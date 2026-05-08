use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per XORI instruction execution.
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus (current execution state).
///   - Sends `(InstructionId::Xori, rd, rs1, imm)` to the "decode" bus.
///   - Sends two operations to the "memory" bus: read rs1, write rd.
///   - Sends four byte-pair tuples to the "bytes_xor" bus, proving
///     `rd_new_value[i] = rs1_value[i] ^ imm_se_bytes[i]`.
///   - Sends `(next_pc[0..4], timestamp + 2)` to the "trace" bus (next execution state).
#[repr(C)]
pub struct XoriColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index (from decode bus).
    pub rd: F,
    /// Source register index (from decode bus).
    pub rs1: F,
    /// 12-bit unsigned immediate (from decode bus).
    pub imm: F,

    /// Bit decomposition of bits 8–11 of `imm` (little-endian).
    /// `imm_high_bits[3]` is the sign bit (bit 11).
    pub imm_high_bits: [F; 4],
    /// Sign-extended `imm` as four byte limbs (little-endian u32).
    pub imm_se_bytes: [F; 4],

    /// Value read from register `rs1`.
    pub rs1_value: [F; 4],
    /// Old value of register `rd` (before write).
    pub old_rd_value: [F; 4],
    /// New value written to `rd`: `rs1_value ^ sign_extend(imm)`.
    pub rd_new_value: [F; 4],

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    /// When zero, all lookup multiplicities are zero (lookups disabled).
    pub is_dummy: F,
}

pub const NUM_XORI_COLS: usize = size_of::<XoriColumns<u8>>();

impl<T> Borrow<XoriColumns<T>> for [T] {
    fn borrow(&self) -> &XoriColumns<T> {
        debug_assert_eq!(self.len(), NUM_XORI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<XoriColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<XoriColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut XoriColumns<T> {
        debug_assert_eq!(self.len(), NUM_XORI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<XoriColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct XoriAir {
    num_lookups: usize,
}

impl XoriAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for XoriAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for XoriAir {
    fn width(&self) -> usize {
        NUM_XORI_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for XoriAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &XoriColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Bit-decompose bits 8–11 of `imm`. The weighted sum reconstructs
        // the high nibble; bit 3 is the sign bit used for sign extension.
        for bit in local.imm_high_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        let imm_high_nibble: AB::Expr = local.imm_high_bits[0].clone()
            + local.imm_high_bits[1].clone() * AB::Expr::TWO
            + local.imm_high_bits[2].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_high_bits[3].clone() * AB::Expr::from(AB::F::from_u32(8));
        let sign_bit: AB::Expr = local.imm_high_bits[3].clone().into();

        // imm = imm_se_bytes[0] + imm_high_nibble * 256
        builder.assert_eq(
            local.imm.clone(),
            local.imm_se_bytes[0].clone()
                + imm_high_nibble.clone() * AB::Expr::from(AB::F::from_u32(256)),
        );

        // Bytes 1–3 of the sign-extended immediate.
        // byte 1: high nibble of imm, plus 0xF0 replicated sign bits.
        builder.assert_eq(
            local.imm_se_bytes[1].clone(),
            imm_high_nibble + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0)),
        );
        // bytes 2 and 3 are all-ones when the sign bit is set.
        builder.assert_eq(
            local.imm_se_bytes[2].clone(),
            sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.imm_se_bytes[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
    }
}

impl<F: Field> LookupAir<F> for XoriAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let local: &XoriColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        // Consume the current execution state from the "trace" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local.pc.into_iter()
                    .chain(once(local.timestamp))
                    .map(Into::into)
                    .collect(),
                local.is_dummy.into(),
                Direction::Receive,
            )],
        ));

        // Assert the decoded instruction is XORI with (pc, rd, rs1, imm) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                local.pc.into_iter().map(Into::into)
                    .chain(once(F::from_u64(InstructionId::Xori as u64).into()))
                    .chain([local.rd, local.rs1, local.imm].into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Memory bus schema: (timestamp, memory_type, addr[0..4], read[0..4], write[0..4]).
        // Register addresses fit in a single byte; the upper three address limbs are zero.
        // For a register read, write == read (value unchanged).
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once(local.timestamp.into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rs1.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.rs1_value.into_iter().map(Into::into))
                    .chain(local.rs1_value.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write the XOR result to register rd at timestamp + 1.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::ONE).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.rd_new_value.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Four byte-level XOR lookups: rd_new[i] = rs1_value[i] ^ imm_se_bytes[i].
        // These also implicitly range-check all three operand bytes to [0, 255].
        lookups.extend(
            local
                .rs1_value
                .into_iter()
                .zip(local.imm_se_bytes.into_iter())
                .zip(local.rd_new_value.into_iter())
                .map(|((x, y), z)| {
                    self.register_lookup(
                        Kind::Global(String::from("bytes_xor")),
                        &vec![(
                            [x, y, z].into_iter().map(Into::into).collect(),
                            local.is_dummy.into(),
                            Direction::Send,
                        )],
                    )
                }),
        );

        // Emit the next execution state into the "trace" bus.
        // next_pc is constrained to pc + 4 in eval; two memory ops consumed 2 timestamps.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local.next_pc.into_iter()
                    .map(Into::into)
                    .chain(once((local.timestamp.clone() + F::from_u64(2)).into()))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
