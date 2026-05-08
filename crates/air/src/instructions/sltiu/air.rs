use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per SLTIU instruction execution.
///
/// SLTIU: `rd = (rs1 < sign_extend(imm) as u32) ? 1 : 0`
/// (unsigned comparison; imm is sign-extended to 32 bits, then treated as unsigned).
///
/// Comparison technique: compute `rs1 - imm_se` byte-by-byte with borrow chain.
/// The final borrow-out is 1 iff `rs1 < imm_se` (unsigned).
///
/// The 12-bit immediate is sign-extended: `imm_se = sign_extend(imm[11:0])`.
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Sltiu, rd, rs1, imm)` to the "decode" bus.
///   - Sends two operations to the "memory" bus: read rs1, write rd.
///   - Sends byte range-checks for rs1_bytes, diff_bytes to the "bytes" bus.
///   - Sends `(next_pc[0..4], timestamp + 2)` to the "trace" bus.
#[repr(C)]
pub struct SltiuColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index (from decode bus).
    pub rd: F,
    /// Source register index (from decode bus).
    pub rs1: F,
    /// 12-bit unsigned immediate (from decode bus, bits 20–31 of instruction).
    pub imm: F,

    /// Bit decomposition of bits 8–11 of `imm` (little-endian).
    /// `imm_high_bits[3]` is the sign bit (bit 11 of imm).
    pub imm_high_bits: [F; 4],
    /// Sign-extended `imm` as four byte limbs (little-endian u32).
    pub imm_se_bytes: [F; 4],

    /// Value read from register `rs1`, as byte limbs.
    pub rs1_bytes: [F; 4],
    /// Old value of register `rd` (before write), as byte limbs.
    pub old_rd_value: [F; 4],

    /// Byte-level difference `rs1 - imm_se` (wrapping), as byte limbs.
    pub diff_bytes: [F; 4],
    /// Borrow-out bits for each byte of the subtraction.
    pub borrow: [F; 4],

    /// The less-than result: 1 iff rs1 < imm_se (unsigned). Equals `borrow[3]`.
    pub lt_result: F,

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_SLTIU_COLS: usize = size_of::<SltiuColumns<u8>>();

impl<T> Borrow<SltiuColumns<T>> for [T] {
    fn borrow(&self) -> &SltiuColumns<T> {
        debug_assert_eq!(self.len(), NUM_SLTIU_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<SltiuColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<SltiuColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut SltiuColumns<T> {
        debug_assert_eq!(self.len(), NUM_SLTIU_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<SltiuColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct SltiuAir {
    num_lookups: usize,
}

impl SltiuAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for SltiuAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for SltiuAir {
    fn width(&self) -> usize {
        NUM_SLTIU_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for SltiuAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &SltiuColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Bit-decompose bits 8–11 of `imm`. Bit 3 is the sign bit.
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

        // Sign extension: byte 1 = high nibble of imm + 0xF0 * sign_bit.
        builder.assert_eq(
            local.imm_se_bytes[1].clone(),
            imm_high_nibble + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0)),
        );
        // Bytes 2 and 3 are all-ones when the sign bit is set.
        builder.assert_eq(
            local.imm_se_bytes[2].clone(),
            sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.imm_se_bytes[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // Borrow bits must be boolean.
        for b in local.borrow.iter() {
            builder.assert_bool(b.clone());
        }

        // lt_result is boolean.
        builder.assert_bool(local.lt_result.clone());

        // Borrow chain: rs1[i] + borrow[i]*256 = imm_se[i] + diff[i] + borrow_in[i]
        for i in 0..4 {
            let borrow_in: AB::Expr = if i == 0 {
                AB::Expr::ZERO
            } else {
                local.borrow[i - 1].clone().into()
            };
            builder.assert_eq(
                local.rs1_bytes[i].clone()
                    + local.borrow[i].clone() * AB::Expr::from(AB::F::from_u32(256)),
                local.imm_se_bytes[i].clone() + local.diff_bytes[i].clone() + borrow_in,
            );
        }

        // lt_result = borrow[3] (final borrow-out).
        builder.assert_eq(local.lt_result.clone(), local.borrow[3].clone());
    }
}

impl<F: Field> LookupAir<F> for SltiuAir {
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
        let local: &SltiuColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        // Consume the current execution state from the "trace" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .pc
                    .into_iter()
                    .chain(once(local.timestamp))
                    .map(Into::into)
                    .collect(),
                local.is_dummy.into(),
                Direction::Receive,
            )],
        ));

        // Assert the decoded instruction is SLTIU with (pc, rd, rs1, imm) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                local.pc.into_iter().map(Into::into)
                    .chain(once(F::from_u64(InstructionId::Sltiu as u64).into()))
                    .chain([local.rd, local.rs1, local.imm].into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Read rs1 at timestamp.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once(local.timestamp.into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rs1.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.rs1_bytes.into_iter().map(Into::into))
                    .chain(local.rs1_bytes.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write rd = lt_result at timestamp + 1.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::ONE).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(
                        once(local.lt_result.into())
                            .chain([F::ZERO; 3].into_iter().map(Into::into)),
                    )
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rs1_bytes[i].
        lookups.extend(local.rs1_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for diff_bytes[i].
        lookups.extend(local.diff_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Emit the next execution state into the "trace" bus.
        // Two memory ops consumed 2 timestamps.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
                    .map(Into::into)
                    .chain(once(
                        (local.timestamp.clone() + F::from_u64(2)).into(),
                    ))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
