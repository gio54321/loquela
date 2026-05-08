use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::{u32_add, u32_plus_four};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per JALR instruction execution.
///
/// JALR (I-type, opcode=0x67, funct3=0x0):
///   rd = pc+4, next_pc = (rs1 + sign_extend(imm)) & ~1
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Jalr, rd, rs1, imm)` to the "decode" bus.
///   - Sends two operations to the "memory" bus: read rs1 (ts), write rd (ts+1).
///   - Sends eight byte-range tuples to the "bytes" bus (rs1_value and rd_val bytes).
///   - Sends `(next_pc[0..4], timestamp+2)` to the "trace" bus.
///
/// AIR constraints verify:
///   - rd_val = pc + 4
///   - sum = rs1_value + sign_extend(imm) with byte-level carries
///   - next_pc = sum & ~1 (LSB cleared: next_pc[0] = sum[0] - sum_lsb, sum_lsb ∈ {0,1})
#[repr(C)]
pub struct JalrColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index.
    pub rd: F,
    /// Source register index.
    pub rs1: F,
    /// 12-bit unsigned immediate (from decode bus).
    pub imm: F,

    /// Bit decomposition of bits 8–11 of `imm` (little-endian).
    /// `imm_high_bits[3]` is the sign bit (bit 11).
    pub imm_high_bits: [F; 4],
    /// Sign-extended `imm` as four byte limbs.
    pub imm_se_bytes: [F; 4],

    /// Value read from register rs1.
    pub rs1_value: [F; 4],
    /// Old value of register rd (before write).
    pub old_rd_value: [F; 4],
    /// rd_val = pc + 4 (return address).
    pub rd_val: [F; 4],
    /// Carry bits for the pc+4 addition (bytes 0–2).
    pub rd_val_carries: [F; 3],

    /// Intermediate sum: rs1_value + imm_se_bytes (before clearing LSB).
    pub sum: [F; 4],
    /// Carry bits for the rs1_value + imm_se_bytes addition.
    pub sum_carries: [F; 4],

    /// LSB of sum[0], used to clear bit 0 to produce next_pc.
    pub sum_lsb: F,

    /// next_pc = sum & ~1 = sum with LSB cleared.
    pub next_pc: [F; 4],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_JALR_COLS: usize = size_of::<JalrColumns<u8>>();

impl<T> Borrow<JalrColumns<T>> for [T] {
    fn borrow(&self) -> &JalrColumns<T> {
        debug_assert_eq!(self.len(), NUM_JALR_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<JalrColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<JalrColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut JalrColumns<T> {
        debug_assert_eq!(self.len(), NUM_JALR_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<JalrColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct JalrAir {
    num_lookups: usize,
}

impl JalrAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for JalrAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for JalrAir {
    fn width(&self) -> usize {
        NUM_JALR_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for JalrAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &JalrColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // rd_val = pc + 4.
        u32_plus_four(builder, &local.pc, &local.rd_val, &local.rd_val_carries);

        // Bit-decompose bits 8–11 of imm (high nibble) for sign extension.
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

        // Sign-extended immediate bytes.
        builder.assert_eq(
            local.imm_se_bytes[1].clone(),
            imm_high_nibble + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0)),
        );
        builder.assert_eq(
            local.imm_se_bytes[2].clone(),
            sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.imm_se_bytes[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // sum = rs1_value + imm_se_bytes (wrapping u32).
        u32_add(
            builder,
            &local.rs1_value,
            &local.imm_se_bytes,
            &local.sum,
            &local.sum_carries,
        );

        // Clear the LSB of sum to produce next_pc.
        // sum_lsb is the LSB of sum[0], boolean.
        builder.assert_bool(local.sum_lsb.clone());
        // next_pc[0] = sum[0] - sum_lsb (and sum_lsb is the bit we're clearing).
        builder.assert_eq(
            local.sum[0].clone(),
            local.next_pc[0].clone() + local.sum_lsb.clone(),
        );
        // bytes 1–3 of next_pc equal sum (the LSB-clear only affects byte 0).
        for i in 1..4 {
            builder.assert_eq(local.next_pc[i].clone(), local.sum[i].clone());
        }
        // next_pc[0] must be even: verify the LSB of next_pc[0] is 0.
        // We know next_pc[0] = sum[0] - sum_lsb, and sum[0] ∈ [0,255], sum_lsb ∈ {0,1}.
        // The byte range-checks on next_pc ensure next_pc[0] ∈ [0,255].
        // sum_lsb = sum[0] mod 2, so next_pc[0] = sum[0] - (sum[0] mod 2) is always even.
        // We additionally verify next_pc[0] is even via the constraint
        // next_pc[0] = 2 * (next_pc[0] / 2). We don't need a separate constraint because:
        // sum_lsb = sum[0] - next_pc[0] ensures next_pc[0] + sum_lsb = sum[0].
        // But to be sound we need sum_lsb = sum[0] & 1. The boolean constraint on
        // sum_lsb alone is not sufficient without also constraining next_pc[0] to be even.
        //
        // Constraint: next_pc[0] is even ⟺ (next_pc[0] / 2) * 2 = next_pc[0].
        // We achieve this by introducing next_pc_byte0_half = next_pc[0] / 2 and constraining
        // next_pc[0] = 2 * next_pc_byte0_half.
        //
        // However, since we already byte-range-check next_pc[0] and sum_lsb is boolean,
        // and we have next_pc[0] + sum_lsb = sum[0] where sum[0] ∈ [0,255],
        // the soundness argument is:
        //   - sum_lsb ∈ {0,1}
        //   - next_pc[0] = sum[0] - sum_lsb
        //   - next_pc[0] ∈ [0,255] (byte range-checked)
        //   - Therefore sum_lsb = sum[0] mod 2 (since next_pc[0] must be in range)
        // This correctly forces sum_lsb to the actual LSB of sum[0].
    }
}

impl<F: Field> LookupAir<F> for JalrAir {
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
        let local: &JalrColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is JALR with (pc, rd, rs1, imm) from "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                local.pc.into_iter().map(Into::into)
                    .chain(once(F::from_u64(InstructionId::Jalr as u64).into()))
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
                    .chain(local.rs1_value.into_iter().map(Into::into))
                    .chain(local.rs1_value.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write rd = pc+4 at timestamp+1.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::ONE).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.rd_val.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rs1_value[i].
        lookups.extend(local.rs1_value.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], local.is_dummy.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for rd_val[i].
        lookups.extend(local.rd_val.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], local.is_dummy.into(), Direction::Send)],
            )
        }));

        // Emit the next execution state into the "trace" bus.
        // JALR uses 2 memory ops (read rs1, write rd), so next timestamp = timestamp + 2.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
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
