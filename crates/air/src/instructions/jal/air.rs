use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::{u32_add, u32_plus_four};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per JAL instruction execution.
///
/// JAL (J-type, opcode=0x6F): rd = pc+4, next_pc = pc + sign_extend(offset).
/// The J-type immediate is scrambled across instruction bits; the decode AIR
/// exposes the raw bit fields via the "decode_j" bus:
///   - imm_high12 = bits 31:20 of the instruction word = {imm[20], imm[10:1], imm[11]}
///   - imm_lo8    = bits 19:12 of the instruction word = imm[19:12]
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Jal, rd, imm_high12, imm_lo8)` to the "decode_j" bus.
///   - Sends one write to the "memory" bus: write rd = rd_val = pc+4 at timestamp.
///   - Sends eight byte-range tuples to the "bytes" bus (pc and rd_val bytes).
///   - Sends `(next_pc[0..4], timestamp+1)` to the "trace" bus.
///
/// AIR constraints verify:
///   - rd_val = pc + 4
///   - The J-type immediate is reconstructed byte-by-byte from the bit fields.
///   - next_pc = pc + imm_j
#[repr(C)]
pub struct JalColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index.
    pub rd: F,
    /// Upper 12 bits of instruction (bits 31:20): {imm[20], imm[10:1], imm[11]}.
    pub imm_high12: F,
    /// Lower 8 bits of the immediate (bits 19:12 of instruction): imm[19:12].
    pub imm_lo8: F,

    /// Bit decomposition of imm_high12 (12 bits, little-endian).
    /// Layout: bits[0..10] = imm[10:1] (10 bits of the immediate), bit[10] = imm[11], bit[11] = imm[20].
    pub imm_high12_bits: [F; 12],

    /// Bit decomposition of imm_lo8 (8 bits, little-endian).
    /// These are the 8 bits of imm[19:12].
    pub imm_lo8_bits: [F; 8],

    /// The sign-extended J-type immediate as four byte limbs.
    pub imm_j: [F; 4],
    /// Carry bits for the pc + imm_j addition.
    pub jmp_carries: [F; 4],

    /// rd_val = pc + 4, as four byte limbs.
    pub rd_val: [F; 4],
    /// Carry bits for the pc + 4 addition (bytes 0-2).
    pub rd_val_carries: [F; 3],
    /// Old value of register rd (before write).
    pub old_rd_value: [F; 4],

    /// next_pc = pc + imm_j as four byte limbs.
    pub next_pc: [F; 4],

    /// 1 if rd == 0 (i.e. x0, where the return-address write is silently
    /// dropped per RISC-V semantics), else 0.
    pub rd_is_zero: F,
    /// Witness inverse of rd. Required by the rd_is_zero "is-zero" gadget:
    /// when rd != 0 it equals rd^{-1}; when rd == 0 it can be any value
    /// (and the rd*rd_is_zero=0 constraint forces rd_is_zero=1).
    pub rd_inv: F,

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_JAL_COLS: usize = size_of::<JalColumns<u8>>();

impl<T> Borrow<JalColumns<T>> for [T] {
    fn borrow(&self) -> &JalColumns<T> {
        debug_assert_eq!(self.len(), NUM_JAL_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<JalColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<JalColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut JalColumns<T> {
        debug_assert_eq!(self.len(), NUM_JAL_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<JalColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct JalAir {
    num_lookups: usize,
}

impl JalAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for JalAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for JalAir {
    fn width(&self) -> usize {
        NUM_JAL_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for JalAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &JalColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Bit-decompose imm_high12 into 12 bits.
        for bit in local.imm_high12_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        // Reconstruct imm_high12 from its bit decomposition.
        let mut recon_high12 = AB::Expr::ZERO;
        for (i, bit) in local.imm_high12_bits.iter().enumerate() {
            recon_high12 = recon_high12 + bit.clone() * AB::Expr::from(AB::F::from_u32(1u32 << i));
        }
        builder.assert_eq(local.imm_high12.clone(), recon_high12);

        // Bit-decompose imm_lo8 into 8 bits.
        for bit in local.imm_lo8_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        // Reconstruct imm_lo8 from bits.
        let mut recon_lo8 = AB::Expr::ZERO;
        for (i, bit) in local.imm_lo8_bits.iter().enumerate() {
            recon_lo8 = recon_lo8 + bit.clone() * AB::Expr::from(AB::F::from_u32(1u32 << i));
        }
        builder.assert_eq(local.imm_lo8.clone(), recon_lo8);

        // The sign bit is imm_high12_bits[11] (imm[20]).
        let sign_bit: AB::Expr = local.imm_high12_bits[11].clone().into();

        // Reconstruct the 4 bytes of imm_j from the bit fields:
        //
        // J-type immediate layout:
        //   bit[0]    = 0 (always)
        //   bit[10:1] = imm_high12_bits[0..10]  (bits 0..9 of imm_high12 = imm[10:1])
        //   bit[11]   = imm_high12_bits[10]       (bit 10 of imm_high12 = imm[11])
        //   bit[19:12]= imm_lo8_bits[0..8]         (imm_lo8 = imm[19:12])
        //   bit[20]   = imm_high12_bits[11]        (bit 11 of imm_high12 = imm[20] = sign)
        //   bit[31:21]= sign_bit (sign extension)
        //
        // byte0 = {imm[7:1], 0} in bits[7:0]:
        //   = imm_high12_bits[0]*2 + ... + imm_high12_bits[6]*128
        let byte0: AB::Expr = local.imm_high12_bits[0].clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.imm_high12_bits[1].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_high12_bits[2].clone() * AB::Expr::from(AB::F::from_u32(8))
            + local.imm_high12_bits[3].clone() * AB::Expr::from(AB::F::from_u32(16))
            + local.imm_high12_bits[4].clone() * AB::Expr::from(AB::F::from_u32(32))
            + local.imm_high12_bits[5].clone() * AB::Expr::from(AB::F::from_u32(64))
            + local.imm_high12_bits[6].clone() * AB::Expr::from(AB::F::from_u32(128));
        builder.assert_eq(local.imm_j[0].clone(), byte0);

        // byte1 = bits[15:8] of imm_j:
        //   imm[10:8] = imm_high12_bits[7..10] → bits 0..2 of byte1
        //   imm[11]   = imm_high12_bits[10]     → bit 3 of byte1
        //   imm[15:12]= imm_lo8_bits[0..4]       → bits 4..7 of byte1
        let byte1: AB::Expr = local.imm_high12_bits[7].clone()
            + local.imm_high12_bits[8].clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.imm_high12_bits[9].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_high12_bits[10].clone() * AB::Expr::from(AB::F::from_u32(8))
            + local.imm_lo8_bits[0].clone() * AB::Expr::from(AB::F::from_u32(16))
            + local.imm_lo8_bits[1].clone() * AB::Expr::from(AB::F::from_u32(32))
            + local.imm_lo8_bits[2].clone() * AB::Expr::from(AB::F::from_u32(64))
            + local.imm_lo8_bits[3].clone() * AB::Expr::from(AB::F::from_u32(128));
        builder.assert_eq(local.imm_j[1].clone(), byte1);

        // byte2 = bits[23:16] of imm_j:
        //   imm[19:16]= imm_lo8_bits[4..8] → bits 0..3 of byte2
        //   imm[23:20]= sign_bit replicated → bits 4..7 of byte2 = sign_bit * 0xF0
        let byte2: AB::Expr = local.imm_lo8_bits[4].clone()
            + local.imm_lo8_bits[5].clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.imm_lo8_bits[6].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_lo8_bits[7].clone() * AB::Expr::from(AB::F::from_u32(8))
            + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0));
        builder.assert_eq(local.imm_j[2].clone(), byte2);

        // byte3 = bits[31:24] of imm_j: all sign bits = sign_bit * 0xFF
        builder.assert_eq(
            local.imm_j[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // next_pc = pc + imm_j (wrapping u32).
        u32_add(
            builder,
            &local.pc,
            &local.imm_j,
            &local.next_pc,
            &local.jmp_carries,
        );

        // rd_val = pc + 4
        u32_plus_four(builder, &local.pc, &local.rd_val, &local.rd_val_carries);

        // rd_is_zero indicator: 1 if rd == 0, else 0.
        // Standard "is-zero" gadget using a witness inverse rd_inv:
        //   rd_is_zero is boolean
        //   rd * rd_is_zero == 0       (if rd != 0, then rd_is_zero must be 0)
        //   rd * rd_inv + rd_is_zero == 1   (if rd == 0, then rd_is_zero must be 1)
        builder.assert_bool(local.rd_is_zero.clone());
        builder.assert_zero(local.rd.clone() * local.rd_is_zero.clone());
        builder.assert_eq(
            local.rd.clone() * local.rd_inv.clone() + local.rd_is_zero.clone().into(),
            AB::Expr::ONE,
        );
    }
}

impl<F: Field> LookupAir<F> for JalAir {
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
        let local: &JalColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is JAL with (rd, imm_high12, imm_lo8) from "decode_j" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_j")),
            &vec![(
                vec![
                    F::from_u64(InstructionId::Jal as u64).into(),
                    local.rd.into(),
                    local.imm_high12.into(),
                    local.imm_lo8.into(),
                ],
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write rd = pc+4 at timestamp. Suppressed for rd == x0 (the VM
        // silently drops that write); gate the send with (1 - rd_is_zero).
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once(local.timestamp.into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.rd_val.into_iter().map(Into::into))
                    .collect(),
                {
                    let one_minus_rdz: p3_air::SymbolicExpression<F> =
                        Into::<p3_air::SymbolicExpression<F>>::into(F::ONE)
                            - Into::<p3_air::SymbolicExpression<F>>::into(local.rd_is_zero);
                    let is_dummy: p3_air::SymbolicExpression<F> = local.is_dummy.into();
                    is_dummy * one_minus_rdz
                },
                Direction::Send,
            )],
        ));

        // Byte range-checks for pc[i].
        lookups.extend(local.pc.into_iter().map(|byte| {
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
        // JAL uses 1 memory op (write rd), so next timestamp = timestamp + 1.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
                    .map(Into::into)
                    .chain(once((local.timestamp.clone() + F::ONE).into()))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
