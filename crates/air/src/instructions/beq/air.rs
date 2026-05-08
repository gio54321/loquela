use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per BEQ instruction execution.
///
/// BEQ (B-type, opcode=0x63, funct3=0x0):
///   if rs1 == rs2, next_pc = pc + imm; else next_pc = pc + 4.
///
/// Equality check technique:
///   diff_bytes = rs1_bytes - rs2_bytes (borrow chain, wrapping).
///   For each byte i: byte_is_zero[i] = 1 iff diff_bytes[i] == 0,
///   enforced via inverse witness: diff_bytes[i] * diff_byte_inv[i] = 1 - byte_is_zero[i].
///   taken = product of byte_is_zero[0..3].
///
/// next_pc selection:
///   taken * (next_pc[i] - jmp_target[i]) = 0
///   (1-taken) * (next_pc[i] - pc_plus4[i]) = 0
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Beq, rs1, rs2, imm_top7, imm_lo5)` to the "decode_b" bus.
///   - Sends two reads to the "memory" bus: read rs1 (ts), read rs2 (ts+1).
///   - Sends byte range-checks for rs1_bytes, rs2_bytes, diff_bytes to the "bytes" bus.
///   - Sends `(next_pc[0..4], timestamp+2)` to the "trace" bus.
#[repr(C)]
pub struct BeqColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// First source register index.
    pub rs1: F,
    /// Second source register index.
    pub rs2: F,

    /// Top 7 bits of B-type scrambled immediate: bits 31:25 = {imm[12], imm[10:5]}.
    pub imm_top7: F,
    /// Low 5 bits of B-type scrambled immediate: bits 11:7 = {imm[4:1], imm[11]}.
    pub imm_lo5: F,

    /// Bit decomposition of imm_top7 (7 bits, little-endian).
    /// bits[0..5] = imm[10:5], bits[6] = imm[12] (sign).
    pub imm_top7_bits: [F; 7],

    /// Bit decomposition of imm_lo5 (5 bits, little-endian).
    /// bits[0..3] = imm[4:1], bits[4] = imm[11].
    pub imm_lo5_bits: [F; 5],

    /// The sign-extended B-type immediate as four byte limbs.
    pub imm_b: [F; 4],

    /// pc + imm_b = jmp_target (branch target), as four byte limbs.
    pub jmp_target: [F; 4],
    /// Carry bits for the pc + imm_b addition.
    pub jmp_carries: [F; 4],

    /// pc + 4 as four byte limbs.
    pub pc_plus4: [F; 4],
    /// Carry bits for pc + 4 addition (bytes 0-2).
    pub pc_plus4_carries: [F; 3],

    /// Value read from register rs1, as byte limbs.
    pub rs1_bytes: [F; 4],
    /// Value read from register rs2, as byte limbs.
    pub rs2_bytes: [F; 4],

    /// Byte-level difference rs1 - rs2 (wrapping), as byte limbs.
    pub diff_bytes: [F; 4],
    /// Borrow-out bits for each byte of the subtraction.
    pub borrow: [F; 4],

    /// Inverse witnesses for zero-check on each diff byte.
    /// diff_byte_inv[i] = (diff_bytes[i])^{-1} if diff_bytes[i] != 0, else 0.
    pub diff_byte_inv: [F; 4],
    /// byte_is_zero[i] = 1 iff diff_bytes[i] == 0.
    pub byte_is_zero: [F; 4],

    /// taken = 1 iff rs1 == rs2.
    pub taken: F,

    /// next_pc as four byte limbs: jmp_target if taken, pc_plus4 otherwise.
    pub next_pc: [F; 4],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_BEQ_COLS: usize = size_of::<BeqColumns<u8>>();

impl<T> Borrow<BeqColumns<T>> for [T] {
    fn borrow(&self) -> &BeqColumns<T> {
        debug_assert_eq!(self.len(), NUM_BEQ_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<BeqColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<BeqColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut BeqColumns<T> {
        debug_assert_eq!(self.len(), NUM_BEQ_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<BeqColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct BeqAir {
    num_lookups: usize,
}

impl BeqAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for BeqAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for BeqAir {
    fn width(&self) -> usize {
        NUM_BEQ_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for BeqAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &BeqColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Bit-decompose imm_top7 into 7 bits.
        for bit in local.imm_top7_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        let mut recon_top7 = AB::Expr::ZERO;
        for (i, bit) in local.imm_top7_bits.iter().enumerate() {
            recon_top7 = recon_top7 + bit.clone() * AB::Expr::from(AB::F::from_u32(1u32 << i));
        }
        builder.assert_eq(local.imm_top7.clone(), recon_top7);

        // Bit-decompose imm_lo5 into 5 bits.
        for bit in local.imm_lo5_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        let mut recon_lo5 = AB::Expr::ZERO;
        for (i, bit) in local.imm_lo5_bits.iter().enumerate() {
            recon_lo5 = recon_lo5 + bit.clone() * AB::Expr::from(AB::F::from_u32(1u32 << i));
        }
        builder.assert_eq(local.imm_lo5.clone(), recon_lo5);

        // Sign bit is imm_top7_bits[6] = imm[12].
        let sign_bit: AB::Expr = local.imm_top7_bits[6].clone().into();

        // Reconstruct the 4 bytes of imm_b from the bit fields.
        //
        // B-type immediate layout (with bit[0]=0):
        //   bit[4:1]  = imm_lo5_bits[3:0]
        //   bit[10:5] = imm_top7_bits[5:0]
        //   bit[11]   = imm_lo5_bits[4]
        //   bit[12]   = imm_top7_bits[6] (sign)
        //   bit[31:13]= sign extension
        //
        // byte0 = bits[7:0]:
        //   = imm_lo5_bits[0]*2 + imm_lo5_bits[1]*4 + imm_lo5_bits[2]*8 + imm_lo5_bits[3]*16
        //     + imm_top7_bits[0]*32 + imm_top7_bits[1]*64 + imm_top7_bits[2]*128
        let byte0: AB::Expr = local.imm_lo5_bits[0].clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.imm_lo5_bits[1].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_lo5_bits[2].clone() * AB::Expr::from(AB::F::from_u32(8))
            + local.imm_lo5_bits[3].clone() * AB::Expr::from(AB::F::from_u32(16))
            + local.imm_top7_bits[0].clone() * AB::Expr::from(AB::F::from_u32(32))
            + local.imm_top7_bits[1].clone() * AB::Expr::from(AB::F::from_u32(64))
            + local.imm_top7_bits[2].clone() * AB::Expr::from(AB::F::from_u32(128));
        builder.assert_eq(local.imm_b[0].clone(), byte0);

        // byte1 = bits[15:8]:
        //   imm[10:8] = imm_top7_bits[3..5]  (bits 0..2 of byte1)
        //   imm[11]   = imm_lo5_bits[4]       (bit 3 of byte1)
        //   imm[12]   = imm_top7_bits[6]=sign (bit 4 of byte1)
        //   imm[15:13]= sign extension         (bits 5..7 = sign*0x70 + sign*0x80 - ugh)
        // Simpler: bits 15:13 = {sign,sign,sign}, so byte1 bits 7:4 = sign * 0xF0
        let byte1: AB::Expr = local.imm_top7_bits[3].clone()
            + local.imm_top7_bits[4].clone() * AB::Expr::TWO
            + local.imm_top7_bits[5].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_lo5_bits[4].clone() * AB::Expr::from(AB::F::from_u32(8))
            + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0));
        builder.assert_eq(local.imm_b[1].clone(), byte1);

        // byte2 = all sign bits: sign_bit * 0xFF
        builder.assert_eq(
            local.imm_b[2].clone(),
            sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // byte3 = all sign bits: sign_bit * 0xFF
        builder.assert_eq(
            local.imm_b[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // Constrain jmp_target = pc + imm_b via jmp_carries.
        for i in 0..4 {
            builder.assert_bool(local.jmp_carries[i].clone());
            let carry_in: AB::Expr = if i == 0 {
                AB::Expr::ZERO
            } else {
                local.jmp_carries[i - 1].clone().into()
            };
            builder.assert_eq(
                local.jmp_target[i].clone()
                    + local.jmp_carries[i].clone() * AB::Expr::from(AB::F::from_u32(256)),
                local.pc[i].clone() + local.imm_b[i].clone() + carry_in,
            );
        }

        // Constrain pc_plus4 = pc + 4 via pc_plus4_carries.
        {
            builder.assert_bool(local.pc_plus4_carries[0].clone());
            builder.assert_bool(local.pc_plus4_carries[1].clone());
            builder.assert_bool(local.pc_plus4_carries[2].clone());
            let four = AB::Expr::from(AB::F::from_u32(4));
            let c0: AB::Expr = local.pc_plus4_carries[0].clone().into();
            let c1: AB::Expr = local.pc_plus4_carries[1].clone().into();
            let c2: AB::Expr = local.pc_plus4_carries[2].clone().into();
            builder.assert_eq(
                local.pc[0].clone() + four,
                local.pc_plus4[0].clone() + c0.clone() * AB::Expr::from(AB::F::from_u32(256)),
            );
            builder.assert_eq(
                local.pc[1].clone() + c0,
                local.pc_plus4[1].clone() + c1.clone() * AB::Expr::from(AB::F::from_u32(256)),
            );
            builder.assert_eq(
                local.pc[2].clone() + c1,
                local.pc_plus4[2].clone() + c2.clone() * AB::Expr::from(AB::F::from_u32(256)),
            );
            builder.assert_eq(local.pc[3].clone() + c2, local.pc_plus4[3].clone());
        }

        // Borrow chain for rs1 - rs2 = diff_bytes.
        for b in local.borrow.iter() {
            builder.assert_bool(b.clone());
        }
        for i in 0..4 {
            let borrow_in: AB::Expr = if i == 0 {
                AB::Expr::ZERO
            } else {
                local.borrow[i - 1].clone().into()
            };
            builder.assert_eq(
                local.rs1_bytes[i].clone()
                    + local.borrow[i].clone() * AB::Expr::from(AB::F::from_u32(256)),
                local.rs2_bytes[i].clone() + local.diff_bytes[i].clone() + borrow_in,
            );
        }

        // Zero check per diff byte.
        for i in 0..4 {
            builder.assert_bool(local.byte_is_zero[i].clone());
            builder.assert_eq(
                local.byte_is_zero[i].clone(),
                AB::Expr::ONE - local.diff_bytes[i].clone() * local.diff_byte_inv[i].clone(),
            );
            builder.assert_eq(
                local.diff_bytes[i].clone() * local.byte_is_zero[i].clone(),
                AB::Expr::ZERO,
            );
        }

        // taken = 1 iff all diff bytes are zero.
        builder.assert_bool(local.taken.clone());
        let all_zero: AB::Expr = local.byte_is_zero[0].clone()
            * local.byte_is_zero[1].clone()
            * local.byte_is_zero[2].clone()
            * local.byte_is_zero[3].clone();
        builder.assert_eq(local.taken.clone(), all_zero);

        // next_pc selection: taken * (next_pc - jmp_target) = 0,
        //                    (1-taken) * (next_pc - pc_plus4) = 0.
        let not_taken: AB::Expr = AB::Expr::ONE - local.taken.clone();
        for i in 0..4 {
            builder.assert_eq(
                local.taken.clone() * local.next_pc[i].clone(),
                local.taken.clone() * local.jmp_target[i].clone(),
            );
            builder.assert_eq(
                not_taken.clone() * local.next_pc[i].clone(),
                not_taken.clone() * local.pc_plus4[i].clone(),
            );
        }
    }
}

impl<F: Field> LookupAir<F> for BeqAir {
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
        let local: &BeqColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is BEQ from the "decode_b" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_b")),
            &vec![(
                vec![
                    F::from_u64(InstructionId::Beq as u64).into(),
                    local.rs1.into(),
                    local.rs2.into(),
                    local.imm_top7.into(),
                    local.imm_lo5.into(),
                ],
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

        // Read rs2 at timestamp + 1.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::ONE).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rs2.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.rs2_bytes.into_iter().map(Into::into))
                    .chain(local.rs2_bytes.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rs1_bytes.
        lookups.extend(local.rs1_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for rs2_bytes.
        lookups.extend(local.rs2_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for diff_bytes.
        lookups.extend(local.diff_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Emit the next execution state into the "trace" bus.
        // Two memory ops: timestamp + 2.
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
