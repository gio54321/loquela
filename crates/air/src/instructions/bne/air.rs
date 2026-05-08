use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per BNE instruction execution.
///
/// BNE (B-type, opcode=0x63, funct3=0x1):
///   if rs1 != rs2, next_pc = pc + imm; else next_pc = pc + 4.
///
/// Same equality check as BEQ, but taken = 1 - eq_taken:
///   taken = 1 iff NOT all diff bytes are zero.
///   Equivalently: taken = 1 - byte_is_zero[0]*byte_is_zero[1]*byte_is_zero[2]*byte_is_zero[3].
///
/// Wiring mirrors BEQ exactly, with InstructionId::Bne instead of Beq.
#[repr(C)]
pub struct BneColumns<F> {
    pub pc: [F; 4],
    pub timestamp: F,
    pub rs1: F,
    pub rs2: F,
    pub imm_top7: F,
    pub imm_lo5: F,
    pub imm_top7_bits: [F; 7],
    pub imm_lo5_bits: [F; 5],
    pub imm_b: [F; 4],
    pub jmp_target: [F; 4],
    pub jmp_carries: [F; 4],
    pub pc_plus4: [F; 4],
    pub pc_plus4_carries: [F; 3],
    pub rs1_bytes: [F; 4],
    pub rs2_bytes: [F; 4],
    pub diff_bytes: [F; 4],
    pub borrow: [F; 4],
    pub diff_byte_inv: [F; 4],
    pub byte_is_zero: [F; 4],
    /// taken = 1 iff rs1 != rs2 (invert of BEQ).
    pub taken: F,
    pub next_pc: [F; 4],
    pub is_dummy: F,
}

pub const NUM_BNE_COLS: usize = size_of::<BneColumns<u8>>();

impl<T> Borrow<BneColumns<T>> for [T] {
    fn borrow(&self) -> &BneColumns<T> {
        debug_assert_eq!(self.len(), NUM_BNE_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<BneColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<BneColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut BneColumns<T> {
        debug_assert_eq!(self.len(), NUM_BNE_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<BneColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct BneAir {
    num_lookups: usize,
}

impl BneAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for BneAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for BneAir {
    fn width(&self) -> usize {
        NUM_BNE_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for BneAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &BneColumns<AB::Var> = main.current_slice().borrow();

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

        let sign_bit: AB::Expr = local.imm_top7_bits[6].clone().into();

        // Reconstruct imm_b bytes.
        let byte0: AB::Expr = local.imm_lo5_bits[0].clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.imm_lo5_bits[1].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_lo5_bits[2].clone() * AB::Expr::from(AB::F::from_u32(8))
            + local.imm_lo5_bits[3].clone() * AB::Expr::from(AB::F::from_u32(16))
            + local.imm_top7_bits[0].clone() * AB::Expr::from(AB::F::from_u32(32))
            + local.imm_top7_bits[1].clone() * AB::Expr::from(AB::F::from_u32(64))
            + local.imm_top7_bits[2].clone() * AB::Expr::from(AB::F::from_u32(128));
        builder.assert_eq(local.imm_b[0].clone(), byte0);

        let byte1: AB::Expr = local.imm_top7_bits[3].clone()
            + local.imm_top7_bits[4].clone() * AB::Expr::TWO
            + local.imm_top7_bits[5].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_lo5_bits[4].clone() * AB::Expr::from(AB::F::from_u32(8))
            + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0));
        builder.assert_eq(local.imm_b[1].clone(), byte1);

        builder.assert_eq(
            local.imm_b[2].clone(),
            sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.imm_b[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // jmp_target = pc + imm_b.
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

        // pc_plus4 = pc + 4.
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

        // Borrow chain for rs1 - rs2.
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

        // taken = 1 iff NOT equal (invert BEQ logic).
        builder.assert_bool(local.taken.clone());
        let all_zero: AB::Expr = local.byte_is_zero[0].clone()
            * local.byte_is_zero[1].clone()
            * local.byte_is_zero[2].clone()
            * local.byte_is_zero[3].clone();
        // taken = 1 - all_zero
        builder.assert_eq(local.taken.clone(), AB::Expr::ONE - all_zero);

        // next_pc selection.
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

impl<F: Field> LookupAir<F> for BneAir {
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
        let local: &BneColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

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

        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_b")),
            &vec![(
                vec![
                    F::from_u64(InstructionId::Bne as u64).into(),
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

        lookups.extend(local.rs1_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        lookups.extend(local.rs2_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        lookups.extend(local.diff_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

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
