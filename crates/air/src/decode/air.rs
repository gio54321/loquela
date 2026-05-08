use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
    vec,
};

use crate::primitives::bit_decompose::{check_bit_decomposition, pack_bits};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    WindowAccess,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

pub struct Instruction<F> {
    pub is_addi: F,
    pub is_xori: F,
    pub is_ori: F,
    pub is_andi: F,
    pub is_add: F,
    pub is_sub: F,
    pub is_xor: F,
    pub is_or: F,
    pub is_and: F,
    pub is_sll: F,
    pub is_srl: F,
}

#[repr(u8)]
pub enum InstructionId {
    Addi = 0,
    Xori = 1,
    Ori = 2,
    Andi = 3,
    Add = 4,
    Sub = 5,
    Xor = 6,
    Or = 7,
    And = 8,
    Sll = 9,
    Srl = 10,
}

#[repr(C)]
pub struct DecodeColumns<F> {
    pub pc: [F; 4],
    pub instruction: [F; 4],
    pub decompositions: [[F; 8]; 4],
    pub instr_type: Instruction<F>,
    pub instr_type_packed: F,
    pub rd: F,
    pub rs1: F,
    /// Unsigned 12-bit immediate (I-type). Holds bits 20–31 for ADDI/XORI; unconstrained for ADD.
    pub imm: F,
    /// Source register 2 index (R-type). Holds bits 20–24 for all instruction types.
    pub rs2: F,
    pub mult: F,
}

pub const NUM_DECODE_COLS: usize = size_of::<DecodeColumns<u8>>();

impl<T> Borrow<DecodeColumns<T>> for [T] {
    fn borrow(&self) -> &DecodeColumns<T> {
        debug_assert_eq!(self.len(), NUM_DECODE_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<DecodeColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<DecodeColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut DecodeColumns<T> {
        debug_assert_eq!(self.len(), NUM_DECODE_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<DecodeColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct DecodeAir {
    num_lookups: usize,
}

impl DecodeAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for DecodeAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for DecodeAir {
    fn width(&self) -> usize {
        NUM_DECODE_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for DecodeAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &DecodeColumns<AB::Var> = main.current_slice().borrow();

        // is_* flags are one-hot booleans.
        builder.assert_bool(local.instr_type.is_addi.clone());
        builder.assert_bool(local.instr_type.is_xori.clone());
        builder.assert_bool(local.instr_type.is_ori.clone());
        builder.assert_bool(local.instr_type.is_andi.clone());
        builder.assert_bool(local.instr_type.is_add.clone());
        builder.assert_bool(local.instr_type.is_sub.clone());
        builder.assert_bool(local.instr_type.is_xor.clone());
        builder.assert_bool(local.instr_type.is_or.clone());
        builder.assert_bool(local.instr_type.is_and.clone());
        builder.assert_bool(local.instr_type.is_sll.clone());
        builder.assert_bool(local.instr_type.is_srl.clone());
        builder.assert_eq(
            local.instr_type.is_addi.clone()
                + local.instr_type.is_xori.clone()
                + local.instr_type.is_ori.clone()
                + local.instr_type.is_andi.clone()
                + local.instr_type.is_add.clone()
                + local.instr_type.is_sub.clone()
                + local.instr_type.is_xor.clone()
                + local.instr_type.is_or.clone()
                + local.instr_type.is_and.clone()
                + local.instr_type.is_sll.clone()
                + local.instr_type.is_srl.clone(),
            AB::Expr::ONE,
        );

        // Bit-decompose each instruction byte limb.
        for (i, limb) in local.instruction.iter().enumerate() {
            check_bit_decomposition(builder, limb.clone(), &local.decompositions[i]);
        }

        // Opcode (bits 0..7) == 0b0010011 for ADDI, XORI, ORI, and ANDI (I-type).
        let mut when_op_immediate = builder.when(
            local.instr_type.is_addi.clone()
                + local.instr_type.is_xori.clone()
                + local.instr_type.is_ori.clone()
                + local.instr_type.is_andi.clone(),
        );
        for i in 0..7 {
            let expected = if (0b0010011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_op_immediate.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // Opcode (bits 0..7) == 0b0110011 for ADD, SUB, XOR, OR, AND, SLL, and SRL (R-type).
        let mut when_r_type = builder.when(
            local.instr_type.is_add.clone()
                + local.instr_type.is_sub.clone()
                + local.instr_type.is_xor.clone()
                + local.instr_type.is_or.clone()
                + local.instr_type.is_and.clone()
                + local.instr_type.is_sll.clone()
                + local.instr_type.is_srl.clone(),
        );
        for i in 0..7 {
            let expected = if (0b0110011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_r_type.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // funct3 = bits 12..15 of the instruction word — that's bits 4..7 of byte 1.
        // ADDI: funct3 == 0b000
        let mut when_addi = builder.when(local.instr_type.is_addi.clone());
        for i in 0..3 {
            let expected = if (0b000u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_addi.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // XORI: funct3 == 0b100
        let mut when_xori = builder.when(local.instr_type.is_xori.clone());
        for i in 0..3 {
            let expected = if (0b100u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_xori.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // ORI: funct3 == 0b110
        let mut when_ori = builder.when(local.instr_type.is_ori.clone());
        for i in 0..3 {
            let expected = if (0b110u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_ori.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // ANDI: funct3 == 0b111
        let mut when_andi = builder.when(local.instr_type.is_andi.clone());
        for i in 0..3 {
            let expected = if (0b111u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_andi.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // ADD: funct3 == 0b000
        let mut when_add = builder.when(local.instr_type.is_add.clone());
        for i in 0..3 {
            when_add.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
        }

        // OR: funct3 == 0b110
        let mut when_or = builder.when(local.instr_type.is_or.clone());
        for i in 0..3 {
            let expected = if (0b110u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_or.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // AND: funct3 == 0b111
        let mut when_and = builder.when(local.instr_type.is_and.clone());
        for i in 0..3 {
            when_and.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ONE);
        }

        // SLL: funct3 == 0b001
        let mut when_sll = builder.when(local.instr_type.is_sll.clone());
        for i in 0..3 {
            let expected = if (0b001u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_sll.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // SRL: funct3 == 0b101
        let mut when_srl = builder.when(local.instr_type.is_srl.clone());
        for i in 0..3 {
            let expected = if (0b101u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_srl.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // ADD: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3.
        let mut when_add = builder.when(local.instr_type.is_add.clone());
        for i in 1..8 {
            when_add.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SUB: funct3 == 0b000
        let mut when_sub = builder.when(local.instr_type.is_sub.clone());
        for i in 0..3 {
            when_sub.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
        }

        // SUB: funct7 (bits 25..32) == 0b0100000 — bit 30 (decompositions[3][6]) is 1,
        // all other funct7 bits (decompositions[3][1..6] and [3][7]) are 0.
        let mut when_sub = builder.when(local.instr_type.is_sub.clone());
        when_sub.assert_eq(local.decompositions[3][6].clone(), AB::Expr::ONE);
        let mut when_sub = builder.when(local.instr_type.is_sub.clone());
        for i in [1, 2, 3, 4, 5, 7] {
            when_sub.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // XOR: funct3 == 0b100
        let mut when_xor = builder.when(local.instr_type.is_xor.clone());
        for i in 0..3 {
            let expected = if (0b100u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_xor.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // XOR: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3 are all zero.
        let mut when_xor = builder.when(local.instr_type.is_xor.clone());
        for i in 1..8 {
            when_xor.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // OR: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3.
        let mut when_or = builder.when(local.instr_type.is_or.clone());
        for i in 1..8 {
            when_or.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // AND: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3.
        let mut when_and = builder.when(local.instr_type.is_and.clone());
        for i in 1..8 {
            when_and.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SLL: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3.
        let mut when_sll = builder.when(local.instr_type.is_sll.clone());
        for i in 1..8 {
            when_sll.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SRL: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3.
        let mut when_srl = builder.when(local.instr_type.is_srl.clone());
        for i in 1..8 {
            when_srl.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // rd = bits 7..12 (1 bit in byte 0, 4 bits in byte 1).
        let rd_expr = pack_bits::<AB, 4>(
            &local.decompositions,
            &[(0, 7), (1, 0), (1, 1), (1, 2), (1, 3)],
        );
        builder.assert_eq(local.rd.clone(), rd_expr);

        // rs1 = bits 15..20 (1 bit in byte 1, 4 bits in byte 2).
        let rs1_expr = pack_bits::<AB, 4>(
            &local.decompositions,
            &[(1, 7), (2, 0), (2, 1), (2, 2), (2, 3)],
        );
        builder.assert_eq(local.rs1.clone(), rs1_expr);

        // imm = bits 20..32 (4 bits in byte 2, 8 bits in byte 3) — unsigned 12-bit value.
        // Constrained for all rows; for ADD rows the value is unused by the bus.
        let imm_expr = pack_bits::<AB, 4>(
            &local.decompositions,
            &[
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 7),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (3, 7),
            ],
        );
        builder.assert_eq(local.imm.clone(), imm_expr);

        // rs2 = bits 20..25 (4 bits in byte 2, 1 bit in byte 3).
        // Constrained for all rows; for I-type rows the value is unused by the bus.
        let rs2_expr = pack_bits::<AB, 4>(
            &local.decompositions,
            &[(2, 4), (2, 5), (2, 6), (2, 7), (3, 0)],
        );
        builder.assert_eq(local.rs2.clone(), rs2_expr);

        let packed = local.instr_type.is_addi.clone()
            * AB::Expr::from(AB::F::from_u64(InstructionId::Addi as u64))
            + local.instr_type.is_xori.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Xori as u64))
            + local.instr_type.is_ori.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Ori as u64))
            + local.instr_type.is_andi.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Andi as u64))
            + local.instr_type.is_add.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Add as u64))
            + local.instr_type.is_sub.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Sub as u64))
            + local.instr_type.is_xor.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Xor as u64))
            + local.instr_type.is_or.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Or as u64))
            + local.instr_type.is_and.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::And as u64))
            + local.instr_type.is_sll.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Sll as u64))
            + local.instr_type.is_srl.clone()
                * AB::Expr::from(AB::F::from_u64(InstructionId::Srl as u64));
        builder.assert_eq(local.instr_type_packed.clone(), packed);
    }
}

impl<F: Field> LookupAir<F> for DecodeAir {
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
        let local: &DecodeColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        // Fetch each byte of the instruction from the program table.
        // PC is 4-byte aligned, so adding 0–3 to the low limb never produces a carry.
        for (i, byte) in local.instruction.iter().enumerate() {
            lookups.push(self.register_lookup(
                Kind::Global(String::from("program")),
                &vec![(
                    once((local.pc[0] + F::from_u64(i as u64)).into())
                        .chain(local.pc[1..].iter().cloned().map(Into::into))
                        .chain(once(byte.clone().into()))
                        .collect(),
                    F::ONE.into(),
                    Direction::Send,
                )],
            ));
        }

        // For the decode bus field4: use imm for I-type instructions and rs2 for R-type.
        // is_i_type = is_addi + is_xori + is_ori + is_andi (one-hot so sum is 0 or 1).
        // is_r_type = is_add + is_sub + is_xor + is_or + is_and + is_sll + is_srl (one-hot so sum is 0 or 1).
        let is_i_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_addi)
            + SymbolicExpression::from(local.instr_type.is_xori)
            + SymbolicExpression::from(local.instr_type.is_ori)
            + SymbolicExpression::from(local.instr_type.is_andi);
        let is_r_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_add)
            + SymbolicExpression::from(local.instr_type.is_sub)
            + SymbolicExpression::from(local.instr_type.is_xor)
            + SymbolicExpression::from(local.instr_type.is_or)
            + SymbolicExpression::from(local.instr_type.is_and)
            + SymbolicExpression::from(local.instr_type.is_sll)
            + SymbolicExpression::from(local.instr_type.is_srl);
        let field4: SymbolicExpression<F> = is_i_type * SymbolicExpression::from(local.imm)
            + is_r_type * SymbolicExpression::from(local.rs2);

        // export the decoded instruction
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                local.pc.iter().cloned().map(Into::into)
                    .chain([
                        local.instr_type_packed.into(),
                        local.rd.into(),
                        local.rs1.into(),
                        field4,
                    ])
                    .collect(),
                local.mult.into(),
                Direction::Receive,
            )],
        ));

        lookups
    }
}
