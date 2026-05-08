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
    pub is_add: F,
    pub is_and: F,
    pub is_sll: F,
    pub is_srl: F,
    pub is_sra: F,
    pub is_slli: F,
    pub is_srli: F,
    pub is_srai: F,
    pub is_slt: F,
    pub is_sltu: F,
    pub is_slti: F,
    pub is_sltiu: F,
    pub is_lui: F,
    pub is_auipc: F,
    pub is_jal: F,
    pub is_jalr: F,
    pub is_beq: F,
    pub is_bne: F,
    pub is_blt: F,
    pub is_bge: F,
    pub is_bltu: F,
    pub is_bgeu: F,
    pub is_lw: F,
    pub is_lh: F,
    pub is_lb: F,
    pub is_lhu: F,
    pub is_lbu: F,
}

#[repr(u8)]
pub enum InstructionId {
    Addi = 0,
    Xori = 1,
    Add = 2,
    And = 3,
    Sll = 4,
    Srl = 5,
    Sra = 6,
    Slli = 7,
    Srli = 8,
    Srai = 9,
    Slt = 10,
    Sltu = 11,
    Slti = 12,
    Sltiu = 13,
    Lui = 14,
    Auipc = 15,
    Jal = 16,
    Jalr = 17,
    Beq = 18,
    Bne = 19,
    Blt = 20,
    Bge = 21,
    Bltu = 22,
    Bgeu = 23,
    Lw = 24,
    Lh = 25,
    Lb = 26,
    Lhu = 27,
    Lbu = 28,
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
    /// For U-type (LUI/AUIPC), holds bits 31:20 (the upper 12 of the 20-bit U-immediate).
    pub imm: F,
    /// Source register 2 index (R-type). Holds bits 20–24 for all instruction types.
    pub rs2: F,
    /// For U-type (LUI/AUIPC): lower 8 bits of the raw 20-bit immediate (bits 19:12).
    /// Equals bits 19:16 | bits 15:12 of the instruction word.
    /// Zero for non-U-type instructions.
    pub imm_low8: F,
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
        builder.assert_bool(local.instr_type.is_add.clone());
        builder.assert_bool(local.instr_type.is_and.clone());
        builder.assert_bool(local.instr_type.is_sll.clone());
        builder.assert_bool(local.instr_type.is_srl.clone());
        builder.assert_bool(local.instr_type.is_sra.clone());
        builder.assert_bool(local.instr_type.is_slli.clone());
        builder.assert_bool(local.instr_type.is_srli.clone());
        builder.assert_bool(local.instr_type.is_srai.clone());
        builder.assert_bool(local.instr_type.is_slt.clone());
        builder.assert_bool(local.instr_type.is_sltu.clone());
        builder.assert_bool(local.instr_type.is_slti.clone());
        builder.assert_bool(local.instr_type.is_sltiu.clone());
        builder.assert_bool(local.instr_type.is_lui.clone());
        builder.assert_bool(local.instr_type.is_auipc.clone());
        builder.assert_bool(local.instr_type.is_jal.clone());
        builder.assert_bool(local.instr_type.is_jalr.clone());
        builder.assert_bool(local.instr_type.is_beq.clone());
        builder.assert_bool(local.instr_type.is_bne.clone());
        builder.assert_bool(local.instr_type.is_blt.clone());
        builder.assert_bool(local.instr_type.is_bge.clone());
        builder.assert_bool(local.instr_type.is_bltu.clone());
        builder.assert_bool(local.instr_type.is_bgeu.clone());
        builder.assert_bool(local.instr_type.is_lw.clone());
        builder.assert_bool(local.instr_type.is_lh.clone());
        builder.assert_bool(local.instr_type.is_lb.clone());
        builder.assert_bool(local.instr_type.is_lhu.clone());
        builder.assert_bool(local.instr_type.is_lbu.clone());
        builder.assert_eq(
            local.instr_type.is_addi.clone()
                + local.instr_type.is_xori.clone()
                + local.instr_type.is_add.clone()
                + local.instr_type.is_and.clone()
                + local.instr_type.is_sll.clone()
                + local.instr_type.is_srl.clone()
                + local.instr_type.is_sra.clone()
                + local.instr_type.is_slli.clone()
                + local.instr_type.is_srli.clone()
                + local.instr_type.is_srai.clone()
                + local.instr_type.is_slt.clone()
                + local.instr_type.is_sltu.clone()
                + local.instr_type.is_slti.clone()
                + local.instr_type.is_sltiu.clone()
                + local.instr_type.is_lui.clone()
                + local.instr_type.is_auipc.clone()
                + local.instr_type.is_jal.clone()
                + local.instr_type.is_jalr.clone()
                + local.instr_type.is_beq.clone()
                + local.instr_type.is_bne.clone()
                + local.instr_type.is_blt.clone()
                + local.instr_type.is_bge.clone()
                + local.instr_type.is_bltu.clone()
                + local.instr_type.is_bgeu.clone()
                + local.instr_type.is_lw.clone()
                + local.instr_type.is_lh.clone()
                + local.instr_type.is_lb.clone()
                + local.instr_type.is_lhu.clone()
                + local.instr_type.is_lbu.clone(),
            AB::Expr::ONE,
        );

        // Bit-decompose each instruction byte limb.
        for (i, limb) in local.instruction.iter().enumerate() {
            check_bit_decomposition(builder, limb.clone(), &local.decompositions[i]);
        }

        // Opcode (bits 0..7) == 0b0010011 for ADDI, XORI, SLLI, SRLI, SRAI, SLTI, SLTIU (I-type).
        let mut when_op_immediate = builder.when(
            local.instr_type.is_addi.clone()
                + local.instr_type.is_xori.clone()
                + local.instr_type.is_slli.clone()
                + local.instr_type.is_srli.clone()
                + local.instr_type.is_srai.clone()
                + local.instr_type.is_slti.clone()
                + local.instr_type.is_sltiu.clone(),
        );
        for i in 0..7 {
            let expected = if (0b0010011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_op_immediate.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // Opcode (bits 0..7) == 0b0110011 for ADD, AND, SLL, SRL, SRA, SLT, SLTU (R-type).
        let mut when_r_type = builder.when(
            local.instr_type.is_add.clone()
                + local.instr_type.is_and.clone()
                + local.instr_type.is_sll.clone()
                + local.instr_type.is_srl.clone()
                + local.instr_type.is_sra.clone()
                + local.instr_type.is_slt.clone()
                + local.instr_type.is_sltu.clone(),
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

        // ADD: funct3 == 0b000
        let mut when_add = builder.when(local.instr_type.is_add.clone());
        for i in 0..3 {
            when_add.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
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

        // SRA: funct3 == 0b101 (same as SRL).
        let mut when_sra = builder.when(local.instr_type.is_sra.clone());
        for i in 0..3 {
            let expected = if (0b101u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_sra.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // SRA: funct7 == 0b0100000 — bit 30 is 1, all other funct7 bits are 0.
        // funct7 = bits 25..32 of the instruction = bits 1..7 of byte 3 (0-indexed).
        // bit 30 of the instruction = bit 6 of byte 3 = decompositions[3][6].
        // So decompositions[3][1..6] == 0, decompositions[3][6] == 1, decompositions[3][7] == 0.
        let mut when_sra = builder.when(local.instr_type.is_sra.clone());
        for i in 1..6 {
            when_sra.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }
        let mut when_sra = builder.when(local.instr_type.is_sra.clone());
        when_sra.assert_eq(local.decompositions[3][6].clone(), AB::Expr::ONE);
        let mut when_sra = builder.when(local.instr_type.is_sra.clone());
        when_sra.assert_eq(local.decompositions[3][7].clone(), AB::Expr::ZERO);

        // SLLI: funct3 == 0b001, imm[11:5] == 0b0000000 (bits 25..32 all zero).
        // funct3 = bits 12..15 = bits 4..7 of byte 1.
        let mut when_slli = builder.when(local.instr_type.is_slli.clone());
        for i in 0..3 {
            let expected = if (0b001u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_slli.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }
        // imm[11:5] == 0 means bits 25..32 of the instruction word = bits 1..7 of byte 3.
        // For SLLI this is 0b0000000 (all zero).
        let mut when_slli = builder.when(local.instr_type.is_slli.clone());
        for i in 1..8 {
            when_slli.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SRLI: funct3 == 0b101, imm[11:5] == 0b0000000 (bits 25..32 all zero).
        let mut when_srli = builder.when(local.instr_type.is_srli.clone());
        for i in 0..3 {
            let expected = if (0b101u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_srli.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }
        let mut when_srli = builder.when(local.instr_type.is_srli.clone());
        for i in 1..8 {
            when_srli.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SRAI: funct3 == 0b101, imm[11:5] == 0b0100000 (bit 30 = 1, all others 0).
        // bit 30 of the instruction = bit 6 of byte 3 = decompositions[3][6].
        let mut when_srai = builder.when(local.instr_type.is_srai.clone());
        for i in 0..3 {
            let expected = if (0b101u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_srai.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }
        // bits 25..30 (decompositions[3][1..6]) == 0
        let mut when_srai = builder.when(local.instr_type.is_srai.clone());
        for i in 1..6 {
            when_srai.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }
        // bit 30 (decompositions[3][6]) == 1
        let mut when_srai = builder.when(local.instr_type.is_srai.clone());
        when_srai.assert_eq(local.decompositions[3][6].clone(), AB::Expr::ONE);
        // bit 31 (decompositions[3][7]) == 0
        let mut when_srai = builder.when(local.instr_type.is_srai.clone());
        when_srai.assert_eq(local.decompositions[3][7].clone(), AB::Expr::ZERO);

        // SLT: funct3 == 0b010, funct7 == 0b0000000.
        let mut when_slt = builder.when(local.instr_type.is_slt.clone());
        for i in 0..3 {
            let expected = if (0b010u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_slt.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }
        let mut when_slt = builder.when(local.instr_type.is_slt.clone());
        for i in 1..8 {
            when_slt.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SLTU: funct3 == 0b011, funct7 == 0b0000000.
        let mut when_sltu = builder.when(local.instr_type.is_sltu.clone());
        for i in 0..3 {
            let expected = if (0b011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_sltu.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }
        let mut when_sltu = builder.when(local.instr_type.is_sltu.clone());
        for i in 1..8 {
            when_sltu.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SLTI: funct3 == 0b010.
        let mut when_slti = builder.when(local.instr_type.is_slti.clone());
        for i in 0..3 {
            let expected = if (0b010u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_slti.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // SLTIU: funct3 == 0b011.
        let mut when_sltiu = builder.when(local.instr_type.is_sltiu.clone());
        for i in 0..3 {
            let expected = if (0b011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_sltiu.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // JAL: opcode == 0b1101111 (J-type).
        let mut when_jal = builder.when(local.instr_type.is_jal.clone());
        for i in 0..7 {
            let expected = if (0b1101111u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_jal.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // JALR: opcode == 0b1100111, funct3 == 0b000.
        let mut when_jalr = builder.when(local.instr_type.is_jalr.clone());
        for i in 0..7 {
            let expected = if (0b1100111u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_jalr.assert_eq(local.decompositions[0][i].clone(), expected);
        }
        let mut when_jalr = builder.when(local.instr_type.is_jalr.clone());
        for i in 0..3 {
            when_jalr.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
        }

        // LUI: opcode == 0b0110111.
        let mut when_lui = builder.when(local.instr_type.is_lui.clone());
        for i in 0..7 {
            let expected = if (0b0110111u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_lui.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // AUIPC: opcode == 0b0010111.
        let mut when_auipc = builder.when(local.instr_type.is_auipc.clone());
        for i in 0..7 {
            let expected = if (0b0010111u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_auipc.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // B-type instructions: opcode == 0b1100011.
        let is_b_type: AB::Expr = local.instr_type.is_beq.clone()
            + local.instr_type.is_bne.clone()
            + local.instr_type.is_blt.clone()
            + local.instr_type.is_bge.clone()
            + local.instr_type.is_bltu.clone()
            + local.instr_type.is_bgeu.clone();
        let mut when_b_type = builder.when(is_b_type);
        for i in 0..7 {
            let expected = if (0b1100011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_b_type.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // BEQ: funct3 == 0b000.
        let mut when_beq = builder.when(local.instr_type.is_beq.clone());
        for i in 0..3 {
            when_beq.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
        }

        // BNE: funct3 == 0b001.
        let mut when_bne = builder.when(local.instr_type.is_bne.clone());
        for i in 0..3 {
            let expected = if (0b001u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_bne.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // BLT: funct3 == 0b100.
        let mut when_blt = builder.when(local.instr_type.is_blt.clone());
        for i in 0..3 {
            let expected = if (0b100u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_blt.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // BGE: funct3 == 0b101.
        let mut when_bge = builder.when(local.instr_type.is_bge.clone());
        for i in 0..3 {
            let expected = if (0b101u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_bge.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // BLTU: funct3 == 0b110.
        let mut when_bltu = builder.when(local.instr_type.is_bltu.clone());
        for i in 0..3 {
            let expected = if (0b110u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_bltu.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // BGEU: funct3 == 0b111.
        let mut when_bgeu = builder.when(local.instr_type.is_bgeu.clone());
        for i in 0..3 {
            when_bgeu.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ONE);
        }

        // Load instructions: opcode == 0b0000011 (I-type, opcode=0x03).
        let is_load: AB::Expr = local.instr_type.is_lw.clone()
            + local.instr_type.is_lh.clone()
            + local.instr_type.is_lb.clone()
            + local.instr_type.is_lhu.clone()
            + local.instr_type.is_lbu.clone();
        let mut when_load = builder.when(is_load);
        for i in 0..7 {
            let expected = if (0b0000011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_load.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // LW: funct3 == 0b010.
        let mut when_lw = builder.when(local.instr_type.is_lw.clone());
        for i in 0..3 {
            let expected = if (0b010u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_lw.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // LH: funct3 == 0b001.
        let mut when_lh = builder.when(local.instr_type.is_lh.clone());
        for i in 0..3 {
            let expected = if (0b001u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_lh.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // LB: funct3 == 0b000.
        let mut when_lb = builder.when(local.instr_type.is_lb.clone());
        for i in 0..3 {
            when_lb.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
        }

        // LHU: funct3 == 0b101.
        let mut when_lhu = builder.when(local.instr_type.is_lhu.clone());
        for i in 0..3 {
            let expected = if (0b101u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_lhu.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // LBU: funct3 == 0b100.
        let mut when_lbu = builder.when(local.instr_type.is_lbu.clone());
        for i in 0..3 {
            let expected = if (0b100u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_lbu.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // imm_low8: lower 8 bits of the U-type 20-bit immediate (bits 19:12 of instruction).
        // = bits 15:12 (decompositions[1][4..7]) | bits 19:16 (decompositions[2][0..3]).
        let imm_low8_expr = pack_bits::<AB, 4>(
            &local.decompositions,
            &[
                (1, 4),
                (1, 5),
                (1, 6),
                (1, 7),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
            ],
        );
        // For U-type and JAL instructions, imm_low8 == the 8-bit immediate fragment (bits 19:12).
        let mut when_u_or_jal = builder.when(
            local.instr_type.is_lui.clone()
                + local.instr_type.is_auipc.clone()
                + local.instr_type.is_jal.clone(),
        );
        when_u_or_jal.assert_eq(local.imm_low8.clone(), imm_low8_expr);
        // For all other instructions, imm_low8 == 0.
        let is_not_u_or_jal: AB::Expr = AB::Expr::ONE
            - local.instr_type.is_lui.clone()
            - local.instr_type.is_auipc.clone()
            - local.instr_type.is_jal.clone();
        builder.assert_eq(local.imm_low8.clone() * is_not_u_or_jal, AB::Expr::ZERO);

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

        // instr_type_packed: 0=ADDI, 1=XORI, 2=ADD, 3=AND, 4=SLL, 5=SRL, 6=SRA,
        //                    7=SLLI, 8=SRLI, 9=SRAI, 10=SLT, 11=SLTU, 12=SLTI, 13=SLTIU,
        //                    14=LUI, 15=AUIPC, 16=JAL, 17=JALR,
        //                    18=BEQ, 19=BNE, 20=BLT, 21=BGE, 22=BLTU, 23=BGEU.
        let packed = local.instr_type.is_addi.clone() * AB::Expr::ZERO
            + local.instr_type.is_xori.clone() * AB::Expr::ONE
            + local.instr_type.is_add.clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.instr_type.is_and.clone() * AB::Expr::from(AB::F::from_u32(3))
            + local.instr_type.is_sll.clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.instr_type.is_srl.clone() * AB::Expr::from(AB::F::from_u32(5))
            + local.instr_type.is_sra.clone() * AB::Expr::from(AB::F::from_u32(6))
            + local.instr_type.is_slli.clone() * AB::Expr::from(AB::F::from_u32(7))
            + local.instr_type.is_srli.clone() * AB::Expr::from(AB::F::from_u32(8))
            + local.instr_type.is_srai.clone() * AB::Expr::from(AB::F::from_u32(9))
            + local.instr_type.is_slt.clone() * AB::Expr::from(AB::F::from_u32(10))
            + local.instr_type.is_sltu.clone() * AB::Expr::from(AB::F::from_u32(11))
            + local.instr_type.is_slti.clone() * AB::Expr::from(AB::F::from_u32(12))
            + local.instr_type.is_sltiu.clone() * AB::Expr::from(AB::F::from_u32(13))
            + local.instr_type.is_lui.clone() * AB::Expr::from(AB::F::from_u32(14))
            + local.instr_type.is_auipc.clone() * AB::Expr::from(AB::F::from_u32(15))
            + local.instr_type.is_jal.clone() * AB::Expr::from(AB::F::from_u32(16))
            + local.instr_type.is_jalr.clone() * AB::Expr::from(AB::F::from_u32(17))
            + local.instr_type.is_beq.clone() * AB::Expr::from(AB::F::from_u32(18))
            + local.instr_type.is_bne.clone() * AB::Expr::from(AB::F::from_u32(19))
            + local.instr_type.is_blt.clone() * AB::Expr::from(AB::F::from_u32(20))
            + local.instr_type.is_bge.clone() * AB::Expr::from(AB::F::from_u32(21))
            + local.instr_type.is_bltu.clone() * AB::Expr::from(AB::F::from_u32(22))
            + local.instr_type.is_bgeu.clone() * AB::Expr::from(AB::F::from_u32(23))
            + local.instr_type.is_lw.clone() * AB::Expr::from(AB::F::from_u32(24))
            + local.instr_type.is_lh.clone() * AB::Expr::from(AB::F::from_u32(25))
            + local.instr_type.is_lb.clone() * AB::Expr::from(AB::F::from_u32(26))
            + local.instr_type.is_lhu.clone() * AB::Expr::from(AB::F::from_u32(27))
            + local.instr_type.is_lbu.clone() * AB::Expr::from(AB::F::from_u32(28));
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

        // For the decode bus field4: use imm for standard I-type (ADDI/XORI/SLTI/SLTIU/JALR),
        // rs2 (= imm[4:0] = shamt) for shift-immediate instructions (SLLI/SRLI/SRAI),
        // rs2 for R-type instructions (ADD/AND/SLL/SRL/SRA/SLT/SLTU),
        // and imm (= bits 31:20) for U-type (LUI/AUIPC).
        // JAL uses the "decode_j" bus instead.
        let is_i_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_addi)
            + SymbolicExpression::from(local.instr_type.is_xori)
            + SymbolicExpression::from(local.instr_type.is_slti)
            + SymbolicExpression::from(local.instr_type.is_sltiu)
            + SymbolicExpression::from(local.instr_type.is_jalr)
            + SymbolicExpression::from(local.instr_type.is_lw)
            + SymbolicExpression::from(local.instr_type.is_lh)
            + SymbolicExpression::from(local.instr_type.is_lb)
            + SymbolicExpression::from(local.instr_type.is_lhu)
            + SymbolicExpression::from(local.instr_type.is_lbu);
        // shift-immediate: send rs2 (= shamt = imm[4:0]) as field4.
        let is_shift_imm: SymbolicExpression<F> =
            SymbolicExpression::from(local.instr_type.is_slli)
                + SymbolicExpression::from(local.instr_type.is_srli)
                + SymbolicExpression::from(local.instr_type.is_srai);
        let is_r_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_add)
            + SymbolicExpression::from(local.instr_type.is_and)
            + SymbolicExpression::from(local.instr_type.is_sll)
            + SymbolicExpression::from(local.instr_type.is_srl)
            + SymbolicExpression::from(local.instr_type.is_sra)
            + SymbolicExpression::from(local.instr_type.is_slt)
            + SymbolicExpression::from(local.instr_type.is_sltu);
        // U-type: send imm (bits 31:20) as field4; imm_low8 is on the "decode_u" bus.
        let is_u_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_lui)
            + SymbolicExpression::from(local.instr_type.is_auipc);
        // JAL: handled on the "decode_j" bus.
        let is_jal: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_jal);
        // B-type: handled on the "decode_b" bus.
        let is_b_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_beq)
            + SymbolicExpression::from(local.instr_type.is_bne)
            + SymbolicExpression::from(local.instr_type.is_blt)
            + SymbolicExpression::from(local.instr_type.is_bge)
            + SymbolicExpression::from(local.instr_type.is_bltu)
            + SymbolicExpression::from(local.instr_type.is_bgeu);
        let field4: SymbolicExpression<F> = is_i_type * SymbolicExpression::from(local.imm)
            + (is_r_type + is_shift_imm) * SymbolicExpression::from(local.rs2)
            + is_u_type.clone() * SymbolicExpression::from(local.imm);

        // Export the decoded instruction for non-U-type, non-JAL, non-B-type instructions.
        // U-type (LUI/AUIPC) rows use "decode_u" bus; JAL rows use "decode_j" bus;
        // B-type rows use "decode_b" bus.
        let is_standard: SymbolicExpression<F> = SymbolicExpression::from(F::ONE)
            - is_u_type.clone()
            - is_jal.clone()
            - is_b_type.clone();
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                vec![
                    local.instr_type_packed.into(),
                    local.rd.into(),
                    local.rs1.into(),
                    field4,
                ],
                (local.mult.clone() * is_standard).into(),
                Direction::Receive,
            )],
        ));

        // For U-type (LUI/AUIPC): export decoded instruction on the "decode_u" bus.
        // Schema: (instr_type_packed, rd, imm_high12, imm_low8)
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_u")),
            &vec![(
                vec![
                    local.instr_type_packed.into(),
                    local.rd.into(),
                    local.imm.into(),
                    local.imm_low8.into(),
                ],
                (local.mult.clone() * is_u_type).into(),
                Direction::Receive,
            )],
        ));

        // For JAL: export decoded instruction on the "decode_j" bus.
        // Schema: (instr_type_packed, rd, imm_high12, imm_lo8)
        // imm_high12 = bits 31:20 of instruction = {imm[20], imm[10:1], imm[11]} = local.imm
        // imm_lo8    = bits 19:12 of instruction = imm[19:12] = local.imm_low8
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_j")),
            &vec![(
                vec![
                    local.instr_type_packed.into(),
                    local.rd.into(),
                    local.imm.into(),
                    local.imm_low8.into(),
                ],
                (local.mult.clone() * is_jal).into(),
                Direction::Receive,
            )],
        ));

        // For B-type (BEQ/BNE/BLT/BGE/BLTU/BGEU): export decoded instruction on the "decode_b" bus.
        // Schema: (instr_type_packed, rs1, rs2, imm_top7, imm_lo5)
        // imm_top7 = {imm[12], imm[10:5]} packed as 7-bit value
        //          = decompositions[3][1..=7] packed (bit31=imm[12] is MSB, bits30:25=imm[10:5])
        // imm_lo5  = {imm[11], imm[4:1]} packed as 5-bit value
        //          = decompositions[1][0..=3] (imm[4:1]) and decompositions[0][7] (imm[11])
        let imm_top7_b: SymbolicExpression<F> =
            SymbolicExpression::from(local.decompositions[3][1]) * SymbolicExpression::from(F::ONE)
                + SymbolicExpression::from(local.decompositions[3][2])
                    * SymbolicExpression::from(F::from_u64(2))
                + SymbolicExpression::from(local.decompositions[3][3])
                    * SymbolicExpression::from(F::from_u64(4))
                + SymbolicExpression::from(local.decompositions[3][4])
                    * SymbolicExpression::from(F::from_u64(8))
                + SymbolicExpression::from(local.decompositions[3][5])
                    * SymbolicExpression::from(F::from_u64(16))
                + SymbolicExpression::from(local.decompositions[3][6])
                    * SymbolicExpression::from(F::from_u64(32))
                + SymbolicExpression::from(local.decompositions[3][7])
                    * SymbolicExpression::from(F::from_u64(64));
        let imm_lo5_b: SymbolicExpression<F> = SymbolicExpression::from(local.decompositions[1][0])
            * SymbolicExpression::from(F::ONE)
            + SymbolicExpression::from(local.decompositions[1][1])
                * SymbolicExpression::from(F::from_u64(2))
            + SymbolicExpression::from(local.decompositions[1][2])
                * SymbolicExpression::from(F::from_u64(4))
            + SymbolicExpression::from(local.decompositions[1][3])
                * SymbolicExpression::from(F::from_u64(8))
            + SymbolicExpression::from(local.decompositions[0][7])
                * SymbolicExpression::from(F::from_u64(16));
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_b")),
            &vec![(
                vec![
                    local.instr_type_packed.into(),
                    local.rs1.into(),
                    local.rs2.into(),
                    imm_top7_b,
                    imm_lo5_b,
                ],
                (local.mult.clone() * is_b_type).into(),
                Direction::Receive,
            )],
        ));

        lookups
    }
}
