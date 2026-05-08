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
    pub is_sw: F,
    pub is_sh: F,
    pub is_sb: F,
}

#[repr(u8)]
pub enum InstructionId {
    Addi = 0,
    Xori = 1,
    Add = 2,
    Sw = 3,
    Sh = 4,
    Sb = 5,
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
    /// Unsigned 12-bit immediate (I-type). Holds bits 20–31 for ADDI/XORI; unconstrained for ADD/S-type.
    pub imm: F,
    /// Source register 2 index (R-type and S-type). Holds bits 20–24 for all instruction types.
    pub rs2: F,
    /// S-type immediate (unsigned 12-bit). Bits 11:5 from instruction[31:25], bits 4:0 from instruction[11:7].
    /// Constrained only for S-type instructions (SW, SH, SB); unconstrained for other types.
    pub imm_s: F,
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
        builder.assert_bool(local.instr_type.is_sw.clone());
        builder.assert_bool(local.instr_type.is_sh.clone());
        builder.assert_bool(local.instr_type.is_sb.clone());
        builder.assert_eq(
            local.instr_type.is_addi.clone()
                + local.instr_type.is_xori.clone()
                + local.instr_type.is_add.clone()
                + local.instr_type.is_sw.clone()
                + local.instr_type.is_sh.clone()
                + local.instr_type.is_sb.clone(),
            AB::Expr::ONE,
        );

        // Bit-decompose each instruction byte limb.
        for (i, limb) in local.instruction.iter().enumerate() {
            check_bit_decomposition(builder, limb.clone(), &local.decompositions[i]);
        }

        // Opcode (bits 0..7) == 0b0010011 for ADDI and XORI (I-type).
        let mut when_op_immediate =
            builder.when(local.instr_type.is_addi.clone() + local.instr_type.is_xori.clone());
        for i in 0..7 {
            let expected = if (0b0010011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_op_immediate.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // Opcode (bits 0..7) == 0b0110011 for ADD (R-type).
        let mut when_add = builder.when(local.instr_type.is_add.clone());
        for i in 0..7 {
            let expected = if (0b0110011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_add.assert_eq(local.decompositions[0][i].clone(), expected);
        }

        // Opcode (bits 0..7) == 0b0100011 for SW, SH, SB (S-type).
        let is_s_type: AB::Expr = local.instr_type.is_sw.clone()
            + local.instr_type.is_sh.clone()
            + local.instr_type.is_sb.clone();
        let mut when_s_type = builder.when(is_s_type);
        for i in 0..7 {
            let expected = if (0b0100011u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_s_type.assert_eq(local.decompositions[0][i].clone(), expected);
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

        // ADD: funct7 (bits 25..32) == 0b0000000 — bits 1..7 of byte 3.
        let mut when_add = builder.when(local.instr_type.is_add.clone());
        for i in 1..8 {
            when_add.assert_eq(local.decompositions[3][i].clone(), AB::Expr::ZERO);
        }

        // SW: funct3 == 0b010
        let mut when_sw = builder.when(local.instr_type.is_sw.clone());
        for i in 0..3 {
            let expected = if (0b010u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_sw.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // SH: funct3 == 0b001
        let mut when_sh = builder.when(local.instr_type.is_sh.clone());
        for i in 0..3 {
            let expected = if (0b001u32 >> i) & 1 == 1 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            when_sh.assert_eq(local.decompositions[1][4 + i].clone(), expected);
        }

        // SB: funct3 == 0b000
        let mut when_sb = builder.when(local.instr_type.is_sb.clone());
        for i in 0..3 {
            when_sb.assert_eq(local.decompositions[1][4 + i].clone(), AB::Expr::ZERO);
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

        // imm_s = S-type immediate: bits [11:5] from instruction[31:25], bits [4:0] from instruction[11:7].
        // imm_s[0] = bit7  = dec[0][7]
        // imm_s[1] = bit8  = dec[1][0]
        // imm_s[2] = bit9  = dec[1][1]
        // imm_s[3] = bit10 = dec[1][2]
        // imm_s[4] = bit11 = dec[1][3]
        // imm_s[5] = bit25 = dec[3][1]
        // imm_s[6] = bit26 = dec[3][2]
        // imm_s[7] = bit27 = dec[3][3]
        // imm_s[8] = bit28 = dec[3][4]
        // imm_s[9] = bit29 = dec[3][5]
        // imm_s[10]= bit30 = dec[3][6]
        // imm_s[11]= bit31 = dec[3][7]
        let imm_s_expr = pack_bits::<AB, 4>(
            &local.decompositions,
            &[
                (0, 7),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (3, 7),
            ],
        );
        builder.assert_eq(local.imm_s.clone(), imm_s_expr);

        // instr_type_packed: 0=ADDI, 1=XORI, 2=ADD, 3=SW, 4=SH, 5=SB.
        let packed = local.instr_type.is_addi.clone() * AB::Expr::ZERO
            + local.instr_type.is_xori.clone() * AB::Expr::ONE
            + local.instr_type.is_add.clone() * AB::Expr::from(AB::F::from_u32(2))
            + local.instr_type.is_sw.clone() * AB::Expr::from(AB::F::from_u32(3))
            + local.instr_type.is_sh.clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.instr_type.is_sb.clone() * AB::Expr::from(AB::F::from_u32(5));
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
        // is_i_type = is_addi + is_xori (one-hot so sum is 0 or 1, safe to use as multiplier).
        let is_i_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_addi)
            + SymbolicExpression::from(local.instr_type.is_xori);
        let field4: SymbolicExpression<F> = is_i_type * SymbolicExpression::from(local.imm)
            + SymbolicExpression::from(local.instr_type.is_add)
                * SymbolicExpression::from(local.rs2);

        // is_s_type = is_sw + is_sh + is_sb (mutually exclusive, sum is 0 or 1).
        let is_s_type: SymbolicExpression<F> = SymbolicExpression::from(local.instr_type.is_sw)
            + SymbolicExpression::from(local.instr_type.is_sh)
            + SymbolicExpression::from(local.instr_type.is_sb);

        // Export the decoded instruction on the "decode" bus for non-S-type instructions.
        // Multiplicity = mult * (1 - is_s_type).
        let decode_mult: SymbolicExpression<F> = SymbolicExpression::from(local.mult)
            * (SymbolicExpression::from(F::ONE) - is_s_type.clone());
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                vec![
                    local.instr_type_packed.into(),
                    local.rd.into(),
                    local.rs1.into(),
                    field4,
                ],
                decode_mult,
                Direction::Receive,
            )],
        ));

        // Export the decoded instruction on the "decode_s" bus for S-type instructions.
        // Schema: (instr_type_packed, rs1, rs2, imm_s).
        // Multiplicity = mult * is_s_type.
        let decode_s_mult: SymbolicExpression<F> = SymbolicExpression::from(local.mult) * is_s_type;
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_s")),
            &vec![(
                vec![
                    local.instr_type_packed.into(),
                    local.rs1.into(),
                    local.rs2.into(),
                    local.imm_s.into(),
                ],
                decode_s_mult,
                Direction::Receive,
            )],
        ));

        lookups
    }
}
