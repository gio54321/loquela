use super::air::{DecodeColumns, Instruction, NUM_DECODE_COLS};
use loquela_vm::ExecutionStep;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

fn u32_to_limbs<F: PrimeCharacteristicRing>(v: u32) -> [F; 4] {
    let b = v.to_le_bytes();
    [
        F::from_u64(b[0] as u64),
        F::from_u64(b[1] as u64),
        F::from_u64(b[2] as u64),
        F::from_u64(b[3] as u64),
    ]
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut DecodeColumns<F>, pc: u32, word: u32) {
    row.pc = u32_to_limbs(pc);

    for i in 0..4 {
        let byte = (word >> (i * 8)) & 0xFF;
        row.instruction[i] = F::from_u64(byte as u64);
        for j in 0..8 {
            row.decompositions[i][j] = F::from_u64(((byte >> j) & 1) as u64);
        }
    }

    let rd = ((word >> 7) & 0x1F) as u8;
    let rs1 = ((word >> 15) & 0x1F) as u8;
    let imm = ((word >> 20) & 0xFFF) as u16;
    let rs2 = ((word >> 20) & 0x1F) as u8;

    row.rd = F::from_u64(rd as u64);
    row.rs1 = F::from_u64(rs1 as u64);
    row.imm = F::from_u64(imm as u64);
    row.rs2 = F::from_u64(rs2 as u64);

    let is_addi = word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b000;
    let is_xori = word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b100;
    let is_add =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b000 && (word >> 25) == 0b000_0000;
    let is_and =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b111 && (word >> 25) == 0b000_0000;
    let is_sll =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b001 && (word >> 25) == 0b000_0000;
    let is_srl =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b101 && (word >> 25) == 0b000_0000;
    let is_sra =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b101 && (word >> 25) == 0b010_0000;
    let is_slli =
        word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b001 && (word >> 25) == 0b000_0000;
    let is_srli =
        word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b101 && (word >> 25) == 0b000_0000;
    let is_srai =
        word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b101 && (word >> 25) == 0b010_0000;
    let is_slt =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b010 && (word >> 25) == 0b000_0000;
    let is_sltu =
        word & 0x7F == 0b011_0011 && (word >> 12) & 0x7 == 0b011 && (word >> 25) == 0b000_0000;
    let is_slti = word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b010;
    let is_sltiu = word & 0x7F == 0b001_0011 && (word >> 12) & 0x7 == 0b011;
    // U-type instructions: opcode only (no funct3/funct7).
    let is_lui = word & 0x7F == 0b011_0111;
    let is_auipc = word & 0x7F == 0b001_0111;

    // For U-type, imm_low8 = bits 19:12 of the instruction word.
    let imm_low8 = if is_lui || is_auipc {
        (word >> 12) & 0xFF
    } else {
        0
    };
    row.imm_low8 = F::from_u64(imm_low8 as u64);

    row.instr_type = Instruction {
        is_addi: F::from_bool(is_addi),
        is_xori: F::from_bool(is_xori),
        is_add: F::from_bool(is_add),
        is_and: F::from_bool(is_and),
        is_sll: F::from_bool(is_sll),
        is_srl: F::from_bool(is_srl),
        is_sra: F::from_bool(is_sra),
        is_slli: F::from_bool(is_slli),
        is_srli: F::from_bool(is_srli),
        is_srai: F::from_bool(is_srai),
        is_slt: F::from_bool(is_slt),
        is_sltu: F::from_bool(is_sltu),
        is_slti: F::from_bool(is_slti),
        is_sltiu: F::from_bool(is_sltiu),
        is_lui: F::from_bool(is_lui),
        is_auipc: F::from_bool(is_auipc),
    };
    row.instr_type_packed = if is_xori {
        F::ONE
    } else if is_add {
        F::from_u64(2)
    } else if is_and {
        F::from_u64(3)
    } else if is_sll {
        F::from_u64(4)
    } else if is_srl {
        F::from_u64(5)
    } else if is_sra {
        F::from_u64(6)
    } else if is_slli {
        F::from_u64(7)
    } else if is_srli {
        F::from_u64(8)
    } else if is_srai {
        F::from_u64(9)
    } else if is_slt {
        F::from_u64(10)
    } else if is_sltu {
        F::from_u64(11)
    } else if is_slti {
        F::from_u64(12)
    } else if is_sltiu {
        F::from_u64(13)
    } else if is_lui {
        F::from_u64(14)
    } else if is_auipc {
        F::from_u64(15)
    } else {
        F::ZERO
    };
    row.mult = F::ONE;
}

/// Build the decode trace from the VM execution steps.
///
/// One row per step. The `mult` column is always 1 (each decode is consumed
/// once by the corresponding instruction AIR). Padding rows reuse the first
/// step's instruction word with mult=0 so the decode bus stays balanced.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> RowMajorMatrix<F> {
    let num_steps = steps.len();
    assert!(num_steps > 0, "decode trace requires at least one step");

    let num_rows = num_steps.next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_DECODE_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<DecodeColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(steps.iter()) {
        fill_row(row, step.state.pc, step.instruction_word);
    }
    // Padding rows: reuse the first step's instruction word with mult=0.
    let padding_word = steps[0].instruction_word;
    for row in rows.iter_mut().skip(num_steps) {
        fill_row(row, 0, padding_word);
        row.mult = F::ZERO;
    }

    RowMajorMatrix::new(values, NUM_DECODE_COLS)
}
