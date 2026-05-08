use loquela_vm::{ExecutionStep, Instruction, MemoryOperation, MemoryType};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{SwColumns, NUM_SW_COLS};

struct SwStep {
    pc: u32,
    timestamp: u32,
    rs1: u8,
    rs2: u8,
    imm: i32,
    rs1_value: u32,
    rs2_value: u32,
    addr: u32,
    old_ram_value: u32,
}

fn extract_sw_steps(steps: &[ExecutionStep]) -> Vec<SwStep> {
    steps
        .iter()
        .filter_map(|step| {
            let (rs1, rs2, imm) = match step.instruction {
                Instruction::Sw { rs1, rs2, imm } => (rs1, rs2, imm),
                _ => return None,
            };
            let (timestamp, rs1_value, rs2_value, old_ram_value) = match step.memory_ops.as_slice()
            {
                [MemoryOperation::Read {
                    timestamp,
                    value: v1,
                    memory_type: MemoryType::Register,
                    ..
                }, MemoryOperation::Read {
                    value: v2,
                    memory_type: MemoryType::Register,
                    ..
                }, MemoryOperation::Write {
                    old_value,
                    memory_type: MemoryType::Ram,
                    ..
                }] => (*timestamp, *v1, *v2, *old_value),
                _ => return None,
            };
            let addr = rs1_value.wrapping_add(imm as u32);
            Some(SwStep {
                pc: step.state.pc,
                timestamp,
                rs1,
                rs2,
                imm,
                rs1_value,
                rs2_value,
                addr,
                old_ram_value,
            })
        })
        .collect()
}

fn u32_to_limbs<F: PrimeCharacteristicRing>(v: u32) -> [F; 4] {
    let b = v.to_le_bytes();
    [
        F::from_u64(b[0] as u64),
        F::from_u64(b[1] as u64),
        F::from_u64(b[2] as u64),
        F::from_u64(b[3] as u64),
    ]
}

fn pc_plus4_carries(pc: u32) -> [u8; 3] {
    let b = pc.to_le_bytes();
    let s0 = b[0] as u32 + 4;
    let c0 = (s0 >> 8) as u8;
    let s1 = b[1] as u32 + c0 as u32;
    let c1 = (s1 >> 8) as u8;
    let s2 = b[2] as u32 + c1 as u32;
    let c2 = (s2 >> 8) as u8;
    [c0, c1, c2]
}

fn u32_add_carries(x: u32, y: u32) -> [u8; 4] {
    let xb = x.to_le_bytes();
    let yb = y.to_le_bytes();
    let mut carries = [0u8; 4];
    let mut carry = 0u32;
    for i in 0..4 {
        let sum = xb[i] as u32 + yb[i] as u32 + carry;
        carries[i] = (sum >> 8) as u8;
        carry = carries[i] as u32;
    }
    carries
}

/// Sign-extend a 12-bit immediate to 32 bits, returning unsigned 32-bit bytes.
fn imm_se_bytes(imm: i32) -> [u8; 4] {
    (imm as u32).to_le_bytes()
}

fn fill_row<F: PrimeCharacteristicRing>(row: &mut SwColumns<F>, step: &SwStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.rs1 = F::from_u64(step.rs1 as u64);
    row.rs2 = F::from_u64(step.rs2 as u64);

    // The 12-bit unsigned immediate value as stored in the decode_s bus.
    let imm_unsigned = (step.imm as u32) & 0xFFF;
    row.imm_s = F::from_u64(imm_unsigned as u64);

    // High nibble bits of the 12-bit immediate (bits 8–11).
    let high_nibble = (imm_unsigned >> 8) & 0xF;
    row.imm_high_bits = [
        F::from_u64((high_nibble & 1) as u64),
        F::from_u64(((high_nibble >> 1) & 1) as u64),
        F::from_u64(((high_nibble >> 2) & 1) as u64),
        F::from_u64(((high_nibble >> 3) & 1) as u64),
    ];

    let se = imm_se_bytes(step.imm);
    row.imm_se_bytes = [
        F::from_u64(se[0] as u64),
        F::from_u64(se[1] as u64),
        F::from_u64(se[2] as u64),
        F::from_u64(se[3] as u64),
    ];

    row.rs1_value = u32_to_limbs(step.rs1_value);
    row.rs2_value = u32_to_limbs(step.rs2_value);
    row.addr = u32_to_limbs(step.addr);
    row.old_ram_value = u32_to_limbs(step.old_ram_value);

    let imm_se_u32 = step.imm as u32;
    let carries = u32_add_carries(step.rs1_value, imm_se_u32);
    row.addr_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
        F::from_u64(carries[3] as u64),
    ];

    row.next_pc = u32_to_limbs(step.pc + 4);
    let pc_carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(pc_carries[0] as u64),
        F::from_u64(pc_carries[1] as u64),
        F::from_u64(pc_carries[2] as u64),
    ];

    row.is_dummy = F::ONE;
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut SwColumns<F>) {
    *row = SwColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        rs1: F::ZERO,
        rs2: F::ZERO,
        imm_s: F::ZERO,
        imm_high_bits: [F::ZERO; 4],
        imm_se_bytes: [F::ZERO; 4],
        rs1_value: [F::ZERO; 4],
        rs2_value: [F::ZERO; 4],
        addr: [F::ZERO; 4],
        addr_carries: [F::ZERO; 4],
        old_ram_value: [F::ZERO; 4],
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Build the SW execution trace from the VM execution steps.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let sw_steps = extract_sw_steps(steps);
    assert!(!sw_steps.is_empty(), "no SW steps found in trace");

    let num_rows = sw_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_SW_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<SwColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(sw_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(sw_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_SW_COLS)
}
