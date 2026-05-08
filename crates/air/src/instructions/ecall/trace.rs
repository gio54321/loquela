use loquela_vm::{ExecutionStep, Instruction};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;

use super::air::{EcallColumns, NUM_ECALL_COLS};

struct EcallStep {
    pc: u32,
    timestamp: u32,
    is_ecall: bool,
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

/// Carry bits produced when computing `pc + 4` byte by byte (matching `u32_plus_four`).
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

fn fill_row<F: PrimeCharacteristicRing>(row: &mut EcallColumns<F>, step: &EcallStep) {
    row.pc = u32_to_limbs(step.pc);
    row.timestamp = F::from_u64(step.timestamp as u64);
    row.is_ecall = F::from_bool(step.is_ecall);
    row.next_pc = u32_to_limbs(step.pc + 4);
    let carries = pc_plus4_carries(step.pc);
    row.next_pc_carries = [
        F::from_u64(carries[0] as u64),
        F::from_u64(carries[1] as u64),
        F::from_u64(carries[2] as u64),
    ];
    row.is_dummy = F::ONE;
}

fn fill_padding_row<F: PrimeCharacteristicRing>(row: &mut EcallColumns<F>) {
    *row = EcallColumns {
        pc: [F::ZERO; 4],
        timestamp: F::ZERO,
        is_ecall: F::ZERO,
        next_pc: [F::from_u64(4), F::ZERO, F::ZERO, F::ZERO],
        next_pc_carries: [F::ZERO; 3],
        is_dummy: F::ZERO,
    };
}

/// Compute the entry timestamp for each step by accumulating memory op counts.
///
/// The VM starts at timestamp 0 and increments it once per memory operation.
/// ECALL/EBREAK emits zero memory ops, so its entry timestamp equals the
/// total number of memory ops in all preceding steps.
fn entry_timestamps(steps: &[ExecutionStep]) -> Vec<u32> {
    let mut ts = 0u32;
    steps
        .iter()
        .map(|step| {
            let entry = ts;
            ts += step.memory_ops.len() as u32;
            entry
        })
        .collect()
}

/// Build the ECALL/EBREAK execution trace from the VM execution steps.
///
/// Filters all ECALL and EBREAK steps, fills one row per step, and pads to the
/// next power of two.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    steps: &[ExecutionStep],
) -> DenseMatrix<F> {
    let timestamps = entry_timestamps(steps);

    let ecall_steps: Vec<EcallStep> = steps
        .iter()
        .enumerate()
        .filter_map(|(i, step)| match step.instruction {
            Instruction::Ecall => Some(EcallStep {
                pc: step.state.pc,
                timestamp: timestamps[i],
                is_ecall: true,
            }),
            Instruction::Ebreak => Some(EcallStep {
                pc: step.state.pc,
                timestamp: timestamps[i],
                is_ecall: false,
            }),
            _ => None,
        })
        .collect();

    assert!(
        !ecall_steps.is_empty(),
        "no ECALL/EBREAK steps found in trace"
    );

    let num_rows = ecall_steps.len().next_power_of_two().max(4);
    let mut values = vec![F::ZERO; num_rows * NUM_ECALL_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<EcallColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (row, step) in rows.iter_mut().zip(ecall_steps.iter()) {
        fill_row(row, step);
    }
    for row in rows.iter_mut().skip(ecall_steps.len()) {
        fill_padding_row(row);
    }

    DenseMatrix::new(values, NUM_ECALL_COLS)
}
