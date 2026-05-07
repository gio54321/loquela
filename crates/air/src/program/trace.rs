use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use punctum_vm::ExecutionStep;

use super::air::{ProgramColumns, NUM_MEMORY_COLS};

fn u32_to_limbs<F: PrimeCharacteristicRing>(v: u32) -> [F; 4] {
    let b = v.to_le_bytes();
    [
        F::from_u64(b[0] as u64),
        F::from_u64(b[1] as u64),
        F::from_u64(b[2] as u64),
        F::from_u64(b[3] as u64),
    ]
}

/// Carry bits when computing `addr + 1` byte by byte (for `u32_inc`).
fn inc_carries(addr: u32) -> [u8; 4] {
    let b = addr.to_le_bytes();
    let s0 = b[0] as u32 + 1;
    let c0 = (s0 >> 8) as u8;
    let s1 = b[1] as u32 + c0 as u32;
    let c1 = (s1 >> 8) as u8;
    let s2 = b[2] as u32 + c1 as u32;
    let c2 = (s2 >> 8) as u8;
    let s3 = b[3] as u32 + c2 as u32;
    let c3 = (s3 >> 8) as u8;
    [c0, c1, c2, c3]
}

/// Build the program ROM trace.
///
/// One row per byte in the program, padded to the next power of two.
/// `mult[i]` counts how many times byte address `i` was fetched by the decode AIR:
/// each step at PC `p` sends 4 fetches for bytes `p`, `p+1`, `p+2`, `p+3`.
///
/// `num_decode_padding` additional fetches at PC=0 (bytes 0–3) are added for
/// decode-trace padding rows; pass 0 if not accounting for padding.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    program: &[u8],
    steps: &[ExecutionStep],
    num_decode_padding: usize,
) -> RowMajorMatrix<F> {
    let n_bytes = program.len();
    assert!(n_bytes > 0);
    let num_rows = n_bytes.next_power_of_two().max(4);

    let mut mults = vec![0usize; num_rows];
    for step in steps {
        for i in 0..4 {
            let addr = step.state.pc as usize + i;
            if addr < num_rows {
                mults[addr] += 1;
            }
        }
    }
    for i in 0..4.min(num_rows) {
        mults[i] += num_decode_padding;
    }

    let mut values = vec![F::ZERO; num_rows * NUM_MEMORY_COLS];
    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<ProgramColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (i, row) in rows.iter_mut().enumerate() {
        row.address = u32_to_limbs(i as u32);
        row.value = if i < n_bytes {
            F::from_u64(program[i] as u64)
        } else {
            F::ZERO
        };
        row.mult = F::from_u64(mults[i] as u64);
        let c = inc_carries(i as u32);
        row.inc_carries = [
            F::from_u64(c[0] as u64),
            F::from_u64(c[1] as u64),
            F::from_u64(c[2] as u64),
            F::from_u64(c[3] as u64),
        ];
    }

    RowMajorMatrix::new(values, NUM_MEMORY_COLS)
}
