use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{
    GenericPoseidon2LinearLayersMersenne31, MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL,
    MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL, MERSENNE31_POSEIDON2_RC_16_INTERNAL, Mersenne31,
};
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::FullRound;

use super::columns::{
    BYTES_PER_RATE_ELEM, BYTES_PER_ROW, DIGEST_LEN, HALF_FULL_ROUNDS, NUM_COLS, P2Cols,
    PARTIAL_ROUNDS, ProgramHashColumns, RATE_ELEMS, WIDTH,
};

/// Fill the Poseidon2 witness columns for one permutation invocation. Mirrors
/// `p3_poseidon2_air::generate_trace_rows_for_perm` but works on initialized
/// columns (we wrote zeros earlier) and uses concrete Mersenne31 constants.
fn fill_perm(perm: &mut P2Cols<Mersenne31>, state_in: [Mersenne31; WIDTH]) {
    perm.inputs = state_in;
    let mut state = state_in;
    GenericPoseidon2LinearLayersMersenne31::external_linear_layer::<Mersenne31>(&mut state);

    for r in 0..HALF_FULL_ROUNDS {
        fill_full_round(
            &mut state,
            &mut perm.beginning_full_rounds[r],
            &MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL[r],
        );
    }
    for r in 0..PARTIAL_ROUNDS {
        state[0] += MERSENNE31_POSEIDON2_RC_16_INTERNAL[r];
        let x = state[0];
        let x2 = x * x;
        let x3 = x2 * x;
        perm.partial_rounds[r].sbox.0[0] = x3;
        state[0] = x3 * x2;
        perm.partial_rounds[r].post_sbox = state[0];
        GenericPoseidon2LinearLayersMersenne31::internal_linear_layer::<Mersenne31>(&mut state);
    }
    for r in 0..HALF_FULL_ROUNDS {
        fill_full_round(
            &mut state,
            &mut perm.ending_full_rounds[r],
            &MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL[r],
        );
    }
}

fn fill_full_round(
    state: &mut [Mersenne31; WIDTH],
    round: &mut FullRound<Mersenne31, WIDTH, 5, 1>,
    rc: &[Mersenne31; WIDTH],
) {
    for i in 0..WIDTH {
        state[i] += rc[i];
        let x = state[i];
        let x2 = x * x;
        let x3 = x2 * x;
        round.sbox[i].0[0] = x3;
        state[i] = x3 * x2;
    }
    GenericPoseidon2LinearLayersMersenne31::external_linear_layer::<Mersenne31>(state);
    round.post = *state;
}

/// Apply one Poseidon2-Mersenne31<WIDTH=16> permutation in-host. Mirrors the
/// round chain the AIR constrains.
fn permute_in_host(state: &mut [Mersenne31; WIDTH]) {
    GenericPoseidon2LinearLayersMersenne31::external_linear_layer::<Mersenne31>(state);
    for r in 0..HALF_FULL_ROUNDS {
        for i in 0..WIDTH {
            state[i] += MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL[r][i];
            let x = state[i];
            state[i] = x * x * x * x * x;
        }
        GenericPoseidon2LinearLayersMersenne31::external_linear_layer::<Mersenne31>(state);
    }
    for r in 0..PARTIAL_ROUNDS {
        state[0] += MERSENNE31_POSEIDON2_RC_16_INTERNAL[r];
        let x = state[0];
        state[0] = x * x * x * x * x;
        GenericPoseidon2LinearLayersMersenne31::internal_linear_layer::<Mersenne31>(state);
    }
    for r in 0..HALF_FULL_ROUNDS {
        for i in 0..WIDTH {
            state[i] += MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL[r][i];
            let x = state[i];
            state[i] = x * x * x * x * x;
        }
        GenericPoseidon2LinearLayersMersenne31::external_linear_layer::<Mersenne31>(state);
    }
}

/// Pack 3 bytes from `program[start..start + 3]` (zero-extended past the end)
/// into a single u32 lane: `b0 + 256*b1 + 65536*b2`.
fn pack_lane(program: &[u8], start: usize) -> u32 {
    let mut acc: u32 = 0;
    for k in 0..BYTES_PER_RATE_ELEM {
        if start + k < program.len() {
            acc |= (program[start + k] as u32) << (8 * k);
        }
    }
    acc
}

/// Compute the program-image Poseidon2 digest in-host. Used by trace generation
/// (and tests) to derive the `(digest, length)` public-value pair.
///
/// Absorption is BOTTOM-UP to match the AIR's bottom-up chaining: chunks of 24
/// bytes are absorbed in reverse trace order (program tail first, into INIT;
/// program head absorbed last; digest is the final state).
pub fn compute_program_digest(program: &[u8]) -> ([Mersenne31; DIGEST_LEN], u32) {
    let n = program.len();
    let num_real_rows = n.div_ceil(BYTES_PER_ROW);

    let mut state = [Mersenne31::ZERO; WIDTH];
    for row_idx in (0..num_real_rows).rev() {
        let base = row_idx * BYTES_PER_ROW;
        for g in 0..RATE_ELEMS {
            let packed = pack_lane(program, base + g * BYTES_PER_RATE_ELEM);
            state[g] += Mersenne31::from_u32(packed);
        }
        permute_in_host(&mut state);
    }

    let mut digest = [Mersenne31::ZERO; DIGEST_LEN];
    digest.copy_from_slice(&state[..DIGEST_LEN]);
    (digest, n as u32)
}

/// Build the program-hash trace.
///
/// Layout: `num_real_rows = ceil(n / 24)` real rows (flag=1) followed by ≥1
/// padding rows (flag=0). The bottom-up sponge chain runs from INIT at the
/// first padding row, accumulating up through the real rows. Padding rows have
/// `state = 0`, `perm = 0`, and zero Poseidon work; constraints on padding
/// rows are gated off by `flag = 0`.
pub fn build_trace(program: &[u8]) -> RowMajorMatrix<Mersenne31> {
    let n = program.len();
    assert!(n > 0, "program must be non-empty");
    let num_real_rows = n.div_ceil(BYTES_PER_ROW);
    // At least one padding row at the end (`when_last_row: flag = 0` pins the
    // INIT base of the chain). Round up to a power of two ≥ 4 (CirclePCS
    // minimum) and ≥ num_real_rows + 1.
    let num_rows = (num_real_rows + 1).next_power_of_two().max(4);

    let mut values = vec![Mersenne31::ZERO; num_rows * NUM_COLS];
    {
        let (prefix, rows, suffix) =
            unsafe { values.align_to_mut::<ProgramHashColumns<Mersenne31>>() };
        assert!(prefix.is_empty(), "alignment mismatch");
        assert!(suffix.is_empty(), "alignment mismatch");
        assert_eq!(rows.len(), num_rows);

        // ── Pass 1: fill local-only fields (base_addr, bytes/is_active, flag,
        // cum_active). The base_addr chain runs continuously across all rows.
        let mut cum_active: u32 = 0;
        for (row_idx, row) in rows.iter_mut().enumerate() {
            row.base_addr = Mersenne31::from_u32((row_idx * BYTES_PER_ROW) as u32);

            if row_idx < num_real_rows {
                let base = row_idx * BYTES_PER_ROW;
                for i in 0..BYTES_PER_ROW {
                    let global_addr = base + i;
                    if global_addr < n {
                        row.bytes[i] = Mersenne31::from_u64(program[global_addr] as u64);
                        row.is_active[i] = Mersenne31::ONE;
                        cum_active += 1;
                    }
                }
                row.flag = Mersenne31::ONE;
            } else {
                row.flag = Mersenne31::ZERO;
            }
            row.cum_active = Mersenne31::from_u32(cum_active);
        }

        // ── Pass 2: bottom-up sponge chain. Start at INIT (= [0; WIDTH]) below
        // the lowest real row, absorb each row's chunk, and write the post-
        // permutation state back into the row's `state` column. Padding rows
        // keep `state = 0` and `perm = 0` (already zeroed by `vec![0; ..]`).
        let mut acc = [Mersenne31::ZERO; WIDTH];
        for row_idx in (0..num_real_rows).rev() {
            let row = &mut rows[row_idx];
            let mut perm_in = acc;
            for g in 0..RATE_ELEMS {
                let base = row_idx * BYTES_PER_ROW + g * BYTES_PER_RATE_ELEM;
                let packed = pack_lane(program, base);
                perm_in[g] += Mersenne31::from_u32(packed);
            }
            fill_perm(&mut row.perm, perm_in);
            acc = row.perm.ending_full_rounds[HALF_FULL_ROUNDS - 1].post;
            row.state = acc;
        }
    }

    RowMajorMatrix::new(values, NUM_COLS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::borrow::Borrow;

    fn trace_digest(program: &[u8]) -> [Mersenne31; DIGEST_LEN] {
        // In bottom-up chaining, the digest is at row 0 (top of the chain).
        let matrix = build_trace(program);
        let row_slice: &[Mersenne31] = &matrix.values[..NUM_COLS];
        let row: &ProgramHashColumns<Mersenne31> = row_slice.borrow();
        let mut out = [Mersenne31::ZERO; DIGEST_LEN];
        out.copy_from_slice(&row.state[..DIGEST_LEN]);
        out
    }

    #[test]
    fn digest_matches_host_computation_for_various_lengths() {
        // Boundary cases around the 24-byte chunking plus a larger program.
        for n in [1usize, 7, 23, 24, 25, 48, 49, 128] {
            let program: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)).collect();
            let (expected_digest, expected_len) = compute_program_digest(&program);
            assert_eq!(expected_len as usize, n);
            let trace = trace_digest(&program);
            assert_eq!(trace, expected_digest, "digest mismatch at n={}", n);
        }
    }

    #[test]
    #[should_panic(expected = "program must be non-empty")]
    fn empty_program_rejected() {
        let _ = build_trace(&[]);
    }
}
