use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use p3_poseidon2_air::generate_trace_rows;

use super::air::ROUND_CONSTANTS;
use super::columns::{
    HALF_FULL_ROUNDS, PARTIAL_ROUNDS, SBOX_DEGREE, SBOX_REGISTERS, WIDTH,
};

/// Build the chip's main trace.
///
/// `inputs[i]` is the Poseidon2 input state for the i-th real chip row. The
/// trace is padded to the smallest power of two greater than
/// `num_real_rows + 1` (so `is_real` has at least one row to fall to 0 in the
/// preprocessed trace), with padding rows running `Poseidon(0)`.
///
/// Trace generation is delegated to `p3_poseidon2_air::generate_trace_rows`,
/// which computes the full round witness (per-round S-box intermediates and
/// post-states) for every row.
pub fn build_trace(inputs: &[[Mersenne31; WIDTH]]) -> RowMajorMatrix<Mersenne31> {
    let num_real_rows = inputs.len();
    let height = (num_real_rows + 1).next_power_of_two().max(4);

    let mut padded: Vec<[Mersenne31; WIDTH]> = Vec::with_capacity(height);
    padded.extend_from_slice(inputs);
    padded.resize(height, [Mersenne31::ZERO; WIDTH]);

    generate_trace_rows::<
        Mersenne31,
        p3_mersenne_31::GenericPoseidon2LinearLayersMersenne31,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(padded, &ROUND_CONSTANTS, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::borrow::Borrow;

    use super::super::columns::{ChipCols, NUM_MAIN_COLS};

    #[test]
    fn trace_width_matches_layout() {
        let trace = build_trace(&[[Mersenne31::ZERO; WIDTH]]);
        assert_eq!(trace.values.len() % NUM_MAIN_COLS, 0);
        let height = trace.values.len() / NUM_MAIN_COLS;
        assert!(height >= 2, "trace must include at least one padding row");
        // Sanity: first row's `inputs` field reads back as we wrote it.
        let row0: &ChipCols<Mersenne31> = trace.values[..NUM_MAIN_COLS].borrow();
        assert_eq!(row0.inputs, [Mersenne31::ZERO; WIDTH]);
    }
}
