//! Width-16 Mersenne31 Poseidon2 parameters and column-layout aliases shared
//! with the upstream `p3_poseidon2_air` crate. The chip's main trace is laid
//! out exactly as `Poseidon2Cols` so the upstream `Air<AB>::eval` borrows
//! correctly; per-row "this row is a real permutation request" lives in a
//! separate preprocessed column.

use p3_poseidon2_air::{num_cols, Poseidon2Cols};

pub const WIDTH: usize = 16;
pub const SBOX_DEGREE: u64 = 5;
pub const SBOX_REGISTERS: usize = 1;
pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 14;

pub type ChipCols<T> =
    Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

pub const NUM_MAIN_COLS: usize =
    num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

/// Width of the chip's preprocessed trace (just the `is_real` selector).
pub const NUM_PREPROCESSED_COLS: usize = 1;
