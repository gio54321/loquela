use core::borrow::Borrow;
use std::vec;
use std::vec::Vec;

use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    WindowAccess,
};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{
    GenericPoseidon2LinearLayersMersenne31, MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL,
    MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL, MERSENNE31_POSEIDON2_RC_16_INTERNAL, Mersenne31,
};
use p3_poseidon2_air::{Poseidon2Air, RoundConstants};

use super::columns::{
    ChipCols, HALF_FULL_ROUNDS, NUM_MAIN_COLS, NUM_PREPROCESSED_COLS, PARTIAL_ROUNDS,
    SBOX_DEGREE, SBOX_REGISTERS, WIDTH,
};

/// Type alias for the inner `Poseidon2Air` over Mersenne31 width-16.
pub type InnerAir = Poseidon2Air<
    Mersenne31,
    GenericPoseidon2LinearLayersMersenne31,
    WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

/// Mersenne31 width-16 round constants packaged for the upstream AIR. Used by
/// both the chip (for constraint evaluation) and trace generation.
pub const ROUND_CONSTANTS: RoundConstants<Mersenne31, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> =
    RoundConstants::new(
        MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL,
        MERSENNE31_POSEIDON2_RC_16_INTERNAL,
        MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL,
    );

/// Pre-built `InnerAir` constant. `Poseidon2Air::new` is `const`, so the chip
/// can share a single instance across proves.
pub const INNER_AIR: InnerAir = Poseidon2Air::new(ROUND_CONSTANTS);

/// AIR that wraps the upstream `Poseidon2Air` and exposes it over a global
/// `poseidon2_perm` lookup bus.
///
/// - Main trace = exactly `Poseidon2Cols<F, 16, 5, 1, 4, 14>` (so the upstream
///   `Air<AB>::eval` borrows correctly).
/// - Preprocessed trace = single column `is_real` (1 on the first
///   `num_real_rows`, 0 thereafter).
/// - Lookup: Receives `(perm.inputs[0..16], perm.ending_full_rounds[3].post[0..16])`
///   with multiplicity = `is_real`.
#[derive(Clone)]
pub struct Poseidon2Chip {
    /// Number of real permutation rows. The chip's main+preprocessed traces
    /// are padded to the next power of two greater than or equal to
    /// `num_real_rows + 1` (at least one padding row so `is_real` falls to 0).
    pub num_real_rows: usize,
    num_lookups: usize,
}

impl Poseidon2Chip {
    /// Construct a chip configured for `num_real_rows` real permutation rows.
    pub const fn new(num_real_rows: usize) -> Self {
        Self {
            num_real_rows,
            num_lookups: 0,
        }
    }

    /// Total trace height (real rows + at least one padding row, rounded up).
    pub fn trace_height(&self) -> usize {
        (self.num_real_rows + 1).next_power_of_two().max(4)
    }
}

impl<F: Field> BaseAir<F> for Poseidon2Chip {
    fn width(&self) -> usize {
        NUM_MAIN_COLS
    }

    /// Poseidon2 is single-row only: all constraints reference the current
    /// row. Mirrors `Poseidon2Air::main_next_row_columns`.
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let h = self.trace_height();
        let mut data = vec![F::ZERO; h];
        for v in data.iter_mut().take(self.num_real_rows) {
            *v = F::ONE;
        }
        Some(RowMajorMatrix::new(data, NUM_PREPROCESSED_COLS))
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(SBOX_DEGREE as usize)
    }
}

impl<AB> Air<AB> for Poseidon2Chip
where
    AB: AirBuilder<F = Mersenne31>,
    AB::MainWindow: WindowAccess<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        // The chip's main trace is laid out exactly as `Poseidon2Cols`. Hand
        // it to the upstream `Air<AB>` impl unmodified; all round, S-box, and
        // linear-layer constraints come straight from `p3-poseidon2-air`.
        <InnerAir as Air<AB>>::eval(&INNER_AIR, builder);
    }
}

impl<F: Field> LookupAir<F> for Poseidon2Chip {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: NUM_MAIN_COLS,
            preprocessed_width: NUM_PREPROCESSED_COLS,
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let local: &ChipCols<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();
        let preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        // Lookup tuple: (inputs[0..16], outputs[0..16]) where outputs are the
        // post-state of the final ending full round.
        let inputs = local.inputs.iter().map(|v| SymbolicExpression::from(*v));
        let outputs = local.ending_full_rounds[HALF_FULL_ROUNDS - 1]
            .post
            .iter()
            .map(|v| SymbolicExpression::from(*v));
        let elements: Vec<SymbolicExpression<F>> = inputs.chain(outputs).collect();

        let is_real: SymbolicExpression<F> = preprocessed_local[0].clone().into();

        vec![self.register_lookup(
            Kind::Global(String::from("poseidon2_perm")),
            &vec![(elements, is_real, Direction::Receive)],
        )]
    }
}
