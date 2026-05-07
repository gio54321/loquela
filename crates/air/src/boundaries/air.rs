use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
};

/// Four-row AIR that enforces the initial and final execution boundaries via the "trace" bus.
///
/// Row 0: sends  (pc=0, timestamp=0) — the required initial state.
/// Rows 1–2: neutral padding rows (no bus interactions).
/// Row 3: receives (pc, timestamp)   — the final state, unconstrained by this AIR.
///
/// The preprocessed trace has TWO columns so padding rows can be fully neutral:
///   col 0 = is_first: 1 only on row 0, else 0.
///   col 1 = is_last:  1 only on row 3, else 0.
/// CirclePCS requires ≥ 4 rows per committed matrix, so 4 is the minimum.
#[repr(C)]
pub struct BoundaryColumns<F> {
    pub pc: [F; 4],
    pub timestamp: F,
}

pub const NUM_COLS: usize = size_of::<BoundaryColumns<u8>>();

impl<T> Borrow<BoundaryColumns<T>> for [T] {
    fn borrow(&self) -> &BoundaryColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<BoundaryColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<BoundaryColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut BoundaryColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<BoundaryColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct BoundariesAir {
    num_lookups: usize,
}

impl BoundariesAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F: Field> BaseAir<F> for BoundariesAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        // 4 rows × 2 cols: (is_first, is_last).
        // Row 0: (1,0), Rows 1-2: (0,0), Row 3: (0,1).
        Some(RowMajorMatrix::new(
            vec![
                F::ONE,
                F::ZERO, // row 0
                F::ZERO,
                F::ZERO, // row 1 (padding)
                F::ZERO,
                F::ZERO, // row 2 (padding)
                F::ZERO,
                F::ONE, // row 3
            ],
            2,
        ))
    }
}

impl<AB: AirBuilder> Air<AB> for BoundariesAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    <AB as AirBuilder>::F: Field,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<F: Field> LookupAir<F> for BoundariesAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            preprocessed_width: 2,
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let local: &BoundaryColumns<_> = symbolic_main.current_slice().borrow();
        let preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        // Two independent selectors: only one is non-zero per row.
        let is_first: p3_air::SymbolicExpression<F> = preprocessed_local[0].clone().into();
        let is_last: p3_air::SymbolicExpression<F> = preprocessed_local[1].clone().into();

        let mut lookups = Vec::new();

        // Row 0: send (pc=0, timestamp=0) as the required initial execution state.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                (0..4)
                    .map(|_| Into::<p3_air::SymbolicExpression<F>>::into(F::ZERO))
                    .chain(once(F::ZERO.into()))
                    .collect::<Vec<_>>(),
                is_first,
                Direction::Send,
            )],
        ));

        // Row 3: receive (pc, timestamp) — the final execution state.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .pc
                    .into_iter()
                    .map(Into::into)
                    .chain(once(local.timestamp.clone().into()))
                    .collect::<Vec<_>>(),
                is_last,
                Direction::Receive,
            )],
        ));

        lookups
    }
}

/// Build the main trace for `BoundariesAir`.
///
/// Row 0 holds zeros (the send uses constant 0s).
/// Rows 1–2 are neutral padding (all zeros, no bus contribution).
/// Row 3 holds `final_pc` and `final_timestamp` — the final execution state to be received.
pub fn build_trace<F: Field>(final_pc: [F; 4], final_timestamp: F) -> RowMajorMatrix<F> {
    let mut data = vec![F::ZERO; 4 * NUM_COLS];
    // Row 3: write final pc limbs then timestamp.
    let row3_start = 3 * NUM_COLS;
    data[row3_start..row3_start + 4].copy_from_slice(&final_pc);
    data[row3_start + 4] = final_timestamp;
    RowMajorMatrix::new(data, NUM_COLS)
}
