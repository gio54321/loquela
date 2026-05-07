use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
};

/// Two-row AIR that enforces the initial and final execution boundaries via the "trace" bus.
///
/// Row 0 (first): sends (pc=0, timestamp=0) — the required initial state.
/// Row 1 (last):  receives (pc, timestamp)  — the final state, unconstrained by this AIR.
///
/// The preprocessed trace has a single column that acts as a row selector:
///   row 0 → 0 (first row),  row 1 → 1 (last row).
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
        // Row 0 = 0 (first-row selector), Row 1 = 1 (last-row selector).
        Some(RowMajorMatrix::new(vec![F::ZERO, F::ONE], 1))
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
            preprocessed_width: 1,
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let local: &BoundaryColumns<_> = symbolic_main.current_slice().borrow();
        let preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        let is_last: p3_air::SymbolicExpression<F> = preprocessed_local[0].clone().into();
        let is_first: p3_air::SymbolicExpression<F> =
            Into::<p3_air::SymbolicExpression<F>>::into(F::ONE) - is_last.clone();

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

        // Row 1: receive (pc, timestamp) — the final execution state.
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
/// Row 0 holds zeros (the send uses constant 0s, so column values are irrelevant).
/// Row 1 holds `final_pc` and `final_timestamp` — the final execution state to be received.
pub fn build_trace<F: Field>(final_pc: [F; 4], final_timestamp: F) -> RowMajorMatrix<F> {
    let mut data = vec![F::ZERO; 2 * NUM_COLS];
    // Row 1: write final pc limbs then timestamp.
    let row1_start = NUM_COLS;
    data[row1_start..row1_start + 4].copy_from_slice(&final_pc);
    data[row1_start + 4] = final_timestamp;
    RowMajorMatrix::new(data, NUM_COLS)
}
