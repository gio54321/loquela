use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::integers::QuotientMap;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::iter::once;
use std::{
    borrow::{Borrow, BorrowMut},
    vec,
};

#[repr(C)]
pub struct U32LessThanColumns<F> {
    pub x: [F; 4],
    pub y: [F; 4],

    pub inverses: [F; 4],
    pub is_equals: [F; 4],
    pub is_equal: F,
    pub mult_lts: [F; 4],
    pub mult: F,
}

const NUM_COLS: usize = size_of::<U32LessThanColumns<u8>>();

impl<T> Borrow<U32LessThanColumns<T>> for [T] {
    fn borrow(&self) -> &U32LessThanColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<U32LessThanColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<U32LessThanColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut U32LessThanColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<U32LessThanColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

pub struct U32LessThanAir {
    num_lookups: usize,
}

impl<F: Field> BaseAir<F> for U32LessThanAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for U32LessThanAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    <AB as AirBuilder>::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &U32LessThanColumns<AB::Var> = main.current_slice().borrow();

        // constrain is_equal[i] to be 1 iff x[i] == y[i]
        for i in 0..4 {
            let diff = local.y[i].clone() - local.x[i].clone();

            builder.assert_eq(
                local.inverses[i].clone() * diff.clone() + AB::Expr::ONE,
                local.is_equals[i].clone(),
            );
            builder.assert_eq(diff * local.is_equals[i].clone(), AB::Expr::ZERO);
        }

        // only one mult_lt can be 1, every other is zero
        for i in (0..4).rev() {
            builder.assert_bool(local.mult_lts[i]);
        }

        let mult_sum = local
            .mult_lts
            .iter()
            .fold(AB::Expr::ZERO, |acc, x| acc + x.clone());
        builder.assert_bool(mult_sum.clone());

        // mult_lts[i] should be 1 on the first (most significant) position where x and y differ, and 0 otherwise
        let mut already_seen = AB::Expr::ZERO;
        for i in (0..4).rev() {
            builder
                .when(already_seen.clone())
                .assert_zero(local.mult_lts[i].clone());
            builder
                .when(AB::Expr::ONE - already_seen.clone())
                .assert_eq(
                    local.mult_lts[i].clone(),
                    AB::Expr::ONE - local.is_equals[i],
                );
            already_seen += local.mult_lts[i].clone();
        }

        // when the multiplicities are all zeros, then x and y are equal, so is_equal should be 1, otherwise it should be 0
        builder.assert_eq(local.is_equal.clone(), AB::Expr::ONE - mult_sum);
    }
}

impl<F: Field> LookupAir<F> for U32LessThanAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local: &U32LessThanColumns<_> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        for i in 0..4 {
            lookups.push(self.register_lookup(
                Kind::Global(format!("byte{}_lt", i)),
                &vec![(
                        vec![
                            symbolic_main_local.x[i].clone(),
                            symbolic_main_local.y[i].clone(),
                        ]
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                        symbolic_main_local.mult_lts[i].clone().into(),
                        Direction::Send,
                    )],
            ));
        }

        lookups.push(self.register_lookup(
            Kind::Global(String::from("u32_lt")),
            &vec![(symbolic_main_local.x.into_iter().chain(
                symbolic_main_local
                    .y
                    .into_iter()).chain(
                        once(symbolic_main_local.is_equal.clone()))
                    .map(Into::into)
                    .collect::<Vec<_>>(),
                symbolic_main_local.mult.into(),
                Direction::Receive,
            )],
        ));
        lookups
    }
}

/// Build the main trace for `U32LessThanAir`.
///
/// Each entry `(x, y, mult)` represents a u32 comparison: `mult` is the number of times
/// the tuple `(x, y, is_equal)` is received on the `"u32_lt"` bus.
///
/// All derived columns (`inverses`, `is_equals`, `mult_lts`, `is_equal`) are computed
/// from `x` and `y`. Byte ordering is little-endian: byte index 3 is most significant.
///
/// The trace is padded to the next power of two using rows with `x = y = 0, mult = 0`.
pub fn build_trace<F: Field + QuotientMap<u8>>(entries: &[(u32, u32, F)]) -> RowMajorMatrix<F> {
    let height = entries.len().next_power_of_two().max(1);
    let mut data = vec![F::ZERO; height * NUM_COLS];

    let (prefix, rows, suffix) = unsafe { data.align_to_mut::<U32LessThanColumns<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), height);

    for (row, &(x, y, mult)) in rows.iter_mut().zip(entries.iter()) {
        let x_bytes = x.to_le_bytes();
        let y_bytes = y.to_le_bytes();

        for i in 0..4 {
            row.x[i] = F::from_int(x_bytes[i]);
            row.y[i] = F::from_int(y_bytes[i]);

            if x_bytes[i] == y_bytes[i] {
                row.inverses[i] = F::ZERO;
                row.is_equals[i] = F::ONE;
            } else {
                // constraint: inverses[i] * (y[i] - x[i]) + 1 = 0
                // so inverses[i] = -1 / (y[i] - x[i])
                let diff = F::from_int(y_bytes[i]) - F::from_int(x_bytes[i]);
                row.inverses[i] = -diff.try_inverse().unwrap();
                row.is_equals[i] = F::ZERO;
            }
        }

        // mult_lts[i] = 1 at the highest byte index where x and y differ
        let mut found = false;
        for i in (0..4).rev() {
            if !found && x_bytes[i] != y_bytes[i] {
                row.mult_lts[i] = F::ONE;
                found = true;
            }
            // remaining entries stay F::ZERO (already initialised)
        }

        row.is_equal = if x == y { F::ONE } else { F::ZERO };
        row.mult = mult;
    }

    // Padding rows: x = y = 0, mult = 0.
    // is_equals must be [1,1,1,1] and is_equal must be 1; all others stay zero.
    for row in rows.iter_mut().skip(entries.len()) {
        row.is_equals = [F::ONE; 4];
        row.is_equal = F::ONE;
    }

    RowMajorMatrix::new(data, NUM_COLS)
}
