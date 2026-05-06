use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::integers::QuotientMap;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::borrow::{Borrow, BorrowMut};
use std::vec;

/// AIR for less-than comparison on 24-bit timestamps.
///
/// Range-checks x and y via 24-bit decompositions. Then uses the circomlib trick:
/// bit-decompose v = x + 2^24 - y into 25 bits; x < y iff v_bits[24] == 0.
///
/// Exposes (x, y, is_equal) on the "timestamp_lt" bus with no external lookups.
#[repr(C)]
pub struct TimestampLessThanColumns<F> {
    pub x: F,
    pub y: F,
    /// Bit decomposition of x (little-endian), enforces x ∈ [0, 2^24)
    pub x_bits: [F; 24],
    /// Bit decomposition of y (little-endian), enforces y ∈ [0, 2^24)
    pub y_bits: [F; 24],
    /// 25-bit decomposition of v = x + 2^24 - y; bit 24 is the borrow indicator
    pub v_bits: [F; 25],
    /// Inverse of (x - y), or 0 when x == y (IsZero witness)
    pub diff_inv: F,
    /// 1 when x == y, 0 otherwise
    pub is_equal: F,
    pub mult: F,
}

const NUM_COLS: usize = size_of::<TimestampLessThanColumns<u8>>();

impl<T> Borrow<TimestampLessThanColumns<T>> for [T] {
    fn borrow(&self) -> &TimestampLessThanColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<TimestampLessThanColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<TimestampLessThanColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut TimestampLessThanColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<TimestampLessThanColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

pub struct TimestampLessThanAir {
    num_lookups: usize,
}

impl TimestampLessThanAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F: Field> BaseAir<F> for TimestampLessThanAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for TimestampLessThanAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &TimestampLessThanColumns<AB::Var> = main.current_slice().borrow();

        let two = AB::Expr::TWO;

        // Range-check x via its 24-bit decomposition.
        // Each bit is boolean; the weighted sum must equal x.
        let mut x_sum = AB::Expr::ZERO;
        let mut weight = AB::Expr::ONE;
        for (i, bit) in local.x_bits.iter().enumerate() {
            builder.assert_bool(bit.clone());
            x_sum = x_sum + bit.clone() * weight.clone();
            if i + 1 < 24 {
                weight = weight * two.clone();
            }
        }
        builder.assert_eq(local.x.clone(), x_sum);

        // Range-check y the same way.
        let mut y_sum = AB::Expr::ZERO;
        let mut weight = AB::Expr::ONE;
        for (i, bit) in local.y_bits.iter().enumerate() {
            builder.assert_bool(bit.clone());
            y_sum = y_sum + bit.clone() * weight.clone();
            if i + 1 < 24 {
                weight = weight * two.clone();
            }
        }
        builder.assert_eq(local.y.clone(), y_sum);

        // Bit decomposition of v = x + 2^24 - y.
        // weight at this point is 2^23; one more doubling gives 2^24.
        let two24 = weight * two.clone();
        let v = local.x.clone().into() + two24 - local.y.clone();
        let mut v_sum = AB::Expr::ZERO;
        let mut weight = AB::Expr::ONE;
        for (i, bit) in local.v_bits.iter().enumerate() {
            builder.assert_bool(bit.clone());
            v_sum = v_sum + bit.clone() * weight.clone();
            if i + 1 < 25 {
                weight = weight * two.clone();
            }
        }
        builder.assert_eq(v, v_sum);

        // IsZero gadget: is_equal = 1 iff x == y.
        //   diff_inv * (x - y) + 1 = is_equal
        //   (x - y) * is_equal     = 0
        let diff = local.x.clone().into() - local.y.clone();
        builder.assert_eq(
            local.diff_inv.clone() * diff.clone() + AB::Expr::ONE,
            local.is_equal.clone(),
        );
        builder.assert_eq(diff * local.is_equal.clone(), AB::Expr::ZERO);
    }
}

impl<F: Field> LookupAir<F> for TimestampLessThanAir {
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
        let local: &TimestampLessThanColumns<_> = symbolic_main.current_slice().borrow();

        vec![self.register_lookup(
            Kind::Global(String::from("timestamp_lt")),
            &vec![(
                vec![
                    local.x.clone().into(),
                    local.y.clone().into(),
                    local.is_equal.clone().into(),
                ],
                local.mult.clone().into(),
                Direction::Receive,
            )],
        )]
    }
}

/// Build the main trace for `TimestampLessThanAir`.
///
/// Each entry `(x, y, mult)` compares two 24-bit values; `mult` is the number of times
/// the tuple `(x, y, is_equal)` is received on the `"timestamp_lt"` bus.
/// Both `x` and `y` must be less than `2^24`.
///
/// The trace is padded to the next power of two with rows where `x = y = 0`.
pub fn build_trace<F: Field + QuotientMap<u8> + QuotientMap<u32>>(
    entries: &[(u32, u32, F)],
) -> RowMajorMatrix<F> {
    let height = entries.len().next_power_of_two().max(1);
    let mut data = vec![F::ZERO; height * NUM_COLS];

    let (prefix, rows, suffix) = unsafe { data.align_to_mut::<TimestampLessThanColumns<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), height);

    for (row, &(x, y, mult)) in rows.iter_mut().zip(entries.iter()) {
        debug_assert!(x < (1 << 24), "x must fit in 24 bits");
        debug_assert!(y < (1 << 24), "y must fit in 24 bits");

        row.x = F::from_int(x);
        row.y = F::from_int(y);

        for i in 0..24 {
            row.x_bits[i] = F::from_int(((x >> i) & 1) as u8);
            row.y_bits[i] = F::from_int(((y >> i) & 1) as u8);
        }

        // v = x + 2^24 - y; since x, y < 2^24, v lies in [1, 2^25 - 1].
        let v = x + (1u32 << 24) - y;
        for i in 0..25 {
            row.v_bits[i] = F::from_int(((v >> i) & 1) as u8);
        }

        if x == y {
            row.diff_inv = F::ZERO;
            row.is_equal = F::ONE;
        } else {
            let diff = F::from_int(x) - F::from_int(y);
            // diff_inv * diff + 1 = 0  =>  diff_inv = -diff^{-1}
            row.diff_inv = -diff.try_inverse().unwrap();
            row.is_equal = F::ZERO;
        }

        row.mult = mult;
    }

    // Padding rows: x = y = 0, so v = 2^24, v_bits[24] = 1, is_equal = 1.
    for row in rows.iter_mut().skip(entries.len()) {
        row.v_bits[24] = F::ONE;
        row.is_equal = F::ONE;
    }

    RowMajorMatrix::new(data, NUM_COLS)
}
