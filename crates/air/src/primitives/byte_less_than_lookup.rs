use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::vec;

/// Number of valid (x, y) byte pairs where x < y.
const NUM_VALID_PAIRS: usize = 256 * 255 / 2; // 32640
/// Padded to the next power of two so the trace height is a legal power-of-two.
const NUM_PREPROCESSED_ROWS: usize = 32768; // 2^15

#[derive(Clone)]
pub struct LessThanAir {
    num_lookups: usize,
}

impl LessThanAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F: Field> BaseAir<F> for LessThanAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut bytes_f = Vec::new();
        let mut curr = F::ZERO;
        for _ in 0..256 {
            bytes_f.push(curr);
            curr += F::ONE;
        }

        // Rows 0..NUM_VALID_PAIRS: all (x, y) where x < y, sorted by y then x.
        let mut data = Vec::with_capacity(NUM_PREPROCESSED_ROWS * 2);
        for y in 0..256 {
            for x in 0..y {
                data.push(bytes_f[x]);
                data.push(bytes_f[y]);
            }
        }
        // Padding rows: dummy pair (0, 0) with mult=0 — no bus contribution.
        while data.len() < NUM_PREPROCESSED_ROWS * 2 {
            data.push(F::ZERO);
            data.push(F::ZERO);
        }
        Some(RowMajorMatrix::new(data, 2))
    }
}

impl<AB: AirBuilder> Air<AB> for LessThanAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    <AB as AirBuilder>::F: Field,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<F: Field> LookupAir<F> for LessThanAir {
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
        let symbolic_main_local = symbolic_main.current_slice();
        let symbolic_preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        vec![self.register_lookup(
            Kind::Global(String::from("bytes_lt")),
            &vec![(
                vec![
                    symbolic_preprocessed_local[0].into(),
                    symbolic_preprocessed_local[1].into(),
                ],
                symbolic_main_local[0].into(),
                Direction::Receive,
            )],
        )]
    }
}

/// Build the main trace for `LessThanAir`.
///
/// `multiplicities[i]` is the count for preprocessed row `i`, which holds the pair
/// `(x, y)` ordered as: `for y in 0..256 { for x in 0..y { ... } }`,
/// i.e. row index = `y*(y-1)/2 + x`.
///
/// The slice must have at most `NUM_VALID_PAIRS` = 32640 entries; the function pads
/// to `NUM_PREPROCESSED_ROWS` = 32768 rows automatically.
/// The main trace has width 2; column 0 holds the multiplicity, column 1 is unused.
pub fn build_trace<F: Field>(multiplicities: &[F]) -> RowMajorMatrix<F> {
    assert!(
        multiplicities.len() <= NUM_VALID_PAIRS,
        "LessThanAir: multiplicities length {} exceeds NUM_VALID_PAIRS {}",
        multiplicities.len(),
        NUM_VALID_PAIRS
    );
    let mut data = vec![F::ZERO; NUM_PREPROCESSED_ROWS * 2];
    for (i, &mult) in multiplicities.iter().enumerate() {
        data[i * 2] = mult; // column 0 = multiplicity; column 1 stays zero
    }
    RowMajorMatrix::new(data, 2)
}
