use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::vec;

/// Number of preprocessed rows: all (x, y) byte pairs where x < y.
const NUM_PREPROCESSED_ROWS: usize = 256 * 255 / 2; // 32640

pub struct LessThanAir {
    num_lookups: usize,
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

        // store all byte pairs (x, y) where x < y, sorted by x then y
        let mut data = Vec::new();
        for y in 0..256 {
            for x in 0..y {
                data.push(bytes_f[x]);
                data.push(bytes_f[y]);
            }
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
/// The slice must have exactly `NUM_PREPROCESSED_ROWS` = 32640 elements.
/// The main trace has width 2; column 0 holds the multiplicity, column 1 is unused.
pub fn build_trace<F: Field>(multiplicities: &[F]) -> RowMajorMatrix<F> {
    assert_eq!(
        multiplicities.len(),
        NUM_PREPROCESSED_ROWS,
        "LessThanAir requires exactly {NUM_PREPROCESSED_ROWS} multiplicities"
    );
    let mut data = vec![F::ZERO; NUM_PREPROCESSED_ROWS * 2];
    for (i, &mult) in multiplicities.iter().enumerate() {
        data[i * 2] = mult; // column 0 = multiplicity; column 1 stays zero
    }
    RowMajorMatrix::new(data, 2)
}
