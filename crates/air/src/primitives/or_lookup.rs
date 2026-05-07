use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::vec;

/// Number of preprocessed rows: all (x, y) byte pairs (65536 = 2^16).
const NUM_PREPROCESSED_ROWS: usize = 256 * 256;

#[derive(Clone)]
pub struct OrAir {
    num_lookups: usize,
}

impl OrAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F: Field> BaseAir<F> for OrAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut bytes_f = Vec::new();

        let mut curr = F::ZERO;
        for _ in 0..256 {
            bytes_f.push(curr);
            curr += F::ONE;
        }

        let mut data = Vec::new();
        for x in 0..256 {
            for y in 0..256 {
                data.push(bytes_f[x]);
                data.push(bytes_f[y]);
                data.push(bytes_f[x | y]);
            }
        }
        Some(RowMajorMatrix::new(data, 3))
    }
}

impl<AB: AirBuilder> Air<AB> for OrAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    <AB as AirBuilder>::F: Field,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<F: Field> LookupAir<F> for OrAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            preprocessed_width: 3,
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.current_slice();
        let symbolic_preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        vec![self.register_lookup(
            Kind::Global(String::from("bytes_or")),
            &vec![(
                vec![
                    symbolic_preprocessed_local[0].into(),
                    symbolic_preprocessed_local[1].into(),
                    symbolic_preprocessed_local[2].into(),
                ],
                symbolic_main_local[0].into(),
                Direction::Receive,
            )],
        )]
    }
}

/// Build the main trace for `OrAir`.
///
/// `multiplicities[i]` is the count for preprocessed row `i`, which holds
/// `(x, y, x | y)` ordered as: `for x in 0..256 { for y in 0..256 { ... } }`,
/// i.e. row index = `x * 256 + y`.
///
/// The slice must have exactly `NUM_PREPROCESSED_ROWS` = 65536 elements.
pub fn build_trace<F: Field>(multiplicities: &[F]) -> RowMajorMatrix<F> {
    assert_eq!(
        multiplicities.len(),
        NUM_PREPROCESSED_ROWS,
        "OrAir requires exactly {NUM_PREPROCESSED_ROWS} multiplicities"
    );
    RowMajorMatrix::new(multiplicities.to_vec(), 1)
}
