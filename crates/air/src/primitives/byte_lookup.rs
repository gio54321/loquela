use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::vec;

/// Number of preprocessed rows: one per byte value 0..=255.
const NUM_PREPROCESSED_ROWS: usize = 256;

#[derive(Clone)]
pub struct BytesAir {
    num_lookups: usize,
}

impl BytesAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F: Field> BaseAir<F> for BytesAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut curr = F::ZERO;
        let mut data = Vec::new();
        for _ in 0..256 {
            data.push(curr);
            curr += F::ONE;
        }
        Some(RowMajorMatrix::new(data, 1))
    }
}

impl<AB: AirBuilder> Air<AB> for BytesAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    <AB as AirBuilder>::F: Field,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<F: Field> LookupAir<F> for BytesAir {
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
        let symbolic_main_local = symbolic_main.current_slice();
        let sybolic_preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        vec![self.register_lookup(
            Kind::Global(String::from("bytes")),
            &vec![(
                vec![sybolic_preprocessed_local[0].into()],
                symbolic_main_local[0].into(),
                Direction::Receive,
            )],
        )]
    }
}

/// Build the main trace for `BytesAir`.
///
/// `multiplicities[v]` is the number of times byte value `v` is looked up.
/// The slice must have exactly 256 elements (one per preprocessed row).
pub fn build_trace<F: Field>(multiplicities: &[F]) -> RowMajorMatrix<F> {
    assert_eq!(
        multiplicities.len(),
        NUM_PREPROCESSED_ROWS,
        "BytesAir requires exactly {NUM_PREPROCESSED_ROWS} multiplicities"
    );
    RowMajorMatrix::new(multiplicities.to_vec(), 1)
}
