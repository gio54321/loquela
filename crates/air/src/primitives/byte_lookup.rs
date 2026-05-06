use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::vec;

pub struct BytesAir {
    num_lookups: usize,
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
