use p3_air::{Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, WindowAccess};
use p3_field::Field;
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use std::vec;

/// Number of preprocessed rows: all (byte_val, bit_shamt) pairs.
/// byte_val in 0..256, bit_shamt in 0..8 → 2048 rows.
const NUM_PREPROCESSED_ROWS: usize = 256 * 8;

/// Preprocessed lookup table for byte-level left shift.
///
/// Row `byte_val * 8 + bit_shamt` stores:
///   `(byte_val, bit_shamt, shifted_byte, carry_byte)` where:
///   - `shifted_byte = (byte_val << bit_shamt) & 0xFF`
///   - `carry_byte   = byte_val >> (8 - bit_shamt)`  (0 when bit_shamt == 0)
///
/// The `"byte_sll"` bus schema is `(byte_val, bit_shamt, shifted_byte, carry_byte)`.
#[derive(Clone)]
pub struct ByteShiftLeftAir {
    num_lookups: usize,
}

impl ByteShiftLeftAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for ByteShiftLeftAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> BaseAir<F> for ByteShiftLeftAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut data = Vec::with_capacity(NUM_PREPROCESSED_ROWS * 4);
        for byte_val in 0u32..256 {
            for bit_shamt in 0u32..8 {
                let shifted_byte = (byte_val << bit_shamt) & 0xFF;
                let carry_byte = if bit_shamt == 0 {
                    0u32
                } else {
                    byte_val >> (8 - bit_shamt)
                };
                data.push(F::from_u64(byte_val as u64));
                data.push(F::from_u64(bit_shamt as u64));
                data.push(F::from_u64(shifted_byte as u64));
                data.push(F::from_u64(carry_byte as u64));
            }
        }
        Some(RowMajorMatrix::new(data, 4))
    }
}

impl<AB: AirBuilder> Air<AB> for ByteShiftLeftAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    <AB as AirBuilder>::F: Field,
{
    fn eval(&self, _builder: &mut AB) {}
}

impl<F: Field> LookupAir<F> for ByteShiftLeftAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            preprocessed_width: 4,
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.current_slice();
        let symbolic_preprocessed_local = symbolic_air_builder.preprocessed().current_slice();

        vec![self.register_lookup(
            Kind::Global(String::from("byte_sll")),
            &vec![(
                vec![
                    symbolic_preprocessed_local[0].into(),
                    symbolic_preprocessed_local[1].into(),
                    symbolic_preprocessed_local[2].into(),
                    symbolic_preprocessed_local[3].into(),
                ],
                symbolic_main_local[0].into(),
                Direction::Receive,
            )],
        )]
    }
}

/// Build the main trace for `ByteShiftLeftAir`.
///
/// `multiplicities[i]` is the count for preprocessed row `i`.
/// Row index = `byte_val * 8 + bit_shamt`.
///
/// The slice must have exactly `NUM_PREPROCESSED_ROWS` = 2048 elements.
pub fn build_trace<F: Field>(multiplicities: &[F]) -> RowMajorMatrix<F> {
    assert_eq!(
        multiplicities.len(),
        NUM_PREPROCESSED_ROWS,
        "ByteShiftLeftAir requires exactly {NUM_PREPROCESSED_ROWS} multiplicities"
    );
    RowMajorMatrix::new(multiplicities.to_vec(), 1)
}
