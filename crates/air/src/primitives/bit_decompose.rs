use p3_air::AirBuilder;
use p3_field::PrimeCharacteristicRing;

pub fn check_bit_decomposition<AB: AirBuilder, const N: usize>(
    builder: &mut AB,
    value: AB::Var,
    bits: &[AB::Var; N],
) {
    let mut recomposed = AB::Expr::ZERO;
    let mut weight = AB::Expr::ONE;
    let two = AB::Expr::ONE + AB::Expr::ONE;
    for (i, bit) in bits.iter().enumerate() {
        builder.assert_bool(bit.clone());
        recomposed = recomposed + bit.clone().into() * weight.clone();
        if i + 1 < N {
            weight = weight * two.clone();
        }
    }
    builder.assert_eq(value, recomposed);
}

/// Pack a sub-selection of bits (little-endian) from a `[[F; 8]; B]` decomposition
/// into a single field expression. Each `(byte, bit)` pair indexes into the array.
pub fn pack_bits<AB: AirBuilder, const B: usize>(
    decompositions: &[[AB::Var; 8]; B],
    bits: &[(usize, usize)],
) -> AB::Expr {
    let mut acc = AB::Expr::ZERO;
    let mut weight = AB::Expr::ONE;
    let two = AB::Expr::ONE + AB::Expr::ONE;
    for (i, (byte, bit)) in bits.iter().enumerate() {
        acc = acc + decompositions[*byte][*bit].clone().into() * weight.clone();
        if i + 1 < bits.len() {
            weight = weight * two.clone();
        }
    }
    acc
}
