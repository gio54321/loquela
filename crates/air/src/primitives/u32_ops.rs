use p3_air::AirBuilder;
use p3_field::{integers::QuotientMap, PrimeCharacteristicRing};

/// Assumes that `x`, `y` and `z` are normalized.
/// Asserts that `x + y = z` as u32, and that the carries are correct limb carries.
pub fn u32_add<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Var; 4],
    y: &[AB::Var; 4],
    sum: &[AB::Var; 4],
    carries: &[AB::Var; 4],
) where
    AB::F: QuotientMap<u32>,
{
    for i in 0..4 {
        builder.assert_bool(carries[i]);
        let carry_in = if i == 0 {
            AB::Expr::ZERO
        } else {
            carries[i - 1].into()
        };
        builder.assert_eq(
            sum[i].clone() + carries[i].clone() * AB::F::from_u32(1 << 8),
            x[i].clone() + y[i].clone() + carry_in,
        );
    }
}

/// Assumes that `x` and `y` are normalized byte limbs.
/// Asserts that `y = x + 1` as u32 with carry propagation.
pub fn u32_inc<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Var; 4],
    y: &[AB::Var; 4],
    carries: &[AB::Var; 4],
) where
    AB::F: QuotientMap<u32>,
{
    for i in 0..4 {
        builder.assert_bool(carries[i]);
        let carry_in: AB::Expr = if i == 0 {
            AB::Expr::ONE
        } else {
            carries[i - 1].into()
        };
        builder.assert_eq(
            y[i].clone() + carries[i].clone() * AB::F::from_u32(1 << 8),
            x[i].clone() + carry_in,
        );
    }
}
