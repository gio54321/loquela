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
/// Asserts that `y = x + 4` as u32, used for PC increment in RISC-V.
/// Drops the top carry (overflow past 2^32 is not expected).
pub fn u32_plus_four<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Var; 4],
    y: &[AB::Var; 4],
    carries: &[AB::Var; 3],
) where
    AB::F: QuotientMap<u32>,
{
    for i in 0..4 {
        let carry_in: AB::Expr = if i == 0 {
            AB::Expr::from(AB::F::from_u32(4))
        } else {
            carries[i - 1].into()
        };
        let carry_out: AB::Expr = if i < 3 {
            builder.assert_bool(carries[i]);
            carries[i].into()
        } else {
            AB::Expr::ZERO
        };
        builder.assert_eq(
            y[i].clone() + carry_out * AB::F::from_u32(1 << 8),
            x[i].clone() + carry_in,
        );
    }
}

/// Assumes that `x`, `y` and `diff` are normalized byte limbs.
/// Asserts that `diff = x - y` as a wrapping u32, with borrow-chain bits.
/// The borrow equation per limb: `x[i] + 256 * borrow_out[i] = y[i] + borrow_in[i] + diff[i]`,
/// where `borrow_in[0] = 0` and `borrow_in[i] = borrow_out[i-1]` for i > 0.
pub fn u32_sub<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Var; 4],
    y: &[AB::Var; 4],
    diff: &[AB::Var; 4],
    borrows: &[AB::Var; 4],
) where
    AB::F: QuotientMap<u32>,
{
    for i in 0..4 {
        builder.assert_bool(borrows[i]);
        let borrow_in: AB::Expr = if i == 0 {
            AB::Expr::ZERO
        } else {
            borrows[i - 1].into()
        };
        // x[i] + 256 * borrow_out = y[i] + borrow_in + diff[i]
        builder.assert_eq(
            x[i].clone() + borrows[i].clone() * AB::F::from_u32(1 << 8),
            y[i].clone() + borrow_in + diff[i].clone(),
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
