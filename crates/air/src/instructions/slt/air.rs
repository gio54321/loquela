use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per SLT instruction execution.
///
/// SLT: `rd = ((rs1 as i32) < (rs2 as i32)) ? 1 : 0` (signed comparison).
///
/// Comparison technique: compute `rs1 - rs2` byte-by-byte with borrow chain
/// (same as SLTU), then adjust for sign bits.
///
/// The signed less-than result is:
///   `slt = borrow XOR (sign_rs1 XOR sign_rs2)`
///
/// where `borrow = borrow_out[3]` (the final unsigned borrow),
///   `sign_rs1 = rs1_bytes[3] >> 7`, `sign_rs2 = rs2_bytes[3] >> 7`.
///
/// Sign bit extraction:
///   `rs1_bytes[3] = sign_rs1 * 128 + rs1_byte3_low7`
///   `rs2_bytes[3] = sign_rs2 * 128 + rs2_byte3_low7`
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Slt, rd, rs1, rs2)` to the "decode" bus.
///   - Sends three operations to the "memory" bus: read rs1, read rs2, write rd.
///   - Sends byte range-checks for rs1_bytes, rs2_bytes, diff_bytes,
///     rs1_byte3_low7, rs2_byte3_low7 to the "bytes" bus.
///   - Sends `(next_pc[0..4], timestamp + 3)` to the "trace" bus.
#[repr(C)]
pub struct SltColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index (from decode bus).
    pub rd: F,
    /// First source register index (from decode bus).
    pub rs1: F,
    /// Second source register index (from decode bus).
    pub rs2: F,

    /// Value read from register `rs1`, as byte limbs.
    pub rs1_bytes: [F; 4],
    /// Value read from register `rs2`, as byte limbs.
    pub rs2_bytes: [F; 4],
    /// Old value of register `rd` (before write), as byte limbs.
    pub old_rd_value: [F; 4],

    /// Byte-level difference `rs1 - rs2` (wrapping), as byte limbs.
    pub diff_bytes: [F; 4],
    /// Borrow-out bits for each byte of the subtraction.
    pub borrow: [F; 4],

    /// Sign bit of rs1: bit 7 of rs1_bytes[3] (0 or 1).
    pub sign_rs1: F,
    /// Sign bit of rs2: bit 7 of rs2_bytes[3] (0 or 1).
    pub sign_rs2: F,
    /// Low 7 bits of rs1_bytes[3]: `rs1_bytes[3] - sign_rs1 * 128`.
    pub rs1_byte3_low7: F,
    /// Low 7 bits of rs2_bytes[3]: `rs2_bytes[3] - sign_rs2 * 128`.
    pub rs2_byte3_low7: F,

    /// The signed less-than result: 1 iff (rs1 as i32) < (rs2 as i32).
    /// Equals `borrow[3] XOR (sign_rs1 XOR sign_rs2)`.
    pub lt_result: F,

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_SLT_COLS: usize = size_of::<SltColumns<u8>>();

impl<T> Borrow<SltColumns<T>> for [T] {
    fn borrow(&self) -> &SltColumns<T> {
        debug_assert_eq!(self.len(), NUM_SLT_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<SltColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<SltColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut SltColumns<T> {
        debug_assert_eq!(self.len(), NUM_SLT_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<SltColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct SltAir {
    num_lookups: usize,
}

impl SltAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for SltAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for SltAir {
    fn width(&self) -> usize {
        NUM_SLT_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for SltAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &SltColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Borrow bits must be boolean.
        for b in local.borrow.iter() {
            builder.assert_bool(b.clone());
        }

        // Sign bits must be boolean.
        builder.assert_bool(local.sign_rs1.clone());
        builder.assert_bool(local.sign_rs2.clone());

        // lt_result is boolean.
        builder.assert_bool(local.lt_result.clone());

        // Sign bit extraction:
        // rs1_bytes[3] = sign_rs1 * 128 + rs1_byte3_low7
        builder.assert_eq(
            local.rs1_bytes[3].clone(),
            local.sign_rs1.clone() * AB::Expr::from(AB::F::from_u32(128))
                + local.rs1_byte3_low7.clone(),
        );
        // rs2_bytes[3] = sign_rs2 * 128 + rs2_byte3_low7
        builder.assert_eq(
            local.rs2_bytes[3].clone(),
            local.sign_rs2.clone() * AB::Expr::from(AB::F::from_u32(128))
                + local.rs2_byte3_low7.clone(),
        );

        // Borrow chain: rs1[i] + borrow[i]*256 = rs2[i] + diff[i] + borrow_in[i]
        // where borrow_in[0] = 0, borrow_in[i] = borrow[i-1].
        for i in 0..4 {
            let borrow_in: AB::Expr = if i == 0 {
                AB::Expr::ZERO
            } else {
                local.borrow[i - 1].clone().into()
            };
            builder.assert_eq(
                local.rs1_bytes[i].clone()
                    + local.borrow[i].clone() * AB::Expr::from(AB::F::from_u32(256)),
                local.rs2_bytes[i].clone() + local.diff_bytes[i].clone() + borrow_in,
            );
        }

        // slt = borrow[3] XOR (sign_rs1 XOR sign_rs2)
        // XOR(a, b) = a + b - 2*a*b for boolean a, b.
        let sign_xor: AB::Expr = local.sign_rs1.clone() + local.sign_rs2.clone()
            - local.sign_rs1.clone() * local.sign_rs2.clone() * AB::Expr::TWO;
        let borrow3: AB::Expr = local.borrow[3].clone().into();
        let slt: AB::Expr = borrow3.clone() + sign_xor.clone() - borrow3 * sign_xor * AB::Expr::TWO;
        builder.assert_eq(local.lt_result.clone(), slt);
    }
}

impl<F: Field> LookupAir<F> for SltAir {
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
        let local: &SltColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        // Consume the current execution state from the "trace" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .pc
                    .into_iter()
                    .chain(once(local.timestamp))
                    .map(Into::into)
                    .collect(),
                local.is_dummy.into(),
                Direction::Receive,
            )],
        ));

        // Assert the decoded instruction is SLT with (rd, rs1, rs2) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                once(F::from_u64(InstructionId::Slt as u64).into())
                    .chain([local.rd, local.rs1, local.rs2].into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Read rs1 at timestamp.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once(local.timestamp.into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rs1.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.rs1_bytes.into_iter().map(Into::into))
                    .chain(local.rs1_bytes.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Read rs2 at timestamp + 1.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::ONE).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rs2.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.rs2_bytes.into_iter().map(Into::into))
                    .chain(local.rs2_bytes.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write rd = lt_result at timestamp + 2.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::from_u64(2)).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(
                        once(local.lt_result.into())
                            .chain([F::ZERO; 3].into_iter().map(Into::into)),
                    )
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rs1_bytes[0..3].
        lookups.extend(local.rs1_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for rs2_bytes[0..3].
        lookups.extend(local.rs2_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for diff_bytes[0..3].
        lookups.extend(local.diff_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Range-check rs1_byte3_low7 (value in 0..127) via "bytes" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("bytes")),
            &vec![(
                vec![local.rs1_byte3_low7.into()],
                F::ONE.into(),
                Direction::Send,
            )],
        ));

        // Range-check rs2_byte3_low7 (value in 0..127) via "bytes" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("bytes")),
            &vec![(
                vec![local.rs2_byte3_low7.into()],
                F::ONE.into(),
                Direction::Send,
            )],
        ));

        // Emit the next execution state into the "trace" bus.
        // Three memory ops consumed 3 timestamps.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
                    .map(Into::into)
                    .chain(once(
                        (local.timestamp.clone() + F::from_u64(3)).into(),
                    ))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
