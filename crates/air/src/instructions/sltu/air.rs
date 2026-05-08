use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per SLTU instruction execution.
///
/// SLTU: `rd = (rs1 < rs2) ? 1 : 0` (unsigned comparison).
///
/// Comparison technique: compute `rs1 - rs2` byte-by-byte with borrow chain.
/// The final borrow-out is 1 iff `rs1 < rs2` (unsigned).
///
/// Borrow chain: for each byte i,
///   `rs1_bytes[i] + borrow_out[i] * 256 = rs2_bytes[i] + diff_bytes[i] + borrow_in[i]`
/// where `borrow_in[0] = 0`, `borrow_in[i] = borrow_out[i-1]`.
/// `lt_result = borrow_out[3]`.
///
/// rd = lt_result (a 1-bit value stored as a 32-bit register).
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Sltu, rd, rs1, rs2)` to the "decode" bus.
///   - Sends three operations to the "memory" bus: read rs1, read rs2, write rd.
///   - Sends byte range-checks for rs1_bytes[i], rs2_bytes[i], diff_bytes[i] to the "bytes" bus.
///   - Sends `(next_pc[0..4], timestamp + 3)` to the "trace" bus.
#[repr(C)]
pub struct SltuColumns<F> {
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
    /// `borrow[i] = 1` iff the subtraction of byte i required borrowing.
    pub borrow: [F; 4],

    /// The less-than result: 1 iff rs1 < rs2 (unsigned). Equals `borrow[3]`.
    pub lt_result: F,

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_SLTU_COLS: usize = size_of::<SltuColumns<u8>>();

impl<T> Borrow<SltuColumns<T>> for [T] {
    fn borrow(&self) -> &SltuColumns<T> {
        debug_assert_eq!(self.len(), NUM_SLTU_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<SltuColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<SltuColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut SltuColumns<T> {
        debug_assert_eq!(self.len(), NUM_SLTU_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<SltuColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct SltuAir {
    num_lookups: usize,
}

impl SltuAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for SltuAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for SltuAir {
    fn width(&self) -> usize {
        NUM_SLTU_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for SltuAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &SltuColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Borrow bits must be boolean.
        for b in local.borrow.iter() {
            builder.assert_bool(b.clone());
        }

        // lt_result is boolean.
        builder.assert_bool(local.lt_result.clone());

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

        // lt_result = borrow[3] (final borrow-out).
        builder.assert_eq(local.lt_result.clone(), local.borrow[3].clone());
    }
}

impl<F: Field> LookupAir<F> for SltuAir {
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
        let local: &SltuColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is SLTU with (pc, rd, rs1, rs2) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                local.pc.into_iter().map(Into::into)
                    .chain(once(F::from_u64(InstructionId::Sltu as u64).into()))
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
        // rd value is lt_result in byte 0, zeros in bytes 1-3.
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

        // Byte range-checks for rs1_bytes[i].
        lookups.extend(local.rs1_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for rs2_bytes[i].
        lookups.extend(local.rs2_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for diff_bytes[i].
        lookups.extend(local.diff_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

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
