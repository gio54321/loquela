use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per AND instruction execution.
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::And, rd, rs1, rs2)` to the "decode" bus.
///   - Sends three operations to the "memory" bus: read rs1, read rs2, write rd.
///   - Sends four byte-triple tuples to the "bytes_and" bus, proving
///     `rd_new_value[i] = rs1_value[i] & rs2_value[i]`.
///   - Sends `(next_pc[0..4], timestamp + 3)` to the "trace" bus.
#[repr(C)]
pub struct AndColumns<F> {
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

    /// Value read from register `rs1`.
    pub rs1_value: [F; 4],
    /// Value read from register `rs2`.
    pub rs2_value: [F; 4],
    /// Old value of register `rd` (before write).
    pub old_rd_value: [F; 4],
    /// New value written to `rd`: `rs1_value & rs2_value`.
    pub rd_new_value: [F; 4],

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_AND_COLS: usize = size_of::<AndColumns<u8>>();

impl<T> Borrow<AndColumns<T>> for [T] {
    fn borrow(&self) -> &AndColumns<T> {
        debug_assert_eq!(self.len(), NUM_AND_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<AndColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<AndColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut AndColumns<T> {
        debug_assert_eq!(self.len(), NUM_AND_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<AndColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct AndInstrAir {
    num_lookups: usize,
}

impl AndInstrAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for AndInstrAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for AndInstrAir {
    fn width(&self) -> usize {
        NUM_AND_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for AndInstrAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &AndColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);
    }
}

impl<F: Field> LookupAir<F> for AndInstrAir {
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
        let local: &AndColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        // Consume the current execution state from the "trace" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local.pc.into_iter()
                    .chain(once(local.timestamp))
                    .map(Into::into)
                    .collect(),
                local.is_dummy.into(),
                Direction::Receive,
            )],
        ));

        // Assert the decoded instruction is AND with (rd, rs1, rs2) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                once(F::from_u64(InstructionId::And as u64).into())
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
                    .chain(local.rs1_value.into_iter().map(Into::into))
                    .chain(local.rs1_value.into_iter().map(Into::into))
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
                    .chain(local.rs2_value.into_iter().map(Into::into))
                    .chain(local.rs2_value.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write the AND result to rd at timestamp + 2.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::from_u64(2)).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.rd_new_value.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Four byte-level AND lookups: rd_new[i] = rs1_value[i] & rs2_value[i].
        // These also implicitly range-check all three operand bytes to [0, 255].
        lookups.extend(
            local
                .rs1_value
                .into_iter()
                .zip(local.rs2_value.into_iter())
                .zip(local.rd_new_value.into_iter())
                .map(|((x, y), z)| {
                    self.register_lookup(
                        Kind::Global(String::from("bytes_and")),
                        &vec![(
                            [x, y, z].into_iter().map(Into::into).collect(),
                            local.is_dummy.into(),
                            Direction::Send,
                        )],
                    )
                }),
        );

        // Emit the next execution state into the "trace" bus.
        // Three memory ops consumed 3 timestamps.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local.next_pc.into_iter()
                    .map(Into::into)
                    .chain(once((local.timestamp.clone() + F::from_u64(3)).into()))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
