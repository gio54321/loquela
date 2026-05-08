use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per ECALL or EBREAK instruction execution.
///
/// These instructions halt the VM: they produce no register reads or writes.
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Ecall, 0, 0, 0)` or `(InstructionId::Ebreak, 0, 0, 0)`
///     to the "decode" bus.
///   - No memory bus interactions.
///   - Sends four byte-range tuples to the "bytes" bus, range-checking `pc[i]`.
///   - Sends `(next_pc[0..4], timestamp)` to the "trace" bus (timestamp unchanged
///     since no memory ops were emitted).
///
/// `is_ecall` distinguishes the two: 1 for ECALL, 0 for EBREAK.
#[repr(C)]
pub struct EcallColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction (unchanged by execution).
    pub timestamp: F,

    /// 1 if this row is an ECALL, 0 if it is an EBREAK.
    pub is_ecall: F,

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_ECALL_COLS: usize = size_of::<EcallColumns<u8>>();

impl<T> Borrow<EcallColumns<T>> for [T] {
    fn borrow(&self) -> &EcallColumns<T> {
        debug_assert_eq!(self.len(), NUM_ECALL_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<EcallColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<EcallColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut EcallColumns<T> {
        debug_assert_eq!(self.len(), NUM_ECALL_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<EcallColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct EcallAir {
    num_lookups: usize,
}

impl EcallAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for EcallAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for EcallAir {
    fn width(&self) -> usize {
        NUM_ECALL_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for EcallAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &EcallColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());
        builder.assert_bool(local.is_ecall.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);
    }
}

impl<F: Field> LookupAir<F> for EcallAir {
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
        let local: &EcallColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Declare the decoded instruction on the "decode" bus.
        // is_ecall selects between InstructionId::Ecall and InstructionId::Ebreak.
        let instr_id: p3_air::SymbolicExpression<F> =
            p3_air::SymbolicExpression::from(local.is_ecall)
                * p3_air::SymbolicExpression::from(
                    F::from_u64(InstructionId::Ecall as u64)
                        - F::from_u64(InstructionId::Ebreak as u64),
                )
                + p3_air::SymbolicExpression::from(F::from_u64(InstructionId::Ebreak as u64));
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                vec![instr_id, F::ZERO.into(), F::ZERO.into(), F::ZERO.into()],
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for pc[i]: proves each limb is in [0, 255].
        lookups.extend(local.pc.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Emit the next execution state into the "trace" bus.
        // No memory ops were emitted, so timestamp is unchanged.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
                    .map(Into::into)
                    .chain(once(local.timestamp.into()))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
