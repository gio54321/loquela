use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::{u32_add, u32_plus_four};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per AUIPC instruction execution.
///
/// AUIPC encoding: rd = PC + (imm_raw << 12).
/// `imm_raw` is bits 31:12 of the instruction word (20-bit value), split as:
///   - imm_high12 = bits 31:20 (received via "decode_u" bus)
///   - imm_low8   = bits 19:12 (received via "decode_u" bus)
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Auipc, rd, imm_high12, imm_low8)` to the "decode_u" bus.
///   - Sends one write to the "memory" bus: write rd = rd_val = pc + imm_u.
///   - Sends eight byte-range tuples to the "bytes" bus (pc bytes and rd_val bytes).
///   - Sends `(next_pc[0..4], timestamp + 1)` to the "trace" bus.
///
/// AIR constraints verify:
///   - imm_u[0] == 0
///   - (imm_low8 + imm_high12 * 256) * 16 == imm_u[1] + imm_u[2]*256 + imm_u[3]*65536
///   - rd_val = pc + imm_u (byte-level carry chain)
///   - next_pc = pc + 4
#[repr(C)]
pub struct AuipcColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index.
    pub rd: F,
    /// Lower 8 bits of raw 20-bit immediate (bits 19:12 of instruction word).
    pub imm_low8: F,
    /// Upper 12 bits of raw 20-bit immediate (bits 31:20 of instruction word).
    pub imm_high12: F,

    /// imm_u = imm_raw << 12, stored as four byte limbs.
    pub imm_u: [F; 4],
    /// rd_val = pc + imm_u (wrapping u32), as four byte limbs.
    pub rd_val: [F; 4],
    /// Carry bits for the byte-level addition pc + imm_u.
    pub add_carries: [F; 4],
    /// Old value of register `rd` (before write).
    pub old_rd_value: [F; 4],

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_AUIPC_COLS: usize = size_of::<AuipcColumns<u8>>();

impl<T> Borrow<AuipcColumns<T>> for [T] {
    fn borrow(&self) -> &AuipcColumns<T> {
        debug_assert_eq!(self.len(), NUM_AUIPC_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<AuipcColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<AuipcColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut AuipcColumns<T> {
        debug_assert_eq!(self.len(), NUM_AUIPC_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<AuipcColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct AuipcAir {
    num_lookups: usize,
}

impl AuipcAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for AuipcAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for AuipcAir {
    fn width(&self) -> usize {
        NUM_AUIPC_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for AuipcAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &AuipcColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // AUIPC: imm_u = imm_raw << 12, where imm_raw = imm_low8 + imm_high12 * 256.
        // Byte decomposition of imm_raw << 12:
        //   byte 0: 0
        //   byte 1..3: same pattern as LUI
        // Reconstruction: (imm_low8 + imm_high12 * 256) * 16 == imm_u[1] + imm_u[2]*256 + imm_u[3]*65536

        builder.assert_zero(local.imm_u[0].clone());

        let sixteen = AB::F::from_u32(16);
        let two56 = AB::F::from_u32(256);
        let two56_sq = AB::F::from_u32(65536);

        builder.assert_eq(
            (local.imm_low8.clone() + local.imm_high12.clone() * AB::Expr::from(two56.clone()))
                * AB::Expr::from(sixteen),
            local.imm_u[1].clone()
                + local.imm_u[2].clone() * AB::Expr::from(two56)
                + local.imm_u[3].clone() * AB::Expr::from(two56_sq),
        );

        // Constrain rd_val = pc + imm_u (wrapping u32) with carry bits.
        u32_add(
            builder,
            &local.pc,
            &local.imm_u,
            &local.rd_val,
            &local.add_carries,
        );
    }
}

impl<F: Field> LookupAir<F> for AuipcAir {
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
        let local: &AuipcColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Receive decoded instruction from the "decode_u" bus.
        // Schema: (instr_type_packed, rd, imm_high12, imm_low8)
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_u")),
            &vec![(
                vec![
                    F::from_u64(InstructionId::Auipc as u64).into(),
                    local.rd.into(),
                    local.imm_high12.into(),
                    local.imm_low8.into(),
                ],
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write rd = pc + imm_u at timestamp.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once(local.timestamp.into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.rd_val.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for pc[i]: proves each PC limb is in [0, 255].
        lookups.extend(local.pc.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for rd_val[i]: proves the addition result is byte-normalised.
        lookups.extend(local.rd_val.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Emit the next execution state into the "trace" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
                    .map(Into::into)
                    .chain(once((local.timestamp.clone() + F::ONE).into()))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
