use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::{u32_add, u32_plus_four};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per LW instruction execution.
///
/// LW: `rd = MEM32[rs1 + imm]` (word load, 32-bit, no sign extension needed).
///
/// Memory operations (3 total, consuming 3 timestamps):
///   ts+0: Register read  rs1 → rs1_value
///   ts+1: RAM read       addr → loaded_val  (addr = rs1_value + sign_extend(imm))
///   ts+2: Register write rd  ← loaded_val
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Lw, rd, rs1, imm)` to the "decode" bus.
///   - Sends three operations to the "memory" bus.
///   - Sends byte range-checks to the "bytes" bus.
///   - Sends `(next_pc[0..4], timestamp + 3)` to the "trace" bus.
#[repr(C)]
pub struct LwColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// Destination register index.
    pub rd: F,
    /// Source register index.
    pub rs1: F,
    /// Unsigned 12-bit immediate (bits 31:20 of instruction word).
    pub imm: F,

    /// Sign extension bits for the immediate (same as ADDI).
    pub imm_high_bits: [F; 4],
    /// Sign-extended immediate as four byte limbs.
    pub imm_se_bytes: [F; 4],

    /// Value read from register `rs1`.
    pub rs1_value: [F; 4],
    /// Effective address = rs1_value + sign_extend(imm), as four byte limbs.
    pub addr_bytes: [F; 4],
    /// Carry bits for the address addition.
    pub addr_carries: [F; 4],

    /// The 32-bit value loaded from RAM at `addr`.
    pub loaded_val: [F; 4],
    /// Old value of register `rd` (before write).
    pub old_rd_value: [F; 4],

    /// `pc + 4` as four byte limbs.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition.
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_LW_COLS: usize = size_of::<LwColumns<u8>>();

impl<T> Borrow<LwColumns<T>> for [T] {
    fn borrow(&self) -> &LwColumns<T> {
        debug_assert_eq!(self.len(), NUM_LW_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<LwColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<LwColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut LwColumns<T> {
        debug_assert_eq!(self.len(), NUM_LW_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<LwColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct LwAir {
    num_lookups: usize,
}

impl LwAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for LwAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for LwAir {
    fn width(&self) -> usize {
        NUM_LW_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for LwAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &LwColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Sign-extend the 12-bit immediate (same logic as ADDI).
        for bit in local.imm_high_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        let imm_high_nibble: AB::Expr = local.imm_high_bits[0].clone()
            + local.imm_high_bits[1].clone() * AB::Expr::TWO
            + local.imm_high_bits[2].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_high_bits[3].clone() * AB::Expr::from(AB::F::from_u32(8));
        let sign_bit: AB::Expr = local.imm_high_bits[3].clone().into();

        // imm = imm_se_bytes[0] + imm_high_nibble * 256
        builder.assert_eq(
            local.imm.clone(),
            local.imm_se_bytes[0].clone()
                + imm_high_nibble.clone() * AB::Expr::from(AB::F::from_u32(256)),
        );
        builder.assert_eq(
            local.imm_se_bytes[1].clone(),
            imm_high_nibble + sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xF0)),
        );
        builder.assert_eq(
            local.imm_se_bytes[2].clone(),
            sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.imm_se_bytes[3].clone(),
            sign_bit * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        // Constrain addr_bytes = rs1_value + imm_se_bytes (wrapping u32).
        u32_add(
            builder,
            &local.rs1_value,
            &local.imm_se_bytes,
            &local.addr_bytes,
            &local.addr_carries,
        );
    }
}

impl<F: Field> LookupAir<F> for LwAir {
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
        let local: &LwColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is LW with (rd, rs1, imm) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                once(F::from_u64(InstructionId::Lw as u64).into())
                    .chain([local.rd, local.rs1, local.imm].into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Memory bus schema: (timestamp, memory_type, addr[0..4], read[0..4], write[0..4]).
        // ts+0: Register read rs1 — memory_type=0, addr=rs1, read=write=rs1_value.
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

        // ts+1: RAM read at addr — memory_type=1, addr=addr_bytes, read=write=loaded_val.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::ONE).into())
                    .chain(once(F::ONE.into()))
                    .chain(local.addr_bytes.into_iter().map(Into::into))
                    .chain(local.loaded_val.into_iter().map(Into::into))
                    .chain(local.loaded_val.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // ts+2: Register write rd ← loaded_val.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::from_u64(2)).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.loaded_val.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rs1_value.
        lookups.extend(local.rs1_value.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for addr_bytes.
        lookups.extend(local.addr_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for loaded_val.
        lookups.extend(local.loaded_val.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Emit the next execution state: 3 timestamps consumed.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("trace")),
            &vec![(
                local
                    .next_pc
                    .into_iter()
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
