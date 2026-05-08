use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::{u32_add, u32_plus_four};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per SH instruction execution.
///
/// SH stores the low 16 bits of rs2 (bytes 0–1) into RAM at address rs1 + imm.
/// The stored 32-bit RAM word has bytes 2–3 zeroed.
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Sh, rs1, rs2, imm_s)` to the "decode_s" bus.
///   - Sends three operations to the "memory" bus: read rs1, read rs2, write RAM.
///   - Sends byte-range tuples to the "bytes" bus for rs1_value, rs2_value bytes 0–1, addr.
///   - Sends `(next_pc[0..4], timestamp + 3)` to the "trace" bus.
#[repr(C)]
pub struct ShColumns<F> {
    /// Current program counter as four byte limbs (little-endian u32).
    pub pc: [F; 4],
    /// Timestamp at the start of this instruction.
    pub timestamp: F,

    /// First source register index (address base).
    pub rs1: F,
    /// Second source register index (value to store).
    pub rs2: F,
    /// S-type immediate (12-bit unsigned from decode bus).
    pub imm_s: F,

    /// High 4 bits of imm_s (bits 8–11); bit 3 is the sign bit.
    pub imm_high_bits: [F; 4],
    /// Sign-extended immediate as four byte limbs.
    pub imm_se_bytes: [F; 4],

    /// Value read from register `rs1`.
    pub rs1_value: [F; 4],
    /// Value read from register `rs2` (full 32-bit, but only bytes 0–1 are stored).
    pub rs2_value: [F; 4],

    /// RAM address = rs1_value + sign_extend(imm_s) as four byte limbs.
    pub addr: [F; 4],
    /// Carry bits for the address addition `rs1_value + imm_se_bytes`.
    pub addr_carries: [F; 4],

    /// Old value in RAM at address `addr` (before the write).
    pub old_ram_value: [F; 4],

    /// `pc + 4` as four byte limbs.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_SH_COLS: usize = size_of::<ShColumns<u8>>();

impl<T> Borrow<ShColumns<T>> for [T] {
    fn borrow(&self) -> &ShColumns<T> {
        debug_assert_eq!(self.len(), NUM_SH_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<ShColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<ShColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut ShColumns<T> {
        debug_assert_eq!(self.len(), NUM_SH_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<ShColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct ShAir {
    num_lookups: usize,
}

impl ShAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for ShAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for ShAir {
    fn width(&self) -> usize {
        NUM_SH_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for ShAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ShColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Bit-decompose bits 8–11 of imm_s. Bit 3 is the sign bit.
        for bit in local.imm_high_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        let imm_high_nibble: AB::Expr = local.imm_high_bits[0].clone()
            + local.imm_high_bits[1].clone() * AB::Expr::TWO
            + local.imm_high_bits[2].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_high_bits[3].clone() * AB::Expr::from(AB::F::from_u32(8));
        let sign_bit: AB::Expr = local.imm_high_bits[3].clone().into();

        // imm_s = imm_se_bytes[0] + imm_high_nibble * 256
        builder.assert_eq(
            local.imm_s.clone(),
            local.imm_se_bytes[0].clone()
                + imm_high_nibble.clone() * AB::Expr::from(AB::F::from_u32(256)),
        );

        // Sign-extend: byte 1 = high nibble + 0xF0 * sign_bit.
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

        // Constrain addr = rs1_value + imm_se_bytes (wrapping u32).
        u32_add(
            builder,
            &local.rs1_value,
            &local.imm_se_bytes,
            &local.addr,
            &local.addr_carries,
        );
    }
}

impl<F: Field> LookupAir<F> for ShAir {
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
        let local: &ShColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is SH with (rs1, rs2, imm_s) from "decode_s" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode_s")),
            &vec![(
                once(F::from_u64(InstructionId::Sh as u64).into())
                    .chain([local.rs1, local.rs2, local.imm_s].into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Read rs1 at timestamp (register, read == write).
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

        // Read rs2 at timestamp + 1 (register, read == write).
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

        // Write halfword (bytes 0–1 of rs2, bytes 2–3 = 0) to RAM at addr, timestamp + 2.
        // The stored value is [rs2_value[0], rs2_value[1], 0, 0].
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::from_u64(2)).into())
                    .chain(once(F::ONE.into())) // memory_type = Ram = 1
                    .chain(local.addr.into_iter().map(Into::into))
                    .chain(local.old_ram_value.into_iter().map(Into::into))
                    .chain(
                        [local.rs2_value[0], local.rs2_value[1]]
                            .into_iter()
                            .map(Into::into)
                            .chain([F::ZERO.into(), F::ZERO.into()]),
                    )
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rs1_value[i].
        lookups.extend(local.rs1_value.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for rs2_value bytes 0–1 (the stored halfword).
        lookups.extend(local.rs2_value[0..2].iter().copied().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Byte range-checks for addr[i].
        lookups.extend(local.addr.into_iter().map(|byte| {
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
                    .chain(once((local.timestamp.clone() + F::from_u64(3)).into()))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        lookups
    }
}
