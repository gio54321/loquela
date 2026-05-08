use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::{u32_add, u32_plus_four};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per LB instruction execution.
///
/// LB: `rd = sign_extend_8(MEM8[rs1 + imm])`.
///
/// Sign extension: if bit 7 of loaded_val[0] is set, upper bytes are 0xFF;
/// otherwise they are 0x00.
///
/// Constraints:
///   - `rd_val[0] = loaded_val[0]`
///   - `loaded_val[0] = sign_bit * 128 + byte_low7`
///   - `rd_val[1] = rd_val[2] = rd_val[3] = sign_bit * 0xFF`
#[repr(C)]
pub struct LbColumns<F> {
    pub pc: [F; 4],
    pub timestamp: F,
    pub rd: F,
    pub rs1: F,
    pub imm: F,

    pub imm_high_bits: [F; 4],
    pub imm_se_bytes: [F; 4],

    pub rs1_value: [F; 4],
    pub addr_bytes: [F; 4],
    pub addr_carries: [F; 4],

    pub loaded_val: [F; 4],
    pub old_rd_value: [F; 4],
    /// rd_val = sign_extend_8(loaded_val[0]).
    pub rd_val: [F; 4],

    /// Sign bit of loaded_val[0] (bit 7).
    pub sign_bit: F,
    /// Low 7 bits of loaded_val[0]: `loaded_val[0] = sign_bit * 128 + byte_low7`.
    pub byte_low7: F,

    pub next_pc: [F; 4],
    pub next_pc_carries: [F; 3],

    pub is_dummy: F,
}

pub const NUM_LB_COLS: usize = size_of::<LbColumns<u8>>();

impl<T> Borrow<LbColumns<T>> for [T] {
    fn borrow(&self) -> &LbColumns<T> {
        debug_assert_eq!(self.len(), NUM_LB_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<LbColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<LbColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut LbColumns<T> {
        debug_assert_eq!(self.len(), NUM_LB_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<LbColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct LbAir {
    num_lookups: usize,
}

impl LbAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for LbAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for LbAir {
    fn width(&self) -> usize {
        NUM_LB_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for LbAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &LbColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // Sign-extend the 12-bit immediate.
        for bit in local.imm_high_bits.iter() {
            builder.assert_bool(bit.clone());
        }
        let imm_high_nibble: AB::Expr = local.imm_high_bits[0].clone()
            + local.imm_high_bits[1].clone() * AB::Expr::TWO
            + local.imm_high_bits[2].clone() * AB::Expr::from(AB::F::from_u32(4))
            + local.imm_high_bits[3].clone() * AB::Expr::from(AB::F::from_u32(8));
        let sign_bit_imm: AB::Expr = local.imm_high_bits[3].clone().into();

        builder.assert_eq(
            local.imm.clone(),
            local.imm_se_bytes[0].clone()
                + imm_high_nibble.clone() * AB::Expr::from(AB::F::from_u32(256)),
        );
        builder.assert_eq(
            local.imm_se_bytes[1].clone(),
            imm_high_nibble + sign_bit_imm.clone() * AB::Expr::from(AB::F::from_u32(0xF0)),
        );
        builder.assert_eq(
            local.imm_se_bytes[2].clone(),
            sign_bit_imm.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.imm_se_bytes[3].clone(),
            sign_bit_imm * AB::Expr::from(AB::F::from_u32(0xFF)),
        );

        u32_add(
            builder,
            &local.rs1_value,
            &local.imm_se_bytes,
            &local.addr_bytes,
            &local.addr_carries,
        );

        // Sign bit extraction: loaded_val[0] = sign_bit * 128 + byte_low7.
        builder.assert_bool(local.sign_bit.clone());
        builder.assert_eq(
            local.loaded_val[0].clone(),
            local.sign_bit.clone() * AB::Expr::from(AB::F::from_u32(128)) + local.byte_low7.clone(),
        );

        // rd_val[0] = loaded_val[0] (same byte).
        builder.assert_eq(local.rd_val[0].clone(), local.loaded_val[0].clone());
        // Upper bytes are the sign extension: 0xFF if sign_bit=1, else 0x00.
        builder.assert_eq(
            local.rd_val[1].clone(),
            local.sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.rd_val[2].clone(),
            local.sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
        builder.assert_eq(
            local.rd_val[3].clone(),
            local.sign_bit.clone() * AB::Expr::from(AB::F::from_u32(0xFF)),
        );
    }
}

impl<F: Field> LookupAir<F> for LbAir {
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
        let local: &LbColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

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

        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                once(F::from_u64(InstructionId::Lb as u64).into())
                    .chain([local.rd, local.rs1, local.imm].into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // ts+0: Register read rs1.
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

        // ts+1: RAM read at addr.
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

        // ts+2: Register write rd ← rd_val.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::from_u64(2)).into())
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

        lookups.extend(local.rs1_value.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        lookups.extend(local.addr_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        lookups.extend(local.loaded_val.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Range-check byte_low7 (must be in [0, 127]).
        lookups.push(self.register_lookup(
            Kind::Global(String::from("bytes")),
            &vec![(vec![local.byte_low7.into()], F::ONE.into(), Direction::Send)],
        ));

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
