use std::borrow::{Borrow, BorrowMut};
use std::iter::once;

use crate::decode::air::InstructionId;
use crate::primitives::u32_ops::u32_plus_four;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

/// One row per SRA instruction execution.
///
/// SRA: `rd = (rs1 as i32) >> (rs2 & 0x1F)` (arithmetic right shift, fills with sign bit).
///
/// Shift decomposition:
///   `shamt = rs2_value[0] & 0x1F`   (low 5 bits of rs2)
///   `bit_shamt = shamt % 8`          (lower 3 bits, controls within-byte shift, 0..7)
///   `byte_shamt = shamt / 8`         (upper 2 bits, controls whole-byte shift, 0..3)
///   `rs2_shamt_high = rs2_value[0] >> 5`  (upper 3 bits of rs2_value[0], 0..7)
///
/// Sign extension:
///   `sign_bit = rs1_bytes[3] >> 7`   (MSB of rs1, either 0 or 1)
///   `fill_byte = sign_bit * 255`     (fill byte for vacated high bytes: 0x00 or 0xFF)
///   `rs1_byte3_low7 = rs1_bytes[3] - sign_bit * 128`  (low 7 bits of rs1_bytes[3])
///
/// Within-byte shift: same as SRL — for each byte `i` of rs1:
///   `(rs1_bytes[i], bit_shamt) -> (shifted_bytes[i], carry_bytes[i])` via `"byte_srl"` bus.
///
/// Whole-byte shift: same arrangement as SRL, but vacated high bytes are filled with
/// `fill_byte` instead of 0:
///   rd_bytes[3] = is_bs0 * inter3 + (is_bs1 + is_bs2 + is_bs3) * fill_byte
///   rd_bytes[2] = is_bs0 * inter2 + is_bs1 * inter3 + (is_bs2 + is_bs3) * fill_byte
///   rd_bytes[1] = is_bs0 * inter1 + is_bs1 * inter2 + is_bs2 * inter3 + is_bs3 * fill_byte
///   rd_bytes[0] = is_bs0 * inter0 + is_bs1 * inter1 + is_bs2 * inter2 + is_bs3 * inter3
///
/// Wiring:
///   - Receives `(pc[0..4], timestamp)` from the "trace" bus.
///   - Sends `(InstructionId::Sra, rd, rs1, rs2)` to the "decode" bus.
///   - Sends three operations to the "memory" bus: read rs1, read rs2, write rd.
///   - Sends four byte-quadruple tuples to the "byte_srl" bus.
///   - Sends byte range-checks for rs2_value, rd_bytes, rs1_byte3_low7 to the "bytes" bus.
///   - Sends `(next_pc[0..4], timestamp + 3)` to the "trace" bus.
#[repr(C)]
pub struct SraColumns<F> {
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
    pub rs2_value: [F; 4],
    /// Old value of register `rd` (before write), as byte limbs.
    pub old_rd_value: [F; 4],
    /// New value written to `rd`, as byte limbs.
    pub rd_bytes: [F; 4],

    /// `bit_shamt = shamt % 8` (lower 3 bits of shamt, range 0..7).
    pub bit_shamt: F,
    /// `byte_shamt = shamt / 8` (upper 2 bits of shamt, range 0..3).
    pub byte_shamt: F,
    /// Upper 3 bits of `rs2_value[0]`: `rs2_value[0] >> 5`, range 0..7.
    pub rs2_shamt_high: F,

    /// One-hot selectors for byte_shamt value (exactly one is 1).
    pub is_bs0: F, // byte_shamt == 0
    pub is_bs1: F, // byte_shamt == 1
    pub is_bs2: F, // byte_shamt == 2
    pub is_bs3: F, // byte_shamt == 3

    /// For each byte of rs1: the within-byte shifted result (shifted right).
    /// `shifted_bytes[i] = (rs1_bytes[i] >> bit_shamt) & 0xFF`
    pub shifted_bytes: [F; 4],
    /// For each byte of rs1: the carry that flows into the *previous* (lower index) byte.
    /// `carry_bytes[i] = (rs1_bytes[i] << (8 - bit_shamt)) & 0xFF` (0 when bit_shamt == 0)
    pub carry_bytes: [F; 4],

    /// Sign bit of rs1: MSB of rs1_bytes[3], either 0 or 1.
    pub sign_bit: F,
    /// Fill byte for vacated high positions: `sign_bit * 255` (0x00 or 0xFF).
    pub fill_byte: F,
    /// Low 7 bits of rs1_bytes[3]: `rs1_bytes[3] - sign_bit * 128`, range 0..127.
    pub rs1_byte3_low7: F,

    /// `pc + 4` as four byte limbs, constrained by `u32_plus_four`.
    pub next_pc: [F; 4],
    /// Carry bits for the `pc + 4` addition (bytes 0–2 only; top carry is dropped).
    pub next_pc_carries: [F; 3],

    /// Padding selector: 1 for real execution rows, 0 for dummy/padding rows.
    pub is_dummy: F,
}

pub const NUM_SRA_COLS: usize = size_of::<SraColumns<u8>>();

impl<T> Borrow<SraColumns<T>> for [T] {
    fn borrow(&self) -> &SraColumns<T> {
        debug_assert_eq!(self.len(), NUM_SRA_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<SraColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<SraColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut SraColumns<T> {
        debug_assert_eq!(self.len(), NUM_SRA_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<SraColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct SraAir {
    num_lookups: usize,
}

impl SraAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for SraAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for SraAir {
    fn width(&self) -> usize {
        NUM_SRA_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for SraAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: PrimeCharacteristicRing + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &SraColumns<AB::Var> = main.current_slice().borrow();

        builder.assert_bool(local.is_dummy.clone());

        // Constrain next_pc = pc + 4 with carry propagation.
        u32_plus_four(builder, &local.pc, &local.next_pc, &local.next_pc_carries);

        // is_bs* are boolean.
        builder.assert_bool(local.is_bs0.clone());
        builder.assert_bool(local.is_bs1.clone());
        builder.assert_bool(local.is_bs2.clone());
        builder.assert_bool(local.is_bs3.clone());

        // Exactly one is_bs* is set (byte_shamt is one-hot encoded).
        builder.assert_eq(
            local.is_bs0.clone()
                + local.is_bs1.clone()
                + local.is_bs2.clone()
                + local.is_bs3.clone(),
            AB::Expr::ONE,
        );

        // byte_shamt = 0*is_bs0 + 1*is_bs1 + 2*is_bs2 + 3*is_bs3.
        builder.assert_eq(
            local.byte_shamt.clone(),
            local.is_bs1.clone()
                + local.is_bs2.clone() * AB::Expr::from(AB::F::from_u32(2))
                + local.is_bs3.clone() * AB::Expr::from(AB::F::from_u32(3)),
        );

        // Shamt decomposition: rs2_value[0] = bit_shamt + 8 * byte_shamt + 32 * rs2_shamt_high.
        builder.assert_eq(
            local.rs2_value[0].clone(),
            local.bit_shamt.clone()
                + local.byte_shamt.clone() * AB::Expr::from(AB::F::from_u32(8))
                + local.rs2_shamt_high.clone() * AB::Expr::from(AB::F::from_u32(32)),
        );

        // sign_bit is boolean.
        builder.assert_bool(local.sign_bit.clone());

        // fill_byte = sign_bit * 255.
        builder.assert_eq(
            local.fill_byte.clone(),
            local.sign_bit.clone().into() * AB::Expr::from(AB::F::from_u32(255)),
        );

        // rs1_bytes[3] = sign_bit * 128 + rs1_byte3_low7.
        builder.assert_eq(
            local.rs1_bytes[3].clone(),
            local.sign_bit.clone().into() * AB::Expr::from(AB::F::from_u32(128))
                + local.rs1_byte3_low7.clone().into(),
        );

        // For SRA, carry from byte i flows into byte i-1 (lower index), same as SRL.
        // inter_bytes[3] = shifted_bytes[3]             (high byte, no carry in)
        // inter_bytes[i] = shifted_bytes[i] + carry_bytes[i+1]  for i=0,1,2
        let inter0: AB::Expr =
            local.shifted_bytes[0].clone().into() + local.carry_bytes[1].clone().into();
        let inter1: AB::Expr =
            local.shifted_bytes[1].clone().into() + local.carry_bytes[2].clone().into();
        let inter2: AB::Expr =
            local.shifted_bytes[2].clone().into() + local.carry_bytes[3].clone().into();
        let inter3: AB::Expr = local.shifted_bytes[3].clone().into();

        let fill: AB::Expr = local.fill_byte.clone().into();

        // Whole-byte shift for SRA: each byte of rs1 at position j maps to rd at position
        // j - byte_shamt. Positions below 0 are discarded; positions above 3 are filled
        // with fill_byte (0x00 for positive rs1, 0xFF for negative rs1).
        //
        // rd_bytes[3]: from inter3 (bs=0) or fill_byte (bs=1,2,3).
        builder.assert_eq(
            local.rd_bytes[3].clone(),
            local.is_bs0.clone().into() * inter3.clone()
                + (local.is_bs1.clone().into()
                    + local.is_bs2.clone().into()
                    + local.is_bs3.clone().into())
                    * fill.clone(),
        );

        // rd_bytes[2]: from inter2 (bs=0), inter3 (bs=1), fill_byte (bs=2,3).
        builder.assert_eq(
            local.rd_bytes[2].clone(),
            local.is_bs0.clone().into() * inter2.clone()
                + local.is_bs1.clone().into() * inter3.clone()
                + (local.is_bs2.clone().into() + local.is_bs3.clone().into()) * fill.clone(),
        );

        // rd_bytes[1]: from inter1 (bs=0), inter2 (bs=1), inter3 (bs=2), fill_byte (bs=3).
        builder.assert_eq(
            local.rd_bytes[1].clone(),
            local.is_bs0.clone().into() * inter1.clone()
                + local.is_bs1.clone().into() * inter2.clone()
                + local.is_bs2.clone().into() * inter3.clone()
                + local.is_bs3.clone().into() * fill.clone(),
        );

        // rd_bytes[0]: from inter0 (bs=0), inter1 (bs=1), inter2 (bs=2), inter3 (bs=3).
        builder.assert_eq(
            local.rd_bytes[0].clone(),
            local.is_bs0.clone().into() * inter0.clone()
                + local.is_bs1.clone().into() * inter1.clone()
                + local.is_bs2.clone().into() * inter2.clone()
                + local.is_bs3.clone().into() * inter3.clone(),
        );
    }
}

impl<F: Field> LookupAir<F> for SraAir {
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
        let local: &SraColumns<SymbolicVariable<F>> = symbolic_main.current_slice().borrow();

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

        // Assert the decoded instruction is SRA with (rd, rs1, rs2) from the "decode" bus.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("decode")),
            &vec![(
                once(F::from_u64(InstructionId::Sra as u64).into())
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
                    .chain(local.rs2_value.into_iter().map(Into::into))
                    .chain(local.rs2_value.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Write the shift result to rd at timestamp + 2.
        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                once((local.timestamp.clone() + F::from_u64(2)).into())
                    .chain(once(F::ZERO.into()))
                    .chain(once(local.rd.into()))
                    .chain([F::ZERO; 3].into_iter().map(Into::into))
                    .chain(local.old_rd_value.into_iter().map(Into::into))
                    .chain(local.rd_bytes.into_iter().map(Into::into))
                    .collect(),
                local.is_dummy.into(),
                Direction::Send,
            )],
        ));

        // Four byte_srl lookups: one per byte of rs1.
        // Each lookup sends (rs1_bytes[i], bit_shamt, shifted_bytes[i], carry_bytes[i]).
        // Reuses the same "byte_srl" bus as SRL (same per-byte shift computation).
        lookups.extend(
            local
                .rs1_bytes
                .into_iter()
                .zip(local.shifted_bytes.into_iter())
                .zip(local.carry_bytes.into_iter())
                .map(|((byte, shifted), carry)| {
                    self.register_lookup(
                        Kind::Global(String::from("byte_srl")),
                        &vec![(
                            vec![
                                byte.into(),
                                local.bit_shamt.into(),
                                shifted.into(),
                                carry.into(),
                            ],
                            local.is_dummy.into(),
                            Direction::Send,
                        )],
                    )
                }),
        );

        // Byte range-checks for rs2_value[i] via "bytes" bus.
        lookups.extend(local.rs2_value.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Range-check rs2_shamt_high via "bytes" bus (value in 0..7).
        lookups.push(self.register_lookup(
            Kind::Global(String::from("bytes")),
            &vec![(
                vec![local.rs2_shamt_high.into()],
                F::ONE.into(),
                Direction::Send,
            )],
        ));

        // Byte range-checks for rd_bytes[i] via "bytes" bus.
        lookups.extend(local.rd_bytes.into_iter().map(|byte| {
            self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(vec![byte.into()], F::ONE.into(), Direction::Send)],
            )
        }));

        // Range-check rs1_byte3_low7 via "bytes" bus (value in 0..127).
        lookups.push(self.register_lookup(
            Kind::Global(String::from("bytes")),
            &vec![(
                vec![local.rs1_byte3_low7.into()],
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
