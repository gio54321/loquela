use std::{borrow::Borrow, borrow::BorrowMut, iter::once, vec};

use crate::primitives::u32_ops::u32_inc;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

#[repr(C)]
pub struct ProgramColumns<F> {
    pub address: [F; 4],
    pub value: F,
    pub mult: F,
    pub inc_carries: [F; 4],
    /// 1 for rows holding real program bytes, 0 for padding rows appended to
    /// round the table to a power of two. Monotone-falling along the trace.
    pub is_real: F,
}

pub const NUM_MEMORY_COLS: usize = size_of::<ProgramColumns<u8>>();

impl<T> Borrow<ProgramColumns<T>> for [T] {
    fn borrow(&self) -> &ProgramColumns<T> {
        debug_assert_eq!(self.len(), NUM_MEMORY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<ProgramColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<ProgramColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut ProgramColumns<T> {
        debug_assert_eq!(self.len(), NUM_MEMORY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<ProgramColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct ProgramAir {
    num_lookups: usize,
}

impl ProgramAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F> BaseAir<F> for ProgramAir {
    fn width(&self) -> usize {
        NUM_MEMORY_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for ProgramAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ProgramColumns<AB::Var> = main.current_slice().borrow();
        let next: &ProgramColumns<AB::Var> = main.next_slice().borrow();

        for limb in local.address.iter() {
            builder.when_first_row().assert_zero(limb.clone());
        }

        let mut t = builder.when_transition();
        u32_inc(&mut t, &local.address, &next.address, &local.inc_carries);

        // The program-image bus uses a single-field-element address derived as
        // a linear combination of `address[0..4]`. Two protections are needed:
        //
        //   1. `inc_carries[3] = 0` — the top-byte carry can't fire, since
        //      that would let the packed address wrap (`addr.next - addr.local
        //      = 1 - 2^32 ≡ -1 (mod p)` at the boundary). Honest traces never
        //      overflow into the top byte.
        //
        //   2. `address[3] = 0` — caps the 32-bit address space to its low 24
        //      bits (< 16 MiB), so the packed field-element address is always
        //      below `2^24 < Mersenne31 modulus`. Without this bound, a
        //      malicious prover could construct a sufficiently long trace
        //      where two distinct ROM positions `j` and `j + p` collide on
        //      the single-element bus key. Loquela programs are far smaller
        //      than 16 MiB, so this is satisfied by every real trace.
        builder.assert_zero(local.inc_carries[3].clone());
        builder.assert_zero(local.address[3].clone());

        // `is_real` is boolean and monotone-falling: once it drops to 0 it
        // stays at 0. This gives the program-image hash a clean "real bytes"
        // selector on the ROM side.
        builder.assert_bool(local.is_real.clone());
        builder
            .when_transition()
            .when(AB::Expr::ONE - local.is_real.clone())
            .assert_zero(next.is_real.clone());

        // No fetches on padding rows: their address is uncommitted/garbage,
        // so binding a fetch to it would let a malicious prover bypass the
        // ROM contents.
        builder.assert_zero(local.mult.clone() * (AB::Expr::ONE - local.is_real.clone()));
    }
}

impl<F: Field> LookupAir<F> for ProgramAir {
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
        let symbolic_main_local: &ProgramColumns<SymbolicVariable<F>> =
            symbolic_main.current_slice().borrow();

        vec![
            self.register_lookup(
                Kind::Global(String::from("program")),
                &vec![(
                    symbolic_main_local
                        .address
                        .into_iter()
                        .chain(once(symbolic_main_local.value))
                        .map(Into::into)
                        .collect(),
                    symbolic_main_local.mult.into(),
                    Direction::Receive,
                )],
            ),
            // Program image bus: each real ROM byte is offered once with the
            // address packed into a single field element. ProgramHashAir sends
            // matching tuples (Send mult = is_active per absorbed byte) to
            // compute a running Poseidon2 hash of the program.
            //
            // `addr_combined = address[0] + 256 * address[1] + 65536 *
            // address[2] + 16777216 * address[3]`, expressed symbolically over
            // the existing 4-limb columns. No new columns are needed on the
            // ROM side; the `inc_carries[3] = 0` assertion above guarantees
            // this combination is a pure `+1` field-element chain pinned to 0
            // on row 0.
            self.register_lookup(
                Kind::Global(String::from("program_image")),
                &vec![(
                    {
                        let addr_combined: SymbolicExpression<F> =
                            SymbolicExpression::from(symbolic_main_local.address[0])
                                + SymbolicExpression::from(symbolic_main_local.address[1])
                                    * SymbolicExpression::from(F::from_u32(1u32 << 8))
                                + SymbolicExpression::from(symbolic_main_local.address[2])
                                    * SymbolicExpression::from(F::from_u32(1u32 << 16))
                                + SymbolicExpression::from(symbolic_main_local.address[3])
                                    * SymbolicExpression::from(F::from_u32(1u32 << 24));
                        vec![addr_combined, symbolic_main_local.value.into()]
                    },
                    symbolic_main_local.is_real.into(),
                    Direction::Receive,
                )],
            ),
        ]
    }
}
