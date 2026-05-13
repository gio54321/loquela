use core::borrow::Borrow;
use std::vec;
use std::vec::Vec;

use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

use super::columns::{
    ProgramHashColumns, BYTES_PER_RATE_ELEM, BYTES_PER_ROW, DIGEST_LEN, NUM_COLS,
    NUM_PUBLIC_VALUES, PV_DIGEST_OFFSET, PV_LENGTH_INDEX, RATE_ELEMS, WIDTH,
};

/// AIR for the Poseidon2 sponge over the program ROM.
///
/// The trace is a real prefix (`flag = 1`) followed by a padding suffix
/// (`flag = 0`). The sponge chains BOTTOM-UP across rows: each real row
/// computes `state.cur = Poseidon(next.state, chunk.cur)`, with the bottom
/// of the chain (the first padding row) pinned to the all-zero `INIT` state.
/// The digest therefore lives in `state` at row 0 and is exposed via
/// `public_values`.
///
/// The Poseidon2 permutation itself is *not* constrained here. Each row
/// commits the permutation input/output as `perm_in` / `perm_out` columns and
/// Sends them on the `poseidon2_perm` bus; the round-, S-box-, and
/// linear-layer constraints live on the `Poseidon2Chip` AIR, which wraps the
/// upstream `p3_poseidon2_air::Poseidon2Air` directly.
#[derive(Clone)]
pub struct ProgramHashAir {
    num_lookups: usize,
}

impl ProgramHashAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl Default for ProgramHashAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BaseAir<F> for ProgramHashAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn num_public_values(&self) -> usize {
        NUM_PUBLIC_VALUES
    }
}

impl<AB> Air<AB> for ProgramHashAir
where
    AB: AirBuilder,
    AB::F: Field + QuotientMap<u32>,
    AB::MainWindow: WindowAccess<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ProgramHashColumns<AB::Var> = main.current_slice().borrow();
        let next: &ProgramHashColumns<AB::Var> = main.next_slice().borrow();

        // ─── 1. Selectors: flag and is_active are boolean ────────────────────
        builder.assert_bool(local.flag.clone());
        for i in 0..BYTES_PER_ROW {
            builder.assert_bool(local.is_active[i].clone());
        }

        // ─── 2. flag is monotone-falling: once 0, stays 0 ────────────────────
        // Equivalently next.flag = 1 ⇒ local.flag = 1, i.e. next.flag * (1 - local.flag) = 0.
        builder
            .when_transition()
            .assert_zero(next.flag.clone() * (AB::Expr::ONE - local.flag.clone().into()));

        // ─── 3. Last row is padding: flag = 0 ────────────────────────────────
        // Guarantees the bottom of the bottom-up sponge chain is pinned to INIT.
        builder.when_last_row().assert_zero(local.flag.clone());

        // ─── 4. is_active non-increasing within a row, zero on padding rows ──
        for i in 0..BYTES_PER_ROW - 1 {
            builder.assert_zero(
                local.is_active[i + 1].clone()
                    * (AB::Expr::ONE - local.is_active[i].clone().into()),
            );
        }
        for i in 0..BYTES_PER_ROW {
            builder.assert_zero(
                (AB::Expr::ONE - local.flag.clone().into()) * local.is_active[i].clone(),
            );
        }

        // ─── 5. Canonical chunking ───────────────────────────────────────────
        //   (a) Every real row absorbs at least one byte: is_active[0] = flag.
        //   (b) Every non-last real row is FULL: flag.cur * flag.next *
        //       (1 - is_active[K-1]) = 0. Only the last real row may be partial.
        // Without these, a prover could insert empty real rows (flag=1,
        // is_active=0) after the program ends, shifting the digest assertion
        // and picking a different digest for the same program bytes.
        builder.assert_eq(local.is_active[0].clone(), local.flag.clone());
        builder.when_transition().assert_zero(
            local.flag.clone()
                * next.flag.clone()
                * (AB::Expr::ONE - local.is_active[BYTES_PER_ROW - 1].clone().into()),
        );

        // ─── 6. Inactive bytes are zero (canonicalizes the absorbed payload) ──
        for i in 0..BYTES_PER_ROW {
            builder.assert_zero(
                (AB::Expr::ONE - local.is_active[i].clone().into()) * local.bytes[i].clone(),
            );
        }

        // ─── 7. base_addr chain: 0 on row 0, +BYTES_PER_ROW per transition ──
        // Per-byte addresses are derived as `base_addr + i` (a degree-1
        // symbolic expression of `base_addr` plus a constant); intra- and
        // cross-row carry math is no longer needed because the chain lives in
        // a single field element (Mersenne31 has 31 bits of headroom; Loquela
        // programs are far smaller than 2^31 bytes). Soundness of the bus is
        // unaffected: ProgramAir's `inc_carries[3] = 0` keeps its
        // single-element address combination a pure `+1` chain pinned to 0,
        // and ProgramHash's chain is `+24` pinned to 0, so the tuples align
        // by induction.
        builder
            .when_first_row()
            .assert_zero(local.base_addr.clone());
        builder.when_transition().assert_eq(
            next.base_addr.clone(),
            local.base_addr.clone() + AB::F::from_u32(BYTES_PER_ROW as u32),
        );

        // ─── 10. Padding-row state pin: (1 - flag) * state[i] = 0 ────────────
        // INIT is all-zero. On padding rows, state[i] = 0 means perm input is
        // irrelevant and the round constraints (gated by flag below) don't
        // fire — padding rows therefore cost no Poseidon witness.
        for i in 0..WIDTH {
            builder
                .assert_zero((AB::Expr::ONE - local.flag.clone().into()) * local.state[i].clone());
        }

        // ─── 11. Bottom-up sponge absorption (real rows only) ────────────────
        // For each real row: perm.inputs[g] = next.state[g] + packed[g] for
        // rate lanes, perm.inputs[g] = next.state[g] for capacity lanes. The
        // permutation output equals local.state[i].
        //
        // All Poseidon constraints (including the absorption hookup and the
        // round chain) are multiplied by `local.flag`, so on padding rows the
        // entire Poseidon witness is freely zero. Degree of the inner round
        // constraint is 3 (the (5,1) S-box recipe); multiplied by flag gives
        // degree 4, within the SBOX_DEGREE=5 budget reported by
        // `max_constraint_degree`.

        // Absorption inputs to the permutation.
        // Rate absorption: perm_in[g] = next.state[g] + packed[g] for g < RATE,
        // gated by flag.
        for g in 0..RATE_ELEMS {
            let b0 = local.bytes[BYTES_PER_RATE_ELEM * g].clone();
            let b1 = local.bytes[BYTES_PER_RATE_ELEM * g + 1].clone();
            let b2 = local.bytes[BYTES_PER_RATE_ELEM * g + 2].clone();
            let packed: AB::Expr = b0.into()
                + b1.into() * AB::F::from_u32(1u32 << 8)
                + b2.into() * AB::F::from_u32(1u32 << 16);
            builder.assert_zero(
                local.flag.clone()
                    * (local.perm_in[g].clone().into() - next.state[g].clone().into() - packed),
            );
        }
        // Capacity passthrough: perm_in[g] = next.state[g] for g >= RATE.
        for g in RATE_ELEMS..WIDTH {
            builder.assert_zero(
                local.flag.clone()
                    * (local.perm_in[g].clone().into() - next.state[g].clone().into()),
            );
        }

        // The Poseidon2 permutation itself is enforced on the Poseidon2Chip
        // AIR (via the upstream `Poseidon2Air`); on this AIR we only need to
        // commit the input/output and link the output back to `state`.

        // state[i] = perm_out[i] on real rows. On padding rows (flag = 0)
        // `state` is already pinned to zero by the padding-row state pin
        // above, and `perm_out` is unconstrained (it never reaches the bus
        // because the Send has multiplicity `flag`).
        for i in 0..WIDTH {
            builder.assert_zero(
                local.flag.clone()
                    * (local.state[i].clone().into() - local.perm_out[i].clone().into()),
            );
        }

        // ─── 12. cum_active accumulator (forward) ────────────────────────────
        // Counts real bytes seen up to and including this row.
        let mut row_sum: AB::Expr = AB::Expr::ZERO;
        for i in 0..BYTES_PER_ROW {
            row_sum = row_sum + local.is_active[i].clone().into();
        }
        let mut next_row_sum: AB::Expr = AB::Expr::ZERO;
        for i in 0..BYTES_PER_ROW {
            next_row_sum = next_row_sum + next.is_active[i].clone().into();
        }
        builder
            .when_first_row()
            .assert_eq(local.cum_active.clone(), row_sum);
        builder.when_transition().assert_eq(
            next.cum_active.clone(),
            local.cum_active.clone() + next_row_sum,
        );

        // ─── 13. Public values: digest at row 0, length at last row ──────────
        // Snapshot public-value expressions before re-borrowing `builder` for
        // the boundary sub-builders (`builder.public_values()` immutably
        // borrows `builder`).
        let pv_digest: [AB::Expr; DIGEST_LEN] = {
            let public_values = builder.public_values();
            assert_eq!(public_values.len(), NUM_PUBLIC_VALUES);
            core::array::from_fn(|i| public_values[PV_DIGEST_OFFSET + i].into())
        };
        let pv_length: AB::Expr = {
            let public_values = builder.public_values();
            public_values[PV_LENGTH_INDEX].into()
        };
        for (i, pv) in pv_digest.into_iter().enumerate() {
            builder
                .when_first_row()
                .assert_eq(local.state[i].clone(), pv);
        }
        builder
            .when_last_row()
            .assert_eq(local.cum_active.clone(), pv_length);
    }
}

impl<F: Field> LookupAir<F> for ProgramHashAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: NUM_COLS,
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_local: &ProgramHashColumns<SymbolicVariable<F>> =
            symbolic_main.current_slice().borrow();

        let mut lookups = Vec::new();

        // Program-image bus: one tuple per byte, (base_addr + i, byte), mult =
        // is_active[i]. ProgramAir Receives matching tuples (its own
        // single-element address comes from a symbolic combination of the
        // 4-limb `address` columns); zero-sum forces the prover to absorb
        // exactly the real ROM bytes in order.
        for i in 0..BYTES_PER_ROW {
            let addr_expr: SymbolicExpression<F> =
                SymbolicExpression::from(symbolic_local.base_addr)
                    + SymbolicExpression::from(F::from_u32(i as u32));
            let elements: Vec<SymbolicExpression<F>> =
                vec![addr_expr, symbolic_local.bytes[i].clone().into()];
            let mult: SymbolicExpression<F> = symbolic_local.is_active[i].clone().into();
            lookups.push(self.register_lookup(
                Kind::Global(String::from("program_image")),
                &vec![(elements, mult, Direction::Send)],
            ));
        }

        // Byte range checks for every absorbed byte: required because the
        // ProgramAir does not byte-range-check its `value` column. Without this
        // a malicious prover could put a non-byte field element in
        // `local.bytes[i]` and exploit the 24-bit packing.
        for i in 0..BYTES_PER_ROW {
            let elements: Vec<SymbolicExpression<F>> = vec![symbolic_local.bytes[i].clone().into()];
            let mult: SymbolicExpression<F> = symbolic_local.is_active[i].clone().into();
            lookups.push(self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(elements, mult, Direction::Send)],
            ));
        }

        // Poseidon2 permutation: Send (perm_in[0..16], perm_out[0..16]) with
        // multiplicity `flag`. The Poseidon2Chip AIR Receives the matching
        // tuples (with multiplicity = its own preprocessed `is_real`) and
        // enforces the Poseidon2 round constraints via the upstream AIR.
        {
            let elements: Vec<SymbolicExpression<F>> = symbolic_local
                .perm_in
                .iter()
                .chain(symbolic_local.perm_out.iter())
                .map(|v| (*v).into())
                .collect();
            let mult: SymbolicExpression<F> = symbolic_local.flag.clone().into();
            lookups.push(self.register_lookup(
                Kind::Global(String::from("poseidon2_perm")),
                &vec![(elements, mult, Direction::Send)],
            ));
        }

        lookups
    }
}
