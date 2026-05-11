use core::borrow::Borrow;
use std::vec;
use std::vec::Vec;

use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    WindowAccess,
};
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing, PrimeField32};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};
use p3_mersenne_31::{
    GenericPoseidon2LinearLayersMersenne31, MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL,
    MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL, MERSENNE31_POSEIDON2_RC_16_INTERNAL,
};
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{FullRound, PartialRound, SBox};

use super::columns::{
    NUM_COLS, NUM_PUBLIC_VALUES, ProgramHashColumns, BYTES_PER_RATE_ELEM, BYTES_PER_ROW,
    DIGEST_LEN, HALF_FULL_ROUNDS, PARTIAL_ROUNDS, PV_DIGEST_OFFSET, PV_LENGTH_INDEX, RATE_ELEMS,
    SBOX_DEGREE, SBOX_REGISTERS, WIDTH,
};

/// AIR for the Poseidon2 sponge over the program ROM.
///
/// The trace is a real prefix (`flag = 1`) followed by a padding suffix
/// (`flag = 0`). The sponge chains BOTTOM-UP across rows:
///   - Padding rows pin `state = INIT = 0`. No Poseidon witness is needed; all
///     round constraints are gated by `flag` so the AIR pays zero cost on
///     padding rows.
///   - Real rows compute `state.cur = Poseidon(next.state, chunk.cur)` where
///     `chunk.cur` is the 24 absorbed bytes packed 3 per rate lane. The bottom
///     of the chain (row `k - 1`) pulls `next.state` from the first padding
///     row, which is `INIT`.
///
/// Consequences:
///   - The digest naturally lives at row 0 (`state[0..DIGEST_LEN]`) and is
///     pinned to the public-value vector via a clean `when_first_row` boundary.
///   - Program length `L = cum_active` (forward accumulator) is pinned to the
///     last row's `cum_active` via `when_last_row`.
///   - Absorption order is reverse of trace order: bytes of `chunk[k-1]`
///     (program's tail) are absorbed first into INIT, bytes of `chunk[0]`
///     (program's head) are absorbed last. The host-side helper
///     `compute_program_digest` mirrors this order.
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

    /// All constraints are bounded by the Poseidon2 S-box recipe at degree 3
    /// (committing `x^3` and then asserting `x^5 = x^3 * x^2`). Boolean and
    /// `is_active` × `bytes` cross-products are also degree 2-3. Mirror the
    /// upstream `Poseidon2Air` bound of `SBOX_DEGREE` (5) as an upper limit so
    /// p3-batch-stark allocates the right number of quotient chunks.
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(SBOX_DEGREE as usize)
    }
}

/// S-box helper for `(DEGREE=5, REGISTERS=1)`. Equivalent in shape to
/// `p3_poseidon2_air::eval_sbox` but inlined because that one is crate-private.
///
/// One witness register `r = x^3` (constrained as `r - x*x*x = 0`, degree 3).
/// Returns `x^5` as `r * x * x` (degree 3 in the witness).
fn eval_sbox_5_1<AB: AirBuilder>(sbox: &SBox<AB::Var, 5, 1>, x: &mut AB::Expr, builder: &mut AB) {
    let committed_x3: AB::Expr = sbox.0[0].into();
    let x_clone = x.clone();
    let x2 = x_clone.clone() * x_clone.clone();
    // Force the register to hold x^3.
    builder.assert_eq(committed_x3.clone(), x2.clone() * x_clone.clone());
    // Output x^5 = x^3 * x^2.
    *x = committed_x3 * x2;
}

fn eval_full_round<AB, LinearLayers>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[AB::Expr; WIDTH],
    builder: &mut AB,
) where
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
{
    for (i, (s, rc)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s = s.clone() + rc.clone();
        eval_sbox_5_1(&full_round.sbox[i], s, builder);
    }
    <LinearLayers as GenericPoseidon2LinearLayers<WIDTH>>::external_linear_layer::<AB::Expr>(state);
    for (state_i, post_i) in state.iter_mut().zip(full_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

fn eval_partial_round<AB, LinearLayers>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
{
    state[0] = state[0].clone() + round_constant;
    eval_sbox_5_1(&partial_round.sbox, &mut state[0], builder);
    builder.assert_eq(state[0].clone(), partial_round.post_sbox);
    state[0] = partial_round.post_sbox.into();
    <LinearLayers as GenericPoseidon2LinearLayers<WIDTH>>::internal_linear_layer::<AB::Expr>(state);
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
        builder.when_transition().assert_zero(
            next.flag.clone() * (AB::Expr::ONE - local.flag.clone().into()),
        );

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
            builder.assert_zero(
                (AB::Expr::ONE - local.flag.clone().into()) * local.state[i].clone(),
            );
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
        for g in 0..RATE_ELEMS {
            let b0 = local.bytes[BYTES_PER_RATE_ELEM * g].clone();
            let b1 = local.bytes[BYTES_PER_RATE_ELEM * g + 1].clone();
            let b2 = local.bytes[BYTES_PER_RATE_ELEM * g + 2].clone();
            let packed: AB::Expr = b0.into()
                + b1.into() * AB::F::from_u32(1u32 << 8)
                + b2.into() * AB::F::from_u32(1u32 << 16);
            builder.assert_zero(
                local.flag.clone()
                    * (local.perm.inputs[g].clone().into()
                        - next.state[g].clone().into()
                        - packed),
            );
        }
        for g in RATE_ELEMS..WIDTH {
            builder.assert_zero(
                local.flag.clone()
                    * (local.perm.inputs[g].clone().into() - next.state[g].clone().into()),
            );
        }

        // Poseidon2 round chain: initial MDS → 4 full → 14 partial → 4 full.
        // Wrapped in a `when(flag)` filtered builder so every assert_eq inside
        // is gated by flag. State propagates as symbolic expressions through
        // the chain; on padding rows the chain still executes symbolically but
        // every assertion is multiplied by 0.
        {
            let mut fb = builder.when(local.flag.clone());
            let mut state: [AB::Expr; WIDTH] =
                local.perm.inputs.clone().map(AB::Expr::from);
            <GenericPoseidon2LinearLayersMersenne31 as GenericPoseidon2LinearLayers<WIDTH>>::external_linear_layer::<AB::Expr>(&mut state);
            for r in 0..HALF_FULL_ROUNDS {
                let rc_row = &MERSENNE31_POSEIDON2_RC_16_EXTERNAL_INITIAL[r];
                let rc_exprs: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
                    AB::Expr::from(AB::F::from_u32(rc_row[i].as_canonical_u32()))
                });
                eval_full_round::<_, GenericPoseidon2LinearLayersMersenne31>(
                    &mut state,
                    &local.perm.beginning_full_rounds[r],
                    &rc_exprs,
                    &mut fb,
                );
            }
            for r in 0..PARTIAL_ROUNDS {
                let rc: AB::Expr = AB::Expr::from(AB::F::from_u32(
                    MERSENNE31_POSEIDON2_RC_16_INTERNAL[r].as_canonical_u32(),
                ));
                eval_partial_round::<_, GenericPoseidon2LinearLayersMersenne31>(
                    &mut state,
                    &local.perm.partial_rounds[r],
                    rc,
                    &mut fb,
                );
            }
            for r in 0..HALF_FULL_ROUNDS {
                let rc_row = &MERSENNE31_POSEIDON2_RC_16_EXTERNAL_FINAL[r];
                let rc_exprs: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
                    AB::Expr::from(AB::F::from_u32(rc_row[i].as_canonical_u32()))
                });
                eval_full_round::<_, GenericPoseidon2LinearLayersMersenne31>(
                    &mut state,
                    &local.perm.ending_full_rounds[r],
                    &rc_exprs,
                    &mut fb,
                );
            }
        }

        // local.state[i] = local.perm.ending_full_rounds[last].post[i], gated by flag.
        for i in 0..WIDTH {
            builder.assert_zero(
                local.flag.clone()
                    * (local.state[i].clone().into()
                        - local.perm.ending_full_rounds[HALF_FULL_ROUNDS - 1].post[i]
                            .clone()
                            .into()),
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
            let elements: Vec<SymbolicExpression<F>> =
                vec![symbolic_local.bytes[i].clone().into()];
            let mult: SymbolicExpression<F> = symbolic_local.is_active[i].clone().into();
            lookups.push(self.register_lookup(
                Kind::Global(String::from("bytes")),
                &vec![(elements, mult, Direction::Send)],
            ));
        }

        lookups
    }
}
