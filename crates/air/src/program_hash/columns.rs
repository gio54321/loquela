use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use p3_poseidon2_air::Poseidon2Cols;

/// Width-16 Mersenne31 Poseidon2 parameters.
pub const WIDTH: usize = 16;
pub const SBOX_DEGREE: u64 = 5;
pub const SBOX_REGISTERS: usize = 1;
pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 14;

/// Number of program bytes absorbed per Poseidon2 row.
///
/// 8 rate elements, 3 bytes packed into each as little-endian (24 bits, safely
/// below the 31-bit Mersenne31 prime).
pub const BYTES_PER_ROW: usize = 24;
pub const RATE_ELEMS: usize = 8;
pub const BYTES_PER_RATE_ELEM: usize = 3;

/// Number of field elements in the exposed digest (half-state output).
pub const DIGEST_LEN: usize = 8;

/// Public-value layout: 8 digest elements followed by the program length.
pub const NUM_PUBLIC_VALUES: usize = DIGEST_LEN + 1;
pub const PV_DIGEST_OFFSET: usize = 0;
pub const PV_LENGTH_INDEX: usize = DIGEST_LEN;

pub type P2Cols<T> =
    Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

/// Columns for the program-image Poseidon2 sponge AIR.
///
/// The trace is split into a real prefix (`flag = 1`) and a padding suffix
/// (`flag = 0`). The sponge chains bottom-up: each real row computes
/// `state.cur = Poseidon(state.next, chunk.cur)`, with the bottom of the chain
/// (the first padding row) pinned to the all-zero `INIT` state. The digest
/// therefore lives in `state` at row 0 and is exposed via `public_values`.
///
/// Padding rows have `state = INIT = 0` and zero Poseidon witness; round
/// constraints are gated by `flag`, so the prover pays no Poseidon cost on the
/// padding suffix.
#[repr(C)]
pub struct ProgramHashColumns<T> {
    /// Sponge state at this row.
    /// - Real rows (`flag = 1`): `state = Poseidon(next.state, chunk.cur)`,
    ///   i.e. the digest accumulated by absorbing chunks at rows
    ///   `[row_idx, k-1]` starting from `INIT` below.
    /// - Padding rows (`flag = 0`): constrained to `INIT = [0; WIDTH]`.
    /// - Row 0: `state[0..DIGEST_LEN]` is asserted equal to the public-value
    ///   digest.
    pub state: [T; WIDTH],

    /// Program bytes absorbed on this row, in order.
    pub bytes: [T; BYTES_PER_ROW],

    /// Address of each absorbed byte as 4 byte-limbs. `addrs[0]` is the row's
    /// base address; `addrs[i + 1] = addrs[i] + 1` is enforced via
    /// `addr_inc_carries[i]`. The 4-limb form matches the `program_image`
    /// bus tuple shape.
    pub addrs: [[T; 4]; BYTES_PER_ROW],

    /// Carries for `addrs[i + 1] = addrs[i] + 1` (intra-row `u32_inc`).
    pub addr_inc_carries: [[T; 4]; BYTES_PER_ROW - 1],

    /// Carries linking the last in-row address `addrs[BYTES_PER_ROW - 1]` to
    /// the next row's `addrs[0]` (cross-row `u32_inc`).
    pub cross_row_addr_inc_carries: [T; 4],

    /// Per-byte activity flag. 1 = real program byte (Sent on the
    /// `program_image` and `bytes` buses), 0 = trailing pad bytes on the last
    /// real row or entirely-padding rows. Non-increasing left-to-right.
    pub is_active: [T; BYTES_PER_ROW],

    /// Real-row selector. 1 on rows that contain real program bytes, 0 on the
    /// padding suffix. Monotone-falling (once 0, stays 0). The last row is
    /// pinned to `flag = 0`, so the bottom-up chain always has a well-grounded
    /// `INIT` base.
    pub flag: T,

    /// Running count of absorbed real bytes so far (including this row).
    /// Asserted against `public_values[PV_LENGTH_INDEX]` on the last row.
    pub cum_active: T,

    /// Poseidon2 permutation witness (inputs + per-round S-box registers and
    /// post-states). On real rows the AIR constrains
    /// `perm.inputs[g] = next.state[g] + packed[g]` for rate lanes and
    /// `perm.inputs[g] = next.state[g]` for capacity lanes, and
    /// `state[i] = perm.ending_full_rounds[..].post[i]`. All Poseidon
    /// constraints are gated by `flag`, so this witness is freely zero on
    /// padding rows.
    pub perm: P2Cols<T>,
}

pub const NUM_COLS: usize = size_of::<ProgramHashColumns<u8>>();

impl<T> Borrow<ProgramHashColumns<T>> for [T] {
    fn borrow(&self) -> &ProgramHashColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<ProgramHashColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<ProgramHashColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut ProgramHashColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<ProgramHashColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
