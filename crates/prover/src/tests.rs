use p3_batch_stark::verify_batch;
use p3_field::PrimeCharacteristicRing;

use crate::{build_config, do_prove, generate_traces, prove, prove_traces, AllTraces, Val};

fn encode_addi(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
    let word =
        ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0b001_0011;
    word.to_le_bytes()
}

fn encode_add(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    let word = ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0b011_0011;
    word.to_le_bytes()
}

fn encode_sub(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    // funct7=0b0100000 occupies bits 25..32; bit 30 (funct7 bit 5) is set.
    let word = (0b010_0000u32 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | ((rd as u32) << 7)
        | 0b011_0011;
    word.to_le_bytes()
}

fn encode_xori(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
    let word = ((imm as u32 & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (0b100 << 12)
        | ((rd as u32) << 7)
        | 0b001_0011;
    word.to_le_bytes()
}

fn encode_ori(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
    let word = ((imm as u32 & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (0b110 << 12)
        | ((rd as u32) << 7)
        | 0b001_0011;
    word.to_le_bytes()
}

fn encode_andi(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
    let word = ((imm as u32 & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (0b111 << 12)
        | ((rd as u32) << 7)
        | 0b001_0011;
    word.to_le_bytes()
}

fn encode_xor(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
    let word = ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (0b100 << 12)
        | ((rd as u32) << 7)
        | 0b011_0011;
    word.to_le_bytes()
}

/// Prove and verify a set of (possibly modified) traces.
/// Returns `true` if verification succeeds, `false` otherwise.
fn prove_and_verify(traces: AllTraces) -> bool {
    let (airs, trace_vecs) = traces.into_vecs();
    let config = build_config();
    let (proof, common) = do_prove(&config, &airs, &trace_vecs);
    let pvs = vec![vec![]; airs.len()];
    verify_batch(&config, &airs, &proof, &pvs, &common).is_ok()
}

// ── Positive tests ────────────────────────────────────────────────────────────

/// Single ADDI: x1 = x0 + 1.  Exercises the bytes and trace buses.
#[test]
fn prove_single_addi() {
    let program = encode_addi(1, 0, 1).to_vec();
    prove(&program);
}

/// Single XORI: x1 = x0 ^ 0xFF.  Exercises the bytes_xor bus.
#[test]
fn prove_single_xori() {
    let program = encode_xori(1, 0, 0xFF).to_vec();
    prove(&program);
}

/// Single ADD: x3 = x1 + x2. Exercises the register-register addition path.
#[test]
fn prove_single_add() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
    program.extend_from_slice(&encode_addi(2, 0, 7)); // x2 = 7
    program.extend_from_slice(&encode_add(3, 1, 2)); // x3 = 17
    prove(&program);
}

/// ADD wrapping overflow: 0xFFFF_FFFF + 1 = 0.
#[test]
fn prove_add_wrapping() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
    program.extend_from_slice(&encode_addi(2, 0, 1)); // x2 = 1
    program.extend_from_slice(&encode_add(3, 1, 2)); // x3 = 0 (wraps)
    prove(&program);
}

/// Mixed ADDI + ADD program exercising both I-type and R-type paths together.
#[test]
fn prove_mixed_addi_add() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
    program.extend_from_slice(&encode_addi(2, 0, 3)); // x2 = 3
    program.extend_from_slice(&encode_add(3, 1, 2)); // x3 = 8
    program.extend_from_slice(&encode_add(4, 3, 1)); // x4 = 13
    program.extend_from_slice(&encode_addi(5, 4, -1)); // x5 = 12
    prove(&program);
}

/// Multiple ADDI steps touching different registers; exercises the u32_lt
/// and timestamp_lt buses through the memory sort.
#[test]
fn prove_addi_chain() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
    program.extend_from_slice(&encode_addi(2, 1, 3)); // x2 = 8
    program.extend_from_slice(&encode_addi(3, 2, -1)); // x3 = 7
    program.extend_from_slice(&encode_addi(3, 2, -1)); // x3 = 7
    prove(&program);
}

/// Mixed ADDI + XORI program, identical to the guest test.s fixture.
#[test]
fn prove_mixed_addi_xori() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
    program.extend_from_slice(&encode_addi(2, 0, 0)); // x2 = 0
    program.extend_from_slice(&encode_addi(3, 0, 1)); // x3 = 1
    program.extend_from_slice(&encode_xori(4, 3, -1i16)); // x4 = x3 ^ 0xFFFF_FFFF
    program.extend_from_slice(&encode_xori(5, 2, 0x55)); // x5 = x2 ^ 0x55
    prove(&program);
}

/// Overwriting the same register twice; checks that the memory AIR handles
/// multiple writes at the same address in sorted order.
#[test]
fn prove_overwrite_register() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
    program.extend_from_slice(&encode_addi(1, 0, 20)); // x1 = 20
    program.extend_from_slice(&encode_addi(1, 1, 5)); // x1 = 25
    prove(&program);
}

/// Confirm that generate_traces + prove_traces round-trips correctly.
#[test]
fn generate_then_prove() {
    let program = encode_addi(1, 0, 42).to_vec();
    let traces = generate_traces(&program);
    prove_traces(traces);
}

// ── Negative tests ────────────────────────────────────────────────────────────
//
// Each test corrupts one field in an otherwise-valid witness and asserts
// that the resulting proof fails verification.
//
// BoundaryColumns layout (5 cols): [pc0, pc1, pc2, pc3, timestamp]
// MemoryColumns layout (17 cols):
//   [memory_type, addr0..3, timestamp, read0..3, write0..3,
//    is_memory_type_equal, is_timestamp_equal, is_address_equal]
// AddiColumns layout (40 cols):
//   [pc0..3, timestamp, rd, rs1, imm, imm_high_bits0..3, imm_se_bytes0..3,
//    rs1_value0..3, old_rd_value0..3, rd_new_value0..3, add_carries0..3,
//    next_pc0..3, next_pc_carries0..2, is_dummy]

/// Corrupt the initial PC in row 0 of the boundaries trace.
///
/// The boundaries AIR sends (pc, ts) = (0, 0) from row 0 onto the bus.
/// Setting pc[0] = 1 unbalances the bus: the CPU expects an initial pc of 0.
#[test]
fn negative_boundaries_initial_pc() {
    let program = encode_addi(1, 0, 1).to_vec();
    let mut traces = generate_traces(&program);
    // Row 0, col 0 = initial PC byte 0, must be 0.
    traces.boundaries.values[0] = Val::ONE;
    assert!(
        !prove_and_verify(traces),
        "corrupted initial PC should fail verification"
    );
}

/// Corrupt rd_new_value[0] in the ADDI trace.
///
/// The ADDI AIR constrains rs1_value + sign_extend(imm) == rd_new_value
/// byte-by-byte. Incrementing the low byte of rd_new_value breaks this.
#[test]
fn negative_addi_arithmetic() {
    let program = encode_addi(1, 0, 1).to_vec();
    let mut traces = generate_traces(&program);
    let addi = traces.addi.as_mut().expect("program has ADDI");
    // rd_new_value[0] is at column index 24 (see AddiColumns layout above).
    addi.values[24] += Val::ONE;
    assert!(
        !prove_and_verify(traces),
        "corrupted ADDI result should fail verification"
    );
}

/// Corrupt write[0] in the first row of the memory trace.
///
/// The memory AIR exposes (address, value) pairs on the memory bus.
/// Changing a stored write value breaks the balance with what the CPU sent.
#[test]
fn negative_memory_write_value() {
    let program = encode_addi(1, 0, 5).to_vec();
    let mut traces = generate_traces(&program);
    // write[0] is at column index 10 (see MemoryColumns layout above).
    traces.memory.values[10] += Val::ONE;
    assert!(
        !prove_and_verify(traces),
        "corrupted memory write should fail verification"
    );
}

/// Single SUB: x3 = x1 - x2. Exercises the register-register subtraction path.
#[test]
fn prove_single_sub() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
    program.extend_from_slice(&encode_addi(2, 0, 3)); // x2 = 3
    program.extend_from_slice(&encode_sub(3, 1, 2)); // x3 = 7
    prove(&program);
}

/// SUB wrapping underflow: 0 - 1 = 0xFFFF_FFFF.
#[test]
fn prove_sub_wrapping() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0)); // x1 = 0
    program.extend_from_slice(&encode_addi(2, 0, 1)); // x2 = 1
    program.extend_from_slice(&encode_sub(3, 1, 2)); // x3 = 0xFFFF_FFFF (wraps)
    prove(&program);
}

/// Mixed ADD + SUB: exercises both R-type instructions together.
#[test]
fn prove_mixed_add_sub() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 20)); // x1 = 20
    program.extend_from_slice(&encode_addi(2, 0, 7)); // x2 = 7
    program.extend_from_slice(&encode_add(3, 1, 2)); // x3 = 27
    program.extend_from_slice(&encode_sub(4, 3, 2)); // x4 = 20
    prove(&program);
}

/// Single XOR: x3 = x1 ^ x2. Exercises the register-register XOR path.
#[test]
fn prove_single_xor() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0b1010)); // x1 = 0b1010
    program.extend_from_slice(&encode_addi(2, 0, 0b1100)); // x2 = 0b1100
    program.extend_from_slice(&encode_xor(3, 1, 2)); // x3 = 0b0110
    prove(&program);
}

/// XOR of a register with itself produces zero: x ^ x == 0.
#[test]
fn prove_xor_self_is_zero() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0x7FF)); // x1 = 0x7FF
    program.extend_from_slice(&encode_xor(2, 1, 1)); // x2 = x1 ^ x1 = 0
    prove(&program);
}

/// XOR with all-ones (from XORI -1) produces bitwise complement.
#[test]
fn prove_xor_complement() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 42)); // x1 = 42
    program.extend_from_slice(&encode_addi(2, 0, -1i16)); // x2 = 0xFFFF_FFFF
    program.extend_from_slice(&encode_xor(3, 1, 2)); // x3 = ~42 = 0xFFFF_FFD5
    prove(&program);
}

/// Mixed XOR and ADD: exercises both R-type instruction paths together.
#[test]
fn prove_mixed_xor_add() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0xFF)); // x1 = 0xFF
    program.extend_from_slice(&encode_addi(2, 0, 0x0F)); // x2 = 0x0F
    program.extend_from_slice(&encode_xor(3, 1, 2)); // x3 = 0xF0
    program.extend_from_slice(&encode_add(4, 3, 1)); // x4 = 0xF0 + 0xFF = 0x1EF
    prove(&program);
}

/// Single ORI: x2 = x1 | 0xFF. Exercises the bytes_or bus.
#[test]
fn prove_single_ori() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
    program.extend_from_slice(&encode_ori(2, 1, 0xFF)); // x2 = 5 | 0xFF = 0xFF
    prove(&program);
}

/// ORI with a negative (sign-extended) immediate: x2 = x1 | 0xFFFF_FFFF.
#[test]
fn prove_ori_negative_immediate() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 42)); // x1 = 42
    program.extend_from_slice(&encode_ori(2, 1, -1i16)); // x2 = 42 | 0xFFFF_FFFF = 0xFFFF_FFFF
    prove(&program);
}

/// Mixed ADDI + ORI program.
#[test]
fn prove_mixed_addi_ori() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0b1010)); // x1 = 0b1010
    program.extend_from_slice(&encode_ori(2, 1, 0b0101)); // x2 = 0b1010 | 0b0101 = 0b1111
    program.extend_from_slice(&encode_ori(3, 2, 0x00)); // x3 = 0b1111 | 0 = 0b1111
    prove(&program);
}

/// Single ANDI: x2 = x1 & 0x0F. Exercises the bytes_and bus.
#[test]
fn prove_single_andi() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0xFF)); // x1 = 0xFF
    program.extend_from_slice(&encode_andi(2, 1, 0x0F)); // x2 = 0xFF & 0x0F = 0x0F
    prove(&program);
}

/// ANDI masking pattern: AND with 0xFF extracts the low byte.
#[test]
fn prove_andi_byte_mask() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
    program.extend_from_slice(&encode_andi(2, 1, 0xFF)); // x2 = 0xFFFF_FFFF & 0xFF = 0xFF
    prove(&program);
}

/// ANDI with a negative (sign-extended) immediate: x2 = x1 & 0xFFFF_FFFF.
#[test]
fn prove_andi_negative_immediate() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 42)); // x1 = 42
    program.extend_from_slice(&encode_andi(2, 1, -1i16)); // x2 = 42 & 0xFFFF_FFFF = 42
    prove(&program);
}

/// Mixed ADDI + ANDI program.
#[test]
fn prove_mixed_addi_andi() {
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 0b1111)); // x1 = 0b1111
    program.extend_from_slice(&encode_andi(2, 1, 0b1010)); // x2 = 0b1111 & 0b1010 = 0b1010
    program.extend_from_slice(&encode_andi(3, 2, 0b0100)); // x3 = 0b1010 & 0b0100 = 0b0000
    prove(&program);
}

/// Set `is_address_equal` to a non-boolean value (2) in a memory row.
///
/// The memory AIR asserts this column is boolean.  A value of 2 violates
/// that constraint directly, regardless of the actual addresses.
#[test]
fn negative_nonboolean_is_address_equal() {
    // Two writes to the same register guarantee is_address_equal == 1 somewhere.
    let mut program = Vec::new();
    program.extend_from_slice(&encode_addi(1, 0, 5));
    program.extend_from_slice(&encode_addi(1, 1, 3));
    let mut traces = generate_traces(&program);
    let width = traces.memory.width; // == 17
                                     // is_address_equal is at column index 16 within each row.
    let col = 16;
    let mut found = false;
    for row in 0..traces.memory.values.len() / width {
        let idx = row * width + col;
        if traces.memory.values[idx] == Val::ONE {
            // Replace the valid flag with a non-boolean field element.
            traces.memory.values[idx] = Val::from_u64(2);
            found = true;
            break;
        }
    }
    assert!(
        found,
        "expected at least one row with is_address_equal == 1"
    );
    assert!(
        !prove_and_verify(traces),
        "non-boolean is_address_equal should fail verification"
    );
}
