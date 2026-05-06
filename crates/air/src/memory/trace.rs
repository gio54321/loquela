use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::DenseMatrix;
use punctum_vm::{MemoryOperation, MemoryType};

use super::air::{MemoryColumns, NUM_COLS};

struct MemoryOp {
    memory_type: MemoryType,
    address: u32,
    timestamp: u32,
    read: u32,
    write: u32,
}

impl From<&MemoryOperation> for MemoryOp {
    fn from(op: &MemoryOperation) -> Self {
        match *op {
            MemoryOperation::Read { memory_type, address, timestamp, value } => Self {
                memory_type,
                address,
                timestamp,
                read: value,
                write: value,
            },
            MemoryOperation::Write { memory_type, address, timestamp, old_value, new_value } => Self {
                memory_type,
                address,
                timestamp,
                read: old_value,
                write: new_value,
            },
        }
    }
}

fn u32_to_limbs<F: PrimeCharacteristicRing>(v: u32) -> [F; 4] {
    let b = v.to_le_bytes();
    [
        F::from_u64(b[0] as u64),
        F::from_u64(b[1] as u64),
        F::from_u64(b[2] as u64),
        F::from_u64(b[3] as u64),
    ]
}

/// Build the memory-argument trace matrix from raw VM operations.
///
/// `vm_ops` is the flat list of all memory operations in execution order.
/// They are converted, sorted by `(memory_type, address, timestamp)`, and
/// written into the matrix.  Transition equality flags are derived
/// automatically from consecutive sorted rows.
pub fn build_trace<F: PrimeCharacteristicRing + Send + Sync>(
    vm_ops: &[MemoryOperation],
) -> DenseMatrix<F> {
    let mut ops: Vec<MemoryOp> = vm_ops.iter().map(MemoryOp::from).collect();
    ops.sort_by_key(|op| (op.memory_type as u8, op.address, op.timestamp));

    let num_ops = ops.len();
    assert!(num_ops > 0, "memory trace must have at least one operation");
    let num_rows = num_ops.next_power_of_two();

    let mut values = vec![F::ZERO; num_rows * NUM_COLS];

    let (prefix, rows, suffix) = unsafe { values.align_to_mut::<MemoryColumns<F>>() };
    assert!(prefix.is_empty(), "alignment mismatch");
    assert!(suffix.is_empty(), "alignment mismatch");
    assert_eq!(rows.len(), num_rows);

    for (i, op) in ops.iter().enumerate() {
        let row = &mut rows[i];

        row.memory_type = F::from_u64(op.memory_type as u64); // Register=0, Ram=1
        row.address = u32_to_limbs(op.address);
        row.timestamp = F::from_u64(op.timestamp as u64);
        row.read = u32_to_limbs(op.read);
        row.write = u32_to_limbs(op.write);

        if let Some(next) = ops.get(i + 1) {
            row.is_memory_type_equal = F::from_bool(op.memory_type as u8 == next.memory_type as u8);
            row.is_address_equal = F::from_bool(op.address == next.address);
            row.is_timestamp_equal = F::from_bool(op.timestamp == next.timestamp);
        }
        // Last real row and padding rows keep F::ZERO for all equality flags.
    }

    DenseMatrix::new(values, NUM_COLS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_mersenne_31::Mersenne31;
    use punctum_vm::{MemoryOperation, MemoryType, VM};
    use std::borrow::Borrow;

    type F = Mersenne31;

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    fn limbs(v: u32) -> [F; 4] {
        u32_to_limbs(v)
    }

    fn row(trace: &DenseMatrix<F>, i: usize) -> &MemoryColumns<F> {
        let w = trace.width;
        trace.values[i * w..(i + 1) * w].borrow()
    }

    fn encode_addi(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
        let word = ((imm as u32 & 0xFFF) << 20)
            | ((rs1 as u32) << 15)
            | ((rd as u32) << 7)
            | 0b001_0011;
        word.to_le_bytes()
    }

    // Ops given in random order must appear sorted by (memory_type, address, timestamp).
    #[test]
    fn ops_are_sorted() {
        let ops = vec![
            MemoryOperation::Read  { memory_type: MemoryType::Ram,      address: 10, timestamp: 5, value: 99 },
            MemoryOperation::Read  { memory_type: MemoryType::Register, address:  5, timestamp: 3, value:  7 },
            MemoryOperation::Write { memory_type: MemoryType::Register, address:  5, timestamp: 1, old_value: 0, new_value: 7 },
            MemoryOperation::Read  { memory_type: MemoryType::Ram,      address: 10, timestamp: 2, value: 42 },
        ];
        let trace: DenseMatrix<F> = build_trace(&ops);

        // Expected order: (0,5,1), (0,5,3), (1,10,2), (1,10,5)
        assert_eq!(row(&trace, 0).memory_type, f(0)); assert_eq!(row(&trace, 0).address, limbs(5));  assert_eq!(row(&trace, 0).timestamp, f(1));
        assert_eq!(row(&trace, 1).memory_type, f(0)); assert_eq!(row(&trace, 1).address, limbs(5));  assert_eq!(row(&trace, 1).timestamp, f(3));
        assert_eq!(row(&trace, 2).memory_type, f(1)); assert_eq!(row(&trace, 2).address, limbs(10)); assert_eq!(row(&trace, 2).timestamp, f(2));
        assert_eq!(row(&trace, 3).memory_type, f(1)); assert_eq!(row(&trace, 3).address, limbs(10)); assert_eq!(row(&trace, 3).timestamp, f(5));
    }

    // is_memory_type_equal / is_address_equal / is_timestamp_equal are set correctly
    // for each consecutive pair, and cleared on the last real row.
    #[test]
    fn transition_flags() {
        let ops = vec![
            // (Register, addr=1, ts=0) -> (Register, addr=1, ts=2): same type, same addr, diff ts
            MemoryOperation::Write { memory_type: MemoryType::Register, address: 1,   timestamp: 0, old_value: 0, new_value: 5 },
            MemoryOperation::Read  { memory_type: MemoryType::Register, address: 1,   timestamp: 2, value: 5 },
            // (Register, addr=1, ts=2) -> (Register, addr=3, ts=4): same type, diff addr
            MemoryOperation::Write { memory_type: MemoryType::Register, address: 3,   timestamp: 4, old_value: 0, new_value: 9 },
            // (Register, addr=3, ts=4) -> (Ram, addr=100, ts=6): diff type
            MemoryOperation::Read  { memory_type: MemoryType::Ram,      address: 100, timestamp: 6, value: 0 },
        ];
        let trace: DenseMatrix<F> = build_trace(&ops);

        let r0 = row(&trace, 0);
        assert_eq!(r0.is_memory_type_equal, F::ONE);
        assert_eq!(r0.is_address_equal,     F::ONE);
        assert_eq!(r0.is_timestamp_equal,   F::ZERO);

        let r1 = row(&trace, 1);
        assert_eq!(r1.is_memory_type_equal, F::ONE);
        assert_eq!(r1.is_address_equal,     F::ZERO);
        assert_eq!(r1.is_timestamp_equal,   F::ZERO);

        let r2 = row(&trace, 2);
        assert_eq!(r2.is_memory_type_equal, F::ZERO);
        assert_eq!(r2.is_address_equal,     F::ZERO);
        assert_eq!(r2.is_timestamp_equal,   F::ZERO);

        // Last real row: all flags zero
        let r3 = row(&trace, 3);
        assert_eq!(r3.is_memory_type_equal, F::ZERO);
        assert_eq!(r3.is_address_equal,     F::ZERO);
        assert_eq!(r3.is_timestamp_equal,   F::ZERO);
    }

    // A Read maps to read==write.
    #[test]
    fn read_op_has_same_read_write() {
        let ops = vec![MemoryOperation::Read { memory_type: MemoryType::Register, address: 2, timestamp: 0, value: 42 }];
        let trace: DenseMatrix<F> = build_trace(&ops);
        let r = row(&trace, 0);
        assert_eq!(r.read,  limbs(42));
        assert_eq!(r.write, limbs(42));
    }

    // A Write maps old_value→read, new_value→write.
    #[test]
    fn write_op_splits_old_and_new() {
        let ops = vec![MemoryOperation::Write { memory_type: MemoryType::Register, address: 2, timestamp: 0, old_value: 7, new_value: 42 }];
        let trace: DenseMatrix<F> = build_trace(&ops);
        let r = row(&trace, 0);
        assert_eq!(r.read,  limbs(7));
        assert_eq!(r.write, limbs(42));
    }

    // 3 ops are padded to 4 rows; the padding row is all-zero.
    #[test]
    fn padding_rows_are_zero() {
        let ops = vec![
            MemoryOperation::Read  { memory_type: MemoryType::Register, address: 0, timestamp: 0, value: 0 },
            MemoryOperation::Write { memory_type: MemoryType::Register, address: 1, timestamp: 1, old_value: 0, new_value: 5 },
            MemoryOperation::Read  { memory_type: MemoryType::Register, address: 1, timestamp: 2, value: 5 },
        ];
        let trace: DenseMatrix<F> = build_trace(&ops);
        assert_eq!(trace.values.len() / trace.width, 4);

        let pad = row(&trace, 3);
        assert_eq!(pad.memory_type,          F::ZERO);
        assert_eq!(pad.address,              [F::ZERO; 4]);
        assert_eq!(pad.timestamp,            F::ZERO);
        assert_eq!(pad.read,                 [F::ZERO; 4]);
        assert_eq!(pad.write,                [F::ZERO; 4]);
        assert_eq!(pad.is_memory_type_equal, F::ZERO);
        assert_eq!(pad.is_address_equal,     F::ZERO);
        assert_eq!(pad.is_timestamp_equal,   F::ZERO);
    }

    // Run the VM, build the trace, and verify key cells end-to-end.
    #[test]
    fn end_to_end_two_addi() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
        program.extend_from_slice(&encode_addi(2, 1, 3)); // x2 = 8

        let mut vm = VM::new(program);
        vm.run().unwrap();

        let owned: Vec<MemoryOperation> = vm.get_memory_ops().into_iter().cloned().collect();
        let trace: DenseMatrix<F> = build_trace(&owned);

        // 4 ops, all type=0; sorted by (addr, ts):
        //  row 0: addr=0, ts=0, read x0=0         → read=0,  write=0
        //  row 1: addr=1, ts=1, write x1 0→5      → read=0,  write=5
        //  row 2: addr=1, ts=2, read x1=5          → read=5,  write=5
        //  row 3: addr=2, ts=3, write x2 0→8      → read=0,  write=8
        assert_eq!(trace.values.len() / trace.width, 4);

        assert_eq!(row(&trace, 0).address, limbs(0)); assert_eq!(row(&trace, 0).read, limbs(0)); assert_eq!(row(&trace, 0).write, limbs(0));
        assert_eq!(row(&trace, 1).address, limbs(1)); assert_eq!(row(&trace, 1).read, limbs(0)); assert_eq!(row(&trace, 1).write, limbs(5));
        assert_eq!(row(&trace, 2).address, limbs(1)); assert_eq!(row(&trace, 2).read, limbs(5)); assert_eq!(row(&trace, 2).write, limbs(5));
        assert_eq!(row(&trace, 3).address, limbs(2)); assert_eq!(row(&trace, 3).read, limbs(0)); assert_eq!(row(&trace, 3).write, limbs(8));

        // Transition flags for the sorted sequence
        assert_eq!(row(&trace, 0).is_memory_type_equal, F::ONE);  // same type all the way
        assert_eq!(row(&trace, 0).is_address_equal,     F::ZERO); // addr 0 → addr 1
        assert_eq!(row(&trace, 1).is_address_equal,     F::ONE);  // addr 1 → addr 1
        assert_eq!(row(&trace, 2).is_address_equal,     F::ZERO); // addr 1 → addr 2
    }
}

