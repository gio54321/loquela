use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Register = 0,
    Ram = 1,
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    AddI { rd: u8, rs1: u8, imm: i16 },
    XorI { rd: u8, rs1: u8, imm: i16 },
}

#[derive(Debug, Clone)]
pub struct VMState {
    pub pc: u32,
    pub registers: [u32; 32],
}

#[derive(Debug, Clone)]
pub enum MemoryOperation {
    /// A read that does not change the cell's value.
    Read {
        memory_type: MemoryType,
        address: u32,
        timestamp: u32,
        value: u32,
    },
    /// A write that changes the cell's value.
    Write {
        memory_type: MemoryType,
        address: u32,
        timestamp: u32,
        old_value: u32,
        new_value: u32,
    },
}

pub struct VM {
    pub program: Vec<u8>,
    pub memory: HashMap<u32, u32>,
    pub pc: u32,
    pub timestamp: u32,
    pub trace: Vec<(VMState, Vec<MemoryOperation>)>,
}

impl VM {
    pub fn new(program: Vec<u8>) -> Self {
        Self {
            program,
            memory: HashMap::new(),
            pc: 0,
            timestamp: 0,
            trace: Vec::new(),
        }
    }

    pub fn run(&mut self) -> Result<(), String> {
        while (self.pc as usize) < self.program.len() {
            self.step()?;
        }
        Ok(())
    }

    pub fn step(&mut self) -> Result<(), String> {
        // Snapshot current register state for this step's trace entry.
        let current_registers = self
            .trace
            .last()
            .map(|(s, _)| s.registers)
            .unwrap_or([0u32; 32]);
        self.trace
            .push((VMState { pc: self.pc, registers: current_registers }, Vec::new()));

        let pc = self.pc as usize;
        let word = u32::from_le_bytes([
            self.program[pc],
            self.program[pc + 1],
            self.program[pc + 2],
            self.program[pc + 3],
        ]);
        let decoded = Self::decode_instruction(word);
        let mut ops = Vec::new();
        match decoded {
            Instruction::AddI { rd, rs1, imm } => {
                let rs1_val = self.trace.last().unwrap().0.registers[rs1 as usize];
                let old_rd = self.trace.last().unwrap().0.registers[rd as usize];
                let result = rs1_val.wrapping_add(imm as u32);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                self.trace.last_mut().unwrap().0.registers[rd as usize] = result;
            }
            Instruction::XorI { rd, rs1, imm } => {
                let rs1_val = self.trace.last().unwrap().0.registers[rs1 as usize];
                let old_rd = self.trace.last().unwrap().0.registers[rd as usize];
                let result = rs1_val ^ (imm as u32);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                self.trace.last_mut().unwrap().0.registers[rd as usize] = result;
            }
        }
        self.trace.last_mut().unwrap().1.extend(ops);
        self.pc += 4;
        Ok(())
    }

    /// Return all memory operations in execution order, across all steps.
    pub fn get_memory_ops(&self) -> Vec<&MemoryOperation> {
        self.trace.iter().flat_map(|(_, ops)| ops.iter()).collect()
    }

    /// Return the VMState snapshot at the start of each executed step.
    pub fn get_trace(&self) -> Vec<VMState> {
        self.trace.iter().map(|(state, _)| state.clone()).collect()
    }

    fn decode_instruction(bytes: u32) -> Instruction {
        if bytes & 0b1111111 == 0b0010011 && (bytes >> 12) & 0b111 == 0b000 {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes as i32) >> 20) as i16;
            Instruction::AddI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0010011 && (bytes >> 12) & 0b111 == 0b100 {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes as i32) >> 20) as i16;
            Instruction::XorI { rd, rs1, imm }
        } else {
            unimplemented!("not supported");
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn encode_addi(rd: u8, rs1: u8, imm: i16) -> [u8; 4] {
        let word =
            ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0b001_0011;
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

    // addi x1, x0, 5  →  read x0 (val=0, ts=0), write x1 (old=0, new=5, ts=1)
    #[test]
    fn addi_logs_read_then_write() {
        let mut vm = VM::new(encode_addi(1, 0, 5).to_vec());
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        assert_eq!(ops.len(), 2);

        assert!(matches!(
            ops[0],
            MemoryOperation::Read {
                memory_type: MemoryType::Register,
                address: 0,
                timestamp: 0,
                value: 0
            }
        ));
        assert!(matches!(
            ops[1],
            MemoryOperation::Write {
                memory_type: MemoryType::Register,
                address: 1,
                timestamp: 1,
                old_value: 0,
                new_value: 5
            }
        ));
    }

    // Two addi instructions: timestamps must be globally monotone.
    #[test]
    fn timestamps_increment_across_instructions() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 5)); // addi x1, x0, 5
        program.extend_from_slice(&encode_addi(2, 1, 3)); // addi x2, x1, 3
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        assert_eq!(ops.len(), 4);

        assert!(matches!(
            ops[0],
            MemoryOperation::Read {
                address: 0,
                timestamp: 0,
                value: 0,
                ..
            }
        ));
        assert!(matches!(
            ops[1],
            MemoryOperation::Write {
                address: 1,
                timestamp: 1,
                old_value: 0,
                new_value: 5,
                ..
            }
        ));
        assert!(matches!(
            ops[2],
            MemoryOperation::Read {
                address: 1,
                timestamp: 2,
                value: 5,
                ..
            }
        ));
        assert!(matches!(
            ops[3],
            MemoryOperation::Write {
                address: 2,
                timestamp: 3,
                old_value: 0,
                new_value: 8,
                ..
            }
        ));
    }

    // xori uses old register value for read and produces new value for write.
    #[test]
    fn xori_logs_read_then_write() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
        program.extend_from_slice(&encode_xori(2, 1, -1i16)); // x2 = 5 ^ 0xFFFF_FFFF = 0xFFFF_FFFA
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        assert!(matches!(
            ops[2],
            MemoryOperation::Read {
                address: 1,
                timestamp: 2,
                value: 5,
                ..
            }
        ));
        assert!(matches!(
            ops[3],
            MemoryOperation::Write {
                address: 2,
                timestamp: 3,
                old_value: 0,
                new_value: 0xFFFF_FFFA,
                ..
            }
        ));
    }

    // Writing to the same register twice: old_value of second write is new_value of first.
    #[test]
    fn overwrite_same_register_tracks_old_value() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
        program.extend_from_slice(&encode_addi(1, 0, 20)); // x1 = 20, old = 10
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        assert_eq!(ops.len(), 4);
        assert!(matches!(
            ops[3],
            MemoryOperation::Write {
                address: 1,
                old_value: 10,
                new_value: 20,
                ..
            }
        ));
    }

    #[test]
    fn decode_test_bin() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../guest-programs/test.bin");
        let bytes = fs::read(&path).expect("failed to read test.bin");
        assert!(
            bytes.len() % 4 == 0,
            "test.bin length {} is not a multiple of 4",
            bytes.len()
        );

        let program: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for (i, word) in program.iter().enumerate() {
            let word = *word;
            let decoded = std::panic::catch_unwind(|| VM::decode_instruction(word));
            match decoded {
                Ok(instr) => println!("{:04x}: {:08x}  {:?}", i * 4, word, instr),
                Err(_) => println!(
                    "{:04x}: {:08x}  <unsupported opcode 0x{:02x}>",
                    i * 4,
                    word,
                    word & 0x7f
                ),
            }
        }
    }

    fn print_regs(regs: &[u32; 32]) {
        for chunk in 0..4 {
            let mut line = String::new();
            for i in 0..8 {
                let r = chunk * 8 + i;
                line.push_str(&format!(" x{:<2}={:08x}", r, regs[r]));
            }
            println!("    {}", line);
        }
    }

    #[test]
    fn execute_test_bin() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../guest-programs/test.bin");
        let program = fs::read(&path).expect("failed to read test.bin");

        let mut vm = VM::new(program);

        println!("== initial state ==");
        println!("  pc=0x{:08x}", vm.pc);
        print_regs(&[0u32; 32]);

        let mut step_idx = 0;
        while (vm.pc as usize) < vm.program.len() {
            let pc_before = vm.pc;
            let word = u32::from_le_bytes([
                vm.program[pc_before as usize],
                vm.program[pc_before as usize + 1],
                vm.program[pc_before as usize + 2],
                vm.program[pc_before as usize + 3],
            ]);
            let instr = VM::decode_instruction(word);

            vm.step().expect("step failed");

            let state = &vm.trace.last().unwrap().0;
            println!(
                "== step {} == pc 0x{:08x} -> 0x{:08x}  {:08x}  {:?}",
                step_idx, pc_before, vm.pc, word, instr
            );
            print_regs(&state.registers);
            step_idx += 1;
        }

        // Sanity-check final register values for the program in test.s:
        //   addi x1, x1, -1  ; addi x2, x2, 0  ; addi x3, x3, 1
        //   xori x4, x3, -1  ; xori x5, x2, 0x55
        let regs = vm.trace.last().unwrap().0.registers;
        assert_eq!(regs[1], u32::MAX);
        assert_eq!(regs[2], 0);
        assert_eq!(regs[3], 1);
        assert_eq!(regs[4], 0xFFFF_FFFE);
        assert_eq!(regs[5], 0x55);
    }
}
