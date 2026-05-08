use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Register = 0,
    Ram = 1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    AddI { rd: u8, rs1: u8, imm: i16 },
    XorI { rd: u8, rs1: u8, imm: i16 },
    OriI { rd: u8, rs1: u8, imm: i16 },
    AndiI { rd: u8, rs1: u8, imm: i16 },
    SlliI { rd: u8, rs1: u8, imm: i32 },
    SrliI { rd: u8, rs1: u8, imm: i32 },
    SraiI { rd: u8, rs1: u8, imm: i32 },
    Add { rd: u8, rs1: u8, rs2: u8 },
    Sub { rd: u8, rs1: u8, rs2: u8 },
    Xor { rd: u8, rs1: u8, rs2: u8 },
    Or { rd: u8, rs1: u8, rs2: u8 },
    And { rd: u8, rs1: u8, rs2: u8 },
    Sll { rd: u8, rs1: u8, rs2: u8 },
    Srl { rd: u8, rs1: u8, rs2: u8 },
    Sra { rd: u8, rs1: u8, rs2: u8 },
    Slt { rd: u8, rs1: u8, rs2: u8 },
    Sltu { rd: u8, rs1: u8, rs2: u8 },
    SltiI { rd: u8, rs1: u8, imm: i32 },
    SltiuI { rd: u8, rs1: u8, imm: i32 },
    Lui { rd: u8, imm: i32 },
    Auipc { rd: u8, imm: i32 },
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

/// One executed step: the VM state after execution, the raw instruction word,
/// the decoded instruction, and all memory operations emitted during the step.
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Register state after execution; PC is the address of this instruction.
    pub state: VMState,
    /// Raw 32-bit instruction word fetched from the program ROM.
    pub instruction_word: u32,
    /// Decoded instruction (same information, typed).
    pub instruction: Instruction,
    /// Memory operations emitted in program order during this step.
    pub memory_ops: Vec<MemoryOperation>,
}

pub struct VM {
    pub program: Vec<u8>,
    pub memory: HashMap<u32, u32>,
    pub pc: u32,
    pub timestamp: u32,
    pub trace: Vec<ExecutionStep>,
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
        let mut registers = self
            .trace
            .last()
            .map(|s| s.state.registers)
            .unwrap_or([0u32; 32]);

        let pc = self.pc;
        let off = pc as usize;
        let instruction_word = u32::from_le_bytes([
            self.program[off],
            self.program[off + 1],
            self.program[off + 2],
            self.program[off + 3],
        ]);
        let instruction = Self::decode_instruction(instruction_word);

        let mut ops = Vec::new();
        match &instruction {
            Instruction::AddI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val.wrapping_add(*imm as u32);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::XorI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val ^ (*imm as u32);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::OriI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val | (*imm as u32);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::SlliI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let shamt = (*imm & 0x1F) as u32;
                let result = rs1_val.wrapping_shl(shamt);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::AndiI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val & (*imm as u32);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::SrliI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let shamt = (*imm & 0x1F) as u32;
                let result = rs1_val.wrapping_shr(shamt);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::SraiI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let shamt = (*imm & 0x1F) as u32;
                let result = ((rs1_val as i32) >> shamt) as u32;

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Add { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val.wrapping_add(rs2_val);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Sub { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val.wrapping_sub(rs2_val);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Xor { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val ^ rs2_val;

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Or { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val | rs2_val;

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::And { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = rs1_val & rs2_val;

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Sll { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let shamt = rs2_val & 0x1F;
                let result = rs1_val.wrapping_shl(shamt);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Srl { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let shamt = rs2_val & 0x1F;
                let result = rs1_val.wrapping_shr(shamt);

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Sra { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let shamt = rs2_val & 0x1F;
                let result = ((rs1_val as i32) >> shamt) as u32;

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Slt { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = if (rs1_val as i32) < (rs2_val as i32) {
                    1u32
                } else {
                    0u32
                };

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Sltu { rd, rs1, rs2 } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];
                let old_rd = registers[*rd as usize];
                let result = if rs1_val < rs2_val { 1u32 } else { 0u32 };

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::SltiI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let result = if (rs1_val as i32) < *imm { 1u32 } else { 0u32 };

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::SltiuI { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                // imm is sign-extended; comparison is unsigned
                let imm_u32 = *imm as u32;
                let result = if rs1_val < imm_u32 { 1u32 } else { 0u32 };

                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Lui { rd, imm } => {
                let old_rd = registers[*rd as usize];
                // imm is the raw upper-20 bits (inst >> 12); shift left 12 to get value
                let result = (*imm as u32) << 12;

                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
            Instruction::Auipc { rd, imm } => {
                let old_rd = registers[*rd as usize];
                // imm is the raw upper-20 bits (inst >> 12); shift left 12 and add PC
                let imm_u = (*imm as u32) << 12;
                let result = pc.wrapping_add(imm_u);

                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: result,
                });
                self.timestamp += 1;

                registers[*rd as usize] = result;
            }
        }

        self.trace.push(ExecutionStep {
            state: VMState { pc, registers },
            instruction_word,
            instruction,
            memory_ops: ops,
        });
        self.pc += 4;
        Ok(())
    }

    /// Return all memory operations in execution order, across all steps.
    pub fn get_memory_ops(&self) -> Vec<&MemoryOperation> {
        self.trace
            .iter()
            .flat_map(|s| s.memory_ops.iter())
            .collect()
    }

    /// Return the VMState snapshot (post-execution) for each executed step.
    pub fn get_trace(&self) -> Vec<VMState> {
        self.trace.iter().map(|s| s.state.clone()).collect()
    }

    pub fn decode_instruction(bytes: u32) -> Instruction {
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
        } else if bytes & 0b1111111 == 0b0010011 && (bytes >> 12) & 0b111 == 0b110 {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes as i32) >> 20) as i16;
            Instruction::OriI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0010011 && (bytes >> 12) & 0b111 == 0b111 {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes as i32) >> 20) as i16;
            Instruction::AndiI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0010011
            && (bytes >> 12) & 0b111 == 0b001
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes >> 20) & 0x1F) as i32;
            Instruction::SlliI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0010011
            && (bytes >> 12) & 0b111 == 0b101
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes >> 20) & 0x1F) as i32;
            Instruction::SrliI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0010011
            && (bytes >> 12) & 0b111 == 0b101
            && (bytes >> 25) == 0b0100000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = ((bytes >> 20) & 0x1F) as i32;
            Instruction::SraiI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b000
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Add { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b000
            && (bytes >> 25) == 0b0100000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Sub { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b100
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Xor { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b110
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Or { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b111
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::And { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b001
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Sll { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b101
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Srl { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b101
            && (bytes >> 25) == 0b0100000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Sra { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b010
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Slt { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0110011
            && (bytes >> 12) & 0b111 == 0b011
            && (bytes >> 25) == 0b0000000
        {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            Instruction::Sltu { rd, rs1, rs2 }
        } else if bytes & 0b1111111 == 0b0010011 && (bytes >> 12) & 0b111 == 0b010 {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::SltiI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0010011 && (bytes >> 12) & 0b111 == 0b011 {
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::SltiuI { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0110111 {
            // LUI: U-type, opcode=0x37
            let rd = ((bytes >> 7) & 0b11111) as u8;
            // imm is the raw upper-20 bits (bits 31:12), NOT yet shifted
            let imm = ((bytes as i32) >> 12) & 0xF_FFFF;
            Instruction::Lui { rd, imm }
        } else if bytes & 0b1111111 == 0b0010111 {
            // AUIPC: U-type, opcode=0x17
            let rd = ((bytes >> 7) & 0b11111) as u8;
            // imm is the raw upper-20 bits (bits 31:12), NOT yet shifted
            let imm = ((bytes as i32) >> 12) & 0xF_FFFF;
            Instruction::Auipc { rd, imm }
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

    fn encode_add(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
        let word = ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0b011_0011;
        word.to_le_bytes()
    }

    fn encode_sub(rd: u8, rs1: u8, rs2: u8) -> [u8; 4] {
        let word = (0b010_0000u32 << 25)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | ((rd as u32) << 7)
            | 0b011_0011;
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

    // add x3, x1, x2 → read x1 (ts=0), read x2 (ts=1), write x3 (ts=2)
    #[test]
    fn add_logs_two_reads_then_write() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
        program.extend_from_slice(&encode_addi(2, 0, 7)); // x2 = 7
        program.extend_from_slice(&encode_add(3, 1, 2)); // x3 = x1 + x2 = 17
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        // 2 ops for addi x1, 2 ops for addi x2, 3 ops for add x3
        assert_eq!(ops.len(), 7);

        assert!(matches!(
            ops[4],
            MemoryOperation::Read {
                address: 1,
                timestamp: 4,
                value: 10,
                ..
            }
        ));
        assert!(matches!(
            ops[5],
            MemoryOperation::Read {
                address: 2,
                timestamp: 5,
                value: 7,
                ..
            }
        ));
        assert!(matches!(
            ops[6],
            MemoryOperation::Write {
                address: 3,
                timestamp: 6,
                old_value: 0,
                new_value: 17,
                ..
            }
        ));
    }

    // sub x3, x1, x2 → read x1 (ts=0), read x2 (ts=1), write x3 (ts=2)
    #[test]
    fn sub_logs_two_reads_then_write() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 10)); // x1 = 10
        program.extend_from_slice(&encode_addi(2, 0, 3)); // x2 = 3
        program.extend_from_slice(&encode_sub(3, 1, 2)); // x3 = x1 - x2 = 7
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        // 2 ops for addi x1, 2 ops for addi x2, 3 ops for sub x3
        assert_eq!(ops.len(), 7);

        assert!(matches!(
            ops[4],
            MemoryOperation::Read {
                address: 1,
                timestamp: 4,
                value: 10,
                ..
            }
        ));
        assert!(matches!(
            ops[5],
            MemoryOperation::Read {
                address: 2,
                timestamp: 5,
                value: 3,
                ..
            }
        ));
        assert!(matches!(
            ops[6],
            MemoryOperation::Write {
                address: 3,
                timestamp: 6,
                old_value: 0,
                new_value: 7,
                ..
            }
        ));
    }

    #[test]
    fn sub_wrapping_underflow() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 0)); // x1 = 0
        program.extend_from_slice(&encode_addi(2, 0, 1)); // x2 = 1
        program.extend_from_slice(&encode_sub(3, 1, 2)); // x3 = 0 - 1 = 0xFFFF_FFFF (wraps)
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[3], u32::MAX);
    }

    #[test]
    fn add_wrapping_overflow() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
        program.extend_from_slice(&encode_addi(2, 0, 1)); // x2 = 1
        program.extend_from_slice(&encode_add(3, 1, 2)); // x3 = 0 (wraps)
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[3], 0);
    }

    // xor x3, x1, x2 → read x1 (ts=0), read x2 (ts=1), write x3 (ts=2)
    #[test]
    fn xor_logs_two_reads_then_write() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 0b1010)); // x1 = 0b1010
        program.extend_from_slice(&encode_addi(2, 0, 0b1100)); // x2 = 0b1100
        program.extend_from_slice(&encode_xor(3, 1, 2)); // x3 = 0b0110
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        // 2 ops for addi x1, 2 ops for addi x2, 3 ops for xor x3
        assert_eq!(ops.len(), 7);

        assert!(matches!(
            ops[4],
            MemoryOperation::Read {
                address: 1,
                timestamp: 4,
                value: 0b1010,
                ..
            }
        ));
        assert!(matches!(
            ops[5],
            MemoryOperation::Read {
                address: 2,
                timestamp: 5,
                value: 0b1100,
                ..
            }
        ));
        assert!(matches!(
            ops[6],
            MemoryOperation::Write {
                address: 3,
                timestamp: 6,
                old_value: 0,
                new_value: 0b0110,
                ..
            }
        ));
    }

    // ori x2, x1, imm → read x1, write x2 with x1 | sign_extend(imm).
    #[test]
    fn ori_logs_read_then_write() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 5)); // x1 = 5
        program.extend_from_slice(&encode_ori(2, 1, 0xFF)); // x2 = 5 | 0xFF = 0xFF
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
                new_value: 0xFF,
                ..
            }
        ));
    }

    // ori with negative immediate sign-extends to 32 bits.
    #[test]
    fn ori_sign_extended_immediate() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 0)); // x1 = 0
        program.extend_from_slice(&encode_ori(2, 1, -1i16)); // x2 = 0 | 0xFFFF_FFFF = 0xFFFF_FFFF
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[2], u32::MAX);
    }

    // andi x2, x1, imm → read x1, write x2 with x1 & sign_extend(imm).
    #[test]
    fn andi_logs_read_then_write() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 0xFF)); // x1 = 0xFF
        program.extend_from_slice(&encode_andi(2, 1, 0x0F)); // x2 = 0xFF & 0x0F = 0x0F
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let ops = vm.get_memory_ops();
        assert!(matches!(
            ops[2],
            MemoryOperation::Read {
                address: 1,
                timestamp: 2,
                value: 0xFF,
                ..
            }
        ));
        assert!(matches!(
            ops[3],
            MemoryOperation::Write {
                address: 2,
                timestamp: 3,
                old_value: 0,
                new_value: 0x0F,
                ..
            }
        ));
    }

    // andi with negative immediate sign-extends to 32 bits.
    #[test]
    fn andi_sign_extended_immediate() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
        program.extend_from_slice(&encode_andi(2, 1, -1i16)); // x2 = 0xFFFF_FFFF & 0xFFFF_FFFF = 0xFFFF_FFFF
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[2], u32::MAX);
    }

    // andi with 0xFF mask extracts low byte.
    #[test]
    fn andi_mask_low_byte() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, -1i16)); // x1 = 0xFFFF_FFFF
        program.extend_from_slice(&encode_andi(2, 1, 0xFF)); // x2 = 0xFFFF_FFFF & 0xFF = 0xFF
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[2], 0xFF);
    }

    // xor of a register with itself should produce zero.
    #[test]
    fn xor_self_produces_zero() {
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 0x7FF)); // x1 = 0x7FF
        program.extend_from_slice(&encode_xor(2, 1, 1)); // x2 = x1 ^ x1 = 0
        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[2], 0);
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
            vm.step().expect("step failed");

            let step = vm.trace.last().unwrap();
            println!(
                "== step {} == pc 0x{:08x} -> 0x{:08x}  {:08x}  {:?}",
                step_idx, pc_before, vm.pc, step.instruction_word, step.instruction
            );
            print_regs(&step.state.registers);
            step_idx += 1;
        }

        // Sanity-check final register values for the program in test.s:
        //   addi x1, x1, -1  ; addi x2, x2, 0  ; addi x3, x3, 1
        //   xori x4, x3, -1  ; xori x5, x2, 0x55
        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[1], u32::MAX);
        assert_eq!(regs[2], 0);
        assert_eq!(regs[3], 1);
        assert_eq!(regs[4], 0xFFFF_FFFE);
        assert_eq!(regs[5], 0x55);
    }
}
