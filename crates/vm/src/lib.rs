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
    SlliI { rd: u8, rs1: u8, imm: i32 },
    SrliI { rd: u8, rs1: u8, imm: i32 },
    SraiI { rd: u8, rs1: u8, imm: i32 },
    Add { rd: u8, rs1: u8, rs2: u8 },
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
    Jal { rd: u8, imm: i32 },
    Jalr { rd: u8, rs1: u8, imm: i32 },
    Beq { rs1: u8, rs2: u8, imm: i32 },
    Bne { rs1: u8, rs2: u8, imm: i32 },
    Blt { rs1: u8, rs2: u8, imm: i32 },
    Bge { rs1: u8, rs2: u8, imm: i32 },
    Bltu { rs1: u8, rs2: u8, imm: i32 },
    Bgeu { rs1: u8, rs2: u8, imm: i32 },
    Lw { rd: u8, rs1: u8, imm: i32 },
    Lh { rd: u8, rs1: u8, imm: i32 },
    Lb { rd: u8, rs1: u8, imm: i32 },
    Lhu { rd: u8, rs1: u8, imm: i32 },
    Lbu { rd: u8, rs1: u8, imm: i32 },
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
            Instruction::Jal { rd, imm } => {
                let old_rd = registers[*rd as usize];
                let return_addr = if *rd == 0 { 0 } else { pc.wrapping_add(4) };
                let next_pc = pc.wrapping_add(*imm as u32);

                // Write return address to rd (or 0 if rd=x0).
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: return_addr,
                });
                self.timestamp += 1;

                if *rd != 0 {
                    registers[*rd as usize] = return_addr;
                }

                self.trace.push(ExecutionStep {
                    state: VMState { pc, registers },
                    instruction_word,
                    instruction,
                    memory_ops: ops,
                });
                self.pc = next_pc;
                return Ok(());
            }
            Instruction::Jalr { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let return_addr = if *rd == 0 { 0 } else { pc.wrapping_add(4) };
                let next_pc = rs1_val.wrapping_add(*imm as u32) & !1u32;

                // Read rs1.
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;

                // Write return address to rd (or 0 if rd=x0).
                ops.push(MemoryOperation::Write {
                    memory_type: MemoryType::Register,
                    address: *rd as u32,
                    timestamp: self.timestamp,
                    old_value: old_rd,
                    new_value: return_addr,
                });
                self.timestamp += 1;

                if *rd != 0 {
                    registers[*rd as usize] = return_addr;
                }

                self.trace.push(ExecutionStep {
                    state: VMState { pc, registers },
                    instruction_word,
                    instruction,
                    memory_ops: ops,
                });
                self.pc = next_pc;
                return Ok(());
            }
            Instruction::Beq { rs1, rs2, imm }
            | Instruction::Bne { rs1, rs2, imm }
            | Instruction::Blt { rs1, rs2, imm }
            | Instruction::Bge { rs1, rs2, imm }
            | Instruction::Bltu { rs1, rs2, imm }
            | Instruction::Bgeu { rs1, rs2, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let rs2_val = registers[*rs2 as usize];

                let taken = match &instruction {
                    Instruction::Beq { .. } => rs1_val == rs2_val,
                    Instruction::Bne { .. } => rs1_val != rs2_val,
                    Instruction::Blt { .. } => (rs1_val as i32) < (rs2_val as i32),
                    Instruction::Bge { .. } => (rs1_val as i32) >= (rs2_val as i32),
                    Instruction::Bltu { .. } => rs1_val < rs2_val,
                    Instruction::Bgeu { .. } => rs1_val >= rs2_val,
                    _ => unreachable!(),
                };

                let next_pc = if taken {
                    pc.wrapping_add(*imm as u32)
                } else {
                    pc.wrapping_add(4)
                };

                // Read rs1 at timestamp.
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;

                // Read rs2 at timestamp + 1.
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs2 as u32,
                    timestamp: self.timestamp,
                    value: rs2_val,
                });
                self.timestamp += 1;

                self.trace.push(ExecutionStep {
                    state: VMState { pc, registers },
                    instruction_word,
                    instruction,
                    memory_ops: ops,
                });
                self.pc = next_pc;
                return Ok(());
            }
            Instruction::Lw { rd, rs1, imm }
            | Instruction::Lh { rd, rs1, imm }
            | Instruction::Lb { rd, rs1, imm }
            | Instruction::Lhu { rd, rs1, imm }
            | Instruction::Lbu { rd, rs1, imm } => {
                let rs1_val = registers[*rs1 as usize];
                let old_rd = registers[*rd as usize];
                let addr = rs1_val.wrapping_add(*imm as u32);

                // Read from rs1 register.
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Register,
                    address: *rs1 as u32,
                    timestamp: self.timestamp,
                    value: rs1_val,
                });
                self.timestamp += 1;

                // Read from RAM.
                let ram_val = *self.memory.get(&addr).unwrap_or(&0);
                ops.push(MemoryOperation::Read {
                    memory_type: MemoryType::Ram,
                    address: addr,
                    timestamp: self.timestamp,
                    value: ram_val,
                });
                self.timestamp += 1;

                // Compute result based on instruction type.
                let result = match &instruction {
                    Instruction::Lw { .. } => ram_val,
                    Instruction::Lhu { .. } => ram_val & 0xFFFF,
                    Instruction::Lbu { .. } => ram_val & 0xFF,
                    Instruction::Lh { .. } => {
                        let half = (ram_val & 0xFFFF) as u16;
                        (half as i16 as i32) as u32
                    }
                    Instruction::Lb { .. } => {
                        let byte = (ram_val & 0xFF) as u8;
                        (byte as i8 as i32) as u32
                    }
                    _ => unreachable!(),
                };

                // Write result to rd register.
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
        } else if bytes & 0b1111111 == 0b1101111 {
            // JAL: J-type, opcode=0x6F
            let rd = ((bytes >> 7) & 0b11111) as u8;
            // J-type immediate: bits scrambled across the instruction word.
            // imm[20]   = bit 31
            // imm[10:1] = bits 30:21
            // imm[11]   = bit 20
            // imm[19:12]= bits 19:12
            // imm[0]    = 0 (always)
            let imm20 = (bytes >> 31) & 1;
            let imm10_1 = (bytes >> 21) & 0x3FF;
            let imm11 = (bytes >> 20) & 1;
            let imm19_12 = (bytes >> 12) & 0xFF;
            let imm_raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            // Sign-extend from bit 20.
            let imm = ((imm_raw << 11) as i32) >> 11;
            Instruction::Jal { rd, imm }
        } else if bytes & 0b1111111 == 0b1100111 && (bytes >> 12) & 0b111 == 0b000 {
            // JALR: I-type, opcode=0x67, funct3=0x0
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::Jalr { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b1100011 {
            // B-type: opcode=0x63
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let rs2 = ((bytes >> 20) & 0b11111) as u8;
            let funct3 = (bytes >> 12) & 0b111;
            // B-type immediate encoding:
            // imm[12]  = bit 31
            // imm[10:5]= bits 30:25
            // imm[4:1] = bits 11:8
            // imm[11]  = bit 7
            // imm[0]   = 0 (always)
            let imm12 = (bytes >> 31) & 1;
            let imm10_5 = (bytes >> 25) & 0x3F;
            let imm4_1 = (bytes >> 8) & 0xF;
            let imm11 = (bytes >> 7) & 1;
            let imm_raw = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
            // Sign-extend from bit 12.
            let imm = ((imm_raw << 19) as i32) >> 19;
            match funct3 {
                0b000 => Instruction::Beq { rs1, rs2, imm },
                0b001 => Instruction::Bne { rs1, rs2, imm },
                0b100 => Instruction::Blt { rs1, rs2, imm },
                0b101 => Instruction::Bge { rs1, rs2, imm },
                0b110 => Instruction::Bltu { rs1, rs2, imm },
                0b111 => Instruction::Bgeu { rs1, rs2, imm },
                _ => unimplemented!("unsupported B-type funct3"),
            }
        } else if bytes & 0b1111111 == 0b0000011 && (bytes >> 12) & 0b111 == 0b010 {
            // LW: I-type, opcode=0x03, funct3=0x2
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::Lw { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0000011 && (bytes >> 12) & 0b111 == 0b001 {
            // LH: I-type, opcode=0x03, funct3=0x1
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::Lh { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0000011 && (bytes >> 12) & 0b111 == 0b000 {
            // LB: I-type, opcode=0x03, funct3=0x0
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::Lb { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0000011 && (bytes >> 12) & 0b111 == 0b101 {
            // LHU: I-type, opcode=0x03, funct3=0x5
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::Lhu { rd, rs1, imm }
        } else if bytes & 0b1111111 == 0b0000011 && (bytes >> 12) & 0b111 == 0b100 {
            // LBU: I-type, opcode=0x03, funct3=0x4
            let rd = ((bytes >> 7) & 0b11111) as u8;
            let rs1 = ((bytes >> 15) & 0b11111) as u8;
            let imm = (bytes as i32) >> 20;
            Instruction::Lbu { rd, rs1, imm }
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

    fn encode_jal(rd: u8, imm: i32) -> [u8; 4] {
        // J-type encoding:
        // imm[20|10:1|11|19:12] | rd | opcode
        let imm = imm as u32;
        let imm20 = (imm >> 20) & 1;
        let imm10_1 = (imm >> 1) & 0x3FF;
        let imm11 = (imm >> 11) & 1;
        let imm19_12 = (imm >> 12) & 0xFF;
        let word = (imm20 << 31)
            | (imm10_1 << 21)
            | (imm11 << 20)
            | (imm19_12 << 12)
            | ((rd as u32) << 7)
            | 0b110_1111;
        word.to_le_bytes()
    }

    fn encode_jalr(rd: u8, rs1: u8, imm: i32) -> [u8; 4] {
        let word = ((imm as u32 & 0xFFF) << 20)
            | ((rs1 as u32) << 15)
            | (0b000 << 12)
            | ((rd as u32) << 7)
            | 0b110_0111;
        word.to_le_bytes()
    }

    #[test]
    fn jal_jumps_forward() {
        // Program:
        //   pc=0: jal x1, +8  → x1 = 4, jump to pc=8
        //   pc=4: addi x2, x0, 99  (should be skipped)
        //   pc=8: addi x3, x0, 42
        let mut program = Vec::new();
        program.extend_from_slice(&encode_jal(1, 8)); // jal x1, 8
        program.extend_from_slice(&encode_addi(2, 0, 99)); // should be skipped
        program.extend_from_slice(&encode_addi(3, 0, 42));

        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[1], 4, "x1 should be return address pc+4=4");
        assert_eq!(regs[2], 0, "x2 should not be set (skipped)");
        assert_eq!(regs[3], 42, "x3 should be 42");
        assert_eq!(
            vm.trace.len(),
            2,
            "should execute jal + addi, skipping middle"
        );
    }

    #[test]
    fn jal_rd_zero_unconditional_jump() {
        // jal x0, +8 — unconditional jump, return address discarded (rd=x0 → 0)
        let mut program = Vec::new();
        program.extend_from_slice(&encode_jal(0, 8)); // jal x0, 8
        program.extend_from_slice(&encode_addi(1, 0, 99)); // skipped
        program.extend_from_slice(&encode_addi(2, 0, 7));

        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        // x0 is hardwired to 0 — the write emits new_value=0
        assert_eq!(regs[0], 0, "x0 hardwired to 0");
        assert_eq!(regs[1], 0, "x1 should not be set (skipped)");
        assert_eq!(regs[2], 7, "x2 should be 7");
    }

    #[test]
    fn jalr_jumps_to_register_plus_imm() {
        // addi x1, x0, 12  → x1 = 12
        // jalr x2, x1, 0   → x2 = pc+4, jump to x1+0=12
        // addi x3, x0, 99  (at pc=8, should be skipped)
        // addi x4, x0, 55  (at pc=12, target)
        let mut program = Vec::new();
        program.extend_from_slice(&encode_addi(1, 0, 12)); // pc=0: x1=12
        program.extend_from_slice(&encode_jalr(2, 1, 0)); // pc=4: x2=8, jump to 12
        program.extend_from_slice(&encode_addi(3, 0, 99)); // pc=8: skipped
        program.extend_from_slice(&encode_addi(4, 0, 55)); // pc=12: target

        let mut vm = VM::new(program);
        vm.run().unwrap();

        let regs = vm.trace.last().unwrap().state.registers;
        assert_eq!(regs[1], 12, "x1 = 12");
        assert_eq!(regs[2], 8, "x2 = return address = 4+4=8");
        assert_eq!(regs[3], 0, "x3 skipped");
        assert_eq!(regs[4], 55, "x4 = 55 (target)");
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
