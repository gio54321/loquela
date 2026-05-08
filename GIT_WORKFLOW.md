You are an implementor for a zkvm called loquela. Spawn a subagent that is tasked to implement new instructions, use sonnet with medium effort.

The subagent will do the following tasks:
- Pick an opcode from the list of opcodes in the README that is not yet implemented. Start with the ones that are closest to the ones already implemented (e.g. if ADDI is implemented, ADD, SUB, SUBI are good next steps).
- Create a new branch named `opcodes/<opcode>` (e.g. `opcodes/add`) from `main`.
- Reference the RISC-V specification to understand the instruction format, semantics, and encoding. There is an html page at the root of the repo with the spec, but you can also refer to the official RISC-V spec online.
- Implement the instruction AIR: it always receives `(pc, ts)` on the `"trace"` bus, reads source registers (if any) via `"memory"`, writes the destination register (if any) via `"memory"`, and performs the appropriate arithmetic/logic operation using constraints and lookup tables as needed. You can refer to the existing AIR implementations for guidance.
- Implement the instruction trace generation for the AIR.
- Implement the instruction execution in the VM.
- Add tests for the instruction, reference existing tests for guidance.
- Commit after each step with a clear commit message describing what was done. Refer to the commit messages in the existing code for examples of good commit messages.