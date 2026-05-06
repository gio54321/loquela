#!/bin/sh

riscv64-unknown-elf-as -march=rv32i -mabi=ilp32 test.s -o test.o
riscv64-unknown-elf-objcopy -O binary test.o test.bin