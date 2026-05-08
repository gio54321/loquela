use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{prove_batch, BatchProof, CommonData, ProverData, StarkInstance};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::{
    extension::BinomialExtensionField, integers::QuotientMap, Field, PrimeCharacteristicRing,
};
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_lookup::{Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::StarkConfig;

use loquela_air::boundaries::air::BoundariesAir;
use loquela_air::decode::air::DecodeAir;
use loquela_air::instructions::add::air::AddAir;
use loquela_air::instructions::addi::air::AddiAir;
use loquela_air::instructions::and::air::AndInstrAir;
use loquela_air::instructions::andi::air::AndiAir;
use loquela_air::instructions::auipc::air::AuipcAir;
use loquela_air::instructions::jal::air::JalAir;
use loquela_air::instructions::jalr::air::JalrAir;
use loquela_air::instructions::lui::air::LuiAir;
use loquela_air::instructions::or::air::OrInstrAir;
use loquela_air::instructions::ori::air::OriAir;
use loquela_air::instructions::sll::air::SllAir;
use loquela_air::instructions::slli::air::SlliAir;
use loquela_air::instructions::slt::air::SltAir;
use loquela_air::instructions::slti::air::SltiAir;
use loquela_air::instructions::sltiu::air::SltiuAir;
use loquela_air::instructions::sltu::air::SltuAir;
use loquela_air::instructions::sra::air::SraAir;
use loquela_air::instructions::srai::air::SraiAir;
use loquela_air::instructions::srl::air::SrlAir;
use loquela_air::instructions::srli::air::SrliAir;
use loquela_air::instructions::sub::air::SubAir;
use loquela_air::instructions::xor::air::XorInstrAir;
use loquela_air::instructions::xori::air::XoriAir;
use loquela_air::memory::air::MemoryAir;
use loquela_air::primitives::and_lookup::AndAir;
use loquela_air::primitives::byte_less_than_lookup::LessThanAir;
use loquela_air::primitives::byte_lookup::BytesAir;
use loquela_air::primitives::byte_shift_left_lookup::ByteShiftLeftAir;
use loquela_air::primitives::byte_shift_right_lookup::ByteShiftRightAir;
use loquela_air::primitives::or_lookup::OrAir;
use loquela_air::primitives::timestamp_less_than::TimestampLessThanAir;
use loquela_air::primitives::u32_less_than_lookup::U32LessThanAir;
use loquela_air::primitives::xor_lookup::XorAir;
use loquela_air::program::air::ProgramAir;
use loquela_vm::{Instruction, MemoryOperation, VM};

// ── Config ────────────────────────────────────────────────────────────────────

type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn build_config() -> MyConfig {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 1);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = CirclePcs::new(val_mmcs, fri_params);
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    MyConfig::new(pcs, challenger)
}

// ── LoquelAir enum ────────────────────────────────────────────────────────────

/// Wraps every AIR variant so all instances share a single type for `prove_batch`.
#[derive(Clone)]
pub enum LoquelAir {
    Boundaries(BoundariesAir),
    Decode(DecodeAir),
    Add(AddAir),
    Addi(AddiAir),
    Andi(AndiAir),
    Ori(OriAir),
    Sub(SubAir),
    XorInstr(XorInstrAir),
    Xori(XoriAir),
    AndInstr(AndInstrAir),
    Sll(SllAir),
    Srl(SrlAir),
    Sra(SraAir),
    Slli(SlliAir),
    Srli(SrliAir),
    Srai(SraiAir),
    OrInstr(OrInstrAir),
    Memory(MemoryAir),
    Program(ProgramAir),
    Bytes(BytesAir),
    And(AndAir),
    ByteSll(ByteShiftLeftAir),
    ByteSrl(ByteShiftRightAir),
    Or(OrAir),
    Xor(XorAir),
    AndPrim(AndAir),
    OrPrim(OrAir),
    U32Lt(U32LessThanAir),
    TimestampLt(TimestampLessThanAir),
    BytesLt(LessThanAir),
    Slt(SltAir),
    Sltu(SltuAir),
    Slti(SltiAir),
    Sltiu(SltiuAir),
    Lui(LuiAir),
    Auipc(AuipcAir),
    Jal(JalAir),
    Jalr(JalrAir),
}

impl<F: Field> BaseAir<F> for LoquelAir {
    fn width(&self) -> usize {
        match self {
            LoquelAir::Boundaries(a) => BaseAir::<F>::width(a),
            LoquelAir::Decode(a) => BaseAir::<F>::width(a),
            LoquelAir::Add(a) => BaseAir::<F>::width(a),
            LoquelAir::Addi(a) => BaseAir::<F>::width(a),
            LoquelAir::Andi(a) => BaseAir::<F>::width(a),
            LoquelAir::Ori(a) => BaseAir::<F>::width(a),
            LoquelAir::Sub(a) => BaseAir::<F>::width(a),
            LoquelAir::XorInstr(a) => BaseAir::<F>::width(a),
            LoquelAir::Xori(a) => BaseAir::<F>::width(a),
            LoquelAir::AndInstr(a) => BaseAir::<F>::width(a),
            LoquelAir::Sll(a) => BaseAir::<F>::width(a),
            LoquelAir::Srl(a) => BaseAir::<F>::width(a),
            LoquelAir::Sra(a) => BaseAir::<F>::width(a),
            LoquelAir::Slli(a) => BaseAir::<F>::width(a),
            LoquelAir::Srli(a) => BaseAir::<F>::width(a),
            LoquelAir::Srai(a) => BaseAir::<F>::width(a),
            LoquelAir::OrInstr(a) => BaseAir::<F>::width(a),
            LoquelAir::Memory(a) => BaseAir::<F>::width(a),
            LoquelAir::Program(a) => BaseAir::<F>::width(a),
            LoquelAir::Bytes(a) => BaseAir::<F>::width(a),
            LoquelAir::And(a) => BaseAir::<F>::width(a),
            LoquelAir::ByteSll(a) => BaseAir::<F>::width(a),
            LoquelAir::ByteSrl(a) => BaseAir::<F>::width(a),
            LoquelAir::Or(a) => BaseAir::<F>::width(a),
            LoquelAir::Xor(a) => BaseAir::<F>::width(a),
            LoquelAir::AndPrim(a) => BaseAir::<F>::width(a),
            LoquelAir::OrPrim(a) => BaseAir::<F>::width(a),
            LoquelAir::U32Lt(a) => BaseAir::<F>::width(a),
            LoquelAir::TimestampLt(a) => BaseAir::<F>::width(a),
            LoquelAir::BytesLt(a) => BaseAir::<F>::width(a),
            LoquelAir::Slt(a) => BaseAir::<F>::width(a),
            LoquelAir::Sltu(a) => BaseAir::<F>::width(a),
            LoquelAir::Slti(a) => BaseAir::<F>::width(a),
            LoquelAir::Sltiu(a) => BaseAir::<F>::width(a),
            LoquelAir::Lui(a) => BaseAir::<F>::width(a),
            LoquelAir::Auipc(a) => BaseAir::<F>::width(a),
            LoquelAir::Jal(a) => BaseAir::<F>::width(a),
            LoquelAir::Jalr(a) => BaseAir::<F>::width(a),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            LoquelAir::Boundaries(a) => a.preprocessed_trace(),
            LoquelAir::Bytes(a) => a.preprocessed_trace(),
            LoquelAir::And(a) => a.preprocessed_trace(),
            LoquelAir::ByteSll(a) => a.preprocessed_trace(),
            LoquelAir::ByteSrl(a) => a.preprocessed_trace(),
            LoquelAir::Or(a) => a.preprocessed_trace(),
            LoquelAir::Xor(a) => a.preprocessed_trace(),
            LoquelAir::AndPrim(a) => a.preprocessed_trace(),
            LoquelAir::OrPrim(a) => a.preprocessed_trace(),
            LoquelAir::BytesLt(a) => a.preprocessed_trace(),
            _ => None,
        }
    }
}

impl<AB> Air<AB> for LoquelAir
where
    AB: AirBuilder,
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: Field + QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            LoquelAir::Boundaries(a) => a.eval(builder),
            LoquelAir::Decode(a) => a.eval(builder),
            LoquelAir::Add(a) => a.eval(builder),
            LoquelAir::Addi(a) => a.eval(builder),
            LoquelAir::Andi(a) => a.eval(builder),
            LoquelAir::Ori(a) => a.eval(builder),
            LoquelAir::Sub(a) => a.eval(builder),
            LoquelAir::XorInstr(a) => a.eval(builder),
            LoquelAir::Xori(a) => a.eval(builder),
            LoquelAir::AndInstr(a) => a.eval(builder),
            LoquelAir::Sll(a) => a.eval(builder),
            LoquelAir::Srl(a) => a.eval(builder),
            LoquelAir::Sra(a) => a.eval(builder),
            LoquelAir::Slli(a) => a.eval(builder),
            LoquelAir::Srli(a) => a.eval(builder),
            LoquelAir::Srai(a) => a.eval(builder),
            LoquelAir::OrInstr(a) => a.eval(builder),
            LoquelAir::Memory(a) => a.eval(builder),
            LoquelAir::Program(a) => a.eval(builder),
            LoquelAir::Bytes(a) => a.eval(builder),
            LoquelAir::And(a) => a.eval(builder),
            LoquelAir::ByteSll(a) => a.eval(builder),
            LoquelAir::ByteSrl(a) => a.eval(builder),
            LoquelAir::Or(a) => a.eval(builder),
            LoquelAir::Xor(a) => a.eval(builder),
            LoquelAir::AndPrim(a) => a.eval(builder),
            LoquelAir::OrPrim(a) => a.eval(builder),
            LoquelAir::U32Lt(a) => a.eval(builder),
            LoquelAir::TimestampLt(a) => a.eval(builder),
            LoquelAir::BytesLt(a) => a.eval(builder),
            LoquelAir::Slt(a) => a.eval(builder),
            LoquelAir::Sltu(a) => a.eval(builder),
            LoquelAir::Slti(a) => a.eval(builder),
            LoquelAir::Sltiu(a) => a.eval(builder),
            LoquelAir::Lui(a) => a.eval(builder),
            LoquelAir::Auipc(a) => a.eval(builder),
            LoquelAir::Jal(a) => a.eval(builder),
            LoquelAir::Jalr(a) => a.eval(builder),
        }
    }
}

impl<F: Field> LookupAir<F> for LoquelAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            LoquelAir::Boundaries(a) => <BoundariesAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Decode(a) => <DecodeAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Add(a) => <AddAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Addi(a) => <AddiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Andi(a) => <AndiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Ori(a) => <OriAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Sub(a) => <SubAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::XorInstr(a) => <XorInstrAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Xori(a) => <XoriAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::AndInstr(a) => <AndInstrAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Sll(a) => <SllAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Srl(a) => <SrlAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Sra(a) => <SraAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Slli(a) => <SlliAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Srli(a) => <SrliAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Srai(a) => <SraiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::OrInstr(a) => <OrInstrAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Memory(a) => <MemoryAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Program(a) => <ProgramAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Bytes(a) => <BytesAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::And(a) => <AndAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::ByteSll(a) => <ByteShiftLeftAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::ByteSrl(a) => <ByteShiftRightAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Or(a) => <OrAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Xor(a) => <XorAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::AndPrim(a) => <AndAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::OrPrim(a) => <OrAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::U32Lt(a) => <U32LessThanAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::TimestampLt(a) => {
                <TimestampLessThanAir as LookupAir<F>>::add_lookup_columns(a)
            }
            LoquelAir::BytesLt(a) => <LessThanAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Slt(a) => <SltAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Sltu(a) => <SltuAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Slti(a) => <SltiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Sltiu(a) => <SltiuAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Lui(a) => <LuiAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Auipc(a) => <AuipcAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Jal(a) => <JalAir as LookupAir<F>>::add_lookup_columns(a),
            LoquelAir::Jalr(a) => <JalrAir as LookupAir<F>>::add_lookup_columns(a),
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        match self {
            LoquelAir::Boundaries(a) => <BoundariesAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Decode(a) => <DecodeAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Add(a) => <AddAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Addi(a) => <AddiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Andi(a) => <AndiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Ori(a) => <OriAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Sub(a) => <SubAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::XorInstr(a) => <XorInstrAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Xori(a) => <XoriAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::AndInstr(a) => <AndInstrAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Sll(a) => <SllAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Srl(a) => <SrlAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Sra(a) => <SraAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Slli(a) => <SlliAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Srli(a) => <SrliAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Srai(a) => <SraiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::OrInstr(a) => <OrInstrAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Memory(a) => <MemoryAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Program(a) => <ProgramAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Bytes(a) => <BytesAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::And(a) => <AndAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::ByteSll(a) => <ByteShiftLeftAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::ByteSrl(a) => <ByteShiftRightAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Or(a) => <OrAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Xor(a) => <XorAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::AndPrim(a) => <AndAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::OrPrim(a) => <OrAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::U32Lt(a) => <U32LessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::TimestampLt(a) => <TimestampLessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::BytesLt(a) => <LessThanAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Slt(a) => <SltAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Sltu(a) => <SltuAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Slti(a) => <SltiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Sltiu(a) => <SltiuAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Lui(a) => <LuiAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Auipc(a) => <AuipcAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Jal(a) => <JalAir as LookupAir<F>>::get_lookups(a),
            LoquelAir::Jalr(a) => <JalrAir as LookupAir<F>>::get_lookups(a),
        }
    }
}

// ── AllTraces ─────────────────────────────────────────────────────────────────

/// All witness matrices for a single execution, bundled with their AIR instances.
pub struct AllTraces {
    pub airs: Vec<LoquelAir>,
    pub boundaries: RowMajorMatrix<Val>,
    pub decode: RowMajorMatrix<Val>,
    pub memory: RowMajorMatrix<Val>,
    pub program: RowMajorMatrix<Val>,
    pub bytes: RowMajorMatrix<Val>,
    pub and: RowMajorMatrix<Val>,
    pub or: RowMajorMatrix<Val>,
    pub xor: RowMajorMatrix<Val>,
    pub u32_lt: RowMajorMatrix<Val>,
    pub timestamp_lt: RowMajorMatrix<Val>,
    pub bytes_lt: RowMajorMatrix<Val>,
    /// Present when the program contains ADD instructions.
    pub add: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains ADDI instructions.
    pub addi: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains ANDI instructions.
    pub andi: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains ORI instructions.
    pub ori: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SUB instructions.
    pub sub: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains XORI instructions.
    pub xori: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains XOR instructions.
    pub xor_instr: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains OR instructions.
    pub or_instr: Option<RowMajorMatrix<Val>>,
    /// OR primitive lookup table (always present).
    pub or_prim: RowMajorMatrix<Val>,
    /// Present when the program contains AND instructions (instruction AIR).
    pub and_instr: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains AND instructions (bytes_and lookup table).
    pub and_prim: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLL instructions (instruction AIR).
    pub sll: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLL instructions (byte_sll lookup table).
    pub byte_sll: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SRL instructions (instruction AIR).
    pub srl: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SRL instructions (byte_srl lookup table).
    pub byte_srl: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SRA instructions (instruction AIR).
    pub sra: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLLI instructions (instruction AIR).
    pub slli: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SRLI instructions (instruction AIR).
    pub srli: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SRAI instructions (instruction AIR).
    pub srai: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLT instructions (instruction AIR).
    pub slt: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLTU instructions (instruction AIR).
    pub sltu: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLTI instructions (instruction AIR).
    pub slti: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains SLTIU instructions (instruction AIR).
    pub sltiu: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains LUI instructions.
    pub lui: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains AUIPC instructions.
    pub auipc: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains JAL instructions.
    pub jal: Option<RowMajorMatrix<Val>>,
    /// Present when the program contains JALR instructions.
    pub jalr: Option<RowMajorMatrix<Val>>,
}

impl AllTraces {
    /// Flatten into parallel vecs in the same order as `airs`.
    pub fn into_vecs(self) -> (Vec<LoquelAir>, Vec<RowMajorMatrix<Val>>) {
        let AllTraces {
            airs,
            boundaries,
            decode,
            memory,
            program,
            bytes,
            and,
            or,
            xor,
            u32_lt,
            timestamp_lt,
            bytes_lt,
            add,
            addi,
            andi,
            ori,
            sub,
            xori,
            xor_instr,
            or_instr,
            or_prim,
            and_instr,
            and_prim,
            sll,
            byte_sll,
            srl,
            byte_srl,
            sra,
            slli,
            srli,
            srai,
            slt,
            sltu,
            slti,
            sltiu,
            lui,
            auipc,
            jal,
            jalr,
        } = self;
        let mut traces = vec![
            boundaries,
            decode,
            memory,
            program,
            bytes,
            and,
            or,
            xor,
            u32_lt,
            timestamp_lt,
            bytes_lt,
            or_prim,
        ];
        if let Some(t) = add {
            traces.push(t);
        }
        if let Some(t) = addi {
            traces.push(t);
        }
        if let Some(t) = andi {
            traces.push(t);
        }
        if let Some(t) = ori {
            traces.push(t);
        }
        if let Some(t) = sub {
            traces.push(t);
        }
        if let Some(t) = xori {
            traces.push(t);
        }
        if let Some(t) = xor_instr {
            traces.push(t);
        }
        if let Some(t) = or_instr {
            traces.push(t);
        }
        if let Some(t) = and_instr {
            traces.push(t);
        }
        if let Some(t) = and_prim {
            traces.push(t);
        }
        if let Some(t) = sll {
            traces.push(t);
        }
        if let Some(t) = byte_sll {
            traces.push(t);
        }
        if let Some(t) = srl {
            traces.push(t);
        }
        if let Some(t) = byte_srl {
            traces.push(t);
        }
        if let Some(t) = sra {
            traces.push(t);
        }
        if let Some(t) = slli {
            traces.push(t);
        }
        if let Some(t) = srli {
            traces.push(t);
        }
        if let Some(t) = srai {
            traces.push(t);
        }
        if let Some(t) = slt {
            traces.push(t);
        }
        if let Some(t) = sltu {
            traces.push(t);
        }
        if let Some(t) = slti {
            traces.push(t);
        }
        if let Some(t) = sltiu {
            traces.push(t);
        }
        if let Some(t) = lui {
            traces.push(t);
        }
        if let Some(t) = auipc {
            traces.push(t);
        }
        if let Some(t) = jal {
            traces.push(t);
        }
        if let Some(t) = jalr {
            traces.push(t);
        }
        (airs, traces)
    }
}

// ── Multiplicity helpers ───────────────────────────────────────────────────────

/// Count byte occurrences across all u32 values that are byte-range-checked.
/// Returns `mults[v]` = number of times byte value `v` appeared.
fn bytes_multiplicities(vals: &[u32]) -> [Val; 256] {
    let mut mults = [Val::ZERO; 256];
    for v in vals {
        for byte in v.to_le_bytes() {
            mults[byte as usize] += Val::ONE;
        }
    }
    mults
}

/// Count XOR triples from XORI/XOR steps: row index = x * 256 + y.
fn xor_multiplicities(xori_ops: &[(u32, u32, u32)]) -> Vec<Val> {
    let mut mults = vec![Val::ZERO; 256 * 256];
    for &(x, y, _) in xori_ops {
        mults[x as usize * 256 + y as usize] += Val::ONE;
    }
    mults
}

/// Count OR triples (used for both ORI and OR steps): row index = x * 256 + y.
fn or_multiplicities(ops: &[(u32, u32, u32)]) -> Vec<Val> {
    let mut mults = vec![Val::ZERO; 256 * 256];
    for &(x, y, _) in ops {
        mults[x as usize * 256 + y as usize] += Val::ONE;
    }
    mults
}

/// Count AND triples (used for both ANDI and AND steps): row index = x * 256 + y.
fn and_multiplicities(ops: &[(u32, u32, u32)]) -> Vec<Val> {
    let mut mults = vec![Val::ZERO; 256 * 256];
    for &(x, y, _) in ops {
        mults[x as usize * 256 + y as usize] += Val::ONE;
    }
    mults
}

/// Collect sorted memory ops and derive all u32_lt and timestamp_lt entries,
/// plus the byte-level lt multiplicities consumed by LessThanAir.
fn memory_lookup_entries(
    all_ops: &[MemoryOperation],
) -> (
    Vec<(u32, u32, Val)>, // u32_lt entries: (addr_x, addr_y, mult)
    Vec<(u32, u32, Val)>, // timestamp_lt entries: (ts_x, ts_y, mult)
    Vec<Val>,             // bytes_lt mults (length <= 32640)
) {
    struct Op {
        memory_type: u8,
        address: u32,
        timestamp: u32,
    }
    let mut ops: Vec<Op> = all_ops
        .iter()
        .map(|op| match op {
            MemoryOperation::Read {
                memory_type,
                address,
                timestamp,
                ..
            } => Op {
                memory_type: *memory_type as u8,
                address: *address,
                timestamp: *timestamp,
            },
            MemoryOperation::Write {
                memory_type,
                address,
                timestamp,
                ..
            } => Op {
                memory_type: *memory_type as u8,
                address: *address,
                timestamp: *timestamp,
            },
        })
        .collect();
    ops.sort_by_key(|op| (op.memory_type, op.address, op.timestamp));

    let mut u32_lt_entries: Vec<(u32, u32, Val)> = Vec::new();
    let mut timestamp_lt_entries: Vec<(u32, u32, Val)> = Vec::new();
    let mut bytes_lt_mults_map = std::collections::HashMap::<(u8, u8), u64>::new();

    for i in 0..ops.len().saturating_sub(1) {
        let cur = &ops[i];
        let nxt = &ops[i + 1];

        let same_type = cur.memory_type == nxt.memory_type;
        if !same_type {
            continue;
        }

        let is_addr_equal = cur.address == nxt.address;
        u32_lt_entries.push((cur.address, nxt.address, Val::ONE));

        if !is_addr_equal {
            let x_bytes = cur.address.to_le_bytes();
            let y_bytes = nxt.address.to_le_bytes();
            for b in (0..4).rev() {
                if x_bytes[b] != y_bytes[b] {
                    let (lo, hi) = (x_bytes[b].min(y_bytes[b]), x_bytes[b].max(y_bytes[b]));
                    *bytes_lt_mults_map.entry((lo, hi)).or_insert(0) += 1;
                    break;
                }
            }
        }

        if is_addr_equal {
            timestamp_lt_entries.push((cur.timestamp, nxt.timestamp, Val::ONE));
        }
    }

    let num_valid = 256 * 255 / 2;
    let mut bytes_lt_mults = vec![Val::ZERO; num_valid];
    for ((x, y), count) in &bytes_lt_mults_map {
        let row = (*y as usize) * (*y as usize - 1) / 2 + (*x as usize);
        bytes_lt_mults[row] = Val::from_u64(*count);
    }

    (u32_lt_entries, timestamp_lt_entries, bytes_lt_mults)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Execute `program` and build all witness matrices (without proving).
///
/// The returned `AllTraces` can be inspected or mutated before passing to
/// `prove_traces`, which is useful for negative testing.
pub fn generate_traces(program: &[u8]) -> AllTraces {
    // 1. Execute the VM.
    let mut vm = VM::new(program.to_vec());
    vm.run().expect("VM execution failed");

    let steps = &vm.trace;
    println!("VM executed in {} steps", steps.len());
    let all_ops: Vec<MemoryOperation> = vm.get_memory_ops().into_iter().cloned().collect();

    // 2. Classify steps by instruction type.
    let has_add = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Add { .. }));
    let has_addi = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::AddI { .. }));
    let has_andi = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::AndiI { .. }));
    let has_ori = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::OriI { .. }));
    let has_sub = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Sub { .. }));
    let has_xori = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::XorI { .. }));
    let has_xor = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Xor { .. }));
    let has_or = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Or { .. }));
    let has_and = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::And { .. }));
    let has_sll = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Sll { .. }));
    let has_srl = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Srl { .. }));
    let has_sra = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Sra { .. }));
    let has_slli = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::SlliI { .. }));
    let has_srli = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::SrliI { .. }));
    let has_srai = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::SraiI { .. }));
    let has_slt = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Slt { .. }));
    let has_sltu = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Sltu { .. }));
    let has_slti = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::SltiI { .. }));
    let has_sltiu = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::SltiuI { .. }));
    let has_lui = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Lui { .. }));
    let has_auipc = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Auipc { .. }));
    let has_jal = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Jal { .. }));
    let has_jalr = steps
        .iter()
        .any(|s| matches!(s.instruction, Instruction::Jalr { .. }));

    // 3. Build traces.
    println!("Building AIR instances and traces...");

    let _final_step = steps.last().expect("no steps");
    // vm.pc is the PC after the last instruction executed — the next_pc sent on the trace bus.
    let final_pc_bytes = vm.pc.to_le_bytes();
    let final_pc: [Val; 4] = final_pc_bytes.map(|b| Val::from_u64(b as u64));
    let final_ts = Val::from_u64(vm.timestamp as u64);
    let boundaries = loquela_air::boundaries::air::build_trace(final_pc, final_ts);

    let decode = loquela_air::decode::trace::build_trace::<Val>(steps);

    let add_trace = if has_add {
        Some(loquela_air::instructions::add::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let addi_trace = if has_addi {
        Some(loquela_air::instructions::addi::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let sub_trace = if has_sub {
        Some(loquela_air::instructions::sub::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let andi_trace = if has_andi {
        Some(loquela_air::instructions::andi::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let ori_trace = if has_ori {
        Some(loquela_air::instructions::ori::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let xori_trace = if has_xori {
        Some(loquela_air::instructions::xori::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let xor_instr_trace = if has_xor {
        Some(loquela_air::instructions::xor::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let or_instr_trace = if has_or {
        Some(loquela_air::instructions::or::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let and_instr_trace = if has_and {
        Some(loquela_air::instructions::and::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let sll_trace = if has_sll {
        Some(loquela_air::instructions::sll::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let srl_trace = if has_srl {
        Some(loquela_air::instructions::srl::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let sra_trace = if has_sra {
        Some(loquela_air::instructions::sra::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let slli_trace = if has_slli {
        Some(loquela_air::instructions::slli::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let srli_trace = if has_srli {
        Some(loquela_air::instructions::srli::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let srai_trace = if has_srai {
        Some(loquela_air::instructions::srai::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let slt_trace = if has_slt {
        Some(loquela_air::instructions::slt::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let sltu_trace = if has_sltu {
        Some(loquela_air::instructions::sltu::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let slti_trace = if has_slti {
        Some(loquela_air::instructions::slti::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let sltiu_trace = if has_sltiu {
        Some(loquela_air::instructions::sltiu::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let lui_trace = if has_lui {
        Some(loquela_air::instructions::lui::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let auipc_trace = if has_auipc {
        Some(loquela_air::instructions::auipc::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let jal_trace = if has_jal {
        Some(loquela_air::instructions::jal::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };
    let jalr_trace = if has_jalr {
        Some(loquela_air::instructions::jalr::trace::build_trace::<Val>(
            steps,
        ))
    } else {
        None
    };

    let memory = loquela_air::memory::trace::build_trace::<Val>(&all_ops);

    let n_decode_steps = steps.len();
    // decode::trace::build_trace pads to next_power_of_two().max(4); mirror that here
    // so the program AIR's PC=0..3 multiplicities account for every padding-row fetch.
    let num_decode_padding = n_decode_steps
        .next_power_of_two()
        .max(4)
        .saturating_sub(n_decode_steps);
    let program_trace =
        loquela_air::program::trace::build_trace::<Val>(program, steps, num_decode_padding);

    let (u32_lt_entries, timestamp_lt_entries, bytes_lt_mults) = memory_lookup_entries(&all_ops);

    // Collect all u32 values that are byte-range-checked (each is expanded to
    // 4 byte multiplicities) and all single-byte values that get range-checked
    // individually (e.g. rs2_shamt_high, rs1_byte3_low7). Mixing them was the
    // source of a subtle bug: pushing a single-byte value as u32 would incorrectly
    // bump multiplicities[0] by 3 for the three high zero bytes.
    let mut byte_checked_vals: Vec<u32> = Vec::new();
    let mut byte_checked_singles: Vec<u8> = Vec::new();
    for s in steps.iter() {
        match s.memory_ops.as_slice() {
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::AddI { .. }) =>
            {
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(
                    s.instruction,
                    Instruction::Add { .. } | Instruction::Sub { .. }
                ) =>
            {
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rs2);
                byte_checked_vals.push(*rd);
            }
            // XOR: range-check rs1 and rs2 bytes via "bytes" bus (rd is checked by bytes_xor).
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { .. }]
                if matches!(s.instruction, Instruction::Xor { .. }) =>
            {
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rs2);
            }
            [MemoryOperation::Read { value: _rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Sll { .. }) =>
            {
                // SLL byte-range checks: rs2_value bytes, rd bytes.
                // rs1 bytes are covered by the byte_sll lookup.
                // rs2_shamt_high is a small value (0..7) also range-checked individually.
                byte_checked_vals.push(*rs2);
                byte_checked_vals.push(*rd);
                let rs2_shamt_high = ((*rs2 & 0xFF) >> 5) as u8;
                byte_checked_singles.push(rs2_shamt_high);
            }
            [MemoryOperation::Read { value: _rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Srl { .. }) =>
            {
                // SRL byte-range checks: rs2_value bytes, rd bytes.
                // rs1 bytes are covered by the byte_srl lookup.
                // rs2_shamt_high is a small value (0..7) also range-checked individually.
                byte_checked_vals.push(*rs2);
                byte_checked_vals.push(*rd);
                let rs2_shamt_high = ((*rs2 & 0xFF) >> 5) as u8;
                byte_checked_singles.push(rs2_shamt_high);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Sra { .. }) =>
            {
                // SRA byte-range checks: rs2_value bytes, rd bytes, rs1_byte3_low7.
                // rs1 bytes are covered by the byte_srl lookup.
                // rs2_shamt_high is a small value (0..7) also range-checked individually.
                byte_checked_vals.push(*rs2);
                byte_checked_vals.push(*rd);
                // rs2_shamt_high: upper 3 bits of rs2 low byte
                let rs2_shamt_high = ((*rs2 & 0xFF) >> 5) as u8;
                byte_checked_singles.push(rs2_shamt_high);
                // rs1_byte3_low7: low 7 bits of rs1's high byte
                let rs1_byte3_low7 = ((rs1 >> 24) & 0x7F) as u8;
                byte_checked_singles.push(rs1_byte3_low7);
            }
            [MemoryOperation::Read { .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::SlliI { .. }) =>
            {
                // SLLI: rd bytes are range-checked; rs1 bytes covered by byte_sll lookup.
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Read { .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::SrliI { .. }) =>
            {
                // SRLI: rd bytes are range-checked; rs1 bytes covered by byte_srl lookup.
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::SraiI { .. }) =>
            {
                // SRAI: rd bytes and rs1_byte3_low7 are range-checked;
                // rs1 bytes covered by byte_srl lookup.
                byte_checked_vals.push(*rd);
                let rs1_byte3_low7 = ((rs1 >> 24) & 0x7F) as u8;
                byte_checked_singles.push(rs1_byte3_low7);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Sltu { .. }) =>
            {
                // SLTU: rs1_bytes, rs2_bytes, diff_bytes are range-checked.
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rs2);
                let diff = rs1.wrapping_sub(*rs2);
                byte_checked_vals.push(diff);
                let _ = rd;
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Slt { .. }) =>
            {
                // SLT: rs1_bytes, rs2_bytes, diff_bytes, rs1_byte3_low7, rs2_byte3_low7 range-checked.
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rs2);
                let diff = rs1.wrapping_sub(*rs2);
                byte_checked_vals.push(diff);
                let rs1_byte3_low7 = ((rs1 >> 24) & 0x7F) as u8;
                byte_checked_singles.push(rs1_byte3_low7);
                let rs2_byte3_low7 = ((rs2 >> 24) & 0x7F) as u8;
                byte_checked_singles.push(rs2_byte3_low7);
                let _ = rd;
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::SltiuI { .. }) =>
            {
                // SLTIU: rs1_bytes and diff_bytes are range-checked.
                let imm = match s.instruction {
                    Instruction::SltiuI { imm, .. } => imm as u32,
                    _ => unreachable!(),
                };
                byte_checked_vals.push(*rs1);
                let diff = rs1.wrapping_sub(imm);
                byte_checked_vals.push(diff);
                let _ = rd;
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::SltiI { .. }) =>
            {
                // SLTI: rs1_bytes, diff_bytes, rs1_byte3_low7 are range-checked.
                let imm = match s.instruction {
                    Instruction::SltiI { imm, .. } => imm as u32,
                    _ => unreachable!(),
                };
                byte_checked_vals.push(*rs1);
                let diff = rs1.wrapping_sub(imm);
                byte_checked_vals.push(diff);
                let rs1_byte3_low7 = ((rs1 >> 24) & 0x7F) as u8;
                byte_checked_singles.push(rs1_byte3_low7);
                let _ = rd;
            }
            [MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Lui { .. }) =>
            {
                // LUI: rd_val bytes are range-checked.
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Auipc { .. }) =>
            {
                // AUIPC: pc bytes, imm_u bytes 1..3, and rd_val bytes are all
                // range-checked by the AIR. imm_u[0] is constrained to 0 in eval
                // and skipped here.
                byte_checked_vals.push(s.state.pc);
                byte_checked_vals.push(*rd);
                let imm = match s.instruction {
                    Instruction::Auipc { imm, .. } => imm as u32,
                    _ => unreachable!(),
                };
                let imm_u = imm << 12;
                let imm_u_bytes = imm_u.to_le_bytes();
                byte_checked_singles.push(imm_u_bytes[1]);
                byte_checked_singles.push(imm_u_bytes[2]);
                byte_checked_singles.push(imm_u_bytes[3]);
            }
            [MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Jal { .. }) =>
            {
                // JAL: pc bytes and rd_val (=pc+4) bytes are range-checked.
                byte_checked_vals.push(s.state.pc);
                byte_checked_vals.push(*rd);
            }
            [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }]
                if matches!(s.instruction, Instruction::Jalr { .. }) =>
            {
                // JALR: rs1_value bytes and rd_val (=pc+4) bytes are range-checked.
                byte_checked_vals.push(*rs1);
                byte_checked_vals.push(*rd);
            }
            _ => {}
        }
    }
    let mut bytes_mults = bytes_multiplicities(&byte_checked_vals);
    for &b in &byte_checked_singles {
        bytes_mults[b as usize] += Val::ONE;
    }
    let bytes = loquela_air::primitives::byte_lookup::build_trace::<Val>(&bytes_mults);

    let mut xori_triples: Vec<(u32, u32, u32)> = Vec::new();
    let mut ori_triples: Vec<(u32, u32, u32)> = Vec::new();
    let mut andi_triples: Vec<(u32, u32, u32)> = Vec::new();
    for s in steps.iter() {
        match s.instruction {
            Instruction::XorI { imm, .. } => {
                let imm_se = imm as i32 as u32;
                if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }] =
                    s.memory_ops.as_slice()
                {
                    let rs1_b = rs1.to_le_bytes();
                    let imm_b = imm_se.to_le_bytes();
                    let rd_b = rd.to_le_bytes();
                    for i in 0..4 {
                        xori_triples.push((rs1_b[i] as u32, imm_b[i] as u32, rd_b[i] as u32));
                    }
                }
            }
            Instruction::Xor { .. } => {
                if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }] =
                    s.memory_ops.as_slice()
                {
                    let rs1_b = rs1.to_le_bytes();
                    let rs2_b = rs2.to_le_bytes();
                    let rd_b = rd.to_le_bytes();
                    for i in 0..4 {
                        xori_triples.push((rs1_b[i] as u32, rs2_b[i] as u32, rd_b[i] as u32));
                    }
                }
            }
            Instruction::OriI { imm, .. } => {
                let imm_se = imm as i32 as u32;
                if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }] =
                    s.memory_ops.as_slice()
                {
                    let rs1_b = rs1.to_le_bytes();
                    let imm_b = imm_se.to_le_bytes();
                    let rd_b = rd.to_le_bytes();
                    for i in 0..4 {
                        ori_triples.push((rs1_b[i] as u32, imm_b[i] as u32, rd_b[i] as u32));
                    }
                }
            }
            Instruction::AndiI { imm, .. } => {
                let imm_se = imm as i32 as u32;
                if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Write { new_value: rd, .. }] =
                    s.memory_ops.as_slice()
                {
                    let rs1_b = rs1.to_le_bytes();
                    let imm_b = imm_se.to_le_bytes();
                    let rd_b = rd.to_le_bytes();
                    for i in 0..4 {
                        andi_triples.push((rs1_b[i] as u32, imm_b[i] as u32, rd_b[i] as u32));
                    }
                }
            }
            _ => {}
        }
    }
    let xor_mults = xor_multiplicities(&xori_triples);
    let xor = loquela_air::primitives::xor_lookup::build_trace::<Val>(&xor_mults);
    let or_mults = or_multiplicities(&ori_triples);
    let or = loquela_air::primitives::or_lookup::build_trace::<Val>(&or_mults);
    let and_mults = and_multiplicities(&andi_triples);
    let and = loquela_air::primitives::and_lookup::build_trace::<Val>(&and_mults);

    let mut or_triples: Vec<(u32, u32, u32)> = Vec::new();
    for s in steps.iter() {
        if !matches!(s.instruction, Instruction::Or { .. }) {
            continue;
        }
        if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }] =
            s.memory_ops.as_slice()
        {
            let rs1_b = rs1.to_le_bytes();
            let rs2_b = rs2.to_le_bytes();
            let rd_b = rd.to_le_bytes();
            for i in 0..4 {
                or_triples.push((rs1_b[i] as u32, rs2_b[i] as u32, rd_b[i] as u32));
            }
        }
    }
    let or_mults = or_multiplicities(&or_triples);
    let or_prim = loquela_air::primitives::or_lookup::build_trace::<Val>(&or_mults);

    let mut and_triples: Vec<(u32, u32, u32)> = Vec::new();
    for s in steps.iter() {
        if !matches!(s.instruction, Instruction::And { .. }) {
            continue;
        }
        if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, MemoryOperation::Write { new_value: rd, .. }] =
            s.memory_ops.as_slice()
        {
            let rs1_b = rs1.to_le_bytes();
            let rs2_b = rs2.to_le_bytes();
            let rd_b = rd.to_le_bytes();
            for i in 0..4 {
                and_triples.push((rs1_b[i] as u32, rs2_b[i] as u32, rd_b[i] as u32));
            }
        }
    }
    let and_prim_trace = if has_and {
        let and_mults = and_multiplicities(&and_triples);
        Some(loquela_air::primitives::and_lookup::build_trace::<Val>(
            &and_mults,
        ))
    } else {
        None
    };

    // Compute byte_sll multiplicities for SLL and SLLI instructions.
    // For each SLL/SLLI step, we emit 4 lookups: (rs1_bytes[i], bit_shamt, shifted, carry).
    // Row index in the byte_sll table = byte_val * 8 + bit_shamt.
    let byte_sll_prim_trace = if has_sll || has_slli {
        let mut byte_sll_mults = vec![Val::ZERO; 256 * 8];
        for s in steps.iter() {
            let (rs1_val, shamt) = match &s.instruction {
                Instruction::Sll { .. } => {
                    if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, ..] =
                        s.memory_ops.as_slice()
                    {
                        (*rs1, *rs2 & 0x1F)
                    } else {
                        continue;
                    }
                }
                Instruction::SlliI { imm, .. } => {
                    if let [MemoryOperation::Read { value: rs1, .. }, ..] = s.memory_ops.as_slice()
                    {
                        (*rs1, (*imm & 0x1F) as u32)
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            let bit_shamt = (shamt % 8) as usize;
            let rs1_b = rs1_val.to_le_bytes();
            for i in 0..4 {
                let byte_val = rs1_b[i] as usize;
                let row = byte_val * 8 + bit_shamt;
                byte_sll_mults[row] += Val::ONE;
            }
        }
        Some(loquela_air::primitives::byte_shift_left_lookup::build_trace::<Val>(&byte_sll_mults))
    } else {
        None
    };

    // Compute byte_srl multiplicities for SRL, SRA, SRLI, and SRAI instructions.
    // For each such step, we emit 4 lookups: (rs1_bytes[i], bit_shamt, shifted, carry).
    // Row index in the byte_srl table = byte_val * 8 + bit_shamt.
    let byte_srl_prim_trace = if has_srl || has_sra || has_srli || has_srai {
        let mut byte_srl_mults = vec![Val::ZERO; 256 * 8];
        for s in steps.iter() {
            let (rs1_val, shamt) = match &s.instruction {
                Instruction::Srl { .. } | Instruction::Sra { .. } => {
                    if let [MemoryOperation::Read { value: rs1, .. }, MemoryOperation::Read { value: rs2, .. }, ..] =
                        s.memory_ops.as_slice()
                    {
                        (*rs1, *rs2 & 0x1F)
                    } else {
                        continue;
                    }
                }
                Instruction::SrliI { imm, .. } | Instruction::SraiI { imm, .. } => {
                    if let [MemoryOperation::Read { value: rs1, .. }, ..] = s.memory_ops.as_slice()
                    {
                        (*rs1, (*imm & 0x1F) as u32)
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            let bit_shamt = (shamt % 8) as usize;
            let rs1_b = rs1_val.to_le_bytes();
            for i in 0..4 {
                let byte_val = rs1_b[i] as usize;
                let row = byte_val * 8 + bit_shamt;
                byte_srl_mults[row] += Val::ONE;
            }
        }
        Some(loquela_air::primitives::byte_shift_right_lookup::build_trace::<Val>(&byte_srl_mults))
    } else {
        None
    };

    let u32_lt = loquela_air::primitives::u32_less_than_lookup::build_trace::<Val>(&u32_lt_entries);
    let timestamp_lt =
        loquela_air::primitives::timestamp_less_than::build_trace::<Val>(&timestamp_lt_entries);
    let bytes_lt =
        loquela_air::primitives::byte_less_than_lookup::build_trace::<Val>(&bytes_lt_mults);

    // 4. Assemble AIRs in the same order as the traces vec built in `into_vecs`.
    let mut airs: Vec<LoquelAir> = vec![
        LoquelAir::Boundaries(BoundariesAir::new()),
        LoquelAir::Decode(DecodeAir::new()),
        LoquelAir::Memory(MemoryAir::new()),
        LoquelAir::Program(ProgramAir::new()),
        LoquelAir::Bytes(BytesAir::new()),
        LoquelAir::And(AndAir::new()),
        LoquelAir::Or(OrAir::new()),
        LoquelAir::Xor(XorAir::new()),
        LoquelAir::U32Lt(U32LessThanAir::new()),
        LoquelAir::TimestampLt(TimestampLessThanAir::new()),
        LoquelAir::BytesLt(LessThanAir::new()),
        LoquelAir::OrPrim(OrAir::new()),
    ];

    if has_add {
        airs.push(LoquelAir::Add(AddAir::new()));
    }
    if has_addi {
        airs.push(LoquelAir::Addi(AddiAir::new()));
    }
    if has_andi {
        airs.push(LoquelAir::Andi(AndiAir::new()));
    }
    if has_ori {
        airs.push(LoquelAir::Ori(OriAir::new()));
    }
    if has_sub {
        airs.push(LoquelAir::Sub(SubAir::new()));
    }
    if has_xori {
        airs.push(LoquelAir::Xori(XoriAir::new()));
    }
    if has_xor {
        airs.push(LoquelAir::XorInstr(XorInstrAir::new()));
    }
    if has_or {
        airs.push(LoquelAir::OrInstr(OrInstrAir::new()));
    }
    if has_and {
        airs.push(LoquelAir::AndInstr(AndInstrAir::new()));
        airs.push(LoquelAir::AndPrim(AndAir::new()));
    }
    if has_sll {
        airs.push(LoquelAir::Sll(SllAir::new()));
    }
    if has_sll || has_slli {
        airs.push(LoquelAir::ByteSll(ByteShiftLeftAir::new()));
    }
    if has_srl {
        airs.push(LoquelAir::Srl(SrlAir::new()));
    }
    if has_srl || has_sra || has_srli || has_srai {
        airs.push(LoquelAir::ByteSrl(ByteShiftRightAir::new()));
    }
    if has_sra {
        airs.push(LoquelAir::Sra(SraAir::new()));
    }
    if has_slli {
        airs.push(LoquelAir::Slli(SlliAir::new()));
    }
    if has_srli {
        airs.push(LoquelAir::Srli(SrliAir::new()));
    }
    if has_srai {
        airs.push(LoquelAir::Srai(SraiAir::new()));
    }
    if has_slt {
        airs.push(LoquelAir::Slt(SltAir::new()));
    }
    if has_sltu {
        airs.push(LoquelAir::Sltu(SltuAir::new()));
    }
    if has_slti {
        airs.push(LoquelAir::Slti(SltiAir::new()));
    }
    if has_sltiu {
        airs.push(LoquelAir::Sltiu(SltiuAir::new()));
    }
    if has_lui {
        airs.push(LoquelAir::Lui(LuiAir::new()));
    }
    if has_auipc {
        airs.push(LoquelAir::Auipc(AuipcAir::new()));
    }
    if has_jal {
        airs.push(LoquelAir::Jal(JalAir::new()));
    }
    if has_jalr {
        airs.push(LoquelAir::Jalr(JalrAir::new()));
    }

    AllTraces {
        airs,
        boundaries,
        decode,
        memory,
        program: program_trace,
        bytes,
        and,
        or,
        xor,
        u32_lt,
        timestamp_lt,
        bytes_lt,
        add: add_trace,
        addi: addi_trace,
        andi: andi_trace,
        ori: ori_trace,
        sub: sub_trace,
        xori: xori_trace,
        xor_instr: xor_instr_trace,
        or_instr: or_instr_trace,
        or_prim,
        and_instr: and_instr_trace,
        and_prim: and_prim_trace,
        sll: sll_trace,
        byte_sll: byte_sll_prim_trace,
        srl: srl_trace,
        byte_srl: byte_srl_prim_trace,
        sra: sra_trace,
        slli: slli_trace,
        srli: srli_trace,
        srai: srai_trace,
        slt: slt_trace,
        sltu: sltu_trace,
        slti: slti_trace,
        sltiu: sltiu_trace,
        lui: lui_trace,
        auipc: auipc_trace,
        jal: jal_trace,
        jalr: jalr_trace,
    }
}

/// Low-level prove: commit to `traces`, run FRI, return a batch proof.
///
/// Callers are responsible for ensuring `airs` and `traces` are parallel and
/// ordered consistently (use `AllTraces::into_vecs` to guarantee this).
fn do_prove(
    config: &MyConfig,
    airs: &[LoquelAir],
    traces: &[RowMajorMatrix<Val>],
) -> (BatchProof<MyConfig>, CommonData<MyConfig>) {
    let trace_refs: Vec<&RowMajorMatrix<Val>> = traces.iter().collect();
    let pvs: Vec<Vec<Val>> = vec![vec![]; airs.len()];

    let initial_instances: Vec<StarkInstance<'_, MyConfig, LoquelAir>> = airs
        .iter()
        .zip(trace_refs.iter())
        .map(|(air, trace)| StarkInstance {
            air,
            trace,
            public_values: vec![],
            lookups: vec![],
        })
        .collect();

    println!("Building prover data ({} AIRs)...", airs.len());
    let prover_data = ProverData::from_instances(config, &initial_instances);
    let instances = StarkInstance::new_multiple(airs, &trace_refs, &pvs, &prover_data.common);

    println!("Proving...");
    let proof = prove_batch(config, &instances, &prover_data);
    (proof, prover_data.common)
}

/// Prove a set of traces produced by `generate_traces` (or a mutation thereof).
pub fn prove_traces(all_traces: AllTraces) -> BatchProof<MyConfig> {
    let (airs, traces) = all_traces.into_vecs();
    let config = build_config();
    do_prove(&config, &airs, &traces).0
}

/// Execute `program`, build all traces, and return a batch STARK proof.
pub fn prove(program: &[u8]) -> BatchProof<MyConfig> {
    prove_traces(generate_traces(program))
}

pub mod debug;

#[cfg(test)]
mod tests;
