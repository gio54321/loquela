//! Debug helpers: dump every AIR's main + preprocessed trace, then evaluate
//! every lookup tuple on every row and print it. Also exposes a wrapper
//! around p3-lookup's `check_lookups` so any global-bus imbalance is reported
//! with concrete (instance, lookup, row) locations.

use p3_air::{
    AirBuilder, BaseAir, ExtensionBuilder, PermutationAirBuilder, RowWindow,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::{
    debug_util::{check_lookups, LookupDebugInstance},
    lookup_traits::symbolic_to_expr,
    Kind, LookupAir,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{LoquelAir, Val};

pub fn air_name(air: &LoquelAir) -> &'static str {
    match air {
        LoquelAir::Boundaries(_) => "Boundaries",
        LoquelAir::Decode(_) => "Decode",
        LoquelAir::Add(_) => "Add",
        LoquelAir::Addi(_) => "Addi",
        LoquelAir::Andi(_) => "Andi",
        LoquelAir::Ori(_) => "Ori",
        LoquelAir::Sub(_) => "Sub",
        LoquelAir::XorInstr(_) => "XorInstr",
        LoquelAir::Xori(_) => "Xori",
        LoquelAir::AndInstr(_) => "AndInstr",
        LoquelAir::Sll(_) => "Sll",
        LoquelAir::Srl(_) => "Srl",
        LoquelAir::Sra(_) => "Sra",
        LoquelAir::Slli(_) => "Slli",
        LoquelAir::Srli(_) => "Srli",
        LoquelAir::Srai(_) => "Srai",
        LoquelAir::OrInstr(_) => "OrInstr",
        LoquelAir::Memory(_) => "Memory",
        LoquelAir::Program(_) => "Program",
        LoquelAir::ProgramHash(_) => "ProgramHash",
        LoquelAir::Poseidon2Chip(_) => "Poseidon2Chip",
        LoquelAir::Bytes(_) => "Bytes",
        LoquelAir::And(_) => "And(andi-prim)",
        LoquelAir::ByteSll(_) => "ByteSll",
        LoquelAir::ByteSrl(_) => "ByteSrl",
        LoquelAir::Or(_) => "Or(ori-prim)",
        LoquelAir::Xor(_) => "Xor",
        LoquelAir::AndPrim(_) => "AndPrim",
        LoquelAir::OrPrim(_) => "OrPrim",
        LoquelAir::U32Lt(_) => "U32Lt",
        LoquelAir::TimestampLt(_) => "TimestampLt",
        LoquelAir::BytesLt(_) => "BytesLt",
        LoquelAir::Slt(_) => "Slt",
        LoquelAir::Sltu(_) => "Sltu",
        LoquelAir::Slti(_) => "Slti",
        LoquelAir::Sltiu(_) => "Sltiu",
        LoquelAir::Lui(_) => "Lui",
        LoquelAir::Auipc(_) => "Auipc",
        LoquelAir::Jal(_) => "Jal",
        LoquelAir::Jalr(_) => "Jalr",
    }
}

/// Mini AIR builder that evaluates symbolic expressions to concrete field elements.
struct MiniBuilder<'a> {
    main: RowWindow<'a, Val>,
    preprocessed: RowWindow<'a, Val>,
    public_values: &'a [Val],
    row: usize,
    height: usize,
}

impl<'a> AirBuilder for MiniBuilder<'a> {
    type F = Val;
    type Expr = Val;
    type Var = Val;
    type PreprocessedWindow = RowWindow<'a, Val>;
    type MainWindow = RowWindow<'a, Val>;
    type PublicVar = Val;

    fn main(&self) -> Self::MainWindow {
        self.main
    }
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }
    fn is_first_row(&self) -> Self::Expr {
        Val::from_bool(self.row == 0)
    }
    fn is_last_row(&self) -> Self::Expr {
        Val::from_bool(self.row + 1 == self.height)
    }
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2);
        Val::from_bool(self.row + 1 < self.height)
    }
    fn assert_zero<I: Into<Self::Expr>>(&mut self, _: I) {}
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a> ExtensionBuilder for MiniBuilder<'a> {
    type EF = Val;
    type ExprEF = Val;
    type VarEF = Val;
    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, _: I) {}
}

impl<'a> PermutationAirBuilder for MiniBuilder<'a> {
    type MP = RowWindow<'a, Val>;
    type RandomVar = Val;
    type PermutationVar = Val;
    fn permutation(&self) -> Self::MP {
        RowWindow::from_two_rows(&[], &[])
    }
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &[]
    }
    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &[]
    }
}

fn fmt_row(row: &[Val]) -> String {
    row.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Materialize main and (optional) preprocessed traces as `Vec<Vec<Val>>`
/// so that we can hand stable `&[Val]` slices to `RowWindow::from_two_rows`.
fn materialize(matrix: &RowMajorMatrix<Val>) -> Vec<Vec<Val>> {
    (0..matrix.height())
        .map(|r| matrix.row_slice(r).unwrap().to_vec())
        .collect()
}

/// Print every AIR's trace + preprocessed trace + every lookup, evaluated row-by-row.
/// Lookup rows whose multiplicity is zero are skipped.
pub fn dump_traces_and_lookups(airs: &mut [LoquelAir], traces: &[RowMajorMatrix<Val>]) {
    assert_eq!(airs.len(), traces.len());

    let prep_traces: Vec<Option<RowMajorMatrix<Val>>> = airs
        .iter()
        .map(|a| <LoquelAir as BaseAir<Val>>::preprocessed_trace(a))
        .collect();
    let lookups_per_air: Vec<_> = airs
        .iter_mut()
        .map(|a| <LoquelAir as LookupAir<Val>>::get_lookups(a))
        .collect();

    for (i, (air, trace)) in airs.iter().zip(traces.iter()).enumerate() {
        let name = air_name(air);
        let height = trace.height();
        let width = trace.width;
        println!(
            "\n========== AIR[{i}] {name}  ({height} rows × {width} cols) =========="
        );

        let main_rows: Vec<Vec<Val>> = materialize(trace);
        for (r, row) in main_rows.iter().enumerate() {
            println!("  main[{r:>3}]: {}", fmt_row(row));
        }

        let prep_rows: Option<Vec<Vec<Val>>> = prep_traces[i].as_ref().map(materialize);
        if let Some(prep) = &prep_rows {
            println!(
                "  -- preprocessed ({} rows × {} cols) --",
                prep.len(),
                prep.first().map(|r| r.len()).unwrap_or(0)
            );
            for (r, row) in prep.iter().enumerate() {
                println!("  prep[{r:>3}]: {}", fmt_row(row));
            }
        }

        let lookups = &lookups_per_air[i];
        if lookups.is_empty() {
            continue;
        }
        println!("  -- lookups ({}) --", lookups.len());

        let empty: Vec<Val> = Vec::new();
        for (lk, lookup) in lookups.iter().enumerate() {
            let kind_str = match &lookup.kind {
                Kind::Local => "Local".to_string(),
                Kind::Global(name) => format!("Global({name})"),
            };
            println!(
                "    lookup #{lk}  kind={kind_str}  tuples_per_row={}",
                lookup.element_exprs.len()
            );

            for r in 0..height {
                let next_r = (r + 1) % height;
                let main = RowWindow::from_two_rows(&main_rows[r], &main_rows[next_r]);
                let preprocessed = match &prep_rows {
                    Some(prep) => {
                        let pn = (r + 1) % prep.len();
                        RowWindow::from_two_rows(&prep[r], &prep[pn])
                    }
                    None => RowWindow::from_two_rows(&empty, &empty),
                };

                let builder = MiniBuilder {
                    main,
                    preprocessed,
                    public_values: &[],
                    row: r,
                    height,
                };

                for (t, exprs) in lookup.element_exprs.iter().enumerate() {
                    let mult: Val = symbolic_to_expr(&builder, &lookup.multiplicities_exprs[t]);
                    if mult.is_zero() {
                        continue;
                    }
                    let tuple: Vec<Val> = exprs
                        .iter()
                        .map(|e| symbolic_to_expr(&builder, e))
                        .collect();
                    println!(
                        "      row {r:>3}  t{t}  mult={}  tuple=({})",
                        mult,
                        fmt_row(&tuple),
                    );
                }
            }
        }
    }
}

/// Run p3-lookup's `check_lookups`. Panics with concrete (instance, lookup, row)
/// locations on any global-bus imbalance.
pub fn check_all_lookups(airs: &mut [LoquelAir], traces: &[RowMajorMatrix<Val>]) {
    let prep_traces: Vec<Option<RowMajorMatrix<Val>>> = airs
        .iter()
        .map(|a| <LoquelAir as BaseAir<Val>>::preprocessed_trace(a))
        .collect();
    let lookups_per_air: Vec<_> = airs
        .iter_mut()
        .map(|a| <LoquelAir as LookupAir<Val>>::get_lookups(a))
        .collect();
    let public_values: Vec<Vec<Val>> = vec![vec![]; airs.len()];

    let instances: Vec<LookupDebugInstance<'_, Val>> = (0..airs.len())
        .map(|i| LookupDebugInstance {
            main_trace: &traces[i],
            preprocessed_trace: &prep_traces[i],
            public_values: &public_values[i],
            lookups: &lookups_per_air[i],
            permutation_challenges: &[],
        })
        .collect();

    check_lookups(&instances);
}
