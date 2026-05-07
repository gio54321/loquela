use std::{borrow::Borrow, borrow::BorrowMut, iter::once, vec};

use crate::primitives::u32_ops::u32_inc;
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{integers::QuotientMap, Field};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

#[repr(C)]
pub struct ProgramColumns<F> {
    pub address: [F; 4],
    pub value: F,
    pub mult: F,
    pub inc_carries: [F; 4],
}

pub const NUM_MEMORY_COLS: usize = size_of::<ProgramColumns<u8>>();

impl<T> Borrow<ProgramColumns<T>> for [T] {
    fn borrow(&self) -> &ProgramColumns<T> {
        debug_assert_eq!(self.len(), NUM_MEMORY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<ProgramColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<ProgramColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut ProgramColumns<T> {
        debug_assert_eq!(self.len(), NUM_MEMORY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<ProgramColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

pub struct ProgramAir {
    num_lookups: usize,
}

impl<F> BaseAir<F> for ProgramAir {
    fn width(&self) -> usize {
        NUM_MEMORY_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for ProgramAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
    AB::F: QuotientMap<u32>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ProgramColumns<AB::Var> = main.current_slice().borrow();
        let next: &ProgramColumns<AB::Var> = main.next_slice().borrow();

        // todo: export commitment of the program
        for limb in local.address.iter() {
            builder.when_first_row().assert_zero(limb.clone());
        }

        let mut t = builder.when_transition();
        u32_inc(&mut t, &local.address, &next.address, &local.inc_carries);
    }
}

impl<F: Field> LookupAir<F> for ProgramAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;

        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: BaseAir::<F>::width(self),
            ..Default::default()
        });
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local: &ProgramColumns<SymbolicVariable<F>> =
            symbolic_main.current_slice().borrow();

        vec![self.register_lookup(
            Kind::Global(String::from("program")),
            &vec![(
                symbolic_main_local
                    .address
                    .into_iter()
                    .chain(once(symbolic_main_local.value))
                    .map(Into::into)
                    .collect(),
                symbolic_main_local.mult.into(),
                Direction::Receive,
            )],
        )]
    }
}
