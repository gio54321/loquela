use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
    vec,
};

use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, SymbolicAirBuilder, SymbolicVariable, WindowAccess,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::{Direction, Kind, Lookup, LookupAir};

#[repr(C)]
pub struct MemoryColumns<F> {
    pub memory_type: F, // 0 -> registers, 1 -> memory
    pub address: [F; 4],
    pub timestamp: F,
    pub read: [F; 4],
    pub write: [F; 4],

    pub is_memory_type_equal: F,
    pub is_timestamp_equal: F,
    pub is_address_equal: F,

    pub is_padding: F,
}

pub const NUM_COLS: usize = size_of::<MemoryColumns<u8>>();

impl<T> Borrow<MemoryColumns<T>> for [T] {
    fn borrow(&self) -> &MemoryColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<MemoryColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<MemoryColumns<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut MemoryColumns<T> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<MemoryColumns<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

#[derive(Clone)]
pub struct MemoryAir {
    num_lookups: usize,
}

impl MemoryAir {
    pub fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F> BaseAir<F> for MemoryAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for MemoryAir
where
    AB::MainWindow: WindowAccess<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &MemoryColumns<AB::Var> = main.current_slice().borrow();
        let next: &MemoryColumns<AB::Var> = main.next_slice().borrow();

        builder.assert_bool(local.is_padding);
        builder
            .when_transition()
            .when(local.is_padding)
            .assert_one(next.is_padding);

        // constrain is_memory_type_equal to be 1 iff we are comparing the same memory type
        builder.assert_bool(local.is_memory_type_equal);

        builder
            .when_transition()
            .when(local.is_memory_type_equal)
            .assert_eq(next.memory_type.clone(), local.memory_type.clone());

        let s = local.memory_type.clone() + next.memory_type.clone();
        builder
            .when_transition()
            .when(AB::Expr::ONE - local.is_memory_type_equal)
            .assert_zero((s.clone() - AB::Expr::ONE) * (s - AB::Expr::TWO));

        // constrain memory_type to be sorted, so once the type is 1 we cannot go back to 0
        builder
            .when_transition()
            .when(local.memory_type.clone())
            .assert_one(next.memory_type.clone());

        // we want memory operations in the following order:
        // - first by memory type (registers first, then memory)
        // - then by address
        // - then by timestamp
        // this is enforced by looking up into the u32 lt

        // on the first row and when the address changes the read value should be zero
        for i in 0..4 {
            builder.when_first_row().assert_zero(local.read[i].clone());
            builder
                .when_transition()
                .when(AB::Expr::ONE - local.is_address_equal.clone())
                .assert_zero(next.read[i].clone());
        }

        // if the address is the same, then the next read value should be the same as the current write value
        for i in 0..4 {
            builder
                .when_transition()
                .when(local.is_address_equal.clone())
                .assert_eq(next.read[i].clone(), local.write[i].clone());
        }
    }
}

impl<F: Field> LookupAir<F> for MemoryAir {
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
        let symbolic_main_local: &MemoryColumns<SymbolicVariable<F>> =
            symbolic_main.current_slice().borrow();
        let symbolic_main_next: &MemoryColumns<SymbolicVariable<F>> =
            symbolic_main.next_slice().borrow();

        let mut lookups = Vec::new();

        // Sortedness only matters between two real rows. Gate every cross-AIR
        // sortedness send with (1 - next.is_padding): once we transition into
        // padding, the addresses/timestamps are all zero and would otherwise
        // emit spurious lookups that the U32Lt / TimestampLt AIRs don't receive.
        let real_transition: p3_air::SymbolicExpression<F> =
            p3_air::SymbolicExpression::from(F::ONE) - symbolic_main_next.is_padding.clone();

        // address must always be sorted unless the memory type is different, so we send with mult is_memory_type_equal
        lookups.push(self.register_lookup(
            Kind::Global(String::from("u32_lt")),
            &vec![(symbolic_main_local.address.into_iter().chain(
                symbolic_main_next.address.into_iter()).chain(
                        once(symbolic_main_local.is_address_equal.clone()))
                    .map(Into::into)
                    .collect::<Vec<_>>(),
            (Into::<p3_air::SymbolicExpression<F>>::into(symbolic_main_local.is_memory_type_equal.clone())
                * real_transition.clone()).into(),
            Direction::Send,
        )],
        ));

        // timestamp must be sorted unless the address is different, so we send with mult is_address_equal
        lookups.push(self.register_lookup(
            Kind::Global(String::from("timestamp_lt")),
            &vec![(
                vec![
                    symbolic_main_local.timestamp.clone().into(),
                    symbolic_main_next.timestamp.clone().into(),
                    symbolic_main_local.is_timestamp_equal.clone().into(),
                ],
                (Into::<p3_air::SymbolicExpression<F>>::into(symbolic_main_local.is_address_equal.clone())
                    * real_transition).into(),
                Direction::Send,
            )],
        ));

        lookups.push(self.register_lookup(
            Kind::Global(String::from("memory")),
            &vec![(
                    once(symbolic_main_local.timestamp.clone())
                        .chain(once(symbolic_main_local.memory_type))
                        .chain(symbolic_main_local.address.into_iter())
                        .chain(symbolic_main_local.read.into_iter())
                        .chain(symbolic_main_local.write.into_iter())
                        .map(Into::into)
                        .collect::<Vec<_>>(),
                    (Into::<p3_air::SymbolicExpression<F>>::into(F::ONE)
                        - symbolic_main_local.is_padding.clone())
                    .into(),
                    Direction::Receive,
                )],
        ));

        lookups
    }
}
