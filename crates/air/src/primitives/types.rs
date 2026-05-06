pub struct Byte<F> {
    value: F,
}

impl<F> Byte<F> {
    pub fn from_field_unchecked(value: F) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &F {
        &self.value
    }
}

#[repr(C)]
pub struct Unsigned32<F> {
    limbs: [Byte<F>; 4],
}

impl<F> Unsigned32<F> {
    pub fn from_limbs_unchecked(limbs: [Byte<F>; 4]) -> Self {
        Self { limbs }
    }

    pub fn limbs(&self) -> &[Byte<F>; 4] {
        &self.limbs
    }
}

pub struct Boolean<F> {
    value: F,
}

impl<F> Boolean<F> {
    pub fn from_field_unchecked(value: F) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &F {
        &self.value
    }
}
