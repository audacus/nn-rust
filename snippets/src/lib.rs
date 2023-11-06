use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

pub fn random_seed() -> u64 {
    RandomState::new().build_hasher().finish()
}

pub fn random_numbers() -> impl Iterator<Item = u64> {
    let mut random = random_seed();
    std::iter::repeat_with(move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_numbers_works() {
        let result = random_numbers().next();
        assert_ne!(result, None)
    }
}
