// crates/mid-math/src/swizzle/wide_lane_engine.rs
//! Lane-shuffle for this crate's opaque, single-register wide types
//! (`f32x4`, `i32x4`/`u32x4`, `i16x8`/`u16x8`/`i32x8`/`u32x8`,
//! `i8x16`/`u8x16`/`i16x16`/`u16x16`, `i8x32`/`u8x32`) — these wrap one SIMD
//! register with no named fields at all (`i32x4(pub(crate) __m128i)`), so
//! there's no x/y/z/w axis meaning the way `Vec3x4`/`QuatX4` have. A
//! "swizzle" here is a lane permutation instead: reorder which lane holds
//! what value.
//!
//! One trait per width (`LaneShuffle4`/`8`/`16`/`32`) rather than one generic
//! trait taking a slice — matches this crate's own preference for
//! compile-time-checked array lengths over a runtime length check (same
//! reasoning `Vec2Swizzles`/`Vec3Swizzles`/`Vec4Swizzles` are three traits,
//! not one generic over width).
//!
//! Every method here goes through `to_array()` + `from_array()` — confirmed
//! present, with matching names, on every wide/int and wide/float type
//! across every backend before writing this — never a raw shuffle
//! intrinsic. Same reasoning as `engine.rs`'s getters: no compiler available
//! to verify a shuffle-immediate encoding, and this path reuses each type's
//! own already-correct round-trip instead of risking a new one. `get()`
//! already panics on an out-of-range lane in every one of these types, so
//! `shuffle()` panicking on an out-of-range index is the existing
//! convention, not a new one.

// --- LaneShuffle4 ---
pub trait LaneShuffle4: Sized + Copy {
    /// `result.get(i) == self.get(indices[i])` for every `i`. Panics if any
    /// index is out of range, same as `get()` already does.
    #[must_use]
    fn shuffle(self, indices: [usize; 4]) -> Self;
    #[must_use]
    fn reverse_lanes(self) -> Self;
    #[must_use]
    fn splat_lane(self, lane: usize) -> Self;
    #[must_use]
    fn rotate_left(self, n: usize) -> Self;
    #[must_use]
    fn rotate_right(self, n: usize) -> Self;
}

// --- LaneShuffle8 ---
pub trait LaneShuffle8: Sized + Copy {
    /// `result.get(i) == self.get(indices[i])` for every `i`. Panics if any
    /// index is out of range, same as `get()` already does.
    #[must_use]
    fn shuffle(self, indices: [usize; 8]) -> Self;
    #[must_use]
    fn reverse_lanes(self) -> Self;
    #[must_use]
    fn splat_lane(self, lane: usize) -> Self;
    #[must_use]
    fn rotate_left(self, n: usize) -> Self;
    #[must_use]
    fn rotate_right(self, n: usize) -> Self;
}

// --- LaneShuffle16 ---
pub trait LaneShuffle16: Sized + Copy {
    /// `result.get(i) == self.get(indices[i])` for every `i`. Panics if any
    /// index is out of range, same as `get()` already does.
    #[must_use]
    fn shuffle(self, indices: [usize; 16]) -> Self;
    #[must_use]
    fn reverse_lanes(self) -> Self;
    #[must_use]
    fn splat_lane(self, lane: usize) -> Self;
    #[must_use]
    fn rotate_left(self, n: usize) -> Self;
    #[must_use]
    fn rotate_right(self, n: usize) -> Self;
}

// --- LaneShuffle32 ---
pub trait LaneShuffle32: Sized + Copy {
    /// `result.get(i) == self.get(indices[i])` for every `i`. Panics if any
    /// index is out of range, same as `get()` already does.
    #[must_use]
    fn shuffle(self, indices: [usize; 32]) -> Self;
    #[must_use]
    fn reverse_lanes(self) -> Self;
    #[must_use]
    fn splat_lane(self, lane: usize) -> Self;
    #[must_use]
    fn rotate_left(self, n: usize) -> Self;
    #[must_use]
    fn rotate_right(self, n: usize) -> Self;
}

// --- impl_lane_shuffle4! ---
#[macro_export]
macro_rules! impl_lane_shuffle4 {
    ($Self:ty) => {
        impl $crate::swizzle::wide_lane_engine::LaneShuffle4 for $Self {
            #[inline(always)]
            fn shuffle(self, indices: [usize; 4]) -> Self {
                let a = self.to_array();
                Self::from_array([a[indices[0]], a[indices[1]], a[indices[2]], a[indices[3]]])
            }
            #[inline(always)]
            fn reverse_lanes(self) -> Self {
                let mut a = self.to_array();
                a.reverse();
                Self::from_array(a)
            }
            #[inline(always)]
            fn splat_lane(self, lane: usize) -> Self {
                Self::from_array([self.get(lane); 4])
            }
            #[inline(always)]
            fn rotate_left(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_left(n % 4);
                Self::from_array(a)
            }
            #[inline(always)]
            fn rotate_right(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_right(n % 4);
                Self::from_array(a)
            }
        }
    };
}

// --- impl_lane_shuffle8! ---
#[macro_export]
macro_rules! impl_lane_shuffle8 {
    ($Self:ty) => {
        impl $crate::swizzle::wide_lane_engine::LaneShuffle8 for $Self {
            #[inline(always)]
            fn shuffle(self, indices: [usize; 8]) -> Self {
                let a = self.to_array();
                Self::from_array([a[indices[0]], a[indices[1]], a[indices[2]], a[indices[3]], a[indices[4]], a[indices[5]], a[indices[6]], a[indices[7]]])
            }
            #[inline(always)]
            fn reverse_lanes(self) -> Self {
                let mut a = self.to_array();
                a.reverse();
                Self::from_array(a)
            }
            #[inline(always)]
            fn splat_lane(self, lane: usize) -> Self {
                Self::from_array([self.get(lane); 8])
            }
            #[inline(always)]
            fn rotate_left(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_left(n % 8);
                Self::from_array(a)
            }
            #[inline(always)]
            fn rotate_right(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_right(n % 8);
                Self::from_array(a)
            }
        }
    };
}

// --- impl_lane_shuffle16! ---
#[macro_export]
macro_rules! impl_lane_shuffle16 {
    ($Self:ty) => {
        impl $crate::swizzle::wide_lane_engine::LaneShuffle16 for $Self {
            #[inline(always)]
            fn shuffle(self, indices: [usize; 16]) -> Self {
                let a = self.to_array();
                Self::from_array([a[indices[0]], a[indices[1]], a[indices[2]], a[indices[3]], a[indices[4]], a[indices[5]], a[indices[6]], a[indices[7]], a[indices[8]], a[indices[9]], a[indices[10]], a[indices[11]], a[indices[12]], a[indices[13]], a[indices[14]], a[indices[15]]])
            }
            #[inline(always)]
            fn reverse_lanes(self) -> Self {
                let mut a = self.to_array();
                a.reverse();
                Self::from_array(a)
            }
            #[inline(always)]
            fn splat_lane(self, lane: usize) -> Self {
                Self::from_array([self.get(lane); 16])
            }
            #[inline(always)]
            fn rotate_left(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_left(n % 16);
                Self::from_array(a)
            }
            #[inline(always)]
            fn rotate_right(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_right(n % 16);
                Self::from_array(a)
            }
        }
    };
}

// --- impl_lane_shuffle32! ---
#[macro_export]
macro_rules! impl_lane_shuffle32 {
    ($Self:ty) => {
        impl $crate::swizzle::wide_lane_engine::LaneShuffle32 for $Self {
            #[inline(always)]
            fn shuffle(self, indices: [usize; 32]) -> Self {
                let a = self.to_array();
                Self::from_array([a[indices[0]], a[indices[1]], a[indices[2]], a[indices[3]], a[indices[4]], a[indices[5]], a[indices[6]], a[indices[7]], a[indices[8]], a[indices[9]], a[indices[10]], a[indices[11]], a[indices[12]], a[indices[13]], a[indices[14]], a[indices[15]], a[indices[16]], a[indices[17]], a[indices[18]], a[indices[19]], a[indices[20]], a[indices[21]], a[indices[22]], a[indices[23]], a[indices[24]], a[indices[25]], a[indices[26]], a[indices[27]], a[indices[28]], a[indices[29]], a[indices[30]], a[indices[31]]])
            }
            #[inline(always)]
            fn reverse_lanes(self) -> Self {
                let mut a = self.to_array();
                a.reverse();
                Self::from_array(a)
            }
            #[inline(always)]
            fn splat_lane(self, lane: usize) -> Self {
                Self::from_array([self.get(lane); 32])
            }
            #[inline(always)]
            fn rotate_left(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_left(n % 32);
                Self::from_array(a)
            }
            #[inline(always)]
            fn rotate_right(self, n: usize) -> Self {
                let mut a = self.to_array();
                a.rotate_right(n % 32);
                Self::from_array(a)
            }
        }
    };
}

