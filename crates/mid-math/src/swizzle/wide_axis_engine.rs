// crates/mid-math/src/swizzle/wide_axis_engine.rs
//! Axis-swizzle for this crate's SoA "batch of N Vec3s" wide types
//! (`Vec3x4`, `Vec3x8`) — reordering which *register* holds x/y/z for all N
//! packed vectors at once.
//!
//! This is a separate, smaller engine from `engine.rs`, not a reuse of
//! `Vec3Swizzles`, because it genuinely can't share that trait's shape:
//! there's no `Vec2x4`/`Vec4x4` type for the narrowing/widening 2-letter and
//! 4-letter results `Vec3Swizzles` needs, and these types have no `new(x,y,z)`
//! matching `Vec3Swizzles`'s calling convention (they build via
//! `from_vec3s`/`splat`/etc. instead). So: same-width-only (all 27
//! permutations of x/y/z, dim=3), getters only. Every field here is `pub`
//! already (`Vec3x4`/`Vec3x8`'s own doc comments say so explicitly — "Fields
//! are public for advanced intrinsic use"), so construction is a plain
//! struct literal, not `::new(...)` — and unlike the scalar case, this needs
//! ZERO data movement: reordering which whole register is x/y/z is free,
//! there's no padding-lane risk, no shuffle-immediate to get wrong.
//!
//! No setters (`with_xy`-style): there's no natural 2-component carrier type
//! to receive a setter's `rhs` here (no `Vec2x4`), and inventing one
//! (a raw tuple, say) has no precedent elsewhere in this crate's API — left
//! out rather than guessed at.
//!
//! No `QuatX4` here, deliberately, matching this crate's scalar swizzle
//! scope: `Quat`/`DQuat` were never given `Vec4Swizzles` either (matches
//! glam's own scope too — it has no `QuatSwizzles` trait). Reordering a
//! quaternion's components isn't a meaningful "swizzle" — it doesn't
//! generally produce a valid rotation — so this stays Vec-family only, same
//! as every other file in `swizzle/`.

pub trait Vec3AxisSwizzle: Sized + Copy + Clone {
    #[must_use]
    fn xxx(self) -> Self;
    #[must_use]
    fn xxy(self) -> Self;
    #[must_use]
    fn xxz(self) -> Self;
    #[must_use]
    fn xyx(self) -> Self;
    #[must_use]
    fn xyy(self) -> Self;
    #[must_use]
    fn xyz(self) -> Self;
    #[must_use]
    fn xzx(self) -> Self;
    #[must_use]
    fn xzy(self) -> Self;
    #[must_use]
    fn xzz(self) -> Self;
    #[must_use]
    fn yxx(self) -> Self;
    #[must_use]
    fn yxy(self) -> Self;
    #[must_use]
    fn yxz(self) -> Self;
    #[must_use]
    fn yyx(self) -> Self;
    #[must_use]
    fn yyy(self) -> Self;
    #[must_use]
    fn yyz(self) -> Self;
    #[must_use]
    fn yzx(self) -> Self;
    #[must_use]
    fn yzy(self) -> Self;
    #[must_use]
    fn yzz(self) -> Self;
    #[must_use]
    fn zxx(self) -> Self;
    #[must_use]
    fn zxy(self) -> Self;
    #[must_use]
    fn zxz(self) -> Self;
    #[must_use]
    fn zyx(self) -> Self;
    #[must_use]
    fn zyy(self) -> Self;
    #[must_use]
    fn zyz(self) -> Self;
    #[must_use]
    fn zzx(self) -> Self;
    #[must_use]
    fn zzy(self) -> Self;
    #[must_use]
    fn zzz(self) -> Self;
}

#[macro_export]
macro_rules! impl_vec3_axis_swizzle {
    ($Self:ty) => {
        impl $crate::swizzle::wide_axis_engine::Vec3AxisSwizzle for $Self {
            #[inline(always)]
            fn xxx(self) -> Self { Self { x: self.x, y: self.x, z: self.x } }
            #[inline(always)]
            fn xxy(self) -> Self { Self { x: self.x, y: self.x, z: self.y } }
            #[inline(always)]
            fn xxz(self) -> Self { Self { x: self.x, y: self.x, z: self.z } }
            #[inline(always)]
            fn xyx(self) -> Self { Self { x: self.x, y: self.y, z: self.x } }
            #[inline(always)]
            fn xyy(self) -> Self { Self { x: self.x, y: self.y, z: self.y } }
            #[inline(always)]
            fn xyz(self) -> Self { Self { x: self.x, y: self.y, z: self.z } }
            #[inline(always)]
            fn xzx(self) -> Self { Self { x: self.x, y: self.z, z: self.x } }
            #[inline(always)]
            fn xzy(self) -> Self { Self { x: self.x, y: self.z, z: self.y } }
            #[inline(always)]
            fn xzz(self) -> Self { Self { x: self.x, y: self.z, z: self.z } }
            #[inline(always)]
            fn yxx(self) -> Self { Self { x: self.y, y: self.x, z: self.x } }
            #[inline(always)]
            fn yxy(self) -> Self { Self { x: self.y, y: self.x, z: self.y } }
            #[inline(always)]
            fn yxz(self) -> Self { Self { x: self.y, y: self.x, z: self.z } }
            #[inline(always)]
            fn yyx(self) -> Self { Self { x: self.y, y: self.y, z: self.x } }
            #[inline(always)]
            fn yyy(self) -> Self { Self { x: self.y, y: self.y, z: self.y } }
            #[inline(always)]
            fn yyz(self) -> Self { Self { x: self.y, y: self.y, z: self.z } }
            #[inline(always)]
            fn yzx(self) -> Self { Self { x: self.y, y: self.z, z: self.x } }
            #[inline(always)]
            fn yzy(self) -> Self { Self { x: self.y, y: self.z, z: self.y } }
            #[inline(always)]
            fn yzz(self) -> Self { Self { x: self.y, y: self.z, z: self.z } }
            #[inline(always)]
            fn zxx(self) -> Self { Self { x: self.z, y: self.x, z: self.x } }
            #[inline(always)]
            fn zxy(self) -> Self { Self { x: self.z, y: self.x, z: self.y } }
            #[inline(always)]
            fn zxz(self) -> Self { Self { x: self.z, y: self.x, z: self.z } }
            #[inline(always)]
            fn zyx(self) -> Self { Self { x: self.z, y: self.y, z: self.x } }
            #[inline(always)]
            fn zyy(self) -> Self { Self { x: self.z, y: self.y, z: self.y } }
            #[inline(always)]
            fn zyz(self) -> Self { Self { x: self.z, y: self.y, z: self.z } }
            #[inline(always)]
            fn zzx(self) -> Self { Self { x: self.z, y: self.z, z: self.x } }
            #[inline(always)]
            fn zzy(self) -> Self { Self { x: self.z, y: self.z, z: self.y } }
            #[inline(always)]
            fn zzz(self) -> Self { Self { x: self.z, y: self.z, z: self.z } }
        }
    };
}
