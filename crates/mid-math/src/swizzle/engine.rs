// crates/mid-math/src/swizzle/engine.rs
//! The swizzle "engine": `Vec2Swizzles`/`Vec3Swizzles`/`Vec4Swizzles` trait
//! definitions and the `impl_vec{2,3,4}_swizzle!` macros that implement them.
//!
//! This file is numeric-family-agnostic — nothing here mentions f32, f64, or
//! any int type. Each macro only needs `self.x`/`self.y`/`self.z`/`self.w`
//! (works whether the concrete type has real pub fields or reaches them via
//! `Deref`, see `crate::deref`) and the target type's own `new()`. Per-family
//! invocations live in sibling files (`f32.rs`, `f64.rs`, ...) — see
//! `swizzle/mod.rs`.
//!
//! Getters: read-only permutation accessors covering every same-width,
//! narrowing, and widening combination (`.xy()`, `.xzy()`, `.wzyx()`, ...).
//! Setters: `.with_xy(rhs)`/`.with_xyz(rhs)`-style same-or-narrower-width
//! replacement, returning a new value with the named components replaced
//! from `rhs` and everything else left as-is. No `with_` setter widens (no
//! 4-letter `with_` on `Vec4Swizzles`) — replacing every component isn't a
//! partial update, just build a new value with `::new()` instead.
//!
//! Perf note: every getter goes through `<Output>::new(...)`, never a single
//! in-register SIMD shuffle the way glam's SSE2/NEON/WASM/coresimd backends
//! do for same-width swizzles. Real, well-defined follow-up (see
//! `crates/mid-math/README.md`), deliberately deferred: needs each backend's
//! shuffle-immediate encoding *and* `Vec3`'s padding-lane behavior verified
//! with a compiler, across several hundred call sites per backend, which
//! this pass couldn't do safely with none available.

// --- Vec2Swizzles ---
pub trait Vec2Swizzles: Sized + Copy + Clone {
    type Vec3;
    type Vec4;

    #[must_use]
    fn xy(self) -> Self;

    #[must_use]
    fn xx(self) -> Self;
    #[must_use]
    fn yx(self) -> Self;
    #[must_use]
    fn yy(self) -> Self;

    #[must_use]
    fn xxx(self) -> Self::Vec3;
    #[must_use]
    fn xxy(self) -> Self::Vec3;
    #[must_use]
    fn xyx(self) -> Self::Vec3;
    #[must_use]
    fn xyy(self) -> Self::Vec3;
    #[must_use]
    fn yxx(self) -> Self::Vec3;
    #[must_use]
    fn yxy(self) -> Self::Vec3;
    #[must_use]
    fn yyx(self) -> Self::Vec3;
    #[must_use]
    fn yyy(self) -> Self::Vec3;

    #[must_use]
    fn xxxx(self) -> Self::Vec4;
    #[must_use]
    fn xxxy(self) -> Self::Vec4;
    #[must_use]
    fn xxyx(self) -> Self::Vec4;
    #[must_use]
    fn xxyy(self) -> Self::Vec4;
    #[must_use]
    fn xyxx(self) -> Self::Vec4;
    #[must_use]
    fn xyxy(self) -> Self::Vec4;
    #[must_use]
    fn xyyx(self) -> Self::Vec4;
    #[must_use]
    fn xyyy(self) -> Self::Vec4;
    #[must_use]
    fn yxxx(self) -> Self::Vec4;
    #[must_use]
    fn yxxy(self) -> Self::Vec4;
    #[must_use]
    fn yxyx(self) -> Self::Vec4;
    #[must_use]
    fn yxyy(self) -> Self::Vec4;
    #[must_use]
    fn yyxx(self) -> Self::Vec4;
    #[must_use]
    fn yyxy(self) -> Self::Vec4;
    #[must_use]
    fn yyyx(self) -> Self::Vec4;
    #[must_use]
    fn yyyy(self) -> Self::Vec4;

}

// --- Vec3Swizzles ---
pub trait Vec3Swizzles: Sized + Copy + Clone {
    type Vec2;
    type Vec4;

    #[must_use]
    fn xyz(self) -> Self;

    #[must_use]
    fn xx(self) -> Self::Vec2;
    #[must_use]
    fn xy(self) -> Self::Vec2;
    #[must_use]
    fn with_xy(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn xz(self) -> Self::Vec2;
    #[must_use]
    fn with_xz(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn yx(self) -> Self::Vec2;
    #[must_use]
    fn with_yx(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn yy(self) -> Self::Vec2;
    #[must_use]
    fn yz(self) -> Self::Vec2;
    #[must_use]
    fn with_yz(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn zx(self) -> Self::Vec2;
    #[must_use]
    fn with_zx(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn zy(self) -> Self::Vec2;
    #[must_use]
    fn with_zy(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn zz(self) -> Self::Vec2;

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

    #[must_use]
    fn xxxx(self) -> Self::Vec4;
    #[must_use]
    fn xxxy(self) -> Self::Vec4;
    #[must_use]
    fn xxxz(self) -> Self::Vec4;
    #[must_use]
    fn xxyx(self) -> Self::Vec4;
    #[must_use]
    fn xxyy(self) -> Self::Vec4;
    #[must_use]
    fn xxyz(self) -> Self::Vec4;
    #[must_use]
    fn xxzx(self) -> Self::Vec4;
    #[must_use]
    fn xxzy(self) -> Self::Vec4;
    #[must_use]
    fn xxzz(self) -> Self::Vec4;
    #[must_use]
    fn xyxx(self) -> Self::Vec4;
    #[must_use]
    fn xyxy(self) -> Self::Vec4;
    #[must_use]
    fn xyxz(self) -> Self::Vec4;
    #[must_use]
    fn xyyx(self) -> Self::Vec4;
    #[must_use]
    fn xyyy(self) -> Self::Vec4;
    #[must_use]
    fn xyyz(self) -> Self::Vec4;
    #[must_use]
    fn xyzx(self) -> Self::Vec4;
    #[must_use]
    fn xyzy(self) -> Self::Vec4;
    #[must_use]
    fn xyzz(self) -> Self::Vec4;
    #[must_use]
    fn xzxx(self) -> Self::Vec4;
    #[must_use]
    fn xzxy(self) -> Self::Vec4;
    #[must_use]
    fn xzxz(self) -> Self::Vec4;
    #[must_use]
    fn xzyx(self) -> Self::Vec4;
    #[must_use]
    fn xzyy(self) -> Self::Vec4;
    #[must_use]
    fn xzyz(self) -> Self::Vec4;
    #[must_use]
    fn xzzx(self) -> Self::Vec4;
    #[must_use]
    fn xzzy(self) -> Self::Vec4;
    #[must_use]
    fn xzzz(self) -> Self::Vec4;
    #[must_use]
    fn yxxx(self) -> Self::Vec4;
    #[must_use]
    fn yxxy(self) -> Self::Vec4;
    #[must_use]
    fn yxxz(self) -> Self::Vec4;
    #[must_use]
    fn yxyx(self) -> Self::Vec4;
    #[must_use]
    fn yxyy(self) -> Self::Vec4;
    #[must_use]
    fn yxyz(self) -> Self::Vec4;
    #[must_use]
    fn yxzx(self) -> Self::Vec4;
    #[must_use]
    fn yxzy(self) -> Self::Vec4;
    #[must_use]
    fn yxzz(self) -> Self::Vec4;
    #[must_use]
    fn yyxx(self) -> Self::Vec4;
    #[must_use]
    fn yyxy(self) -> Self::Vec4;
    #[must_use]
    fn yyxz(self) -> Self::Vec4;
    #[must_use]
    fn yyyx(self) -> Self::Vec4;
    #[must_use]
    fn yyyy(self) -> Self::Vec4;
    #[must_use]
    fn yyyz(self) -> Self::Vec4;
    #[must_use]
    fn yyzx(self) -> Self::Vec4;
    #[must_use]
    fn yyzy(self) -> Self::Vec4;
    #[must_use]
    fn yyzz(self) -> Self::Vec4;
    #[must_use]
    fn yzxx(self) -> Self::Vec4;
    #[must_use]
    fn yzxy(self) -> Self::Vec4;
    #[must_use]
    fn yzxz(self) -> Self::Vec4;
    #[must_use]
    fn yzyx(self) -> Self::Vec4;
    #[must_use]
    fn yzyy(self) -> Self::Vec4;
    #[must_use]
    fn yzyz(self) -> Self::Vec4;
    #[must_use]
    fn yzzx(self) -> Self::Vec4;
    #[must_use]
    fn yzzy(self) -> Self::Vec4;
    #[must_use]
    fn yzzz(self) -> Self::Vec4;
    #[must_use]
    fn zxxx(self) -> Self::Vec4;
    #[must_use]
    fn zxxy(self) -> Self::Vec4;
    #[must_use]
    fn zxxz(self) -> Self::Vec4;
    #[must_use]
    fn zxyx(self) -> Self::Vec4;
    #[must_use]
    fn zxyy(self) -> Self::Vec4;
    #[must_use]
    fn zxyz(self) -> Self::Vec4;
    #[must_use]
    fn zxzx(self) -> Self::Vec4;
    #[must_use]
    fn zxzy(self) -> Self::Vec4;
    #[must_use]
    fn zxzz(self) -> Self::Vec4;
    #[must_use]
    fn zyxx(self) -> Self::Vec4;
    #[must_use]
    fn zyxy(self) -> Self::Vec4;
    #[must_use]
    fn zyxz(self) -> Self::Vec4;
    #[must_use]
    fn zyyx(self) -> Self::Vec4;
    #[must_use]
    fn zyyy(self) -> Self::Vec4;
    #[must_use]
    fn zyyz(self) -> Self::Vec4;
    #[must_use]
    fn zyzx(self) -> Self::Vec4;
    #[must_use]
    fn zyzy(self) -> Self::Vec4;
    #[must_use]
    fn zyzz(self) -> Self::Vec4;
    #[must_use]
    fn zzxx(self) -> Self::Vec4;
    #[must_use]
    fn zzxy(self) -> Self::Vec4;
    #[must_use]
    fn zzxz(self) -> Self::Vec4;
    #[must_use]
    fn zzyx(self) -> Self::Vec4;
    #[must_use]
    fn zzyy(self) -> Self::Vec4;
    #[must_use]
    fn zzyz(self) -> Self::Vec4;
    #[must_use]
    fn zzzx(self) -> Self::Vec4;
    #[must_use]
    fn zzzy(self) -> Self::Vec4;
    #[must_use]
    fn zzzz(self) -> Self::Vec4;

}

// --- Vec4Swizzles ---
pub trait Vec4Swizzles: Sized + Copy + Clone {
    type Vec2;
    type Vec3;

    #[must_use]
    fn xyzw(self) -> Self;

    #[must_use]
    fn xx(self) -> Self::Vec2;
    #[must_use]
    fn xy(self) -> Self::Vec2;
    #[must_use]
    fn with_xy(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn xz(self) -> Self::Vec2;
    #[must_use]
    fn with_xz(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn xw(self) -> Self::Vec2;
    #[must_use]
    fn with_xw(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn yx(self) -> Self::Vec2;
    #[must_use]
    fn with_yx(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn yy(self) -> Self::Vec2;
    #[must_use]
    fn yz(self) -> Self::Vec2;
    #[must_use]
    fn with_yz(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn yw(self) -> Self::Vec2;
    #[must_use]
    fn with_yw(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn zx(self) -> Self::Vec2;
    #[must_use]
    fn with_zx(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn zy(self) -> Self::Vec2;
    #[must_use]
    fn with_zy(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn zz(self) -> Self::Vec2;
    #[must_use]
    fn zw(self) -> Self::Vec2;
    #[must_use]
    fn with_zw(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn wx(self) -> Self::Vec2;
    #[must_use]
    fn with_wx(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn wy(self) -> Self::Vec2;
    #[must_use]
    fn with_wy(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn wz(self) -> Self::Vec2;
    #[must_use]
    fn with_wz(self, rhs: Self::Vec2) -> Self;
    #[must_use]
    fn ww(self) -> Self::Vec2;

    #[must_use]
    fn xxx(self) -> Self::Vec3;
    #[must_use]
    fn xxy(self) -> Self::Vec3;
    #[must_use]
    fn xxz(self) -> Self::Vec3;
    #[must_use]
    fn xxw(self) -> Self::Vec3;
    #[must_use]
    fn xyx(self) -> Self::Vec3;
    #[must_use]
    fn xyy(self) -> Self::Vec3;
    #[must_use]
    fn xyz(self) -> Self::Vec3;
    #[must_use]
    fn with_xyz(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn xyw(self) -> Self::Vec3;
    #[must_use]
    fn with_xyw(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn xzx(self) -> Self::Vec3;
    #[must_use]
    fn xzy(self) -> Self::Vec3;
    #[must_use]
    fn with_xzy(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn xzz(self) -> Self::Vec3;
    #[must_use]
    fn xzw(self) -> Self::Vec3;
    #[must_use]
    fn with_xzw(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn xwx(self) -> Self::Vec3;
    #[must_use]
    fn xwy(self) -> Self::Vec3;
    #[must_use]
    fn with_xwy(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn xwz(self) -> Self::Vec3;
    #[must_use]
    fn with_xwz(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn xww(self) -> Self::Vec3;
    #[must_use]
    fn yxx(self) -> Self::Vec3;
    #[must_use]
    fn yxy(self) -> Self::Vec3;
    #[must_use]
    fn yxz(self) -> Self::Vec3;
    #[must_use]
    fn with_yxz(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn yxw(self) -> Self::Vec3;
    #[must_use]
    fn with_yxw(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn yyx(self) -> Self::Vec3;
    #[must_use]
    fn yyy(self) -> Self::Vec3;
    #[must_use]
    fn yyz(self) -> Self::Vec3;
    #[must_use]
    fn yyw(self) -> Self::Vec3;
    #[must_use]
    fn yzx(self) -> Self::Vec3;
    #[must_use]
    fn with_yzx(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn yzy(self) -> Self::Vec3;
    #[must_use]
    fn yzz(self) -> Self::Vec3;
    #[must_use]
    fn yzw(self) -> Self::Vec3;
    #[must_use]
    fn with_yzw(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn ywx(self) -> Self::Vec3;
    #[must_use]
    fn with_ywx(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn ywy(self) -> Self::Vec3;
    #[must_use]
    fn ywz(self) -> Self::Vec3;
    #[must_use]
    fn with_ywz(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn yww(self) -> Self::Vec3;
    #[must_use]
    fn zxx(self) -> Self::Vec3;
    #[must_use]
    fn zxy(self) -> Self::Vec3;
    #[must_use]
    fn with_zxy(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn zxz(self) -> Self::Vec3;
    #[must_use]
    fn zxw(self) -> Self::Vec3;
    #[must_use]
    fn with_zxw(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn zyx(self) -> Self::Vec3;
    #[must_use]
    fn with_zyx(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn zyy(self) -> Self::Vec3;
    #[must_use]
    fn zyz(self) -> Self::Vec3;
    #[must_use]
    fn zyw(self) -> Self::Vec3;
    #[must_use]
    fn with_zyw(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn zzx(self) -> Self::Vec3;
    #[must_use]
    fn zzy(self) -> Self::Vec3;
    #[must_use]
    fn zzz(self) -> Self::Vec3;
    #[must_use]
    fn zzw(self) -> Self::Vec3;
    #[must_use]
    fn zwx(self) -> Self::Vec3;
    #[must_use]
    fn with_zwx(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn zwy(self) -> Self::Vec3;
    #[must_use]
    fn with_zwy(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn zwz(self) -> Self::Vec3;
    #[must_use]
    fn zww(self) -> Self::Vec3;
    #[must_use]
    fn wxx(self) -> Self::Vec3;
    #[must_use]
    fn wxy(self) -> Self::Vec3;
    #[must_use]
    fn with_wxy(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn wxz(self) -> Self::Vec3;
    #[must_use]
    fn with_wxz(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn wxw(self) -> Self::Vec3;
    #[must_use]
    fn wyx(self) -> Self::Vec3;
    #[must_use]
    fn with_wyx(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn wyy(self) -> Self::Vec3;
    #[must_use]
    fn wyz(self) -> Self::Vec3;
    #[must_use]
    fn with_wyz(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn wyw(self) -> Self::Vec3;
    #[must_use]
    fn wzx(self) -> Self::Vec3;
    #[must_use]
    fn with_wzx(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn wzy(self) -> Self::Vec3;
    #[must_use]
    fn with_wzy(self, rhs: Self::Vec3) -> Self;
    #[must_use]
    fn wzz(self) -> Self::Vec3;
    #[must_use]
    fn wzw(self) -> Self::Vec3;
    #[must_use]
    fn wwx(self) -> Self::Vec3;
    #[must_use]
    fn wwy(self) -> Self::Vec3;
    #[must_use]
    fn wwz(self) -> Self::Vec3;
    #[must_use]
    fn www(self) -> Self::Vec3;

    #[must_use]
    fn xxxx(self) -> Self;
    #[must_use]
    fn xxxy(self) -> Self;
    #[must_use]
    fn xxxz(self) -> Self;
    #[must_use]
    fn xxxw(self) -> Self;
    #[must_use]
    fn xxyx(self) -> Self;
    #[must_use]
    fn xxyy(self) -> Self;
    #[must_use]
    fn xxyz(self) -> Self;
    #[must_use]
    fn xxyw(self) -> Self;
    #[must_use]
    fn xxzx(self) -> Self;
    #[must_use]
    fn xxzy(self) -> Self;
    #[must_use]
    fn xxzz(self) -> Self;
    #[must_use]
    fn xxzw(self) -> Self;
    #[must_use]
    fn xxwx(self) -> Self;
    #[must_use]
    fn xxwy(self) -> Self;
    #[must_use]
    fn xxwz(self) -> Self;
    #[must_use]
    fn xxww(self) -> Self;
    #[must_use]
    fn xyxx(self) -> Self;
    #[must_use]
    fn xyxy(self) -> Self;
    #[must_use]
    fn xyxz(self) -> Self;
    #[must_use]
    fn xyxw(self) -> Self;
    #[must_use]
    fn xyyx(self) -> Self;
    #[must_use]
    fn xyyy(self) -> Self;
    #[must_use]
    fn xyyz(self) -> Self;
    #[must_use]
    fn xyyw(self) -> Self;
    #[must_use]
    fn xyzx(self) -> Self;
    #[must_use]
    fn xyzy(self) -> Self;
    #[must_use]
    fn xyzz(self) -> Self;
    #[must_use]
    fn xywx(self) -> Self;
    #[must_use]
    fn xywy(self) -> Self;
    #[must_use]
    fn xywz(self) -> Self;
    #[must_use]
    fn xyww(self) -> Self;
    #[must_use]
    fn xzxx(self) -> Self;
    #[must_use]
    fn xzxy(self) -> Self;
    #[must_use]
    fn xzxz(self) -> Self;
    #[must_use]
    fn xzxw(self) -> Self;
    #[must_use]
    fn xzyx(self) -> Self;
    #[must_use]
    fn xzyy(self) -> Self;
    #[must_use]
    fn xzyz(self) -> Self;
    #[must_use]
    fn xzyw(self) -> Self;
    #[must_use]
    fn xzzx(self) -> Self;
    #[must_use]
    fn xzzy(self) -> Self;
    #[must_use]
    fn xzzz(self) -> Self;
    #[must_use]
    fn xzzw(self) -> Self;
    #[must_use]
    fn xzwx(self) -> Self;
    #[must_use]
    fn xzwy(self) -> Self;
    #[must_use]
    fn xzwz(self) -> Self;
    #[must_use]
    fn xzww(self) -> Self;
    #[must_use]
    fn xwxx(self) -> Self;
    #[must_use]
    fn xwxy(self) -> Self;
    #[must_use]
    fn xwxz(self) -> Self;
    #[must_use]
    fn xwxw(self) -> Self;
    #[must_use]
    fn xwyx(self) -> Self;
    #[must_use]
    fn xwyy(self) -> Self;
    #[must_use]
    fn xwyz(self) -> Self;
    #[must_use]
    fn xwyw(self) -> Self;
    #[must_use]
    fn xwzx(self) -> Self;
    #[must_use]
    fn xwzy(self) -> Self;
    #[must_use]
    fn xwzz(self) -> Self;
    #[must_use]
    fn xwzw(self) -> Self;
    #[must_use]
    fn xwwx(self) -> Self;
    #[must_use]
    fn xwwy(self) -> Self;
    #[must_use]
    fn xwwz(self) -> Self;
    #[must_use]
    fn xwww(self) -> Self;
    #[must_use]
    fn yxxx(self) -> Self;
    #[must_use]
    fn yxxy(self) -> Self;
    #[must_use]
    fn yxxz(self) -> Self;
    #[must_use]
    fn yxxw(self) -> Self;
    #[must_use]
    fn yxyx(self) -> Self;
    #[must_use]
    fn yxyy(self) -> Self;
    #[must_use]
    fn yxyz(self) -> Self;
    #[must_use]
    fn yxyw(self) -> Self;
    #[must_use]
    fn yxzx(self) -> Self;
    #[must_use]
    fn yxzy(self) -> Self;
    #[must_use]
    fn yxzz(self) -> Self;
    #[must_use]
    fn yxzw(self) -> Self;
    #[must_use]
    fn yxwx(self) -> Self;
    #[must_use]
    fn yxwy(self) -> Self;
    #[must_use]
    fn yxwz(self) -> Self;
    #[must_use]
    fn yxww(self) -> Self;
    #[must_use]
    fn yyxx(self) -> Self;
    #[must_use]
    fn yyxy(self) -> Self;
    #[must_use]
    fn yyxz(self) -> Self;
    #[must_use]
    fn yyxw(self) -> Self;
    #[must_use]
    fn yyyx(self) -> Self;
    #[must_use]
    fn yyyy(self) -> Self;
    #[must_use]
    fn yyyz(self) -> Self;
    #[must_use]
    fn yyyw(self) -> Self;
    #[must_use]
    fn yyzx(self) -> Self;
    #[must_use]
    fn yyzy(self) -> Self;
    #[must_use]
    fn yyzz(self) -> Self;
    #[must_use]
    fn yyzw(self) -> Self;
    #[must_use]
    fn yywx(self) -> Self;
    #[must_use]
    fn yywy(self) -> Self;
    #[must_use]
    fn yywz(self) -> Self;
    #[must_use]
    fn yyww(self) -> Self;
    #[must_use]
    fn yzxx(self) -> Self;
    #[must_use]
    fn yzxy(self) -> Self;
    #[must_use]
    fn yzxz(self) -> Self;
    #[must_use]
    fn yzxw(self) -> Self;
    #[must_use]
    fn yzyx(self) -> Self;
    #[must_use]
    fn yzyy(self) -> Self;
    #[must_use]
    fn yzyz(self) -> Self;
    #[must_use]
    fn yzyw(self) -> Self;
    #[must_use]
    fn yzzx(self) -> Self;
    #[must_use]
    fn yzzy(self) -> Self;
    #[must_use]
    fn yzzz(self) -> Self;
    #[must_use]
    fn yzzw(self) -> Self;
    #[must_use]
    fn yzwx(self) -> Self;
    #[must_use]
    fn yzwy(self) -> Self;
    #[must_use]
    fn yzwz(self) -> Self;
    #[must_use]
    fn yzww(self) -> Self;
    #[must_use]
    fn ywxx(self) -> Self;
    #[must_use]
    fn ywxy(self) -> Self;
    #[must_use]
    fn ywxz(self) -> Self;
    #[must_use]
    fn ywxw(self) -> Self;
    #[must_use]
    fn ywyx(self) -> Self;
    #[must_use]
    fn ywyy(self) -> Self;
    #[must_use]
    fn ywyz(self) -> Self;
    #[must_use]
    fn ywyw(self) -> Self;
    #[must_use]
    fn ywzx(self) -> Self;
    #[must_use]
    fn ywzy(self) -> Self;
    #[must_use]
    fn ywzz(self) -> Self;
    #[must_use]
    fn ywzw(self) -> Self;
    #[must_use]
    fn ywwx(self) -> Self;
    #[must_use]
    fn ywwy(self) -> Self;
    #[must_use]
    fn ywwz(self) -> Self;
    #[must_use]
    fn ywww(self) -> Self;
    #[must_use]
    fn zxxx(self) -> Self;
    #[must_use]
    fn zxxy(self) -> Self;
    #[must_use]
    fn zxxz(self) -> Self;
    #[must_use]
    fn zxxw(self) -> Self;
    #[must_use]
    fn zxyx(self) -> Self;
    #[must_use]
    fn zxyy(self) -> Self;
    #[must_use]
    fn zxyz(self) -> Self;
    #[must_use]
    fn zxyw(self) -> Self;
    #[must_use]
    fn zxzx(self) -> Self;
    #[must_use]
    fn zxzy(self) -> Self;
    #[must_use]
    fn zxzz(self) -> Self;
    #[must_use]
    fn zxzw(self) -> Self;
    #[must_use]
    fn zxwx(self) -> Self;
    #[must_use]
    fn zxwy(self) -> Self;
    #[must_use]
    fn zxwz(self) -> Self;
    #[must_use]
    fn zxww(self) -> Self;
    #[must_use]
    fn zyxx(self) -> Self;
    #[must_use]
    fn zyxy(self) -> Self;
    #[must_use]
    fn zyxz(self) -> Self;
    #[must_use]
    fn zyxw(self) -> Self;
    #[must_use]
    fn zyyx(self) -> Self;
    #[must_use]
    fn zyyy(self) -> Self;
    #[must_use]
    fn zyyz(self) -> Self;
    #[must_use]
    fn zyyw(self) -> Self;
    #[must_use]
    fn zyzx(self) -> Self;
    #[must_use]
    fn zyzy(self) -> Self;
    #[must_use]
    fn zyzz(self) -> Self;
    #[must_use]
    fn zyzw(self) -> Self;
    #[must_use]
    fn zywx(self) -> Self;
    #[must_use]
    fn zywy(self) -> Self;
    #[must_use]
    fn zywz(self) -> Self;
    #[must_use]
    fn zyww(self) -> Self;
    #[must_use]
    fn zzxx(self) -> Self;
    #[must_use]
    fn zzxy(self) -> Self;
    #[must_use]
    fn zzxz(self) -> Self;
    #[must_use]
    fn zzxw(self) -> Self;
    #[must_use]
    fn zzyx(self) -> Self;
    #[must_use]
    fn zzyy(self) -> Self;
    #[must_use]
    fn zzyz(self) -> Self;
    #[must_use]
    fn zzyw(self) -> Self;
    #[must_use]
    fn zzzx(self) -> Self;
    #[must_use]
    fn zzzy(self) -> Self;
    #[must_use]
    fn zzzz(self) -> Self;
    #[must_use]
    fn zzzw(self) -> Self;
    #[must_use]
    fn zzwx(self) -> Self;
    #[must_use]
    fn zzwy(self) -> Self;
    #[must_use]
    fn zzwz(self) -> Self;
    #[must_use]
    fn zzww(self) -> Self;
    #[must_use]
    fn zwxx(self) -> Self;
    #[must_use]
    fn zwxy(self) -> Self;
    #[must_use]
    fn zwxz(self) -> Self;
    #[must_use]
    fn zwxw(self) -> Self;
    #[must_use]
    fn zwyx(self) -> Self;
    #[must_use]
    fn zwyy(self) -> Self;
    #[must_use]
    fn zwyz(self) -> Self;
    #[must_use]
    fn zwyw(self) -> Self;
    #[must_use]
    fn zwzx(self) -> Self;
    #[must_use]
    fn zwzy(self) -> Self;
    #[must_use]
    fn zwzz(self) -> Self;
    #[must_use]
    fn zwzw(self) -> Self;
    #[must_use]
    fn zwwx(self) -> Self;
    #[must_use]
    fn zwwy(self) -> Self;
    #[must_use]
    fn zwwz(self) -> Self;
    #[must_use]
    fn zwww(self) -> Self;
    #[must_use]
    fn wxxx(self) -> Self;
    #[must_use]
    fn wxxy(self) -> Self;
    #[must_use]
    fn wxxz(self) -> Self;
    #[must_use]
    fn wxxw(self) -> Self;
    #[must_use]
    fn wxyx(self) -> Self;
    #[must_use]
    fn wxyy(self) -> Self;
    #[must_use]
    fn wxyz(self) -> Self;
    #[must_use]
    fn wxyw(self) -> Self;
    #[must_use]
    fn wxzx(self) -> Self;
    #[must_use]
    fn wxzy(self) -> Self;
    #[must_use]
    fn wxzz(self) -> Self;
    #[must_use]
    fn wxzw(self) -> Self;
    #[must_use]
    fn wxwx(self) -> Self;
    #[must_use]
    fn wxwy(self) -> Self;
    #[must_use]
    fn wxwz(self) -> Self;
    #[must_use]
    fn wxww(self) -> Self;
    #[must_use]
    fn wyxx(self) -> Self;
    #[must_use]
    fn wyxy(self) -> Self;
    #[must_use]
    fn wyxz(self) -> Self;
    #[must_use]
    fn wyxw(self) -> Self;
    #[must_use]
    fn wyyx(self) -> Self;
    #[must_use]
    fn wyyy(self) -> Self;
    #[must_use]
    fn wyyz(self) -> Self;
    #[must_use]
    fn wyyw(self) -> Self;
    #[must_use]
    fn wyzx(self) -> Self;
    #[must_use]
    fn wyzy(self) -> Self;
    #[must_use]
    fn wyzz(self) -> Self;
    #[must_use]
    fn wyzw(self) -> Self;
    #[must_use]
    fn wywx(self) -> Self;
    #[must_use]
    fn wywy(self) -> Self;
    #[must_use]
    fn wywz(self) -> Self;
    #[must_use]
    fn wyww(self) -> Self;
    #[must_use]
    fn wzxx(self) -> Self;
    #[must_use]
    fn wzxy(self) -> Self;
    #[must_use]
    fn wzxz(self) -> Self;
    #[must_use]
    fn wzxw(self) -> Self;
    #[must_use]
    fn wzyx(self) -> Self;
    #[must_use]
    fn wzyy(self) -> Self;
    #[must_use]
    fn wzyz(self) -> Self;
    #[must_use]
    fn wzyw(self) -> Self;
    #[must_use]
    fn wzzx(self) -> Self;
    #[must_use]
    fn wzzy(self) -> Self;
    #[must_use]
    fn wzzz(self) -> Self;
    #[must_use]
    fn wzzw(self) -> Self;
    #[must_use]
    fn wzwx(self) -> Self;
    #[must_use]
    fn wzwy(self) -> Self;
    #[must_use]
    fn wzwz(self) -> Self;
    #[must_use]
    fn wzww(self) -> Self;
    #[must_use]
    fn wwxx(self) -> Self;
    #[must_use]
    fn wwxy(self) -> Self;
    #[must_use]
    fn wwxz(self) -> Self;
    #[must_use]
    fn wwxw(self) -> Self;
    #[must_use]
    fn wwyx(self) -> Self;
    #[must_use]
    fn wwyy(self) -> Self;
    #[must_use]
    fn wwyz(self) -> Self;
    #[must_use]
    fn wwyw(self) -> Self;
    #[must_use]
    fn wwzx(self) -> Self;
    #[must_use]
    fn wwzy(self) -> Self;
    #[must_use]
    fn wwzz(self) -> Self;
    #[must_use]
    fn wwzw(self) -> Self;
    #[must_use]
    fn wwwx(self) -> Self;
    #[must_use]
    fn wwwy(self) -> Self;
    #[must_use]
    fn wwwz(self) -> Self;
    #[must_use]
    fn wwww(self) -> Self;

}

// --- impl_vec2_swizzle! ---
#[macro_export]
macro_rules! impl_vec2_swizzle {
    ($Self:ty, $Vec3:ty, $Vec4:ty) => {
        impl $crate::swizzle::Vec2Swizzles for $Self {
    type Vec3 = $Vec3;
    type Vec4 = $Vec4;

    #[inline(always)]
    fn xy(self) -> Self { self }

    #[inline(always)]
    fn xx(self) -> Self { Self::new(self.x, self.x) }
    #[inline(always)]
    fn yx(self) -> Self { Self::new(self.y, self.x) }
    #[inline(always)]
    fn yy(self) -> Self { Self::new(self.y, self.y) }

    #[inline(always)]
    fn xxx(self) -> $Vec3 { <$Vec3>::new(self.x, self.x, self.x) }
    #[inline(always)]
    fn xxy(self) -> $Vec3 { <$Vec3>::new(self.x, self.x, self.y) }
    #[inline(always)]
    fn xyx(self) -> $Vec3 { <$Vec3>::new(self.x, self.y, self.x) }
    #[inline(always)]
    fn xyy(self) -> $Vec3 { <$Vec3>::new(self.x, self.y, self.y) }
    #[inline(always)]
    fn yxx(self) -> $Vec3 { <$Vec3>::new(self.y, self.x, self.x) }
    #[inline(always)]
    fn yxy(self) -> $Vec3 { <$Vec3>::new(self.y, self.x, self.y) }
    #[inline(always)]
    fn yyx(self) -> $Vec3 { <$Vec3>::new(self.y, self.y, self.x) }
    #[inline(always)]
    fn yyy(self) -> $Vec3 { <$Vec3>::new(self.y, self.y, self.y) }

    #[inline(always)]
    fn xxxx(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.x, self.x) }
    #[inline(always)]
    fn xxxy(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.x, self.y) }
    #[inline(always)]
    fn xxyx(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.y, self.x) }
    #[inline(always)]
    fn xxyy(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.y, self.y) }
    #[inline(always)]
    fn xyxx(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.x, self.x) }
    #[inline(always)]
    fn xyxy(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.x, self.y) }
    #[inline(always)]
    fn xyyx(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.y, self.x) }
    #[inline(always)]
    fn xyyy(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.y, self.y) }
    #[inline(always)]
    fn yxxx(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.x, self.x) }
    #[inline(always)]
    fn yxxy(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.x, self.y) }
    #[inline(always)]
    fn yxyx(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.y, self.x) }
    #[inline(always)]
    fn yxyy(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.y, self.y) }
    #[inline(always)]
    fn yyxx(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.x, self.x) }
    #[inline(always)]
    fn yyxy(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.x, self.y) }
    #[inline(always)]
    fn yyyx(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.y, self.x) }
    #[inline(always)]
    fn yyyy(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.y, self.y) }

        }
    };
}

// --- impl_vec3_swizzle! ---
#[macro_export]
macro_rules! impl_vec3_swizzle {
    ($Self:ty, $Vec2:ty, $Vec4:ty) => {
        impl $crate::swizzle::Vec3Swizzles for $Self {
    type Vec2 = $Vec2;
    type Vec4 = $Vec4;

    #[inline(always)]
    fn xyz(self) -> Self { self }

    #[inline(always)]
    fn xx(self) -> $Vec2 { <$Vec2>::new(self.x, self.x) }
    #[inline(always)]
    fn xy(self) -> $Vec2 { <$Vec2>::new(self.x, self.y) }
    #[inline(always)]
    fn with_xy(self, rhs: $Vec2) -> Self { Self::new(rhs.x, rhs.y, self.z) }
    #[inline(always)]
    fn xz(self) -> $Vec2 { <$Vec2>::new(self.x, self.z) }
    #[inline(always)]
    fn with_xz(self, rhs: $Vec2) -> Self { Self::new(rhs.x, self.y, rhs.y) }
    #[inline(always)]
    fn yx(self) -> $Vec2 { <$Vec2>::new(self.y, self.x) }
    #[inline(always)]
    fn with_yx(self, rhs: $Vec2) -> Self { Self::new(rhs.y, rhs.x, self.z) }
    #[inline(always)]
    fn yy(self) -> $Vec2 { <$Vec2>::new(self.y, self.y) }
    #[inline(always)]
    fn yz(self) -> $Vec2 { <$Vec2>::new(self.y, self.z) }
    #[inline(always)]
    fn with_yz(self, rhs: $Vec2) -> Self { Self::new(self.x, rhs.x, rhs.y) }
    #[inline(always)]
    fn zx(self) -> $Vec2 { <$Vec2>::new(self.z, self.x) }
    #[inline(always)]
    fn with_zx(self, rhs: $Vec2) -> Self { Self::new(rhs.y, self.y, rhs.x) }
    #[inline(always)]
    fn zy(self) -> $Vec2 { <$Vec2>::new(self.z, self.y) }
    #[inline(always)]
    fn with_zy(self, rhs: $Vec2) -> Self { Self::new(self.x, rhs.y, rhs.x) }
    #[inline(always)]
    fn zz(self) -> $Vec2 { <$Vec2>::new(self.z, self.z) }

    #[inline(always)]
    fn xxx(self) -> Self { Self::new(self.x, self.x, self.x) }
    #[inline(always)]
    fn xxy(self) -> Self { Self::new(self.x, self.x, self.y) }
    #[inline(always)]
    fn xxz(self) -> Self { Self::new(self.x, self.x, self.z) }
    #[inline(always)]
    fn xyx(self) -> Self { Self::new(self.x, self.y, self.x) }
    #[inline(always)]
    fn xyy(self) -> Self { Self::new(self.x, self.y, self.y) }
    #[inline(always)]
    fn xzx(self) -> Self { Self::new(self.x, self.z, self.x) }
    #[inline(always)]
    fn xzy(self) -> Self { Self::new(self.x, self.z, self.y) }
    #[inline(always)]
    fn xzz(self) -> Self { Self::new(self.x, self.z, self.z) }
    #[inline(always)]
    fn yxx(self) -> Self { Self::new(self.y, self.x, self.x) }
    #[inline(always)]
    fn yxy(self) -> Self { Self::new(self.y, self.x, self.y) }
    #[inline(always)]
    fn yxz(self) -> Self { Self::new(self.y, self.x, self.z) }
    #[inline(always)]
    fn yyx(self) -> Self { Self::new(self.y, self.y, self.x) }
    #[inline(always)]
    fn yyy(self) -> Self { Self::new(self.y, self.y, self.y) }
    #[inline(always)]
    fn yyz(self) -> Self { Self::new(self.y, self.y, self.z) }
    #[inline(always)]
    fn yzx(self) -> Self { Self::new(self.y, self.z, self.x) }
    #[inline(always)]
    fn yzy(self) -> Self { Self::new(self.y, self.z, self.y) }
    #[inline(always)]
    fn yzz(self) -> Self { Self::new(self.y, self.z, self.z) }
    #[inline(always)]
    fn zxx(self) -> Self { Self::new(self.z, self.x, self.x) }
    #[inline(always)]
    fn zxy(self) -> Self { Self::new(self.z, self.x, self.y) }
    #[inline(always)]
    fn zxz(self) -> Self { Self::new(self.z, self.x, self.z) }
    #[inline(always)]
    fn zyx(self) -> Self { Self::new(self.z, self.y, self.x) }
    #[inline(always)]
    fn zyy(self) -> Self { Self::new(self.z, self.y, self.y) }
    #[inline(always)]
    fn zyz(self) -> Self { Self::new(self.z, self.y, self.z) }
    #[inline(always)]
    fn zzx(self) -> Self { Self::new(self.z, self.z, self.x) }
    #[inline(always)]
    fn zzy(self) -> Self { Self::new(self.z, self.z, self.y) }
    #[inline(always)]
    fn zzz(self) -> Self { Self::new(self.z, self.z, self.z) }

    #[inline(always)]
    fn xxxx(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.x, self.x) }
    #[inline(always)]
    fn xxxy(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.x, self.y) }
    #[inline(always)]
    fn xxxz(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.x, self.z) }
    #[inline(always)]
    fn xxyx(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.y, self.x) }
    #[inline(always)]
    fn xxyy(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.y, self.y) }
    #[inline(always)]
    fn xxyz(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.y, self.z) }
    #[inline(always)]
    fn xxzx(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.z, self.x) }
    #[inline(always)]
    fn xxzy(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.z, self.y) }
    #[inline(always)]
    fn xxzz(self) -> $Vec4 { <$Vec4>::new(self.x, self.x, self.z, self.z) }
    #[inline(always)]
    fn xyxx(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.x, self.x) }
    #[inline(always)]
    fn xyxy(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.x, self.y) }
    #[inline(always)]
    fn xyxz(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.x, self.z) }
    #[inline(always)]
    fn xyyx(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.y, self.x) }
    #[inline(always)]
    fn xyyy(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.y, self.y) }
    #[inline(always)]
    fn xyyz(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.y, self.z) }
    #[inline(always)]
    fn xyzx(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.z, self.x) }
    #[inline(always)]
    fn xyzy(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.z, self.y) }
    #[inline(always)]
    fn xyzz(self) -> $Vec4 { <$Vec4>::new(self.x, self.y, self.z, self.z) }
    #[inline(always)]
    fn xzxx(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.x, self.x) }
    #[inline(always)]
    fn xzxy(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.x, self.y) }
    #[inline(always)]
    fn xzxz(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.x, self.z) }
    #[inline(always)]
    fn xzyx(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.y, self.x) }
    #[inline(always)]
    fn xzyy(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.y, self.y) }
    #[inline(always)]
    fn xzyz(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.y, self.z) }
    #[inline(always)]
    fn xzzx(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.z, self.x) }
    #[inline(always)]
    fn xzzy(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.z, self.y) }
    #[inline(always)]
    fn xzzz(self) -> $Vec4 { <$Vec4>::new(self.x, self.z, self.z, self.z) }
    #[inline(always)]
    fn yxxx(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.x, self.x) }
    #[inline(always)]
    fn yxxy(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.x, self.y) }
    #[inline(always)]
    fn yxxz(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.x, self.z) }
    #[inline(always)]
    fn yxyx(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.y, self.x) }
    #[inline(always)]
    fn yxyy(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.y, self.y) }
    #[inline(always)]
    fn yxyz(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.y, self.z) }
    #[inline(always)]
    fn yxzx(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.z, self.x) }
    #[inline(always)]
    fn yxzy(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.z, self.y) }
    #[inline(always)]
    fn yxzz(self) -> $Vec4 { <$Vec4>::new(self.y, self.x, self.z, self.z) }
    #[inline(always)]
    fn yyxx(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.x, self.x) }
    #[inline(always)]
    fn yyxy(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.x, self.y) }
    #[inline(always)]
    fn yyxz(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.x, self.z) }
    #[inline(always)]
    fn yyyx(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.y, self.x) }
    #[inline(always)]
    fn yyyy(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.y, self.y) }
    #[inline(always)]
    fn yyyz(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.y, self.z) }
    #[inline(always)]
    fn yyzx(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.z, self.x) }
    #[inline(always)]
    fn yyzy(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.z, self.y) }
    #[inline(always)]
    fn yyzz(self) -> $Vec4 { <$Vec4>::new(self.y, self.y, self.z, self.z) }
    #[inline(always)]
    fn yzxx(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.x, self.x) }
    #[inline(always)]
    fn yzxy(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.x, self.y) }
    #[inline(always)]
    fn yzxz(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.x, self.z) }
    #[inline(always)]
    fn yzyx(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.y, self.x) }
    #[inline(always)]
    fn yzyy(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.y, self.y) }
    #[inline(always)]
    fn yzyz(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.y, self.z) }
    #[inline(always)]
    fn yzzx(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.z, self.x) }
    #[inline(always)]
    fn yzzy(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.z, self.y) }
    #[inline(always)]
    fn yzzz(self) -> $Vec4 { <$Vec4>::new(self.y, self.z, self.z, self.z) }
    #[inline(always)]
    fn zxxx(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.x, self.x) }
    #[inline(always)]
    fn zxxy(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.x, self.y) }
    #[inline(always)]
    fn zxxz(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.x, self.z) }
    #[inline(always)]
    fn zxyx(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.y, self.x) }
    #[inline(always)]
    fn zxyy(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.y, self.y) }
    #[inline(always)]
    fn zxyz(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.y, self.z) }
    #[inline(always)]
    fn zxzx(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.z, self.x) }
    #[inline(always)]
    fn zxzy(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.z, self.y) }
    #[inline(always)]
    fn zxzz(self) -> $Vec4 { <$Vec4>::new(self.z, self.x, self.z, self.z) }
    #[inline(always)]
    fn zyxx(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.x, self.x) }
    #[inline(always)]
    fn zyxy(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.x, self.y) }
    #[inline(always)]
    fn zyxz(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.x, self.z) }
    #[inline(always)]
    fn zyyx(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.y, self.x) }
    #[inline(always)]
    fn zyyy(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.y, self.y) }
    #[inline(always)]
    fn zyyz(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.y, self.z) }
    #[inline(always)]
    fn zyzx(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.z, self.x) }
    #[inline(always)]
    fn zyzy(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.z, self.y) }
    #[inline(always)]
    fn zyzz(self) -> $Vec4 { <$Vec4>::new(self.z, self.y, self.z, self.z) }
    #[inline(always)]
    fn zzxx(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.x, self.x) }
    #[inline(always)]
    fn zzxy(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.x, self.y) }
    #[inline(always)]
    fn zzxz(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.x, self.z) }
    #[inline(always)]
    fn zzyx(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.y, self.x) }
    #[inline(always)]
    fn zzyy(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.y, self.y) }
    #[inline(always)]
    fn zzyz(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.y, self.z) }
    #[inline(always)]
    fn zzzx(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.z, self.x) }
    #[inline(always)]
    fn zzzy(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.z, self.y) }
    #[inline(always)]
    fn zzzz(self) -> $Vec4 { <$Vec4>::new(self.z, self.z, self.z, self.z) }

        }
    };
}

// --- impl_vec4_swizzle! ---
#[macro_export]
macro_rules! impl_vec4_swizzle {
    ($Self:ty, $Vec2:ty, $Vec3:ty) => {
        impl $crate::swizzle::Vec4Swizzles for $Self {
    type Vec2 = $Vec2;
    type Vec3 = $Vec3;

    #[inline(always)]
    fn xyzw(self) -> Self { self }

    #[inline(always)]
    fn xx(self) -> $Vec2 { <$Vec2>::new(self.x, self.x) }
    #[inline(always)]
    fn xy(self) -> $Vec2 { <$Vec2>::new(self.x, self.y) }
    #[inline(always)]
    fn with_xy(self, rhs: $Vec2) -> Self { Self::new(rhs.x, rhs.y, self.z, self.w) }
    #[inline(always)]
    fn xz(self) -> $Vec2 { <$Vec2>::new(self.x, self.z) }
    #[inline(always)]
    fn with_xz(self, rhs: $Vec2) -> Self { Self::new(rhs.x, self.y, rhs.y, self.w) }
    #[inline(always)]
    fn xw(self) -> $Vec2 { <$Vec2>::new(self.x, self.w) }
    #[inline(always)]
    fn with_xw(self, rhs: $Vec2) -> Self { Self::new(rhs.x, self.y, self.z, rhs.y) }
    #[inline(always)]
    fn yx(self) -> $Vec2 { <$Vec2>::new(self.y, self.x) }
    #[inline(always)]
    fn with_yx(self, rhs: $Vec2) -> Self { Self::new(rhs.y, rhs.x, self.z, self.w) }
    #[inline(always)]
    fn yy(self) -> $Vec2 { <$Vec2>::new(self.y, self.y) }
    #[inline(always)]
    fn yz(self) -> $Vec2 { <$Vec2>::new(self.y, self.z) }
    #[inline(always)]
    fn with_yz(self, rhs: $Vec2) -> Self { Self::new(self.x, rhs.x, rhs.y, self.w) }
    #[inline(always)]
    fn yw(self) -> $Vec2 { <$Vec2>::new(self.y, self.w) }
    #[inline(always)]
    fn with_yw(self, rhs: $Vec2) -> Self { Self::new(self.x, rhs.x, self.z, rhs.y) }
    #[inline(always)]
    fn zx(self) -> $Vec2 { <$Vec2>::new(self.z, self.x) }
    #[inline(always)]
    fn with_zx(self, rhs: $Vec2) -> Self { Self::new(rhs.y, self.y, rhs.x, self.w) }
    #[inline(always)]
    fn zy(self) -> $Vec2 { <$Vec2>::new(self.z, self.y) }
    #[inline(always)]
    fn with_zy(self, rhs: $Vec2) -> Self { Self::new(self.x, rhs.y, rhs.x, self.w) }
    #[inline(always)]
    fn zz(self) -> $Vec2 { <$Vec2>::new(self.z, self.z) }
    #[inline(always)]
    fn zw(self) -> $Vec2 { <$Vec2>::new(self.z, self.w) }
    #[inline(always)]
    fn with_zw(self, rhs: $Vec2) -> Self { Self::new(self.x, self.y, rhs.x, rhs.y) }
    #[inline(always)]
    fn wx(self) -> $Vec2 { <$Vec2>::new(self.w, self.x) }
    #[inline(always)]
    fn with_wx(self, rhs: $Vec2) -> Self { Self::new(rhs.y, self.y, self.z, rhs.x) }
    #[inline(always)]
    fn wy(self) -> $Vec2 { <$Vec2>::new(self.w, self.y) }
    #[inline(always)]
    fn with_wy(self, rhs: $Vec2) -> Self { Self::new(self.x, rhs.y, self.z, rhs.x) }
    #[inline(always)]
    fn wz(self) -> $Vec2 { <$Vec2>::new(self.w, self.z) }
    #[inline(always)]
    fn with_wz(self, rhs: $Vec2) -> Self { Self::new(self.x, self.y, rhs.y, rhs.x) }
    #[inline(always)]
    fn ww(self) -> $Vec2 { <$Vec2>::new(self.w, self.w) }

    #[inline(always)]
    fn xxx(self) -> $Vec3 { <$Vec3>::new(self.x, self.x, self.x) }
    #[inline(always)]
    fn xxy(self) -> $Vec3 { <$Vec3>::new(self.x, self.x, self.y) }
    #[inline(always)]
    fn xxz(self) -> $Vec3 { <$Vec3>::new(self.x, self.x, self.z) }
    #[inline(always)]
    fn xxw(self) -> $Vec3 { <$Vec3>::new(self.x, self.x, self.w) }
    #[inline(always)]
    fn xyx(self) -> $Vec3 { <$Vec3>::new(self.x, self.y, self.x) }
    #[inline(always)]
    fn xyy(self) -> $Vec3 { <$Vec3>::new(self.x, self.y, self.y) }
    #[inline(always)]
    fn xyz(self) -> $Vec3 { <$Vec3>::new(self.x, self.y, self.z) }
    #[inline(always)]
    fn with_xyz(self, rhs: $Vec3) -> Self { Self::new(rhs.x, rhs.y, rhs.z, self.w) }
    #[inline(always)]
    fn xyw(self) -> $Vec3 { <$Vec3>::new(self.x, self.y, self.w) }
    #[inline(always)]
    fn with_xyw(self, rhs: $Vec3) -> Self { Self::new(rhs.x, rhs.y, self.z, rhs.z) }
    #[inline(always)]
    fn xzx(self) -> $Vec3 { <$Vec3>::new(self.x, self.z, self.x) }
    #[inline(always)]
    fn xzy(self) -> $Vec3 { <$Vec3>::new(self.x, self.z, self.y) }
    #[inline(always)]
    fn with_xzy(self, rhs: $Vec3) -> Self { Self::new(rhs.x, rhs.z, rhs.y, self.w) }
    #[inline(always)]
    fn xzz(self) -> $Vec3 { <$Vec3>::new(self.x, self.z, self.z) }
    #[inline(always)]
    fn xzw(self) -> $Vec3 { <$Vec3>::new(self.x, self.z, self.w) }
    #[inline(always)]
    fn with_xzw(self, rhs: $Vec3) -> Self { Self::new(rhs.x, self.y, rhs.y, rhs.z) }
    #[inline(always)]
    fn xwx(self) -> $Vec3 { <$Vec3>::new(self.x, self.w, self.x) }
    #[inline(always)]
    fn xwy(self) -> $Vec3 { <$Vec3>::new(self.x, self.w, self.y) }
    #[inline(always)]
    fn with_xwy(self, rhs: $Vec3) -> Self { Self::new(rhs.x, rhs.z, self.z, rhs.y) }
    #[inline(always)]
    fn xwz(self) -> $Vec3 { <$Vec3>::new(self.x, self.w, self.z) }
    #[inline(always)]
    fn with_xwz(self, rhs: $Vec3) -> Self { Self::new(rhs.x, self.y, rhs.z, rhs.y) }
    #[inline(always)]
    fn xww(self) -> $Vec3 { <$Vec3>::new(self.x, self.w, self.w) }
    #[inline(always)]
    fn yxx(self) -> $Vec3 { <$Vec3>::new(self.y, self.x, self.x) }
    #[inline(always)]
    fn yxy(self) -> $Vec3 { <$Vec3>::new(self.y, self.x, self.y) }
    #[inline(always)]
    fn yxz(self) -> $Vec3 { <$Vec3>::new(self.y, self.x, self.z) }
    #[inline(always)]
    fn with_yxz(self, rhs: $Vec3) -> Self { Self::new(rhs.y, rhs.x, rhs.z, self.w) }
    #[inline(always)]
    fn yxw(self) -> $Vec3 { <$Vec3>::new(self.y, self.x, self.w) }
    #[inline(always)]
    fn with_yxw(self, rhs: $Vec3) -> Self { Self::new(rhs.y, rhs.x, self.z, rhs.z) }
    #[inline(always)]
    fn yyx(self) -> $Vec3 { <$Vec3>::new(self.y, self.y, self.x) }
    #[inline(always)]
    fn yyy(self) -> $Vec3 { <$Vec3>::new(self.y, self.y, self.y) }
    #[inline(always)]
    fn yyz(self) -> $Vec3 { <$Vec3>::new(self.y, self.y, self.z) }
    #[inline(always)]
    fn yyw(self) -> $Vec3 { <$Vec3>::new(self.y, self.y, self.w) }
    #[inline(always)]
    fn yzx(self) -> $Vec3 { <$Vec3>::new(self.y, self.z, self.x) }
    #[inline(always)]
    fn with_yzx(self, rhs: $Vec3) -> Self { Self::new(rhs.z, rhs.x, rhs.y, self.w) }
    #[inline(always)]
    fn yzy(self) -> $Vec3 { <$Vec3>::new(self.y, self.z, self.y) }
    #[inline(always)]
    fn yzz(self) -> $Vec3 { <$Vec3>::new(self.y, self.z, self.z) }
    #[inline(always)]
    fn yzw(self) -> $Vec3 { <$Vec3>::new(self.y, self.z, self.w) }
    #[inline(always)]
    fn with_yzw(self, rhs: $Vec3) -> Self { Self::new(self.x, rhs.x, rhs.y, rhs.z) }
    #[inline(always)]
    fn ywx(self) -> $Vec3 { <$Vec3>::new(self.y, self.w, self.x) }
    #[inline(always)]
    fn with_ywx(self, rhs: $Vec3) -> Self { Self::new(rhs.z, rhs.x, self.z, rhs.y) }
    #[inline(always)]
    fn ywy(self) -> $Vec3 { <$Vec3>::new(self.y, self.w, self.y) }
    #[inline(always)]
    fn ywz(self) -> $Vec3 { <$Vec3>::new(self.y, self.w, self.z) }
    #[inline(always)]
    fn with_ywz(self, rhs: $Vec3) -> Self { Self::new(self.x, rhs.x, rhs.z, rhs.y) }
    #[inline(always)]
    fn yww(self) -> $Vec3 { <$Vec3>::new(self.y, self.w, self.w) }
    #[inline(always)]
    fn zxx(self) -> $Vec3 { <$Vec3>::new(self.z, self.x, self.x) }
    #[inline(always)]
    fn zxy(self) -> $Vec3 { <$Vec3>::new(self.z, self.x, self.y) }
    #[inline(always)]
    fn with_zxy(self, rhs: $Vec3) -> Self { Self::new(rhs.y, rhs.z, rhs.x, self.w) }
    #[inline(always)]
    fn zxz(self) -> $Vec3 { <$Vec3>::new(self.z, self.x, self.z) }
    #[inline(always)]
    fn zxw(self) -> $Vec3 { <$Vec3>::new(self.z, self.x, self.w) }
    #[inline(always)]
    fn with_zxw(self, rhs: $Vec3) -> Self { Self::new(rhs.y, self.y, rhs.x, rhs.z) }
    #[inline(always)]
    fn zyx(self) -> $Vec3 { <$Vec3>::new(self.z, self.y, self.x) }
    #[inline(always)]
    fn with_zyx(self, rhs: $Vec3) -> Self { Self::new(rhs.z, rhs.y, rhs.x, self.w) }
    #[inline(always)]
    fn zyy(self) -> $Vec3 { <$Vec3>::new(self.z, self.y, self.y) }
    #[inline(always)]
    fn zyz(self) -> $Vec3 { <$Vec3>::new(self.z, self.y, self.z) }
    #[inline(always)]
    fn zyw(self) -> $Vec3 { <$Vec3>::new(self.z, self.y, self.w) }
    #[inline(always)]
    fn with_zyw(self, rhs: $Vec3) -> Self { Self::new(self.x, rhs.y, rhs.x, rhs.z) }
    #[inline(always)]
    fn zzx(self) -> $Vec3 { <$Vec3>::new(self.z, self.z, self.x) }
    #[inline(always)]
    fn zzy(self) -> $Vec3 { <$Vec3>::new(self.z, self.z, self.y) }
    #[inline(always)]
    fn zzz(self) -> $Vec3 { <$Vec3>::new(self.z, self.z, self.z) }
    #[inline(always)]
    fn zzw(self) -> $Vec3 { <$Vec3>::new(self.z, self.z, self.w) }
    #[inline(always)]
    fn zwx(self) -> $Vec3 { <$Vec3>::new(self.z, self.w, self.x) }
    #[inline(always)]
    fn with_zwx(self, rhs: $Vec3) -> Self { Self::new(rhs.z, self.y, rhs.x, rhs.y) }
    #[inline(always)]
    fn zwy(self) -> $Vec3 { <$Vec3>::new(self.z, self.w, self.y) }
    #[inline(always)]
    fn with_zwy(self, rhs: $Vec3) -> Self { Self::new(self.x, rhs.z, rhs.x, rhs.y) }
    #[inline(always)]
    fn zwz(self) -> $Vec3 { <$Vec3>::new(self.z, self.w, self.z) }
    #[inline(always)]
    fn zww(self) -> $Vec3 { <$Vec3>::new(self.z, self.w, self.w) }
    #[inline(always)]
    fn wxx(self) -> $Vec3 { <$Vec3>::new(self.w, self.x, self.x) }
    #[inline(always)]
    fn wxy(self) -> $Vec3 { <$Vec3>::new(self.w, self.x, self.y) }
    #[inline(always)]
    fn with_wxy(self, rhs: $Vec3) -> Self { Self::new(rhs.y, rhs.z, self.z, rhs.x) }
    #[inline(always)]
    fn wxz(self) -> $Vec3 { <$Vec3>::new(self.w, self.x, self.z) }
    #[inline(always)]
    fn with_wxz(self, rhs: $Vec3) -> Self { Self::new(rhs.y, self.y, rhs.z, rhs.x) }
    #[inline(always)]
    fn wxw(self) -> $Vec3 { <$Vec3>::new(self.w, self.x, self.w) }
    #[inline(always)]
    fn wyx(self) -> $Vec3 { <$Vec3>::new(self.w, self.y, self.x) }
    #[inline(always)]
    fn with_wyx(self, rhs: $Vec3) -> Self { Self::new(rhs.z, rhs.y, self.z, rhs.x) }
    #[inline(always)]
    fn wyy(self) -> $Vec3 { <$Vec3>::new(self.w, self.y, self.y) }
    #[inline(always)]
    fn wyz(self) -> $Vec3 { <$Vec3>::new(self.w, self.y, self.z) }
    #[inline(always)]
    fn with_wyz(self, rhs: $Vec3) -> Self { Self::new(self.x, rhs.y, rhs.z, rhs.x) }
    #[inline(always)]
    fn wyw(self) -> $Vec3 { <$Vec3>::new(self.w, self.y, self.w) }
    #[inline(always)]
    fn wzx(self) -> $Vec3 { <$Vec3>::new(self.w, self.z, self.x) }
    #[inline(always)]
    fn with_wzx(self, rhs: $Vec3) -> Self { Self::new(rhs.z, self.y, rhs.y, rhs.x) }
    #[inline(always)]
    fn wzy(self) -> $Vec3 { <$Vec3>::new(self.w, self.z, self.y) }
    #[inline(always)]
    fn with_wzy(self, rhs: $Vec3) -> Self { Self::new(self.x, rhs.z, rhs.y, rhs.x) }
    #[inline(always)]
    fn wzz(self) -> $Vec3 { <$Vec3>::new(self.w, self.z, self.z) }
    #[inline(always)]
    fn wzw(self) -> $Vec3 { <$Vec3>::new(self.w, self.z, self.w) }
    #[inline(always)]
    fn wwx(self) -> $Vec3 { <$Vec3>::new(self.w, self.w, self.x) }
    #[inline(always)]
    fn wwy(self) -> $Vec3 { <$Vec3>::new(self.w, self.w, self.y) }
    #[inline(always)]
    fn wwz(self) -> $Vec3 { <$Vec3>::new(self.w, self.w, self.z) }
    #[inline(always)]
    fn www(self) -> $Vec3 { <$Vec3>::new(self.w, self.w, self.w) }

    #[inline(always)]
    fn xxxx(self) -> Self { Self::new(self.x, self.x, self.x, self.x) }
    #[inline(always)]
    fn xxxy(self) -> Self { Self::new(self.x, self.x, self.x, self.y) }
    #[inline(always)]
    fn xxxz(self) -> Self { Self::new(self.x, self.x, self.x, self.z) }
    #[inline(always)]
    fn xxxw(self) -> Self { Self::new(self.x, self.x, self.x, self.w) }
    #[inline(always)]
    fn xxyx(self) -> Self { Self::new(self.x, self.x, self.y, self.x) }
    #[inline(always)]
    fn xxyy(self) -> Self { Self::new(self.x, self.x, self.y, self.y) }
    #[inline(always)]
    fn xxyz(self) -> Self { Self::new(self.x, self.x, self.y, self.z) }
    #[inline(always)]
    fn xxyw(self) -> Self { Self::new(self.x, self.x, self.y, self.w) }
    #[inline(always)]
    fn xxzx(self) -> Self { Self::new(self.x, self.x, self.z, self.x) }
    #[inline(always)]
    fn xxzy(self) -> Self { Self::new(self.x, self.x, self.z, self.y) }
    #[inline(always)]
    fn xxzz(self) -> Self { Self::new(self.x, self.x, self.z, self.z) }
    #[inline(always)]
    fn xxzw(self) -> Self { Self::new(self.x, self.x, self.z, self.w) }
    #[inline(always)]
    fn xxwx(self) -> Self { Self::new(self.x, self.x, self.w, self.x) }
    #[inline(always)]
    fn xxwy(self) -> Self { Self::new(self.x, self.x, self.w, self.y) }
    #[inline(always)]
    fn xxwz(self) -> Self { Self::new(self.x, self.x, self.w, self.z) }
    #[inline(always)]
    fn xxww(self) -> Self { Self::new(self.x, self.x, self.w, self.w) }
    #[inline(always)]
    fn xyxx(self) -> Self { Self::new(self.x, self.y, self.x, self.x) }
    #[inline(always)]
    fn xyxy(self) -> Self { Self::new(self.x, self.y, self.x, self.y) }
    #[inline(always)]
    fn xyxz(self) -> Self { Self::new(self.x, self.y, self.x, self.z) }
    #[inline(always)]
    fn xyxw(self) -> Self { Self::new(self.x, self.y, self.x, self.w) }
    #[inline(always)]
    fn xyyx(self) -> Self { Self::new(self.x, self.y, self.y, self.x) }
    #[inline(always)]
    fn xyyy(self) -> Self { Self::new(self.x, self.y, self.y, self.y) }
    #[inline(always)]
    fn xyyz(self) -> Self { Self::new(self.x, self.y, self.y, self.z) }
    #[inline(always)]
    fn xyyw(self) -> Self { Self::new(self.x, self.y, self.y, self.w) }
    #[inline(always)]
    fn xyzx(self) -> Self { Self::new(self.x, self.y, self.z, self.x) }
    #[inline(always)]
    fn xyzy(self) -> Self { Self::new(self.x, self.y, self.z, self.y) }
    #[inline(always)]
    fn xyzz(self) -> Self { Self::new(self.x, self.y, self.z, self.z) }
    #[inline(always)]
    fn xywx(self) -> Self { Self::new(self.x, self.y, self.w, self.x) }
    #[inline(always)]
    fn xywy(self) -> Self { Self::new(self.x, self.y, self.w, self.y) }
    #[inline(always)]
    fn xywz(self) -> Self { Self::new(self.x, self.y, self.w, self.z) }
    #[inline(always)]
    fn xyww(self) -> Self { Self::new(self.x, self.y, self.w, self.w) }
    #[inline(always)]
    fn xzxx(self) -> Self { Self::new(self.x, self.z, self.x, self.x) }
    #[inline(always)]
    fn xzxy(self) -> Self { Self::new(self.x, self.z, self.x, self.y) }
    #[inline(always)]
    fn xzxz(self) -> Self { Self::new(self.x, self.z, self.x, self.z) }
    #[inline(always)]
    fn xzxw(self) -> Self { Self::new(self.x, self.z, self.x, self.w) }
    #[inline(always)]
    fn xzyx(self) -> Self { Self::new(self.x, self.z, self.y, self.x) }
    #[inline(always)]
    fn xzyy(self) -> Self { Self::new(self.x, self.z, self.y, self.y) }
    #[inline(always)]
    fn xzyz(self) -> Self { Self::new(self.x, self.z, self.y, self.z) }
    #[inline(always)]
    fn xzyw(self) -> Self { Self::new(self.x, self.z, self.y, self.w) }
    #[inline(always)]
    fn xzzx(self) -> Self { Self::new(self.x, self.z, self.z, self.x) }
    #[inline(always)]
    fn xzzy(self) -> Self { Self::new(self.x, self.z, self.z, self.y) }
    #[inline(always)]
    fn xzzz(self) -> Self { Self::new(self.x, self.z, self.z, self.z) }
    #[inline(always)]
    fn xzzw(self) -> Self { Self::new(self.x, self.z, self.z, self.w) }
    #[inline(always)]
    fn xzwx(self) -> Self { Self::new(self.x, self.z, self.w, self.x) }
    #[inline(always)]
    fn xzwy(self) -> Self { Self::new(self.x, self.z, self.w, self.y) }
    #[inline(always)]
    fn xzwz(self) -> Self { Self::new(self.x, self.z, self.w, self.z) }
    #[inline(always)]
    fn xzww(self) -> Self { Self::new(self.x, self.z, self.w, self.w) }
    #[inline(always)]
    fn xwxx(self) -> Self { Self::new(self.x, self.w, self.x, self.x) }
    #[inline(always)]
    fn xwxy(self) -> Self { Self::new(self.x, self.w, self.x, self.y) }
    #[inline(always)]
    fn xwxz(self) -> Self { Self::new(self.x, self.w, self.x, self.z) }
    #[inline(always)]
    fn xwxw(self) -> Self { Self::new(self.x, self.w, self.x, self.w) }
    #[inline(always)]
    fn xwyx(self) -> Self { Self::new(self.x, self.w, self.y, self.x) }
    #[inline(always)]
    fn xwyy(self) -> Self { Self::new(self.x, self.w, self.y, self.y) }
    #[inline(always)]
    fn xwyz(self) -> Self { Self::new(self.x, self.w, self.y, self.z) }
    #[inline(always)]
    fn xwyw(self) -> Self { Self::new(self.x, self.w, self.y, self.w) }
    #[inline(always)]
    fn xwzx(self) -> Self { Self::new(self.x, self.w, self.z, self.x) }
    #[inline(always)]
    fn xwzy(self) -> Self { Self::new(self.x, self.w, self.z, self.y) }
    #[inline(always)]
    fn xwzz(self) -> Self { Self::new(self.x, self.w, self.z, self.z) }
    #[inline(always)]
    fn xwzw(self) -> Self { Self::new(self.x, self.w, self.z, self.w) }
    #[inline(always)]
    fn xwwx(self) -> Self { Self::new(self.x, self.w, self.w, self.x) }
    #[inline(always)]
    fn xwwy(self) -> Self { Self::new(self.x, self.w, self.w, self.y) }
    #[inline(always)]
    fn xwwz(self) -> Self { Self::new(self.x, self.w, self.w, self.z) }
    #[inline(always)]
    fn xwww(self) -> Self { Self::new(self.x, self.w, self.w, self.w) }
    #[inline(always)]
    fn yxxx(self) -> Self { Self::new(self.y, self.x, self.x, self.x) }
    #[inline(always)]
    fn yxxy(self) -> Self { Self::new(self.y, self.x, self.x, self.y) }
    #[inline(always)]
    fn yxxz(self) -> Self { Self::new(self.y, self.x, self.x, self.z) }
    #[inline(always)]
    fn yxxw(self) -> Self { Self::new(self.y, self.x, self.x, self.w) }
    #[inline(always)]
    fn yxyx(self) -> Self { Self::new(self.y, self.x, self.y, self.x) }
    #[inline(always)]
    fn yxyy(self) -> Self { Self::new(self.y, self.x, self.y, self.y) }
    #[inline(always)]
    fn yxyz(self) -> Self { Self::new(self.y, self.x, self.y, self.z) }
    #[inline(always)]
    fn yxyw(self) -> Self { Self::new(self.y, self.x, self.y, self.w) }
    #[inline(always)]
    fn yxzx(self) -> Self { Self::new(self.y, self.x, self.z, self.x) }
    #[inline(always)]
    fn yxzy(self) -> Self { Self::new(self.y, self.x, self.z, self.y) }
    #[inline(always)]
    fn yxzz(self) -> Self { Self::new(self.y, self.x, self.z, self.z) }
    #[inline(always)]
    fn yxzw(self) -> Self { Self::new(self.y, self.x, self.z, self.w) }
    #[inline(always)]
    fn yxwx(self) -> Self { Self::new(self.y, self.x, self.w, self.x) }
    #[inline(always)]
    fn yxwy(self) -> Self { Self::new(self.y, self.x, self.w, self.y) }
    #[inline(always)]
    fn yxwz(self) -> Self { Self::new(self.y, self.x, self.w, self.z) }
    #[inline(always)]
    fn yxww(self) -> Self { Self::new(self.y, self.x, self.w, self.w) }
    #[inline(always)]
    fn yyxx(self) -> Self { Self::new(self.y, self.y, self.x, self.x) }
    #[inline(always)]
    fn yyxy(self) -> Self { Self::new(self.y, self.y, self.x, self.y) }
    #[inline(always)]
    fn yyxz(self) -> Self { Self::new(self.y, self.y, self.x, self.z) }
    #[inline(always)]
    fn yyxw(self) -> Self { Self::new(self.y, self.y, self.x, self.w) }
    #[inline(always)]
    fn yyyx(self) -> Self { Self::new(self.y, self.y, self.y, self.x) }
    #[inline(always)]
    fn yyyy(self) -> Self { Self::new(self.y, self.y, self.y, self.y) }
    #[inline(always)]
    fn yyyz(self) -> Self { Self::new(self.y, self.y, self.y, self.z) }
    #[inline(always)]
    fn yyyw(self) -> Self { Self::new(self.y, self.y, self.y, self.w) }
    #[inline(always)]
    fn yyzx(self) -> Self { Self::new(self.y, self.y, self.z, self.x) }
    #[inline(always)]
    fn yyzy(self) -> Self { Self::new(self.y, self.y, self.z, self.y) }
    #[inline(always)]
    fn yyzz(self) -> Self { Self::new(self.y, self.y, self.z, self.z) }
    #[inline(always)]
    fn yyzw(self) -> Self { Self::new(self.y, self.y, self.z, self.w) }
    #[inline(always)]
    fn yywx(self) -> Self { Self::new(self.y, self.y, self.w, self.x) }
    #[inline(always)]
    fn yywy(self) -> Self { Self::new(self.y, self.y, self.w, self.y) }
    #[inline(always)]
    fn yywz(self) -> Self { Self::new(self.y, self.y, self.w, self.z) }
    #[inline(always)]
    fn yyww(self) -> Self { Self::new(self.y, self.y, self.w, self.w) }
    #[inline(always)]
    fn yzxx(self) -> Self { Self::new(self.y, self.z, self.x, self.x) }
    #[inline(always)]
    fn yzxy(self) -> Self { Self::new(self.y, self.z, self.x, self.y) }
    #[inline(always)]
    fn yzxz(self) -> Self { Self::new(self.y, self.z, self.x, self.z) }
    #[inline(always)]
    fn yzxw(self) -> Self { Self::new(self.y, self.z, self.x, self.w) }
    #[inline(always)]
    fn yzyx(self) -> Self { Self::new(self.y, self.z, self.y, self.x) }
    #[inline(always)]
    fn yzyy(self) -> Self { Self::new(self.y, self.z, self.y, self.y) }
    #[inline(always)]
    fn yzyz(self) -> Self { Self::new(self.y, self.z, self.y, self.z) }
    #[inline(always)]
    fn yzyw(self) -> Self { Self::new(self.y, self.z, self.y, self.w) }
    #[inline(always)]
    fn yzzx(self) -> Self { Self::new(self.y, self.z, self.z, self.x) }
    #[inline(always)]
    fn yzzy(self) -> Self { Self::new(self.y, self.z, self.z, self.y) }
    #[inline(always)]
    fn yzzz(self) -> Self { Self::new(self.y, self.z, self.z, self.z) }
    #[inline(always)]
    fn yzzw(self) -> Self { Self::new(self.y, self.z, self.z, self.w) }
    #[inline(always)]
    fn yzwx(self) -> Self { Self::new(self.y, self.z, self.w, self.x) }
    #[inline(always)]
    fn yzwy(self) -> Self { Self::new(self.y, self.z, self.w, self.y) }
    #[inline(always)]
    fn yzwz(self) -> Self { Self::new(self.y, self.z, self.w, self.z) }
    #[inline(always)]
    fn yzww(self) -> Self { Self::new(self.y, self.z, self.w, self.w) }
    #[inline(always)]
    fn ywxx(self) -> Self { Self::new(self.y, self.w, self.x, self.x) }
    #[inline(always)]
    fn ywxy(self) -> Self { Self::new(self.y, self.w, self.x, self.y) }
    #[inline(always)]
    fn ywxz(self) -> Self { Self::new(self.y, self.w, self.x, self.z) }
    #[inline(always)]
    fn ywxw(self) -> Self { Self::new(self.y, self.w, self.x, self.w) }
    #[inline(always)]
    fn ywyx(self) -> Self { Self::new(self.y, self.w, self.y, self.x) }
    #[inline(always)]
    fn ywyy(self) -> Self { Self::new(self.y, self.w, self.y, self.y) }
    #[inline(always)]
    fn ywyz(self) -> Self { Self::new(self.y, self.w, self.y, self.z) }
    #[inline(always)]
    fn ywyw(self) -> Self { Self::new(self.y, self.w, self.y, self.w) }
    #[inline(always)]
    fn ywzx(self) -> Self { Self::new(self.y, self.w, self.z, self.x) }
    #[inline(always)]
    fn ywzy(self) -> Self { Self::new(self.y, self.w, self.z, self.y) }
    #[inline(always)]
    fn ywzz(self) -> Self { Self::new(self.y, self.w, self.z, self.z) }
    #[inline(always)]
    fn ywzw(self) -> Self { Self::new(self.y, self.w, self.z, self.w) }
    #[inline(always)]
    fn ywwx(self) -> Self { Self::new(self.y, self.w, self.w, self.x) }
    #[inline(always)]
    fn ywwy(self) -> Self { Self::new(self.y, self.w, self.w, self.y) }
    #[inline(always)]
    fn ywwz(self) -> Self { Self::new(self.y, self.w, self.w, self.z) }
    #[inline(always)]
    fn ywww(self) -> Self { Self::new(self.y, self.w, self.w, self.w) }
    #[inline(always)]
    fn zxxx(self) -> Self { Self::new(self.z, self.x, self.x, self.x) }
    #[inline(always)]
    fn zxxy(self) -> Self { Self::new(self.z, self.x, self.x, self.y) }
    #[inline(always)]
    fn zxxz(self) -> Self { Self::new(self.z, self.x, self.x, self.z) }
    #[inline(always)]
    fn zxxw(self) -> Self { Self::new(self.z, self.x, self.x, self.w) }
    #[inline(always)]
    fn zxyx(self) -> Self { Self::new(self.z, self.x, self.y, self.x) }
    #[inline(always)]
    fn zxyy(self) -> Self { Self::new(self.z, self.x, self.y, self.y) }
    #[inline(always)]
    fn zxyz(self) -> Self { Self::new(self.z, self.x, self.y, self.z) }
    #[inline(always)]
    fn zxyw(self) -> Self { Self::new(self.z, self.x, self.y, self.w) }
    #[inline(always)]
    fn zxzx(self) -> Self { Self::new(self.z, self.x, self.z, self.x) }
    #[inline(always)]
    fn zxzy(self) -> Self { Self::new(self.z, self.x, self.z, self.y) }
    #[inline(always)]
    fn zxzz(self) -> Self { Self::new(self.z, self.x, self.z, self.z) }
    #[inline(always)]
    fn zxzw(self) -> Self { Self::new(self.z, self.x, self.z, self.w) }
    #[inline(always)]
    fn zxwx(self) -> Self { Self::new(self.z, self.x, self.w, self.x) }
    #[inline(always)]
    fn zxwy(self) -> Self { Self::new(self.z, self.x, self.w, self.y) }
    #[inline(always)]
    fn zxwz(self) -> Self { Self::new(self.z, self.x, self.w, self.z) }
    #[inline(always)]
    fn zxww(self) -> Self { Self::new(self.z, self.x, self.w, self.w) }
    #[inline(always)]
    fn zyxx(self) -> Self { Self::new(self.z, self.y, self.x, self.x) }
    #[inline(always)]
    fn zyxy(self) -> Self { Self::new(self.z, self.y, self.x, self.y) }
    #[inline(always)]
    fn zyxz(self) -> Self { Self::new(self.z, self.y, self.x, self.z) }
    #[inline(always)]
    fn zyxw(self) -> Self { Self::new(self.z, self.y, self.x, self.w) }
    #[inline(always)]
    fn zyyx(self) -> Self { Self::new(self.z, self.y, self.y, self.x) }
    #[inline(always)]
    fn zyyy(self) -> Self { Self::new(self.z, self.y, self.y, self.y) }
    #[inline(always)]
    fn zyyz(self) -> Self { Self::new(self.z, self.y, self.y, self.z) }
    #[inline(always)]
    fn zyyw(self) -> Self { Self::new(self.z, self.y, self.y, self.w) }
    #[inline(always)]
    fn zyzx(self) -> Self { Self::new(self.z, self.y, self.z, self.x) }
    #[inline(always)]
    fn zyzy(self) -> Self { Self::new(self.z, self.y, self.z, self.y) }
    #[inline(always)]
    fn zyzz(self) -> Self { Self::new(self.z, self.y, self.z, self.z) }
    #[inline(always)]
    fn zyzw(self) -> Self { Self::new(self.z, self.y, self.z, self.w) }
    #[inline(always)]
    fn zywx(self) -> Self { Self::new(self.z, self.y, self.w, self.x) }
    #[inline(always)]
    fn zywy(self) -> Self { Self::new(self.z, self.y, self.w, self.y) }
    #[inline(always)]
    fn zywz(self) -> Self { Self::new(self.z, self.y, self.w, self.z) }
    #[inline(always)]
    fn zyww(self) -> Self { Self::new(self.z, self.y, self.w, self.w) }
    #[inline(always)]
    fn zzxx(self) -> Self { Self::new(self.z, self.z, self.x, self.x) }
    #[inline(always)]
    fn zzxy(self) -> Self { Self::new(self.z, self.z, self.x, self.y) }
    #[inline(always)]
    fn zzxz(self) -> Self { Self::new(self.z, self.z, self.x, self.z) }
    #[inline(always)]
    fn zzxw(self) -> Self { Self::new(self.z, self.z, self.x, self.w) }
    #[inline(always)]
    fn zzyx(self) -> Self { Self::new(self.z, self.z, self.y, self.x) }
    #[inline(always)]
    fn zzyy(self) -> Self { Self::new(self.z, self.z, self.y, self.y) }
    #[inline(always)]
    fn zzyz(self) -> Self { Self::new(self.z, self.z, self.y, self.z) }
    #[inline(always)]
    fn zzyw(self) -> Self { Self::new(self.z, self.z, self.y, self.w) }
    #[inline(always)]
    fn zzzx(self) -> Self { Self::new(self.z, self.z, self.z, self.x) }
    #[inline(always)]
    fn zzzy(self) -> Self { Self::new(self.z, self.z, self.z, self.y) }
    #[inline(always)]
    fn zzzz(self) -> Self { Self::new(self.z, self.z, self.z, self.z) }
    #[inline(always)]
    fn zzzw(self) -> Self { Self::new(self.z, self.z, self.z, self.w) }
    #[inline(always)]
    fn zzwx(self) -> Self { Self::new(self.z, self.z, self.w, self.x) }
    #[inline(always)]
    fn zzwy(self) -> Self { Self::new(self.z, self.z, self.w, self.y) }
    #[inline(always)]
    fn zzwz(self) -> Self { Self::new(self.z, self.z, self.w, self.z) }
    #[inline(always)]
    fn zzww(self) -> Self { Self::new(self.z, self.z, self.w, self.w) }
    #[inline(always)]
    fn zwxx(self) -> Self { Self::new(self.z, self.w, self.x, self.x) }
    #[inline(always)]
    fn zwxy(self) -> Self { Self::new(self.z, self.w, self.x, self.y) }
    #[inline(always)]
    fn zwxz(self) -> Self { Self::new(self.z, self.w, self.x, self.z) }
    #[inline(always)]
    fn zwxw(self) -> Self { Self::new(self.z, self.w, self.x, self.w) }
    #[inline(always)]
    fn zwyx(self) -> Self { Self::new(self.z, self.w, self.y, self.x) }
    #[inline(always)]
    fn zwyy(self) -> Self { Self::new(self.z, self.w, self.y, self.y) }
    #[inline(always)]
    fn zwyz(self) -> Self { Self::new(self.z, self.w, self.y, self.z) }
    #[inline(always)]
    fn zwyw(self) -> Self { Self::new(self.z, self.w, self.y, self.w) }
    #[inline(always)]
    fn zwzx(self) -> Self { Self::new(self.z, self.w, self.z, self.x) }
    #[inline(always)]
    fn zwzy(self) -> Self { Self::new(self.z, self.w, self.z, self.y) }
    #[inline(always)]
    fn zwzz(self) -> Self { Self::new(self.z, self.w, self.z, self.z) }
    #[inline(always)]
    fn zwzw(self) -> Self { Self::new(self.z, self.w, self.z, self.w) }
    #[inline(always)]
    fn zwwx(self) -> Self { Self::new(self.z, self.w, self.w, self.x) }
    #[inline(always)]
    fn zwwy(self) -> Self { Self::new(self.z, self.w, self.w, self.y) }
    #[inline(always)]
    fn zwwz(self) -> Self { Self::new(self.z, self.w, self.w, self.z) }
    #[inline(always)]
    fn zwww(self) -> Self { Self::new(self.z, self.w, self.w, self.w) }
    #[inline(always)]
    fn wxxx(self) -> Self { Self::new(self.w, self.x, self.x, self.x) }
    #[inline(always)]
    fn wxxy(self) -> Self { Self::new(self.w, self.x, self.x, self.y) }
    #[inline(always)]
    fn wxxz(self) -> Self { Self::new(self.w, self.x, self.x, self.z) }
    #[inline(always)]
    fn wxxw(self) -> Self { Self::new(self.w, self.x, self.x, self.w) }
    #[inline(always)]
    fn wxyx(self) -> Self { Self::new(self.w, self.x, self.y, self.x) }
    #[inline(always)]
    fn wxyy(self) -> Self { Self::new(self.w, self.x, self.y, self.y) }
    #[inline(always)]
    fn wxyz(self) -> Self { Self::new(self.w, self.x, self.y, self.z) }
    #[inline(always)]
    fn wxyw(self) -> Self { Self::new(self.w, self.x, self.y, self.w) }
    #[inline(always)]
    fn wxzx(self) -> Self { Self::new(self.w, self.x, self.z, self.x) }
    #[inline(always)]
    fn wxzy(self) -> Self { Self::new(self.w, self.x, self.z, self.y) }
    #[inline(always)]
    fn wxzz(self) -> Self { Self::new(self.w, self.x, self.z, self.z) }
    #[inline(always)]
    fn wxzw(self) -> Self { Self::new(self.w, self.x, self.z, self.w) }
    #[inline(always)]
    fn wxwx(self) -> Self { Self::new(self.w, self.x, self.w, self.x) }
    #[inline(always)]
    fn wxwy(self) -> Self { Self::new(self.w, self.x, self.w, self.y) }
    #[inline(always)]
    fn wxwz(self) -> Self { Self::new(self.w, self.x, self.w, self.z) }
    #[inline(always)]
    fn wxww(self) -> Self { Self::new(self.w, self.x, self.w, self.w) }
    #[inline(always)]
    fn wyxx(self) -> Self { Self::new(self.w, self.y, self.x, self.x) }
    #[inline(always)]
    fn wyxy(self) -> Self { Self::new(self.w, self.y, self.x, self.y) }
    #[inline(always)]
    fn wyxz(self) -> Self { Self::new(self.w, self.y, self.x, self.z) }
    #[inline(always)]
    fn wyxw(self) -> Self { Self::new(self.w, self.y, self.x, self.w) }
    #[inline(always)]
    fn wyyx(self) -> Self { Self::new(self.w, self.y, self.y, self.x) }
    #[inline(always)]
    fn wyyy(self) -> Self { Self::new(self.w, self.y, self.y, self.y) }
    #[inline(always)]
    fn wyyz(self) -> Self { Self::new(self.w, self.y, self.y, self.z) }
    #[inline(always)]
    fn wyyw(self) -> Self { Self::new(self.w, self.y, self.y, self.w) }
    #[inline(always)]
    fn wyzx(self) -> Self { Self::new(self.w, self.y, self.z, self.x) }
    #[inline(always)]
    fn wyzy(self) -> Self { Self::new(self.w, self.y, self.z, self.y) }
    #[inline(always)]
    fn wyzz(self) -> Self { Self::new(self.w, self.y, self.z, self.z) }
    #[inline(always)]
    fn wyzw(self) -> Self { Self::new(self.w, self.y, self.z, self.w) }
    #[inline(always)]
    fn wywx(self) -> Self { Self::new(self.w, self.y, self.w, self.x) }
    #[inline(always)]
    fn wywy(self) -> Self { Self::new(self.w, self.y, self.w, self.y) }
    #[inline(always)]
    fn wywz(self) -> Self { Self::new(self.w, self.y, self.w, self.z) }
    #[inline(always)]
    fn wyww(self) -> Self { Self::new(self.w, self.y, self.w, self.w) }
    #[inline(always)]
    fn wzxx(self) -> Self { Self::new(self.w, self.z, self.x, self.x) }
    #[inline(always)]
    fn wzxy(self) -> Self { Self::new(self.w, self.z, self.x, self.y) }
    #[inline(always)]
    fn wzxz(self) -> Self { Self::new(self.w, self.z, self.x, self.z) }
    #[inline(always)]
    fn wzxw(self) -> Self { Self::new(self.w, self.z, self.x, self.w) }
    #[inline(always)]
    fn wzyx(self) -> Self { Self::new(self.w, self.z, self.y, self.x) }
    #[inline(always)]
    fn wzyy(self) -> Self { Self::new(self.w, self.z, self.y, self.y) }
    #[inline(always)]
    fn wzyz(self) -> Self { Self::new(self.w, self.z, self.y, self.z) }
    #[inline(always)]
    fn wzyw(self) -> Self { Self::new(self.w, self.z, self.y, self.w) }
    #[inline(always)]
    fn wzzx(self) -> Self { Self::new(self.w, self.z, self.z, self.x) }
    #[inline(always)]
    fn wzzy(self) -> Self { Self::new(self.w, self.z, self.z, self.y) }
    #[inline(always)]
    fn wzzz(self) -> Self { Self::new(self.w, self.z, self.z, self.z) }
    #[inline(always)]
    fn wzzw(self) -> Self { Self::new(self.w, self.z, self.z, self.w) }
    #[inline(always)]
    fn wzwx(self) -> Self { Self::new(self.w, self.z, self.w, self.x) }
    #[inline(always)]
    fn wzwy(self) -> Self { Self::new(self.w, self.z, self.w, self.y) }
    #[inline(always)]
    fn wzwz(self) -> Self { Self::new(self.w, self.z, self.w, self.z) }
    #[inline(always)]
    fn wzww(self) -> Self { Self::new(self.w, self.z, self.w, self.w) }
    #[inline(always)]
    fn wwxx(self) -> Self { Self::new(self.w, self.w, self.x, self.x) }
    #[inline(always)]
    fn wwxy(self) -> Self { Self::new(self.w, self.w, self.x, self.y) }
    #[inline(always)]
    fn wwxz(self) -> Self { Self::new(self.w, self.w, self.x, self.z) }
    #[inline(always)]
    fn wwxw(self) -> Self { Self::new(self.w, self.w, self.x, self.w) }
    #[inline(always)]
    fn wwyx(self) -> Self { Self::new(self.w, self.w, self.y, self.x) }
    #[inline(always)]
    fn wwyy(self) -> Self { Self::new(self.w, self.w, self.y, self.y) }
    #[inline(always)]
    fn wwyz(self) -> Self { Self::new(self.w, self.w, self.y, self.z) }
    #[inline(always)]
    fn wwyw(self) -> Self { Self::new(self.w, self.w, self.y, self.w) }
    #[inline(always)]
    fn wwzx(self) -> Self { Self::new(self.w, self.w, self.z, self.x) }
    #[inline(always)]
    fn wwzy(self) -> Self { Self::new(self.w, self.w, self.z, self.y) }
    #[inline(always)]
    fn wwzz(self) -> Self { Self::new(self.w, self.w, self.z, self.z) }
    #[inline(always)]
    fn wwzw(self) -> Self { Self::new(self.w, self.w, self.z, self.w) }
    #[inline(always)]
    fn wwwx(self) -> Self { Self::new(self.w, self.w, self.w, self.x) }
    #[inline(always)]
    fn wwwy(self) -> Self { Self::new(self.w, self.w, self.w, self.y) }
    #[inline(always)]
    fn wwwz(self) -> Self { Self::new(self.w, self.w, self.w, self.z) }
    #[inline(always)]
    fn wwww(self) -> Self { Self::new(self.w, self.w, self.w, self.w) }

        }
    };
}

