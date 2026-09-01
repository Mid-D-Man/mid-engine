// crates/mid-math/src/f32/scalar/vec3.rs
//! Scalar Vec3 — fallback for non-SIMD targets and correctness reference.
//!
//! This is the implementation used on targets without SSE2/NEON/WASM SIMD.
//! On x86/x86_64, `sse2/vec3.rs` supersedes this — additions made here
//! must be mirrored there too (and to neon/vec3.rs, wasm/vec3.rs) for parity.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{BVec3, EPSILON};
use crate::f32::vec2::Vec2;
use crate::f32::scalar::vec4::Vec4;

/// 3D vector. 16 bytes (12 data + 4 pad), align(16). Scalar storage.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl Vec3 {
    pub const ZERO:         Self = Self { x: 0.0,  y: 0.0,  z: 0.0,  _pad: 0.0 };
    pub const ONE:          Self = Self { x: 1.0,  y: 1.0,  z: 1.0,  _pad: 0.0 };
    pub const NEG_ONE:      Self = Self { x:-1.0,  y:-1.0,  z:-1.0,  _pad: 0.0 };
    pub const X:            Self = Self { x: 1.0,  y: 0.0,  z: 0.0,  _pad: 0.0 };
    pub const Y:            Self = Self { x: 0.0,  y: 1.0,  z: 0.0,  _pad: 0.0 };
    pub const Z:            Self = Self { x: 0.0,  y: 0.0,  z: 1.0,  _pad: 0.0 };
    pub const NEG_X:        Self = Self { x:-1.0,  y: 0.0,  z: 0.0,  _pad: 0.0 };
    pub const NEG_Y:        Self = Self { x: 0.0,  y:-1.0,  z: 0.0,  _pad: 0.0 };
    pub const NEG_Z:        Self = Self { x: 0.0,  y: 0.0,  z:-1.0,  _pad: 0.0 };
    pub const MIN:          Self = Self { x: f32::MIN, y: f32::MIN, z: f32::MIN, _pad: 0.0 };
    pub const MAX:          Self = Self { x: f32::MAX, y: f32::MAX, z: f32::MAX, _pad: 0.0 };
    pub const NAN:          Self = Self { x: f32::NAN, y: f32::NAN, z: f32::NAN, _pad: 0.0 };
    pub const INFINITY:     Self = Self { x: f32::INFINITY,     y: f32::INFINITY,     z: f32::INFINITY,     _pad: 0.0 };
    pub const NEG_INFINITY: Self = Self { x: f32::NEG_INFINITY, y: f32::NEG_INFINITY, z: f32::NEG_INFINITY, _pad: 0.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, _pad: 0.0 } }
    #[inline(always)] pub fn splat(v: f32) -> Self { Self::new(v, v, v) }
    #[inline(always)] pub fn from_array(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 3] { [self.x, self.y, self.z] }

    /// Apply a closure to each component.
    #[inline]
    pub fn map<F: Fn(f32) -> f32>(self, f: F) -> Self { Self::new(f(self.x), f(self.y), f(self.z)) }

    /// Conditional select component-wise based on `mask`.
    #[inline]
    pub fn select(mask: BVec3, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
        )
    }

    /// Write components to `slice[0..3]`. Panics if slice is too short.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x; slice[1] = self.y; slice[2] = self.z;
    }

    #[inline(always)] pub fn with_x(mut self, x: f32) -> Self { self.x = x; self }
    #[inline(always)] pub fn with_y(mut self, y: f32) -> Self { self.y = y; self }
    #[inline(always)] pub fn with_z(mut self, z: f32) -> Self { self.z = z; self }

    /// Extend to Vec4 by appending `w`.
    #[inline(always)]
    pub fn extend(self, w: f32) -> super::vec4::Vec4 { super::vec4::Vec4::new(self.x, self.y, self.z, w) }

    /// Truncate to Vec2 by dropping `z`.
    #[inline(always)]
    pub fn truncate(self) -> crate::f32::vec2::Vec2 { crate::f32::vec2::Vec2::new(self.x, self.y) }

    /// Build from a homogeneous Vec4 — divides xyz by w. Returns ZERO if w ≈ 0.
    #[inline]
    pub fn from_homogeneous(v: super::vec4::Vec4) -> Self {
        if v.w.abs() < EPSILON { Self::ZERO } else {
            let w_recip = 1.0 / v.w;
            Self::new(v.x * w_recip, v.y * w_recip, v.z * w_recip)
        }
    }

    /// Convert to homogeneous Vec4 with `w = 1.0`.
    #[inline(always)]
    pub fn to_homogeneous(self) -> super::vec4::Vec4 { self.extend(1.0) }

    // ── Dot / cross ───────────────────────────────────────────────────────────

    #[inline(always)] pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    /// Dot product broadcast into a `Vec3`.
    #[inline(always)] pub fn dot_into_vec(self, r: Self) -> Self { Self::splat(self.dot(r)) }
    #[inline(always)]
    pub fn cross(self, r: Self) -> Self {
        Self::new(
            self.y*r.z - self.z*r.y,
            self.z*r.x - self.x*r.z,
            self.x*r.y - self.y*r.x,
        )
    }

    // ── Length ────────────────────────────────────────────────────────────────

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }
    /// Alias for `length_sq` (glam-compat name).
    #[inline(always)] pub fn length_squared(self) -> f32 { self.length_sq() }
    #[inline(always)] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline(always)]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    // ── Normalize ─────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }
    #[inline(always)]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }
    /// Normalize, returning `fallback` if zero-length.
    #[inline(always)]
    pub fn normalize_or(self, fallback: Self) -> Self { self.try_normalize().unwrap_or(fallback) }
    #[inline(always)] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline(always)] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }
    /// Normalize and return `(normalized, original_length)` in one pass.
    #[inline]
    pub fn normalize_and_length(self) -> (Self, f32) {
        let l = self.length();
        if l < EPSILON { (Self::ZERO, 0.0) } else { (self / l, l) }
    }

    // ── Distance ──────────────────────────────────────────────────────────────

    #[inline(always)] pub fn distance(self, r: Self) -> f32 { (self - r).length() }
    #[inline(always)] pub fn distance_sq(self, r: Self) -> f32 { (self - r).length_sq() }
    /// Alias for `distance_sq` (glam-compat name).
    #[inline(always)] pub fn distance_squared(self, r: Self) -> f32 { self.distance_sq(r) }

    // ── Reduction ─────────────────────────────────────────────────────────────

    #[inline(always)] pub fn min_element(self) -> f32 { self.x.min(self.y).min(self.z) }
    #[inline(always)] pub fn max_element(self) -> f32 { self.x.max(self.y).max(self.z) }
    #[inline(always)] pub fn element_sum(self) -> f32 { self.x + self.y + self.z }
    #[inline(always)] pub fn element_product(self) -> f32 { self.x * self.y * self.z }
    /// Index (0-2) of the minimum component.
    #[inline]
    pub fn min_position(self) -> usize {
        if self.x <= self.y && self.x <= self.z { 0 } else if self.y <= self.z { 1 } else { 2 }
    }
    /// Index (0-2) of the maximum component.
    #[inline]
    pub fn max_position(self) -> usize {
        if self.x >= self.y && self.x >= self.z { 0 } else if self.y >= self.z { 1 } else { 2 }
    }

    // ── Comparisons → BVec3 ───────────────────────────────────────────────────

    #[inline(always)] pub fn cmpeq(self, r: Self) -> BVec3 { BVec3::new(self.x==r.x, self.y==r.y, self.z==r.z) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> BVec3 { BVec3::new(self.x!=r.x, self.y!=r.y, self.z!=r.z) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> BVec3 { BVec3::new(self.x>=r.x, self.y>=r.y, self.z>=r.z) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> BVec3 { BVec3::new(self.x> r.x, self.y> r.y, self.z> r.z) }
    #[inline(always)] pub fn cmple(self, r: Self) -> BVec3 { BVec3::new(self.x<=r.x, self.y<=r.y, self.z<=r.z) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> BVec3 { BVec3::new(self.x< r.x, self.y< r.y, self.z< r.z) }

    // ── Sign ──────────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum(), self.z.signum()) }
    #[inline(always)]
    pub fn copysign(self, rhs: Self) -> Self {
        Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y), self.z.copysign(rhs.z))
    }
    /// Bitmask of sign bits: bit 0=x, bit 1=y, bit 2=z.
    #[inline]
    pub fn is_negative_bitmask(self) -> u32 {
          (self.x.is_sign_negative() as u32)
        | ((self.y.is_sign_negative() as u32) << 1)
        | ((self.z.is_sign_negative() as u32) << 2)
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn is_finite(self) -> bool { self.x.is_finite() && self.y.is_finite() && self.z.is_finite() }
    #[inline(always)]
    pub fn is_nan(self) -> bool { self.x.is_nan() || self.y.is_nan() || self.z.is_nan() }
    #[inline(always)]
    pub fn is_finite_mask(self) -> BVec3 { BVec3::new(self.x.is_finite(), self.y.is_finite(), self.z.is_finite()) }
    #[inline(always)]
    pub fn is_nan_mask(self) -> BVec3 { BVec3::new(self.x.is_nan(), self.y.is_nan(), self.z.is_nan()) }

    // ── Component-wise math ───────────────────────────────────────────────────

    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(),   self.y.abs(),   self.z.abs())   }
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y), self.z.min(r.z)) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y), self.z.max(r.z)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor(), self.z.floor()) }
    #[inline(always)] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil(),  self.z.ceil())  }
    #[inline(always)] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round(), self.z.round()) }
    #[inline(always)] pub fn trunc(self) -> Self { Self::new(self.x.trunc(), self.y.trunc(), self.z.trunc()) }
    #[inline(always)] pub fn fract(self) -> Self { Self::new(self.x.fract(), self.y.fract(), self.z.fract()) }
    /// GLSL `fract` — always non-negative.
    #[inline(always)] pub fn fract_gl(self) -> Self { self - self.floor() }
    /// Heaviside step: 0.0 if `self < rhs`, else 1.0.
    #[inline(always)]
    pub fn step(self, rhs: Self) -> Self {
        Self::new(
            if self.x < rhs.x { 0.0 } else { 1.0 },
            if self.y < rhs.y { 0.0 } else { 1.0 },
            if self.z < rhs.z { 0.0 } else { 1.0 },
        )
    }
    /// Clamp each component to `[0.0, 1.0]`.
    #[inline(always)] pub fn saturate(self) -> Self { self.clamp(Self::ZERO, Self::ONE) }
    #[inline(always)] pub fn recip(self) -> Self { Self::new(self.x.recip(), self.y.recip(), self.z.recip()) }
    #[inline(always)] pub fn sqrt(self)  -> Self { Self::new(self.x.sqrt(),  self.y.sqrt(),  self.z.sqrt())  }
    #[inline(always)] pub fn exp(self)   -> Self { Self::new(self.x.exp(),   self.y.exp(),   self.z.exp())   }
    #[inline(always)] pub fn exp2(self)  -> Self { Self::new(self.x.exp2(),  self.y.exp2(),  self.z.exp2())  }
    #[inline(always)] pub fn ln(self)    -> Self { Self::new(self.x.ln(),    self.y.ln(),    self.z.ln())    }
    #[inline(always)] pub fn log2(self)  -> Self { Self::new(self.x.log2(),  self.y.log2(),  self.z.log2())  }
    #[inline(always)] pub fn powf(self, n: f32) -> Self { Self::new(self.x.powf(n), self.y.powf(n), self.z.powf(n)) }
    #[inline(always)] pub fn sin(self)   -> Self { Self::new(self.x.sin(),   self.y.sin(),   self.z.sin())   }
    #[inline(always)] pub fn cos(self)   -> Self { Self::new(self.x.cos(),   self.y.cos(),   self.z.cos())   }
    #[inline(always)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (sx, cx) = self.x.sin_cos();
        let (sy, cy) = self.y.sin_cos();
        let (sz, cz) = self.z.sin_cos();
        (Self::new(sx, sy, sz), Self::new(cx, cy, cz))
    }
    #[inline(always)]
    pub fn div_euclid(self, rhs: Self) -> Self {
        Self::new(self.x.div_euclid(rhs.x), self.y.div_euclid(rhs.y), self.z.div_euclid(rhs.z))
    }
    #[inline(always)]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        Self::new(self.x.rem_euclid(rhs.x), self.y.rem_euclid(rhs.y), self.z.rem_euclid(rhs.z))
    }

    // ── FMA ───────────────────────────────────────────────────────────────────

    /// Fused multiply-add: `self * a + b`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.x.mul_add(a.x, b.x), self.y.mul_add(a.y, b.y), self.z.mul_add(a.z, b.z))
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    #[inline(always)] pub fn lerp(self, r: Self, t: f32) -> Self { self + (r - self) * t }
    #[inline(always)] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    /// Move toward `target` by at most `max_dist`. Never overshoots.
    #[inline(always)]
    pub fn move_towards(self, target: Self, max_dist: f32) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < EPSILON { target } else { self + d / len * max_dist }
    }

    /// Reflect `self` off a surface defined by unit `normal`.
    #[inline(always)] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }

    /// Refract `self` through a surface with unit `normal` and ratio `eta` (n_i/n_t).
    /// Returns `ZERO` on total internal reflection.
    #[inline]
    pub fn refract(self, normal: Self, eta: f32) -> Self {
        let n_dot_i = normal.dot(self);
        let k = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
        if k < 0.0 { Self::ZERO } else { self * eta - normal * (eta * n_dot_i + k.sqrt()) }
    }

    /// Clamp vector length to `[min, max]`.
    #[inline(always)]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let len = self.length();
        if len < EPSILON { return Self::ZERO; }
        let c = len.clamp(min, max);
        if (c - len).abs() < EPSILON { self } else { self * (c / len) }
    }
    #[inline(always)]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let len = self.length();
        if len > max && len > EPSILON { self * (max / len) } else { self }
    }
    #[inline(always)]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let len = self.length();
        if len < min && len > EPSILON { self * (min / len) } else { self }
    }

    /// Project `self` onto `rhs`. Returns `ZERO` if `rhs` is zero-length.
    #[inline(always)]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < EPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }
    #[inline(always)] pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }
    /// Project assuming `rhs` is already unit length (no division).
    #[inline(always)] pub fn project_onto_normalized(self, rhs: Self) -> Self { rhs * self.dot(rhs) }
    #[inline(always)] pub fn reject_from_normalized(self, rhs: Self) -> Self { self - self.project_onto_normalized(rhs) }

    /// Angle in radians between `self` and `rhs`. Returns `0.0` for zero-length inputs.
    #[inline(always)]
    pub fn angle_between(self, rhs: Self) -> f32 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < EPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    #[inline(always)] pub fn is_parallel(self, rhs: Self) -> bool { self.cross(rhs).length_sq() < EPSILON * EPSILON }
    #[inline(always)] pub fn is_perpendicular(self, rhs: Self) -> bool { self.dot(rhs).abs() < EPSILON }

    // ── Rotation ──────────────────────────────────────────────────────────────

    /// Rotate around the +X axis by `angle` radians.
    #[inline]
    pub fn rotate_x(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(self.x, self.y * c - self.z * s, self.y * s + self.z * c)
    }

    /// Rotate around the +Y axis by `angle` radians.
    #[inline]
    pub fn rotate_y(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(self.x * c + self.z * s, self.y, -self.x * s + self.z * c)
    }

    /// Rotate around the +Z axis by `angle` radians.
    #[inline]
    pub fn rotate_z(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c, self.z)
    }

    /// Rotate around an arbitrary `axis` by `angle` radians (Rodrigues' formula).
    /// `axis` is normalized internally.
    #[inline]
    pub fn rotate_axis(self, axis: Self, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        let k = axis.normalize();
        // v·cos + (k×v)·sin + k·(k·v)·(1-cos)
        self * cos + k.cross(self) * sin + k * k.dot(self) * (1.0 - cos)
    }

    /// Rotate toward `rhs` by at most `max_angle` radians using slerp.
    #[inline]
    pub fn rotate_towards(self, rhs: Self, max_angle: f32) -> Self {
        let angle = self.angle_between(rhs);
        if angle < EPSILON { return self; }
        self.slerp(rhs, (max_angle / angle).clamp(0.0, 1.0))
    }

    // ── Slerp ────────────────────────────────────────────────────────────────

    /// Spherical linear interpolation — preserves both length and angular path.
    ///
    /// Interpolates vector length linearly while keeping the rotation arc smooth.
    /// Falls back to `lerp` when vectors are (anti)parallel to avoid NaN.
    #[inline]
    pub fn slerp(self, rhs: Self, s: f32) -> Self {
        let self_len = self.length();
        let rhs_len  = rhs.length();
        // Guard zero-length inputs
        if self_len < EPSILON || rhs_len < EPSILON {
            return self.lerp(rhs, s);
        }
        let dot = (self.dot(rhs) / (self_len * rhs_len)).clamp(-1.0, 1.0);
        // Fall back to lerp when nearly parallel or anti-parallel
        if dot.abs() >= 1.0 - 3e-7 {
            return self.lerp(rhs, s);
        }
        let theta     = dot.acos();
        let sin_theta = theta.sin();
        let t0 = ((1.0 - s) * theta).sin() / sin_theta;
        let t1 = (s * theta).sin() / sin_theta;
        // Interpolate direction, then restore interpolated length
        let result_len = self_len + (rhs_len - self_len) * s;
        (self.normalize() * t0 + rhs.normalize() * t1) * result_len
    }

    // ── Orthogonal helpers ────────────────────────────────────────────────────

    /// Return some vector orthogonal to `self`. Undefined which one — fast, not stable.
    #[inline]
    pub fn any_orthogonal_vector(self) -> Self {
        // Pick the build axis least aligned to self to minimise cancellation.
        if self.x.abs() > self.y.abs() {
            Self::new(-self.z, 0.0, self.x) // ≈ self × Y
        } else {
            Self::new(0.0, self.z, -self.y) // ≈ self × X
        }
    }

    /// Return a unit vector orthogonal to `self`.
    #[inline]
    pub fn any_orthonormal_vector(self) -> Self {
        self.any_orthogonal_vector().normalize()
    }

    /// Return two orthonormal vectors perpendicular to `self` (and to each other).
    /// `self` is assumed to be unit length. Efficient: only one normalize call.
    #[inline]
    pub fn any_orthonormal_pair(self) -> (Self, Self) {
        let t = self.any_orthonormal_vector();
        let b = self.cross(t);
        (t, b)
    }

    // ── Spherical coordinates ─────────────────────────────────────────────────

    /// Convert to spherical `(radius, theta, phi)`.
    /// `theta` = polar angle from +Z ∈ `[0, π]`. `phi` = azimuthal from +X ∈ `[-π, π]`.
    #[inline]
    pub fn to_spherical(self) -> (f32, f32, f32) {
        let r = self.length();
        if r < EPSILON { return (0.0, 0.0, 0.0); }
        let theta = (self.z / r).clamp(-1.0, 1.0).acos();
        let phi   = self.y.atan2(self.x);
        (r, theta, phi)
    }

    /// Build from spherical `(r, theta, phi)`.
    #[inline]
    pub fn from_spherical(r: f32, theta: f32, phi: f32) -> Self {
        let sin_theta = theta.sin();
        Self::new(r * sin_theta * phi.cos(), r * sin_theta * phi.sin(), r * theta.cos())
    }

    // ── Approx equality ───────────────────────────────────────────────────────

    #[inline(always)]
    pub fn approx_eq(self, r: Self) -> bool {
        (self.x-r.x).abs() < EPSILON && (self.y-r.y).abs() < EPSILON && (self.z-r.z).abs() < EPSILON
    }
    /// Approximate equality with explicit `max_abs_diff` (glam-compat name).
    #[inline(always)]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        (self.x-rhs.x).abs() < max_abs_diff
            && (self.y-rhs.y).abs() < max_abs_diff
            && (self.z-rhs.z).abs() < max_abs_diff
    }
}

// ── PartialEq / Default / Display ─────────────────────────────────────────────

impl PartialEq for Vec3 {
    fn eq(&self, r: &Self) -> bool { self.x == r.x && self.y == r.y && self.z == r.z }
}
impl Default for Vec3 { fn default() -> Self { Self::ZERO } }
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── Arithmetic operators ──────────────────────────────────────────────────────

impl Add  for Vec3 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)} }
impl Sub  for Vec3 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)} }
impl Neg  for Vec3 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)} }
// scalar × Vec3
impl Mul<f32> for Vec3 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)} }
impl Mul<Vec3> for f32 { type Output=Vec3; #[inline(always)] fn mul(self,v:Vec3)->Vec3{Vec3::new(self*v.x,self*v.y,self*v.z)} }
// component-wise Vec3 × Vec3
impl Mul for Vec3 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y,self.z*r.z)} }
impl Div<f32> for Vec3 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s)} }
impl Div for Vec3 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z)} }

impl AddAssign for Vec3 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;} }
impl SubAssign for Vec3 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;} }
impl MulAssign<f32> for Vec3 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;self.z*=s;} }
impl MulAssign for Vec3 { #[inline(always)] fn mul_assign(&mut self,r:Self){self.x*=r.x;self.y*=r.y;self.z*=r.z;} }
impl DivAssign<f32> for Vec3 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;self.z/=s;} }
impl DivAssign for Vec3 { #[inline(always)] fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;self.z/=r.z;} }

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f32;3]> for Vec3   { fn from(a:[f32;3])->Self{Self::new(a[0],a[1],a[2])} }
impl From<Vec3> for [f32;3]   { fn from(v:Vec3)->[f32;3]{[v.x,v.y,v.z]} }
impl From<(f32,f32,f32)> for Vec3 { fn from(t:(f32,f32,f32))->Self{Self::new(t.0,t.1,t.2)} }
impl From<Vec3> for (f32,f32,f32) { fn from(v:Vec3)->(f32,f32,f32){(v.x,v.y,v.z)} }

// ── Swizzle ────────────────────────────────────────────────────────────────

crate::impl_vec3_swizzle!(Vec3, Vec2, Vec4);
