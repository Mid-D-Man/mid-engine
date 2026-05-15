// crates/mid-math/src/rotor.rs
//! Rotor3 — 3D rotation in Geometric Algebra (PGA/VGA).
//!
//! A rotor R = s + b_yz·e₂₃ + b_xz·e₁₃ + b_xy·e₁₂ represents a rotation
//! exactly like a quaternion, but derived from the geometric product of two
//! unit vectors: R = v₂v₁ rotates by twice the angle between v₁ and v₂.
//!
//! Advantages over quaternions:
//!   - Geometric interpretation: rotate by "the plane" rather than "the axis"
//!   - Generalises cleanly to N dimensions (no magic i,j,k)
//!   - More intuitive construction from two vectors or a plane+angle
//!
//! Correspondence to Quat: s↔w, b_yz↔x, -b_xz↔y, b_xy↔z.
//! Conversion is zero-cost — same memory layout as Quat.
//!
//! Reference: Dorst, Fontijne, Mann "Geometric Algebra for Computer Science"

use core::fmt;
use core::ops::{Mul, MulAssign, Neg};
use crate::{Quat, Vec3, EPSILON};

/// 3D rotor in Geometric Algebra. 16 bytes, 16-byte aligned.
///
/// R = s + b_yz·e₂₃ + b_xz·e₁₃ + b_xy·e₁₂
/// Unit rotor: s² + b_yz² + b_xz² + b_xy² = 1
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Rotor3 {
    /// Scalar part.
    pub s:    f32,
    /// Bivector e₂₃ (yz plane = rotation about x).
    pub b_yz: f32,
    /// Bivector e₁₃ (xz plane = rotation about y, note sign convention).
    pub b_xz: f32,
    /// Bivector e₁₂ (xy plane = rotation about z).
    pub b_xy: f32,
}

impl Rotor3 {
    /// Identity rotor — no rotation.
    pub const IDENTITY: Self = Self { s: 1.0, b_yz: 0.0, b_xz: 0.0, b_xy: 0.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub const fn new(s: f32, b_yz: f32, b_xz: f32, b_xy: f32) -> Self {
        Self { s, b_yz, b_xz, b_xy }
    }

    /// Build from a rotation axis + angle (radians).
    ///
    /// Axis does not need to be pre-normalised.
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (sin_half, cos_half) = (angle * 0.5).sin_cos();
        let n = axis.normalize();
        // Bivector components = sin(θ/2) * n_i  (note: axis = (byz, bxz, bxy) direction)
        Self {
            s:    cos_half,
            b_yz: sin_half * n.x,  // rotation in yz-plane ~ x-axis
            b_xz: sin_half * n.y,  // rotation in xz-plane ~ y-axis (sign handled in rotate)
            b_xy: sin_half * n.z,  // rotation in xy-plane ~ z-axis
        }
    }

    /// Build the shortest rotor that rotates unit vector `from` into unit vector `to`.
    ///
    /// Returns IDENTITY if vectors are already aligned.
    /// Handles anti-parallel vectors (returns 180° rotation about a perpendicular axis).
    pub fn from_vec_to_vec(from: Vec3, to: Vec3) -> Self {
        let dot = from.dot(to);
        if dot >= 1.0 - EPSILON { return Self::IDENTITY; }
        if dot <= -1.0 + EPSILON {
            // Anti-parallel: rotate 180° about any perpendicular axis
            let perp = if from.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0).cross(from).normalize()
            } else {
                Vec3::new(0.0, 1.0, 0.0).cross(from).normalize()
            };
            return Self { s: 0.0, b_yz: perp.x, b_xz: perp.y, b_xy: perp.z };
        }
        // R = (1 + t·f) / |1 + t·f| where t = to, f = from (no, wrong)
        // R = normalize(1 + t*f) in GA: geometric product of two unit vectors
        // t*f = t·f + t∧f = dot + (byz, bxz, bxy)
        // The wedge t∧f gives the bivector components
        let half_angle_s = (0.5 * (1.0 + dot)).sqrt();
        let half_sin_over_sin = if half_angle_s > EPSILON { 1.0 / (2.0 * half_angle_s) } else { 0.0 };
        let b = to.cross(from) * half_sin_over_sin; // bivector direction from cross product
        // Note: from∧to = from × to in 3D (dual relationship)
        Self { s: half_angle_s, b_yz: b.x, b_xz: b.y, b_xy: b.z }
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Rotate vector v: R v R̃  (where R̃ is the reverse of R).
    ///
    /// This is the sandwich product. Equivalent to Quat::rotate.
    pub fn rotate(self, v: Vec3) -> Vec3 {
        // R = s + b  where b is the bivector
        // R v R̃ can be expanded as 2 cross-products (same as Quat::rotate formula)
        // Using the Quat isomorphism for efficiency:
        let q = self.to_quat();
        q.rotate(v)
    }

    /// Geometric product (composition): `self * rhs` applies rhs first, then self.
    pub fn geometric_product(self, rhs: Self) -> Self {
        // Rotor product is isomorphic to quaternion product
        Self {
            s:    self.s*rhs.s    - self.b_yz*rhs.b_yz - self.b_xz*rhs.b_xz - self.b_xy*rhs.b_xy,
            b_yz: self.s*rhs.b_yz + self.b_yz*rhs.s    - self.b_xz*rhs.b_xy + self.b_xy*rhs.b_xz,
            b_xz: self.s*rhs.b_xz + self.b_xz*rhs.s    + self.b_yz*rhs.b_xy - self.b_xy*rhs.b_yz,
            b_xy: self.s*rhs.b_xy + self.b_xy*rhs.s    - self.b_yz*rhs.b_xz + self.b_xz*rhs.b_yz,
        }
    }

    /// Reverse (conjugate of bivector parts, scalar unchanged). R̃ = s - bivector.
    #[inline]
    pub fn reverse(self) -> Self {
        Self { s: self.s, b_yz: -self.b_yz, b_xz: -self.b_xz, b_xy: -self.b_xy }
    }

    #[inline] pub fn length_sq(self) -> f32 {
        self.s*self.s + self.b_yz*self.b_yz + self.b_xz*self.b_xz + self.b_xy*self.b_xy
    }
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }

    #[inline] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::IDENTITY }
        else { let inv = 1.0/l; Self { s:self.s*inv, b_yz:self.b_yz*inv, b_xz:self.b_xz*inv, b_xy:self.b_xy*inv } }
    }

    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }
    #[inline] pub fn is_finite(self) -> bool {
        self.s.is_finite() && self.b_yz.is_finite() && self.b_xz.is_finite() && self.b_xy.is_finite()
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Normalised linear interpolation (nlerp). Fast, nearly constant angular velocity.
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        let dot = self.s*rhs.s + self.b_yz*rhs.b_yz + self.b_xz*rhs.b_xz + self.b_xy*rhs.b_xy;
        let rhs = if dot < 0.0 { -rhs } else { rhs };
        Self {
            s:    self.s    + (rhs.s    - self.s)    * t,
            b_yz: self.b_yz + (rhs.b_yz - self.b_yz) * t,
            b_xz: self.b_xz + (rhs.b_xz - self.b_xz) * t,
            b_xy: self.b_xy + (rhs.b_xy - self.b_xy) * t,
        }.normalize()
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Convert to `Quat`. Zero-cost identity mapping.
    ///
    /// s↔w, b_yz↔x, -b_xz↔y, b_xy↔z.
    /// The sign flip on b_xz comes from the difference between
    /// bivector e₁₃ and the j quaternion component convention.
    #[inline]
    pub fn to_quat(self) -> Quat {
        Quat::new(self.b_yz, -self.b_xz, self.b_xy, self.s)
    }

    /// Build from `Quat`.
    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        Self { s: q.w, b_yz: q.x, b_xz: -q.y, b_xy: q.z }
    }

    /// Euler angles → Rotor3 (ZYX convention, same as Quat).
    #[inline]
    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        Self::from_quat(Quat::from_euler(roll, pitch, yaw))
    }
}

impl Mul for Rotor3 {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.geometric_product(rhs) }
}
impl MulAssign for Rotor3 { #[inline] fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; } }
impl Neg for Rotor3 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self { s: -self.s, b_yz: -self.b_yz, b_xz: -self.b_xz, b_xy: -self.b_xy } }
}
impl PartialEq for Rotor3 {
    fn eq(&self, r: &Self) -> bool {
        self.s == r.s && self.b_yz == r.b_yz && self.b_xz == r.b_xz && self.b_xy == r.b_xy
    }
}
impl Default for Rotor3 { fn default() -> Self { Self::IDENTITY } }
impl From<Quat>   for Rotor3 { #[inline] fn from(q: Quat)   -> Self { Self::from_quat(q) } }
impl From<Rotor3> for Quat   { #[inline] fn from(r: Rotor3) -> Self { r.to_quat() } }
impl fmt::Debug for Rotor3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rotor3(s={:.4}, b_yz={:.4}, b_xz={:.4}, b_xy={:.4})", self.s, self.b_yz, self.b_xz, self.b_xy)
    }
  }
