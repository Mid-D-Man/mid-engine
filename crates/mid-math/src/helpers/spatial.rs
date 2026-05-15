// crates/mid-math/src/spatial.rs
//! Spatial vectors (V6 / screws) for rigid-body physics.
//!
//! A spatial vector combines 3D angular and 3D linear components into one
//! 6-element entity. This is Roy Featherstone's spatial algebra notation,
//! used in articulated-body dynamics (ABA), constraint solvers, and impulse propagation.
//!
//! Layout (Featherstone convention): [angular; linear] — angular comes first.
//!
//! Key types:
//!   SpatialVelocity — rigid-body velocity: [ω; v] (angular velocity, linear velocity)
//!   SpatialForce    — generalised force: [τ; f]  (torque, force)
//!
//! Spatial cross product:
//!   v × f = [ω × τ + v × f_lin; ω × f_lin]  (velocity cross force → force)
//!   v × v2 = [ω × ω2; ω × v2_lin + v_lin × ω2]  (velocity cross velocity)
//!
//! Reference: Featherstone "Rigid Body Dynamics Algorithms" (2008).

use core::fmt;
use core::ops::{Add, AddAssign, Mul, Neg, Sub};
use crate::Vec3;

// ── SpatialVelocity ───────────────────────────────────────────────────────────

/// Spatial velocity: [angular ω, linear v]. 24 bytes.
///
/// Represents the instantaneous velocity of a rigid body in world space.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct SpatialVelocity {
    /// Angular velocity ω (rad/s about each axis).
    pub angular: Vec3,
    /// Linear velocity v (m/s of the body's reference point).
    pub linear:  Vec3,
}

impl SpatialVelocity {
    pub const ZERO: Self = Self { angular: Vec3::ZERO, linear: Vec3::ZERO };

    #[inline(always)]
    pub fn new(angular: Vec3, linear: Vec3) -> Self { Self { angular, linear } }

    /// Spatial cross product with a spatial velocity: v × w.
    ///
    /// Result = [ω_a × ω_b; ω_a × v_b + v_a × ω_b]
    #[inline]
    pub fn cross_vel(self, rhs: Self) -> Self {
        Self {
            angular: self.angular.cross(rhs.angular),
            linear:  self.angular.cross(rhs.linear) + self.linear.cross(rhs.angular),
        }
    }

    /// Spatial cross product with a spatial force: v × f.
    ///
    /// Result = [ω × τ + v × f; ω × f]
    #[inline]
    pub fn cross_force(self, rhs: SpatialForce) -> SpatialForce {
        SpatialForce {
            torque: self.angular.cross(rhs.torque) + self.linear.cross(rhs.force),
            force:  self.angular.cross(rhs.force),
        }
    }

    /// Spatial dot product (motion · force): scalar power.
    ///
    /// = ω·τ + v·f  (Watts for unit-consistent inputs)
    #[inline]
    pub fn dot_force(self, f: SpatialForce) -> f32 {
        self.angular.dot(f.torque) + self.linear.dot(f.force)
    }

    #[inline] pub fn scale(self, s: f32) -> Self {
        Self { angular: self.angular * s, linear: self.linear * s }
    }
    #[inline] pub fn is_finite(self) -> bool { self.angular.is_finite() && self.linear.is_finite() }
}

// ── SpatialForce ──────────────────────────────────────────────────────────────

/// Spatial force: [torque τ, force f]. 24 bytes.
///
/// Represents a generalised force applied to a rigid body:
/// force component `f` at the body's reference point, torque component `τ`.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct SpatialForce {
    /// Torque τ (N·m).
    pub torque: Vec3,
    /// Force  f (N).
    pub force:  Vec3,
}

impl SpatialForce {
    pub const ZERO: Self = Self { torque: Vec3::ZERO, force: Vec3::ZERO };

    #[inline(always)]
    pub fn new(torque: Vec3, force: Vec3) -> Self { Self { torque, force } }

    /// Dot product with a spatial velocity. Same as `vel.dot_force(self)`.
    #[inline]
    pub fn dot_vel(self, v: SpatialVelocity) -> f32 { v.dot_force(self) }

    #[inline] pub fn scale(self, s: f32) -> Self {
        Self { torque: self.torque * s, force: self.force * s }
    }
    #[inline] pub fn is_finite(self) -> bool { self.torque.is_finite() && self.force.is_finite() }
}

// ── SpatialInertia ────────────────────────────────────────────────────────────

/// Spatial inertia: 6×6 symmetric matrix represented in compact form.
///
/// Compact form: mass `m`, center of mass offset `c` from reference point,
/// and 3×3 rotational inertia tensor `I` about the reference point.
///
/// The full 6×6 is: [[I + m*c×c×ᵀ, m*c×]; [-m*c×ᵀ, m·I₃]]
/// where c× is the skew-symmetric matrix of c.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct SpatialInertia {
    /// Total mass (kg).
    pub mass: f32,
    /// Center of mass position relative to body reference point (m).
    pub com:  Vec3,
    /// Inertia tensor about the reference point (not COM) in kg·m².
    /// Stored as diagonal + off-diagonal: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].
    pub inertia: [f32; 6],
}

impl SpatialInertia {
    /// Apply spatial inertia to a velocity: I * v → SpatialForce (momentum).
    pub fn mul_vel(self, v: SpatialVelocity) -> SpatialForce {
        // Unpack symmetric inertia tensor
        let (ixx, iyy, izz, ixy, ixz, iyz) = (
            self.inertia[0], self.inertia[1], self.inertia[2],
            self.inertia[3], self.inertia[4], self.inertia[5],
        );
        let w = v.angular;
        let vel = v.linear;
        let m = self.mass;
        let c = self.com;

        // Angular momentum h = I*ω + m*(c × v_lin)
        let cv = c.cross(vel);
        let h = Vec3::new(
            ixx * w.x + ixy * w.y + ixz * w.z + m * cv.x,
            ixy * w.x + iyy * w.y + iyz * w.z + m * cv.y,
            ixz * w.x + iyz * w.y + izz * w.z + m * cv.z,
        );
        // Linear momentum p = m*(v_lin - c × ω)
        let cw = c.cross(w);
        let p = (vel - cw) * m;

        SpatialForce { torque: h, force: p }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for SpatialVelocity {
    type Output=Self;
    #[inline] fn add(self,r:Self)->Self{Self{angular:self.angular+r.angular,linear:self.linear+r.linear}}
}
impl AddAssign for SpatialVelocity { #[inline] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for SpatialVelocity {
    type Output=Self;
    #[inline] fn sub(self,r:Self)->Self{Self{angular:self.angular-r.angular,linear:self.linear-r.linear}}
}
impl Neg for SpatialVelocity {
    type Output=Self;
    #[inline] fn neg(self)->Self{Self{angular:-self.angular,linear:-self.linear}}
}
impl Mul<f32> for SpatialVelocity { type Output=Self; #[inline] fn mul(self,s:f32)->Self{self.scale(s)} }

impl Add for SpatialForce {
    type Output=Self;
    #[inline] fn add(self,r:Self)->Self{Self{torque:self.torque+r.torque,force:self.force+r.force}}
}
impl AddAssign for SpatialForce { #[inline] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for SpatialForce {
    type Output=Self;
    #[inline] fn sub(self,r:Self)->Self{Self{torque:self.torque-r.torque,force:self.force-r.force}}
}
impl Neg for SpatialForce {
    type Output=Self;
    #[inline] fn neg(self)->Self{Self{torque:-self.torque,force:-self.force}}
}
impl Mul<f32> for SpatialForce { type Output=Self; #[inline] fn mul(self,s:f32)->Self{self.scale(s)} }

impl fmt::Debug for SpatialVelocity {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        write!(f,"SpatialVelocity(ω={:?}, v={:?})",self.angular,self.linear)
    }
}
impl fmt::Debug for SpatialForce {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        write!(f,"SpatialForce(τ={:?}, f={:?})",self.torque,self.force)
    }
      }
