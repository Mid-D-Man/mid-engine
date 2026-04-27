// crates/mid-math/src/f32/scalar/quat.rs
//! Scalar Quat — fallback for non-SIMD targets and correctness reference.

use core::fmt;
use core::ops::{Mul, MulAssign, Neg, Add, Sub};
use crate::f32::scalar::vec3::Vec3;
use crate::f32::scalar::mat4::Mat4;
use crate::f32::math;
use crate::EPSILON;

/// Quaternion. 16 bytes, 16-byte aligned. Convention: (x, y, z, w).
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x:0.0, y:0.0, z:0.0, w:1.0 };
    const ZERO: Self         = Self { x:0.0, y:0.0, z:0.0, w:0.0 };

    #[inline(always)]
    pub fn new(x:f32, y:f32, z:f32, w:f32) -> Self { Self{x,y,z,w} }

    #[inline(always)]
    pub fn from_xyzw(x:f32, y:f32, z:f32, w:f32) -> Self { Self::new(x,y,z,w) }

    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let (s, c) = math::sin_cos(angle_rad * 0.5);
        let n = axis.normalize();
        Self::new(n.x*s, n.y*s, n.z*s, c)
    }

    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sx, cx) = math::sin_cos(roll  * 0.5);
        let (sy, cy) = math::sin_cos(pitch * 0.5);
        let (sz, cz) = math::sin_cos(yaw   * 0.5);
        Self::new(
            cz*cy*sx - sz*sy*cx,
            cz*sy*cx + sz*cy*sx,
            sz*cy*cx - cz*sy*sx,
            cz*cy*cx + sz*sy*sx,
        ).normalize()
    }

    pub fn to_euler(self) -> (f32, f32, f32) {
        let sinp  = 2.0 * (self.w*self.y - self.z*self.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * core::f32::consts::FRAC_PI_2
        } else {
            sinp.asin()
        };
        let roll = (2.0*(self.w*self.x + self.y*self.z))
            .atan2(1.0 - 2.0*(self.x*self.x + self.y*self.y));
        let yaw  = (2.0*(self.w*self.z + self.x*self.y))
            .atan2(1.0 - 2.0*(self.y*self.y + self.z*self.z));
        (roll, pitch, yaw)
    }

    #[inline]
    pub fn dot(self, r: Self) -> f32 {
        self.x*r.x + self.y*r.y + self.z*r.z + self.w*r.w
    }

    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self)    -> f32 { self.length_sq().sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::IDENTITY }
        else { Self::new(self.x/l, self.y/l, self.z/l, self.w/l) }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { return Self::IDENTITY; }
        let r = 1.0 / sq;
        Self::new(-self.x*r, -self.y*r, -self.z*r, self.w*r)
    }

    /// Rotate a Vec3 via sandwich product q v q*.
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    #[inline]
    pub fn mul_quat(self, r: Self) -> Self {
        Self::new(
            self.w*r.x + self.x*r.w + self.y*r.z - self.z*r.y,
            self.w*r.y - self.x*r.z + self.y*r.w + self.z*r.x,
            self.w*r.z + self.x*r.y - self.y*r.x + self.z*r.w,
            self.w*r.w - self.x*r.x - self.y*r.y - self.z*r.z,
        )
    }

    pub fn nlerp(self, mut rhs: Self, t: f32) -> Self {
        if self.dot(rhs) < 0.0 { rhs = -rhs; }
        Self::new(
            self.x + (rhs.x - self.x)*t,
            self.y + (rhs.y - self.y)*t,
            self.z + (rhs.z - self.z)*t,
            self.w + (rhs.w - self.w)*t,
        ).normalize()
    }

    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 { rhs = -rhs; cos_theta = -cos_theta; }
        if cos_theta > 1.0 - EPSILON {
            return self.nlerp(rhs, t);
        }
        let angle = math::acos_approx(cos_theta);
        let sin_a = (1.0 - cos_theta*cos_theta).sqrt();
        let s0    = ((1.0-t)*angle).sin() / sin_a;
        let s1    = (t*angle).sin()       / sin_a;
        Self::new(
            self.x*s0 + rhs.x*s1,
            self.y*s0 + rhs.y*s1,
            self.z*s0 + rhs.z*s1,
            self.w*s0 + rhs.w*s1,
        )
    }

    pub fn to_mat4(self) -> Mat4 {
        let q = self.normalize();
        let (x,y,z,w) = (q.x,q.y,q.z,q.w);
        let (x2,y2,z2) = (x+x,y+y,z+z);
        let (xx,yy,zz) = (x*x2,y*y2,z*z2);
        let (xy,xz,yz) = (x*y2,x*z2,y*z2);
        let (wx,wy,wz) = (w*x2,w*y2,w*z2);
        Mat4::from_cols(
            [1.0-yy-zz, xy+wz,     xz-wy,     0.0],
            [xy-wz,     1.0-xx-zz, yz+wx,     0.0],
            [xz+wy,     yz-wx,     1.0-xx-yy, 0.0],
            [0.0,       0.0,       0.0,       1.0],
        )
    }

    #[inline]
    pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() &&
        self.z.is_finite() && self.w.is_finite()
    }
}

impl Mul for Quat {
    type Output = Self;
    #[inline] fn mul(self, r: Self) -> Self { self.mul_quat(r) }
}
impl MulAssign for Quat {
    #[inline] fn mul_assign(&mut self, r: Self) { *self = self.mul_quat(r); }
}
impl Neg for Quat {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x,-self.y,-self.z,-self.w) }
}
impl Add for Quat {
    type Output = Self;
    #[inline] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z,self.w+r.w)}
}
impl Sub for Quat {
    type Output = Self;
    #[inline] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z,self.w-r.w)}
}
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s,self.w*s)}
}

impl PartialEq for Quat {
    fn eq(&self, r:&Self) -> bool {
        self.x==r.x && self.y==r.y && self.z==r.z && self.w==r.w
    }
}
impl Default for Quat { fn default() -> Self { Self::IDENTITY } }

impl fmt::Display for Quat {
    fn fmt(&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quat({:.4}, {:.4}, {:.4}, {:.4})", self.x, self.y, self.z, self.w)
    }
  }
