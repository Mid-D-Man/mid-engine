// crates/mid-math/src/f64/dvec3.rs
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::dvec2::DEPSILON;

/// 3D double-precision vector. 24 bytes, align(8). Always scalar.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct DVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DVec3 {
    pub const ZERO:  Self = Self { x:  0.0, y:  0.0, z:  0.0 };
    pub const ONE:   Self = Self { x:  1.0, y:  1.0, z:  1.0 };
    pub const X:     Self = Self { x:  1.0, y:  0.0, z:  0.0 };
    pub const Y:     Self = Self { x:  0.0, y:  1.0, z:  0.0 };
    pub const Z:     Self = Self { x:  0.0, y:  0.0, z:  1.0 };
    pub const NEG_X: Self = Self { x: -1.0, y:  0.0, z:  0.0 };
    pub const NEG_Y: Self = Self { x:  0.0, y: -1.0, z:  0.0 };
    pub const NEG_Z: Self = Self { x:  0.0, y:  0.0, z: -1.0 };

    #[inline(always)] pub const fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    #[inline(always)] pub fn splat(v: f64) -> Self { Self::new(v, v, v) }
    #[inline(always)] pub fn from_array(a: [f64; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 3] { [self.x, self.y, self.z] }

    #[inline(always)]
    pub fn extend(self, w: f64) -> crate::DVec4 {
        crate::DVec4::new(self.x, self.y, self.z, w)
    }

    #[inline(always)]
    pub fn truncate(self) -> crate::DVec2 {
        crate::DVec2::new(self.x, self.y)
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }
    #[inline(always)] pub fn length(self) -> f64 { self.length_sq().sqrt() }

    #[inline(always)]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < DEPSILON { Self::ZERO } else { self * (1.0 / l) }
    }

    #[inline(always)]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline(always)] pub fn normalize_or_zero(self) -> Self { self.try_normalize().unwrap_or(Self::ZERO) }
    #[inline(always)] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-10 }

    #[inline(always)] pub fn lerp(self, rhs: Self, t: f64) -> Self { self + (rhs - self) * t }
    #[inline(always)] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline(always)] pub fn distance(self, rhs: Self) -> f64 { (self - rhs).length() }
    #[inline(always)] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    #[inline(always)]
    pub fn angle_between(self, rhs: Self) -> f64 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < DEPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    // ── p2: project / reject ──────────────────────────────────────────────────

    #[inline(always)]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < DEPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }

    #[inline(always)]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }

    // ── p7: movement / clamping helpers ──────────────────────────────────────

    #[inline(always)]
    pub fn move_towards(self, target: Self, max_dist: f64) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < DEPSILON { target } else { self + d / len * max_dist }
    }

    #[inline(always)]
    pub fn clamp_length(self, min: f64, max: f64) -> Self {
        let len = self.length();
        if len < DEPSILON { return Self::ZERO; }
        let clamped = len.clamp(min, max);
        if (clamped - len).abs() < DEPSILON { self } else { self * (clamped / len) }
    }

    #[inline(always)]
    pub fn clamp_length_max(self, max: f64) -> Self {
        let len = self.length();
        if len > max && len > DEPSILON { self * (max / len) } else { self }
    }

    #[inline(always)]
    pub fn clamp_length_min(self, min: f64) -> Self {
        let len = self.length();
        if len < min && len > DEPSILON { self * (min / len) } else { self }
    }

    #[inline(always)] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    #[inline(always)]
    pub fn is_parallel(self, rhs: Self) -> bool {
        self.cross(rhs).length_sq() < DEPSILON * DEPSILON
    }

    #[inline(always)]
    pub fn is_perpendicular(self, rhs: Self) -> bool { self.dot(rhs).abs() < DEPSILON }

    // ── p6: spherical coordinates ─────────────────────────────────────────────

    /// Convert to `(radius, theta, phi)`. `theta` ∈ `[0, π]`, `phi` ∈ `[-π, π]`.
    #[inline]
    pub fn to_spherical(self) -> (f64, f64, f64) {
        let r = self.length();
        if r < DEPSILON { return (0.0, 0.0, 0.0); }
        let theta = (self.z / r).clamp(-1.0, 1.0).acos();
        let phi   = self.y.atan2(self.x);
        (r, theta, phi)
    }

    /// Build from spherical `(r, theta, phi)`.
    #[inline]
    pub fn from_spherical(r: f64, theta: f64, phi: f64) -> Self {
        let sin_theta = theta.sin();
        Self::new(r * sin_theta * phi.cos(), r * sin_theta * phi.sin(), r * theta.cos())
    }

    #[inline(always)]
    pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs()) }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor(), self.z.floor()) }
    #[inline(always)] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil(),  self.z.ceil()) }
    #[inline(always)] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round(), self.z.round()) }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline(always)]
    pub fn is_nan(self) -> bool { self.x.is_nan() || self.y.is_nan() || self.z.is_nan() }

    #[inline(always)]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < DEPSILON
            && (self.y - rhs.y).abs() < DEPSILON
            && (self.z - rhs.z).abs() < DEPSILON
    }

    #[inline(always)]
    pub fn as_vec3(self) -> crate::Vec3 {
        crate::Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    #[inline(always)]
    pub fn as_vec3a(self) -> crate::Vec3 { self.as_vec3() }

    /// Large World Coordinates: shifts `self` so `origin` becomes the
    /// new coordinate zero, in f64 (the subtraction that matters for
    /// precision), then truncates to f32. Unlike a plain [`Self::as_vec3`]
    /// cast, this is safe to call regardless of how far `self` is from
    /// the *world* origin — jitter comes from the magnitude of the
    /// value being cast, not from anything about `self` on its own, so
    /// shifting first (typically `origin` = the camera's own current
    /// `DVec3` position) is what actually prevents it. Precision is
    /// highest exactly where `origin` is, which is the point of calling
    /// this with the camera's position specifically before sending
    /// per-vertex or per-instance data to the GPU.
    #[inline(always)]
    pub fn to_view_relative(self, origin: Self) -> crate::Vec3 {
        (self - origin).as_vec3()
    }
}

impl Add  for DVec3 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)} }
impl Sub  for DVec3 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)} }
impl Neg  for DVec3 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)} }
impl Mul<f64> for DVec3 { type Output=Self; #[inline(always)] fn mul(self,s:f64)->Self{Self::new(self.x*s,self.y*s,self.z*s)} }
impl Mul<DVec3> for f64  { type Output=DVec3; #[inline(always)] fn mul(self,v:DVec3)->DVec3{DVec3::new(self*v.x,self*v.y,self*v.z)} }
impl Mul  for DVec3 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y,self.z*r.z)} }
impl Div<f64> for DVec3 { type Output=Self; #[inline(always)] fn div(self,s:f64)->Self{Self::new(self.x/s,self.y/s,self.z/s)} }
impl Div  for DVec3 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z)} }

impl AddAssign for DVec3 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;} }
impl SubAssign for DVec3 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;} }
impl MulAssign<f64> for DVec3 { #[inline(always)] fn mul_assign(&mut self,s:f64){self.x*=s;self.y*=s;self.z*=s;} }
impl DivAssign<f64> for DVec3 { #[inline(always)] fn div_assign(&mut self,s:f64){self.x/=s;self.y/=s;self.z/=s;} }

impl PartialEq for DVec3 {
    fn eq(&self, rhs: &Self) -> bool { self.x==rhs.x && self.y==rhs.y && self.z==rhs.z }
}
impl Default for DVec3 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for DVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec3").field(&self.x).field(&self.y).field(&self.z).finish()
    }
}
impl fmt::Display for DVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl From<[f64; 3]> for DVec3 { fn from(a:[f64;3])->Self{Self::new(a[0],a[1],a[2])} }
impl From<DVec3> for [f64; 3] { fn from(v:DVec3)->[f64;3]{[v.x,v.y,v.z]} }
impl From<(f64,f64,f64)> for DVec3 { fn from(t:(f64,f64,f64))->Self{Self::new(t.0,t.1,t.2)} }
impl From<DVec3> for (f64,f64,f64) { fn from(v:DVec3)->(f64,f64,f64){(v.x,v.y,v.z)} }
