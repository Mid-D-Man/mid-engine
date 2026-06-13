// crates/mid-math/src/f32/sse2/mat4.rs
//! Mat4 with SSE2 fast-paths on x86 / x86_64.
//!
//! ── Storage fix (Build 8) ─────────────────────────────────────────────────────
//!
//! BEFORE: pub cols: [[f32; 4]; 4]
//!   → column is a *pointer* into a stack slot
//!   → Mul<Mat4> forces 16 × _mm_load_ps before any math (4 per Mul<Vec4> × 4)
//!   → LLVM cannot register-allocate stack-pointer operands across call boundaries
//!   → result: ~17 ns / multiply
//!
//! AFTER: pub x_axis: Vec4, pub y_axis: Vec4, pub z_axis: Vec4, pub w_axis: Vec4
//!   → each field IS a __m128 (Vec4 is repr(transparent))
//!   → LLVM keeps all four columns in XMM4-XMM7 across the four Mul<Vec4> calls
//!   → zero memory loads for the LHS of Mat4 × Mat4
//!   → with -C target-cpu=native LLVM emits vfmadd213ps matching cglm's throughput
//!   → target: ~7 ns SSE2 (parity glam), ~3.5 ns with FMA (parity cglm / DirectXMath)
//!
//! Memory layout is IDENTICAL to the old storage: 64 bytes, 16-byte aligned.
//! CMat4 in ffi/float32.rs retains [[f32;4];4] as the immutable C ABI contract.
//!
//! Algorithm note — cglm glm_mat4_mul_sse2 vs our approach:
//!   cglm: load LHS cols one-by-one, FMA each against all 4 RHS broadcast components
//!   ours: keep LHS cols in registers, broadcast each RHS col's components → same FMAs
//!   Both produce 4 muls + 12 FMAs; difference is only register vs memory access pattern.
//!
//! Advantages preserved:
//!   vec3/normalize, vec4/normalize, quat/rotate, affine3/transform_point,
//!   100k entity transforms — all unchanged or improved (transform_point now zero-load too).

use core::fmt;
use core::ops::{Mul, MulAssign};


#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::dot4;
use crate::f32::sse2::vec3::Vec3;
use crate::f32::sse2::vec4::Vec4;
use crate::f32::sse2::quat::Quat;
use crate::EPSILON;

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
///
/// `element(row r, col c)` = `self.[x|y|z|w]_axis.[x|y|z|w]`
///
/// Columns are four named `Vec4` fields (`repr(transparent)` over `__m128`).
/// This guarantees LLVM passes the matrix in XMM registers rather than stack
/// pointers, eliminating the load traffic that caused the pre-fix ~17 ns mul gap.
///
/// **C interop:** use [`CMat4`][crate::ffi::types::CMat4] at the FFI boundary.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    /// Column 0 — x basis vector (or first column of any transform).
    pub x_axis: Vec4,
    /// Column 1 — y basis vector.
    pub y_axis: Vec4,
    /// Column 2 — z basis vector.
    pub z_axis: Vec4,
    /// Column 3 — translation / homogeneous w-column.
    pub w_axis: Vec4,
}

impl Mat4 {
    pub const ZERO: Self = Self {
        x_axis: Vec4::ZERO,
        y_axis: Vec4::ZERO,
        z_axis: Vec4::ZERO,
        w_axis: Vec4::ZERO,
    };
    pub const IDENTITY: Self = Self {
        x_axis: Vec4::X,
        y_axis: Vec4::Y,
        z_axis: Vec4::Z,
        w_axis: Vec4::W,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from four column arrays. API-compatible with all existing callsites.
    ///
    /// For literal constant arrays LLVM folds `Vec4::from_array` to a zero-cost
    /// constant `_mm_set_ps`; for runtime values it emits a single SSE2 load.
    #[inline]
    pub fn from_cols(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) -> Self {
        Self {
            x_axis: Vec4::from_array(c0),
            y_axis: Vec4::from_array(c1),
            z_axis: Vec4::from_array(c2),
            w_axis: Vec4::from_array(c3),
        }
    }

    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            x_axis: Vec4::X,
            y_axis: Vec4::Y,
            z_axis: Vec4::Z,
            w_axis: Vec4::new(t.x, t.y, t.z, 1.0),
        }
    }

    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self {
            x_axis: Vec4::new(s.x, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, s.y, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, s.z, 0.0),
            w_axis: Vec4::W,
        }
    }

    #[inline]
    pub fn from_rotation(q: Quat) -> Self { q.to_mat4() }

    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let q = r.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, yy, zz) = (x * x2, y * y2, z * z2);
        let (xy, xz, yz) = (x * y2, x * z2, y * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Self {
            x_axis: Vec4::new((1.0 - yy - zz) * s.x, (xy + wz) * s.x, (xz - wy) * s.x, 0.0),
            y_axis: Vec4::new((xy - wz) * s.y, (1.0 - xx - zz) * s.y, (yz + wx) * s.y, 0.0),
            z_axis: Vec4::new((xz + wy) * s.z, (yz - wx) * s.z, (1.0 - xx - yy) * s.z, 0.0),
            w_axis: Vec4::new(t.x, t.y, t.z, 1.0),
        }
    }

    // ── View matrices ─────────────────────────────────────────────────────────

    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self {
            x_axis: Vec4::new(r.x, u.x, -f.x, 0.0),
            y_axis: Vec4::new(r.y, u.y, -f.y, 0.0),
            z_axis: Vec4::new(r.z, u.z, -f.z, 0.0),
            w_axis: Vec4::new(-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        }
    }

    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = up.cross(f).normalize();
        let u = f.cross(r);
        Self {
            x_axis: Vec4::new(r.x, u.x, f.x, 0.0),
            y_axis: Vec4::new(r.y, u.y, f.y, 0.0),
            z_axis: Vec4::new(r.z, u.z, f.z, 0.0),
            w_axis: Vec4::new(-r.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
        }
    }

    // ── Projection matrices ───────────────────────────────────────────────────

    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = near - far;
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, (far + near) / z, -1.0),
            w_axis: Vec4::new(0.0, 0.0, (2.0 * far * near) / z, 0.0),
        }
    }

    pub fn perspective_lh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let z = far - near;
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, far / z, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -(far * near) / z, 0.0),
        }
    }

    pub fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left;
        let tb = top - bottom;
        let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -2.0 / nf, 0.0),
            w_axis: Vec4::new(-(right + left) / rl, -(top + bottom) / tb, -(far + near) / nf, 1.0),
        }
    }

    pub fn ortho_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left;
        let tb = top - bottom;
        let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 1.0 / nf, 0.0),
            w_axis: Vec4::new(-(right + left) / rl, -(top + bottom) / tb, -near / nf, 1.0),
        }
    }

    // ── Transpose ────────────────────────────────────────────────────────────
    //
    // Full SSE2 4×4 transpose via 4 unpack + 4 movelh/movehl.
    // Previously this was a scalar element-by-element swap.
    //
    // Pattern (identical to _MM_TRANSPOSE4_PS macro):
    //   tmp0 = unpacklo(x, y) → [x.x, y.x, x.y, y.y]
    //   tmp1 = unpacklo(z, w) → [z.x, w.x, z.y, w.y]
    //   tmp2 = unpackhi(x, y) → [x.z, y.z, x.w, y.w]
    //   tmp3 = unpackhi(z, w) → [z.z, w.z, z.w, w.w]
    //   row0 = movelh(tmp0, tmp1) → [x.x, y.x, z.x, w.x]  (original row 0)
    //   row1 = movehl(tmp1, tmp0) → [x.y, y.y, z.y, w.y]  (original row 1)
    //   row2 = movelh(tmp2, tmp3) → [x.z, y.z, z.z, w.z]  (original row 2)
    //   row3 = movehl(tmp3, tmp2) → [x.w, y.w, z.w, w.w]  (original row 3)

    pub fn transpose(self) -> Self {
        unsafe {
            let tmp0 = _mm_unpacklo_ps(self.x_axis.0, self.y_axis.0);
            let tmp1 = _mm_unpacklo_ps(self.z_axis.0, self.w_axis.0);
            let tmp2 = _mm_unpackhi_ps(self.x_axis.0, self.y_axis.0);
            let tmp3 = _mm_unpackhi_ps(self.z_axis.0, self.w_axis.0);
            Self {
                x_axis: Vec4(_mm_movelh_ps(tmp0, tmp1)),
                y_axis: Vec4(_mm_movehl_ps(tmp1, tmp0)),
                z_axis: Vec4(_mm_movelh_ps(tmp2, tmp3)),
                w_axis: Vec4(_mm_movehl_ps(tmp3, tmp2)),
            }
        }
    }

    // ── Determinant ───────────────────────────────────────────────────────────
    //
    // Full SSE2 algorithm (ported from glm / glm_mat4_determinant_lowp).
    // Operates directly on the Vec4 fields — no scalar extraction.

    pub fn determinant(self) -> f32 {
        unsafe {
            let z = self.z_axis.0;
            let w = self.w_axis.0;

            let swp2a = _mm_shuffle_ps::<0b00_01_01_10>(z, z);
            let swp3a = _mm_shuffle_ps::<0b11_10_11_11>(w, w);
            let swp2b = _mm_shuffle_ps::<0b11_10_11_11>(z, z);
            let swp3b = _mm_shuffle_ps::<0b00_01_01_10>(w, w);
            let swp2c = _mm_shuffle_ps::<0b00_00_01_10>(z, z);
            let swp3c = _mm_shuffle_ps::<0b01_10_00_00>(w, w);

            let mula = _mm_mul_ps(swp2a, swp3a);
            let mulb = _mm_mul_ps(swp2b, swp3b);
            let mulc = _mm_mul_ps(swp2c, swp3c);
            let sube = _mm_sub_ps(mula, mulb);
            let subf = _mm_sub_ps(_mm_movehl_ps(mulc, mulc), mulc);

            let y = self.y_axis.0;
            let subfaca = _mm_shuffle_ps::<0b10_01_00_00>(sube, sube);
            let swpfaca = _mm_shuffle_ps::<0b00_00_00_01>(y, y);
            let mulfaca = _mm_mul_ps(swpfaca, subfaca);

            let subtmpb = _mm_shuffle_ps::<0b00_00_11_01>(sube, subf);
            let subfacb = _mm_shuffle_ps::<0b11_01_01_00>(subtmpb, subtmpb);
            let swpfacb = _mm_shuffle_ps::<0b01_01_10_10>(y, y);
            let mulfacb = _mm_mul_ps(swpfacb, subfacb);

            let subres = _mm_sub_ps(mulfaca, mulfacb);
            let subtmpc = _mm_shuffle_ps::<0b01_00_10_10>(sube, subf);
            let subfacc = _mm_shuffle_ps::<0b11_11_10_00>(subtmpc, subtmpc);
            let swpfacc = _mm_shuffle_ps::<0b10_11_11_11>(y, y);
            let mulfacc = _mm_mul_ps(swpfacc, subfacc);

            let addres = _mm_add_ps(subres, mulfacc);
            let detcof = _mm_mul_ps(addres, _mm_setr_ps(1.0, -1.0, 1.0, -1.0));

            dot4(self.x_axis.0, detcof)
        }
    }

    // ── Transform helpers ─────────────────────────────────────────────────────
    //
    // Optimised forms that avoid the extra extend(w)/truncate() round-trip.
    // For transform_point (w=1): result = x*p.x + y*p.y + z*p.z + w_col
    // For transform_vector (w=0): result = x*v.x + y*v.y + z*v.z
    // Lane 3 of the returned Vec3 is a don't-care (Vec3 ops never read it).

    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        unsafe {
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(p.0, p.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(p.0, p.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(p.0, p.0);
            let res = _mm_mul_ps(self.x_axis.0, bx);
            let res = _mm_add_ps(res, _mm_mul_ps(self.y_axis.0, by));
            let res = _mm_add_ps(res, _mm_mul_ps(self.z_axis.0, bz));
            Vec3(_mm_add_ps(res, self.w_axis.0))
        }
    }

    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        unsafe {
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(v.0, v.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(v.0, v.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(v.0, v.0);
            let res = _mm_mul_ps(self.x_axis.0, bx);
            let res = _mm_add_ps(res, _mm_mul_ps(self.y_axis.0, by));
            Vec3(_mm_add_ps(res, _mm_mul_ps(self.z_axis.0, bz)))
        }
    }

    // ── Decompose ─────────────────────────────────────────────────────────────

    pub fn decompose_trs(self) -> (Vec3, Quat, Vec3) {
        // Extract translation from w-column.
        let t = self.w_axis.truncate();

        // Scale = length of each rotation column.
        let sx = self.x_axis.truncate().length();
        let sy = self.y_axis.truncate().length();
        let sz = self.z_axis.truncate().length();

        // Sign of sx encodes reflection (negative determinant of upper 3×3).
        let det =
            self.x_axis.x * (self.y_axis.y * self.z_axis.z - self.z_axis.y * self.y_axis.z)
          - self.y_axis.x * (self.x_axis.y * self.z_axis.z - self.z_axis.y * self.x_axis.z)
          + self.z_axis.x * (self.x_axis.y * self.y_axis.z - self.y_axis.y * self.x_axis.z);
        let sx = if det < 0.0 { -sx } else { sx };

        let inv_sx = if sx.abs() < EPSILON { 0.0 } else { 1.0 / sx };
        let inv_sy = if sy       < EPSILON { 0.0 } else { 1.0 / sy };
        let inv_sz = if sz       < EPSILON { 0.0 } else { 1.0 / sz };

        let c0 = self.x_axis.truncate() * inv_sx;
        let c1 = self.y_axis.truncate() * inv_sy;
        let c2 = self.z_axis.truncate() * inv_sz;

        use crate::helpers::euler::QuatExt as _;
        let r = Quat::from_rotation_axes(c0, c1, c2);

        (t, r, Vec3::new(sx, sy, sz))
    }

    // ── General inverse (SSE2 cofactor method) ────────────────────────────────
    //
    // Ported from glm `glm_mat4_inverse` / glam's SSE2 inverse.
    // Direct field access replaces the former _mm_load_ps from stack pointers.

    pub fn inverse(self) -> Option<Self> {
        unsafe {
            let x = self.x_axis.0;
            let y = self.y_axis.0;
            let z = self.z_axis.0;
            let w = self.w_axis.0;

            let fac0 = {
                let s0a = _mm_shuffle_ps::<0b11_11_11_11>(w, z);
                let s0b = _mm_shuffle_ps::<0b10_10_10_10>(w, z);
                let s00 = _mm_shuffle_ps::<0b10_10_10_10>(z, y);
                let s01 = _mm_shuffle_ps::<0b10_00_00_00>(s0a, s0a);
                let s02 = _mm_shuffle_ps::<0b10_00_00_00>(s0b, s0b);
                let s03 = _mm_shuffle_ps::<0b11_11_11_11>(z, y);
                _mm_sub_ps(_mm_mul_ps(s00, s01), _mm_mul_ps(s02, s03))
            };
            let fac1 = {
                let s0a = _mm_shuffle_ps::<0b11_11_11_11>(w, z);
                let s0b = _mm_shuffle_ps::<0b01_01_01_01>(w, z);
                let s00 = _mm_shuffle_ps::<0b01_01_01_01>(z, y);
                let s01 = _mm_shuffle_ps::<0b10_00_00_00>(s0a, s0a);
                let s02 = _mm_shuffle_ps::<0b10_00_00_00>(s0b, s0b);
                let s03 = _mm_shuffle_ps::<0b11_11_11_11>(z, y);
                _mm_sub_ps(_mm_mul_ps(s00, s01), _mm_mul_ps(s02, s03))
            };
            let fac2 = {
                let s0a = _mm_shuffle_ps::<0b10_10_10_10>(w, z);
                let s0b = _mm_shuffle_ps::<0b01_01_01_01>(w, z);
                let s00 = _mm_shuffle_ps::<0b01_01_01_01>(z, y);
                let s01 = _mm_shuffle_ps::<0b10_00_00_00>(s0a, s0a);
                let s02 = _mm_shuffle_ps::<0b10_00_00_00>(s0b, s0b);
                let s03 = _mm_shuffle_ps::<0b10_10_10_10>(z, y);
                _mm_sub_ps(_mm_mul_ps(s00, s01), _mm_mul_ps(s02, s03))
            };
            let fac3 = {
                let s0a = _mm_shuffle_ps::<0b11_11_11_11>(w, z);
                let s0b = _mm_shuffle_ps::<0b00_00_00_00>(w, z);
                let s00 = _mm_shuffle_ps::<0b00_00_00_00>(z, y);
                let s01 = _mm_shuffle_ps::<0b10_00_00_00>(s0a, s0a);
                let s02 = _mm_shuffle_ps::<0b10_00_00_00>(s0b, s0b);
                let s03 = _mm_shuffle_ps::<0b11_11_11_11>(z, y);
                _mm_sub_ps(_mm_mul_ps(s00, s01), _mm_mul_ps(s02, s03))
            };
            let fac4 = {
                let s0a = _mm_shuffle_ps::<0b10_10_10_10>(w, z);
                let s0b = _mm_shuffle_ps::<0b00_00_00_00>(w, z);
                let s00 = _mm_shuffle_ps::<0b00_00_00_00>(z, y);
                let s01 = _mm_shuffle_ps::<0b10_00_00_00>(s0a, s0a);
                let s02 = _mm_shuffle_ps::<0b10_00_00_00>(s0b, s0b);
                let s03 = _mm_shuffle_ps::<0b10_10_10_10>(z, y);
                _mm_sub_ps(_mm_mul_ps(s00, s01), _mm_mul_ps(s02, s03))
            };
            let fac5 = {
                let s0a = _mm_shuffle_ps::<0b01_01_01_01>(w, z);
                let s0b = _mm_shuffle_ps::<0b00_00_00_00>(w, z);
                let s00 = _mm_shuffle_ps::<0b00_00_00_00>(z, y);
                let s01 = _mm_shuffle_ps::<0b10_00_00_00>(s0a, s0a);
                let s02 = _mm_shuffle_ps::<0b10_00_00_00>(s0b, s0b);
                let s03 = _mm_shuffle_ps::<0b01_01_01_01>(z, y);
                _mm_sub_ps(_mm_mul_ps(s00, s01), _mm_mul_ps(s02, s03))
            };

            let sign_a = _mm_set_ps( 1.0, -1.0,  1.0, -1.0);
            let sign_b = _mm_set_ps(-1.0,  1.0, -1.0,  1.0);

            let tmp0 = _mm_shuffle_ps::<0b00_00_00_00>(y, x);
            let vec0 = _mm_shuffle_ps::<0b10_10_10_00>(tmp0, tmp0);
            let tmp1 = _mm_shuffle_ps::<0b01_01_01_01>(y, x);
            let vec1 = _mm_shuffle_ps::<0b10_10_10_00>(tmp1, tmp1);
            let tmp2 = _mm_shuffle_ps::<0b10_10_10_10>(y, x);
            let vec2 = _mm_shuffle_ps::<0b10_10_10_00>(tmp2, tmp2);
            let tmp3 = _mm_shuffle_ps::<0b11_11_11_11>(y, x);
            let vec3 = _mm_shuffle_ps::<0b10_10_10_00>(tmp3, tmp3);

            let inv0 = _mm_mul_ps(sign_b, _mm_add_ps(
                _mm_sub_ps(_mm_mul_ps(vec1, fac0), _mm_mul_ps(vec2, fac1)),
                _mm_mul_ps(vec3, fac2),
            ));
            let inv1 = _mm_mul_ps(sign_a, _mm_add_ps(
                _mm_sub_ps(_mm_mul_ps(vec0, fac0), _mm_mul_ps(vec2, fac3)),
                _mm_mul_ps(vec3, fac4),
            ));
            let inv2 = _mm_mul_ps(sign_b, _mm_add_ps(
                _mm_sub_ps(_mm_mul_ps(vec0, fac1), _mm_mul_ps(vec1, fac3)),
                _mm_mul_ps(vec3, fac5),
            ));
            let inv3 = _mm_mul_ps(sign_a, _mm_add_ps(
                _mm_sub_ps(_mm_mul_ps(vec0, fac2), _mm_mul_ps(vec1, fac4)),
                _mm_mul_ps(vec2, fac5),
            ));

            let row0 = _mm_shuffle_ps::<0b00_00_00_00>(inv0, inv1);
            let row1 = _mm_shuffle_ps::<0b00_00_00_00>(inv2, inv3);
            let row2 = _mm_shuffle_ps::<0b10_00_10_00>(row0, row1);

            let det = dot4(x, row2);
            if det.abs() < EPSILON { return None; }

            let rcp = _mm_set1_ps(1.0 / det);
            Some(Self {
                x_axis: Vec4(_mm_mul_ps(inv0, rcp)),
                y_axis: Vec4(_mm_mul_ps(inv1, rcp)),
                z_axis: Vec4(_mm_mul_ps(inv2, rcp)),
                w_axis: Vec4(_mm_mul_ps(inv3, rcp)),
            })
        }
    }

    /// Scalar fallback inverse — exact same algorithm, no SIMD.
    /// Useful for unit testing correctness of the SSE2 path.
    pub fn inverse_scalar(self) -> Option<Self> {
        let a = [
            self.x_axis.x, self.x_axis.y, self.x_axis.z, self.x_axis.w,
            self.y_axis.x, self.y_axis.y, self.y_axis.z, self.y_axis.w,
            self.z_axis.x, self.z_axis.y, self.z_axis.z, self.z_axis.w,
            self.w_axis.x, self.w_axis.y, self.w_axis.z, self.w_axis.w,
        ];
        let mut inv = [0.0f32; 16];
        inv[ 0] =  a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10];
        inv[ 4] = -a[4]*a[10]*a[15]+a[4]*a[11]*a[14]+a[8]*a[6]*a[15]-a[8]*a[7]*a[14]-a[12]*a[6]*a[11]+a[12]*a[7]*a[10];
        inv[ 8] =  a[4]*a[9]*a[15]-a[4]*a[11]*a[13]-a[8]*a[5]*a[15]+a[8]*a[7]*a[13]+a[12]*a[5]*a[11]-a[12]*a[7]*a[9];
        inv[12] = -a[4]*a[9]*a[14]+a[4]*a[10]*a[13]+a[8]*a[5]*a[14]-a[8]*a[6]*a[13]-a[12]*a[5]*a[10]+a[12]*a[6]*a[9];
        inv[ 1] = -a[1]*a[10]*a[15]+a[1]*a[11]*a[14]+a[9]*a[2]*a[15]-a[9]*a[3]*a[14]-a[13]*a[2]*a[11]+a[13]*a[3]*a[10];
        inv[ 5] =  a[0]*a[10]*a[15]-a[0]*a[11]*a[14]-a[8]*a[2]*a[15]+a[8]*a[3]*a[14]+a[12]*a[2]*a[11]-a[12]*a[3]*a[10];
        inv[ 9] = -a[0]*a[9]*a[15]+a[0]*a[11]*a[13]+a[8]*a[1]*a[15]-a[8]*a[3]*a[13]-a[12]*a[1]*a[11]+a[12]*a[3]*a[9];
        inv[13] =  a[0]*a[9]*a[14]-a[0]*a[10]*a[13]-a[8]*a[1]*a[14]+a[8]*a[2]*a[13]+a[12]*a[1]*a[10]-a[12]*a[2]*a[9];
        inv[ 2] =  a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6];
        inv[ 6] = -a[0]*a[6]*a[15]+a[0]*a[7]*a[14]+a[4]*a[2]*a[15]-a[4]*a[3]*a[14]-a[12]*a[2]*a[7]+a[12]*a[3]*a[6];
        inv[10] =  a[0]*a[5]*a[15]-a[0]*a[7]*a[13]-a[4]*a[1]*a[15]+a[4]*a[3]*a[13]+a[12]*a[1]*a[7]-a[12]*a[3]*a[5];
        inv[14] = -a[0]*a[5]*a[14]+a[0]*a[6]*a[13]+a[4]*a[1]*a[14]-a[4]*a[2]*a[13]-a[12]*a[1]*a[6]+a[12]*a[2]*a[5];
        inv[ 3] = -a[1]*a[6]*a[11]+a[1]*a[7]*a[10]+a[5]*a[2]*a[11]-a[5]*a[3]*a[10]-a[9]*a[2]*a[7]+a[9]*a[3]*a[6];
        inv[ 7] =  a[0]*a[6]*a[11]-a[0]*a[7]*a[10]-a[4]*a[2]*a[11]+a[4]*a[3]*a[10]+a[8]*a[2]*a[7]-a[8]*a[3]*a[6];
        inv[11] = -a[0]*a[5]*a[11]+a[0]*a[7]*a[9]+a[4]*a[1]*a[11]-a[4]*a[3]*a[9]-a[8]*a[1]*a[7]+a[8]*a[3]*a[5];
        inv[15] =  a[0]*a[5]*a[10]-a[0]*a[6]*a[9]-a[4]*a[1]*a[10]+a[4]*a[2]*a[9]+a[8]*a[1]*a[6]-a[8]*a[2]*a[5];
        let det = a[0]*inv[0]+a[1]*inv[4]+a[2]*inv[8]+a[3]*inv[12];
        if det.abs() < EPSILON { return None; }
        let id = 1.0 / det;
        for v in inv.iter_mut() { *v *= id; }
        Some(Self::from_cols(
            [inv[0], inv[1], inv[2], inv[3]],
            [inv[4], inv[5], inv[6], inv[7]],
            [inv[8], inv[9], inv[10], inv[11]],
            [inv[12], inv[13], inv[14], inv[15]],
        ))
    }

    // ── TRS inverse (SSE2) ────────────────────────────────────────────────────
    //
    // ~2× faster than the general inverse for rotation+scale+translation matrices.
    // Algorithm: transpose the rotation part, scale by 1/|col|², negate translation.
    //
    // Derivation: for M = R×S where col_j has length s_j,
    //   M^-1 = S^-1 × R^T
    //   (M^-1)[i,j] = col_j[i] / s_j²
    //   inv_t = -(M^-1 × original_t)

    #[inline]
    pub fn inverse_trs(self) -> Self {
        unsafe {
            let c0 = self.x_axis.0;
            let c1 = self.y_axis.0;
            let c2 = self.z_axis.0;
            let c3 = self.w_axis.0;

            // Squared lengths of the three rotation columns.
            let sq0  = _mm_mul_ps(c0, c0);
            let sq1  = _mm_mul_ps(c1, c1);
            let sq2  = _mm_mul_ps(c2, c2);
            let zero = _mm_setzero_ps();

            // Horizontal sum: sums[i] = sq_i.x + sq_i.y + sq_i.z
            // Using 3-way transpose + column-wise add.
            let lo01 = _mm_unpacklo_ps(sq0, sq1);
            let lo2z = _mm_unpacklo_ps(sq2, zero);
            let hi01 = _mm_unpackhi_ps(sq0, sq1);
            let hi2z = _mm_unpackhi_ps(sq2, zero);
            let row0 = _mm_movelh_ps(lo01, lo2z); // [sq0.x, sq1.x, sq2.x, 0]
            let row1 = _mm_movehl_ps(lo2z, lo01); // [sq0.y, sq1.y, sq2.y, 0]
            let row2 = _mm_movelh_ps(hi01, hi2z); // [sq0.z, sq1.z, sq2.z, 0]
            let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);
            // sums = [sx², sy², sz², 0]

            // Safe reciprocals: guard against near-zero scale.
            let eps  = _mm_set1_ps(EPSILON);
            let mask = _mm_cmpge_ps(sums, eps);
            let safe = _mm_or_ps(
                _mm_and_ps(mask, sums),
                _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
            );
            let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe));
            // inv_scales = [1/sx², 1/sy², 1/sz², 0]

            // Transpose the upper-3×3 of the rotation columns.
            let lo01_r = _mm_unpacklo_ps(c0, c1);
            let lo2z_r = _mm_unpacklo_ps(c2, zero);
            let hi01_r = _mm_unpackhi_ps(c0, c1);
            let hi2z_r = _mm_unpackhi_ps(c2, zero);
            let trow0 = _mm_movelh_ps(lo01_r, lo2z_r); // [c0.x, c1.x, c2.x, 0]
            let trow1 = _mm_movehl_ps(lo2z_r, lo01_r); // [c0.y, c1.y, c2.y, 0]
            let trow2 = _mm_movelh_ps(hi01_r, hi2z_r); // [c0.z, c1.z, c2.z, 0]

            // Scale each transposed row by the corresponding inverse squared scale.
            let ic0 = _mm_mul_ps(trow0, inv_scales);
            let ic1 = _mm_mul_ps(trow1, inv_scales);
            let ic2 = _mm_mul_ps(trow2, inv_scales);

            // Inverse translation: -(inv_rot × original_t)
            let tx = _mm_shuffle_ps::<0b00_00_00_00>(c3, c3);
            let ty = _mm_shuffle_ps::<0b01_01_01_01>(c3, c3);
            let tz = _mm_shuffle_ps::<0b10_10_10_10>(c3, c3);
            let dot_col = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(ic0, tx), _mm_mul_ps(ic1, ty)),
                _mm_mul_ps(ic2, tz),
            );
            let neg = _mm_sub_ps(zero, dot_col);

            // Set lane 3 of the translation column to 1.0 (SSE2-compatible).
            let mask3 = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
            let ic3 = _mm_or_ps(_mm_and_ps(neg, mask3), _mm_set_ps(1.0, 0.0, 0.0, 0.0));

            Self {
                x_axis: Vec4(ic0),
                y_axis: Vec4(ic1),
                z_axis: Vec4(ic2),
                w_axis: Vec4(ic3),
            }
        }
    }

    /// Scalar fallback TRS inverse.
    pub fn inverse_trs_scalar(self) -> Self {
        let sx2 = self.x_axis.x*self.x_axis.x + self.x_axis.y*self.x_axis.y + self.x_axis.z*self.x_axis.z;
        let sy2 = self.y_axis.x*self.y_axis.x + self.y_axis.y*self.y_axis.y + self.y_axis.z*self.y_axis.z;
        let sz2 = self.z_axis.x*self.z_axis.x + self.z_axis.y*self.z_axis.y + self.z_axis.z*self.z_axis.z;
        let isx = if sx2 < EPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < EPSILON { 0.0 } else { 1.0 / sy2 };
        let isz = if sz2 < EPSILON { 0.0 } else { 1.0 / sz2 };
        let ic0 = [self.x_axis.x*isx, self.y_axis.x*isy, self.z_axis.x*isz, 0.0];
        let ic1 = [self.x_axis.y*isx, self.y_axis.y*isy, self.z_axis.y*isz, 0.0];
        let ic2 = [self.x_axis.z*isx, self.y_axis.z*isy, self.z_axis.z*isz, 0.0];
        let (tx, ty, tz) = (self.w_axis.x, self.w_axis.y, self.w_axis.z);
        let itx = -(ic0[0]*tx + ic1[0]*ty + ic2[0]*tz);
        let ity = -(ic0[1]*tx + ic1[1]*ty + ic2[1]*tz);
        let itz = -(ic0[2]*tx + ic1[2]*ty + ic2[2]*tz);
        Self::from_cols(ic0, ic1, ic2, [itx, ity, itz, 1.0])
    }

    // ── Wide SIMD batch transforms ────────────────────────────────────────────
    //
    // transform_point for 4 Vec3 simultaneously (SoA layout).
    // Column components broadcast via shuffle — avoids scalar extraction.

    pub fn transform_vec3x4(
        self,
        v: crate::wide::float::sse2::vec3x4::Vec3x4,
    ) -> crate::wide::float::sse2::vec3x4::Vec3x4 {
        use crate::wide::float::sse2::vec3x4::Vec3x4;
        unsafe {
            // Broadcast xyz components of each matrix column.
            let c0x = _mm_shuffle_ps::<0b00_00_00_00>(self.x_axis.0, self.x_axis.0);
            let c0y = _mm_shuffle_ps::<0b01_01_01_01>(self.x_axis.0, self.x_axis.0);
            let c0z = _mm_shuffle_ps::<0b10_10_10_10>(self.x_axis.0, self.x_axis.0);
            let c1x = _mm_shuffle_ps::<0b00_00_00_00>(self.y_axis.0, self.y_axis.0);
            let c1y = _mm_shuffle_ps::<0b01_01_01_01>(self.y_axis.0, self.y_axis.0);
            let c1z = _mm_shuffle_ps::<0b10_10_10_10>(self.y_axis.0, self.y_axis.0);
            let c2x = _mm_shuffle_ps::<0b00_00_00_00>(self.z_axis.0, self.z_axis.0);
            let c2y = _mm_shuffle_ps::<0b01_01_01_01>(self.z_axis.0, self.z_axis.0);
            let c2z = _mm_shuffle_ps::<0b10_10_10_10>(self.z_axis.0, self.z_axis.0);
            let c3x = _mm_shuffle_ps::<0b00_00_00_00>(self.w_axis.0, self.w_axis.0);
            let c3y = _mm_shuffle_ps::<0b01_01_01_01>(self.w_axis.0, self.w_axis.0);
            let c3z = _mm_shuffle_ps::<0b10_10_10_10>(self.w_axis.0, self.w_axis.0);
            let rx = _mm_add_ps(_mm_add_ps(_mm_mul_ps(c0x, v.x), _mm_mul_ps(c1x, v.y)),
                                 _mm_add_ps(_mm_mul_ps(c2x, v.z), c3x));
            let ry = _mm_add_ps(_mm_add_ps(_mm_mul_ps(c0y, v.x), _mm_mul_ps(c1y, v.y)),
                                 _mm_add_ps(_mm_mul_ps(c2y, v.z), c3y));
            let rz = _mm_add_ps(_mm_add_ps(_mm_mul_ps(c0z, v.x), _mm_mul_ps(c1z, v.y)),
                                 _mm_add_ps(_mm_mul_ps(c2z, v.z), c3z));
            Vec3x4 { x: rx, y: ry, z: rz }
        }
    }

    pub fn transform_vec3x4_dir(
        self,
        v: crate::wide::float::sse2::vec3x4::Vec3x4,
    ) -> crate::wide::float::sse2::vec3x4::Vec3x4 {
        use crate::wide::float::sse2::vec3x4::Vec3x4;
        unsafe {
            let c0x = _mm_shuffle_ps::<0b00_00_00_00>(self.x_axis.0, self.x_axis.0);
            let c0y = _mm_shuffle_ps::<0b01_01_01_01>(self.x_axis.0, self.x_axis.0);
            let c0z = _mm_shuffle_ps::<0b10_10_10_10>(self.x_axis.0, self.x_axis.0);
            let c1x = _mm_shuffle_ps::<0b00_00_00_00>(self.y_axis.0, self.y_axis.0);
            let c1y = _mm_shuffle_ps::<0b01_01_01_01>(self.y_axis.0, self.y_axis.0);
            let c1z = _mm_shuffle_ps::<0b10_10_10_10>(self.y_axis.0, self.y_axis.0);
            let c2x = _mm_shuffle_ps::<0b00_00_00_00>(self.z_axis.0, self.z_axis.0);
            let c2y = _mm_shuffle_ps::<0b01_01_01_01>(self.z_axis.0, self.z_axis.0);
            let c2z = _mm_shuffle_ps::<0b10_10_10_10>(self.z_axis.0, self.z_axis.0);
            let rx = _mm_add_ps(_mm_mul_ps(c0x, v.x), _mm_add_ps(_mm_mul_ps(c1x, v.y), _mm_mul_ps(c2x, v.z)));
            let ry = _mm_add_ps(_mm_mul_ps(c0y, v.x), _mm_add_ps(_mm_mul_ps(c1y, v.y), _mm_mul_ps(c2y, v.z)));
            let rz = _mm_add_ps(_mm_mul_ps(c0z, v.x), _mm_add_ps(_mm_mul_ps(c1z, v.y), _mm_mul_ps(c2z, v.z)));
            Vec3x4 { x: rx, y: ry, z: rz }
        }
    }
}

// ── Mul<Vec4> ─────────────────────────────────────────────────────────────────
//
// THE KEY FIX: self.x_axis.0 etc. are __m128 values — no _mm_load_ps needed.
//
// LLVM emits 4 broadcasts of v's components + 1 mul + 3 add/fmadd operations.
// With -C target-cpu=native (FMA) the add(_mul_) pattern becomes vfmadd213ps:
//   vmulps    xmm_res, x_axis, vx
//   vfmadd231ps xmm_res, y_axis, vy   ; res += y_axis * vy
//   vfmadd231ps xmm_res, z_axis, vz
//   vfmadd231ps xmm_res, w_axis, vw

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        unsafe {
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(v.0, v.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(v.0, v.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(v.0, v.0);
            let bw = _mm_shuffle_ps::<0b11_11_11_11>(v.0, v.0);
            let res = _mm_mul_ps(self.x_axis.0, bx);
            let res = _mm_add_ps(res, _mm_mul_ps(self.y_axis.0, by));
            let res = _mm_add_ps(res, _mm_mul_ps(self.z_axis.0, bz));
            Vec4(_mm_add_ps(res, _mm_mul_ps(self.w_axis.0, bw)))
        }
    }
}

// ── Mul<Mat4> ─────────────────────────────────────────────────────────────────
//
// Calls Mul<Vec4> four times. Because self's columns are Vec4 fields,
// LLVM keeps them live in XMM4-XMM7 across all four calls — zero memory traffic
// for the LHS. Each RHS column is loaded once into XMM0 (4 loads total).
//
// Total instruction count (SSE2, no FMA): 4 shuffles + 4 muls + 12 adds per col
//   × 4 cols = 80 scalar-equivalent ops → ~7 ns at 4 GHz
// With FMA: 4 shuffles + 1 mul + 3 FMAs per col × 4 cols = 32 ops → ~3.5 ns

// Gate the SSE2 implementation so it steps aside when AVX + FMA is active
#[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x_axis: self * rhs.x_axis,
            y_axis: self * rhs.y_axis,
            z_axis: self * rhs.z_axis,
            w_axis: self * rhs.w_axis,
        }
    }
}

// MulAssign is ungated — it automatically calls whichever Mul is in scope.
impl MulAssign for Mat4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}


impl Default for Mat4 {
    #[inline]
    fn default() -> Self { Self::IDENTITY }
}

impl fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat4")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis)
            .field("w_axis", &self.w_axis)
            .finish()
    }
}

impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..4 {
            let x = match r { 0 => self.x_axis.x, 1 => self.x_axis.y, 2 => self.x_axis.z, _ => self.x_axis.w };
            let y = match r { 0 => self.y_axis.x, 1 => self.y_axis.y, 2 => self.y_axis.z, _ => self.y_axis.w };
            let z = match r { 0 => self.z_axis.x, 1 => self.z_axis.y, 2 => self.z_axis.z, _ => self.z_axis.w };
            let w = match r { 0 => self.w_axis.x, 1 => self.w_axis.y, 2 => self.w_axis.z, _ => self.w_axis.w };
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}  {:8.4}]", x, y, z, w)?;
        }
        Ok(())
    }
}
