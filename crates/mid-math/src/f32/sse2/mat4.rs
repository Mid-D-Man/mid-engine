// crates/mid-math/src/f32/sse2/mat4.rs
//!
//! ── Build history ──────────────────────────────────────────────────────────
//!
//! Build 8:  Vec4 field storage — killed 2.5× mat4/mul gap (17 ns → 7 ns).
//! Build 19: perspective_rh sin_cos fix (closed). look_at transpose (partial).
//!           from_trs cvtss+shuffle extraction — still 2.33× gap because
//!           scale muls stay scalar inside _mm_set_ps args.
//!
//! Build 20 (this file):
//!   quat_to_axes_sse2 — fully vectorized, zero scalar intermediates, zero
//!   stack spills. All 9 products computed as __m128 shuffle+mul chains.
//!   Three output columns assembled with unpack/movelh/shuffle sequences.
//!
//!   from_trs — calls quat_to_axes_sse2, then scales each column with a
//!   single vmulps per axis. Scale extraction stays in XMM via splat-shuffle
//!   (no cvtss_f32). Translation: OR t.0 with W_ONE constant.
//!
//!   look_at_rh / look_at_lh — SoA w-axis dot: after the transpose that
//!   builds x/y/z columns, the 3 dot products with eye are one SIMD
//!   matrix-vector multiply (3 broadcast-muls + 2 adds) rather than 3
//!   sequential dot3 calls. Closes the remaining 1.13× gap.

use core::fmt;
use core::ops::{Mul, MulAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4, m128_from_f32x4};
use crate::f32::math;
use crate::f32::sse2::vec3::Vec3;
use crate::f32::sse2::vec4::Vec4;
use crate::f32::sse2::quat::Quat;
use crate::EPSILON;

// ── Module-level SIMD constants ───────────────────────────────────────────────

/// All-ones lanes 0-2, lane 3 = 0.  Used for AND-masking the w/padding lane.
const XYZ_MASK: __m128 = m128_from_f32x4([
    f32::from_bits(0xFFFF_FFFF),
    f32::from_bits(0xFFFF_FFFF),
    f32::from_bits(0xFFFF_FFFF),
    0.0_f32,
]);

/// 0.0 in lanes 0-2, 1.0 in lane 3.  OR into a result to set w = 1.
const W_ONE: __m128 = m128_from_f32x4([0.0, 0.0, 0.0, 1.0]);

/// Sign-flip mask for lanes 0-2 only (XOR to negate xyz, leave w unchanged).
const NEG_XYZ: __m128 = m128_from_f32x4([-0.0, -0.0, -0.0, 0.0]);

// ── quat_to_axes_sse2 ─────────────────────────────────────────────────────────

/// Convert a normalized quaternion to three rotation-matrix columns.
///
/// **Fully vectorized** — no `_mm_cvtss_f32`, no scalar f32 temporaries,
/// no stack spills.  All 9 products (xx yy zz xy xz yz wx wy wz) are
/// computed as `__m128` shuffle + mul chains and assembled into the three
/// output columns without ever leaving XMM registers.
///
/// # Algorithm
///
/// ```text
/// q = [x, y, z, w]      q2 = q + q = [x2, y2, z2, w2]
///
/// v_x  = splat_x(q) * q2 = [xx, xy, xz, xw]   (xw unused)
/// v_y  = splat_y(q) * q2 = [xy, yy, yz, yw]   (xy/yy/yw unused)
/// v_w  = splat_w(q) * q2 = [wx, wy, wz, ww]   (ww unused after mask)
/// diag = q * q2           = [xx, yy, zz, ww]   (ww not picked by t1/t2)
///
/// v_cross   = shuffle(v_x,v_y, 0xA9) & XYZ_MASK = [xy, xz, yz, 0]
/// v_w_rev   = shuffle(v_w,v_w, 0x06) & XYZ_MASK = [wz, wy, wx, 0]
/// v_add     = v_cross + v_w_rev = [xy+wz, xz+wy, yz+wx, 0]
/// v_sub     = v_cross - v_w_rev = [xy-wz, xz-wy, yz-wx, 0]
///
/// t1         = shuffle(diag, 0x01) = [yy, xx, xx, xx]
/// t2         = shuffle(diag, 0x1A) = [zz, zz, yy, xx]
/// one_minus  = 1 - (t1 + t2)      = [1-(yy+zz), 1-(xx+zz), 1-(xx+yy), *]
///
/// x_axis = [one_minus[0], v_add[0], v_sub[1], 0]
/// y_axis = [v_sub[0],  one_minus[1], v_add[2], 0]
/// z_axis = [v_add[1],  v_sub[2],  one_minus[2], 0]
/// ```
///
/// # Safety
/// Requires SSE2 target feature (always present on x86_64).
#[inline(always)]
pub(crate) unsafe fn quat_to_axes_sse2(q: __m128) -> (__m128, __m128, __m128) {
    // ── products ──────────────────────────────────────────────────────────────
    let q2 = _mm_add_ps(q, q);

    let x_splat = _mm_shuffle_ps::<0b00_00_00_00>(q, q);   // [x,x,x,x]
    let y_splat = _mm_shuffle_ps::<0b01_01_01_01>(q, q);   // [y,y,y,y]
    let w_splat = _mm_shuffle_ps::<0b11_11_11_11>(q, q);   // [w,w,w,w]

    let v_x  = _mm_mul_ps(x_splat, q2);   // [xx, xy, xz, xw]  — xw unused
    let v_y  = _mm_mul_ps(y_splat, q2);   // [xy, yy, yz, yw]  — others unused
    let v_w  = _mm_mul_ps(w_splat, q2);   // [wx, wy, wz, ww]
    let diag = _mm_mul_ps(q, q2);         // [xx, yy, zz, ww]  — ww not selected

    // ── cross and w terms ─────────────────────────────────────────────────────
    //
    // v_cross = [xy, xz, yz, 0]
    //   _mm_shuffle_ps::<0xA9>(v_x, v_y):
    //     0xA9 = 0b10_10_10_01 → i0=1,i1=2 from v_x, i2=2,i3=2 from v_y
    //     = [v_x[1], v_x[2], v_y[2], v_y[2]] = [xy, xz, yz, yz]
    //     AND with XYZ_MASK → [xy, xz, yz, 0]
    let v_cross = _mm_and_ps(
        _mm_shuffle_ps::<0xA9>(v_x, v_y),
        XYZ_MASK,
    );

    // v_w_rev = [wz, wy, wx, 0]   (reversed so add/sub gives correct sign pairs)
    //   _mm_shuffle_ps::<0x06>(v_w, v_w):
    //     0x06 = 0b00_00_01_10 → i0=2,i1=1,i2=0,i3=0
    //     = [v_w[2], v_w[1], v_w[0], v_w[0]] = [wz, wy, wx, wx]
    //     AND with XYZ_MASK → [wz, wy, wx, 0]
    let v_w_rev = _mm_and_ps(
        _mm_shuffle_ps::<0x06>(v_w, v_w),
        XYZ_MASK,
    );

    // v_add = [xy+wz, xz+wy, yz+wx, 0]
    // v_sub = [xy-wz, xz-wy, yz-wx, 0]
    let v_add = _mm_add_ps(v_cross, v_w_rev);
    let v_sub = _mm_sub_ps(v_cross, v_w_rev);

    // ── diagonal (one-minus) terms ────────────────────────────────────────────
    //
    // We need: [1-(yy+zz), 1-(xx+zz), 1-(xx+yy), *] for the three diagonal slots.
    //
    // t1: 0x01 = 0b00_00_00_01 → [diag[1], diag[0], diag[0], diag[0]] = [yy, xx, xx, xx]
    // t2: 0x1A = 0b00_01_10_10 → [diag[2], diag[2], diag[1], diag[0]] = [zz, zz, yy, xx]
    // sums: [yy+zz, xx+zz, xx+yy, 2xx]   (lane 3 garbage — never selected)
    let t1          = _mm_shuffle_ps::<0x01>(diag, diag);
    let t2          = _mm_shuffle_ps::<0x1A>(diag, diag);
    let diag_sums   = _mm_add_ps(t1, t2);
    let one_minus   = _mm_sub_ps(_mm_set1_ps(1.0_f32), diag_sums);
    // one_minus = [1-(yy+zz), 1-(xx+zz), 1-(xx+yy), garbage]

    // ── column assembly ───────────────────────────────────────────────────────
    //
    // Target:
    //   x_axis = [one_minus[0], v_add[0], v_sub[1], 0]  → [a, p, R, 0]
    //   y_axis = [v_sub[0], one_minus[1], v_add[2], 0]  → [P, b, q, 0]
    //   z_axis = [v_add[1], v_sub[2], one_minus[2], 0]  → [r, Q, c, 0]
    //
    // Naming: a=1-(yy+zz), b=1-(xx+zz), c=1-(xx+yy)
    //         p=xy+wz, q=yz+wx, r=xz+wy
    //         P=xy-wz, Q=yz-wx, R=xz-wy

    let zero = _mm_setzero_ps();

    // x_axis: [a, p, R, 0]
    //   t_sub_lo    = unpacklo(v_sub, zero) = [P, 0, R, 0]
    //   t_om_add_lo = unpacklo(one_minus, v_add) = [a, p, b, r]
    //   shuffle::<0x64>(t_om_add_lo, t_sub_lo):
    //     0x64 = 0b01_10_01_00 → i0=0,i1=1 from t_om_add_lo, i2=2,i3=1 from t_sub_lo
    //     = [a, p, R, 0] ✓
    let t_sub_lo    = _mm_unpacklo_ps(v_sub, zero);
    let t_om_add_lo = _mm_unpacklo_ps(one_minus, v_add);
    let x_axis      = _mm_shuffle_ps::<0x64>(t_om_add_lo, t_sub_lo);

    // y_axis: [P, b, q, 0]
    //   t_lo_y = movelh(v_sub, one_minus) = [P, R, a, b]
    //   shuffle::<0xEC>(t_lo_y, v_add):
    //     0xEC = 0b11_10_11_00 → i0=0,i1=3 from t_lo_y, i2=2,i3=3 from v_add
    //     = [P, b, q, 0] ✓  (v_add[3] = 0)
    let t_lo_y = _mm_movelh_ps(v_sub, one_minus);
    let y_axis = _mm_shuffle_ps::<0xEC>(t_lo_y, v_add);

    // z_axis: [r, Q, c, 0]
    //   t_add_sub_hi = unpackhi(v_add, v_sub) = [q, Q, 0, 0]
    //   t_r_c        = shuffle::<0xA5>(v_add, one_minus)
    //                    0xA5 = 0b10_10_01_01 → [r, r, c, c]
    //   t_blend      = shuffle::<0x40>(t_r_c, t_add_sub_hi)
    //                    0x40 = 0b01_00_00_00 → [r, r, q, Q]
    //   zero_c       = shuffle::<0x0A>(t_r_c, zero)
    //                    0x0A = 0b00_00_10_10 → [c, c, 0, 0]
    //   z_axis       = shuffle::<0x8C>(t_blend, zero_c)
    //                    0x8C = 0b10_00_11_00 → [r, Q, c, 0] ✓
    let t_add_sub_hi = _mm_unpackhi_ps(v_add, v_sub);
    let t_r_c        = _mm_shuffle_ps::<0xA5>(v_add, one_minus);
    let t_blend      = _mm_shuffle_ps::<0x40>(t_r_c, t_add_sub_hi);
    let zero_c       = _mm_shuffle_ps::<0x0A>(t_r_c, zero);
    let z_axis       = _mm_shuffle_ps::<0x8C>(t_blend, zero_c);

    (x_axis, y_axis, z_axis)
}

// ─────────────────────────────────────────────────────────────────────────────

/// 4×4 column-major matrix. 64 bytes, 16-byte aligned.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    pub x_axis: Vec4,
    pub y_axis: Vec4,
    pub z_axis: Vec4,
    pub w_axis: Vec4,
}

impl Mat4 {
    pub const ZERO: Self = Self {
        x_axis: Vec4::ZERO, y_axis: Vec4::ZERO,
        z_axis: Vec4::ZERO, w_axis: Vec4::ZERO,
    };
    pub const IDENTITY: Self = Self {
        x_axis: Vec4::X, y_axis: Vec4::Y,
        z_axis: Vec4::Z, w_axis: Vec4::W,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline]
    pub fn from_cols(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) -> Self {
        Self {
            x_axis: Vec4::from_array(c0), y_axis: Vec4::from_array(c1),
            z_axis: Vec4::from_array(c2), w_axis: Vec4::from_array(c3),
        }
    }

    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            x_axis: Vec4::X, y_axis: Vec4::Y, z_axis: Vec4::Z,
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

    /// Full TRS — scale, then rotate, then translate.
    ///
    /// Build 20 fix: `quat_to_axes_sse2` produces three rotation columns as
    /// `__m128` with zero scalar intermediates. Scale is applied via three
    /// `vmulps` (one per column). Translation is set by ORing `t.0` (which
    /// already has lane 3 = 0 from Vec3 contract) with `W_ONE = [0,0,0,1]`.
    ///
    /// No `_mm_cvtss_f32`, no `_mm_set_ps`, no scalar multiplications, no
    /// stack spills. Expected: ~6 ns (parity with glam).
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        unsafe {
            let q = r.normalize().0;
            let (xc, yc, zc) = quat_to_axes_sse2(q);

            // Scale: broadcast each component then vmulps — stays in XMM.
            // No cvtss_f32: shuffle directly to get splat.
            let sx = _mm_shuffle_ps::<0b00_00_00_00>(s.0, s.0);
            let sy = _mm_shuffle_ps::<0b01_01_01_01>(s.0, s.0);
            let sz = _mm_shuffle_ps::<0b10_10_10_10>(s.0, s.0);

            // Translation: t.0 = [tx, ty, tz, 0] → OR with W_ONE → [tx, ty, tz, 1]
            let w = _mm_or_ps(t.0, W_ONE);

            Self {
                x_axis: Vec4(_mm_mul_ps(xc, sx)),
                y_axis: Vec4(_mm_mul_ps(yc, sy)),
                z_axis: Vec4(_mm_mul_ps(zc, sz)),
                w_axis: Vec4(w),
            }
        }
    }

    // ── View matrices ─────────────────────────────────────────────────────────

    /// Right-handed look-at view matrix.
    ///
    /// Build 20: w_axis computed as a SoA dot product using the already-built
    /// columns instead of three sequential `dot3` calls.
    ///
    /// After the unpack/movelh/movehl transpose:
    ///   x_axis = [r.x, u.x, -f.x, 0]
    ///   y_axis = [r.y, u.y, -f.y, 0]
    ///   z_axis = [r.z, u.z, -f.z, 0]
    ///
    /// The w translation is:
    ///   w.xyz = -(x_axis*eye.x + y_axis*eye.y + z_axis*eye.z).xyz
    ///         = [-r·eye, -u·eye, f·eye, 0]   (note: -f is already in col z lanes)
    ///
    /// One SIMD matrix-vector multiply (3 broadcast-muls + 2 adds) replaces
    /// three independent dot3 chains.
    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f  = (center - eye).normalize();
        let r  = f.cross(up).normalize();
        let u  = r.cross(f);
        let nf = -f;
        unsafe {
            // Transpose to column-major view layout
            let tmp0 = _mm_unpacklo_ps(r.0, u.0);   // [r.x, u.x, r.y, u.y]
            let tmp1 = _mm_unpacklo_ps(nf.0, _mm_setzero_ps()); // [nf.x, 0, nf.y, 0]
            let tmp2 = _mm_unpackhi_ps(r.0, u.0);   // [r.z, u.z, 0,    0]
            let tmp3 = _mm_unpackhi_ps(nf.0, _mm_setzero_ps()); // [nf.z, 0, 0,    0]

            let xc = _mm_movelh_ps(tmp0, tmp1);  // [r.x,  u.x,  -f.x, 0]
            let yc = _mm_movehl_ps(tmp1, tmp0);  // [r.y,  u.y,  -f.y, 0]
            let zc = _mm_movelh_ps(tmp2, tmp3);  // [r.z,  u.z,  -f.z, 0]

            // SoA dot: dot_xyz[i] = column_i · eye
            //   dot_xyz[0] = r·eye, [1] = u·eye, [2] = (-f)·eye = -f·eye, [3] = 0
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(eye.0, eye.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(eye.0, eye.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(eye.0, eye.0);
            let dot_xyz = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(xc, bx), _mm_mul_ps(yc, by)),
                _mm_mul_ps(zc, bz),
            );
            // dot_xyz = [r·eye, u·eye, -f·eye, 0]
            // w_axis  = [-r·eye, -u·eye, f·eye, 1]  — negate xyz, set w=1
            let neg = _mm_xor_ps(dot_xyz, NEG_XYZ);  // [-r·eye, -u·eye, f·eye, 0]
            let wc  = _mm_or_ps(neg, W_ONE);          // [-r·eye, -u·eye, f·eye, 1]

            Self {
                x_axis: Vec4(xc), y_axis: Vec4(yc),
                z_axis: Vec4(zc), w_axis: Vec4(wc),
            }
        }
    }

    /// Left-handed look-at view matrix.  Same SoA w-dot as `look_at_rh`.
    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let r = up.cross(f).normalize();
        let u = f.cross(r);
        unsafe {
            let zero = _mm_setzero_ps();
            let tmp0 = _mm_unpacklo_ps(r.0, u.0);
            let tmp1 = _mm_unpacklo_ps(f.0, zero);
            let tmp2 = _mm_unpackhi_ps(r.0, u.0);
            let tmp3 = _mm_unpackhi_ps(f.0, zero);

            let xc = _mm_movelh_ps(tmp0, tmp1);  // [r.x, u.x, f.x, 0]
            let yc = _mm_movehl_ps(tmp1, tmp0);  // [r.y, u.y, f.y, 0]
            let zc = _mm_movelh_ps(tmp2, tmp3);  // [r.z, u.z, f.z, 0]

            // dot_xyz = [r·eye, u·eye, f·eye, 0]
            // w_axis  = [-r·eye, -u·eye, -f·eye, 1]
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(eye.0, eye.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(eye.0, eye.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(eye.0, eye.0);
            let dot_xyz = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(xc, bx), _mm_mul_ps(yc, by)),
                _mm_mul_ps(zc, bz),
            );
            let neg = _mm_xor_ps(dot_xyz, NEG_XYZ);
            let wc  = _mm_or_ps(neg, W_ONE);

            Self {
                x_axis: Vec4(xc), y_axis: Vec4(yc),
                z_axis: Vec4(zc), w_axis: Vec4(wc),
            }
        }
    }

    // ── Projection matrices ───────────────────────────────────────────────────

    /// `cos/sin` instead of `1/tan` — one `sin_cos` call, faster on all targets.
    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin_fov, cos_fov) = math::sin_cos(fov_y * 0.5);
        let f = cos_fov / sin_fov;
        let z = near - far;
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, (far + near) / z, -1.0),
            w_axis: Vec4::new(0.0, 0.0, (2.0 * far * near) / z, 0.0),
        }
    }

    pub fn perspective_lh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin_fov, cos_fov) = math::sin_cos(fov_y * 0.5);
        let f = cos_fov / sin_fov;
        let z = far - near;
        Self {
            x_axis: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, far / z, 1.0),
            w_axis: Vec4::new(0.0, 0.0, -(far * near) / z, 0.0),
        }
    }

    pub fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -2.0 / nf, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rl, -(top + bottom) / tb, -(far + near) / nf, 1.0,
            ),
        }
    }

    pub fn ortho_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = right - left; let tb = top - bottom; let nf = far - near;
        Self {
            x_axis: Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 1.0 / nf, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rl, -(top + bottom) / tb, -near / nf, 1.0,
            ),
        }
    }

    // ── Transpose ─────────────────────────────────────────────────────────────

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

            let subres  = _mm_sub_ps(mulfaca, mulfacb);
            let subtmpc = _mm_shuffle_ps::<0b01_00_10_10>(sube, subf);
            let subfacc = _mm_shuffle_ps::<0b11_11_10_00>(subtmpc, subtmpc);
            let swpfacc = _mm_shuffle_ps::<0b10_11_11_11>(y, y);
            let mulfacc = _mm_mul_ps(swpfacc, subfacc);

            let addres  = _mm_add_ps(subres, mulfacc);
            let detcof  = _mm_mul_ps(addres, _mm_setr_ps(1.0, -1.0, 1.0, -1.0));

            dot4(self.x_axis.0, detcof)
        }
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

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
        let t  = self.w_axis.truncate();
        let sx = self.x_axis.truncate().length();
        let sy = self.y_axis.truncate().length();
        let sz = self.z_axis.truncate().length();
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

    // ── General inverse (SSE2 cofactor) ──────────────────────────────────────

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

    #[inline]
    pub fn inverse_trs(self) -> Self {
        unsafe {
            let c0 = self.x_axis.0; let c1 = self.y_axis.0;
            let c2 = self.z_axis.0; let c3 = self.w_axis.0;

            let sq0  = _mm_mul_ps(c0, c0);
            let sq1  = _mm_mul_ps(c1, c1);
            let sq2  = _mm_mul_ps(c2, c2);
            let zero = _mm_setzero_ps();

            let lo01 = _mm_unpacklo_ps(sq0, sq1);
            let lo2z = _mm_unpacklo_ps(sq2, zero);
            let hi01 = _mm_unpackhi_ps(sq0, sq1);
            let hi2z = _mm_unpackhi_ps(sq2, zero);
            let row0 = _mm_movelh_ps(lo01, lo2z);
            let row1 = _mm_movehl_ps(lo2z, lo01);
            let row2 = _mm_movelh_ps(hi01, hi2z);
            let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);

            let eps  = _mm_set1_ps(EPSILON);
            let mask = _mm_cmpge_ps(sums, eps);
            let safe = _mm_or_ps(
                _mm_and_ps(mask, sums),
                _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
            );
            let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe));

            let lo01_r = _mm_unpacklo_ps(c0, c1);
            let lo2z_r = _mm_unpacklo_ps(c2, zero);
            let hi01_r = _mm_unpackhi_ps(c0, c1);
            let hi2z_r = _mm_unpackhi_ps(c2, zero);
            let trow0  = _mm_movelh_ps(lo01_r, lo2z_r);
            let trow1  = _mm_movehl_ps(lo2z_r, lo01_r);
            let trow2  = _mm_movelh_ps(hi01_r, hi2z_r);

            let ic0 = _mm_mul_ps(trow0, inv_scales);
            let ic1 = _mm_mul_ps(trow1, inv_scales);
            let ic2 = _mm_mul_ps(trow2, inv_scales);

            let tx = _mm_shuffle_ps::<0b00_00_00_00>(c3, c3);
            let ty = _mm_shuffle_ps::<0b01_01_01_01>(c3, c3);
            let tz = _mm_shuffle_ps::<0b10_10_10_10>(c3, c3);
            let dot_col = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(ic0, tx), _mm_mul_ps(ic1, ty)),
                _mm_mul_ps(ic2, tz),
            );
            let neg = _mm_sub_ps(zero, dot_col);
            let mask3 = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
            let ic3   = _mm_or_ps(_mm_and_ps(neg, mask3), _mm_set_ps(1.0, 0.0, 0.0, 0.0));

            Self {
                x_axis: Vec4(ic0), y_axis: Vec4(ic1),
                z_axis: Vec4(ic2), w_axis: Vec4(ic3),
            }
        }
    }

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

    pub fn transform_vec3x4(
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

// ── Mul<Mat4> — gated so AVX+FMA path in avx/mat4.rs takes over ──────────────

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

impl MulAssign for Mat4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl Default for Mat4 {
    #[inline] fn default() -> Self { Self::IDENTITY }
}

impl fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat4")
            .field("x_axis", &self.x_axis).field("y_axis", &self.y_axis)
            .field("z_axis", &self.z_axis).field("w_axis", &self.w_axis)
            .finish()
    }
}

impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..4 {
            let x = match r { 0=>self.x_axis.x, 1=>self.x_axis.y, 2=>self.x_axis.z, _=>self.x_axis.w };
            let y = match r { 0=>self.y_axis.x, 1=>self.y_axis.y, 2=>self.y_axis.z, _=>self.y_axis.w };
            let z = match r { 0=>self.z_axis.x, 1=>self.z_axis.y, 2=>self.z_axis.z, _=>self.z_axis.w };
            let w = match r { 0=>self.w_axis.x, 1=>self.w_axis.y, 2=>self.w_axis.z, _=>self.w_axis.w };
            writeln!(f, "  [{:8.4}  {:8.4}  {:8.4}  {:8.4}]", x, y, z, w)?;
        }
        Ok(())
    }
}
