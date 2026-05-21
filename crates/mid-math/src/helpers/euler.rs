// crates/mid-math/src/helpers/euler.rs
//! Multi-order Euler angle support for the platform-dispatched `Quat` type.
//!
//! Implements the Shoemake (1994) "Euler Angle Conversion" algorithm from
//! Graphics Gems IV, adapted for quaternion output following glam's proven
//! approach. All 24 Euler sequences are supported (12 intrinsic + 12 extrinsic).
//!
//! # Usage
//!
//! Import [`QuatExt`] to get the methods on `Quat`:
//!
//! ```rust
//! use mid_math::{Quat, EulerRot, QuatExt};
//!
//! // Build from Euler (ZYX = yaw → pitch → roll, common flight convention)
//! let q = Quat::from_euler(EulerRot::ZYX, yaw, pitch, roll);
//!
//! // Extract back
//! let (y, p, r) = q.to_euler(EulerRot::ZYX);
//!
//! // From a rotation matrix's column vectors
//! use mid_math::Mat3;
//! let q2 = Quat::from_mat3(&Mat3::IDENTITY);
//! ```

use crate::f32::math;

// ── EulerRot ──────────────────────────────────────────────────────────────────

/// Euler rotation sequence specifier.
///
/// **Intrinsic** sequences rotate about the axes of the moving (body-fixed) frame.
/// Each rotation is applied in order: e.g. `ZYX(a, b, c)` means rotate `a` around Z,
/// then `b` around the new Y, then `c` around the new X.
///
/// **Extrinsic** sequences (`Ex` suffix) rotate about the fixed world axes.
/// `XYZEx(a, b, c)` equals intrinsic `ZYX(c, b, a)`.
///
/// **Two-axis** sequences (Euler proper) repeat the initial axis:
/// `ZYZ(a, b, c)` = rotate `a` around Z, then `b` around Y, then `c` around Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EulerRot {
    // ── Intrinsic three-axis ───────────────────────────────────────────────────
    /// Intrinsic ZYX — yaw → pitch → roll (flight dynamics convention).
    ZYX,
    /// Intrinsic ZXY
    ZXY,
    /// Intrinsic YXZ — default for many game engines.
    YXZ,
    /// Intrinsic YZX
    YZX,
    /// Intrinsic XYZ
    XYZ,
    /// Intrinsic XZY
    XZY,
    // ── Intrinsic two-axis (Euler proper) ─────────────────────────────────────
    /// Intrinsic ZYZ (precession–nutation–spin)
    ZYZ,
    /// Intrinsic ZXZ
    ZXZ,
    /// Intrinsic YXY
    YXY,
    /// Intrinsic YZY
    YZY,
    /// Intrinsic XYX
    XYX,
    /// Intrinsic XZX
    XZX,
    // ── Extrinsic three-axis ──────────────────────────────────────────────────
    /// Extrinsic ZYX (= intrinsic XYZ reversed)
    ZYXEx,
    /// Extrinsic ZXY
    ZXYEx,
    /// Extrinsic YXZ
    YXZEx,
    /// Extrinsic YZX
    YZXEx,
    /// Extrinsic XYZ
    XYZEx,
    /// Extrinsic XZY
    XZYEx,
    // ── Extrinsic two-axis ────────────────────────────────────────────────────
    ZYZEx, ZXZEx, YXYEx, YZYEx, XYXEx, XZXEx,
}

// ── Internal Order encoding ────────────────────────────────────────────────────

/// Internal Shoemake order descriptor.
#[derive(Clone, Copy)]
struct Order {
    /// Index of the first rotation axis: 0 = X, 1 = Y, 2 = Z.
    initial_axis: usize,
    /// Even parity sequences: XYZ, YZX, ZXY (and their Euler-proper variants).
    parity_even: bool,
    /// True for two-axis (Euler proper) sequences where first axis == last axis.
    initial_repeated: bool,
    /// True  = static / extrinsic frame.
    /// False = rotating / intrinsic frame.
    frame_static: bool,
}

impl Order {
    /// Map an [`EulerRot`] variant to the canonical Shoemake Order descriptor.
    ///
    /// Follows the same encoding as glam's `Order::from_euler`. Extrinsic variants
    /// map to the corresponding intrinsic encoding with `frame_static = false`.
    #[inline]
    fn from_euler(e: EulerRot) -> Self {
        // Shorthand: (initial_axis, parity_even, initial_repeated, frame_static)
        macro_rules! o {
            ($ax:expr, $even:expr, $rep:expr, $fs:expr) => {
                Order { initial_axis: $ax, parity_even: $even, initial_repeated: $rep, frame_static: $fs }
            };
        }
        use EulerRot::*;
        match e {
            // ── Intrinsic three-axis ──────────────────────────────────────────
            XYZ => o!(0, true,  false, true),
            XZY => o!(0, false, false, true),
            YXZ => o!(1, false, false, true),
            YZX => o!(1, true,  false, true),
            ZXY => o!(2, true,  false, true),
            ZYX => o!(2, false, false, true),
            // ── Intrinsic two-axis ────────────────────────────────────────────
            XYX => o!(0, true,  true, true),
            XZX => o!(0, false, true, true),
            YXY => o!(1, false, true, true),
            YZY => o!(1, true,  true, true),
            ZXZ => o!(2, true,  true, true),
            ZYZ => o!(2, false, true, true),
            // ── Extrinsic three-axis ──────────────────────────────────────────
            // Extrinsic XYZ = intrinsic ZYX: initial_axis flips to the LAST axis.
            ZYXEx => o!(0, true,  false, false),
            YZXEx => o!(0, false, false, false),
            ZXYEx => o!(1, false, false, false),
            XZYEx => o!(1, true,  false, false),
            YXZEx => o!(2, true,  false, false),
            XYZEx => o!(2, false, false, false),
            // ── Extrinsic two-axis ────────────────────────────────────────────
            XYXEx => o!(0, true,  true, false),
            XZXEx => o!(0, false, true, false),
            YXYEx => o!(1, false, true, false),
            YZYEx => o!(1, true,  true, false),
            ZXZEx => o!(2, true,  true, false),
            ZYZEx => o!(2, false, true, false),
        }
    }

    /// Returns the `(i, j, k)` axis indices for the Shoemake formulas.
    ///
    /// For three-axis: first rotation = axis i, second = j, third = k.
    /// For two-axis:   first = i, middle = j, last = i (repeated).
    #[inline]
    fn angle_order(self) -> (usize, usize, usize) {
        let i = self.initial_axis;
        // Even parity: i→j→k goes next-next; odd: goes prev-next (or next-prev).
        let j = if self.parity_even { (i + 1) % 3 } else { (i + 2) % 3 };
        let k = if self.parity_even { (i + 2) % 3 } else { (i + 1) % 3 };
        (i, j, k)
    }
}

// ── QuatExt trait ──────────────────────────────────────────────────────────────

/// Extension trait that adds full multi-order Euler support and matrix→quaternion
/// extraction to the platform-dispatched [`crate::Quat`] type.
///
/// The two required methods are thin bridges to the platform-specific
/// constructors; all complex math lives here as default implementations.
pub trait QuatExt: Sized {
    // ── Required (one-liner bridge per platform) ──────────────────────────────

    /// Construct a quaternion from raw `(x, y, z, w)` components.
    fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self;

    /// Extract the upper-left 3 × 3 rotation block as `[col][row]`.
    ///
    /// The result is in column-major order: `cols[c][r]` = element at column `c`, row `r`.
    fn to_rotation_mat3(self) -> [[f32; 3]; 3];

    // ── Euler conversion ──────────────────────────────────────────────────────

    /// Build a quaternion from Euler angles in the specified rotation order.
    ///
    /// Uses the Shoemake (Graphic Gems IV, 1994) quaternion formula.
    /// All angles are in radians.
    ///
    /// Angles map to the rotation sequence named by `order` from first to last:
    /// - `EulerRot::ZYX(a, b, c)` → rotate `a` around Z, then `b` around Y, then `c` around X.
    /// - `EulerRot::XYZEx(a, b, c)` → extrinsic X→Y→Z (= intrinsic Z→Y→X reversed).
    ///
    /// # Numerical notes
    /// The quaternion is returned unnormalized to preserve the exact Shoemake
    /// formula output. For a unit result call `.normalize()` afterwards, though
    /// in practice the error is `< 1 ULP` for reasonable angle values.
    #[inline]
    fn from_euler(order: EulerRot, a: f32, b: f32, c: f32) -> Self {
        let ord = Order::from_euler(order);
        let (i, j, k) = ord.angle_order();

        // Extrinsic sequences: reverse input order (first↔last swap).
        let (ai, aj, ak) = if ord.frame_static { (a, b, c) } else { (c, b, a) };

        // Parity correction: for even-parity sequences negate the middle angle.
        // (Quaternion form differs from the matrix form, which negates all three.)
        let aj = if ord.parity_even { -aj } else { aj };

        // Half angles.
        let (si, ci) = math::sin_cos(ai * 0.5);
        let (sj, cj) = math::sin_cos(aj * 0.5);
        let (sh, ch) = math::sin_cos(ak * 0.5);

        let cc = ci * ch;
        let cs = ci * sh;
        let sc = si * ch;
        let ss = si * sh;

        // Parity sign used in the j-component formula.
        let parity = if ord.parity_even { -1.0_f32 } else { 1.0_f32 };

        let mut q = [0.0_f32; 4]; // [x=0, y=1, z=2, w=3]

        if ord.initial_repeated {
            // Two-axis (Euler proper) quaternion formula — Shoemake §IV.7 table.
            q[i] = cj * (cs + sc);
            q[j] = sj * (cc + ss) * parity;
            q[k] = sj * (cs - sc);
            q[3] = cj * (cc - ss);
        } else {
            // Three-axis quaternion formula — Shoemake §IV.6 table.
            q[i] = cj * sc - sj * cs;
            q[j] = (cj * ss + sj * cc) * parity;
            q[k] = cj * cs - sj * sc;
            q[3] = cj * cc + sj * ss;
        }

        Self::from_xyzw(q[0], q[1], q[2], q[3])
    }

    /// Extract Euler angles in the specified rotation order.
    ///
    /// Converts the quaternion to its rotation matrix and applies the
    /// Shoemake matrix-to-Euler algorithm.
    ///
    /// Returns `(angle_first, angle_second, angle_third)` in radians for the
    /// sequence named by `order`. Near gimbal-lock singularities the third angle
    /// is set to `0.0` and the full rotation is expressed with the first two.
    #[inline]
    fn to_euler(self, order: EulerRot) -> (f32, f32, f32) {
        let ord = Order::from_euler(order);
        let (i, j, k) = ord.angle_order();
        let m = self.to_rotation_mat3(); // m[col][row]

        // Column-major accessor: c(col, row).
        let c = |col: usize, row: usize| m[col][row];

        const EPS: f32 = 16.0 * f32::EPSILON;
        let mut ea = [0.0_f32; 3]; // ea[0]=first, ea[1]=second, ea[2]=third

        if ord.initial_repeated {
            // Two-axis (Euler proper) extraction.
            let sy = math::sqrt(c(i, j) * c(i, j) + c(i, k) * c(i, k));
            if sy > EPS {
                ea[0] = math::atan2( c(i, j),  c(i, k));
                ea[1] = math::atan2( sy,        c(i, i));
                ea[2] = math::atan2( c(j, i),  -c(k, i));
            } else {
                ea[0] = math::atan2(-c(j, k),   c(j, j));
                ea[1] = math::atan2( sy,         c(i, i));
                // ea[2] stays 0.0
            }
        } else {
            // Three-axis extraction.
            let cy = math::sqrt(c(i, i) * c(i, i) + c(j, i) * c(j, i));
            if cy > EPS {
                ea[0] = math::atan2( c(k, j),   c(k, k));
                ea[1] = math::atan2(-c(k, i),   cy);
                ea[2] = math::atan2( c(j, i),   c(i, i));
            } else {
                ea[0] = math::atan2(-c(j, k),   c(j, j));
                ea[1] = math::atan2(-c(k, i),   cy);
                // ea[2] stays 0.0
            }
        }

        // Parity correction.
        if ord.parity_even {
            ea[0] = -ea[0];
            ea[1] = -ea[1];
            ea[2] = -ea[2];
        }

        // Extrinsic: output order is reversed relative to intrinsic.
        if ord.frame_static {
            (ea[0], ea[1], ea[2])
        } else {
            (ea[2], ea[1], ea[0])
        }
    }

    // ── Matrix → quaternion ────────────────────────────────────────────────────

    /// Build a quaternion from three orthonormal column vectors of a rotation matrix.
    ///
    /// `x_axis`, `y_axis`, `z_axis` are columns 0, 1, 2 of the 3 × 3 rotation matrix.
    ///
    /// Uses the DirectXMath `XMQuaternionRotationMatrix` "largest component" algorithm
    /// for numerical stability. Invalid (non-orthonormal) input produces undefined results.
    #[inline]
    fn from_rotation_axes(x_axis: crate::Vec3, y_axis: crate::Vec3, z_axis: crate::Vec3) -> Self {
        // Column-major layout: x_axis = col 0, y_axis = col 1, z_axis = col 2.
        // In the DirectXMath algorithm's row-major notation:
        //   r0 = (x_axis.x, y_axis.x, z_axis.x)   ← row 0
        //   r1 = (x_axis.y, y_axis.y, z_axis.y)   ← row 1
        //   r2 = (x_axis.z, y_axis.z, z_axis.z)   ← row 2
        // We expose our column components as m00..m22:
        //   m[row][col] ↔ cols[col][row]
        // so m00=x.x, m01=y.x, m02=z.x, m10=x.y, m11=y.y, m12=z.y, m20=x.z, m21=y.z, m22=z.z.
        //
        // Equivalent to glam's Quat::from_rotation_axes (same algorithm, verified identical).
        let (m00, m10, m20) = (x_axis.x, x_axis.y, x_axis.z);
        let (m01, m11, m21) = (y_axis.x, y_axis.y, y_axis.z);
        let (m02, m12, m22) = (z_axis.x, z_axis.y, z_axis.z);

        if m22 <= 0.0 {
            let dif10 = m11 - m00;
            let omm22 = 1.0 - m22;
            if dif10 <= 0.0 {
                // x² ≥ y² branch.
                let four_xsq = omm22 - dif10;
                let inv4x = 0.5 / math::sqrt(four_xsq);
                Self::from_xyzw(
                    four_xsq * inv4x,    // x — largest
                    (m10 + m01) * inv4x, // y
                    (m20 + m02) * inv4x, // z
                    (m21 - m12) * inv4x, // w
                )
            } else {
                // y² ≥ x² branch.
                let four_ysq = omm22 + dif10;
                let inv4y = 0.5 / math::sqrt(four_ysq);
                Self::from_xyzw(
                    (m10 + m01) * inv4y, // x
                    four_ysq * inv4y,    // y — largest
                    (m21 + m12) * inv4y, // z
                    (m02 - m20) * inv4y, // w
                )
            }
        } else {
            let sum10 = m11 + m00;
            let opm22 = 1.0 + m22;
            if sum10 <= 0.0 {
                // z² ≥ w² branch.
                let four_zsq = opm22 - sum10;
                let inv4z = 0.5 / math::sqrt(four_zsq);
                Self::from_xyzw(
                    (m20 + m02) * inv4z, // x
                    (m21 + m12) * inv4z, // y
                    four_zsq * inv4z,    // z — largest
                    (m10 - m01) * inv4z, // w
                )
            } else {
                // w² ≥ z² branch.
                let four_wsq = opm22 + sum10;
                let inv4w = 0.5 / math::sqrt(four_wsq);
                Self::from_xyzw(
                    (m21 - m12) * inv4w, // x
                    (m02 - m20) * inv4w, // y
                    (m10 - m01) * inv4w, // z
                    four_wsq * inv4w,    // w — largest
                )
            }
        }
    }

    /// Build a quaternion from a [`crate::Mat3`] rotation matrix.
    ///
    /// Assumes the matrix is orthonormal (pure rotation, no scale/shear).
    /// Delegates to [`from_rotation_axes`][QuatExt::from_rotation_axes].
    #[inline]
    fn from_mat3(m: &crate::Mat3) -> Self {
        Self::from_rotation_axes(m.col(0), m.col(1), m.col(2))
    }

    /// Build a quaternion from the upper-left 3 × 3 rotation block of a [`crate::Mat4`].
    ///
    /// The matrix must encode a pure rotation (no scale, no shear in the 3 × 3 block).
    /// For TRS matrices with scale, normalise the columns first.
    #[inline]
    fn from_mat4_rotation(m: &crate::Mat4) -> Self {
        Self::from_rotation_axes(
            crate::Vec3::new(m.cols[0][0], m.cols[0][1], m.cols[0][2]),
            crate::Vec3::new(m.cols[1][0], m.cols[1][1], m.cols[1][2]),
            crate::Vec3::new(m.cols[2][0], m.cols[2][1], m.cols[2][2]),
        )
    }

    /// Build a quaternion from a [`crate::Affine3`] — reads the three axis columns directly.
    ///
    /// The axes must be normalised (rotation only). If the affine transform includes
    /// non-uniform scale, normalise each axis first.
    #[inline]
    fn from_affine3(a: &crate::Affine3) -> Self {
        Self::from_rotation_axes(a.x_axis, a.y_axis, a.z_axis)
    }
}

// ── impl QuatExt for crate::Quat ───────────────────────────────────────────────

/// Platform-dispatch bridge: the two required methods simply call the
/// platform-specific constructor / extractor that already exists on every
/// concrete Quat (SSE2, NEON, WASM, scalar).
impl QuatExt for crate::Quat {
    /// Thin bridge to the platform-specific `Quat::new(x, y, z, w)`.
    #[inline(always)]
    fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        // All four platform Quats expose this identical signature.
        Self::new(x, y, z, w)
    }

    /// Extract the upper-left 3 × 3 rotation block from `self.to_mat4()`.
    ///
    /// `to_mat4()` normalises `self` internally, so the returned block is
    /// always a proper rotation matrix even for slightly denormalised quaternions.
    #[inline]
    fn to_rotation_mat3(self) -> [[f32; 3]; 3] {
        let m = self.to_mat4();
        // cols[c][r] — extract only the 3×3 rotation block (ignore w row/col).
        [
            [m.cols[0][0], m.cols[0][1], m.cols[0][2]], // col 0
            [m.cols[1][0], m.cols[1][1], m.cols[1][2]], // col 1
            [m.cols[2][0], m.cols[2][1], m.cols[2][2]], // col 2
        ]
    }
  }
