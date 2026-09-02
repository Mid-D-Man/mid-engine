// crates/mid-math/src/features/mint_conversions/f32.rs
//! mint conversions for `Quat`/`Mat2`/`Mat3`/`Mat4` — hand-written, not
//! macro-generated like `vectors.rs`, because these three matrix types
//! genuinely don't share one shape: `Mat2`/`Mat4` store named
//! `x_axis`/`y_axis`(/`z_axis`/`w_axis`) `Vec2`/`Vec4` fields, `Mat3` stores
//! a public `cols: [[f32; 3]; 3]` array but its `from_cols` takes `Vec3`
//! args (not arrays) and has a `col(i) -> Vec3` accessor. Confirmed each of
//! these directly against source before writing anything here, not assumed
//! uniform from Mat2/Mat4's shape.
//!
//! Every conversion below reuses `vectors.rs`'s already-defined
//! `Vec2`/`Vec3`/`Vec4` <-> `mint::VectorN<f32>` conversions via `.into()` —
//! including the `Mat3`/`Mat4` cases where the target is actually a raw
//! `[f32; N]` array or `Vec3`/`Vec4` depending on the call site: `.into()`
//! resolves to whichever `From` impl matches what the function signature
//! actually wants (mint's own `Vector3<f32> -> [f32; 3]`, or this crate's
//! `Vector3<f32> -> Vec3`), so the same code works either way without
//! needing to special-case which shape a given matrix type happens to use.
//!
//! Row-major support mirrors glam exactly: build/read as if the mint value
//! were column-major, then `.transpose()` — `Mat2`/`Mat3`/`Mat4` all confirmed
//! to have `.transpose()` before relying on it here.

use crate::{Mat2, Mat3, Mat4, Quat};

// ── Quat ─────────────────────────────────────────────────────────────────────

impl From<mint::Quaternion<f32>> for Quat {
    fn from(q: mint::Quaternion<f32>) -> Self {
        Self::new(q.v.x, q.v.y, q.v.z, q.s)
    }
}

impl From<Quat> for mint::Quaternion<f32> {
    fn from(q: Quat) -> Self {
        Self { s: q.w, v: mint::Vector3 { x: q.x, y: q.y, z: q.z } }
    }
}

impl mint::IntoMint for Quat {
    type MintType = mint::Quaternion<f32>;
}

// ── Mat2 ─────────────────────────────────────────────────────────────────────

impl From<mint::ColumnMatrix2<f32>> for Mat2 {
    fn from(m: mint::ColumnMatrix2<f32>) -> Self {
        Self { x_axis: m.x.into(), y_axis: m.y.into() }
    }
}

impl From<Mat2> for mint::ColumnMatrix2<f32> {
    fn from(m: Mat2) -> Self {
        Self { x: m.x_axis.into(), y: m.y_axis.into() }
    }
}

impl From<mint::RowMatrix2<f32>> for Mat2 {
    fn from(m: mint::RowMatrix2<f32>) -> Self {
        let col: Mat2 = Mat2 { x_axis: m.x.into(), y_axis: m.y.into() };
        col.transpose()
    }
}

impl From<Mat2> for mint::RowMatrix2<f32> {
    fn from(m: Mat2) -> Self {
        let mt = m.transpose();
        Self { x: mt.x_axis.into(), y: mt.y_axis.into() }
    }
}

impl mint::IntoMint for Mat2 {
    type MintType = mint::ColumnMatrix2<f32>;
}

// ── Mat3 ─────────────────────────────────────────────────────────────────────

impl From<mint::ColumnMatrix3<f32>> for Mat3 {
    fn from(m: mint::ColumnMatrix3<f32>) -> Self {
        Self::from_cols(m.x.into(), m.y.into(), m.z.into())
    }
}

impl From<Mat3> for mint::ColumnMatrix3<f32> {
    fn from(m: Mat3) -> Self {
        Self { x: m.col(0).into(), y: m.col(1).into(), z: m.col(2).into() }
    }
}

impl From<mint::RowMatrix3<f32>> for Mat3 {
    fn from(m: mint::RowMatrix3<f32>) -> Self {
        Mat3::from_cols(m.x.into(), m.y.into(), m.z.into()).transpose()
    }
}

impl From<Mat3> for mint::RowMatrix3<f32> {
    fn from(m: Mat3) -> Self {
        let mt = m.transpose();
        Self { x: mt.col(0).into(), y: mt.col(1).into(), z: mt.col(2).into() }
    }
}

impl mint::IntoMint for Mat3 {
    type MintType = mint::ColumnMatrix3<f32>;
}

// ── Mat4 ─────────────────────────────────────────────────────────────────────

impl From<mint::ColumnMatrix4<f32>> for Mat4 {
    fn from(m: mint::ColumnMatrix4<f32>) -> Self {
        Self { x_axis: m.x.into(), y_axis: m.y.into(), z_axis: m.z.into(), w_axis: m.w.into() }
    }
}

impl From<Mat4> for mint::ColumnMatrix4<f32> {
    fn from(m: Mat4) -> Self {
        Self { x: m.x_axis.into(), y: m.y_axis.into(), z: m.z_axis.into(), w: m.w_axis.into() }
    }
}

impl From<mint::RowMatrix4<f32>> for Mat4 {
    fn from(m: mint::RowMatrix4<f32>) -> Self {
        let col: Mat4 = Mat4 {
            x_axis: m.x.into(), y_axis: m.y.into(), z_axis: m.z.into(), w_axis: m.w.into(),
        };
        col.transpose()
    }
}

impl From<Mat4> for mint::RowMatrix4<f32> {
    fn from(m: Mat4) -> Self {
        let mt = m.transpose();
        Self { x: mt.x_axis.into(), y: mt.y_axis.into(), z: mt.z_axis.into(), w: mt.w_axis.into() }
    }
}

impl mint::IntoMint for Mat4 {
    type MintType = mint::ColumnMatrix4<f32>;
}
