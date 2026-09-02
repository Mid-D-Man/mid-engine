// crates/mid-math/src/features/mint_conversions/f64.rs
//! mint conversions for `DQuat`/`DMat2`/`DMat3`/`DMat4`. Mirrors `f32.rs`,
//! but confirmed separately against source rather than assumed parallel —
//! real differences turned up: `DMat3`'s `from_cols` takes raw `[f64; 3]`
//! arrays (unlike `Mat3::from_cols`, which takes `Vec3` directly), and
//! `DMat4` has no `col(i)` accessor at all (unlike `Mat3`/`DMat3`) — reads
//! its public `cols: [[f64; 4]; 4]` array directly instead. `DMat2`/`DMat3`/
//! `DMat4` are all canonical (no backend split, unlike `DVec2`/`DVec4`/
//! `DQuat`), which doesn't change anything here — mint conversions only
//! ever target the public alias regardless, same as `f32.rs`.
//!
//! `.into()` resolves to whichever `From` impl the call site's expected
//! type actually needs (mint's own `Vector3<f64> -> [f64; 3]`, or this
//! crate's `Vector3<f64> -> DVec3` from `vectors.rs`) — same reasoning as
//! `f32.rs`, works for both of `DMat3`/`DMat4`'s different shapes without
//! needing separate code per shape.

use crate::{DMat2, DMat3, DMat4, DQuat};

// ── DQuat ────────────────────────────────────────────────────────────────────

impl From<mint::Quaternion<f64>> for DQuat {
    fn from(q: mint::Quaternion<f64>) -> Self {
        Self::new(q.v.x, q.v.y, q.v.z, q.s)
    }
}

impl From<DQuat> for mint::Quaternion<f64> {
    fn from(q: DQuat) -> Self {
        Self { s: q.w, v: mint::Vector3 { x: q.x, y: q.y, z: q.z } }
    }
}

impl mint::IntoMint for DQuat {
    type MintType = mint::Quaternion<f64>;
}

// ── DMat2 ────────────────────────────────────────────────────────────────────

impl From<mint::ColumnMatrix2<f64>> for DMat2 {
    fn from(m: mint::ColumnMatrix2<f64>) -> Self {
        Self { x_axis: m.x.into(), y_axis: m.y.into() }
    }
}

impl From<DMat2> for mint::ColumnMatrix2<f64> {
    fn from(m: DMat2) -> Self {
        Self { x: m.x_axis.into(), y: m.y_axis.into() }
    }
}

impl From<mint::RowMatrix2<f64>> for DMat2 {
    fn from(m: mint::RowMatrix2<f64>) -> Self {
        let col: DMat2 = DMat2 { x_axis: m.x.into(), y_axis: m.y.into() };
        col.transpose()
    }
}

impl From<DMat2> for mint::RowMatrix2<f64> {
    fn from(m: DMat2) -> Self {
        let mt = m.transpose();
        Self { x: mt.x_axis.into(), y: mt.y_axis.into() }
    }
}

impl mint::IntoMint for DMat2 {
    type MintType = mint::ColumnMatrix2<f64>;
}

// ── DMat3 ────────────────────────────────────────────────────────────────────

impl From<mint::ColumnMatrix3<f64>> for DMat3 {
    fn from(m: mint::ColumnMatrix3<f64>) -> Self {
        Self::from_cols(m.x.into(), m.y.into(), m.z.into())
    }
}

impl From<DMat3> for mint::ColumnMatrix3<f64> {
    fn from(m: DMat3) -> Self {
        Self { x: m.col(0).into(), y: m.col(1).into(), z: m.col(2).into() }
    }
}

impl From<mint::RowMatrix3<f64>> for DMat3 {
    fn from(m: mint::RowMatrix3<f64>) -> Self {
        DMat3::from_cols(m.x.into(), m.y.into(), m.z.into()).transpose()
    }
}

impl From<DMat3> for mint::RowMatrix3<f64> {
    fn from(m: DMat3) -> Self {
        let mt = m.transpose();
        Self { x: mt.col(0).into(), y: mt.col(1).into(), z: mt.col(2).into() }
    }
}

impl mint::IntoMint for DMat3 {
    type MintType = mint::ColumnMatrix3<f64>;
}

// ── DMat4 ────────────────────────────────────────────────────────────────────

impl From<mint::ColumnMatrix4<f64>> for DMat4 {
    fn from(m: mint::ColumnMatrix4<f64>) -> Self {
        Self::from_cols(m.x.into(), m.y.into(), m.z.into(), m.w.into())
    }
}

impl From<DMat4> for mint::ColumnMatrix4<f64> {
    fn from(m: DMat4) -> Self {
        Self { x: m.cols[0].into(), y: m.cols[1].into(), z: m.cols[2].into(), w: m.cols[3].into() }
    }
}

impl From<mint::RowMatrix4<f64>> for DMat4 {
    fn from(m: mint::RowMatrix4<f64>) -> Self {
        DMat4::from_cols(m.x.into(), m.y.into(), m.z.into(), m.w.into()).transpose()
    }
}

impl From<DMat4> for mint::RowMatrix4<f64> {
    fn from(m: DMat4) -> Self {
        let mt = m.transpose();
        Self { x: mt.cols[0].into(), y: mt.cols[1].into(), z: mt.cols[2].into(), w: mt.cols[3].into() }
    }
}

impl mint::IntoMint for DMat4 {
    type MintType = mint::ColumnMatrix4<f64>;
}
