// crates/mid-math/src/deref.rs
//! "View" structs that SIMD-backed types `Deref` into for `.x/.y/.z/.w` access.
//!
//! On x86_64, Vec3 / Vec4 / Quat store a `__m128` register internally.
//! `__m128` is 16 bytes at 16-byte alignment, with four f32 values at byte
//! offsets 0, 4, 8, 12 — exactly matching the fields of these structs.
//!
//! The Deref impl in each SIMD type does:
//!   `unsafe { &*(self as *const Self).cast::<crate::deref::Vec3<f32>>() }`
//!
//! This gives zero-cost field access without breaking the SIMD storage.
//! The fourth lane of Vec3's __m128 (offset 12) is inaccessible through
//! `deref::Vec3` — it is the padding lane and must be kept as 0.0.
//!
//! Adapted from glam/src/deref.rs (MIT/Apache-2.0).

use core::ops::{Deref, DerefMut};

/// Component view for 2D types. Used by [`Vec2`][crate::f32::Vec2].
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct XY<T> {
    pub x: T,
    pub y: T,
}

/// Component view for 3D SIMD types. Covers x, y, z only.
/// The fourth lane (__m128 offset 12) is not accessible through this view.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct XYZ<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// Component view for 4D SIMD types and quaternions.
/// All four lanes are accessible.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct XYZW<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// ── Vec2 deref (scalar storage — direct field access) ─────────────────────────
// Vec2 doesn't use SIMD storage so it implements Deref to XY normally.
// This macro is used by the SSE2 Vec3/Vec4/Quat implementations below.

/// Implement `Deref` and `DerefMut` to `XYZ<f32>` for a `#[repr(transparent)]`
/// newtype wrapping `__m128`. Caller guarantees memory layout compatibility.
///
/// # Safety invariant
/// The concrete type must store `__m128` as its only field (transparent).
/// Lane 0 = x, lane 1 = y, lane 2 = z, lane 3 = padding (ignored by XYZ).
#[macro_export]
macro_rules! impl_vec3_deref {
    ($ty:ty) => {
        impl core::ops::Deref for $ty {
            type Target = $crate::deref::XYZ<f32>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self).cast() }
            }
        }
        impl core::ops::DerefMut for $ty {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self).cast() }
            }
        }
    };
}

/// Implement `Deref` / `DerefMut` to `XYZW<f32>` for a transparent `__m128` newtype.
#[macro_export]
macro_rules! impl_vec4_deref {
    ($ty:ty) => {
        impl core::ops::Deref for $ty {
            type Target = $crate::deref::XYZW<f32>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self).cast() }
            }
        }
        impl core::ops::DerefMut for $ty {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self).cast() }
            }
        }
    };
}
