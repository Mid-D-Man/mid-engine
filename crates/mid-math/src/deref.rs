// crates/mid-math/src/deref.rs
// crates/mid-math/src/deref.rs
//! View structs that SIMD-backed types Deref into for field access.

/// Column view for 2×2 matrices packed into a single register.
/// Gives `.x_axis` / `.y_axis` field access on the SSE2 Mat2.
/// Memory layout must match: x_axis at offset 0, y_axis at offset 8.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Cols2<T> {
    pub x_axis: T,
    pub y_axis: T,
}

/// Component view for 2D types.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct XY<T> {
    pub x: T,
    pub y: T,
}

/// Component view for 3D SIMD types (covers x, y, z — lane 3 is padding).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct XYZ<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// Component view for 4D SIMD types and quaternions.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct XYZW<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// ── f32 SIMD deref macros ─────────────────────────────────────────────────────

/// Implement Deref/DerefMut to XYZ<f32> for a #[repr(transparent)] __m128 newtype.
/// Lane layout must be: 0=x, 1=y, 2=z, 3=padding.
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

/// Implement Deref/DerefMut to XYZW<f32> for a #[repr(transparent)] __m128 newtype.
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

// ── f64 SIMD deref macros ─────────────────────────────────────────────────────

/// Implement Deref/DerefMut to XY<f64> for a #[repr(transparent)] __m128d newtype.
///
/// Memory layout requirement: lane 0 = x (bytes 0-7), lane 1 = y (bytes 8-15).
#[macro_export]
macro_rules! impl_dvec2_deref {
    ($ty:ty) => {
        impl core::ops::Deref for $ty {
            type Target = $crate::deref::XY<f64>;
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

/// Implement Deref/DerefMut to XYZW<f64> for a `#[repr(C, align(32))]` type
/// whose first 32 bytes map to [x, y, z, w] as consecutive f64.
#[macro_export]
macro_rules! impl_dvec4_deref {
    ($ty:ty) => {
        impl core::ops::Deref for $ty {
            type Target = $crate::deref::XYZW<f64>;
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
