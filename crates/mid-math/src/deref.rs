// crates/mid-math/src/deref.rs
//! View structs that SIMD-backed types Deref into for .x/.y/.z/.w access.

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
