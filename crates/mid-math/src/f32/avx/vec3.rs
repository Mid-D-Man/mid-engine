// crates/mid-math/src/f32/avx/vec3.rs
//! AVX + FMA `Vec3::cross`.
//!
//! Closes a gap in the original G3 pass — that work covered `Vec4::lerp`
//! and `Quat::mul_quat`/`nlerp` but never actually added `avx/vec3.rs`,
//! despite the plan calling for Vec3 too. `dot`/`normalize` were checked
//! and confirmed to have no FMA opportunity (the shared `dot3`/
//! `dot3_into_m128` helpers are a single `mul` followed by a pure
//! shuffle-add reduction chain — no repeated mul-then-add-different-operand
//! pattern for FMA to fuse) — only `cross` benefits here.
//!
//! ## Win
//! SSE2 baseline: `sub(mul(a_yzx, b_zxy), mul(a_zxy, b_yzx))` — 2 muls + 1
//! sub. Fusing the second mul directly into the subtraction via
//! `_mm_fmsub_ps` (computes `a*b - c` in one instruction) drops it to 1 mul
//! + 1 fmsub — same instruction count reduction pattern as the mat4 FMA
//! work, one fewer op on the critical path plus one fewer rounding step.
//!
//! `sse2/vec3.rs` gates its own `cross()` out via
//! `#[cfg(not(all(target_feature = "avx", target_feature = "fma")))]` so
//! exactly one implementation exists per target, same pattern as
//! `avx/vec4.rs`/`avx/quat.rs`.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::vec3::Vec3;

impl Vec3 {
    /// AVX+FMA cross product: `fmsub(a_yzx, b_zxy, mul(a_zxy, b_yzx))`.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            let a_yzx = _mm_shuffle_ps::<0b00_00_10_01>(self.0, self.0);
            let b_zxy = _mm_shuffle_ps::<0b00_01_00_10>(rhs.0,  rhs.0);
            let a_zxy = _mm_shuffle_ps::<0b00_01_00_10>(self.0, self.0);
            let b_yzx = _mm_shuffle_ps::<0b00_00_10_01>(rhs.0,  rhs.0);
            Self(_mm_fmsub_ps(a_yzx, b_zxy, _mm_mul_ps(a_zxy, b_yzx)))
        }
    }
              }
