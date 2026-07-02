// crates/mid-math/src/f32/avx/vec4.rs
//! AVX + FMA `Vec4::lerp`.
//!
//! Active when: avx + fma present (module gated in `f32/mod.rs`, same
//! condition as `avx::mat4`). No avx512f exclusion needed here — unlike
//! `mat4::mul`, there's no wider-register win to give up; FMA is FMA
//! regardless of avx512f presence.
//!
//! ## Win
//! `self + (rhs - self) * t` is a classic multiply-add. SSE2 baseline in
//! `sse2/vec4.rs` computes it as `sub` → `mul` → `add` (3 instructions after
//! the broadcast). `_mm_fmadd_ps` fuses the last two into one, and — because
//! it's a *fused* multiply-add — rounds once instead of twice, which is
//! marginally more accurate as a bonus, not just faster.
//!
//! `sse2/vec4.rs` gates its own `lerp()` out via
//! `#[cfg(not(all(target_feature = "avx", target_feature = "fma")))]` so
//! exactly one implementation exists per target at all times, same pattern
//! as `avx/mat4.rs`'s `Mul<Mat4>` override.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::vec4::Vec4;

impl Vec4 {
    /// AVX+FMA lerp: `self + (rhs - self) * t` fused into one FMA.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let tt  = _mm_set1_ps(t);
            let sub = _mm_sub_ps(rhs.0, self.0);
            Self(_mm_fmadd_ps(sub, tt, self.0))
        }
    }
  }
