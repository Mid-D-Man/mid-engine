// crates/mid-math/src/f32/avx/quat.rs
//! AVX + FMA `Quat::mul_quat` and `Quat::nlerp`.
//!
//! Active when: avx + fma present (module gated in `f32/mod.rs`).
//!
//! ## mul_quat — from "mul + ±1 mul + add" to "xor-sign + fmadd"
//!
//! The SSE2 baseline (`sse2/quat.rs`) computes the Hamilton product as:
//!   term_w = w1 · rhs
//!   term_x = (x1 · shuffle(rhs, WZYX)) · CONTROL_WZYX     ← 2 muls
//!   term_y = (y1 · shuffle(rhs, ZWXY)) · CONTROL_ZWXY     ← 2 muls
//!   term_z = (z1 · shuffle(rhs, YXWZ)) · CONTROL_YXWZ     ← 2 muls
//!   result = term_w + term_x + term_y + term_z            ← 3 adds
//! = 7 muls + 3 adds, on top of 7 shuffles.
//!
//! `CONTROL_*` are ±1.0 vectors — multiplying by them is really just a
//! per-lane sign flip, which is what `_mm_xor_ps` with a sign-bit mask does
//! in one cheap instruction instead of a full multiply. Pre-computing the
//! signed, shuffled copies of `rhs` up front (they don't depend on `lhs` at
//! all) then lets the four `r_component · signed_rhs` products accumulate
//! as a straight FMA chain:
//!
//!   acc = w1 · rhs                                        ← 1 mul (base term)
//!   acc = fmadd(x1, xor(shuffle(rhs, WZYX), SIGN_WZYX), acc)
//!   acc = fmadd(y1, xor(shuffle(rhs, ZWXY), SIGN_ZWXY), acc)
//!   acc = fmadd(z1, xor(shuffle(rhs, YXWZ), SIGN_YXWZ), acc)
//! = 1 mul + 3 xor + 3 fmadd, same 7 shuffles. Net: 3 fewer arithmetic
//! instructions, one rounding step per accumulation instead of two.
//!
//! Numerically equivalent to the SSE2 path — verified by hand against the
//! Hamilton product component-by-component (see derivation in the handover
//! notes); this is not a different algorithm, just a fused re-expression of
//! the same one.
//!
//! `sse2/quat.rs` gates `mul_quat`/`nlerp` out via
//! `#[cfg(not(all(target_feature = "avx", target_feature = "fma")))]`, same
//! pattern as `avx/mat4.rs`'s `Mul<Mat4>` override — exactly one
//! implementation per target.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4_into_m128, m128_from_f32x4};
use crate::f32::sse2::quat::Quat;

/// XOR mask equivalent of `CONTROL_WZYX = [1,-1,1,-1]` — flips lanes 1 and 3.
const SIGN_WZYX: __m128 = m128_from_f32x4([0.0, -0.0, 0.0, -0.0]);
/// XOR mask equivalent of `CONTROL_ZWXY = [1,1,-1,-1]` — flips lanes 2 and 3.
const SIGN_ZWXY: __m128 = m128_from_f32x4([0.0, 0.0, -0.0, -0.0]);
/// XOR mask equivalent of `CONTROL_YXWZ = [-1,1,1,-1]` — flips lanes 0 and 3.
const SIGN_YXWZ: __m128 = m128_from_f32x4([-0.0, 0.0, 0.0, -0.0]);

impl Quat {
    /// AVX+FMA quaternion Hamilton product. See module docs for the
    /// mul+mul+add → xor+fmadd derivation.
    #[inline]
    pub fn mul_quat(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.0;
            let rhs = rhs.0;

            let r_xxxx = _mm_shuffle_ps::<0b00_00_00_00>(lhs, lhs);
            let r_yyyy = _mm_shuffle_ps::<0b01_01_01_01>(lhs, lhs);
            let r_zzzz = _mm_shuffle_ps::<0b10_10_10_10>(lhs, lhs);
            let r_wwww = _mm_shuffle_ps::<0b11_11_11_11>(lhs, lhs);

            // Same shuffle chain as the SSE2 baseline — reordered copies of
            // rhs, independent of lhs, so these are hoisted before any FMA.
            let l_wzyx = _mm_shuffle_ps::<0b00_01_10_11>(rhs, rhs);
            let l_zwxy = _mm_shuffle_ps::<0b10_11_00_01>(l_wzyx, l_wzyx);
            let l_yxwz = _mm_shuffle_ps::<0b00_01_10_11>(l_zwxy, l_zwxy);

            // Sign-flip via XOR instead of multiply-by-±1.
            let signed_wzyx = _mm_xor_ps(l_wzyx, SIGN_WZYX);
            let signed_zwxy = _mm_xor_ps(l_zwxy, SIGN_ZWXY);
            let signed_yxwz = _mm_xor_ps(l_yxwz, SIGN_YXWZ);

            let acc = _mm_mul_ps(r_wwww, rhs);
            let acc = _mm_fmadd_ps(r_xxxx, signed_wzyx, acc);
            let acc = _mm_fmadd_ps(r_yyyy, signed_zwxy, acc);
            let acc = _mm_fmadd_ps(r_zzzz, signed_yxwz, acc);

            Self(acc)
        }
    }

    /// AVX+FMA nlerp: same shortest-path sign-fix as the SSE2 baseline, but
    /// the final blend `self + (rhs_adj - self) * t` is one `_mm_fmadd_ps`.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let dot_v    = dot4_into_m128(self.0, rhs.0);
            let sign_bit = _mm_and_ps(dot_v, _mm_set1_ps(-0.0f32));
            let rhs_adj  = _mm_xor_ps(rhs.0, sign_bit);
            let tt       = _mm_set1_ps(t);
            let sub      = _mm_sub_ps(rhs_adj, self.0);
            let lerped   = _mm_fmadd_ps(sub, tt, self.0);
            Self(lerped).normalize_fast()
        }
    }
      }
