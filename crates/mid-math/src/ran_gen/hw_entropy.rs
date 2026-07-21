// crates/mid-math/src/ran_gen/hw_entropy.rs
//! Hardware entropy sourcing for RNG seeding — RDSEED / RDRAND.
//!
//! x86_64 only, and only when built with the `rdseed`/`rdrand` target
//! features (they aren't implied by `target-cpu=native` unless the host
//! CPU actually has them — most do since Ivy Bridge/Excavator, but this
//! is opt-in via `-C target-feature=+rdrand,+rdseed` or `target-cpu=native`
//! on hardware that has them). On every other target, or if neither
//! feature is compiled in, [`hardware_seed_u64`] just returns `None`.
//!
//! This is a *seed source*, not a CSPRNG replacement — [`crate::Xorshift64`]
//! and [`crate::Pcg32`] don't become cryptographically secure just because
//! the seed came from real entropy. Use this when you want a different
//! sequence per run without managing a seed yourself; keep a deterministic
//! fallback ready for targets/builds where it returns `None`.

/// Intel's own guidance for both RDRAND and RDSEED: the underlying
/// conditioning hardware can transiently fall behind under contention
/// (heavy concurrent use across cores) and report failure via the carry
/// flag. Retry a bounded number of times before giving up — 10 is Intel's
/// documented recommendation, not a number pulled out of the air.
///
/// #[allow(dead_code)]: only referenced inside the x86_64+rdrand/rdseed
/// branches below. On every other target (confirmed via the neon bench
/// build) neither branch compiles in, so this would otherwise warn as
/// unused there even though it's genuinely used on the targets it's for.
#[allow(dead_code)]
const HW_ENTROPY_RETRY_BUDGET: u32 = 10;

#[cfg(all(target_arch = "x86_64", target_feature = "rdseed"))]
#[inline]
fn try_rdseed64() -> Option<u64> {
    use core::arch::x86_64::_rdseed64_step;
    let mut out: u64 = 0;
    for _ in 0..HW_ENTROPY_RETRY_BUDGET {
        // Returns 1 on success (carry flag set), 0 on transient failure.
        if unsafe { _rdseed64_step(&mut out) } == 1 {
            return Some(out);
        }
    }
    None
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "rdseed")))]
#[inline]
fn try_rdseed64() -> Option<u64> { None }

#[cfg(all(target_arch = "x86_64", target_feature = "rdrand"))]
#[inline]
fn try_rdrand64() -> Option<u64> {
    use core::arch::x86_64::_rdrand64_step;
    let mut out: u64 = 0;
    for _ in 0..HW_ENTROPY_RETRY_BUDGET {
        if unsafe { _rdrand64_step(&mut out) } == 1 {
            return Some(out);
        }
    }
    None
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "rdrand")))]
#[inline]
fn try_rdrand64() -> Option<u64> { None }

/// One 64-bit value sourced from hardware entropy, for seeding
/// [`crate::Xorshift64`] or [`crate::Pcg32`] without a fixed/predictable seed.
///
/// Prefers RDSEED (samples closer to the physical noise source) over RDRAND
/// (a hardware CSPRNG that's periodically reseeded from that same source) —
/// falls back to RDRAND only when RDSEED isn't compiled in.
///
/// Returns `None` on non-x86_64 targets, on builds without these target
/// features enabled, or in the rare case both exhaust their retry budget.
/// Not a hard dependency — treat it as "better seed when available".
#[inline]
pub fn hardware_seed_u64() -> Option<u64> {
    try_rdseed64().or_else(try_rdrand64)
}
