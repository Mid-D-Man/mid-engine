// crates/mid-math/src/wide/float/neon/mod.rs
//! NEON-backed float wide types — aarch64.
//!
//! All three types receive NEON implementations.
//!
//! ## Key advantages over SSE2
//!
//! | Operation              | SSE2                    | NEON                     |
//! |------------------------|-------------------------|--------------------------|
//! | AoS→SoA transpose (4) | 7× _mm_shuffle_ps       | 1× vld4q_f32             |
//! | SoA→AoS transpose (4) | 7× shuffle              | 1× vst4q_f32             |
//! | Per-lane blend/select  | AND + ANDNOT + OR (3)   | 1× vbslq_f32             |
//! | Lerp / madd           | mul + add (2)           | 1× vfmaq_f32 (mandatory) |
//! | rsqrt NR step         | manual 0.5*r*(3-x*r*r)  | 1× vrsqrtsq_f32          |
//! | rcp NR step           | manual r*(2-x*r)        | 1× vrecpsq_f32           |
//! | abs (float)           | ANDNOT sign mask        | 1× vabsq_f32             |
//! | Comparison result     | __m128 float bits       | uint32x4_t (clean)       |
//!
//! ## vld4q_f32 / vst4q_f32 detail
//!
//! Both Vec3 and Quat store their components in `float32x4_t` with a known lane
//! layout ([x,y,z,pad] and [x,y,z,w] respectively). When 4 of these are stored
//! contiguously in memory:
//!
//! ```text
//! memory: [x0,y0,z0,p0, x1,y1,z1,p1, x2,y2,z2,p2, x3,y3,z3,p3]  ← 16 floats
//!
//! vld4q_f32 deinterleaves:
//!   val[0] = [x0,x1,x2,x3]   ← SoA x lane
//!   val[1] = [y0,y1,y2,y3]   ← SoA y lane
//!   val[2] = [z0,z1,z2,z3]   ← SoA z lane
//!   val[3] = [p0,p1,p2,p3]   ← padding / w lane
//! ```
//!
//! For Vec3x4, val[3] (padding) is discarded.
//! For QuatX4, val[3] is the w lane — all 4 components in ONE instruction.
//!
//! `vst4q_f32` performs the exact reverse in one instruction.

pub mod mask4;
pub mod f32x4;
pub mod vec3x4;
pub mod quatx4;

pub use mask4::{Mask4, Mask4LaneIter};
pub use vec3x4::Vec3x4;
pub use quatx4::QuatX4;
// f32x4 type NOT re-exported at module level (name clash with module).
// Parent accesses it as neon::f32x4::f32x4.
