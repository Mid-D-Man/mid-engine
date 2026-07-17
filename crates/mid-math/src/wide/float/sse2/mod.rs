// crates/mid-math/src/wide/float/sse2/mod.rs
pub mod mask4;
pub mod f32x4;   // module — type accessed as f32x4::f32x4 by parent
pub mod vec3x4;
pub mod quatx4;

pub use mask4::{Mask4, Mask4LaneIter};
// f32x4 type NOT re-exported here (name clash). Parent does sse2::f32x4::f32x4.
pub use vec3x4::Vec3x4;
pub use quatx4::QuatX4;
