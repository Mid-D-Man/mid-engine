// crates/mid-math/src/wide/float/scalar/mod.rs
pub mod mask4;
pub mod f32x4;   // keep module public — parent resolves type via f32x4::f32x4
pub mod vec3x4;
pub mod quatx4;

pub use mask4::Mask4;
// NOTE: f32x4 (type) is NOT re-exported here — that would clash with the
// module of the same name. The parent (wide/float/mod.rs) imports the type
// explicitly as scalar::f32x4::f32x4.
pub use vec3x4::Vec3x4;
pub use quatx4::QuatX4;
