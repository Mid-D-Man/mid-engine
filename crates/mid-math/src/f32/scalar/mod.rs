// crates/mid-math/src/f32/scalar/mod.rs
// Fix: suppress unused import warnings — scalar types are the fallback,
// but on x86_64 the sse2 types are active so these re-exports are unused there.

pub mod vec3;
pub mod vec4;
pub mod quat;
pub mod mat4;

#[allow(unused_imports)]
pub use vec3::Vec3;
#[allow(unused_imports)]
pub use vec4::Vec4;
#[allow(unused_imports)]
pub use quat::Quat;
#[allow(unused_imports)]
pub use mat4::Mat4;
