// crates/mid-math/src/curves/mod.rs
//! Curve and spline primitives for Mid Engine.
//!
//! All curve types operate on `Vec2` and `Vec3` via the `Interpolate` trait.
//! The trait is also implemented for `f32`, `f64`, and `Quat` so the same
//! curve machinery works for scalars and rotations.
//!
//! Implemented types (game-dev relevance in descending order):
//!
//! | Type               | Continuity | Passes through points | Game use                      |
//! |--------------------|------------|----------------------|-------------------------------|
//! | `CatmullRom`       | C¹         | Yes                  | Camera paths, movement tracks |
//! | `CubicBezier`      | C⁰ at join | No (approximating)   | UI animation, simple arcs     |
//! | `HermiteSpline`    | C¹         | Yes (+ tangents)     | Animation curve editors       |
//! | `KochanekBartels`  | C¹ (TCB)   | Yes                  | Cinematic keyframe animation  |
//! | `BSpline`          | C²         | No (approximating)   | Smooth paths, local control   |
//! | `CardinalSpline`   | C¹         | Yes                  | Tension-controlled paths      |

pub mod interpolate;
pub mod bezier;
pub mod catmull_rom;
pub mod hermite;
pub mod kochanek_bartels;
pub mod bspline;
pub mod cardinal;

pub use interpolate::Interpolate;
pub use bezier::{QuadraticBezier, CubicBezier};
pub use catmull_rom::CatmullRom;
pub use hermite::HermiteSpline;
pub use kochanek_bartels::KochanekBartels;
pub use bspline::BSpline;
pub use cardinal::CardinalSpline;
