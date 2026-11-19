// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "lib.rs"
// ============================================================================
//! Runtime description of Rust types: what a type's name and fields are, and
//! a way to read and write a value of it without knowing the type at compile
//! time. The foundation under inspection tools, scene files and a remote
//! protocol (`docs/roadmap.md`, Decision 7).
//!
//! - [`Value`] is a small self-describing tree. [`Value::to_json`] and
//!   [`Value::from_json`] move it to and from JSON text; [`Value::get_path`]
//!   and [`Value::set_path`] reach into it with paths like `translation.x`.
//! - [`Reflect`] turns a type into a [`Value`] and back. The macros
//!   [`impl_reflect_struct!`], [`impl_reflect_newtype!`] and
//!   [`impl_reflect_enum!`] implement it, under a stable name you choose.
//! - [`TypeRegistry`] holds the named types a program registered and reads,
//!   writes and constructs them by name, through `dyn Any`.
//!
//! No external dependencies. With the `math` feature, `mid-math`'s vectors,
//! quaternions and affine transforms are reflected too.
//!
//! ```
//! use mid_reflect::{impl_reflect_struct, TypeRegistry, Value};
//!
//! #[derive(Debug, PartialEq)]
//! struct Health { current: u32, max: u32 }
//! impl_reflect_struct!("game::Health" => Health { current: u32, max: u32 });
//!
//! let mut registry = TypeRegistry::new();
//! registry.register::<Health>().unwrap();
//!
//! let mut health = Health { current: 3, max: 10 };
//! registry.mutate("game::Health", &mut health, "current", Value::Int(7)).unwrap();
//! assert_eq!(health, Health { current: 7, max: 10 });
//! ```

mod error;
mod json;
mod macros;
#[cfg(feature = "math")]
mod math;
mod reflect;
mod registry;
mod ty;
mod value;

pub use error::{ErrorKind, PathSeg, ReflectError};
pub use json::{JsonError, JsonErrorKind, DEFAULT_MAX_DEPTH};
#[doc(hidden)]
pub use reflect::private;
pub use reflect::Reflect;
pub use registry::{Registration, RegistryError, TypeRegistry};
pub use ty::{Field, Kind, Ty, TypeInfo};
pub use value::{path_string, Value};
