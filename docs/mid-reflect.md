# mid-reflect

## Overview

`mid-reflect` describes Rust types at runtime. A program registers the types it wants to expose
under stable names, and a tool can then list their fields, read a value as a tree, write a value
back from a tree, and move that tree to and from JSON, without knowing the type at compile time.

It is the foundation for three things the roadmap needs and does not have yet: inspecting and
editing a running world (an inspector, `mid world get` and `set`), scene files, and a remote
protocol (`docs/roadmap.md`, Decision 7).

What it is made of:

- `Value`, a small self-describing tree (null, bool, integers, float, string, list, and a map that
  keeps its order), with JSON in and out and path-based reads and edits such as `translation.x`.
- `Reflect`, a trait that turns a type into a `Value` and back, implemented for the primitives,
  `String`, `Option<T>`, `Vec<T>` and `[T; N]`, and by three declarative macros for structs,
  newtypes and enums.
- `TypeRegistry`, which holds the registered types by name and reads, writes and builds them
  through `dyn Any`, and carries "type data" so other crates can attach their own tables to a type.
- With the `math` feature, `mid-math`'s vectors, quaternions and affine transforms.

There are no external dependencies. The crate builds on the workspace floor, Rust 1.75, and on
current stable.

**Status.** The core crate is complete and tested. The bridge to `mid-ecs` (reading and writing an
entity's components by name) is not built; see "Not built yet". Decision 7 in the roadmap is a
proposal, and the crate name and the macro approach were adopted on the maintainer's word, not
through an explicit confirmation.

```rust
use mid_reflect::{impl_reflect_struct, TypeRegistry, Value};

#[derive(Debug, PartialEq)]
struct Health { current: u32, max: u32 }
impl_reflect_struct!("game::Health" => Health { current: u32, max: u32 });

let mut registry = TypeRegistry::new();
registry.register::<Health>().unwrap();

let mut health = Health { current: 3, max: 10 };
registry.mutate("game::Health", &mut health, "current", Value::Int(7)).unwrap();
assert_eq!(health, Health { current: 7, max: 10 });
```

## Decisions

These apply to the whole crate. Each is a proposal the maintainer can overturn.

- **Registration by `macro_rules!`, no procedural macro.** A derive needs a separate proc-macro
  crate and normally `syn` and `quote`. The declarative macros cost a type name written as a
  string and the fields listed once, and the crate keeps zero dependencies.
- **No field offsets.** `core::mem::offset_of!` needs Rust 1.77 and the floor is 1.75. Nothing
  needs offsets yet.
- **Only unit-variant enums. Newtypes are transparent.** Enums with data in their variants are
  not supported yet. A newtype reads and writes as the type inside it, with no wrapper.
- **Strict struct reads.** A missing field and an unknown field are both errors, so a typo in a
  field name is reported and not ignored. Every error carries the path to the value (`a.b[2]`).
- **Stable names are chosen by the author.** The name is a string literal given to the macro, so
  it stays the same when the type moves or is renamed in Rust. Every field of a struct must be
  listed; a missing one is a compile error.
- **Version 0.1.0,** to match the other crates, although `docs/RUST_AND_CRATE_GUIDELINES.md` says
  0.0.1.

## Modules

### `lib.rs`

**What it does:** the crate documentation, the module list, the re-exports, and the `math` feature
gate. The macros are `#[macro_export]`, so they live at the crate root.

**Tests:** the doctest in the crate documentation (above) is the end-to-end smoke test; the
integration tests below cover the rest.

### `value.rs`

**What it does:** `Value` and its path syntax.

`Value` has eight variants: `Null`, `Bool`, `Int` (an `i64`), `UInt`, `Float` (an `f64`), `Str`,
`List`, and `Map`. The accessors (`as_bool`, `as_i64`, `as_u64`, `as_f64`, `as_str`, `as_list`,
`as_map`, `get`) return `Option`. `get_path` and `set_path` take a path such as
`translation.x` or `items[2].name`; `path_string` renders a path back to text.

**Decisions:**
- `UInt` holds only values above `i64::MAX`. `Value::from_u64` picks the variant, so every number
  has exactly one representation and equal numbers compare equal.
- `Map` is an ordered list of pairs, not a hash map. Order is part of the value: a struct reads
  back in declaration order, and JSON output is stable. Keys are unique in values that were parsed
  or produced by a reflected type.
- `set_path` replaces an existing value only. A typo in a field name is an error, not a new
  field. An empty path replaces the whole value.

**Tests:** `tests/value_json.rs` (24 tests) covers the accessors, the paths and their errors.
`tests/random.rs` checks that random values survive a round trip through text.

### `json.rs`

**What it does:** JSON text for `Value`. `to_json` is compact, `to_json_pretty` is indented, and
`from_json` and `from_json_with_depth` parse.

**Decisions:**
- Strict RFC 8259 on input: no comments, no trailing commas, no leading zeros, no `+1`, `.5` or
  `1.`.
- Nesting deeper than 128 levels is an error (`DEFAULT_MAX_DEPTH`). The limit can be changed with
  `from_json_with_depth`.
- A repeated key in an object is an error, not a silent last-wins.
- A number that does not fit in an `f64` is an error. It is not read as infinity (see "Fixes and
  Problems").
- JSON has no form for NaN or the infinities, so they are written as the strings `"NaN"`,
  `"Infinity"` and `"-Infinity"`, which the float types read back.
- Errors carry the byte offset where they were found.

**Tests:** `tests/value_json.rs`, and `tests/random.rs`, which round-trips random values, feeds
arbitrary text to the parser and checks it never panics, checks that every `f32` survives, and
round-trips random structs.

### `error.rs`

**What it does:** `ReflectError`, which pairs an `ErrorKind` with the path to where it happened,
and `PathSeg` (a field name or a list index). The kinds are `TypeMismatch`, `OutOfRange`,
`MissingField`, `UnknownField`, `UnknownVariant`, `BadLength` (a fixed-size array given the wrong
number of elements), `BadPath`, `UnknownType` and `WrongType` (a value of a different Rust type
than the one registered).

**Decisions:**
- The path is built as the error travels out of nested values (`at_field`, `at_index`), so the
  code that produces an error does not need to know where it sits.
- JSON syntax errors are a separate type, `JsonError`, which carries a byte offset in the text
  instead of a path.

**Tests:** `tests/reflect_types.rs`, in particular `errors_name_the_path_into_nested_values`.

### `ty.rs`

**What it does:** the description of types. `Ty` is a type as written at a use site (a primitive,
`Option`, a list, an array, or a named type). `TypeInfo` describes a named struct, newtype or enum
with its `Kind` and its `Field`s. `TypeInfo::to_value` turns a description into a `Value`, which is
what the registry's schema is made of.

**Decisions:** no offsets or sizes (see "Decisions" above).

**Tests:** `tests/registry.rs` checks the schema.

### `reflect.rs`

**What it does:** the `Reflect` trait (`ty`, `info`, `register_types`, `to_value`, `from_value`)
and its implementations for the primitives, `String`, `Option<T>`, `Vec<T>` and `[T; N]`. The
`private` module holds helper functions the macros call and is not part of the interface.

**Decisions:**
- Integers are range-checked on read. `Int`, `UInt` and a float with no fractional part are all
  accepted where an integer is expected. A value that does not fit is an `OutOfRange` error, not a
  wrap.
- An `f32` is stored as the `f64` nearest its shortest decimal text, so `0.1f32` becomes `0.1`
  and not `0.10000000149011612`. A finite value too large for an `f32` is an `OutOfRange` error.
- `Option<T>` is `Null` or the inner value.
- An array reads only a list of exactly its length (`BadLength` otherwise).

**Tests:** `tests/reflect_types.rs` (18 tests).

### `macros.rs`

**What it does:** three declarative macros.

- `impl_reflect_struct!("name" => Type { field: Type, ... })`
- `impl_reflect_newtype!("name" => Type(Inner))`
- `impl_reflect_enum!("name" => Type { A, B, C })`

Each implements `Reflect`, describes the type for the schema, and registers the type and every
named type its fields contain.

**Decisions:** see "Decisions" above (no proc-macro, unit variants only, transparent newtypes,
strict reads).

**Tests:** `tests/reflect_types.rs`.

### `registry.rs`

**What it does:** `TypeRegistry` and `Registration`.

`register::<T>()` adds `T` and every named type it contains. `get`, `get_of::<T>` and
`get_by_type_id` find a registration. `to_value`, `apply`, `construct` and `mutate` work by name
on a `&dyn Any`. `insert_data`, `data_of` and `data_by_name` attach and read "type data": a table
of any `Send + Sync` type keyed by its own type, which lets another crate hang its own behaviour
on a registered type (the ECS bridge will store a component accessor table this way). `schema`
returns every registered type as a `Value`.

**Decisions:**
- Registering a type twice changes nothing. Registering a different type under a name already in
  use is an error, and the first registration stays.
- Asking for the wrong Rust type, or an unknown name, is an error and never a panic.
- Registration order is kept, so a schema is stable between runs.

**Tests:** `tests/registry.rs` (13 tests).

### `math.rs`

**What it does:** with the `math` feature, `Reflect` for `Vec2`, `Vec3`, `Vec4`, `Quat`, `DVec2`,
`DVec3`, `DVec4`, `DQuat`, `Affine3` and `DAffine3`, each as a map of its components
(`{"x":1.0,"y":2.0,"z":3.0}`).

**Decisions:** vectors and quaternions are maps of named components, not lists.

**Tests:** `tests/math.rs` (6 tests, only with the feature).

## Not built yet

- **The ECS bridge.** `mid-ecs` has no way to enumerate an entity's archetype-core components. That
  API comes first. Then a `ReflectComponent` table (get, apply, insert, remove and has, by name)
  would be attached as type data, behind a feature of `mid-ecs`, and `LocalTransform`,
  `GlobalTransform`, `GlobalTransformLWC`, `ChildOf` and `Children` registered.
- Enums with data in their variants.
- A derive macro.
- Field offsets.

## Verification status

- 69 integration tests with the `math` feature and 63 without, plus 4 doctests. All pass on rustc
  1.97.1 and on rustc 1.75.0, with and without the feature.
- `clippy --no-deps --all-targets` and `rustfmt` are clean.
- 42 mutants over `json`, `value`, `reflect`, `registry`, `error`, `macros` and `ty` were all
  killed by the tests. The mutation run was done when the crate was written; it has not been rerun
  since.
- Not verified: behaviour on real CI before this delivery, and the cost of reading or writing large
  values. Nothing has been measured.

## CI and Workflows

- `.github/workflows/mid-reflect-test.yml` runs the tests on stable, with and without the `math`
  feature, and on Rust 1.75.0 for the same two configurations. Formatting and clippy run on stable
  and do not fail the job, as in the other crate workflows. A run starts on `workflow_dispatch`, or
  when a commit message or pull request title contains `--mid-reflect` or `--mid-all`.

## Fixes and Problems

### `json.rs`

- **A huge exponent became infinity and then a string.** The randomized test fed arbitrary text to
  the parser and compared a value with its own output. A number such as `1958e3538141297` parsed
  to an `f64` of infinity, which the writer then emitted as the string `"Infinity"`, so the text
  round trip changed the value's type. The parser now refuses a number that overflows `f64`. This
  was found by the test before any mutation testing started.
