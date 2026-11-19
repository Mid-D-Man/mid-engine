// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "registry.rs"
// ============================================================================
//! [`TypeRegistry`]: the named types a program has registered, with
//! type-erased read, write and construct operations on each.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use crate::error::{ErrorKind, ReflectError};
use crate::reflect::Reflect;
use crate::ty::TypeInfo;
use crate::value::Value;

/// A failure to register a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Two different Rust types were registered under one name.
    NameConflict(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::NameConflict(name) => {
                write!(f, "two different types are registered as `{name}`")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// What the registry holds for one type.
pub struct Registration {
    info: TypeInfo,
    type_id: TypeId,
    rust_name: &'static str,
    to_value: fn(&dyn Any) -> Option<Value>,
    apply: fn(&mut dyn Any, &Value) -> Option<Result<(), ReflectError>>,
    construct: fn(&Value) -> Result<Box<dyn Any + Send + Sync>, ReflectError>,
    data: Vec<(TypeId, Box<dyn Any + Send + Sync>)>,
}

impl Registration {
    fn of<T: Reflect>(info: TypeInfo) -> Self {
        Self {
            info,
            type_id: TypeId::of::<T>(),
            rust_name: std::any::type_name::<T>(),
            to_value: |any| any.downcast_ref::<T>().map(Reflect::to_value),
            apply: |any, value| {
                any.downcast_mut::<T>().map(|target| {
                    *target = T::from_value(value)?;
                    Ok(())
                })
            },
            construct: |value| {
                T::from_value(value).map(|v| Box::new(v) as Box<dyn Any + Send + Sync>)
            },
            data: Vec::new(),
        }
    }

    /// The type's description.
    pub fn info(&self) -> &TypeInfo {
        &self.info
    }

    /// The registered name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// The Rust `TypeId`.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// The compiler's name for the type. Not stable across compiler
    /// versions; for messages only.
    pub fn rust_name(&self) -> &'static str {
        self.rust_name
    }

    fn wrong_type(&self) -> ReflectError {
        ReflectError::new(ErrorKind::WrongType(self.info.name.clone()))
    }

    /// Reads `value` (which must be a value of this type) as a [`Value`].
    pub fn to_value(&self, value: &dyn Any) -> Result<Value, ReflectError> {
        (self.to_value)(value).ok_or_else(|| self.wrong_type())
    }

    /// Overwrites `target` (which must be a value of this type) from `value`.
    /// Nothing changes if `value` does not fit.
    pub fn apply(&self, target: &mut dyn Any, value: &Value) -> Result<(), ReflectError> {
        (self.apply)(target, value).unwrap_or_else(|| Err(self.wrong_type()))
    }

    /// Builds a new value of this type.
    pub fn construct(&self, value: &Value) -> Result<Box<dyn Any + Send + Sync>, ReflectError> {
        (self.construct)(value)
    }

    /// Sets the part of `target` at `path` (`translation.x`) to `new`.
    /// Nothing changes if the path is wrong or the result does not fit the type.
    pub fn mutate(&self, target: &mut dyn Any, path: &str, new: Value) -> Result<(), ReflectError> {
        let mut current = self.to_value(target)?;
        current.set_path(path, new)?;
        self.apply(target, &current)
    }

    /// Extra data attached to this registration by another crate, by its
    /// type.
    pub fn data<D: Any>(&self) -> Option<&D> {
        let wanted = TypeId::of::<D>();
        self.data
            .iter()
            .find(|(id, _)| *id == wanted)
            .and_then(|(_, d)| d.downcast_ref::<D>())
    }

    fn set_data<D: Any + Send + Sync>(&mut self, data: D) {
        let id = TypeId::of::<D>();
        match self.data.iter_mut().find(|(existing, _)| *existing == id) {
            Some(slot) => slot.1 = Box::new(data),
            None => self.data.push((id, Box::new(data))),
        }
    }
}

/// The named types registered so far, in registration order.
#[derive(Default)]
pub struct TypeRegistry {
    entries: Vec<Registration>,
    by_name: HashMap<String, usize>,
    by_type: HashMap<TypeId, usize>,
    conflicts: Vec<String>,
}

impl TypeRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `T` and every named type it contains. Registering a type
    /// that is already there changes nothing. Types with no name (the
    /// primitives) register nothing.
    pub fn register<T: Reflect>(&mut self) -> Result<(), RegistryError> {
        let before = self.conflicts.len();
        T::register_types(self);
        match self.conflicts.get(before) {
            Some(name) => Err(RegistryError::NameConflict(name.clone())),
            None => Ok(()),
        }
    }

    /// Adds `T` under its own name. Returns whether `T` is new, so a caller
    /// can go on to register what `T` contains. Used by the macros; a
    /// recursive type stops here the second time round.
    #[doc(hidden)]
    pub fn insert_named<T: Reflect>(&mut self) -> bool {
        let Some(info) = T::info() else {
            return false;
        };
        let type_id = TypeId::of::<T>();
        if self.by_type.contains_key(&type_id) {
            return false;
        }
        if self.by_name.contains_key(&info.name) {
            self.conflicts.push(info.name);
            return false;
        }
        let index = self.entries.len();
        self.by_name.insert(info.name.clone(), index);
        self.by_type.insert(type_id, index);
        self.entries.push(Registration::of::<T>(info));
        true
    }

    /// How many types are registered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The registration under `name`.
    pub fn get(&self, name: &str) -> Option<&Registration> {
        self.by_name.get(name).map(|i| &self.entries[*i])
    }

    /// The registration for the Rust type `T`.
    pub fn get_of<T: Reflect>(&self) -> Option<&Registration> {
        self.by_type
            .get(&TypeId::of::<T>())
            .map(|i| &self.entries[*i])
    }

    /// The registration for a `TypeId`.
    pub fn get_by_type_id(&self, id: TypeId) -> Option<&Registration> {
        self.by_type.get(&id).map(|i| &self.entries[*i])
    }

    /// Every registration, in the order they were added.
    pub fn iter(&self) -> impl Iterator<Item = &Registration> {
        self.entries.iter()
    }

    fn require(&self, name: &str) -> Result<&Registration, ReflectError> {
        self.get(name)
            .ok_or_else(|| ReflectError::new(ErrorKind::UnknownType(name.to_string())))
    }

    /// Reads a value of the type registered as `name`.
    pub fn to_value(&self, name: &str, value: &dyn Any) -> Result<Value, ReflectError> {
        self.require(name)?.to_value(value)
    }

    /// Overwrites a value of the type registered as `name`.
    pub fn apply(
        &self,
        name: &str,
        target: &mut dyn Any,
        value: &Value,
    ) -> Result<(), ReflectError> {
        self.require(name)?.apply(target, value)
    }

    /// Builds a value of the type registered as `name`.
    pub fn construct(
        &self,
        name: &str,
        value: &Value,
    ) -> Result<Box<dyn Any + Send + Sync>, ReflectError> {
        self.require(name)?.construct(value)
    }

    /// Sets the part at `path` of a value of the type registered as `name`.
    pub fn mutate(
        &self,
        name: &str,
        target: &mut dyn Any,
        path: &str,
        new: Value,
    ) -> Result<(), ReflectError> {
        self.require(name)?.mutate(target, path, new)
    }

    /// Attaches `data` to the registration of `T`, replacing any of the same
    /// type. Returns `false`, and drops `data`, if `T` is not registered.
    pub fn insert_data<T: Reflect, D: Any + Send + Sync>(&mut self, data: D) -> bool {
        match self.by_type.get(&TypeId::of::<T>()) {
            Some(index) => {
                self.entries[*index].set_data(data);
                true
            }
            None => false,
        }
    }

    /// The `D` attached to `T`'s registration.
    pub fn data_of<T: Reflect, D: Any>(&self) -> Option<&D> {
        self.get_of::<T>().and_then(Registration::data::<D>)
    }

    /// The `D` attached to the registration named `name`.
    pub fn data_by_name<D: Any>(&self, name: &str) -> Option<&D> {
        self.get(name).and_then(Registration::data::<D>)
    }

    /// Every registered type's description, as a map from name to
    /// [`TypeInfo::to_value`], in registration order.
    pub fn schema(&self) -> Value {
        Value::Map(
            self.entries
                .iter()
                .map(|r| (r.info.name.clone(), r.info.to_value()))
                .collect(),
        )
    }
}
