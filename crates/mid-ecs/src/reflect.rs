// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-reflect.md, section "reflect.rs"
// ============================================================================
//! The bridge between `mid-reflect` and the ECS (feature `reflect`).
//!
//! [`ReflectComponent`] is type data a [`TypeRegistry`] holds for a type that
//! is an Archetype Core component: it reads, replaces, inserts and removes
//! that component on an entity through [`Value`]s, without the caller naming
//! the type. The [`World`] methods at the bottom of the file use it by
//! registered name, which is what an inspector, a scene loader or a remote
//! protocol has.
//!
//! [`register_core_components`] registers the engine's own components
//! ([`LocalTransform`], [`GlobalTransform`], [`GlobalTransformLWC`],
//! [`ChildOf`], [`Children`]). The two hierarchy components go through the
//! hierarchy's own methods, so their rules (no cycles, `Children` kept by
//! `ChildOf`) hold for reflected edits too.
//!
//! ```
//! use mid_ecs::reflect::register_core_components;
//! use mid_ecs::{LocalTransform, World};
//! use mid_reflect::{TypeRegistry, Value};
//! use mid_math::Vec3;
//!
//! let mut registry = TypeRegistry::new();
//! register_core_components(&mut registry).unwrap();
//!
//! let mut world = World::new();
//! let e = world.spawn();
//! world.insert_static(e, LocalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0)));
//!
//! world
//!     .set_reflected_path(&registry, e, "mid_ecs::LocalTransform", "translation.y", Value::Float(9.0))
//!     .unwrap();
//! assert_eq!(world.get_static::<LocalTransform>(e).unwrap().translation.y, 9.0);
//! ```

use std::any::TypeId;
use std::fmt;

use mid_math::{Affine3, DAffine3, Quat, Vec3};
use mid_reflect::{
    impl_reflect_newtype, impl_reflect_struct, ErrorKind, Kind, Reflect, ReflectError,
    RegistryError, Ty, TypeInfo, TypeRegistry, Value,
};

use crate::hierarchy::{ChildOf, Children, HierarchyError};
use crate::transform::{GlobalTransform, GlobalTransformLWC, LocalTransform};
use crate::world::{Entity, World};

// ── Reflect for the engine's own types ──────────────────────────────────────

/// An entity is its packed `u64` handle (`Entity::as_ffi`). A handle read
/// back from a value is checked against the world's generations like any
/// other, so a stale or invented one reads as not alive.
impl Reflect for Entity {
    fn ty() -> Ty {
        Ty::Named("mid_ecs::Entity".to_string())
    }

    fn info() -> Option<TypeInfo> {
        Some(TypeInfo {
            name: "mid_ecs::Entity".to_string(),
            kind: Kind::Newtype(<u64 as Reflect>::ty()),
        })
    }

    fn register_types(registry: &mut TypeRegistry) {
        registry.insert_named::<Self>();
    }

    fn to_value(&self) -> Value {
        Value::from_u64(self.as_ffi())
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        <u64 as Reflect>::from_value(value).map(Entity::from_ffi)
    }
}

impl_reflect_struct!("mid_ecs::LocalTransform" => LocalTransform {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
});
impl_reflect_newtype!("mid_ecs::GlobalTransform" => GlobalTransform(Affine3));
impl_reflect_newtype!("mid_ecs::GlobalTransformLWC" => GlobalTransformLWC(DAffine3));
impl_reflect_newtype!("mid_ecs::ChildOf" => ChildOf(Entity));

/// `Children` can be read but never built from a value: only the hierarchy
/// hooks may write it.
impl Reflect for Children {
    fn ty() -> Ty {
        Ty::Named("mid_ecs::Children".to_string())
    }

    fn info() -> Option<TypeInfo> {
        Some(TypeInfo {
            name: "mid_ecs::Children".to_string(),
            kind: Kind::Newtype(<Vec<Entity> as Reflect>::ty()),
        })
    }

    fn register_types(registry: &mut TypeRegistry) {
        if registry.insert_named::<Self>() {
            <Vec<Entity> as Reflect>::register_types(registry);
        }
    }

    fn to_value(&self) -> Value {
        Value::List(self.as_slice().iter().map(Entity::to_value).collect())
    }

    fn from_value(_value: &Value) -> Result<Self, ReflectError> {
        Err(ReflectError::new(ErrorKind::Unsupported(
            "Children is maintained by ChildOf and cannot be built from a value".to_string(),
        )))
    }
}

// ── errors ──────────────────────────────────────────────────────────────────

/// Why a reflected component access failed. Nothing changed.
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentReflectError {
    /// The entity is not alive.
    NotAlive(Entity),
    /// No type is registered under this name.
    UnknownComponent(String),
    /// The type is registered, but not as a component.
    NotAComponent(String),
    /// The entity has no such component.
    Missing {
        /// The entity that was asked.
        entity: Entity,
        /// The registered component name.
        component: String,
    },
    /// The component is maintained by something else and cannot be changed
    /// this way.
    Refused {
        /// The registered component name.
        component: String,
        /// Why.
        reason: &'static str,
    },
    /// The value does not fit the type, or the path is wrong.
    Value(ReflectError),
    /// The hierarchy refused the change (a cycle, a dead entity).
    Hierarchy(HierarchyError),
}

impl fmt::Display for ComponentReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentReflectError::NotAlive(e) => write!(f, "{e} is not alive"),
            ComponentReflectError::UnknownComponent(name) => {
                write!(f, "no type registered as `{name}`")
            }
            ComponentReflectError::NotAComponent(name) => {
                write!(f, "`{name}` is registered, but not as a component")
            }
            ComponentReflectError::Missing { entity, component } => {
                write!(f, "{entity} has no `{component}`")
            }
            ComponentReflectError::Refused { component, reason } => {
                write!(f, "`{component}` cannot be changed this way: {reason}")
            }
            ComponentReflectError::Value(e) => write!(f, "{e}"),
            ComponentReflectError::Hierarchy(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ComponentReflectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ComponentReflectError::Value(e) => Some(e),
            ComponentReflectError::Hierarchy(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ReflectError> for ComponentReflectError {
    fn from(e: ReflectError) -> Self {
        ComponentReflectError::Value(e)
    }
}

impl From<HierarchyError> for ComponentReflectError {
    fn from(e: HierarchyError) -> Self {
        ComponentReflectError::Hierarchy(e)
    }
}

// ── ReflectComponent ────────────────────────────────────────────────────────

/// Whether an entity holds the component.
pub type ContainsFn = fn(&World, Entity) -> bool;
/// The component's current value.
pub type GetFn = fn(&World, Entity) -> Option<Value>;
/// Replaces the value of a component that is present.
pub type ApplyFn = fn(&mut World, Entity, &Value) -> Result<(), ComponentReflectError>;
/// Adds the component, or replaces it if present.
pub type InsertFn = fn(&mut World, Entity, &Value) -> Result<(), ComponentReflectError>;
/// Removes the component; whether it was there.
pub type RemoveFn = fn(&mut World, Entity) -> Result<bool, ComponentReflectError>;

/// Type data for a component: reads and writes it on an entity as [`Value`]s.
/// Attached to a type's registration by [`register_component`].
///
/// Only Archetype Core components are covered (`World::insert_static` and
/// friends). Replacing a present value in place goes through
/// `World::get_static_mut`, so it is stamped as changed, but fires no
/// component hooks; adding and removing go through the usual structural
/// path, so hooks fire.
#[derive(Clone)]
pub struct ReflectComponent {
    name: String,
    contains: ContainsFn,
    get: GetFn,
    apply: ApplyFn,
    insert: InsertFn,
    remove: RemoveFn,
}

impl fmt::Debug for ReflectComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReflectComponent")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

fn component_name<T: Reflect>() -> String {
    T::info()
        .map(|info| info.name)
        .unwrap_or_else(|| std::any::type_name::<T>().to_string())
}

fn missing<T: Reflect>(entity: Entity) -> ComponentReflectError {
    ComponentReflectError::Missing {
        entity,
        component: component_name::<T>(),
    }
}

fn apply_typed<T: Reflect>(
    world: &mut World,
    entity: Entity,
    value: &Value,
) -> Result<(), ComponentReflectError> {
    if !world.is_alive(entity) {
        return Err(ComponentReflectError::NotAlive(entity));
    }
    if !world.has_static::<T>(entity) {
        return Err(missing::<T>(entity));
    }
    let new = T::from_value(value)?;
    if let Some(slot) = world.get_static_mut::<T>(entity) {
        *slot = new;
    }
    Ok(())
}

fn insert_typed<T: Reflect>(
    world: &mut World,
    entity: Entity,
    value: &Value,
) -> Result<(), ComponentReflectError> {
    if !world.is_alive(entity) {
        return Err(ComponentReflectError::NotAlive(entity));
    }
    let new = T::from_value(value)?;
    if world.has_static::<T>(entity) {
        if let Some(slot) = world.get_static_mut::<T>(entity) {
            *slot = new;
        }
    } else {
        world.insert_static(entity, new);
    }
    Ok(())
}

impl ReflectComponent {
    /// The type data for `T`, a plain Archetype Core component.
    pub fn of<T: Reflect>() -> Self {
        Self {
            name: component_name::<T>(),
            contains: |world, entity| world.has_static::<T>(entity),
            get: |world, entity| world.get_static::<T>(entity).map(T::to_value),
            apply: apply_typed::<T>,
            insert: insert_typed::<T>,
            remove: |world, entity| Ok(world.remove_static::<T>(entity).is_some()),
        }
    }

    /// Type data with hand-written operations, for a component whose changes
    /// must go through something else (`ChildOf` goes through
    /// `World::set_parent`). `name` is the registered name, used in errors.
    pub fn new(
        name: impl Into<String>,
        contains: ContainsFn,
        get: GetFn,
        apply: ApplyFn,
        insert: InsertFn,
        remove: RemoveFn,
    ) -> Self {
        Self {
            name: name.into(),
            contains,
            get,
            apply,
            insert,
            remove,
        }
    }

    /// The registered name of the component.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether `entity` is alive and holds the component.
    pub fn contains(&self, world: &World, entity: Entity) -> bool {
        world.is_alive(entity) && (self.contains)(world, entity)
    }

    /// The component's value on `entity`; `None` if it is dead or lacks it.
    pub fn get(&self, world: &World, entity: Entity) -> Option<Value> {
        if !world.is_alive(entity) {
            return None;
        }
        (self.get)(world, entity)
    }

    /// Replaces the component's value on `entity`.
    ///
    /// # Errors
    /// [`ComponentReflectError`] if `entity` is dead or lacks the component,
    /// or `value` does not fit. Nothing changes then.
    pub fn apply(
        &self,
        world: &mut World,
        entity: Entity,
        value: &Value,
    ) -> Result<(), ComponentReflectError> {
        (self.apply)(world, entity, value)
    }

    /// Adds the component to `entity`, or replaces it if it is there.
    ///
    /// # Errors
    /// [`ComponentReflectError`] if `entity` is dead or `value` does not fit.
    /// Nothing changes then.
    pub fn insert(
        &self,
        world: &mut World,
        entity: Entity,
        value: &Value,
    ) -> Result<(), ComponentReflectError> {
        (self.insert)(world, entity, value)
    }

    /// Removes the component from `entity`. Whether it was there.
    pub fn remove(&self, world: &mut World, entity: Entity) -> Result<bool, ComponentReflectError> {
        (self.remove)(world, entity)
    }

    /// Sets the part of the component at `path` (`translation.y`) to `new`.
    ///
    /// # Errors
    /// [`ComponentReflectError`] if `entity` is dead or lacks the component,
    /// the path is wrong, or the result does not fit. Nothing changes then.
    pub fn set_path(
        &self,
        world: &mut World,
        entity: Entity,
        path: &str,
        new: Value,
    ) -> Result<(), ComponentReflectError> {
        if !world.is_alive(entity) {
            return Err(ComponentReflectError::NotAlive(entity));
        }
        let mut current =
            self.get(world, entity)
                .ok_or_else(|| ComponentReflectError::Missing {
                    entity,
                    component: self.name.clone(),
                })?;
        current.set_path(path, new)?;
        self.apply(world, entity, &current)
    }
}

// ── registration ────────────────────────────────────────────────────────────

/// Registers `T` and its contents, and attaches [`ReflectComponent::of`].
/// Registering the same type again changes nothing.
pub fn register_component<T: Reflect>(registry: &mut TypeRegistry) -> Result<(), RegistryError> {
    register_component_with::<T>(registry, ReflectComponent::of::<T>())
}

/// [`register_component`] with hand-written type data.
pub fn register_component_with<T: Reflect>(
    registry: &mut TypeRegistry,
    data: ReflectComponent,
) -> Result<(), RegistryError> {
    registry.register::<T>()?;
    registry.insert_data::<T, _>(data);
    Ok(())
}

fn child_of_data() -> ReflectComponent {
    fn set_parent_from(
        world: &mut World,
        entity: Entity,
        value: &Value,
    ) -> Result<(), ComponentReflectError> {
        let ChildOf(parent) = ChildOf::from_value(value)?;
        world.set_parent(entity, parent)?;
        Ok(())
    }

    ReflectComponent::new(
        component_name::<ChildOf>(),
        |world, entity| world.has_static::<ChildOf>(entity),
        |world, entity| world.get_static::<ChildOf>(entity).map(ChildOf::to_value),
        |world, entity, value| {
            if !world.is_alive(entity) {
                return Err(ComponentReflectError::NotAlive(entity));
            }
            if !world.has_static::<ChildOf>(entity) {
                return Err(missing::<ChildOf>(entity));
            }
            set_parent_from(world, entity, value)
        },
        |world, entity, value| {
            if !world.is_alive(entity) {
                return Err(ComponentReflectError::NotAlive(entity));
            }
            set_parent_from(world, entity, value)
        },
        |world, entity| Ok(world.remove_parent(entity)),
    )
}

fn children_data() -> ReflectComponent {
    fn refused() -> ComponentReflectError {
        ComponentReflectError::Refused {
            component: component_name::<Children>(),
            reason: "it is maintained by ChildOf",
        }
    }

    ReflectComponent::new(
        component_name::<Children>(),
        |world, entity| world.has_static::<Children>(entity),
        |world, entity| world.get_static::<Children>(entity).map(Children::to_value),
        |_, _, _| Err(refused()),
        |_, _, _| Err(refused()),
        |_, _| Err(refused()),
    )
}

/// Registers the engine's own components: [`LocalTransform`],
/// [`GlobalTransform`], [`GlobalTransformLWC`], [`ChildOf`] and
/// [`Children`], with the math types and [`Entity`] they contain.
pub fn register_core_components(registry: &mut TypeRegistry) -> Result<(), RegistryError> {
    register_component::<LocalTransform>(registry)?;
    register_component::<GlobalTransform>(registry)?;
    register_component::<GlobalTransformLWC>(registry)?;
    register_component_with::<ChildOf>(registry, child_of_data())?;
    register_component_with::<Children>(registry, children_data())?;
    Ok(())
}

// ── World access by registered name ─────────────────────────────────────────

/// An entity's reflected components: what [`World::reflect_entity`] returns.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityReflection {
    /// The entity described.
    pub entity: Entity,
    /// Each registered component the entity holds, as its registered name and
    /// current value, in the registry's registration order.
    pub components: Vec<(String, Value)>,
    /// How many Archetype Core components the entity holds that are not
    /// registered as components, so are not in `components`.
    pub unreflected: usize,
}

fn lookup<'r>(
    registry: &'r TypeRegistry,
    name: &str,
) -> Result<&'r ReflectComponent, ComponentReflectError> {
    let registration = registry
        .get(name)
        .ok_or_else(|| ComponentReflectError::UnknownComponent(name.to_string()))?;
    registration
        .data::<ReflectComponent>()
        .ok_or_else(|| ComponentReflectError::NotAComponent(name.to_string()))
}

impl World {
    /// Every registered component `entity` holds, with its value. `None` if
    /// the entity is not alive. Sparse Shell components are not listed.
    pub fn reflect_entity(
        &self,
        registry: &TypeRegistry,
        entity: Entity,
    ) -> Option<EntityReflection> {
        if !self.is_alive(entity) {
            return None;
        }
        let present: Vec<TypeId> = self.static_component_types(entity).collect();
        let mut components = Vec::new();
        for registration in registry.iter() {
            if !present.contains(&registration.type_id()) {
                continue;
            }
            let Some(data) = registration.data::<ReflectComponent>() else {
                continue;
            };
            if let Some(value) = data.get(self, entity) {
                components.push((registration.name().to_string(), value));
            }
        }
        let unreflected = present.len().saturating_sub(components.len());
        Some(EntityReflection {
            entity,
            components,
            unreflected,
        })
    }

    /// The value of `entity`'s component registered as `name`.
    ///
    /// # Errors
    /// [`ComponentReflectError`] if the name is not a registered component,
    /// the entity is dead, or it lacks the component.
    pub fn reflect_component(
        &self,
        registry: &TypeRegistry,
        entity: Entity,
        name: &str,
    ) -> Result<Value, ComponentReflectError> {
        let data = lookup(registry, name)?;
        if !self.is_alive(entity) {
            return Err(ComponentReflectError::NotAlive(entity));
        }
        data.get(self, entity)
            .ok_or_else(|| ComponentReflectError::Missing {
                entity,
                component: name.to_string(),
            })
    }

    /// Whether `entity` is alive and holds the component registered as
    /// `name`.
    ///
    /// # Errors
    /// [`ComponentReflectError`] if the name is not a registered component.
    pub fn has_reflected(
        &self,
        registry: &TypeRegistry,
        entity: Entity,
        name: &str,
    ) -> Result<bool, ComponentReflectError> {
        Ok(lookup(registry, name)?.contains(self, entity))
    }

    /// Replaces the value of `entity`'s component registered as `name`. See
    /// [`ReflectComponent::apply`].
    pub fn apply_reflected(
        &mut self,
        registry: &TypeRegistry,
        entity: Entity,
        name: &str,
        value: &Value,
    ) -> Result<(), ComponentReflectError> {
        lookup(registry, name)?.apply(self, entity, value)
    }

    /// Adds the component registered as `name` to `entity`, or replaces it.
    /// See [`ReflectComponent::insert`].
    pub fn insert_reflected(
        &mut self,
        registry: &TypeRegistry,
        entity: Entity,
        name: &str,
        value: &Value,
    ) -> Result<(), ComponentReflectError> {
        lookup(registry, name)?.insert(self, entity, value)
    }

    /// Removes the component registered as `name` from `entity`. Whether it
    /// was there.
    pub fn remove_reflected(
        &mut self,
        registry: &TypeRegistry,
        entity: Entity,
        name: &str,
    ) -> Result<bool, ComponentReflectError> {
        lookup(registry, name)?.remove(self, entity)
    }

    /// Sets the part of `entity`'s component registered as `name` at `path`
    /// to `new`. See [`ReflectComponent::set_path`].
    pub fn set_reflected_path(
        &mut self,
        registry: &TypeRegistry,
        entity: Entity,
        name: &str,
        path: &str,
        new: Value,
    ) -> Result<(), ComponentReflectError> {
        lookup(registry, name)?.set_path(self, entity, path, new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::ChangeTracker;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Health {
        current: u32,
        max: u32,
    }
    impl_reflect_struct!("test::Health" => Health { current: u32, max: u32 });

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Tag(u8);
    impl_reflect_newtype!("test::Tag" => Tag(u8));

    /// Registered as a type, never as a component.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Plain {
        n: u32,
    }
    impl_reflect_struct!("test::Plain" => Plain { n: u32 });

    /// Held by entities but never registered at all.
    struct Hidden(u8);

    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::new();
        register_core_components(&mut registry).unwrap();
        register_component::<Health>(&mut registry).unwrap();
        register_component::<Tag>(&mut registry).unwrap();
        registry.register::<Plain>().unwrap();
        registry
    }

    fn health(current: u32) -> Health {
        Health { current, max: 10 }
    }

    // ── Reflect impls ──────────────────────────────────────────────────

    #[test]
    fn entity_round_trips_through_a_value() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(Entity::from_value(&e.to_value()).unwrap(), e);
        let via_json = Value::from_json(&e.to_value().to_json()).unwrap();
        assert_eq!(Entity::from_value(&via_json).unwrap(), e);
    }

    #[test]
    fn entity_from_a_non_number_is_a_value_error() {
        assert!(Entity::from_value(&Value::Str("x".into())).is_err());
    }

    #[test]
    fn a_stale_entity_value_reads_as_not_alive() {
        let mut w = World::new();
        let e = w.spawn();
        let value = e.to_value();
        w.despawn(e);
        let _other = w.spawn();
        assert!(!w.is_alive(Entity::from_value(&value).unwrap()));
    }

    #[test]
    fn local_transform_is_a_struct_of_vectors() {
        let t = LocalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let value = t.to_value();
        assert_eq!(value.get_path("translation.z").unwrap(), &Value::Float(3.0));
        assert_eq!(LocalTransform::from_value(&value).unwrap(), t);
    }

    #[test]
    fn children_reads_as_a_list_and_cannot_be_built() {
        let mut w = World::new();
        let parent = w.spawn();
        let a = w.spawn_child(parent).unwrap();
        let b = w.spawn_child(parent).unwrap();
        let children = w.get_static::<Children>(parent).unwrap();
        assert_eq!(
            children.to_value(),
            Value::List(vec![a.to_value(), b.to_value()])
        );
        let err = Children::from_value(&Value::List(vec![])).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::Unsupported(_)));
    }

    // ── registration ───────────────────────────────────────────────────

    #[test]
    fn registering_a_component_attaches_type_data_and_is_idempotent() {
        let mut registry = TypeRegistry::new();
        register_component::<Health>(&mut registry).unwrap();
        register_component::<Health>(&mut registry).unwrap();
        let data = registry
            .data_by_name::<ReflectComponent>("test::Health")
            .unwrap();
        assert_eq!(data.name(), "test::Health");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn core_components_register_with_the_types_they_contain() {
        let registry = registry();
        for name in [
            "mid_ecs::LocalTransform",
            "mid_ecs::GlobalTransform",
            "mid_ecs::GlobalTransformLWC",
            "mid_ecs::ChildOf",
            "mid_ecs::Children",
        ] {
            assert!(
                registry.data_by_name::<ReflectComponent>(name).is_some(),
                "{name}"
            );
        }
        assert!(registry.get("mid_ecs::Entity").is_some());
        assert!(registry.get("mid_math::Vec3").is_some());
        assert!(registry
            .data_by_name::<ReflectComponent>("mid_ecs::Entity")
            .is_none());
    }

    #[test]
    fn a_name_conflict_is_an_error() {
        struct Other;
        impl Reflect for Other {
            fn ty() -> Ty {
                Ty::Named("test::Health".to_string())
            }
            fn info() -> Option<TypeInfo> {
                Some(TypeInfo {
                    name: "test::Health".to_string(),
                    kind: Kind::Newtype(<u8 as Reflect>::ty()),
                })
            }
            fn register_types(registry: &mut TypeRegistry) {
                registry.insert_named::<Self>();
            }
            fn to_value(&self) -> Value {
                Value::Null
            }
            fn from_value(_: &Value) -> Result<Self, ReflectError> {
                Ok(Other)
            }
        }
        let mut registry = TypeRegistry::new();
        register_component::<Health>(&mut registry).unwrap();
        assert!(register_component::<Other>(&mut registry).is_err());
    }

    // ── reading ────────────────────────────────────────────────────────

    #[test]
    fn reflect_component_reads_the_value() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, health(3));
        let value = w.reflect_component(&registry, e, "test::Health").unwrap();
        assert_eq!(value.to_json(), r#"{"current":3,"max":10}"#);
    }

    #[test]
    fn reflect_component_errors_say_what_is_wrong() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        assert!(matches!(
            w.reflect_component(&registry, e, "test::Nope"),
            Err(ComponentReflectError::UnknownComponent(_))
        ));
        assert!(matches!(
            w.reflect_component(&registry, e, "test::Plain"),
            Err(ComponentReflectError::NotAComponent(_))
        ));
        assert!(matches!(
            w.reflect_component(&registry, e, "test::Health"),
            Err(ComponentReflectError::Missing { .. })
        ));
        w.despawn(e);
        assert_eq!(
            w.reflect_component(&registry, e, "test::Health"),
            Err(ComponentReflectError::NotAlive(e))
        );
    }

    #[test]
    fn has_reflected_tracks_the_component() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(w.has_reflected(&registry, e, "test::Health"), Ok(false));
        w.insert_static(e, health(1));
        assert_eq!(w.has_reflected(&registry, e, "test::Health"), Ok(true));
        assert!(w.has_reflected(&registry, e, "test::Nope").is_err());
        w.despawn(e);
        assert_eq!(w.has_reflected(&registry, e, "test::Health"), Ok(false));
    }

    #[test]
    fn reflect_entity_lists_registered_components_in_registration_order() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Tag(7));
        w.insert_static(e, health(2));
        w.insert_static(e, Hidden(1));
        let r = w.reflect_entity(&registry, e).unwrap();
        let names: Vec<&str> = r.components.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["test::Health", "test::Tag"]);
        assert_eq!(r.unreflected, 1);
        assert_eq!(r.entity, e);
    }

    #[test]
    fn reflect_entity_counts_a_registered_type_without_type_data_as_unreflected() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Plain { n: 1 });
        let r = w.reflect_entity(&registry, e).unwrap();
        assert!(r.components.is_empty());
        assert_eq!(r.unreflected, 1);
    }

    #[test]
    fn reflect_entity_is_none_for_a_dead_entity_and_empty_for_a_bare_one() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        let r = w.reflect_entity(&registry, e).unwrap();
        assert!(r.components.is_empty());
        assert_eq!(r.unreflected, 0);
        w.despawn(e);
        assert!(w.reflect_entity(&registry, e).is_none());
    }

    // ── writing ────────────────────────────────────────────────────────

    #[test]
    fn apply_replaces_the_value_and_stamps_it_changed() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, health(3));
        let mut tracker = ChangeTracker::new();
        w.increment_change_tick();
        tracker.update(&w);
        assert_eq!(w.query_changed::<Health>(&tracker).count(), 0);

        w.increment_change_tick();
        let new = Value::from_json(r#"{"current":9,"max":10}"#).unwrap();
        w.apply_reflected(&registry, e, "test::Health", &new)
            .unwrap();
        assert_eq!(*w.get_static::<Health>(e).unwrap(), health(9));
        assert_eq!(w.query_changed::<Health>(&tracker).count(), 1);
    }

    #[test]
    fn apply_with_a_bad_value_changes_nothing_and_stamps_nothing() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, health(3));
        let mut tracker = ChangeTracker::new();
        w.increment_change_tick();
        tracker.update(&w);
        w.increment_change_tick();

        let bad = Value::from_json(r#"{"current":"x","max":10}"#).unwrap();
        let err = w
            .apply_reflected(&registry, e, "test::Health", &bad)
            .unwrap_err();
        assert!(matches!(err, ComponentReflectError::Value(_)));
        assert_eq!(*w.get_static::<Health>(e).unwrap(), health(3));
        assert_eq!(w.query_changed::<Health>(&tracker).count(), 0);
    }

    #[test]
    fn apply_needs_the_component_to_be_present_and_the_entity_alive() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        let value = health(1).to_value();
        assert!(matches!(
            w.apply_reflected(&registry, e, "test::Health", &value),
            Err(ComponentReflectError::Missing { .. })
        ));
        assert!(!w.has_static::<Health>(e));
        w.despawn(e);
        assert_eq!(
            w.apply_reflected(&registry, e, "test::Health", &value),
            Err(ComponentReflectError::NotAlive(e))
        );
    }

    #[test]
    fn insert_adds_a_missing_component_and_replaces_a_present_one() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_reflected(&registry, e, "test::Health", &health(4).to_value())
            .unwrap();
        assert_eq!(*w.get_static::<Health>(e).unwrap(), health(4));
        w.insert_reflected(&registry, e, "test::Health", &health(8).to_value())
            .unwrap();
        assert_eq!(*w.get_static::<Health>(e).unwrap(), health(8));
        assert_eq!(w.static_component_types(e).count(), 1);
    }

    #[test]
    fn insert_with_a_bad_value_or_a_dead_entity_changes_nothing() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        assert!(w
            .insert_reflected(&registry, e, "test::Health", &Value::Null)
            .is_err());
        assert!(!w.has_static::<Health>(e));
        w.despawn(e);
        assert_eq!(
            w.insert_reflected(&registry, e, "test::Health", &health(1).to_value()),
            Err(ComponentReflectError::NotAlive(e))
        );
    }

    #[test]
    fn insert_fires_the_add_hook_but_apply_does_not() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ADDED: AtomicUsize = AtomicUsize::new(0);
        let registry = registry();
        let mut w = World::new();
        w.register_component_hooks::<Health>().on_add(|_, _| {
            ADDED.fetch_add(1, Ordering::SeqCst);
        });
        let e = w.spawn();
        w.insert_reflected(&registry, e, "test::Health", &health(1).to_value())
            .unwrap();
        assert_eq!(ADDED.load(Ordering::SeqCst), 1);
        w.apply_reflected(&registry, e, "test::Health", &health(2).to_value())
            .unwrap();
        assert_eq!(ADDED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn remove_reports_whether_it_was_there() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, health(1));
        assert_eq!(w.remove_reflected(&registry, e, "test::Health"), Ok(true));
        assert!(!w.has_static::<Health>(e));
        assert_eq!(w.remove_reflected(&registry, e, "test::Health"), Ok(false));
        assert!(w.remove_reflected(&registry, e, "test::Nope").is_err());
    }

    #[test]
    fn set_path_changes_one_field_only() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, health(3));
        w.set_reflected_path(&registry, e, "test::Health", "current", Value::Int(7))
            .unwrap();
        assert_eq!(*w.get_static::<Health>(e).unwrap(), health(7));
    }

    #[test]
    fn set_path_with_a_wrong_path_or_type_changes_nothing() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, health(3));
        assert!(w
            .set_reflected_path(&registry, e, "test::Health", "nope", Value::Int(7))
            .is_err());
        assert!(w
            .set_reflected_path(
                &registry,
                e,
                "test::Health",
                "current",
                Value::Str("x".into())
            )
            .is_err());
        assert!(w
            .set_reflected_path(&registry, e, "test::Health", "current", Value::Int(-1))
            .is_err());
        assert_eq!(*w.get_static::<Health>(e).unwrap(), health(3));
    }

    #[test]
    fn set_path_on_a_missing_component_is_missing() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        assert!(matches!(
            w.set_reflected_path(&registry, e, "test::Health", "current", Value::Int(7)),
            Err(ComponentReflectError::Missing { .. })
        ));
    }

    // ── the engine's own components ────────────────────────────────────

    #[test]
    fn local_transform_is_edited_through_a_nested_path() {
        let registry = registry();
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, LocalTransform::IDENTITY);
        w.set_reflected_path(
            &registry,
            e,
            "mid_ecs::LocalTransform",
            "scale.x",
            Value::Float(2.5),
        )
        .unwrap();
        assert_eq!(w.get_static::<LocalTransform>(e).unwrap().scale.x, 2.5);
        assert_eq!(w.get_static::<LocalTransform>(e).unwrap().scale.y, 1.0);
    }

    #[test]
    fn a_transform_inserted_from_json_feeds_propagation() {
        let registry = registry();
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn_child(parent).unwrap();
        let json = r#"{"translation":{"x":1.0,"y":0.0,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"scale":{"x":1.0,"y":1.0,"z":1.0}}"#;
        let value = Value::from_json(json).unwrap();
        w.insert_reflected(&registry, parent, "mid_ecs::LocalTransform", &value)
            .unwrap();
        w.insert_reflected(&registry, child, "mid_ecs::LocalTransform", &value)
            .unwrap();
        w.insert_static(parent, GlobalTransform::IDENTITY);
        w.insert_static(child, GlobalTransform::IDENTITY);
        w.propagate_transforms();
        let global = w.get_static::<GlobalTransform>(child).unwrap();
        assert_eq!(global.0.translation.x, 2.0);
    }

    #[test]
    fn child_of_insert_goes_through_the_hierarchy() {
        let registry = registry();
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        w.insert_reflected(&registry, child, "mid_ecs::ChildOf", &parent.to_value())
            .unwrap();
        assert_eq!(w.parent(child), Some(parent));
        assert_eq!(w.children(parent), &[child]);
    }

    #[test]
    fn child_of_apply_moves_the_child_between_parents() {
        let registry = registry();
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let child = w.spawn_child(a).unwrap();
        w.apply_reflected(&registry, child, "mid_ecs::ChildOf", &b.to_value())
            .unwrap();
        assert_eq!(w.parent(child), Some(b));
        assert!(w.children(a).is_empty());
        assert_eq!(w.children(b), &[child]);
    }

    #[test]
    fn child_of_that_would_make_a_cycle_is_refused_and_nothing_changes() {
        let registry = registry();
        let mut w = World::new();
        let top = w.spawn();
        let below = w.spawn_child(top).unwrap();
        let err = w
            .apply_reflected(&registry, top, "mid_ecs::ChildOf", &below.to_value())
            .unwrap_err();
        assert!(matches!(err, ComponentReflectError::Missing { .. }));
        let err = w
            .insert_reflected(&registry, top, "mid_ecs::ChildOf", &below.to_value())
            .unwrap_err();
        assert!(matches!(
            err,
            ComponentReflectError::Hierarchy(HierarchyError::Cycle { .. })
        ));
        assert_eq!(w.parent(top), None);
        assert_eq!(w.parent(below), Some(top));
    }

    #[test]
    fn child_of_apply_that_would_make_a_cycle_is_refused() {
        let registry = registry();
        let mut w = World::new();
        let root = w.spawn();
        let mid = w.spawn_child(root).unwrap();
        let leaf = w.spawn_child(mid).unwrap();
        let err = w
            .apply_reflected(&registry, mid, "mid_ecs::ChildOf", &leaf.to_value())
            .unwrap_err();
        assert!(matches!(
            err,
            ComponentReflectError::Hierarchy(HierarchyError::Cycle { .. })
        ));
        assert_eq!(w.parent(mid), Some(root));
        assert_eq!(w.children(root), &[mid]);
        assert_eq!(w.children(mid), &[leaf]);
    }

    #[test]
    fn child_of_to_a_dead_parent_is_refused() {
        let registry = registry();
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn();
        w.despawn(parent);
        let err = w
            .insert_reflected(&registry, child, "mid_ecs::ChildOf", &parent.to_value())
            .unwrap_err();
        assert!(matches!(
            err,
            ComponentReflectError::Hierarchy(HierarchyError::NotAlive(_))
        ));
        assert_eq!(w.parent(child), None);
    }

    #[test]
    fn child_of_remove_detaches_and_the_parents_children_follow() {
        let registry = registry();
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn_child(parent).unwrap();
        assert_eq!(
            w.remove_reflected(&registry, child, "mid_ecs::ChildOf"),
            Ok(true)
        );
        assert_eq!(w.parent(child), None);
        assert!(!w.has_static::<Children>(parent));
        assert_eq!(
            w.remove_reflected(&registry, child, "mid_ecs::ChildOf"),
            Ok(false)
        );
    }

    #[test]
    fn children_can_be_read_but_not_written() {
        let registry = registry();
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn_child(parent).unwrap();
        let value = w
            .reflect_component(&registry, parent, "mid_ecs::Children")
            .unwrap();
        assert_eq!(value, Value::List(vec![child.to_value()]));
        let list = Value::List(vec![]);
        for result in [
            w.apply_reflected(&registry, parent, "mid_ecs::Children", &list),
            w.insert_reflected(&registry, parent, "mid_ecs::Children", &list),
        ] {
            assert!(matches!(result, Err(ComponentReflectError::Refused { .. })));
        }
        assert!(matches!(
            w.remove_reflected(&registry, parent, "mid_ecs::Children"),
            Err(ComponentReflectError::Refused { .. })
        ));
        assert_eq!(w.children(parent), &[child]);
    }

    #[test]
    fn reflect_entity_shows_the_hierarchy_components() {
        let registry = registry();
        let mut w = World::new();
        let parent = w.spawn();
        let child = w.spawn_child(parent).unwrap();
        let r = w.reflect_entity(&registry, parent).unwrap();
        assert_eq!(r.components.len(), 1);
        assert_eq!(r.components[0].0, "mid_ecs::Children");
        let r = w.reflect_entity(&registry, child).unwrap();
        assert_eq!(r.components[0].0, "mid_ecs::ChildOf");
        assert_eq!(r.components[0].1, parent.to_value());
    }

    // ── errors ─────────────────────────────────────────────────────────

    #[test]
    fn errors_display_and_chain() {
        use std::error::Error;
        let mut w = World::new();
        let e = w.spawn();
        let value = ComponentReflectError::from(ReflectError::new(ErrorKind::BadPath("x".into())));
        assert!(value.source().is_some());
        let hierarchy = ComponentReflectError::from(HierarchyError::SelfParent(e));
        assert!(hierarchy.source().is_some());
        assert!(ComponentReflectError::NotAlive(e).source().is_none());
        for error in [
            ComponentReflectError::NotAlive(e),
            ComponentReflectError::UnknownComponent("a".into()),
            ComponentReflectError::NotAComponent("a".into()),
            ComponentReflectError::Missing {
                entity: e,
                component: "a".into(),
            },
            ComponentReflectError::Refused {
                component: "a".into(),
                reason: "why",
            },
            value,
            hierarchy,
        ] {
            assert!(!error.to_string().is_empty());
        }
    }
}
