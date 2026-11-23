// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "components.rs"
// ============================================================================
//! The components the editor's own scenes are made of, and the reflection
//! registry that knows them and the engine's.

use mid_ecs::reflect::{register_component, register_core_components};
use mid_math::Vec3;
use mid_reflect::{impl_reflect_newtype, impl_reflect_struct, TypeRegistry};

/// The name shown in the hierarchy. Stands in for the `Name` that `mid-nodes`
/// will define (`docs/mid-nodes.md`, N3).
#[derive(Clone, Debug, PartialEq)]
pub struct Name(pub String);
impl_reflect_newtype!("mid_editor::Name" => Name(String));

/// A color, `0xRRGGBBAA`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgba(pub u32);
impl_reflect_newtype!("mid_editor::Rgba" => Rgba(u32));

/// A box the viewport draws as a wireframe, around the entity's origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxShape {
    /// Half the box's size on each axis, before the entity's own scale.
    pub half: Vec3,
    /// The outline color.
    pub color: Rgba,
}
impl_reflect_struct!("mid_editor::BoxShape" => BoxShape { half: Vec3, color: Rgba });

/// Makes an entity bob and turn while the world is playing: its local height
/// follows `base_y` plus a sine, its local yaw follows time, out of step with
/// other entities by `phase`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bob {
    /// The local height the bob is around.
    pub base_y: f32,
    /// Offset in radians, so neighbours move out of step.
    pub phase: f32,
}
impl_reflect_struct!("mid_editor::Bob" => Bob { base_y: f32, phase: f32 });

/// A registry that knows the engine's own components and the editor's.
pub fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    // The names are distinct and these types register once, so none of these
    // can fail; `expect` states that.
    register_core_components(&mut registry).expect("core component names are distinct");
    register_component::<Name>(&mut registry).expect("editor component names are distinct");
    register_component::<BoxShape>(&mut registry).expect("editor component names are distinct");
    register_component::<Bob>(&mut registry).expect("editor component names are distinct");
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use mid_reflect::{Reflect, Value};

    #[test]
    fn the_registry_knows_every_component_by_name() {
        let r = registry();
        for name in [
            "mid_ecs::LocalTransform",
            "mid_ecs::GlobalTransform",
            "mid_ecs::ChildOf",
            "mid_ecs::Children",
            "mid_editor::Name",
            "mid_editor::BoxShape",
            "mid_editor::Bob",
        ] {
            assert!(
                r.data_by_name::<mid_ecs::ReflectComponent>(name).is_some(),
                "{name}"
            );
        }
        assert!(r.get("mid_editor::Rgba").is_some());
    }

    #[test]
    fn components_round_trip_through_values() {
        let shape = BoxShape {
            half: Vec3::new(0.5, 1.0, 0.25),
            color: Rgba(0x11223344),
        };
        assert_eq!(BoxShape::from_value(&shape.to_value()).unwrap(), shape);
        let name = Name("Cube".to_string());
        assert_eq!(name.to_value(), Value::Str("Cube".to_string()));
        assert_eq!(Name::from_value(&name.to_value()).unwrap(), name);
        let bob = Bob {
            base_y: 0.5,
            phase: 1.4,
        };
        assert_eq!(Bob::from_value(&bob.to_value()).unwrap(), bob);
    }
}
