// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "inspect.rs"
// ============================================================================
//! The inspector's rows: an entity's components and their fields, read
//! through reflection and turned into lines of text.

use mid_ecs::{Entity, World};
use mid_reflect::{Kind, Reflect, Registration, Ty, TypeRegistry, Value};

use crate::components::Name;

/// How deep fields are followed before a value is shown as it is.
const MAX_DEPTH: u8 = 6;
/// How many list items one line shows.
const MAX_ITEMS: usize = 8;
/// Components that are results of others, so read only and large: they show as
/// one line, the position they put the entity at.
const COLLAPSED: [&str; 2] = ["mid_ecs::GlobalTransform", "mid_ecs::GlobalTransformLWC"];

/// One line of the inspector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// How far the line is indented, in levels.
    pub indent: u8,
    /// The component or field name.
    pub label: String,
    /// The value, or empty when the line only introduces fields below it.
    pub text: String,
    /// Whether the line names a component rather than a field.
    pub header: bool,
}

/// The name a component is shown under: the registered name without its
/// namespace.
fn short(name: &str) -> &str {
    name.rsplit("::").next().unwrap_or(name)
}

/// The rows for `entity`: every registered component it holds, then a line
/// counting the ones that are not registered. Empty if it is not alive.
pub fn rows(world: &World, registry: &TypeRegistry, entity: Entity) -> Vec<Row> {
    let Some(reflection) = world.reflect_entity(registry, entity) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (name, value) in &reflection.components {
        let ty = Ty::Named(name.clone());
        if COLLAPSED.contains(&name.as_str()) {
            if let Some(text) = translation_text(world, registry, &ty, value) {
                out.push(Row {
                    indent: 0,
                    label: short(name).to_string(),
                    text: format!("pos {text}"),
                    header: true,
                });
                continue;
            }
        }
        match inline(world, registry, &ty, value) {
            Some(text) => out.push(Row {
                indent: 0,
                label: short(name).to_string(),
                text,
                header: true,
            }),
            None => {
                out.push(Row {
                    indent: 0,
                    label: short(name).to_string(),
                    text: String::new(),
                    header: true,
                });
                fields(world, registry, &ty, value, 1, &mut out);
            }
        }
    }
    if reflection.unreflected > 0 {
        out.push(Row {
            indent: 0,
            label: format!("{} more not shown", reflection.unreflected),
            text: String::new(),
            header: false,
        });
    }
    out
}

/// The `translation` of a newtype around a transform struct, as one line.
fn translation_text(
    world: &World,
    registry: &TypeRegistry,
    ty: &Ty,
    value: &Value,
) -> Option<String> {
    let Kind::Newtype(inner) = &registration(registry, ty)?.info().kind else {
        return None;
    };
    let Kind::Struct(list) = &registration(registry, inner)?.info().kind else {
        return None;
    };
    let field = list.iter().find(|f| f.name == "translation")?;
    inline(world, registry, &field.ty, value.get("translation")?)
}

fn registration<'r>(registry: &'r TypeRegistry, ty: &Ty) -> Option<&'r Registration> {
    match ty {
        Ty::Named(name) => registry.get(name),
        _ => None,
    }
}

/// Adds a line for each field of the struct `ty` (through newtypes), at
/// `indent`.
fn fields(
    world: &World,
    registry: &TypeRegistry,
    ty: &Ty,
    value: &Value,
    indent: u8,
    out: &mut Vec<Row>,
) {
    if indent > MAX_DEPTH {
        return;
    }
    let Some(reg) = registration(registry, ty) else {
        return;
    };
    match &reg.info().kind {
        Kind::Newtype(inner) => fields(world, registry, inner, value, indent, out),
        Kind::Struct(list) => {
            for field in list {
                let Some(v) = value.get(&field.name) else {
                    continue;
                };
                match inline(world, registry, &field.ty, v) {
                    Some(text) => out.push(Row {
                        indent,
                        label: field.name.clone(),
                        text,
                        header: false,
                    }),
                    None => {
                        out.push(Row {
                            indent,
                            label: field.name.clone(),
                            text: String::new(),
                            header: false,
                        });
                        fields(world, registry, &field.ty, v, indent + 1, out);
                    }
                }
            }
        }
        Kind::Enum(_) => {}
    }
}

fn is_primitive(ty: &Ty) -> bool {
    matches!(ty, Ty::Bool | Ty::Int { .. } | Ty::Float { .. } | Ty::Str)
}

fn float_text(f: f64) -> String {
    if f.is_finite() {
        format!("{f:.2}")
    } else {
        "nan".to_string()
    }
}

/// An entity value as the name of the entity it stands for.
fn entity_text(world: &World, value: &Value) -> String {
    match Entity::from_value(value) {
        Ok(e) if world.is_alive(e) => match world.get_static::<Name>(e) {
            Some(name) => name.0.clone(),
            None => format!("Entity {}", e.as_ffi() & 0xFFFF_FFFF),
        },
        Ok(_) => "dead".to_string(),
        Err(_) => "?".to_string(),
    }
}

/// `value` as one line of text, or `None` when it has parts that need their
/// own lines.
fn inline(world: &World, registry: &TypeRegistry, ty: &Ty, value: &Value) -> Option<String> {
    match ty {
        Ty::Bool => value.as_bool().map(|b| b.to_string()),
        Ty::Int { .. } => value
            .as_i64()
            .map(|n| n.to_string())
            .or_else(|| value.as_u64().map(|n| n.to_string())),
        Ty::Float { .. } => value.as_f64().map(float_text),
        Ty::Str => value.as_str().map(str::to_string),
        Ty::Option(inner) => match value {
            Value::Null => Some("none".to_string()),
            other => inline(world, registry, inner, other),
        },
        Ty::List(inner) | Ty::Array(inner, _) => {
            let items = value.as_list()?;
            if items.is_empty() {
                return Some("empty".to_string());
            }
            let mut parts = Vec::new();
            for item in items.iter().take(MAX_ITEMS) {
                parts.push(inline(world, registry, inner, item)?);
            }
            let mut text = parts.join(", ");
            if items.len() > MAX_ITEMS {
                text.push_str(&format!(", and {} more", items.len() - MAX_ITEMS));
            }
            Some(text)
        }
        Ty::Named(name) => match name.as_str() {
            "mid_ecs::Entity" => Some(entity_text(world, value)),
            "mid_editor::Rgba" => value.as_u64().map(|n| format!("#{n:08X}")),
            _ => {
                let reg = registry.get(name)?;
                match &reg.info().kind {
                    Kind::Newtype(inner) => inline(world, registry, inner, value),
                    Kind::Enum(_) => value.as_str().map(str::to_string),
                    Kind::Struct(list) => {
                        if list.len() > 4 || !list.iter().all(|f| is_primitive(&f.ty)) {
                            return None;
                        }
                        let mut parts = Vec::new();
                        for f in list {
                            parts.push(inline(world, registry, &f.ty, value.get(&f.name)?)?);
                        }
                        Some(parts.join("  "))
                    }
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::registry;
    use crate::scene::spawn_demo;

    fn demo() -> (World, TypeRegistry) {
        let mut w = World::new();
        w.enable_hierarchy();
        spawn_demo(&mut w, 9);
        w.propagate_transforms();
        (w, registry())
    }

    fn find(w: &World, name: &str) -> Entity {
        w.query_static::<Name>()
            .find(|(_, n)| n.0 == name)
            .map(|(e, _)| e)
            .unwrap()
    }

    fn text_of(rows: &[Row], label: &str) -> String {
        rows.iter()
            .find(|r| r.label == label)
            .unwrap_or_else(|| panic!("no row {label} in {rows:?}"))
            .text
            .clone()
    }

    #[test]
    fn a_dead_entity_has_no_rows() {
        let (mut w, r) = demo();
        let e = find(&w, "Cube");
        w.despawn(e);
        assert!(rows(&w, &r, e).is_empty());
    }

    #[test]
    fn components_are_headers_and_their_fields_are_not() {
        let (w, r) = demo();
        let rows = rows(&w, &r, find(&w, "Cube.001"));
        let headers: Vec<&str> = rows
            .iter()
            .filter(|r| r.header)
            .map(|r| r.label.as_str())
            .collect();
        for want in [
            "Name",
            "LocalTransform",
            "GlobalTransform",
            "ChildOf",
            "BoxShape",
            "Bob",
        ] {
            assert!(headers.contains(&want), "{want} in {headers:?}");
        }
        assert!(rows
            .iter()
            .filter(|r| !r.header)
            .all(|r| r.indent >= 1 || r.label.contains("not shown")));
    }

    #[test]
    fn simple_components_fit_on_their_header_line() {
        let (w, r) = demo();
        let rows = rows(&w, &r, find(&w, "Cube.003"));
        assert_eq!(text_of(&rows, "Name"), "Cube.003");
        assert_eq!(
            text_of(&rows, "ChildOf"),
            "Cube.002",
            "a parent shows as its name"
        );
    }

    #[test]
    fn a_parent_lists_its_children_by_name() {
        let (w, r) = demo();
        let rows = rows(&w, &r, find(&w, "Cube"));
        assert_eq!(text_of(&rows, "Children"), "Cube.001, Cube.002");
    }

    #[test]
    fn vectors_and_quaternions_are_one_line_of_numbers() {
        let (w, r) = demo();
        let rows = rows(&w, &r, find(&w, "Cube.004"));
        assert_eq!(text_of(&rows, "translation"), "0.00  0.50  0.00");
        assert_eq!(text_of(&rows, "rotation"), "0.00  0.00  0.00  1.00");
        assert_eq!(text_of(&rows, "scale"), "1.00  1.00  1.00");
        let local = rows
            .iter()
            .position(|r| r.label == "LocalTransform")
            .unwrap();
        let translation = rows.iter().position(|r| r.label == "translation").unwrap();
        assert!(translation > local);
        assert_eq!(rows[translation].indent, 1);
    }

    #[test]
    fn nested_structs_get_their_own_lines_one_level_deeper() {
        let (w, r) = demo();
        let value = w
            .get_static::<mid_ecs::GlobalTransform>(find(&w, "Cube"))
            .unwrap()
            .to_value();
        let mut out = Vec::new();
        fields(
            &w,
            &r,
            &Ty::Named("mid_ecs::GlobalTransform".into()),
            &value,
            1,
            &mut out,
        );
        let labels: Vec<&str> = out.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, vec!["x_axis", "y_axis", "z_axis", "translation"]);
        assert!(out.iter().all(|r| r.indent == 1 && !r.header));
        assert_eq!(out[0].text, "1.00  0.00  0.00");
    }

    #[test]
    fn a_derived_transform_is_one_line_with_its_position() {
        let (w, r) = demo();
        let rows = rows(&w, &r, find(&w, "Cube.004"));
        assert_eq!(text_of(&rows, "GlobalTransform"), "pos 0.00  0.50  0.00");
        assert!(
            rows.iter().all(|r| r.label != "x_axis"),
            "its matrix is not listed"
        );
        assert!(
            rows.iter()
                .find(|r| r.label == "GlobalTransform")
                .unwrap()
                .header
        );
    }

    #[test]
    fn colors_show_as_hex() {
        let (w, r) = demo();
        let rows = rows(&w, &r, find(&w, "Cube"));
        assert_eq!(text_of(&rows, "color"), "#E57373FF");
    }

    #[test]
    fn a_component_without_registration_is_counted_not_listed() {
        struct Hidden;
        let (mut w, r) = demo();
        let e = find(&w, "Cube");
        w.insert_static(e, Hidden);
        let rows = rows(&w, &r, e);
        assert_eq!(rows.last().unwrap().label, "1 more not shown");
    }

    #[test]
    fn a_dead_parent_shows_as_dead_not_as_a_number() {
        let (mut w, r) = demo();
        let child = find(&w, "Cube.003");
        let two = find(&w, "Cube.002");
        // Keep the component but point it at a stale handle.
        let stale = w.spawn();
        w.despawn(stale);
        w.get_static_mut::<mid_ecs::ChildOf>(child).unwrap().0 = stale;
        let rows = rows(&w, &r, child);
        assert_eq!(text_of(&rows, "ChildOf"), "dead");
        let _ = two;
    }

    #[test]
    fn long_lists_are_cut_with_a_count() {
        let (w, r) = demo();
        let value = Value::List((0..12).map(|n| Value::Int(n)).collect());
        let ty = Ty::List(Box::new(Ty::Int {
            bits: 32,
            signed: true,
        }));
        let text = inline(&w, &r, &ty, &value).unwrap();
        assert!(text.starts_with("0, 1, 2"));
        assert!(text.ends_with("and 4 more"), "{text}");
        let empty = inline(&w, &r, &ty, &Value::List(vec![])).unwrap();
        assert_eq!(empty, "empty");
    }

    #[test]
    fn depth_is_bounded() {
        let (w, r) = demo();
        let mut out = Vec::new();
        fields(
            &w,
            &r,
            &Ty::Named("mid_ecs::GlobalTransform".into()),
            &w.get_static::<mid_ecs::GlobalTransform>(find(&w, "Cube"))
                .unwrap()
                .to_value(),
            MAX_DEPTH + 1,
            &mut out,
        );
        assert!(out.is_empty());
    }
}
