// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "scene.rs"
// ============================================================================
//! The demo scene the editor shows until scenes can be loaded: cubes on a
//! grid, spawned into a [`World`].

use mid_ecs::{Entity, GlobalTransform, LocalTransform, World};
use mid_math::Vec3;

use crate::components::{Bob, BoxShape, Name, Rgba};
use crate::paint::rgba;

/// Outline colors the cubes cycle through. None is the selection color.
pub const PALETTE: [u32; 6] = [
    rgba(0xE5, 0x73, 0x73, 0xFF),
    rgba(0x81, 0xC7, 0x84, 0xFF),
    rgba(0x64, 0xB5, 0xF6, 0xFF),
    rgba(0xFF, 0x8A, 0x65, 0xFF),
    rgba(0xBA, 0x68, 0xC8, 0xFF),
    rgba(0x4D, 0xD0, 0xE1, 0xFF),
];

/// The twelve edges of a box, as pairs of indices into
/// [`crate::world::Drawable::corners`].
pub const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // along x
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // along y
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // along z
];

const SPACING: f32 = 2.5;

/// Where cube `i` of `count` rests: a square grid, 2.5 units apart, centred on
/// the origin, resting on the ground.
fn grid_position(i: usize, count: usize) -> Vec3 {
    let side = (count as f32).sqrt().ceil().max(1.0) as usize;
    let origin = -((side - 1) as f32) * SPACING * 0.5;
    let (col, row) = (i % side, i / side);
    Vec3::new(
        origin + col as f32 * SPACING,
        0.5,
        origin + row as f32 * SPACING,
    )
}

/// Which cube is cube `i`'s parent, if any: `Cube.001` and `Cube.002` hang
/// under `Cube`, and `Cube.003` under `Cube.002`, so the hierarchy and the
/// transform composition have something to show. Everything else is a root.
fn parent_index(i: usize) -> Option<usize> {
    match i {
        1 | 2 => Some(0),
        3 => Some(2),
        _ => None,
    }
}

/// Spawns `count` cubes into `world`. Names run `Cube`, `Cube.001`,
/// `Cube.002`, and so on. Each cube's world position is its grid position;
/// a child's local position is that minus its parent's, so nothing moves when
/// the hierarchy is built.
///
/// The world needs its hierarchy enabled (`App` with `TransformPlugin` does
/// that); transforms are composed by the first propagation.
pub fn spawn_demo(world: &mut World, count: usize) {
    let mut spawned: Vec<Entity> = Vec::with_capacity(count);
    for i in 0..count {
        let name = if i == 0 {
            "Cube".to_string()
        } else {
            format!("Cube.{i:03}")
        };
        let world_pos = grid_position(i, count);
        let parent = parent_index(i).filter(|&p| p < i);
        let local = match parent {
            Some(p) => world_pos - grid_position(p, count),
            None => world_pos,
        };
        let entity = world.spawn_bundle((
            Name(name),
            LocalTransform::from_translation(local),
            GlobalTransform::IDENTITY,
            BoxShape {
                half: Vec3::new(0.5, 0.5, 0.5),
                color: Rgba(PALETTE[i % PALETTE.len()]),
            },
            Bob {
                base_y: local.y,
                phase: i as f32 * 0.7,
            },
        ));
        if let Some(p) = parent {
            world
                .set_parent(entity, spawned[p])
                .expect("both entities are alive and the new one has no children");
        }
        spawned.push(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewport::SELECTED;

    fn world() -> World {
        let mut w = World::new();
        w.enable_hierarchy();
        w
    }

    #[test]
    fn grid_is_centred_and_rests_on_the_ground() {
        let xs: Vec<f32> = (0..4).map(|i| grid_position(i, 4).x).collect();
        assert_eq!(xs, vec![-1.25, 1.25, -1.25, 1.25]);
        assert!((0..4).all(|i| grid_position(i, 4).y == 0.5));
        let one = grid_position(0, 1);
        assert_eq!((one.x, one.y, one.z), (0.0, 0.5, 0.0));
    }

    #[test]
    fn spawn_demo_names_and_counts() {
        let mut w = world();
        spawn_demo(&mut w, 3);
        assert_eq!(w.entity_count(), 3);
        let mut names: Vec<String> = w.query_static::<Name>().map(|(_, n)| n.0.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["Cube", "Cube.001", "Cube.002"]);
        let mut empty = world();
        spawn_demo(&mut empty, 0);
        assert_eq!(empty.entity_count(), 0);
    }

    #[test]
    fn the_first_cubes_form_a_small_hierarchy() {
        let mut w = world();
        spawn_demo(&mut w, 5);
        let by_name = |w: &World, wanted: &str| {
            w.query_static::<Name>()
                .find(|(_, n)| n.0 == wanted)
                .map(|(e, _)| e)
                .unwrap()
        };
        let root = by_name(&w, "Cube");
        let one = by_name(&w, "Cube.001");
        let two = by_name(&w, "Cube.002");
        let three = by_name(&w, "Cube.003");
        let four = by_name(&w, "Cube.004");
        assert_eq!(w.parent(one), Some(root));
        assert_eq!(w.parent(two), Some(root));
        assert_eq!(w.parent(three), Some(two));
        assert_eq!(w.parent(four), None);
        assert_eq!(w.children(root), &[one, two]);
    }

    #[test]
    fn small_scenes_have_no_dangling_parents() {
        for n in 0..4 {
            let mut w = world();
            spawn_demo(&mut w, n);
            assert_eq!(w.entity_count(), n);
            for (e, _) in w.query_static::<Name>() {
                if let Some(p) = w.parent(e) {
                    assert!(w.is_alive(p));
                }
            }
        }
    }

    #[test]
    fn composing_transforms_puts_every_cube_on_its_grid_position() {
        let mut w = world();
        spawn_demo(&mut w, 9);
        w.propagate_transforms();
        for (e, name) in w.query_static::<Name>() {
            let i: usize = name.0.rsplit('.').next().unwrap().parse().unwrap_or(0);
            let want = grid_position(i, 9);
            let got = w.get_static::<GlobalTransform>(e).unwrap().0.translation;
            assert!(
                (got.x - want.x).abs() < 1e-5
                    && (got.y - want.y).abs() < 1e-5
                    && (got.z - want.z).abs() < 1e-5,
                "{}: {got:?} vs {want:?}",
                name.0
            );
        }
    }

    #[test]
    fn no_cube_color_matches_the_selection_color() {
        let mut w = world();
        spawn_demo(&mut w, PALETTE.len() * 2);
        assert!(w
            .query_static::<BoxShape>()
            .all(|(_, b)| b.color.0 != SELECTED));
    }

    #[test]
    fn every_edge_joins_distinct_corner_indices_one_axis_apart() {
        for (a, b) in EDGES {
            assert_ne!(a, b);
            assert_eq!((a ^ b).count_ones(), 1, "edge {a}-{b} flips one axis bit");
        }
    }
}
