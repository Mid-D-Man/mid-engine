// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "scene.rs"
// ============================================================================
//! The demo scene the editor shows until the ECS is wired in.

use crate::paint::rgba;

/// One thing in the scene: a named box with a parent.
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    /// The name shown in the hierarchy.
    pub name: String,
    /// Index of the parent entity, or `None` for a child of the scene root.
    pub parent: Option<usize>,
    /// World position of the box's center.
    pub pos: [f32; 3],
    /// Half the box's size on each axis.
    pub half: [f32; 3],
    /// Rotation about the Y axis in radians.
    pub yaw: f32,
    /// Outline color, `0xRRGGBBAA`.
    pub color: u32,
    /// Where the box rests, for the play-mode animation.
    base_y: f32,
}

/// A flat list of entities.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    /// The entities. An index into this list identifies an entity.
    pub entities: Vec<Entity>,
}

const PALETTE: [u32; 6] = [
    rgba(0xE5, 0x73, 0x73, 0xFF),
    rgba(0x81, 0xC7, 0x84, 0xFF),
    rgba(0x64, 0xB5, 0xF6, 0xFF),
    rgba(0xFF, 0x8A, 0x65, 0xFF),
    rgba(0xBA, 0x68, 0xC8, 0xFF),
    rgba(0x4D, 0xD0, 0xE1, 0xFF),
];

impl Scene {
    /// A scene of `count` cubes on a square grid, 2.5 units apart, resting on
    /// the ground. Names run `Cube`, `Cube.001`, `Cube.002`, and so on.
    pub fn demo(count: usize) -> Self {
        let side = (count as f32).sqrt().ceil().max(1.0) as usize;
        let spacing = 2.5_f32;
        let origin = -((side - 1) as f32) * spacing * 0.5;
        let mut entities = Vec::with_capacity(count);
        for i in 0..count {
            let (col, row) = (i % side, i / side);
            let name = if i == 0 {
                "Cube".to_string()
            } else {
                format!("Cube.{i:03}")
            };
            entities.push(Entity {
                name,
                parent: None,
                pos: [
                    origin + col as f32 * spacing,
                    0.5,
                    origin + row as f32 * spacing,
                ],
                half: [0.5, 0.5, 0.5],
                yaw: 0.0,
                color: PALETTE[i % PALETTE.len()],
                base_y: 0.5,
            });
        }
        Scene { entities }
    }

    /// Number of entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// True when there are no entities.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Poses every entity for play-mode time `t` in seconds: each cube bobs and
    /// turns, out of step with its neighbours. At `t == 0` every cube is at
    /// rest, which is the edit-mode pose.
    pub fn pose(&mut self, t: f32) {
        for (i, e) in self.entities.iter_mut().enumerate() {
            let phase = i as f32 * 0.7;
            e.pos[1] = e.base_y
                + if t == 0.0 {
                    0.0
                } else {
                    0.3 * (t * 2.0 + phase).sin()
                };
            e.yaw = t * 0.8 + if t == 0.0 { 0.0 } else { phase };
        }
    }

    /// The eight corners of an entity's box in world space, after its yaw.
    pub fn corners(e: &Entity) -> [[f32; 3]; 8] {
        let (s, c) = e.yaw.sin_cos();
        let mut out = [[0.0; 3]; 8];
        for (i, o) in out.iter_mut().enumerate() {
            let x = if i & 1 == 0 { -e.half[0] } else { e.half[0] };
            let y = if i & 2 == 0 { -e.half[1] } else { e.half[1] };
            let z = if i & 4 == 0 { -e.half[2] } else { e.half[2] };
            *o = [
                e.pos[0] + x * c + z * s,
                e.pos[1] + y,
                e.pos[2] - x * s + z * c,
            ];
        }
        out
    }
}

/// The twelve edges of a box, as pairs of indices into [`Scene::corners`].
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_names_and_counts() {
        let s = Scene::demo(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s.entities[0].name, "Cube");
        assert_eq!(s.entities[1].name, "Cube.001");
        assert_eq!(s.entities[2].name, "Cube.002");
        assert!(Scene::demo(0).is_empty());
    }

    #[test]
    fn demo_lays_cubes_out_on_a_centred_grid_resting_on_the_ground() {
        let s = Scene::demo(4);
        let xs: Vec<f32> = s.entities.iter().map(|e| e.pos[0]).collect();
        assert_eq!(xs, vec![-1.25, 1.25, -1.25, 1.25]);
        assert!(s.entities.iter().all(|e| e.pos[1] == 0.5));
        let one = Scene::demo(1);
        assert_eq!(one.entities[0].pos, [0.0, 0.5, 0.0]);
    }

    #[test]
    fn corners_are_eight_distinct_points_at_the_right_distance() {
        let e = &Scene::demo(1).entities[0];
        let c = Scene::corners(e);
        for (i, a) in c.iter().enumerate() {
            for b in &c[i + 1..] {
                assert_ne!(a, b);
            }
            let d =
                ((a[0] - e.pos[0]).powi(2) + (a[1] - e.pos[1]).powi(2) + (a[2] - e.pos[2]).powi(2))
                    .sqrt();
            assert!((d - 0.75_f32.sqrt()).abs() < 1e-5);
        }
    }

    #[test]
    fn every_edge_joins_corners_one_axis_apart() {
        let e = &Scene::demo(1).entities[0];
        let c = Scene::corners(e);
        for (a, b) in EDGES {
            let d = ((c[a][0] - c[b][0]).powi(2)
                + (c[a][1] - c[b][1]).powi(2)
                + (c[a][2] - c[b][2]).powi(2))
            .sqrt();
            assert!((d - 1.0).abs() < 1e-5, "edge {a}-{b} has length {d}");
        }
    }

    #[test]
    fn yaw_turns_the_box_about_its_center() {
        let mut e = Scene::demo(1).entities[0].clone();
        e.yaw = std::f32::consts::FRAC_PI_2;
        let c = Scene::corners(&e);
        let cx: f32 = c.iter().map(|p| p[0]).sum::<f32>() / 8.0;
        let cz: f32 = c.iter().map(|p| p[2]).sum::<f32>() / 8.0;
        assert!(cx.abs() < 1e-5 && cz.abs() < 1e-5);
        assert!(c.iter().all(|p| p[1] >= 0.0 && p[1] <= 1.0));
    }

    #[test]
    fn no_cube_color_matches_the_selection_color() {
        let s = Scene::demo(PALETTE.len() * 2);
        assert!(s
            .entities
            .iter()
            .all(|e| e.color != crate::viewport::SELECTED));
    }

    #[test]
    fn pose_zero_is_the_rest_pose_and_play_time_moves_things() {
        let mut s = Scene::demo(5);
        s.pose(0.0);
        assert!(s.entities.iter().all(|e| e.pos[1] == 0.5 && e.yaw == 0.0));
        s.pose(1.0);
        assert!(s.entities.iter().any(|e| e.pos[1] != 0.5));
        assert!(s
            .entities
            .iter()
            .all(|e| (e.pos[1] - 0.5).abs() <= 0.3 + 1e-5));
        s.pose(0.0);
        assert!(s.entities.iter().all(|e| e.pos[1] == 0.5), "back to rest");
    }
}
