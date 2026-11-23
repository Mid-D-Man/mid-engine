// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "world.rs"
// ============================================================================
//! The world the editor shows and edits: a `mid-app` [`App`] holding a
//! `mid-ecs` world, the reflection registry that knows its components, the
//! boxes the viewport draws, the hierarchy rows the panel lists, and the
//! snapshot that play mode restores.

use std::collections::HashSet;
use std::time::Duration;

use mid_app::mid_time::Clock;
use mid_app::{App, TransformPlugin, Update};
use mid_ecs::{Entity, GlobalTransform, LocalTransform, Mut, ReflectComponent, World};
use mid_math::{Affine3, Quat, Vec3};
use mid_reflect::{TypeRegistry, Value};

use crate::components::{registry, Bob, BoxShape, Name};
use crate::inspect::{self, Row};
use crate::scene::spawn_demo;

/// Components that follow from others and are recomputed after a restore, so
/// the play-mode snapshot does not keep them.
const DERIVED: [&str; 2] = ["mid_ecs::GlobalTransform", "mid_ecs::GlobalTransformLWC"];
const CHILD_OF: &str = "mid_ecs::ChildOf";
const CHILDREN: &str = "mid_ecs::Children";

/// Seconds the world has been playing; advanced by the play-mode system.
#[derive(Default)]
struct PlayTime(f32);

/// Moves every [`Bob`] entity for the time since the last frame.
fn bob(world: &mut World) {
    let dt = world.get_resource::<Clock>().map_or(0.0, Clock::delta_secs);
    let t = match world.get_resource_mut::<PlayTime>() {
        Some(time) => {
            time.0 += dt;
            time.0
        }
        None => {
            world.insert_resource(PlayTime(dt));
            dt
        }
    };
    let up = Vec3::new(0.0, 1.0, 0.0);
    for (b, mut local) in world.query_data_mut::<(&Bob, Mut<LocalTransform>)>() {
        local.translation.y = b.base_y + 0.3 * (t * 2.0 + b.phase).sin();
        local.rotation = Quat::from_axis_angle(up, t * 0.8 + b.phase);
    }
}

/// One box for the viewport to draw, read from the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drawable {
    /// The entity it belongs to.
    pub entity: Entity,
    /// The entity's composed transform.
    pub xf: Affine3,
    /// Half the box's size on each axis, before the transform.
    pub half: [f32; 3],
    /// The box's half size along each of its own axes in world units: `half`
    /// scaled by the length of each axis of the transform.
    pub ext: [f32; 3],
    /// Outline color, `0xRRGGBBAA`.
    pub color: u32,
}

impl Drawable {
    /// The box's center in world space.
    pub fn center(&self) -> [f32; 3] {
        let t = self.xf.translation;
        [t.x, t.y, t.z]
    }

    /// The largest half extent: how far the box reaches from its center along
    /// an axis.
    pub fn radius(&self) -> f32 {
        self.ext[0].max(self.ext[1]).max(self.ext[2])
    }

    /// The eight corners in world space. Bit 0 of the index picks the sign on
    /// x, bit 1 on y, bit 2 on z (set means positive).
    pub fn corners(&self) -> [[f32; 3]; 8] {
        let mut out = [[0.0; 3]; 8];
        for (i, o) in out.iter_mut().enumerate() {
            let x = if i & 1 == 0 {
                -self.half[0]
            } else {
                self.half[0]
            };
            let y = if i & 2 == 0 {
                -self.half[1]
            } else {
                self.half[1]
            };
            let z = if i & 4 == 0 {
                -self.half[2]
            } else {
                self.half[2]
            };
            let p = self.xf.transform_point(Vec3::new(x, y, z));
            *o = [p.x, p.y, p.z];
        }
        out
    }
}

/// One line of the hierarchy panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HierarchyRow {
    /// The entity the row stands for.
    pub entity: Entity,
    /// How many parents it has.
    pub depth: usize,
}

/// What putting the world back after play changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RestoreReport {
    /// Entities that were in the snapshot and still alive, now restored.
    pub restored: usize,
    /// Entities spawned during play, now despawned.
    pub removed: usize,
    /// Entities that were in the snapshot but were despawned during play.
    /// They are not brought back.
    pub lost: usize,
    /// Components that could not be put back.
    pub failed: usize,
}

struct Snapshot {
    entities: Vec<(Entity, Vec<(String, Value)>)>,
}

/// The editor's world.
pub struct EditorWorld {
    app: App,
    registry: TypeRegistry,
    drawables: Vec<Drawable>,
    drawables_stale: bool,
    transforms_fresh: bool,
    roots: Vec<Entity>,
    rows: usize,
    structure_stale: bool,
    snapshot: Option<Snapshot>,
}

impl EditorWorld {
    /// A world of `count` demo cubes, with the systems that animate them in
    /// play mode.
    pub fn demo(count: usize) -> Self {
        let mut app = App::new();
        app.add_plugin(TransformPlugin);
        app.add_systems(Update, bob);
        spawn_demo(&mut app.world, count);
        let mut w = Self {
            app,
            registry: registry(),
            drawables: Vec::new(),
            drawables_stale: true,
            transforms_fresh: false,
            roots: Vec::new(),
            rows: 0,
            structure_stale: true,
            snapshot: None,
        };
        w.refresh();
        w
    }

    /// The world, to read.
    pub fn world(&self) -> &World {
        &self.app.world
    }

    /// The world, to change. Everything derived from it (the boxes, the
    /// hierarchy rows, the composed transforms) is rebuilt at the next
    /// [`Self::refresh`].
    pub fn world_mut(&mut self) -> &mut World {
        self.drawables_stale = true;
        self.transforms_fresh = false;
        self.structure_stale = true;
        &mut self.app.world
    }

    /// The registry of the components the editor can show.
    pub fn registry(&self) -> &TypeRegistry {
        &self.registry
    }

    /// Whether the world is being played.
    pub fn is_playing(&self) -> bool {
        self.snapshot.is_some()
    }

    /// Seconds since play began; zero when not playing.
    pub fn play_time(&self) -> f32 {
        self.app
            .world
            .get_resource::<PlayTime>()
            .map_or(0.0, |t| t.0)
    }

    /// Brings the boxes and hierarchy rows up to date with the world. Does
    /// nothing when nothing changed.
    pub fn refresh(&mut self) {
        if !self.transforms_fresh {
            self.app.world.propagate_transforms();
            self.transforms_fresh = true;
        }
        if self.structure_stale {
            self.rebuild_structure();
        }
        if self.drawables_stale {
            self.rebuild_drawables();
        }
    }

    fn rebuild_drawables(&mut self) {
        self.drawables_stale = false;
        self.drawables.clear();
        let world = &self.app.world;
        for (entity, global, shape) in world.query_data::<(Entity, &GlobalTransform, &BoxShape)>() {
            let m = global.0;
            let half = [shape.half.x, shape.half.y, shape.half.z];
            self.drawables.push(Drawable {
                entity,
                xf: m,
                half,
                ext: [
                    half[0] * m.x_axis.length(),
                    half[1] * m.y_axis.length(),
                    half[2] * m.z_axis.length(),
                ],
                color: shape.color.0,
            });
        }
    }

    fn rebuild_structure(&mut self) {
        self.structure_stale = false;
        let world = &self.app.world;
        self.roots.clear();
        self.roots
            .extend(world.entities().filter(|e| world.parent(*e).is_none()));
        self.roots.sort_by_key(|e| e.as_ffi());
        // Every entity is under some root, so the rows are the entities.
        self.rows = world.entity_count();
    }

    /// The boxes to draw, as of the last [`Self::refresh`].
    pub fn drawables(&self) -> &[Drawable] {
        &self.drawables
    }

    /// How many boxes there are to draw.
    pub fn box_count(&self) -> usize {
        self.drawables.len()
    }

    /// How many rows the hierarchy has: one per entity.
    pub fn row_count(&self) -> usize {
        self.rows
    }

    /// The hierarchy rows from row `first` on, at most `count` of them, in
    /// tree order: roots by age, each followed by its children in order.
    pub fn hierarchy_rows(&self, first: usize, count: usize, out: &mut Vec<HierarchyRow>) {
        out.clear();
        if count == 0 {
            return;
        }
        let world = &self.app.world;
        let mut index = 0usize;
        let mut stack: Vec<(Entity, usize)> = Vec::new();
        for &root in &self.roots {
            stack.push((root, 0));
            while let Some((entity, depth)) = stack.pop() {
                if index >= first {
                    out.push(HierarchyRow { entity, depth });
                    if out.len() == count {
                        return;
                    }
                }
                index += 1;
                for &child in world.children(entity).iter().rev() {
                    stack.push((child, depth + 1));
                }
            }
        }
    }

    /// The name of `entity`, or `Entity <n>` if it has none.
    pub fn name_of(&self, entity: Entity) -> String {
        match self.app.world.get_static::<Name>(entity) {
            Some(name) => name.0.clone(),
            None => format!("Entity {}", entity.as_ffi() & 0xFFFF_FFFF),
        }
    }

    /// The first entity named `name`.
    pub fn find(&self, name: &str) -> Option<Entity> {
        self.app
            .world
            .query_static::<Name>()
            .find(|(_, n)| n.0 == name)
            .map(|(e, _)| e)
    }

    /// Where `entity`'s origin is in world space.
    pub fn position_of(&self, entity: Entity) -> Option<[f32; 3]> {
        self.app
            .world
            .get_static::<GlobalTransform>(entity)
            .map(|g| [g.0.translation.x, g.0.translation.y, g.0.translation.z])
    }

    /// The inspector's rows for `entity`: each registered component and its
    /// fields, read through reflection.
    pub fn inspect(&self, entity: Entity) -> Vec<Row> {
        inspect::rows(&self.app.world, &self.registry, entity)
    }

    /// Starts playing: keeps the world's state so [`Self::exit_play`] can put
    /// it back, and restarts the clock. Does nothing if already playing.
    pub fn enter_play(&mut self) {
        if self.snapshot.is_some() {
            return;
        }
        self.refresh();
        self.snapshot = Some(self.capture());
        let world = &mut self.app.world;
        world.insert_resource(Clock::new());
        world.insert_resource(PlayTime(0.0));
    }

    /// Runs one frame of play of `dt` seconds. Does nothing unless playing.
    pub fn step(&mut self, dt: f32) {
        if self.snapshot.is_none() {
            return;
        }
        self.app.update_with(Duration::from_secs_f32(dt));
        // The app's PostUpdate composed the transforms.
        self.transforms_fresh = true;
        self.drawables_stale = true;
        self.structure_stale = true;
    }

    /// Stops playing and puts the world back as it was when play began.
    /// `None` if the world was not playing.
    pub fn exit_play(&mut self) -> Option<RestoreReport> {
        let snapshot = self.snapshot.take()?;
        let report = self.restore(snapshot);
        self.app.world.insert_resource(PlayTime(0.0));
        self.transforms_fresh = false;
        self.drawables_stale = true;
        self.structure_stale = true;
        self.refresh();
        Some(report)
    }

    /// The components the snapshot keeps: every registered component that is
    /// not derived from others.
    fn capturable(&self) -> Vec<(&str, &ReflectComponent)> {
        self.registry
            .iter()
            .filter(|r| !DERIVED.contains(&r.name()))
            .filter_map(|r| r.data::<ReflectComponent>().map(|d| (r.name(), d)))
            .collect()
    }

    fn capture(&self) -> Snapshot {
        let world = &self.app.world;
        let capturable = self.capturable();
        let mut entities = Vec::with_capacity(world.entity_count());
        for entity in world.entities() {
            let mut components = Vec::new();
            for (name, data) in &capturable {
                if let Some(value) = data.get(world, entity) {
                    components.push((name.to_string(), value));
                }
            }
            entities.push((entity, components));
        }
        Snapshot { entities }
    }

    fn restore(&mut self, snapshot: Snapshot) -> RestoreReport {
        let capturable: Vec<(String, ReflectComponent)> = self
            .capturable()
            .into_iter()
            .map(|(n, d)| (n.to_string(), d.clone()))
            .collect();
        let find = |name: &str| capturable.iter().find(|(n, _)| n == name).map(|(_, d)| d);
        let world = &mut self.app.world;
        let mut report = RestoreReport::default();

        // Entities spawned during play go.
        let kept: HashSet<Entity> = snapshot.entities.iter().map(|(e, _)| *e).collect();
        let extras: Vec<Entity> = world.entities().filter(|e| !kept.contains(e)).collect();
        for entity in extras {
            if world.despawn(entity) {
                report.removed += 1;
            }
        }

        // Detach everything first: parents are put back one at a time below,
        // and a half-restored hierarchy could otherwise look like a cycle.
        for (entity, _) in &snapshot.entities {
            world.remove_parent(*entity);
        }

        // Everything else goes back to its value, and what play added goes.
        for (entity, components) in &snapshot.entities {
            if !world.is_alive(*entity) {
                report.lost += 1;
                continue;
            }
            report.restored += 1;
            for (name, data) in &capturable {
                if name == CHILD_OF || name == CHILDREN {
                    continue;
                }
                if !components.iter().any(|(n, _)| n == name) && data.contains(world, *entity) {
                    match data.remove(world, *entity) {
                        Ok(_) => {}
                        Err(_) => report.failed += 1,
                    }
                }
            }
            for (name, value) in components {
                if name == CHILD_OF || name == CHILDREN {
                    continue;
                }
                let Some(data) = find(name) else {
                    continue;
                };
                if data.insert(world, *entity, value).is_err() {
                    report.failed += 1;
                }
            }
        }

        // Parents, child by child in each parent's own order.
        if let Some(child_of) = find(CHILD_OF) {
            for (parent, components) in &snapshot.entities {
                let Some((_, Value::List(children))) =
                    components.iter().find(|(n, _)| n == CHILDREN)
                else {
                    continue;
                };
                let parent_value = Value::from_u64(parent.as_ffi());
                for child in children {
                    let alive = child
                        .as_u64()
                        .map(Entity::from_ffi)
                        .is_some_and(|c| world.is_alive(c));
                    if !alive || !world.is_alive(*parent) {
                        report.failed += 1;
                        continue;
                    }
                    let child = Entity::from_ffi(child.as_u64().unwrap_or_default());
                    if child_of.insert(world, child, &parent_value).is_err() {
                        report.failed += 1;
                    }
                }
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(w: &EditorWorld, first: usize, count: usize) -> Vec<(String, usize)> {
        let mut rows = Vec::new();
        w.hierarchy_rows(first, count, &mut rows);
        rows.iter()
            .map(|r| (w.name_of(r.entity), r.depth))
            .collect()
    }

    fn rest_positions(w: &EditorWorld) -> Vec<(Entity, [f32; 3])> {
        let mut v: Vec<(Entity, [f32; 3])> = w
            .drawables()
            .iter()
            .map(|d| (d.entity, d.center()))
            .collect();
        v.sort_by_key(|(e, _)| e.as_ffi());
        v
    }

    #[test]
    fn a_demo_world_has_a_box_and_a_row_per_cube() {
        let w = EditorWorld::demo(12);
        assert_eq!(w.box_count(), 12);
        assert_eq!(w.row_count(), 12);
        assert!(!w.is_playing());
        assert_eq!(w.play_time(), 0.0);
    }

    #[test]
    fn an_empty_world_has_nothing_to_draw_or_list() {
        let w = EditorWorld::demo(0);
        assert_eq!((w.box_count(), w.row_count()), (0, 0));
        let mut rows = Vec::new();
        w.hierarchy_rows(0, 10, &mut rows);
        assert!(rows.is_empty());
    }

    #[test]
    fn hierarchy_rows_are_in_tree_order_with_depths() {
        let w = EditorWorld::demo(6);
        assert_eq!(
            names(&w, 0, 10),
            vec![
                ("Cube".to_string(), 0),
                ("Cube.001".to_string(), 1),
                ("Cube.002".to_string(), 1),
                ("Cube.003".to_string(), 2),
                ("Cube.004".to_string(), 0),
                ("Cube.005".to_string(), 0),
            ]
        );
    }

    #[test]
    fn hierarchy_rows_page_from_any_row() {
        let w = EditorWorld::demo(6);
        assert_eq!(
            names(&w, 3, 2),
            vec![("Cube.003".to_string(), 2), ("Cube.004".to_string(), 0)]
        );
        assert_eq!(names(&w, 5, 10).len(), 1);
        assert!(names(&w, 6, 10).is_empty());
        assert!(names(&w, 0, 0).is_empty());
        let mut rows = vec![HierarchyRow {
            entity: w.find("Cube").unwrap(),
            depth: 9,
        }];
        w.hierarchy_rows(100, 5, &mut rows);
        assert!(
            rows.is_empty(),
            "the buffer is cleared even when nothing fits"
        );
    }

    #[test]
    fn composed_positions_match_the_grid_and_drawables_carry_them() {
        let w = EditorWorld::demo(4);
        let p = w.position_of(w.find("Cube.003").unwrap()).unwrap();
        assert_eq!(p, [1.25, 0.5, 1.25]);
        let d = w
            .drawables()
            .iter()
            .find(|d| d.entity == w.find("Cube.003").unwrap())
            .unwrap();
        assert_eq!(d.center(), p);
        assert_eq!(d.ext, [0.5, 0.5, 0.5]);
        assert_eq!(d.radius(), 0.5);
    }

    #[test]
    fn corners_are_eight_distinct_points_one_unit_apart_along_edges() {
        let w = EditorWorld::demo(1);
        let d = w.drawables()[0];
        let c = d.corners();
        for (i, a) in c.iter().enumerate() {
            for b in &c[i + 1..] {
                assert_ne!(a, b);
            }
        }
        for (a, b) in crate::scene::EDGES {
            let len = ((c[a][0] - c[b][0]).powi(2)
                + (c[a][1] - c[b][1]).powi(2)
                + (c[a][2] - c[b][2]).powi(2))
            .sqrt();
            assert!((len - 1.0).abs() < 1e-5, "edge {a}-{b} has length {len}");
        }
    }

    #[test]
    fn a_scaled_and_turned_entity_has_scaled_extents_and_corners() {
        let mut w = EditorWorld::demo(1);
        let e = w.find("Cube").unwrap();
        let world = w.world_mut();
        let local = world.get_static_mut::<LocalTransform>(e).unwrap();
        local.scale = Vec3::new(2.0, 1.0, 3.0);
        local.rotation =
            Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
        w.refresh();
        let d = w.drawables()[0];
        assert!((d.ext[0] - 1.0).abs() < 1e-5);
        assert!((d.ext[2] - 1.5).abs() < 1e-5);
        assert!((d.radius() - 1.5).abs() < 1e-5);
        let c = d.corners();
        let (min_x, max_x) = c.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p[0]), hi.max(p[0]))
        });
        // Turned a quarter, the z extent (3) lies along x.
        assert!((max_x - min_x - 3.0).abs() < 1e-4, "{min_x} {max_x}");
    }

    #[test]
    fn changing_the_world_refreshes_the_boxes_and_rows() {
        let mut w = EditorWorld::demo(3);
        let e = w.find("Cube.001").unwrap();
        w.world_mut().despawn(e);
        w.refresh();
        assert_eq!(w.box_count(), 2);
        assert_eq!(w.row_count(), 2);
        assert!(w.find("Cube.001").is_none());
    }

    #[test]
    fn refresh_without_changes_keeps_the_same_boxes() {
        let mut w = EditorWorld::demo(5);
        let before = w.drawables().to_vec();
        w.refresh();
        assert_eq!(w.drawables(), &before[..]);
    }

    #[test]
    fn names_fall_back_to_the_entity_number() {
        let mut w = EditorWorld::demo(0);
        let e = w.world_mut().spawn();
        assert!(w.name_of(e).starts_with("Entity "));
    }

    // ── play mode ──────────────────────────────────────────────────────

    #[test]
    fn stepping_moves_the_bobbing_entities_and_the_children_follow_their_parent() {
        let mut w = EditorWorld::demo(6);
        let rest = rest_positions(&w);
        w.enter_play();
        assert!(w.is_playing());
        for _ in 0..30 {
            w.step(1.0 / 60.0);
        }
        w.refresh();
        let moved = rest_positions(&w);
        assert_ne!(rest, moved);
        assert!(w.play_time() > 0.4);
        // Cube.001 is a child of Cube: it keeps its offset from the parent's
        // origin, so it is never farther than that offset plus its own bob.
        let parent = w.position_of(w.find("Cube").unwrap()).unwrap();
        let child = w.position_of(w.find("Cube.001").unwrap()).unwrap();
        let rest_offset = ((2.5f32).powi(2)).sqrt();
        let dist = ((parent[0] - child[0]).powi(2)
            + (parent[1] - child[1]).powi(2)
            + (parent[2] - child[2]).powi(2))
        .sqrt();
        assert!((dist - rest_offset).abs() < 0.7, "child {dist} from parent");
    }

    #[test]
    fn step_does_nothing_unless_playing() {
        let mut w = EditorWorld::demo(4);
        let rest = rest_positions(&w);
        w.step(0.5);
        w.refresh();
        assert_eq!(rest_positions(&w), rest);
        assert_eq!(w.play_time(), 0.0);
    }

    #[test]
    fn exit_play_puts_every_position_and_rotation_back() {
        let mut w = EditorWorld::demo(8);
        let before = w.drawables().to_vec();
        w.enter_play();
        for _ in 0..90 {
            w.step(1.0 / 60.0);
        }
        let report = w.exit_play().unwrap();
        assert_eq!(report.restored, 8);
        assert_eq!((report.removed, report.lost, report.failed), (0, 0, 0));
        assert!(!w.is_playing());
        assert_eq!(w.play_time(), 0.0);
        let mut after = w.drawables().to_vec();
        let mut expected = before;
        after.sort_by_key(|d| d.entity.as_ffi());
        expected.sort_by_key(|d| d.entity.as_ffi());
        assert_eq!(after, expected);
    }

    #[test]
    fn entering_play_twice_keeps_the_first_snapshot() {
        let mut w = EditorWorld::demo(4);
        let before = rest_positions(&w);
        w.enter_play();
        for _ in 0..30 {
            w.step(1.0 / 60.0);
        }
        w.enter_play();
        w.exit_play().unwrap();
        assert_eq!(rest_positions(&w), before);
        assert!(w.exit_play().is_none(), "not playing any more");
    }

    #[test]
    fn restart_play_starts_from_time_zero() {
        let mut w = EditorWorld::demo(2);
        w.enter_play();
        w.step(0.5);
        assert!(w.play_time() > 0.4);
        w.exit_play();
        w.enter_play();
        assert_eq!(w.play_time(), 0.0);
    }

    #[test]
    fn exit_play_despawns_what_play_spawned() {
        let mut w = EditorWorld::demo(3);
        w.enter_play();
        let extra = w.world_mut().spawn();
        let kid = w.world_mut().spawn();
        w.world_mut().set_parent(kid, extra).unwrap();
        w.step(0.1);
        let report = w.exit_play().unwrap();
        assert_eq!(report.removed, 2);
        assert_eq!(w.world().entity_count(), 3);
        assert!(!w.world().is_alive(extra));
        assert!(!w.world().is_alive(kid));
    }

    #[test]
    fn exit_play_undoes_edits_to_components_added_and_removed_during_play() {
        let mut w = EditorWorld::demo(3);
        let e = w.find("Cube.002").unwrap();
        w.enter_play();
        {
            let world = w.world_mut();
            world.remove_static::<Bob>(e);
            world.remove_static::<BoxShape>(e);
            world.insert_static(e, Name("renamed".to_string()));
        }
        let report = w.exit_play().unwrap();
        assert_eq!(report.failed, 0);
        assert!(w.world().has_static::<Bob>(e));
        assert!(w.world().has_static::<BoxShape>(e));
        assert_eq!(w.name_of(e), "Cube.002");
    }

    #[test]
    fn exit_play_removes_a_component_that_play_added() {
        let mut w = EditorWorld::demo(2);
        let e = w.find("Cube.001").unwrap();
        w.world_mut().remove_static::<Bob>(e);
        w.enter_play();
        w.world_mut().insert_static(
            e,
            Bob {
                base_y: 0.0,
                phase: 0.0,
            },
        );
        assert!(w.world().has_static::<Bob>(e));
        let report = w.exit_play().unwrap();
        assert_eq!(report.failed, 0);
        assert!(
            !w.world().has_static::<Bob>(e),
            "play gave it a Bob; edit takes it back"
        );
    }

    #[test]
    fn exit_play_restores_the_hierarchy_even_when_play_reparented_it() {
        let mut w = EditorWorld::demo(6);
        let rows_before = names(&w, 0, 10);
        let root = w.find("Cube").unwrap();
        let two = w.find("Cube.002").unwrap();
        let four = w.find("Cube.004").unwrap();
        w.enter_play();
        {
            let world = w.world_mut();
            // Swap a parent and child: root under Cube.002 is a cycle, so
            // first free Cube.002, then hang the old root under it.
            world.remove_parent(two);
            world.set_parent(root, two).unwrap();
            world.set_parent(four, root).unwrap();
        }
        w.refresh();
        assert_ne!(names(&w, 0, 10), rows_before);
        let report = w.exit_play().unwrap();
        assert_eq!(report.failed, 0);
        assert_eq!(names(&w, 0, 10), rows_before);
        assert_eq!(w.world().parent(two), Some(root));
        assert_eq!(w.world().parent(four), None);
    }

    #[test]
    fn a_child_despawned_during_play_is_counted_as_lost_not_resurrected() {
        let mut w = EditorWorld::demo(4);
        let three = w.find("Cube.003").unwrap();
        w.enter_play();
        w.world_mut().despawn(three);
        let report = w.exit_play().unwrap();
        assert_eq!(report.lost, 1);
        assert_eq!(report.restored, 3);
        assert_eq!(w.box_count(), 3);
        assert_eq!(w.row_count(), 3);
    }

    #[test]
    fn inspect_lists_the_selected_entitys_components() {
        let w = EditorWorld::demo(3);
        let rows = w.inspect(w.find("Cube.001").unwrap());
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        for want in ["Name", "LocalTransform", "ChildOf", "BoxShape", "Bob"] {
            assert!(labels.contains(&want), "{want} in {labels:?}");
        }
    }
}
