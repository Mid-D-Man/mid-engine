// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "viewport.rs"
// ============================================================================
//! The scene viewport: a 3D orbit view and a top-down 2D view, both drawn as
//! lines and rectangles projected on the CPU.

use crate::camera::{Area, Orbit, Ortho, Perspective};
use crate::paint::{rgba, PaintList};
use crate::scene::{Entity, Scene, EDGES};

const GRID_HALF: i32 = 10;
const FAR: f32 = 300.0;
const MINOR: u32 = rgba(255, 255, 255, 30);
const AXIS_X: u32 = rgba(229, 115, 115, 200);
const AXIS_Z: u32 = rgba(100, 181, 246, 200);
/// The color of the selection outline.
pub const SELECTED: u32 = rgba(255, 213, 79, 255);

/// True when `value` beats the best candidate so far, or there is none yet.
fn is_better(best: Option<(usize, f32)>, value: f32) -> bool {
    match best {
        Some((_, b)) => value < b,
        None => true,
    }
}

/// The viewport's state: which view, and where each camera is.
#[derive(Clone, Copy, Debug, Default)]
pub struct Viewport {
    /// True for the top-down 2D view.
    pub mode2d: bool,
    /// The 3D camera.
    pub orbit: Orbit,
    /// The 2D camera.
    pub ortho: Ortho,
}

impl Viewport {
    /// One finger moved: turns the 3D camera, or slides the 2D view.
    pub fn drag(&mut self, dx: f32, dy: f32, area_h: f32) {
        if self.mode2d {
            let _ = area_h;
            self.ortho.pan(dx, dy);
        } else {
            self.orbit.orbit(dx, dy);
        }
    }

    /// Two fingers moved: zooms, and slides the view with their midpoint.
    pub fn pinch(&mut self, ratio: f32, dx: f32, dy: f32, area_h: f32) {
        if self.mode2d {
            self.ortho.zoom(ratio);
            self.ortho.pan(dx, dy);
        } else {
            self.orbit.zoom(ratio);
            self.orbit.pan(dx, dy, area_h);
        }
    }

    /// Draws the grid, the scene and the selection into `out`, inside `area`.
    /// `unit` scales line widths with the screen.
    pub fn paint(
        &self,
        scene: &Scene,
        area: Area,
        selected: Option<usize>,
        unit: f32,
        out: &mut PaintList,
    ) {
        let lw = (unit * 0.5).max(1.0);
        if self.mode2d {
            self.paint_2d(scene, area, selected, lw, out);
        } else {
            self.paint_3d(scene, area, selected, lw, out);
        }
    }

    fn paint_3d(
        &self,
        scene: &Scene,
        area: Area,
        selected: Option<usize>,
        lw: f32,
        out: &mut PaintList,
    ) {
        let cam = Perspective::new(&self.orbit, area);
        let g = GRID_HALF as f32;
        for i in -GRID_HALF..=GRID_HALF {
            let f = i as f32;
            let (cx, cz) = if i == 0 {
                (AXIS_Z, AXIS_X)
            } else {
                (MINOR, MINOR)
            };
            if let Some((a, b)) = cam.line([f, 0.0, -g], [f, 0.0, g]) {
                out.line(a[0], a[1], b[0], b[1], lw, cx);
            }
            if let Some((a, b)) = cam.line([-g, 0.0, f], [g, 0.0, f]) {
                out.line(a[0], a[1], b[0], b[1], lw, cz);
            }
        }
        for (i, e) in scene.entities.iter().enumerate() {
            if selected == Some(i) {
                continue;
            }
            self.draw_box_3d(&cam, area, e, e.color, lw, out);
        }
        if let Some(e) = selected.and_then(|i| scene.entities.get(i)) {
            self.draw_box_3d(&cam, area, e, SELECTED, lw * 2.0, out);
        }
    }

    fn draw_box_3d(
        &self,
        cam: &Perspective,
        area: Area,
        e: &Entity,
        color: u32,
        lw: f32,
        out: &mut PaintList,
    ) {
        let Some((c, depth)) = cam.project(e.pos) else {
            return;
        };
        if depth > FAR {
            return;
        }
        let r = cam.pixels(e.half[0].max(e.half[1]).max(e.half[2]) * 1.8, depth);
        if c[0] + r < area.x
            || c[0] - r > area.x + area.w
            || c[1] + r < area.y
            || c[1] - r > area.y + area.h
        {
            return;
        }
        if r < 3.0 {
            out.rect(c[0] - 1.0, c[1] - 1.0, 2.0, 2.0, color, 0.0);
            return;
        }
        let corners = Scene::corners(e);
        for (a, b) in EDGES {
            if let Some((p, q)) = cam.line(corners[a], corners[b]) {
                out.line(p[0], p[1], q[0], q[1], lw, color);
            }
        }
    }

    fn paint_2d(
        &self,
        scene: &Scene,
        area: Area,
        selected: Option<usize>,
        lw: f32,
        out: &mut PaintList,
    ) {
        let cam = &self.ortho;
        let step = if cam.scale >= 12.0 { 1.0 } else { 5.0 };
        let (hw, hh) = (area.w * 0.5 / cam.scale, area.h * 0.5 / cam.scale);
        let (x0, x1) = ((cam.center[0] - hw) / step, (cam.center[0] + hw) / step);
        let (z0, z1) = ((cam.center[1] - hh) / step, (cam.center[1] + hh) / step);
        let mut i = x0.floor() as i32;
        while (i as f32) <= x1.ceil() && i - (x0.floor() as i32) < 400 {
            let w = i as f32 * step;
            let top = cam.project(area, [w, 0.0, cam.center[1] - hh]);
            let bottom = cam.project(area, [w, 0.0, cam.center[1] + hh]);
            out.line(
                top[0],
                top[1],
                bottom[0],
                bottom[1],
                lw,
                if i == 0 { AXIS_Z } else { MINOR },
            );
            i += 1;
        }
        let mut j = z0.floor() as i32;
        while (j as f32) <= z1.ceil() && j - (z0.floor() as i32) < 400 {
            let w = j as f32 * step;
            let left = cam.project(area, [cam.center[0] - hw, 0.0, w]);
            let right = cam.project(area, [cam.center[0] + hw, 0.0, w]);
            out.line(
                left[0],
                left[1],
                right[0],
                right[1],
                lw,
                if j == 0 { AXIS_X } else { MINOR },
            );
            j += 1;
        }
        for (k, e) in scene.entities.iter().enumerate() {
            let c = cam.project(area, e.pos);
            let (rx, rz) = (e.half[0] * cam.scale, e.half[2] * cam.scale);
            if c[0] + rx < area.x
                || c[0] - rx > area.x + area.w
                || c[1] + rz < area.y
                || c[1] - rz > area.y + area.h
            {
                continue;
            }
            let sel = selected == Some(k);
            if rx.max(rz) < 1.5 {
                out.rect(c[0] - 1.0, c[1] - 1.0, 2.0, 2.0, e.color, 0.0);
                continue;
            }
            let fill = (e.color & 0xFFFF_FF00) | 0x50;
            out.rect(c[0] - rx, c[1] - rz, rx * 2.0, rz * 2.0, fill, 0.0);
            let edge = if sel { SELECTED } else { e.color };
            let w = if sel { lw * 2.0 } else { lw };
            let (l, r, t, b) = (c[0] - rx, c[0] + rx, c[1] - rz, c[1] + rz);
            out.line(l, t, r, t, w, edge);
            out.line(r, t, r, b, w, edge);
            out.line(r, b, l, b, w, edge);
            out.line(l, b, l, t, w, edge);
        }
    }

    /// The entity under a touch at `(x, y)`, within `slop` pixels, nearest the
    /// camera when several qualify.
    pub fn pick(&self, scene: &Scene, area: Area, x: f32, y: f32, slop: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        if self.mode2d {
            for (k, e) in scene.entities.iter().enumerate() {
                let c = self.ortho.project(area, e.pos);
                let (rx, rz) = (
                    e.half[0] * self.ortho.scale + slop,
                    e.half[2] * self.ortho.scale + slop,
                );
                if (x - c[0]).abs() <= rx && (y - c[1]).abs() <= rz {
                    let d = (x - c[0]).powi(2) + (y - c[1]).powi(2);
                    if is_better(best, d) {
                        best = Some((k, d));
                    }
                }
            }
        } else {
            let cam = Perspective::new(&self.orbit, area);
            for (k, e) in scene.entities.iter().enumerate() {
                let Some((c, depth)) = cam.project(e.pos) else {
                    continue;
                };
                if depth > FAR {
                    continue;
                }
                let r = cam
                    .pixels(e.half[0].max(e.half[1]).max(e.half[2]), depth)
                    .max(slop);
                if (x - c[0]).powi(2) + (y - c[1]).powi(2) <= r * r && is_better(best, depth) {
                    best = Some((k, depth));
                }
            }
        }
        best.map(|(k, _)| k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::{KIND_LINE, KIND_RECT};

    const AREA: Area = Area {
        x: 0.0,
        y: 0.0,
        w: 400.0,
        h: 400.0,
    };

    fn count(p: &PaintList, kind: u32) -> usize {
        p.items.iter().filter(|i| i.kind == kind).count()
    }

    #[test]
    fn an_empty_3d_scene_still_draws_the_grid() {
        let mut out = PaintList::new();
        Viewport::default().paint(&Scene::default(), AREA, None, 2.0, &mut out);
        assert_eq!(count(&out, KIND_LINE), 42, "21 lines each way");
    }

    #[test]
    fn each_visible_cube_adds_twelve_edges() {
        let mut out = PaintList::new();
        Viewport::default().paint(&Scene::demo(1), AREA, None, 2.0, &mut out);
        assert_eq!(count(&out, KIND_LINE), 42 + 12);
    }

    #[test]
    fn the_selected_cube_is_drawn_last_in_the_selection_color_and_thicker() {
        let mut out = PaintList::new();
        Viewport::default().paint(&Scene::demo(4), AREA, Some(2), 2.0, &mut out);
        let last = out.items.last().unwrap();
        assert_eq!(last.color, SELECTED);
        let first_cube_line = out
            .items
            .iter()
            .find(|i| {
                i.kind == KIND_LINE && i.color != MINOR && i.color != AXIS_X && i.color != AXIS_Z
            })
            .unwrap();
        assert!(last.extra > first_cube_line.extra);
        assert_eq!(out.items.iter().filter(|i| i.color == SELECTED).count(), 12);
    }

    #[test]
    fn tiny_far_cubes_become_dots() {
        let mut out = PaintList::new();
        let v = Viewport {
            orbit: Orbit {
                dist: 250.0,
                ..Orbit::default()
            },
            ..Viewport::default()
        };
        v.paint(&Scene::demo(1), AREA, None, 2.0, &mut out);
        assert_eq!(count(&out, KIND_RECT), 1);
        assert_eq!(count(&out, KIND_LINE), 42);
    }

    #[test]
    fn cubes_off_screen_and_behind_the_camera_are_skipped() {
        let v = Viewport {
            orbit: Orbit {
                target: [500.0, 0.0, 500.0],
                dist: 10.0,
                ..Orbit::default()
            },
            ..Viewport::default()
        };
        let mut out = PaintList::new();
        v.paint(&Scene::demo(9), AREA, None, 2.0, &mut out);
        assert!(out.items.iter().all(|i| i.a.is_finite() && i.b.is_finite()));
        assert_eq!(count(&out, KIND_RECT), 0);
        assert!(
            out.items
                .iter()
                .all(|i| i.color == MINOR || i.color == AXIS_X || i.color == AXIS_Z),
            "only grid lines may remain, never a cube"
        );
    }

    #[test]
    fn two_d_draws_a_grid_and_a_rectangle_per_cube() {
        let v = Viewport {
            mode2d: true,
            ..Viewport::default()
        };
        let mut out = PaintList::new();
        v.paint(&Scene::demo(2), AREA, None, 2.0, &mut out);
        assert!(count(&out, KIND_LINE) > 20 + 8);
        assert_eq!(count(&out, KIND_RECT), 2);
    }

    #[test]
    fn two_d_zoomed_far_out_uses_a_coarser_grid_and_stays_bounded() {
        let v = Viewport {
            mode2d: true,
            ortho: Ortho {
                scale: 4.0,
                ..Ortho::default()
            },
            ..Viewport::default()
        };
        let mut out = PaintList::new();
        v.paint(&Scene::default(), AREA, None, 2.0, &mut out);
        assert!(count(&out, KIND_LINE) < 100, "{}", count(&out, KIND_LINE));
    }

    #[test]
    fn dragging_orbits_in_3d_and_pans_in_2d() {
        let mut v = Viewport::default();
        let yaw = v.orbit.yaw;
        v.drag(50.0, 0.0, 400.0);
        assert_ne!(v.orbit.yaw, yaw);
        assert_eq!(v.ortho, Ortho::default());
        v.mode2d = true;
        v.drag(40.0, 0.0, 400.0);
        assert_ne!(v.ortho.center, [0.0, 0.0]);
    }

    #[test]
    fn pinching_zooms_the_active_camera() {
        let mut v = Viewport::default();
        let d = v.orbit.dist;
        v.pinch(2.0, 0.0, 0.0, 400.0);
        assert!(v.orbit.dist < d);
        v.mode2d = true;
        let s = v.ortho.scale;
        v.pinch(2.0, 0.0, 0.0, 400.0);
        assert!(v.ortho.scale > s);
    }

    #[test]
    fn picking_finds_the_cube_under_a_touch_and_nothing_elsewhere() {
        let v = Viewport::default();
        let scene = Scene::demo(1);
        let cam = Perspective::new(&v.orbit, AREA);
        let (c, _) = cam.project(scene.entities[0].pos).unwrap();
        assert_eq!(v.pick(&scene, AREA, c[0], c[1], 10.0), Some(0));
        assert_eq!(v.pick(&scene, AREA, 5.0, 5.0, 10.0), None);
    }

    #[test]
    fn picking_prefers_the_nearer_cube_when_two_overlap() {
        let mut scene = Scene::demo(2);
        scene.entities[0].pos = [0.0, 0.5, 0.0];
        scene.entities[1].pos = [0.0, 0.5, 6.0];
        let v = Viewport {
            orbit: Orbit {
                yaw: 0.0,
                pitch: 0.0,
                dist: 20.0,
                target: [0.0, 0.5, 0.0],
            },
            ..Viewport::default()
        };
        let cam = Perspective::new(&v.orbit, AREA);
        let (c, _) = cam.project(scene.entities[0].pos).unwrap();
        assert_eq!(
            v.pick(&scene, AREA, c[0], c[1], 30.0),
            Some(1),
            "entity 1 is nearer the camera"
        );
    }

    #[test]
    fn picking_in_2d_uses_the_rectangle() {
        let v = Viewport {
            mode2d: true,
            ..Viewport::default()
        };
        let scene = Scene::demo(1);
        let c = v.ortho.project(AREA, scene.entities[0].pos);
        assert_eq!(v.pick(&scene, AREA, c[0] + 10.0, c[1] - 10.0, 0.0), Some(0));
        assert_eq!(v.pick(&scene, AREA, c[0] + 100.0, c[1], 0.0), None);
    }
}
