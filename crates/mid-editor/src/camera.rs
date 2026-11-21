// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "camera.rs"
// ============================================================================
//! The viewport cameras and their projection onto the screen.

use mid_math::f32::{Mat4, Vec3};

/// A screen rectangle in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Area {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Area {
    /// The center of the rectangle.
    pub fn center(&self) -> [f32; 2] {
        [self.x + self.w * 0.5, self.y + self.h * 0.5]
    }

    /// True when the point is inside the rectangle.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// An orbit camera: it looks at `target` from `dist` away.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    /// Turn about the vertical axis, in radians.
    pub yaw: f32,
    /// Angle above the ground plane, in radians, kept inside about +-86 degrees.
    pub pitch: f32,
    /// Distance from the target.
    pub dist: f32,
    /// The point looked at.
    pub target: [f32; 3],
}

/// Vertical field of view in radians.
pub const FOV_Y: f32 = 1.0;
/// Nearest distance drawn.
pub const NEAR: f32 = 0.1;
const PITCH_LIMIT: f32 = 1.5;

impl Default for Orbit {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.5,
            dist: 14.0,
            target: [0.0, 0.0, 0.0],
        }
    }
}

impl Orbit {
    /// The camera position.
    pub fn eye(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.target[0] + self.dist * cp * self.yaw.sin(),
            self.target[1] + self.dist * self.pitch.sin(),
            self.target[2] + self.dist * cp * self.yaw.cos(),
        ]
    }

    /// Turns the camera by a drag of `dx`, `dy` pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Moves in or out. A ratio above 1 moves in.
    pub fn zoom(&mut self, ratio: f32) {
        if ratio.is_finite() && ratio > 0.0 {
            self.dist = (self.dist / ratio).clamp(2.0, 400.0);
        }
    }

    /// Slides the target along the ground by a drag of `dx`, `dy` pixels, the
    /// scene following the fingers.
    pub fn pan(&mut self, dx: f32, dy: f32, area_h: f32) {
        let per_px = 2.0 * self.dist * (FOV_Y * 0.5).tan() / area_h.max(1.0);
        let (s, c) = self.yaw.sin_cos();
        let (right, forward) = ([c, -s], [-s, -c]);
        self.target[0] += (-right[0] * dx + forward[0] * -dy) * per_px;
        self.target[2] += (-right[1] * dx + forward[1] * -dy) * per_px;
    }
}

/// A top-down orthographic camera over the ground plane (x to the right, z down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ortho {
    /// The ground point at the center of the screen.
    pub center: [f32; 2],
    /// Pixels per world unit.
    pub scale: f32,
}

impl Default for Ortho {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            scale: 40.0,
        }
    }
}

impl Ortho {
    /// Zooms in or out. A ratio above 1 zooms in.
    pub fn zoom(&mut self, ratio: f32) {
        if ratio.is_finite() && ratio > 0.0 {
            self.scale = (self.scale * ratio).clamp(4.0, 600.0);
        }
    }

    /// Slides the view by a drag of `dx`, `dy` pixels, the scene following the fingers.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.center[0] -= dx / self.scale;
        self.center[1] -= dy / self.scale;
    }

    /// A ground point to a screen point.
    pub fn project(&self, area: Area, p: [f32; 3]) -> [f32; 2] {
        let c = area.center();
        [
            c[0] + (p[0] - self.center[0]) * self.scale,
            c[1] + (p[2] - self.center[1]) * self.scale,
        ]
    }
}

/// A perspective projection for one frame: the view matrix, and the focal
/// length in pixels for the area it draws into.
#[derive(Clone, Copy)]
pub struct Perspective {
    view: Mat4,
    focal: f32,
    center: [f32; 2],
}

impl Perspective {
    /// Builds the projection of `orbit` into `area`.
    pub fn new(orbit: &Orbit, area: Area) -> Self {
        let e = orbit.eye();
        let t = orbit.target;
        let view = Mat4::look_at_rh(
            Vec3::new(e[0], e[1], e[2]),
            Vec3::new(t[0], t[1], t[2]),
            Vec3::new(0.0, 1.0, 0.0),
        );
        Self {
            view,
            focal: area.h * 0.5 / (FOV_Y * 0.5).tan(),
            center: area.center(),
        }
    }

    fn view_point(&self, p: [f32; 3]) -> [f32; 3] {
        let v = self.view.transform_point(Vec3::new(p[0], p[1], p[2]));
        [v.x, v.y, v.z]
    }

    fn screen_point(&self, v: [f32; 3]) -> [f32; 2] {
        let k = self.focal / -v[2];
        [self.center[0] + v[0] * k, self.center[1] - v[1] * k]
    }

    /// A world point to a screen point and its distance in front of the camera,
    /// or `None` when it is behind the near plane.
    pub fn project(&self, p: [f32; 3]) -> Option<([f32; 2], f32)> {
        let v = self.view_point(p);
        if -v[2] < NEAR {
            return None;
        }
        Some((self.screen_point(v), -v[2]))
    }

    /// The pixels a world length of `len` covers at distance `depth`.
    pub fn pixels(&self, len: f32, depth: f32) -> f32 {
        len * self.focal / depth.max(NEAR)
    }

    /// A world line to a screen line, cut at the near plane. `None` when the
    /// whole line is behind the camera.
    pub fn line(&self, a: [f32; 3], b: [f32; 3]) -> Option<([f32; 2], [f32; 2])> {
        let (mut va, mut vb) = (self.view_point(a), self.view_point(b));
        let (da, db) = (-va[2] - NEAR, -vb[2] - NEAR);
        if da < 0.0 && db < 0.0 {
            return None;
        }
        if da < 0.0 || db < 0.0 {
            let t = da / (da - db);
            let cut = [
                va[0] + (vb[0] - va[0]) * t,
                va[1] + (vb[1] - va[1]) * t,
                -NEAR,
            ];
            if da < 0.0 {
                va = cut;
            } else {
                vb = cut;
            }
        }
        Some((self.screen_point(va), self.screen_point(vb)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AREA: Area = Area {
        x: 0.0,
        y: 100.0,
        w: 400.0,
        h: 300.0,
    };

    #[test]
    fn area_center_and_containment() {
        assert_eq!(AREA.center(), [200.0, 250.0]);
        assert!(AREA.contains(0.0, 100.0));
        assert!(!AREA.contains(400.0, 150.0));
        assert!(!AREA.contains(10.0, 99.0));
    }

    #[test]
    fn the_target_projects_to_the_center_of_the_area() {
        let o = Orbit::default();
        let p = Perspective::new(&o, AREA);
        let (s, depth) = p.project(o.target).unwrap();
        assert!(
            (s[0] - 200.0).abs() < 1e-2 && (s[1] - 250.0).abs() < 1e-2,
            "{s:?}"
        );
        assert!((depth - o.dist).abs() < 1e-3);
    }

    #[test]
    fn a_point_to_the_right_of_the_view_projects_right_of_center() {
        let o = Orbit {
            yaw: 0.0,
            pitch: 0.0,
            dist: 10.0,
            target: [0.0; 3],
        };
        let p = Perspective::new(&o, AREA);
        let (right, _) = p.project([2.0, 0.0, 0.0]).unwrap();
        let (up, _) = p.project([0.0, 2.0, 0.0]).unwrap();
        assert!(right[0] > 200.0 && (right[1] - 250.0).abs() < 1e-2);
        assert!(
            up[1] < 250.0 && (up[0] - 200.0).abs() < 1e-2,
            "up is smaller y on screen"
        );
    }

    #[test]
    fn farther_points_are_closer_to_the_center() {
        let o = Orbit {
            yaw: 0.0,
            pitch: 0.0,
            dist: 10.0,
            target: [0.0; 3],
        };
        let p = Perspective::new(&o, AREA);
        let (near, _) = p.project([2.0, 0.0, 4.0]).unwrap();
        let (far, _) = p.project([2.0, 0.0, -4.0]).unwrap();
        assert!(near[0] - 200.0 > far[0] - 200.0);
    }

    #[test]
    fn points_behind_the_camera_are_refused() {
        let o = Orbit {
            yaw: 0.0,
            pitch: 0.0,
            dist: 10.0,
            target: [0.0; 3],
        };
        let p = Perspective::new(&o, AREA);
        assert!(p.project([0.0, 0.0, 50.0]).is_none());
        assert!(p.line([0.0, 0.0, 50.0], [1.0, 0.0, 60.0]).is_none());
    }

    #[test]
    fn a_line_through_the_near_plane_is_cut_and_stays_finite() {
        let o = Orbit {
            yaw: 0.0,
            pitch: 0.0,
            dist: 10.0,
            target: [0.0; 3],
        };
        let p = Perspective::new(&o, AREA);
        let (a, b) = p.line([0.0, 0.0, 0.0], [0.0, 0.0, 100.0]).unwrap();
        assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()));
        let both = p.line([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert!(both.0[0] < both.1[0]);
    }

    #[test]
    fn orbit_turns_and_clamps_the_pitch() {
        let mut o = Orbit::default();
        let yaw = o.yaw;
        o.orbit(100.0, 0.0);
        assert!(o.yaw < yaw);
        o.orbit(0.0, 1.0e6);
        assert!((o.pitch - PITCH_LIMIT).abs() < 1e-6);
        o.orbit(0.0, -1.0e6);
        assert!((o.pitch + PITCH_LIMIT).abs() < 1e-6);
        assert!(o.eye().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zoom_clamps_and_ignores_bad_ratios() {
        let mut o = Orbit::default();
        o.zoom(2.0);
        assert!((o.dist - 7.0).abs() < 1e-5);
        o.zoom(1.0e9);
        assert_eq!(o.dist, 2.0);
        o.zoom(1.0e-9);
        assert_eq!(o.dist, 400.0);
        let before = o;
        o.zoom(f32::NAN);
        o.zoom(0.0);
        o.zoom(-2.0);
        assert_eq!(o, before);
    }

    #[test]
    fn pan_moves_the_target_along_the_ground_only() {
        let mut o = Orbit {
            yaw: 0.0,
            pitch: 0.5,
            dist: 10.0,
            target: [0.0, 0.0, 0.0],
        };
        o.pan(100.0, 0.0, 300.0);
        assert!(o.target[0] < 0.0 && o.target[2].abs() < 1e-6 && o.target[1] == 0.0);
        let mut o = Orbit {
            yaw: 0.0,
            pitch: 0.5,
            dist: 10.0,
            target: [0.0; 3],
        };
        o.pan(0.0, 100.0, 300.0);
        assert!(
            o.target[0].abs() < 1e-6 && o.target[2] > 0.0,
            "{:?}",
            o.target
        );
    }

    #[test]
    fn ortho_zoom_pan_and_project() {
        let mut c = Ortho::default();
        assert_eq!(c.project(AREA, [0.0, 0.0, 0.0]), [200.0, 250.0]);
        assert_eq!(c.project(AREA, [1.0, 9.0, 2.0]), [240.0, 330.0]);
        c.pan(40.0, 0.0);
        assert_eq!(c.center, [-1.0, 0.0]);
        c.zoom(2.0);
        assert_eq!(c.scale, 80.0);
        c.zoom(1.0e9);
        assert_eq!(c.scale, 600.0);
        c.zoom(f32::NAN);
        assert_eq!(c.scale, 600.0);
    }

    #[test]
    fn pixels_shrink_with_distance() {
        let o = Orbit::default();
        let p = Perspective::new(&o, AREA);
        assert!(p.pixels(1.0, 5.0) > p.pixels(1.0, 10.0));
        assert!(p.pixels(1.0, 0.0).is_finite());
    }
}
