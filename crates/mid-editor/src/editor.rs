// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "editor.rs"
// ============================================================================
//! The editor: panels on `mid-ui`, the viewport, touch routing and the paint
//! list for one frame.

use std::time::Instant;

use mid_ui::{draw, layout, Color32, DrawCmd, Flow, NodeId, Size, Sizing, Style, Tree};

use crate::camera::Area;
use crate::input::{Gesture, Phase, Tracker};
use crate::paint::{rgba, PaintList};
use crate::scene::Scene;
use crate::stats::Stats;
use crate::text::{fit, TextMeasure};
use crate::viewport::{Viewport, SELECTED};

const BG: u32 = rgba(0x1B, 0x1B, 0x1F, 0xFF);
const BAR: u32 = rgba(0x26, 0x26, 0x2C, 0xFF);
const VIEW_BG: u32 = rgba(0x15, 0x16, 0x1A, 0xFF);
const PANEL: u32 = rgba(0x20, 0x20, 0x25, 0xFF);
const BUTTON: u32 = rgba(0x33, 0x33, 0x3C, 0xFF);
const ACTIVE: u32 = rgba(0x3D, 0x5A, 0x99, 0xFF);
const ROW_SELECTED: u32 = rgba(0x3D, 0x5A, 0x99, 0xB0);
const TEXT: u32 = rgba(0xE6, 0xE6, 0xEA, 0xFF);
const DIM: u32 = rgba(0x9A, 0x9A, 0xA5, 0xFF);

/// Pixels per UI unit for a screen whose short side is 411 pixels. A Galaxy
/// A13 is 1080 pixels across, so a unit is about 2.6 pixels there, close to
/// one Android dp.
const REFERENCE_SHORT_SIDE: f32 = 411.0;
const CUBE_COUNTS: [usize; 4] = [12, 200, 1000, 5000];
const LOG_LINES: usize = 40;

/// Whether the world is being edited or run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The world is still; it is being edited.
    Edit,
    /// The world is stepping.
    Play,
}

/// The panels in the tab strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    /// The entity tree.
    Hierarchy,
    /// The selected entity's properties.
    Inspector,
    /// The project's files.
    Project,
    /// Statistics and the log.
    Console,
}

impl Tab {
    /// Every tab, in strip order.
    pub const ALL: [Tab; 4] = [Tab::Hierarchy, Tab::Inspector, Tab::Project, Tab::Console];

    fn label(self) -> &'static str {
        match self {
            Tab::Hierarchy => "Hierarchy",
            Tab::Inspector => "Inspector",
            Tab::Project => "Project",
            Tab::Console => "Console",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action {
    SetMode(Mode),
    Set2D(bool),
    SetTab(Tab),
    Select(usize),
    SetCount(usize),
}

/// Where the regions of the screen are, in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Areas {
    /// The bar with the mode buttons.
    pub top: Area,
    /// The scene viewport.
    pub viewport: Area,
    /// The tab strip.
    pub tabs: Area,
    /// The panel under (or beside) the viewport.
    pub panel: Area,
}

#[derive(Clone, Copy)]
struct Ids {
    top: NodeId,
    viewport: NodeId,
    tabs: NodeId,
    panel: NodeId,
}

/// A small drawing context: where to paint, how to measure text, where taps land.
struct Ui<'a> {
    out: &'a mut PaintList,
    measure: &'a dyn TextMeasure,
    hot: &'a mut Vec<(Area, Action)>,
    unit: f32,
}

impl Ui<'_> {
    fn text(&mut self, x: f32, y: f32, size: f32, color: u32, s: &str) {
        self.out.text(x, y, size, color, s);
    }

    fn text_fit(&mut self, x: f32, y: f32, size: f32, color: u32, max_w: f32, s: &str) {
        let shown = fit(self.measure, s, size, max_w);
        self.out.text(x, y, size, color, &shown);
    }

    fn centered(&mut self, a: Area, size: f32, color: u32, s: &str) {
        let shown = fit(self.measure, s, size, a.w);
        let w = self.measure.width(&shown, size);
        self.out.text(
            a.x + (a.w - w) * 0.5,
            a.y + (a.h - size) * 0.5,
            size,
            color,
            &shown,
        );
    }

    fn button(&mut self, a: Area, label: &str, active: bool, action: Action) {
        self.out.rect(
            a.x,
            a.y,
            a.w,
            a.h,
            if active { ACTIVE } else { BUTTON },
            8.0 * self.unit,
        );
        self.centered(a, 15.0 * self.unit, TEXT, label);
        self.hot.push((a, action));
    }
}

/// How many parents an entity has, up to a limit that guards a cycle.
fn depth_of(scene: &Scene, mut i: usize) -> usize {
    let mut d = 0;
    while let Some(p) = scene.entities.get(i).and_then(|e| e.parent) {
        d += 1;
        i = p;
        if d >= 8 {
            break;
        }
    }
    d
}

fn area_of(tree: &Tree<f32>, id: NodeId) -> Area {
    tree.rect(id)
        .map(|r| Area {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        })
        .unwrap_or_default()
}

fn color(rgba: u32) -> Color32 {
    Color32::from_u32_rgba(rgba)
}

fn now() -> Option<Instant> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(Instant::now())
    }
}

/// The editor core. A host owns one, feeds it pointer events and frame times,
/// and paints [`Editor::paint`] after each [`Editor::frame`].
pub struct Editor {
    tree: Tree<f32>,
    ids: Ids,
    measure: Box<dyn TextMeasure>,
    size: (f32, f32),
    unit: f32,
    mode: Mode,
    tab: Tab,
    scene: Scene,
    selected: Option<usize>,
    viewport: Viewport,
    tracker: Tracker,
    stats: Stats,
    play_time: f32,
    time: f32,
    paint: PaintList,
    hot: Vec<(Area, Action)>,
    cmds: Vec<DrawCmd<f32>>,
    gestures: Vec<Gesture>,
    log: Vec<String>,
    version: String,
    last_touch: String,
}

impl Editor {
    /// An editor for a screen of `width` by `height` pixels, with the demo
    /// scene. `measure` sizes text.
    pub fn new(width: f32, height: f32, measure: Box<dyn TextMeasure>) -> Self {
        let (w, h) = if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
            (width, height)
        } else {
            (411.0, 891.0)
        };
        let unit = Self::unit_for(w, h);
        let (tree, ids) = Self::build_tree(w, h, unit);
        let mut e = Self {
            tree,
            ids,
            measure,
            size: (w, h),
            unit,
            mode: Mode::Edit,
            tab: Tab::Hierarchy,
            scene: Scene::demo(CUBE_COUNTS[0]),
            selected: None,
            viewport: Viewport::default(),
            tracker: Tracker::new(10.0 * unit),
            stats: Stats::default(),
            play_time: 0.0,
            time: 0.0,
            paint: PaintList::new(),
            hot: Vec::new(),
            cmds: Vec::new(),
            gestures: Vec::new(),
            log: Vec::new(),
            version: String::from("dev"),
            last_touch: String::from("none"),
        };
        e.log(format!("editor core {}", env!("CARGO_PKG_VERSION")));
        e.build_paint();
        e
    }

    fn unit_for(w: f32, h: f32) -> f32 {
        (w.min(h) / REFERENCE_SHORT_SIDE).clamp(0.5, 6.0)
    }

    fn build_tree(w: f32, h: f32, unit: f32) -> (Tree<f32>, Ids) {
        let fill = |weight: f32| Sizing::Fill(weight);
        let mut tree = Tree::new(Style {
            flow: Flow::Column,
            background: color(BG),
            ..Style::default()
        });
        let root = tree.root();
        let top = tree
            .add_child(
                root,
                Style {
                    height: Sizing::Fixed(56.0 * unit),
                    background: color(BAR),
                    ..Style::default()
                },
            )
            .expect("root exists");
        let (viewport, tabs, panel);
        if w <= h {
            viewport = tree
                .add_child(
                    root,
                    Style {
                        height: fill(3.0),
                        background: color(VIEW_BG),
                        ..Style::default()
                    },
                )
                .expect("root exists");
            tabs = tree
                .add_child(
                    root,
                    Style {
                        height: Sizing::Fixed(56.0 * unit),
                        background: color(BAR),
                        ..Style::default()
                    },
                )
                .expect("root exists");
            panel = tree
                .add_child(
                    root,
                    Style {
                        height: fill(2.0),
                        background: color(PANEL),
                        ..Style::default()
                    },
                )
                .expect("root exists");
        } else {
            let body = tree
                .add_child(
                    root,
                    Style {
                        flow: Flow::Row,
                        ..Style::default()
                    },
                )
                .expect("root exists");
            viewport = tree
                .add_child(
                    body,
                    Style {
                        width: fill(3.0),
                        background: color(VIEW_BG),
                        ..Style::default()
                    },
                )
                .expect("body exists");
            let side = tree
                .add_child(
                    body,
                    Style {
                        width: fill(2.0),
                        flow: Flow::Column,
                        ..Style::default()
                    },
                )
                .expect("body exists");
            tabs = tree
                .add_child(
                    side,
                    Style {
                        height: Sizing::Fixed(56.0 * unit),
                        background: color(BAR),
                        ..Style::default()
                    },
                )
                .expect("side exists");
            panel = tree
                .add_child(
                    side,
                    Style {
                        background: color(PANEL),
                        ..Style::default()
                    },
                )
                .expect("side exists");
        }
        layout::compute(&mut tree, Size::new(w, h));
        (
            tree,
            Ids {
                top,
                viewport,
                tabs,
                panel,
            },
        )
    }

    /// Tells the editor the screen is now `width` by `height` pixels. Sizes
    /// that are not positive and finite are ignored.
    pub fn resize(&mut self, width: f32, height: f32) {
        if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
            return;
        }
        self.size = (width, height);
        self.unit = Self::unit_for(width, height);
        self.tracker.slop = 10.0 * self.unit;
        let (tree, ids) = Self::build_tree(width, height, self.unit);
        self.tree = tree;
        self.ids = ids;
    }

    /// The version text a host shows in the top bar.
    pub fn set_version(&mut self, version: &str) {
        self.version = version.chars().take(24).collect();
    }

    /// Adds a line to the console log, keeping the newest 40.
    pub fn log(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > LOG_LINES {
            self.log.remove(0);
        }
    }

    /// Feeds one pointer event: a touch going down, moving or lifting.
    pub fn pointer(&mut self, phase: Phase, id: i32, x: f32, y: f32) {
        let mut found = std::mem::take(&mut self.gestures);
        found.clear();
        self.tracker.event(phase, id, x, y, self.time, &mut found);
        for g in &found {
            self.apply(*g);
        }
        self.gestures = found;
    }

    /// Forgets every touch in progress. A host calls this when the app is
    /// paused or resumed, or when it cannot tell which fingers are still down.
    pub fn cancel_pointers(&mut self) {
        self.tracker.clear();
    }

    fn apply(&mut self, g: Gesture) {
        let vp = area_of(&self.tree, self.ids.viewport);
        match g {
            Gesture::Tap { x, y } => {
                self.last_touch = format!("tap {x:.0},{y:.0}");
                let action = self
                    .hot
                    .iter()
                    .rev()
                    .find(|(a, _)| a.contains(x, y))
                    .map(|(_, act)| *act);
                if let Some(action) = action {
                    self.perform(action);
                } else if vp.contains(x, y) {
                    let hit = self.viewport.pick(&self.scene, vp, x, y, 20.0 * self.unit);
                    self.selected = hit;
                    match hit.and_then(|i| self.scene.entities.get(i)) {
                        Some(e) => self.log(format!("selected {}", e.name)),
                        None => self.log("selection cleared".to_string()),
                    }
                }
            }
            Gesture::Drag {
                dx,
                dy,
                start_x,
                start_y,
                ..
            } => {
                self.last_touch = "drag".to_string();
                if vp.contains(start_x, start_y) {
                    self.viewport.drag(dx, dy, vp.h);
                }
            }
            Gesture::Pinch {
                ratio,
                x,
                y,
                dx,
                dy,
            } => {
                self.last_touch = format!("pinch x{ratio:.2}");
                if vp.contains(x, y) {
                    self.viewport.pinch(ratio, dx, dy, vp.h);
                }
            }
            Gesture::DragEnd => {}
        }
    }

    fn perform(&mut self, action: Action) {
        match action {
            Action::SetMode(m) => {
                self.mode = m;
                if m == Mode::Edit {
                    self.play_time = 0.0;
                    self.scene.pose(0.0);
                }
                self.log(format!("mode {m:?}"));
            }
            Action::Set2D(on) => {
                self.viewport.mode2d = on;
                self.log(format!("view {}", if on { "2D" } else { "3D" }));
            }
            Action::SetTab(t) => self.tab = t,
            Action::Select(i) => {
                if i < self.scene.len() {
                    self.selected = Some(i);
                }
            }
            Action::SetCount(n) => {
                self.scene = Scene::demo(n);
                self.selected = None;
                self.play_time = 0.0;
                self.log(format!("scene of {n} cubes"));
            }
        }
    }

    /// Advances one frame of `dt` seconds and builds the paint list.
    pub fn frame(&mut self, dt: f32) {
        self.stats.push(dt);
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.1)
        } else {
            0.0
        };
        self.time += dt;
        let started = now();
        if self.mode == Mode::Play {
            self.play_time += dt;
            self.scene.pose(self.play_time);
        }
        self.build_paint();
        if let Some(t) = started {
            self.stats.set_core_ms(t.elapsed().as_secs_f32() * 1000.0);
        }
    }

    /// The shapes for the last frame.
    pub fn paint(&self) -> &PaintList {
        &self.paint
    }

    /// Where the screen regions are.
    pub fn areas(&self) -> Areas {
        Areas {
            top: area_of(&self.tree, self.ids.top),
            viewport: area_of(&self.tree, self.ids.viewport),
            tabs: area_of(&self.tree, self.ids.tabs),
            panel: area_of(&self.tree, self.ids.panel),
        }
    }

    /// Edit or play.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The tab showing.
    pub fn tab(&self) -> Tab {
        self.tab
    }

    /// The selected entity, if any.
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// The scene.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// The viewport state.
    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    /// Pixels per UI unit.
    pub fn unit(&self) -> f32 {
        self.unit
    }

    /// The screen size in pixels.
    pub fn size(&self) -> (f32, f32) {
        self.size
    }

    /// The frame statistics.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    fn build_paint(&mut self) {
        let u = self.unit;
        self.paint.clear();
        self.hot.clear();

        self.cmds.clear();
        draw::build_draw_list(&self.tree, &mut self.cmds);
        for cmd in &self.cmds {
            match cmd {
                DrawCmd::FillRect {
                    rect,
                    color,
                    radius,
                } => self
                    .paint
                    .rect(rect.x, rect.y, rect.w, rect.h, color.to_u32_rgba(), *radius),
                DrawCmd::PushClip(r) => self.paint.push_clip(r.x, r.y, r.w, r.h),
                DrawCmd::PopClip => self.paint.pop_clip(),
                _ => {}
            }
        }

        let areas = Areas {
            top: area_of(&self.tree, self.ids.top),
            viewport: area_of(&self.tree, self.ids.viewport),
            tabs: area_of(&self.tree, self.ids.tabs),
            panel: area_of(&self.tree, self.ids.panel),
        };

        // The viewport, clipped to its own rectangle.
        let vp = areas.viewport;
        self.paint.push_clip(vp.x, vp.y, vp.w, vp.h);
        self.viewport
            .paint(&self.scene, vp, self.selected, u, &mut self.paint);
        let hud = format!(
            "{:.0} fps  {:.1} ms  core {:.1} ms",
            self.stats.fps(),
            self.stats.avg_ms(),
            self.stats.core_ms()
        );
        let view = format!(
            "{:?} | {} | {} cubes",
            self.mode,
            if self.viewport.mode2d { "2D" } else { "3D" },
            self.scene.len()
        );
        let measure: &dyn TextMeasure = &*self.measure;
        let size = 12.0 * u;
        self.paint.text(
            vp.x + 8.0 * u,
            vp.y + 6.0 * u,
            size,
            DIM,
            &fit(measure, &hud, size, vp.w - 16.0 * u),
        );
        self.paint.text(
            vp.x + 8.0 * u,
            vp.y + vp.h - 6.0 * u - size,
            size,
            DIM,
            &fit(measure, &view, size, vp.w - 16.0 * u),
        );
        self.paint.pop_clip();

        let mut ui = Ui {
            out: &mut self.paint,
            measure,
            hot: &mut self.hot,
            unit: u,
        };

        // The top bar: mode buttons, view buttons, version.
        let top = areas.top;
        let (bh, gap) = (40.0 * u, 8.0 * u);
        let by = top.y + (top.h - bh) * 0.5;
        let mut x = top.x + gap;
        let items: [(&str, bool, Action, f32); 4] = [
            (
                "Edit",
                self.mode == Mode::Edit,
                Action::SetMode(Mode::Edit),
                0.0,
            ),
            (
                "Play",
                self.mode == Mode::Play,
                Action::SetMode(Mode::Play),
                16.0 * u,
            ),
            ("3D", !self.viewport.mode2d, Action::Set2D(false), 0.0),
            ("2D", self.viewport.mode2d, Action::Set2D(true), 0.0),
        ];
        let mut after_group = 0.0;
        for (label, active, action, extra) in items {
            let w = (ui.measure.width(label, 15.0 * u) + 24.0 * u).max(56.0 * u);
            if label == "3D" {
                x += after_group;
            }
            ui.button(Area { x, y: by, w, h: bh }, label, active, action);
            x += w + gap;
            if label == "Play" {
                after_group = 16.0 * u;
            }
            let _ = extra;
        }
        let ver = format!("v{}", self.version);
        let vw = ui.measure.width(&ver, 12.0 * u);
        if x + vw + gap < top.x + top.w {
            ui.text(
                top.x + top.w - vw - gap,
                top.y + (top.h - 12.0 * u) * 0.5,
                12.0 * u,
                DIM,
                &ver,
            );
        }

        // The tab strip.
        let tabs = areas.tabs;
        let tw = tabs.w / Tab::ALL.len() as f32;
        for (i, t) in Tab::ALL.iter().enumerate() {
            let a = Area {
                x: tabs.x + tw * i as f32 + u,
                y: tabs.y + 4.0 * u,
                w: tw - 2.0 * u,
                h: tabs.h - 8.0 * u,
            };
            ui.button(a, t.label(), self.tab == *t, Action::SetTab(*t));
        }

        // The panel.
        let p = areas.panel;
        let pad = 12.0 * u;
        let inner = Area {
            x: p.x + pad,
            y: p.y + pad,
            w: (p.w - 2.0 * pad).max(0.0),
            h: (p.h - 2.0 * pad).max(0.0),
        };
        ui.out.push_clip(p.x, p.y, p.w, p.h);
        match self.tab {
            Tab::Hierarchy => {
                let (ts, rh) = (16.0 * u, 48.0 * u);
                ui.text(inner.x, inner.y, 13.0 * u, DIM, "SCENE");
                let top = inner.y + 24.0 * u;
                let cap = (((inner.y + inner.h) - top) / rh).floor().max(0.0) as usize;
                let n = self.scene.len();
                let shown = if n > cap && cap > 0 {
                    cap - 1
                } else {
                    n.min(cap)
                };
                for i in 0..shown {
                    let row = Area {
                        x: p.x + u,
                        y: top + rh * i as f32,
                        w: p.w - 2.0 * u,
                        h: rh,
                    };
                    if self.selected == Some(i) {
                        ui.out
                            .rect(row.x, row.y, row.w, row.h, ROW_SELECTED, 6.0 * u);
                    }
                    let indent = 16.0 * u * depth_of(&self.scene, i) as f32;
                    let name = &self.scene.entities[i].name;
                    ui.text_fit(
                        inner.x + indent,
                        row.y + (rh - ts) * 0.5,
                        ts,
                        TEXT,
                        inner.w - indent,
                        name,
                    );
                    ui.hot.push((row, Action::Select(i)));
                }
                if shown < n {
                    ui.text(
                        inner.x,
                        top + rh * shown as f32 + (rh - 14.0 * u) * 0.5,
                        14.0 * u,
                        DIM,
                        &format!("+ {} more", n - shown),
                    );
                }
            }
            Tab::Inspector => {
                let (ts, rh) = (16.0 * u, 40.0 * u);
                match self.selected.and_then(|i| self.scene.entities.get(i)) {
                    Some(e) => {
                        let rows = [
                            e.name.clone(),
                            format!(
                                "Position  {:.2}  {:.2}  {:.2}",
                                e.pos[0], e.pos[1], e.pos[2]
                            ),
                            format!(
                                "Rotation Y  {:.0} deg",
                                e.yaw.to_degrees().rem_euclid(360.0)
                            ),
                            format!(
                                "Size  {:.2} x {:.2} x {:.2}",
                                e.half[0] * 2.0,
                                e.half[1] * 2.0,
                                e.half[2] * 2.0
                            ),
                            format!(
                                "Parent  {}",
                                e.parent
                                    .and_then(|p| self.scene.entities.get(p))
                                    .map_or("Scene", |p| p.name.as_str())
                            ),
                        ];
                        for (i, r) in rows.iter().enumerate() {
                            ui.text_fit(
                                inner.x,
                                inner.y + rh * i as f32,
                                ts,
                                if i == 0 { TEXT } else { DIM },
                                inner.w,
                                r,
                            );
                        }
                    }
                    None => {
                        ui.text_fit(inner.x, inner.y, ts, TEXT, inner.w, "Nothing selected.");
                        ui.text_fit(
                            inner.x,
                            inner.y + rh,
                            14.0 * u,
                            DIM,
                            inner.w,
                            "Tap a cube in the viewport",
                        );
                        ui.text_fit(
                            inner.x,
                            inner.y + rh * 1.6,
                            14.0 * u,
                            DIM,
                            inner.w,
                            "or a row in the hierarchy.",
                        );
                    }
                }
            }
            Tab::Project => {
                let (ts, rh) = (16.0 * u, 40.0 * u);
                let rows = [
                    ("scenes/", TEXT),
                    ("  demo.scene  (not saved)", DIM),
                    ("assets/", TEXT),
                    ("  (empty)", DIM),
                    ("", DIM),
                    ("Files will come from the project folder.", DIM),
                ];
                for (i, (r, c)) in rows.iter().enumerate() {
                    ui.text_fit(inner.x, inner.y + rh * i as f32, ts, *c, inner.w, r);
                }
            }
            Tab::Console => {
                let (ts, rh) = (14.0 * u, 26.0 * u);
                let s = &self.stats;
                let lines = [
                    format!(
                        "fps {:.1}   frame {:.1} ms   worst {:.1} ms",
                        s.fps(),
                        s.avg_ms(),
                        s.worst_ms()
                    ),
                    format!("core {:.2} ms   paint items {}", s.core_ms(), ui.out.len()),
                    format!(
                        "screen {:.0} x {:.0}   unit {:.2} px",
                        self.size.0, self.size.1, u
                    ),
                    format!("touch: {}", self.last_touch),
                ];
                let mut y = inner.y;
                for l in &lines {
                    ui.text_fit(inner.x, y, ts, TEXT, inner.w, l);
                    y += rh;
                }
                y += 4.0 * u;
                ui.text(inner.x, y + (44.0 * u - ts) * 0.5, ts, DIM, "Cubes");
                let lw = ui.measure.width("Cubes", ts) + 12.0 * u;
                let cw = ((inner.w - lw) / CUBE_COUNTS.len() as f32 - 6.0 * u).max(20.0 * u);
                for (i, n) in CUBE_COUNTS.iter().enumerate() {
                    let a = Area {
                        x: inner.x + lw + (cw + 6.0 * u) * i as f32,
                        y,
                        w: cw,
                        h: 44.0 * u,
                    };
                    ui.button(
                        a,
                        &n.to_string(),
                        self.scene.len() == *n,
                        Action::SetCount(*n),
                    );
                }
                y += 44.0 * u + 10.0 * u;
                for l in self.log.iter().rev() {
                    if y + ts > inner.y + inner.h {
                        break;
                    }
                    ui.text_fit(inner.x, y, 13.0 * u, DIM, inner.w, l);
                    y += 22.0 * u;
                }
            }
        }
        ui.out.pop_clip();

        // The selection color is part of the legend of the viewport.
        let _ = SELECTED;
    }
}
