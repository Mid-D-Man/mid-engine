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
use crate::stats::Stats;
use crate::text::{fit, TextMeasure};
use crate::viewport::{Viewport, SELECTED};
use crate::world::{EditorWorld, HierarchyRow};
use mid_ecs::Entity;

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
    Select(Entity),
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
    world: EditorWorld,
    selected: Option<Entity>,
    /// How far the panel's list is scrolled, in pixels, and how far it can go.
    scroll: f32,
    scroll_max: f32,
    rows: Vec<HierarchyRow>,
    viewport: Viewport,
    tracker: Tracker,
    stats: Stats,
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
            world: EditorWorld::demo(CUBE_COUNTS[0]),
            selected: None,
            scroll: 0.0,
            scroll_max: 0.0,
            rows: Vec::new(),
            viewport: Viewport::default(),
            tracker: Tracker::new(10.0 * unit),
            stats: Stats::default(),
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
                    let hit =
                        self.viewport
                            .pick(self.world.drawables(), vp, x, y, 20.0 * self.unit);
                    self.selected = hit;
                    match hit {
                        Some(e) => {
                            let line = format!("selected {}", self.world.name_of(e));
                            self.log(line);
                        }
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
                } else if area_of(&self.tree, self.ids.panel).contains(start_x, start_y) {
                    // A finger moving up moves the list up.
                    self.scroll = (self.scroll - dy).clamp(0.0, self.scroll_max);
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
                match m {
                    Mode::Play => self.world.enter_play(),
                    Mode::Edit => {
                        if let Some(report) = self.world.exit_play() {
                            let mut line = format!("restored {} entities", report.restored);
                            if report.removed + report.lost + report.failed > 0 {
                                line.push_str(&format!(
                                    " ({} removed, {} lost, {} failed)",
                                    report.removed, report.lost, report.failed
                                ));
                            }
                            self.log(line);
                        }
                    }
                }
                self.log(format!("mode {m:?}"));
            }
            Action::Set2D(on) => {
                self.viewport.mode2d = on;
                self.log(format!("view {}", if on { "2D" } else { "3D" }));
            }
            Action::SetTab(t) => {
                self.tab = t;
                self.scroll = 0.0;
            }
            Action::Select(e) => {
                if self.world.world().is_alive(e) {
                    self.selected = Some(e);
                }
            }
            Action::SetCount(n) => {
                self.world = EditorWorld::demo(n);
                if self.mode == Mode::Play {
                    self.world.enter_play();
                }
                self.selected = None;
                self.scroll = 0.0;
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
            self.world.step(dt);
        } else {
            self.world.refresh();
        }
        if self
            .selected
            .is_some_and(|e| !self.world.world().is_alive(e))
        {
            self.selected = None;
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
    pub fn selected(&self) -> Option<Entity> {
        self.selected
    }

    /// The world being edited.
    pub fn world(&self) -> &EditorWorld {
        &self.world
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
        self.world.refresh();
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
        self.viewport.paint(
            self.world.drawables(),
            vp,
            self.selected,
            u,
            &mut self.paint,
        );
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
            self.world.box_count()
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
        if !matches!(self.tab, Tab::Hierarchy | Tab::Inspector) {
            self.scroll_max = 0.0;
            self.scroll = 0.0;
        }
        match self.tab {
            Tab::Hierarchy => {
                let (ts, rh) = (16.0 * u, 48.0 * u);
                ui.text(inner.x, inner.y, 13.0 * u, DIM, "SCENE");
                let top = inner.y + 24.0 * u;
                let cap = (((inner.y + inner.h) - top) / rh).floor().max(0.0) as usize;
                let total = self.world.row_count();
                // The list scrolls a whole row at a time.
                let max_first = total.saturating_sub(cap);
                self.scroll_max = max_first as f32 * rh;
                self.scroll = self.scroll.clamp(0.0, self.scroll_max);
                let first = ((self.scroll / rh).round() as usize).min(max_first);
                let rest = total - first;
                let shown = if rest > cap && cap > 0 {
                    cap - 1
                } else {
                    rest.min(cap)
                };
                self.world.hierarchy_rows(first, shown, &mut self.rows);
                for (i, r) in self.rows.iter().enumerate() {
                    let row = Area {
                        x: p.x + u,
                        y: top + rh * i as f32,
                        w: p.w - 2.0 * u,
                        h: rh,
                    };
                    if self.selected == Some(r.entity) {
                        ui.out
                            .rect(row.x, row.y, row.w, row.h, ROW_SELECTED, 6.0 * u);
                    }
                    let indent = 16.0 * u * r.depth.min(8) as f32;
                    ui.text_fit(
                        inner.x + indent,
                        row.y + (rh - ts) * 0.5,
                        ts,
                        TEXT,
                        inner.w - indent,
                        &self.world.name_of(r.entity),
                    );
                    ui.hot.push((row, Action::Select(r.entity)));
                }
                if first + shown < total {
                    ui.text(
                        inner.x,
                        top + rh * shown as f32 + (rh - 14.0 * u) * 0.5,
                        14.0 * u,
                        DIM,
                        &format!("+ {} more", total - first - shown),
                    );
                }
            }
            Tab::Inspector => {
                let (ts, rh) = (15.0 * u, 34.0 * u);
                let selected = self.selected.filter(|e| self.world.world().is_alive(*e));
                match selected {
                    Some(e) => {
                        let title =
                            format!("{}  #{}", self.world.name_of(e), e.as_ffi() & 0xFFFF_FFFF);
                        ui.text_fit(inner.x, inner.y, 17.0 * u, TEXT, inner.w, &title);
                        let top = inner.y + 36.0 * u;
                        let rows = self.world.inspect(e);
                        let visible = (((inner.y + inner.h) - top) / rh).floor().max(0.0) as usize;
                        let max_first = rows.len().saturating_sub(visible);
                        self.scroll_max = max_first as f32 * rh;
                        self.scroll = self.scroll.clamp(0.0, self.scroll_max);
                        let first = ((self.scroll / rh).round() as usize).min(max_first);
                        let value_x = inner.x + inner.w * 0.42;
                        for (i, row) in rows.iter().enumerate().skip(first).take(visible) {
                            let y = top + rh * (i - first) as f32;
                            if row.header {
                                ui.out.rect(p.x + u, y, p.w - 2.0 * u, rh, BUTTON, 4.0 * u);
                            }
                            let x = inner.x + 14.0 * u * f32::from(row.indent);
                            let ty = y + (rh - ts) * 0.5;
                            let label_w = if row.text.is_empty() {
                                inner.x + inner.w - x
                            } else {
                                (value_x - x - 6.0 * u).max(0.0)
                            };
                            let label_color = if row.header { TEXT } else { DIM };
                            ui.text_fit(x, ty, ts, label_color, label_w, &row.label);
                            if !row.text.is_empty() {
                                let w = inner.x + inner.w - value_x;
                                ui.text_fit(value_x, ty, ts, TEXT, w, &row.text);
                            }
                        }
                    }
                    None => {
                        self.scroll_max = 0.0;
                        ui.text_fit(
                            inner.x,
                            inner.y,
                            16.0 * u,
                            TEXT,
                            inner.w,
                            "Nothing selected.",
                        );
                        ui.text_fit(
                            inner.x,
                            inner.y + 40.0 * u,
                            14.0 * u,
                            DIM,
                            inner.w,
                            "Tap a cube in the viewport",
                        );
                        ui.text_fit(
                            inner.x,
                            inner.y + 64.0 * u,
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
                        self.world.box_count() == *n,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::FixedWidth;

    fn editor() -> Editor {
        Editor::new(1080.0, 2408.0, Box::new(FixedWidth(0.5)))
    }

    #[test]
    fn a_selection_whose_entity_dies_is_cleared_at_the_next_frame() {
        let mut e = editor();
        let victim = e.world.find("Cube.002").unwrap();
        e.selected = Some(victim);
        e.world.world_mut().despawn(victim);
        e.frame(0.016);
        assert_eq!(e.selected, None);
    }

    #[test]
    fn selecting_a_dead_entity_does_nothing() {
        let mut e = editor();
        let victim = e.world.find("Cube.002").unwrap();
        e.world.world_mut().despawn(victim);
        e.perform(Action::Select(victim));
        assert_eq!(e.selected, None);
    }

    #[test]
    fn the_inspector_ignores_a_selection_that_died_before_the_frame() {
        let mut e = editor();
        let victim = e.world.find("Cube.002").unwrap();
        e.selected = Some(victim);
        e.perform(Action::SetTab(Tab::Inspector));
        e.world.world_mut().despawn(victim);
        e.build_paint();
        let p = e.paint();
        assert!(p
            .items
            .iter()
            .any(|i| p.text_of(i) == Some("Nothing selected.")));
    }
}
