// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "Test fixtures"
// ============================================================================
//! Expectations shared by the neutral tests and the DixScript e2e tests.
//!
//! Each fixture in `tests/fixtures/` has a hand-built `MxValue` twin here and
//! one assertion function. The neutral suite runs the twin through the schema
//! reader, so the expected numbers are checked without DixScript; the e2e suite
//! runs the real file through the parser and must meet the same assertions.
#![allow(dead_code, missing_docs)]

use mid_ui::{draw, hit, layout, Color32, DrawCmd, Point, Rect, Size};
use mid_ui_mxui::{BindKey, EnumTables, Loaded, MxValue};

// ── MxValue builders mirroring the fixtures' QuickFuncs ─────────────────────

fn int(i: i64) -> MxValue {
    MxValue::Int(i)
}

fn s(x: &str) -> MxValue {
    MxValue::Str(x.to_string())
}

fn en(t: &str, f: &str) -> MxValue {
    MxValue::enum_value(t, f)
}

/// The enums every fixture declares, numbered the way DixScript numbers them:
/// from 0, in declaration order.
pub fn tables() -> EnumTables {
    let mut t = EnumTables::new();
    t.insert("UiKind", [("Label", 0), ("Button", 1)]);
    t.insert(
        "UiAnchor",
        [("TopLeft", 0), ("Top", 1), ("TopRight", 2), ("Left", 3), ("Center", 4), ("Right", 5), ("BottomLeft", 6), ("Bottom", 7), ("BottomRight", 8)],
    );
    t.insert("UiFlow", [("Stack", 0), ("Row", 1), ("Column", 2)]);
    t
}

/// How an enum value is written into a twin.
pub type Enc = fn(&str, &str) -> MxValue;

/// What DixScript hands back for an enum that went through a QuickFunc
/// argument: the variant's integer (see `docs/mid-ui-mxui.md`).
pub fn as_int(t: &str, f: &str) -> MxValue {
    let v = tables().variant_value(t, f).unwrap_or_else(|| panic!("{t}.{f} is not in the fixture enums"));
    MxValue::Int(v)
}

/// What DixScript hands back for a bare `Enum.Variant` at a data position.
pub fn as_enum(t: &str, f: &str) -> MxValue {
    en(t, f)
}

pub fn fill(w: f64) -> MxValue {
    MxValue::object([("fill", MxValue::Float(w))])
}

#[allow(clippy::too_many_arguments)]
fn boxv(
    enc: Enc,
    id: &str,
    anchor: &str,
    flow: &str,
    width: MxValue,
    height: MxValue,
    margin: MxValue,
    padding: MxValue,
    gap: i64,
    bg: &str,
    radius: i64,
    clip: bool,
    children: Vec<MxValue>,
) -> MxValue {
    MxValue::object([
        ("id", s(id)),
        ("anchor", enc("UiAnchor", anchor)),
        ("flow", enc("UiFlow", flow)),
        ("width", width),
        ("height", height),
        ("margin", margin),
        ("padding", padding),
        ("gap", int(gap)),
        ("background", MxValue::Color(bg.to_string())),
        ("radius", int(radius)),
        ("clip", MxValue::Bool(clip)),
        ("children", MxValue::Array(children)),
    ])
}

#[allow(clippy::too_many_arguments)]
fn leafv(
    enc: Enc,
    id: &str,
    kind: &str,
    anchor: &str,
    width: MxValue,
    height: MxValue,
    margin: i64,
    text: &str,
    bind: &str,
    interactive: bool,
) -> MxValue {
    MxValue::object([
        ("id", s(id)),
        ("kind", enc("UiKind", kind)),
        ("anchor", enc("UiAnchor", anchor)),
        ("width", width),
        ("height", height),
        ("margin", int(margin)),
        ("text", s(text)),
        ("bind", s(bind)),
        ("interactive", MxValue::Bool(interactive)),
    ])
}

// ── twins of the fixtures ───────────────────────────────────────────────────

/// Twin of `fixtures/real.mxui`, in the form DixScript delivers it: enums
/// that went through QuickFunc arguments are integers.
pub fn real_value() -> MxValue {
    real_with(as_int)
}

/// The same tree with real enum values, as a table-style file would give.
pub fn real_value_enum_form() -> MxValue {
    real_with(as_enum)
}

fn real_with(enc: Enc) -> MxValue {
    boxv(enc, "screen", "TopLeft", "Stack", fill(1.0), fill(1.0), int(0), int(0), 0, "#0B0F14", 0, false, vec![
        boxv(enc, "top_bar", "Top", "Row", fill(1.0), int(48), int(0), MxValue::Array(vec![int(12), int(8)]), 8, "#151B23", 0, false, vec![
            leafv(enc, "hp_label", "Label", "TopLeft", int(100), fill(1.0), 0, "HP", "player_health", false),
            leafv(enc, "score_label", "Label", "TopLeft", fill(1.0), fill(1.0), 0, "Score", "player_score", false),
            leafv(enc, "pause_btn", "Button", "TopLeft", int(80), fill(1.0), 0, "Pause", "ui_pause", true),
        ]),
        boxv(enc, "menu", "Center", "Column", int(300), int(220), int(0), int(16), 8, "#1E2430", 8, true, vec![
            leafv(enc, "menu_title", "Label", "TopLeft", fill(1.0), int(32), 0, "Paused", "menu_title_text", false),
            leafv(enc, "play_btn", "Button", "TopLeft", fill(1.0), int(40), 0, "Resume", "ui_resume", true),
            leafv(enc, "options_btn", "Button", "TopLeft", fill(1.0), int(40), 0, "Options", "ui_options", true),
            leafv(enc, "quit_btn", "Button", "TopLeft", fill(1.0), int(40), 0, "Quit", "ui_quit", true),
        ]),
        leafv(enc, "hint", "Label", "BottomRight", int(200), int(20), 12, "Press Esc to pause", "hint_text", false),
    ])
}

/// Twin of `fixtures/row_weights.mxui`.
pub fn row_weights_value() -> MxValue {
    boxv(as_int, "row", "TopLeft", "Row", fill(1.0), fill(1.0), int(0), int(0), 0, "#0000", 0, false, vec![
        leafv(as_int, "fixed", "Label", "TopLeft", int(50), fill(1.0), 0, "fixed", "w_fixed", false),
        leafv(as_int, "one", "Label", "TopLeft", fill(1.0), fill(1.0), 0, "one", "w_one", false),
        leafv(as_int, "three", "Label", "TopLeft", fill(3.0), fill(1.0), 0, "three", "w_three", false),
    ])
}

/// Twin of `fixtures/minimal_table.mxui`.
pub fn minimal_table_value() -> MxValue {
    MxValue::object([
        ("id", s("solo")),
        ("anchor", en("UiAnchor", "Center")),
        ("width", int(100)),
        ("height", int(40)),
    ])
}

/// Twin of `examples/hud.mxui`.
pub fn hud_value() -> MxValue {
    boxv(as_int, "hud_root", "TopLeft", "Row", fill(1.0), int(96), int(0), int(0), 0, "#0000", 0, false, vec![
        leafv(as_int, "hp_label", "Label", "TopLeft", int(120), int(24), 0, "HP", "player_health", false),
        leafv(as_int, "pause_btn", "Button", "TopLeft", int(80), int(24), 0, "Pause", "ui_pause", true),
    ])
}

// ── assertions ──────────────────────────────────────────────────────────────

pub fn rect_of(l: &Loaded<f32>, name: &str) -> (f32, f32, f32, f32) {
    let r = l.tree.rect(l.names[name]).unwrap_or_else(|| panic!("no node `{name}`"));
    (r.x, r.y, r.w, r.h)
}

pub fn hit_name(l: &Loaded<f32>, x: f32, y: f32) -> Option<String> {
    hit::hit_test(&l.tree, Point::new(x, y)).and_then(|id| l.meta.get(&id)).and_then(|m| m.name.clone())
}

fn assert_rects(l: &Loaded<f32>, want: &[(&str, (f32, f32, f32, f32))]) {
    for (name, rect) in want {
        assert_eq!(rect_of(l, name), *rect, "rect of `{name}`");
    }
}

/// Lays out at 800x600 and checks `real.mxui`: rectangles, metadata,
/// interactivity, hit testing and the draw list.
pub fn assert_real_screen(l: &mut Loaded<f32>) {
    layout::compute(&mut l.tree, Size::new(800.0, 600.0));
    assert_eq!(l.tree.len(), 11);

    assert_rects(l, &[
        ("screen", (0.0, 0.0, 800.0, 600.0)),
        ("top_bar", (0.0, 0.0, 800.0, 48.0)),
        ("hp_label", (12.0, 8.0, 100.0, 32.0)),
        ("score_label", (120.0, 8.0, 580.0, 32.0)),
        ("pause_btn", (708.0, 8.0, 80.0, 32.0)),
        ("menu", (250.0, 190.0, 300.0, 220.0)),
        ("menu_title", (266.0, 206.0, 268.0, 32.0)),
        ("play_btn", (266.0, 246.0, 268.0, 40.0)),
        ("options_btn", (266.0, 294.0, 268.0, 40.0)),
        ("quit_btn", (266.0, 342.0, 268.0, 40.0)),
        ("hint", (588.0, 568.0, 200.0, 20.0)),
    ]);

    let leaves = [
        ("hp_label", "Label", "HP", "player_health", false),
        ("score_label", "Label", "Score", "player_score", false),
        ("pause_btn", "Button", "Pause", "ui_pause", true),
        ("menu_title", "Label", "Paused", "menu_title_text", false),
        ("play_btn", "Button", "Resume", "ui_resume", true),
        ("options_btn", "Button", "Options", "ui_options", true),
        ("quit_btn", "Button", "Quit", "ui_quit", true),
        ("hint", "Label", "Press Esc to pause", "hint_text", false),
    ];
    for (name, kind, text, bind, interactive) in leaves {
        let id = l.names[name];
        let m = &l.meta[&id];
        let k = m.kind.as_ref().unwrap_or_else(|| panic!("`{name}` has no kind"));
        assert_eq!((k.type_name.as_str(), k.field.as_str()), ("UiKind", kind), "kind of `{name}`");
        assert_eq!(m.text.as_deref(), Some(text), "text of `{name}`");
        assert_eq!(m.bind.as_ref().map(|b| b.key), Some(BindKey::of(bind)), "bind of `{name}`");
        assert_eq!(l.tree.style(id).unwrap().interactive, interactive, "interactive of `{name}`");
    }
    for name in ["screen", "top_bar", "menu"] {
        let m = &l.meta[&l.names[name]];
        assert!(m.kind.is_none() && m.text.is_none() && m.bind.is_none(), "`{name}` is a plain container");
    }

    // Buttons are hit; labels and containers let the pointer fall through.
    assert_eq!(hit_name(l, 748.0, 24.0).as_deref(), Some("pause_btn"));
    assert_eq!(hit_name(l, 400.0, 266.0).as_deref(), Some("play_btn"));
    assert_eq!(hit_name(l, 400.0, 314.0).as_deref(), Some("options_btn"));
    assert_eq!(hit_name(l, 400.0, 362.0).as_deref(), Some("quit_btn"));
    assert_eq!(hit_name(l, 788.0, 24.0), None, "right edge is outside");
    assert_eq!(hit_name(l, 400.0, 210.0), None, "a label is not interactive");
    assert_eq!(hit_name(l, 260.0, 200.0), None, "menu padding is not interactive");
    assert_eq!(hit_name(l, 10.0, 10.0), None);
    assert!(hit::claims_pointer(&l.tree, Point::new(400.0, 266.0)));
    assert!(!hit::claims_pointer(&l.tree, Point::new(10.0, 10.0)));

    let color = |hex: &str| Color32::from_hex(hex).unwrap();
    let mut list = Vec::new();
    draw::build_draw_list(&l.tree, &mut list);
    assert_eq!(
        list,
        vec![
            DrawCmd::FillRect { rect: Rect::new(0.0, 0.0, 800.0, 600.0), color: color("#0B0F14"), radius: 0.0 },
            DrawCmd::FillRect { rect: Rect::new(0.0, 0.0, 800.0, 48.0), color: color("#151B23"), radius: 0.0 },
            DrawCmd::FillRect { rect: Rect::new(250.0, 190.0, 300.0, 220.0), color: color("#1E2430"), radius: 8.0 },
            DrawCmd::PushClip(Rect::new(250.0, 190.0, 300.0, 220.0)),
            DrawCmd::PopClip,
        ]
    );
}

/// Lays out at 250x100 and checks `row_weights.mxui`.
pub fn assert_row_weights(l: &mut Loaded<f32>) {
    layout::compute(&mut l.tree, Size::new(250.0, 100.0));
    assert_eq!(l.tree.len(), 4);
    assert_rects(l, &[
        ("row", (0.0, 0.0, 250.0, 100.0)),
        ("fixed", (0.0, 0.0, 50.0, 100.0)),
        ("one", (50.0, 0.0, 50.0, 100.0)),
        ("three", (100.0, 0.0, 150.0, 100.0)),
    ]);
    // The row's `#0000` background is fully transparent: nothing to draw.
    let mut list = Vec::new();
    draw::build_draw_list(&l.tree, &mut list);
    assert!(list.is_empty());
}

/// Lays out at 200x100 and checks `minimal_table.mxui`.
pub fn assert_minimal_table(l: &mut Loaded<f32>) {
    layout::compute(&mut l.tree, Size::new(200.0, 100.0));
    assert_eq!(l.tree.len(), 1);
    assert_rects(l, &[("solo", (50.0, 30.0, 100.0, 40.0))]);
}

/// Lays out at 320x200 and checks `examples/hud.mxui`.
pub fn assert_hud(l: &mut Loaded<f32>) {
    layout::compute(&mut l.tree, Size::new(320.0, 200.0));
    assert_eq!(l.tree.len(), 3);
    assert_rects(l, &[
        ("hud_root", (0.0, 0.0, 320.0, 96.0)),
        ("hp_label", (0.0, 0.0, 120.0, 24.0)),
        ("pause_btn", (120.0, 0.0, 80.0, 24.0)),
    ]);
    assert!(l.tree.style(l.names["pause_btn"]).unwrap().interactive);
    assert!(!l.tree.style(l.names["hp_label"]).unwrap().interactive);
}
