// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "dix.rs"
// ============================================================================
//! End-to-end: real `.mxui` source through the published `dixscript` crate.
//!
//! Written from the DixScript v1.0.0 grammar and the crate's own tests
//! without ever being run (DixScript-Rust is not executed in the authoring
//! environment), so the first real run of this file is CI.
#![cfg(feature = "dixscript")]

use mid_ui::{layout, Size};
use mid_ui_mxui::{load_str, BindKey, MxErrorKind};

const HUD: &str = include_str!("../examples/hud.mxui");

fn rect<R: mid_ui::Real>(l: &mid_ui_mxui::Loaded<R>, name: &str) -> (f64, f64, f64, f64) {
    let q = l.tree.rect(l.names[name]).unwrap();
    (q.x.to_f64(), q.y.to_f64(), q.w.to_f64(), q.h.to_f64())
}

#[test]
fn the_example_file_loads_and_lays_out_at_both_precisions() {
    let mut a = load_str::<f32>(HUD).expect("examples/hud.mxui must load");
    let mut b = load_str::<f64>(HUD).expect("examples/hud.mxui must load");
    layout::compute(&mut a.tree, Size::new(320.0, 200.0));
    layout::compute(&mut b.tree, Size::new(320.0, 200.0));
    for l in [rect(&a, "hud_root"), rect(&b, "hud_root")] {
        assert_eq!(l, (0.0, 0.0, 320.0, 96.0));
    }
    for name in ["hp_label", "pause_btn"] {
        assert_eq!(rect(&a, name), rect(&b, name));
    }
    assert_eq!(rect(&a, "hp_label"), (0.0, 0.0, 120.0, 24.0));
    assert_eq!(rect(&a, "pause_btn"), (120.0, 0.0, 80.0, 24.0));
    assert_eq!(a.tree.len(), 3);
}

#[test]
fn kinds_text_bindings_and_interactivity_come_through() {
    let l = load_str::<f32>(HUD).unwrap();
    let hp = &l.meta[&l.names["hp_label"]];
    assert_eq!(hp.kind.as_ref().unwrap().type_name, "UiKind");
    assert_eq!(hp.kind.as_ref().unwrap().field, "Label");
    assert_eq!(hp.text.as_deref(), Some("HP"));
    assert_eq!(hp.bind.as_ref().unwrap().key, BindKey::of("player_health"));

    let pause = l.names["pause_btn"];
    assert_eq!(l.meta[&pause].kind.as_ref().unwrap().field, "Button");
    assert!(l.tree.style(pause).unwrap().interactive);
    assert!(!l.tree.style(l.names["hp_label"]).unwrap().interactive);
}

#[test]
fn root_may_also_be_a_single_table() {
    let src = "@DATA(\n  root:\n    id = \"solo\"\n    width = 100\n)\n";
    let l = load_str::<f32>(src).unwrap();
    assert_eq!(l.tree.len(), 1);
    assert!(l.names.contains_key("solo"));
}

#[test]
fn a_typo_in_a_field_name_is_reported_at_its_path() {
    let src = "@DATA(\n  root:\n    widht = 100\n)\n";
    let e = load_str::<f32>(src).err().unwrap();
    assert_eq!(e.path, "root.widht");
    assert_eq!(e.kind, MxErrorKind::UnknownField("widht".into()));
}

#[test]
fn missing_root_and_broken_source_are_distinct_errors() {
    let e = load_str::<f32>("@DATA(\n  other = 1\n)\n").err().unwrap();
    assert_eq!(e.kind, MxErrorKind::MissingRoot);
    let e = load_str::<f32>("@DATA(").err().unwrap();
    assert!(matches!(e.kind, MxErrorKind::Source(_)));
}
