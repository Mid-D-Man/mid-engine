// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "dix.rs"
// ============================================================================
//! End-to-end: the real `.mxui` fixture files through the published
//! `dixscript` crate, held to the same assertions as their hand-built twins.
//!
//! Written from the DixScript v1.0.0 grammar and the crate's own tests without
//! ever being run (DixScript-Rust is not executed in the authoring
//! environment), so the first real run of this file is CI.
#![cfg(feature = "dixscript")]

mod common;

use mid_ui_mxui::{load_str, parse_str, Loaded, MxErrorKind, MxValue};

const REAL: &str = include_str!("fixtures/real.mxui");
const ROW_WEIGHTS: &str = include_str!("fixtures/row_weights.mxui");
const MINIMAL_TABLE: &str = include_str!("fixtures/minimal_table.mxui");
const BAD_FIELD: &str = include_str!("fixtures/bad_field.mxui");
const NO_ROOT: &str = include_str!("fixtures/no_root.mxui");
const WRONG_ENUM: &str = include_str!("fixtures/wrong_enum.mxui");
const HUD: &str = include_str!("../examples/hud.mxui");

// Loads a fixture, and when that fails shows what the parser actually
// returned, so a mismatch is diagnosable from the CI log alone.
fn load(src: &str, name: &str) -> Loaded<f32> {
    match load_str::<f32>(src) {
        Ok(l) => l,
        Err(e) => {
            let parsed = match parse_str(src) {
                Ok((value, tables)) => format!("{value:#?}\n{tables:#?}"),
                Err(pe) => format!("(parse_str also failed: {pe})"),
            };
            panic!("{name} failed to load: {e}\n--- what the parser returned ---\n{parsed}");
        }
    }
}

fn field<'a>(v: &'a MxValue, name: &str) -> Option<&'a MxValue> {
    match v {
        MxValue::Object(entries) => entries.iter().find(|(k, _)| k == name).map(|(_, v)| v),
        _ => None,
    }
}

// What DixScript really returns for enums, written down so a change in the
// runtime fails here with a clear message instead of deep in a layout test.
#[test]
fn enums_through_quickfunc_arguments_arrive_as_integers() {
    let (root, tables) = parse_str(REAL).expect("real.mxui must parse");
    assert_eq!(field(&root, "anchor"), Some(&MxValue::Int(0)), "root anchor (TopLeft) as an int: {root:#?}");
    assert_eq!(field(&root, "flow"), Some(&MxValue::Int(0)), "root flow (Stack) as an int");
    assert_eq!(tables.variant_name("UiAnchor", 4), Some("Center"));
    assert_eq!(tables.variant_name("UiFlow", 1), Some("Row"));
    assert_eq!(tables.variant_name("UiKind", 1), Some("Button"));
}

#[test]
fn a_bare_enum_in_a_root_table_stays_an_enum_value() {
    let (root, _) = parse_str(MINIMAL_TABLE).expect("minimal_table.mxui must parse");
    assert_eq!(
        field(&root, "anchor"),
        Some(&MxValue::enum_value("UiAnchor", "Center")),
        "anchor in a table property: {root:#?}"
    );
}

#[test]
fn real_mxui_meets_every_expectation() {
    common::assert_real_screen(&mut load(REAL, "real.mxui"));
}

#[test]
fn real_mxui_agrees_between_f32_and_f64() {
    let mut a = load(REAL, "real.mxui");
    let mut b = load_str::<f64>(REAL).unwrap();
    mid_ui::layout::compute(&mut a.tree, mid_ui::Size::new(800.0, 600.0));
    mid_ui::layout::compute(&mut b.tree, mid_ui::Size::new(800.0, 600.0));
    assert_eq!(a.names.len(), b.names.len());
    for (name, id) in &a.names {
        let (ra, rb) = (a.tree.rect(*id).unwrap(), b.tree.rect(b.names[name]).unwrap());
        assert_eq!((ra.x as f64, ra.y as f64, ra.w as f64, ra.h as f64), (rb.x, rb.y, rb.w, rb.h), "{name}");
    }
}

#[test]
fn row_weights_mxui_shares_space_one_to_three() {
    common::assert_row_weights(&mut load(ROW_WEIGHTS, "row_weights.mxui"));
}

#[test]
fn minimal_table_mxui_loads_a_root_table_with_an_enum() {
    common::assert_minimal_table(&mut load(MINIMAL_TABLE, "minimal_table.mxui"));
}

#[test]
fn the_hud_example_loads() {
    common::assert_hud(&mut load(HUD, "examples/hud.mxui"));
}

#[test]
fn a_typo_in_a_field_name_is_reported_at_its_path() {
    let e = load_str::<f32>(BAD_FIELD).err().expect("bad_field.mxui must be rejected");
    assert_eq!(e.path, "root.widht");
    assert_eq!(e.kind, MxErrorKind::UnknownField("widht".into()));
}

#[test]
fn an_enum_of_the_wrong_type_is_reported_at_its_path() {
    let e = load_str::<f32>(WRONG_ENUM).err().expect("wrong_enum.mxui must be rejected");
    assert_eq!(e.path, "root.anchor");
    assert_eq!(e.kind, MxErrorKind::WrongEnum { expected: "UiAnchor", found: "UiFlow".into() });
}

#[test]
fn missing_root_and_broken_source_are_distinct_errors() {
    let e = load_str::<f32>(NO_ROOT).err().unwrap();
    assert_eq!(e.kind, MxErrorKind::MissingRoot);
    let e = load_str::<f32>("@DATA(").err().unwrap();
    assert!(matches!(e.kind, MxErrorKind::Source(_)));
}
