// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "Test fixtures"
// ============================================================================
//! The fixture twins through the schema reader (no DixScript), plus static
//! checks on the fixture files themselves.

mod common;

use mid_ui_mxui::{build_with, Loaded};

const FIXTURES: [(&str, &str); 7] = [
    ("real", include_str!("fixtures/real.mxui")),
    ("row_weights", include_str!("fixtures/row_weights.mxui")),
    ("minimal_table", include_str!("fixtures/minimal_table.mxui")),
    ("bad_field", include_str!("fixtures/bad_field.mxui")),
    ("no_root", include_str!("fixtures/no_root.mxui")),
    ("wrong_enum", include_str!("fixtures/wrong_enum.mxui")),
    ("hud", include_str!("../examples/hud.mxui")),
];

fn loaded(v: &mid_ui_mxui::MxValue) -> Loaded<f32> {
    build_with::<f32>(v, &common::tables()).unwrap()
}

#[test]
fn real_screen_expectations_hold_for_the_twin() {
    common::assert_real_screen(&mut loaded(&common::real_value()));
}

#[test]
fn row_weights_expectations_hold_for_the_twin() {
    common::assert_row_weights(&mut loaded(&common::row_weights_value()));
}

#[test]
fn minimal_table_expectations_hold_for_the_twin() {
    common::assert_minimal_table(&mut loaded(&common::minimal_table_value()));
}

#[test]
fn hud_expectations_hold_for_the_twin() {
    common::assert_hud(&mut loaded(&common::hud_value()));
}

#[test]
fn the_real_screen_also_lays_out_at_f64() {
    let mut a = build_with::<f32>(&common::real_value(), &common::tables()).unwrap();
    let mut b = build_with::<f64>(&common::real_value(), &common::tables()).unwrap();
    mid_ui::layout::compute(&mut a.tree, mid_ui::Size::new(800.0, 600.0));
    mid_ui::layout::compute(&mut b.tree, mid_ui::Size::new(800.0, 600.0));
    for (name, id) in &a.names {
        let (ra, rb) = (a.tree.rect(*id).unwrap(), b.tree.rect(b.names[name]).unwrap());
        assert_eq!((ra.x as f64, ra.y as f64, ra.w as f64, ra.h as f64), (rb.x, rb.y, rb.w, rb.h), "{name}");
    }
}

// DixScript gives back an integer for an enum that went through a QuickFunc
// argument and a real enum value for a bare `Enum.Variant`; both must build
// the same tree, so a file may mix the two freely.
#[test]
fn integer_and_enum_forms_of_the_same_file_build_identical_trees() {
    let mut a = loaded(&common::real_value());
    let mut b = loaded(&common::real_value_enum_form());
    common::assert_real_screen(&mut a);
    common::assert_real_screen(&mut b);
}

#[test]
fn the_twins_really_use_integers_where_quickfuncs_do() {
    use mid_ui_mxui::MxValue;
    let MxValue::Object(fields) = common::real_value() else { panic!("root is an object") };
    let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    assert_eq!(get("anchor"), Some(MxValue::Int(0)));
    assert_eq!(get("flow"), Some(MxValue::Int(0)));
    let MxValue::Object(e) = common::real_value_enum_form() else { panic!() };
    assert!(matches!(e.iter().find(|(n, _)| n == "anchor"), Some((_, MxValue::Enum { .. }))));
}

// The most common hand-writing mistakes from the DixScript writing guide,
// checked on the fixture text so a bad file fails here, not only in CI.
#[test]
fn fixtures_follow_the_house_rules() {
    for (name, src) in FIXTURES {
        assert_eq!(
            src.lines().next(),
            Some("// Brought to u by MidManStudio"),
            "{name}: the signature comment must be the first line"
        );
        let open = src.matches('(').count();
        let close = src.matches(')').count();
        assert_eq!(open, close, "{name}: unbalanced parentheses");
        assert_eq!(src.matches('[').count(), src.matches(']').count(), "{name}: unbalanced brackets");
        assert_eq!(src.matches('{').count(), src.matches('}').count(), "{name}: unbalanced braces");

        // `@CONFIG features` must list every gated section the file uses.
        if let Some(features) = src.lines().find(|l| l.trim_start().starts_with("features")) {
            for (section, keyword) in [("@QUICKFUNCS(", "quickfuncs"), ("@ENUMS(", "enums")] {
                if src.contains(section) {
                    assert!(features.contains(keyword), "{name}: {section} needs `{keyword}` in features");
                }
            }
        }
        assert!(src.contains("@DATA("), "{name}: no @DATA section");
        // Only DixScript line comments; no stray `/*`.
        assert!(!src.contains("/*"), "{name}: block comments are not DixScript");
    }
}

#[test]
fn fixtures_use_only_the_enum_names_the_reader_expects() {
    for (name, src) in FIXTURES {
        for word in src.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.') {
            if let Some(variant) = word.strip_prefix("UiAnchor.") {
                const ANCHORS: [&str; 9] =
                    ["TopLeft", "Top", "TopRight", "Left", "Center", "Right", "BottomLeft", "Bottom", "BottomRight"];
                assert!(ANCHORS.contains(&variant), "{name}: unknown UiAnchor.{variant}");
            }
            if let Some(variant) = word.strip_prefix("UiFlow.") {
                assert!(["Stack", "Row", "Column"].contains(&variant), "{name}: unknown UiFlow.{variant}");
            }
            if let Some(variant) = word.strip_prefix("UiKind.") {
                assert!(["Label", "Button"].contains(&variant), "{name}: unknown UiKind.{variant}");
            }
        }
    }
}
