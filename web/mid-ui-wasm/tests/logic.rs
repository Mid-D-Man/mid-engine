// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-wasm.md, section "logic.rs"
// ============================================================================
//! Native tests of the playground logic. They build trees through the
//! schema reader (no DixScript) and check the JSON by parsing it.

use mid_ui::{Color32, DrawCmd, Rect};
use mid_ui_mxui::{build, MxValue};
use mid_ui_wasm::logic::{describe_json, draw_list_json, json_string, Core};
use serde_json::{json, Value};

fn en(t: &str, f: &str) -> MxValue {
    MxValue::enum_value(t, f)
}

fn s(x: &str) -> MxValue {
    MxValue::Str(x.into())
}

// Root (red) with a labelled button and an unnamed interactive child.
fn sample() -> Core {
    let v = MxValue::object([
        ("id", s("root")),
        ("background", MxValue::Color("#FF0000".into())),
        (
            "children",
            MxValue::Array(vec![
                MxValue::object([
                    ("id", s("ok_btn")),
                    ("kind", en("UiKind", "Button")),
                    ("text", s("OK \"now\"")),
                    ("bind", s("ui_ok")),
                    ("width", MxValue::Int(40)),
                    ("height", MxValue::Int(20)),
                    ("interactive", MxValue::Bool(true)),
                    ("background", MxValue::Color("#00FF0080".into())),
                    ("radius", MxValue::Int(4)),
                ]),
                MxValue::object([
                    ("anchor", en("UiAnchor", "BottomRight")),
                    ("width", MxValue::Int(30)),
                    ("height", MxValue::Int(10)),
                    ("interactive", MxValue::Bool(true)),
                    ("clip", MxValue::Bool(true)),
                ]),
            ]),
        ),
    ]);
    let mut core = Core::new(build::<f32>(&v).unwrap());
    core.layout(100.0, 50.0);
    core
}

fn parse(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("invalid JSON ({e}): {text}"))
}

#[test]
fn json_strings_are_escaped_and_parse_back() {
    for text in ["plain", "quo\"te", "back\\slash", "line\nbreak\ttab\r", "\u{1}\u{1f}", "é😀日本", ""] {
        let mut out = String::new();
        json_string(&mut out, text);
        assert_eq!(parse(&out), Value::String(text.to_string()), "{text:?}");
    }
}

#[test]
fn draw_list_json_has_the_documented_shape() {
    let core = sample();
    assert_eq!(
        parse(&core.draw_list_json()),
        json!([
            {"t":"fill","x":0,"y":0,"w":100,"h":50,"r":0,"c":[255,0,0,255]},
            {"t":"fill","x":0,"y":0,"w":40,"h":20,"r":4,"c":[0,255,0,128]},
            {"t":"clip","x":70,"y":40,"w":30,"h":10},
            {"t":"pop"}
        ])
    );
}

#[test]
fn non_finite_numbers_become_zero_so_the_json_stays_valid() {
    let list = [DrawCmd::FillRect {
        rect: Rect::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.5),
        color: Color32::WHITE,
        radius: f32::NAN,
    }];
    assert_eq!(
        parse(&draw_list_json(&list)),
        json!([{"t":"fill","x":0,"y":0,"w":0,"h":1.5,"r":0,"c":[255,255,255,255]}])
    );
}

#[test]
fn an_empty_draw_list_is_an_empty_array() {
    assert_eq!(draw_list_json(&[]), "[]");
}

#[test]
fn describe_lists_nodes_in_paint_order_with_depth_and_metadata() {
    let core = sample();
    let nodes = parse(&core.describe_json());
    let nodes = nodes.as_array().unwrap();
    assert_eq!(nodes.len(), core.node_count());
    assert_eq!(nodes[0]["id"], json!("root"));
    assert_eq!(nodes[0]["depth"], json!(0));
    assert_eq!(nodes[0]["kind"], Value::Null);

    let btn = &nodes[1];
    assert_eq!(btn["depth"], json!(1));
    assert_eq!(btn["id"], json!("ok_btn"));
    assert_eq!(btn["kind"], json!("UiKind.Button"));
    assert_eq!(btn["text"], json!("OK \"now\""));
    assert_eq!(btn["bind"], json!("ui_ok"));
    assert_eq!((btn["x"].clone(), btn["y"].clone(), btn["w"].clone(), btn["h"].clone()), (json!(0), json!(0), json!(40), json!(20)));
    assert_eq!(btn["interactive"], json!(true));
    assert_eq!(btn["visible"], json!(true));

    let unnamed = &nodes[2];
    assert_eq!(unnamed["id"], Value::Null);
    assert_eq!(unnamed["text"], Value::Null);
    assert_eq!(unnamed["bind"], Value::Null);
    assert_eq!(unnamed["x"], json!(70));
}

#[test]
fn describe_walks_deep_nesting_in_order() {
    let leaf = MxValue::object([("id", s("leaf"))]);
    let mid = MxValue::object([("id", s("mid")), ("children", MxValue::Array(vec![leaf]))]);
    let sibling = MxValue::object([("id", s("sib"))]);
    let root = MxValue::object([("id", s("r")), ("children", MxValue::Array(vec![mid, sibling]))]);
    let mut core = Core::new(build::<f32>(&root).unwrap());
    core.layout(10.0, 10.0);
    let nodes = parse(&core.describe_json());
    let order: Vec<(String, u64)> = nodes
        .as_array()
        .unwrap()
        .iter()
        .map(|n| (n["id"].as_str().unwrap().to_string(), n["depth"].as_u64().unwrap()))
        .collect();
    assert_eq!(order, [("r".into(), 0), ("mid".into(), 1), ("leaf".into(), 2), ("sib".into(), 1)]);
}

#[test]
fn hit_name_reports_named_unnamed_and_none() {
    let core = sample();
    assert_eq!(core.hit_name(10.0, 10.0).as_deref(), Some("ok_btn"));
    assert_eq!(core.hit_name(80.0, 45.0).as_deref(), Some("(unnamed)"));
    assert_eq!(core.hit_name(60.0, 25.0), None);
}

#[test]
fn claims_pointer_matches_hit_name() {
    let core = sample();
    for (x, y) in [(10.0, 10.0), (80.0, 45.0), (60.0, 25.0), (-5.0, -5.0), (100.0, 50.0)] {
        assert_eq!(core.claims_pointer(x, y), core.hit_name(x, y).is_some(), "({x}, {y})");
    }
}

#[test]
fn describe_reflects_a_new_layout() {
    let mut core = sample();
    core.layout(200.0, 100.0);
    let nodes = parse(&core.describe_json());
    assert_eq!(nodes[0]["w"], json!(200));
    assert_eq!(nodes[2]["x"], json!(170));
    assert_eq!(nodes[2]["y"], json!(90));
}

#[test]
fn describe_json_free_function_matches_the_core_method() {
    let v = MxValue::object([("id", s("only"))]);
    let loaded = build::<f32>(&v).unwrap();
    let from_fn = describe_json(&loaded);
    let core = Core::new(loaded);
    assert_eq!(from_fn, core.describe_json());
}
