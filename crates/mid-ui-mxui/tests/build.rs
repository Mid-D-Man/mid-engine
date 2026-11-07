// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "load.rs"
// ============================================================================
//! Schema reader tests, driven by hand-built `MxValue` trees (no DixScript).

use mid_ui::{layout, Anchor, Color32, Flow, Insets, NodeId, Size, Sizing, Style};
use mid_ui_mxui::{build, BindKey, MxErrorKind, MxValue};

fn int(v: i64) -> MxValue {
    MxValue::Int(v)
}

fn text(v: &str) -> MxValue {
    MxValue::Str(v.to_string())
}

fn en(t: &str, f: &str) -> MxValue {
    MxValue::enum_value(t, f)
}

fn node(fields: Vec<(&str, MxValue)>) -> MxValue {
    MxValue::object(fields)
}

fn kids(children: Vec<MxValue>) -> MxValue {
    MxValue::Array(children)
}

fn fill(w: f64) -> MxValue {
    node(vec![("fill", MxValue::Float(w))])
}

fn err_of(v: &MxValue) -> (String, MxErrorKind) {
    match build::<f32>(v) {
        Ok(_) => panic!("expected an error"),
        Err(e) => (e.path, e.kind),
    }
}

fn rect(l: &mid_ui_mxui::Loaded<f32>, id: NodeId) -> (f32, f32, f32, f32) {
    let r = l.tree.rect(id).unwrap();
    (r.x, r.y, r.w, r.h)
}

// ── structure ───────────────────────────────────────────────────────────────

#[test]
fn an_empty_object_is_a_default_root() {
    let l = build::<f32>(&node(vec![])).unwrap();
    assert_eq!(l.tree.len(), 1);
    assert_eq!(*l.tree.style(l.tree.root()).unwrap(), Style::<f32>::default());
    assert!(l.names.is_empty() && l.meta.is_empty());
}

#[test]
fn children_keep_file_order_at_every_level() {
    let v = node(vec![
        ("id", text("r")),
        (
            "children",
            kids(vec![
                node(vec![("id", text("a"))]),
                node(vec![
                    ("id", text("b")),
                    ("children", kids(vec![node(vec![("id", text("b1"))]), node(vec![("id", text("b2"))])])),
                ]),
                node(vec![("id", text("c"))]),
            ]),
        ),
    ]);
    let l = build::<f32>(&v).unwrap();
    let t = &l.tree;
    let order = |id: NodeId| -> Vec<&str> {
        t.children(id).iter().map(|c| l.meta[c].name.as_deref().unwrap()).collect()
    };
    assert_eq!(order(t.root()), ["a", "b", "c"]);
    assert_eq!(order(l.names["b"]), ["b1", "b2"]);
    assert_eq!(t.len(), 6);
    assert_eq!(t.parent(l.names["b2"]), Some(l.names["b"]));
}

#[test]
fn every_style_field_is_read() {
    let v = node(vec![
        ("anchor", en("UiAnchor", "BottomRight")),
        ("flow", en("UiFlow", "Column")),
        ("width", int(120)),
        ("height", fill(2.0)),
        ("margin", int(4)),
        ("padding", kids(vec![int(1), int(2)])),
        ("gap", MxValue::Float(2.5)),
        ("visible", MxValue::Bool(false)),
        ("interactive", MxValue::Bool(true)),
        ("clip", MxValue::Bool(true)),
        ("background", MxValue::Color("#102030".into())),
        ("radius", int(6)),
    ]);
    let l = build::<f32>(&v).unwrap();
    let s = l.tree.style(l.tree.root()).unwrap();
    assert_eq!(s.anchor, Anchor::BottomRight);
    assert_eq!(s.flow, Flow::Column);
    assert_eq!(s.width, Sizing::Fixed(120.0));
    assert_eq!(s.height, Sizing::Fill(2.0));
    assert_eq!(s.margin, Insets::all(4.0));
    assert_eq!(s.padding, Insets::symmetric(1.0, 2.0));
    assert_eq!(s.gap, 2.5);
    assert!(!s.visible && s.interactive && s.clip);
    assert_eq!(s.background, Color32::from_hex("#102030").unwrap());
    assert_eq!(s.radius, 6.0);
}

#[test]
fn a_loaded_tree_lays_out_like_one_built_by_hand() {
    let v = node(vec![
        ("flow", en("UiFlow", "Row")),
        ("height", int(96)),
        (
            "children",
            kids(vec![
                node(vec![("id", text("hp")), ("width", int(120)), ("height", int(24))]),
                node(vec![("id", text("pause")), ("width", int(80)), ("height", int(24))]),
                node(vec![("id", text("rest")), ("width", fill(1.0)), ("height", int(24))]),
            ]),
        ),
    ]);
    let mut l = build::<f32>(&v).unwrap();
    layout::compute(&mut l.tree, Size::new(320.0, 200.0));
    assert_eq!(rect(&l, l.tree.root()), (0.0, 0.0, 320.0, 96.0));
    assert_eq!(rect(&l, l.names["hp"]), (0.0, 0.0, 120.0, 24.0));
    assert_eq!(rect(&l, l.names["pause"]), (120.0, 0.0, 80.0, 24.0));
    assert_eq!(rect(&l, l.names["rest"]), (200.0, 0.0, 120.0, 24.0));
}

#[test]
fn f32_and_f64_builds_agree() {
    let v = node(vec![
        ("flow", en("UiFlow", "Column")),
        ("padding", int(8)),
        ("gap", int(4)),
        (
            "children",
            kids(vec![
                node(vec![("height", int(24))]),
                node(vec![("height", fill(1.0)), ("margin", kids(vec![int(2), int(2)]))]),
            ]),
        ),
    ]);
    let mut a = build::<f32>(&v).unwrap();
    let mut b = build::<f64>(&v).unwrap();
    layout::compute(&mut a.tree, Size::new(320.0, 200.0));
    layout::compute(&mut b.tree, Size::new(320.0, 200.0));
    for (ia, ib) in a.tree.children(a.tree.root()).iter().zip(b.tree.children(b.tree.root())) {
        let (ra, rb) = (a.tree.rect(*ia).unwrap(), b.tree.rect(*ib).unwrap());
        assert_eq!((ra.x as f64, ra.y as f64, ra.w as f64, ra.h as f64), (rb.x, rb.y, rb.w, rb.h));
    }
}

#[test]
fn very_deep_files_build_on_a_tiny_stack() {
    const DEPTH: usize = 50_000;
    // Building and dropping a nested value is itself recursive, so do that on
    // a big stack; the reader under test runs on a deliberately small one.
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(|| {
            let mut v = node(vec![]);
            for _ in 0..DEPTH {
                v = node(vec![("children", kids(vec![v]))]);
            }
            let len = std::thread::scope(|s| {
                std::thread::Builder::new()
                    .stack_size(128 << 10)
                    .spawn_scoped(s, || build::<f32>(&v).map(|l| l.tree.len()))
                    .unwrap()
                    .join()
                    .unwrap()
            });
            assert_eq!(len.unwrap(), DEPTH + 1);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ── names, metadata, bindings ───────────────────────────────────────────────

#[test]
fn metadata_is_kept_only_for_nodes_that_declare_some() {
    let v = node(vec![(
        "children",
        kids(vec![
            node(vec![
                ("id", text("hp")),
                ("kind", en("UiKind", "Label")),
                ("text", text("HP")),
                ("bind", text("player_health")),
            ]),
            node(vec![("width", int(10))]),
        ]),
    )]);
    let l = build::<f32>(&v).unwrap();
    let hp = l.names["hp"];
    let m = &l.meta[&hp];
    assert_eq!(m.kind.as_ref().map(|k| (k.type_name.as_str(), k.field.as_str())), Some(("UiKind", "Label")));
    assert_eq!(m.text.as_deref(), Some("HP"));
    let bind = m.bind.as_ref().unwrap();
    assert_eq!(bind.name, "player_health");
    assert_eq!(bind.key, BindKey::of("player_health"));
    let plain = l.tree.children(l.tree.root())[1];
    assert!(!l.meta.contains_key(&plain));
    assert_eq!(l.meta.len(), 1);
}

#[test]
fn bind_keys_are_fnv1a_32_and_const() {
    const HEALTH: BindKey = BindKey::of("player_health");
    assert_eq!(BindKey::of("").0, 0x811c_9dc5);
    assert_eq!(BindKey::of("a").0, 0xe40c_292c);
    assert_eq!(BindKey::of("foobar").0, 0xbf9c_f968);
    assert_eq!(HEALTH, BindKey::of("player_health"));
    assert_ne!(BindKey::of("player_health"), BindKey::of("player_mana"));
}

#[test]
fn two_nodes_may_share_one_binding_name() {
    let v = node(vec![(
        "children",
        kids(vec![node(vec![("bind", text("score"))]), node(vec![("bind", text("score"))])]),
    )]);
    assert!(build::<f32>(&v).is_ok());
}

#[test]
fn different_names_with_one_hash_are_rejected() {
    // "glbvs" and "yacxa" are both 0xa1bc9a4f under FNV-1a 32.
    assert_eq!(BindKey::of("glbvs"), BindKey::of("yacxa"));
    let v = node(vec![(
        "children",
        kids(vec![node(vec![("bind", text("glbvs"))]), node(vec![("bind", text("yacxa"))])]),
    )]);
    let (path, kind) = err_of(&v);
    assert_eq!(path, "root.children[1].bind");
    assert_eq!(kind, MxErrorKind::BindCollision { first: "glbvs".into(), second: "yacxa".into() });
}

#[test]
fn duplicate_ids_are_rejected_with_the_second_nodes_path() {
    let v = node(vec![
        ("id", text("x")),
        ("children", kids(vec![node(vec![]), node(vec![("id", text("x"))])])),
    ]);
    let (path, kind) = err_of(&v);
    assert_eq!(path, "root.children[1].id");
    assert_eq!(kind, MxErrorKind::DuplicateName("x".into()));
}

// ── field values ────────────────────────────────────────────────────────────

#[test]
fn sizing_accepts_a_number_or_a_fill_object_only() {
    let one = |w: MxValue| build::<f32>(&node(vec![("width", w)])).map(|l| l.tree.style(l.tree.root()).unwrap().width);
    assert_eq!(one(int(50)).unwrap(), Sizing::Fixed(50.0));
    assert_eq!(one(MxValue::Float(0.5)).unwrap(), Sizing::Fixed(0.5));
    assert_eq!(one(fill(3.0)).unwrap(), Sizing::Fill(3.0));
    assert_eq!(one(node(vec![("px", int(1))])).unwrap_err().kind, MxErrorKind::BadSizing);
    assert_eq!(one(node(vec![])).unwrap_err().kind, MxErrorKind::BadSizing);
    assert_eq!(
        one(text("wide")).unwrap_err().kind,
        MxErrorKind::WrongType { expected: "number or fill object", found: "string" }
    );
}

#[test]
fn insets_take_one_two_or_four_numbers() {
    let one = |m: MxValue| build::<f32>(&node(vec![("margin", m)])).map(|l| l.tree.style(l.tree.root()).unwrap().margin);
    assert_eq!(one(int(3)).unwrap(), Insets::all(3.0));
    assert_eq!(one(kids(vec![int(1), int(2)])).unwrap(), Insets::symmetric(1.0, 2.0));
    assert_eq!(one(kids(vec![int(1), int(2), int(3), int(4)])).unwrap(), Insets::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(one(kids(vec![int(1), int(2), int(3)])).unwrap_err().kind, MxErrorKind::BadInsets(3));
    assert_eq!(one(kids(vec![])).unwrap_err().kind, MxErrorKind::BadInsets(0));
    assert_eq!(
        one(kids(vec![int(1), text("x")])).unwrap_err().kind,
        MxErrorKind::WrongType { expected: "number", found: "string" }
    );
}

#[test]
fn colors_accept_three_four_six_and_eight_digit_forms() {
    let one = |c: &str| {
        build::<f32>(&node(vec![("background", MxValue::Color(c.into()))]))
            .map(|l| l.tree.style(l.tree.root()).unwrap().background)
    };
    assert_eq!(one("#F80").unwrap(), Color32::from_hex("#FF8800").unwrap());
    assert_eq!(one("#F808").unwrap(), Color32::from_hex("#FF880088").unwrap());
    assert_eq!(one("#1E1E2E").unwrap(), Color32::from_hex("#1E1E2E").unwrap());
    assert_eq!(one("#1E1E2E80").unwrap(), Color32::from_hex("#1E1E2E80").unwrap());
    for bad in ["#12", "#12345", "#GGGGGG", "#+F0000", "FF0000", "", "#"] {
        assert_eq!(one(bad).unwrap_err().kind, MxErrorKind::BadColor(bad.into()), "{bad}");
    }
}

#[test]
fn enums_must_be_the_right_type_and_a_known_variant() {
    let anchor = |v: MxValue| err_of(&node(vec![("anchor", v)])).1;
    assert_eq!(
        anchor(en("UiFlow", "Row")),
        MxErrorKind::WrongEnum { expected: "UiAnchor", found: "UiFlow".into() }
    );
    assert_eq!(
        anchor(en("UiAnchor", "Middle")),
        MxErrorKind::UnknownVariant { enum_name: "UiAnchor", found: "Middle".into() }
    );
    assert_eq!(anchor(text("Center")), MxErrorKind::WrongType { expected: "enum", found: "string" });
    let flow = err_of(&node(vec![("flow", en("UiFlow", "Grid"))])).1;
    assert_eq!(flow, MxErrorKind::UnknownVariant { enum_name: "UiFlow", found: "Grid".into() });
}

#[test]
fn kind_accepts_any_enum() {
    let l = build::<f32>(&node(vec![("kind", en("MyWidgets", "Gauge"))])).unwrap();
    assert_eq!(l.meta[&l.tree.root()].kind.as_ref().unwrap().type_name, "MyWidgets");
}

// ── errors carry paths ──────────────────────────────────────────────────────

#[test]
fn an_unknown_field_is_an_error_at_its_own_path() {
    let v = node(vec![("children", kids(vec![node(vec![]), node(vec![("widht", int(5))])]))]);
    let (path, kind) = err_of(&v);
    assert_eq!(path, "root.children[1].widht");
    assert_eq!(kind, MxErrorKind::UnknownField("widht".into()));
}

#[test]
fn wrong_types_name_the_field_and_what_was_found() {
    let cases: Vec<(&str, MxValue, &str, &str)> = vec![
        ("visible", int(1), "bool", "int"),
        ("clip", text("yes"), "bool", "string"),
        ("gap", text("2"), "number", "string"),
        ("id", int(7), "string", "int"),
        ("background", text("#FFF"), "color", "string"),
        ("children", int(0), "array", "int"),
    ];
    for (field, value, expected, found) in cases {
        let (path, kind) = err_of(&node(vec![(field, value)]));
        assert_eq!(path, format!("root.{field}"));
        assert_eq!(kind, MxErrorKind::WrongType { expected, found }, "{field}");
    }
}

#[test]
fn nodes_must_be_objects() {
    let (path, kind) = err_of(&int(3));
    assert_eq!(path, "root");
    assert_eq!(kind, MxErrorKind::WrongType { expected: "object", found: "int" });
    let (path, _) = err_of(&node(vec![("children", kids(vec![node(vec![]), text("x")]))]));
    assert_eq!(path, "root.children[1]");
}

#[test]
fn errors_deep_in_the_tree_report_the_full_path() {
    // Indices differ per level (2, then 1) so a reversed path cannot pass.
    let bad = node(vec![("margin", kids(vec![int(1), int(2), int(3)]))]);
    let v = node(vec![(
        "children",
        kids(vec![
            node(vec![]),
            node(vec![]),
            node(vec![("children", kids(vec![node(vec![]), bad]))]),
        ]),
    )]);
    let (path, kind) = err_of(&v);
    assert_eq!(path, "root.children[2].children[1].margin");
    assert_eq!(kind, MxErrorKind::BadInsets(3));
}

#[test]
fn errors_display_their_path_and_message() {
    let e = build::<f32>(&node(vec![("widht", int(1))])).err().unwrap();
    assert_eq!(e.to_string(), "root.widht: unknown field `widht`");
}
