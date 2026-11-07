// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "layout.rs"
// ============================================================================
//! Anchor, margin, padding and row/column layout.

use mid_ui::{layout, Anchor, Flow, Insets, NodeId, Size, Sizing, Style, Tree};

type S = Style<f32>;

const VIEW: Size<f32> = Size::new(200.0, 100.0);

fn fixed(w: f32, h: f32) -> S {
    S { width: Sizing::Fixed(w), height: Sizing::Fixed(h), ..S::default() }
}

fn rect(t: &Tree<f32>, id: NodeId) -> (f32, f32, f32, f32) {
    let r = t.rect(id).unwrap();
    (r.x, r.y, r.w, r.h)
}

fn run(t: &mut Tree<f32>) {
    layout::compute(t, VIEW);
}

#[test]
fn default_root_fills_the_viewport() {
    let mut t = Tree::new(S::default());
    run(&mut t);
    assert_eq!(rect(&t, t.root()), (0.0, 0.0, 200.0, 100.0));
}

#[test]
fn fixed_root_is_anchored_inside_the_viewport() {
    let mut t = Tree::new(S { anchor: Anchor::Center, ..fixed(100.0, 50.0) });
    run(&mut t);
    assert_eq!(rect(&t, t.root()), (50.0, 25.0, 100.0, 50.0));
}

#[test]
fn stack_anchors_place_a_fixed_child_on_a_three_by_three_grid() {
    let cases = [
        (Anchor::TopLeft, (0.0, 0.0)),
        (Anchor::Top, (75.0, 0.0)),
        (Anchor::TopRight, (150.0, 0.0)),
        (Anchor::Left, (0.0, 40.0)),
        (Anchor::Center, (75.0, 40.0)),
        (Anchor::Right, (150.0, 40.0)),
        (Anchor::BottomLeft, (0.0, 80.0)),
        (Anchor::Bottom, (75.0, 80.0)),
        (Anchor::BottomRight, (150.0, 80.0)),
    ];
    for (anchor, (x, y)) in cases {
        let mut t = Tree::new(S::default());
        let c = t.add_child(t.root(), S { anchor, ..fixed(50.0, 20.0) }).unwrap();
        run(&mut t);
        assert_eq!(rect(&t, c), (x, y, 50.0, 20.0), "{anchor:?}");
    }
}

#[test]
fn margin_keeps_a_fixed_child_off_the_edges() {
    let mut t = Tree::new(S::default());
    let tl = t.add_child(t.root(), S { margin: Insets::all(10.0), ..fixed(50.0, 20.0) }).unwrap();
    let br = t
        .add_child(
            t.root(),
            S { anchor: Anchor::BottomRight, margin: Insets::all(10.0), ..fixed(50.0, 20.0) },
        )
        .unwrap();
    run(&mut t);
    assert_eq!(rect(&t, tl), (10.0, 10.0, 50.0, 20.0));
    assert_eq!(rect(&t, br), (140.0, 70.0, 50.0, 20.0));
}

#[test]
fn fill_child_shrinks_by_its_margin() {
    let mut t = Tree::new(S::default());
    let c = t.add_child(t.root(), S { margin: Insets::symmetric(10.0, 5.0), ..S::default() }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, c), (10.0, 5.0, 180.0, 90.0));
}

#[test]
fn padding_shrinks_the_area_children_see() {
    let mut t = Tree::new(S { padding: Insets::all(10.0), ..S::default() });
    let fill = t.add_child(t.root(), S::default()).unwrap();
    let mid = t.add_child(t.root(), S { anchor: Anchor::Center, ..fixed(20.0, 20.0) }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, fill), (10.0, 10.0, 180.0, 80.0));
    assert_eq!(rect(&t, mid), (90.0, 40.0, 20.0, 20.0));
}

#[test]
fn row_places_fixed_children_left_to_right_with_gap() {
    let mut t = Tree::new(S { flow: Flow::Row, gap: 10.0, ..S::default() });
    let kid = S { width: Sizing::Fixed(40.0), ..S::default() };
    let ids: Vec<_> = (0..3).map(|_| t.add_child(t.root(), kid).unwrap()).collect();
    run(&mut t);
    assert_eq!(rect(&t, ids[0]), (0.0, 0.0, 40.0, 100.0));
    assert_eq!(rect(&t, ids[1]), (50.0, 0.0, 40.0, 100.0));
    assert_eq!(rect(&t, ids[2]), (100.0, 0.0, 40.0, 100.0));
}

#[test]
fn row_shares_leftover_width_by_weight() {
    let mut t = Tree::new(S { flow: Flow::Row, ..S::default() });
    let a = t.add_child(t.root(), S { width: Sizing::Fixed(50.0), ..S::default() }).unwrap();
    let b = t.add_child(t.root(), S { width: Sizing::Fill(1.0), ..S::default() }).unwrap();
    let c = t.add_child(t.root(), S { width: Sizing::Fill(3.0), ..S::default() }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, a), (0.0, 0.0, 50.0, 100.0));
    assert_eq!(rect(&t, b), (50.0, 0.0, 37.5, 100.0));
    assert_eq!(rect(&t, c), (87.5, 0.0, 112.5, 100.0));
}

#[test]
fn row_counts_margins_and_gaps_before_sharing_space() {
    let mut t = Tree::new(S { flow: Flow::Row, gap: 10.0, ..S::default() });
    let m = Insets::new(5.0, 0.0, 5.0, 0.0);
    let a = t.add_child(t.root(), S { margin: m, width: Sizing::Fixed(30.0), ..S::default() }).unwrap();
    let b = t.add_child(t.root(), S { margin: m, width: Sizing::Fixed(30.0), ..S::default() }).unwrap();
    let c = t.add_child(t.root(), S { margin: m, width: Sizing::Fill(1.0), ..S::default() }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, a).0, 5.0);
    assert_eq!(rect(&t, b).0, 55.0);
    // 200 - (3 margin pairs = 30) - (60 fixed) - (2 gaps = 20) = 90 left for c.
    assert_eq!(rect(&t, c), (105.0, 0.0, 90.0, 100.0));
}

#[test]
fn row_cross_axis_uses_the_vertical_anchor_only() {
    let mut t = Tree::new(S { flow: Flow::Row, ..S::default() });
    let top = t.add_child(t.root(), S { anchor: Anchor::Top, ..fixed(30.0, 20.0) }).unwrap();
    let mid = t.add_child(t.root(), S { anchor: Anchor::Center, ..fixed(30.0, 20.0) }).unwrap();
    let bot = t.add_child(t.root(), S { anchor: Anchor::BottomRight, ..fixed(30.0, 20.0) }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, top), (0.0, 0.0, 30.0, 20.0));
    assert_eq!(rect(&t, mid), (30.0, 40.0, 30.0, 20.0));
    assert_eq!(rect(&t, bot), (60.0, 80.0, 30.0, 20.0));
}

#[test]
fn column_stacks_top_to_bottom_and_stretches_width() {
    let mut t = Tree::new(S { flow: Flow::Column, gap: 5.0, ..S::default() });
    let kid = S { height: Sizing::Fixed(20.0), ..S::default() };
    let ids: Vec<_> = (0..3).map(|_| t.add_child(t.root(), kid).unwrap()).collect();
    run(&mut t);
    assert_eq!(rect(&t, ids[0]), (0.0, 0.0, 200.0, 20.0));
    assert_eq!(rect(&t, ids[1]), (0.0, 25.0, 200.0, 20.0));
    assert_eq!(rect(&t, ids[2]), (0.0, 50.0, 200.0, 20.0));
}

#[test]
fn column_cross_axis_uses_the_horizontal_anchor_only() {
    let mut t = Tree::new(S { flow: Flow::Column, ..S::default() });
    let l = t.add_child(t.root(), S { anchor: Anchor::Left, ..fixed(20.0, 10.0) }).unwrap();
    let c = t.add_child(t.root(), S { anchor: Anchor::Center, ..fixed(20.0, 10.0) }).unwrap();
    let r = t.add_child(t.root(), S { anchor: Anchor::BottomRight, ..fixed(20.0, 10.0) }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, l), (0.0, 0.0, 20.0, 10.0));
    assert_eq!(rect(&t, c), (90.0, 10.0, 20.0, 10.0));
    assert_eq!(rect(&t, r), (180.0, 20.0, 20.0, 10.0));
}

#[test]
fn column_fill_takes_what_fixed_siblings_leave() {
    let mut t = Tree::new(S { flow: Flow::Column, ..S::default() });
    let a = t.add_child(t.root(), S { height: Sizing::Fixed(40.0), ..S::default() }).unwrap();
    let b = t.add_child(t.root(), S::default()).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, a), (0.0, 0.0, 200.0, 40.0));
    assert_eq!(rect(&t, b), (0.0, 40.0, 200.0, 60.0));
}

#[test]
fn hidden_children_take_no_space() {
    let mut t = Tree::new(S { flow: Flow::Row, gap: 10.0, ..S::default() });
    let kid = S { width: Sizing::Fixed(40.0), ..S::default() };
    let a = t.add_child(t.root(), kid).unwrap();
    let hidden = t.add_child(t.root(), S { visible: false, ..kid }).unwrap();
    let c = t.add_child(t.root(), kid).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, a).0, 0.0);
    assert_eq!(rect(&t, c).0, 50.0);
    assert_eq!(rect(&t, hidden), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn a_hidden_node_keeps_its_old_rectangle() {
    let mut t = Tree::new(S::default());
    let c = t.add_child(t.root(), fixed(50.0, 20.0)).unwrap();
    run(&mut t);
    let before = rect(&t, c);
    t.style_mut(c).unwrap().visible = false;
    layout::compute(&mut t, Size::new(400.0, 300.0));
    assert_eq!(rect(&t, c), before);
}

#[test]
fn overflowing_fixed_children_leave_fill_children_empty() {
    let mut t = Tree::new(S { flow: Flow::Row, ..S::default() });
    let a = t.add_child(t.root(), S { width: Sizing::Fixed(150.0), ..S::default() }).unwrap();
    let b = t.add_child(t.root(), S { width: Sizing::Fixed(150.0), ..S::default() }).unwrap();
    let c = t.add_child(t.root(), S { width: Sizing::Fill(1.0), ..S::default() }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, a).0, 0.0);
    assert_eq!(rect(&t, b).0, 150.0);
    assert_eq!(rect(&t, c), (300.0, 0.0, 0.0, 100.0));
}

#[test]
fn zero_and_negative_weights_get_no_main_axis_space() {
    let mut t = Tree::new(S { flow: Flow::Row, ..S::default() });
    let w0 = t.add_child(t.root(), S { width: Sizing::Fill(0.0), ..S::default() }).unwrap();
    let neg = t.add_child(t.root(), S { width: Sizing::Fill(-2.0), ..S::default() }).unwrap();
    let one = t.add_child(t.root(), S { width: Sizing::Fill(1.0), ..S::default() }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, w0).2, 0.0);
    assert_eq!(rect(&t, neg).2, 0.0);
    assert_eq!(rect(&t, one).2, 200.0);
}

#[test]
fn negative_fixed_sizes_clamp_to_zero() {
    let mut t = Tree::new(S::default());
    let c = t.add_child(t.root(), fixed(-5.0, -1.0)).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, c), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn nested_flows_resolve_against_their_own_content_area() {
    let mut t = Tree::new(S { flow: Flow::Column, ..S::default() });
    let header = t.add_child(t.root(), S { height: Sizing::Fixed(30.0), ..S::default() }).unwrap();
    let body = t
        .add_child(t.root(), S { flow: Flow::Row, padding: Insets::all(10.0), ..S::default() })
        .unwrap();
    let left = t.add_child(body, S::default()).unwrap();
    let right = t.add_child(body, S::default()).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, header), (0.0, 0.0, 200.0, 30.0));
    assert_eq!(rect(&t, body), (0.0, 30.0, 200.0, 70.0));
    // body content area is (10, 40, 180, 50); two equal fills split 180.
    assert_eq!(rect(&t, left), (10.0, 40.0, 90.0, 50.0));
    assert_eq!(rect(&t, right), (100.0, 40.0, 90.0, 50.0));
}

#[test]
fn recomputing_after_a_style_or_viewport_change_updates_rects() {
    let mut t = Tree::new(S::default());
    let c = t.add_child(t.root(), S { anchor: Anchor::Center, ..fixed(50.0, 20.0) }).unwrap();
    run(&mut t);
    assert_eq!(rect(&t, c), (75.0, 40.0, 50.0, 20.0));
    t.style_mut(c).unwrap().anchor = Anchor::BottomRight;
    run(&mut t);
    assert_eq!(rect(&t, c), (150.0, 80.0, 50.0, 20.0));
    layout::compute(&mut t, Size::new(400.0, 300.0));
    assert_eq!(rect(&t, c), (350.0, 280.0, 50.0, 20.0));
}

#[test]
fn deep_trees_do_not_overflow_the_stack() {
    let mut t = Tree::new(S::default());
    let mut parent = t.root();
    for _ in 0..50_000 {
        parent = t.add_child(parent, S::default()).unwrap();
    }
    run(&mut t);
    assert_eq!(rect(&t, parent), (0.0, 0.0, 200.0, 100.0));
}

#[test]
fn non_finite_inputs_never_panic() {
    let mut t = Tree::new(S { flow: Flow::Row, gap: f32::NAN, ..S::default() });
    t.add_child(t.root(), S { width: Sizing::Fixed(f32::NAN), ..S::default() }).unwrap();
    t.add_child(t.root(), S { width: Sizing::Fill(f32::INFINITY), ..S::default() }).unwrap();
    t.add_child(t.root(), S { width: Sizing::Fill(f32::NAN), ..S::default() }).unwrap();
    layout::compute(&mut t, Size::new(f32::INFINITY, f32::NAN));
    layout::compute(&mut t, Size::new(-10.0, -10.0));
}
