// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "hit.rs" and "draw.rs"
// ============================================================================
//! Hit testing and draw-list generation.

use mid_ui::hit::{claims_pointer, hit_test};
use mid_ui::{draw, layout, Anchor, Color32, DrawCmd, NodeId, Point, Rect, Size, Sizing, Style, Tree};

type S = Style<f32>;

fn fixed(w: f32, h: f32) -> S {
    S { width: Sizing::Fixed(w), height: Sizing::Fixed(h), ..S::default() }
}

fn live(w: f32, h: f32) -> S {
    S { interactive: true, ..fixed(w, h) }
}

fn laid_out(mut t: Tree<f32>) -> Tree<f32> {
    layout::compute(&mut t, Size::new(200.0, 100.0));
    t
}

fn at(x: f32, y: f32) -> Point<f32> {
    Point::new(x, y)
}

fn commands(t: &Tree<f32>) -> Vec<DrawCmd<f32>> {
    let mut out = Vec::new();
    draw::build_draw_list(t, &mut out);
    out
}

// ── hit testing ─────────────────────────────────────────────────────────────

#[test]
fn nothing_interactive_means_no_hit_and_no_claim() {
    let mut t = Tree::new(S::default());
    t.add_child(t.root(), fixed(50.0, 50.0)).unwrap();
    let t = laid_out(t);
    assert_eq!(hit_test(&t, at(10.0, 10.0)), None);
    assert!(!claims_pointer(&t, at(10.0, 10.0)));
}

#[test]
fn the_later_sibling_wins_where_two_overlap() {
    let mut t = Tree::new(S::default());
    let a = t.add_child(t.root(), live(100.0, 100.0)).unwrap();
    let b = t.add_child(t.root(), live(50.0, 50.0)).unwrap();
    let t = laid_out(t);
    assert_eq!(hit_test(&t, at(10.0, 10.0)), Some(b));
    assert_eq!(hit_test(&t, at(75.0, 75.0)), Some(a));
    assert!(claims_pointer(&t, at(75.0, 75.0)));
}

#[test]
fn a_descendant_beats_its_ancestor() {
    let mut t = Tree::new(S::default());
    let parent = t.add_child(t.root(), live(100.0, 100.0)).unwrap();
    let child = t.add_child(parent, live(20.0, 20.0)).unwrap();
    let t = laid_out(t);
    assert_eq!(hit_test(&t, at(5.0, 5.0)), Some(child));
    assert_eq!(hit_test(&t, at(50.0, 50.0)), Some(parent));
}

#[test]
fn non_interactive_nodes_let_the_pointer_fall_through() {
    let mut t = Tree::new(S::default());
    let below = t.add_child(t.root(), live(100.0, 100.0)).unwrap();
    t.add_child(t.root(), fixed(100.0, 100.0)).unwrap();
    let t = laid_out(t);
    assert_eq!(hit_test(&t, at(10.0, 10.0)), Some(below));
}

#[test]
fn hidden_subtrees_are_invisible_to_the_pointer() {
    let mut t = Tree::new(S::default());
    let shown = t.add_child(t.root(), live(100.0, 100.0)).unwrap();
    let cover = t.add_child(t.root(), live(100.0, 100.0)).unwrap();
    let inner = t.add_child(cover, live(10.0, 10.0)).unwrap();
    let mut t = laid_out(t);
    assert_eq!(hit_test(&t, at(5.0, 5.0)), Some(inner));
    t.style_mut(cover).unwrap().visible = false;
    assert_eq!(hit_test(&t, at(5.0, 5.0)), Some(shown));
}

#[test]
fn edges_are_half_open() {
    let mut t = Tree::new(S::default());
    let a = t.add_child(t.root(), live(50.0, 50.0)).unwrap();
    let t = laid_out(t);
    assert_eq!(hit_test(&t, at(0.0, 0.0)), Some(a));
    assert_eq!(hit_test(&t, at(50.0, 25.0)), None);
    assert_eq!(hit_test(&t, at(25.0, 50.0)), None);
}

#[test]
fn a_clipping_parent_hides_children_outside_its_rect() {
    let build = |clip: bool| {
        let mut t = Tree::new(S::default());
        let parent = t.add_child(t.root(), S { clip, ..fixed(50.0, 50.0) }).unwrap();
        let child = t.add_child(parent, live(100.0, 100.0)).unwrap();
        (laid_out(t), child)
    };
    let (clipped, _) = build(true);
    assert_eq!(hit_test(&clipped, at(75.0, 75.0)), None);
    let (open, child) = build(false);
    assert_eq!(hit_test(&open, at(75.0, 75.0)), Some(child));
    let (clipped, child) = build(true);
    assert_eq!(hit_test(&clipped, at(25.0, 25.0)), Some(child));
}

#[test]
fn removed_nodes_stop_receiving_hits() {
    let mut t = Tree::new(S::default());
    let a = t.add_child(t.root(), live(50.0, 50.0)).unwrap();
    let mut t = laid_out(t);
    assert_eq!(hit_test(&t, at(1.0, 1.0)), Some(a));
    assert!(t.remove(a));
    assert_eq!(hit_test(&t, at(1.0, 1.0)), None);
}

// ── draw list ───────────────────────────────────────────────────────────────

fn filled(w: f32, h: f32, c: Color32) -> S {
    S { background: c, ..fixed(w, h) }
}

#[test]
fn fills_come_out_in_paint_order() {
    let mut t = Tree::new(S { background: Color32::RED, ..S::default() });
    t.add_child(t.root(), filled(50.0, 20.0, Color32::BLUE)).unwrap();
    t.add_child(t.root(), S { anchor: Anchor::BottomRight, ..filled(10.0, 10.0, Color32::GREEN) }).unwrap();
    let list = commands(&laid_out(t));
    assert_eq!(
        list,
        vec![
            DrawCmd::FillRect { rect: Rect::new(0.0, 0.0, 200.0, 100.0), color: Color32::RED, radius: 0.0 },
            DrawCmd::FillRect { rect: Rect::new(0.0, 0.0, 50.0, 20.0), color: Color32::BLUE, radius: 0.0 },
            DrawCmd::FillRect { rect: Rect::new(190.0, 90.0, 10.0, 10.0), color: Color32::GREEN, radius: 0.0 },
        ]
    );
}

#[test]
fn transparent_and_empty_fills_are_skipped() {
    let mut t = Tree::new(S::default());
    t.add_child(t.root(), fixed(50.0, 50.0)).unwrap();
    t.add_child(t.root(), filled(0.0, 50.0, Color32::RED)).unwrap();
    t.add_child(t.root(), filled(50.0, 0.0, Color32::RED)).unwrap();
    assert!(commands(&laid_out(t)).is_empty());
}

#[test]
fn clip_wraps_exactly_the_clipping_subtree() {
    let mut t = Tree::new(S::default());
    let panel = t.add_child(t.root(), S { clip: true, ..filled(80.0, 80.0, Color32::WHITE) }).unwrap();
    t.add_child(panel, filled(10.0, 10.0, Color32::RED)).unwrap();
    t.add_child(panel, filled(20.0, 20.0, Color32::BLUE)).unwrap();
    t.add_child(t.root(), filled(5.0, 5.0, Color32::GREEN)).unwrap();
    let list = commands(&laid_out(t));
    let colors: Vec<Option<Color32>> = list
        .iter()
        .map(|c| match c {
            DrawCmd::FillRect { color, .. } => Some(*color),
            _ => None,
        })
        .collect();
    assert_eq!(
        colors,
        vec![
            Some(Color32::WHITE),
            None, // PushClip
            Some(Color32::RED),
            Some(Color32::BLUE),
            None, // PopClip
            Some(Color32::GREEN),
        ]
    );
    assert_eq!(list[1], DrawCmd::PushClip(Rect::new(0.0, 0.0, 80.0, 80.0)));
    assert_eq!(list[4], DrawCmd::PopClip);
}

#[test]
fn nested_clips_stay_balanced() {
    let mut t = Tree::new(S::default());
    let outer = t.add_child(t.root(), S { clip: true, ..fixed(100.0, 100.0) }).unwrap();
    let inner = t.add_child(outer, S { clip: true, ..fixed(50.0, 50.0) }).unwrap();
    t.add_child(inner, filled(10.0, 10.0, Color32::RED)).unwrap();
    let list = commands(&laid_out(t));
    let mut depth = 0_i32;
    for c in &list {
        match c {
            DrawCmd::PushClip(_) => depth += 1,
            DrawCmd::PopClip => depth -= 1,
            _ => {}
        }
        assert!(depth >= 0);
    }
    assert_eq!(depth, 0);
    assert_eq!(list.len(), 5);
}

#[test]
fn hidden_subtrees_produce_nothing() {
    let mut t = Tree::new(S::default());
    let a = t.add_child(t.root(), S { clip: true, ..filled(50.0, 50.0, Color32::RED) }).unwrap();
    t.add_child(a, filled(10.0, 10.0, Color32::BLUE)).unwrap();
    let mut t = laid_out(t);
    assert_eq!(commands(&t).len(), 4);
    t.style_mut(a).unwrap().visible = false;
    assert!(commands(&t).is_empty());
}

#[test]
fn radius_is_carried_onto_the_fill() {
    let mut t = Tree::new(S::default());
    t.add_child(t.root(), S { radius: 6.0, ..filled(40.0, 20.0, Color32::WHITE) }).unwrap();
    match commands(&laid_out(t))[0] {
        DrawCmd::FillRect { radius, .. } => assert_eq!(radius, 6.0),
        other => panic!("unexpected command {other:?}"),
    }
}

#[test]
fn building_appends_to_the_buffer_without_clearing_it() {
    let mut t = Tree::new(S::default());
    t.add_child(t.root(), filled(10.0, 10.0, Color32::RED)).unwrap();
    let t = laid_out(t);
    let mut out = Vec::new();
    draw::build_draw_list(&t, &mut out);
    draw::build_draw_list(&t, &mut out);
    assert_eq!(out.len(), 2);
}

#[test]
fn deep_trees_draw_and_hit_without_overflowing_the_stack() {
    let mut t = Tree::new(S::default());
    let mut parent: NodeId = t.root();
    for _ in 0..50_000 {
        parent = t.add_child(parent, S { background: Color32::RED, interactive: true, ..S::default() }).unwrap();
    }
    let t = laid_out(t);
    assert_eq!(commands(&t).len(), 50_000);
    assert_eq!(hit_test(&t, at(1.0, 1.0)), Some(parent));
}
