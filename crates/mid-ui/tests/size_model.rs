// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "solve.rs"
// ============================================================================
//! The size model: `Fit`, minimum and maximum sizes, shrinking, aspect ratio
//! and text-driven sizes, against a fake measure.

use mid_ui::{
    layout, measure::NoMeasure, Anchor, ContentKey, Flow, Insets, Measure, NodeId, Real, Size,
    Sizing, Style, Tree,
};

// ── The fake measure ──────────────────────────────────────────────────────────
//
// Every character is `CW` wide, every line `LH` tall, a word is a run of
// characters and one space separates two words. A content key picks a list of
// word lengths. Wrapping is greedy, as the contract says a first
// implementation is allowed to be.

const CW: f64 = 10.0;
const LH: f64 = 20.0;

struct Fake {
    texts: Vec<Vec<u32>>,
}

impl Fake {
    /// Key 0: three words of five characters. Line width 170, minimum 50.
    /// Heights: 20 at 170 and above, 40 from 110, 60 below.
    fn new() -> Self {
        Fake { texts: vec![vec![5, 5, 5], vec![10], vec![]] }
    }
}

impl<R: Real> Measure<R> for Fake {
    fn line_width(&self, key: ContentKey) -> R {
        let w = &self.texts[key.0 as usize];
        if w.is_empty() {
            return R::ZERO;
        }
        let chars: u32 = w.iter().sum::<u32>() + (w.len() as u32 - 1);
        R::from_f64(chars as f64 * CW)
    }

    fn min_content_width(&self, key: ContentKey) -> R {
        let longest = self.texts[key.0 as usize].iter().copied().max().unwrap_or(0);
        R::from_f64(longest as f64 * CW)
    }

    fn wrapped_height(&self, key: ContentKey, max_width: R) -> R {
        let w = &self.texts[key.0 as usize];
        if w.is_empty() {
            return R::ZERO;
        }
        let limit = max_width.to_f64();
        let (mut lines, mut cur) = (1_u32, 0_u32);
        for &word in w {
            if cur == 0 {
                cur = word;
            } else if (cur + 1 + word) as f64 * CW <= limit {
                cur += 1 + word;
            } else {
                lines += 1;
                cur = word;
            }
        }
        R::from_f64(lines as f64 * LH)
    }
}

const A: ContentKey = ContentKey(0);
const B: ContentKey = ContentKey(1);
const EMPTY: ContentKey = ContentKey(2);

type S = Style<f32>;

fn fit() -> S {
    S { width: Sizing::Fit, height: Sizing::Fit, ..S::default() }
}

fn fixed(w: f32, h: f32) -> S {
    S { width: Sizing::Fixed(w), height: Sizing::Fixed(h), ..S::default() }
}

fn text(t: &mut Tree<f32>, parent: NodeId, key: ContentKey, style: S) -> NodeId {
    let id = t.add_child(parent, style).unwrap();
    assert!(t.set_content(id, Some(key)));
    id
}

fn text_root(t: &mut Tree<f32>, key: ContentKey, style: S) -> NodeId {
    let root = t.root();
    text(t, root, key, style)
}

fn rect(t: &Tree<f32>, id: NodeId) -> (f32, f32, f32, f32) {
    let r = t.rect(id).unwrap();
    (r.x, r.y, r.w, r.h)
}

fn solve(t: &mut Tree<f32>, w: f32, h: f32) {
    layout::compute_with(t, Size::new(w, h), &Fake::new());
}

// ── Fit on a leaf ─────────────────────────────────────────────────────────────

#[test]
fn a_fit_leaf_takes_its_line_width_and_its_height_at_that_width() {
    let mut t = Tree::new(S::default());
    let a = text_root(&mut t, A, fit());
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 170.0, 20.0));
}

#[test]
fn a_fit_leaf_wraps_when_the_space_is_short() {
    let mut t = Tree::new(S::default());
    let a = text_root(&mut t, A, fit());
    solve(&mut t, 100.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 100.0, 60.0));
    solve(&mut t, 110.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 110.0, 40.0));
}

#[test]
fn a_fit_leaf_never_goes_below_its_minimum_content_width() {
    let mut t = Tree::new(S::default());
    let a = text_root(&mut t, A, fit());
    solve(&mut t, 40.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 50.0, 60.0));
}

#[test]
fn fit_without_content_is_zero_and_without_a_measure_it_is_zero_too() {
    let mut t = Tree::new(S::default());
    let plain = t.add_child(t.root(), fit()).unwrap();
    let a = text_root(&mut t, A, fit());
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, plain), (0.0, 0.0, 0.0, 0.0));
    layout::compute(&mut t, Size::new(400.0, 300.0));
    assert_eq!(rect(&t, a), (0.0, 0.0, 0.0, 0.0));
    layout::compute_with(&mut t, Size::new(400.0, 300.0), &NoMeasure);
    assert_eq!(rect(&t, a), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn empty_text_has_no_size() {
    let mut t = Tree::new(S::default());
    let e = text_root(&mut t, EMPTY, fit());
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, e), (0.0, 0.0, 0.0, 0.0));
}

// ── Fit on a container ────────────────────────────────────────────────────────

#[test]
fn a_fit_container_hugs_its_children_padding_margins_and_gap_included() {
    let mut t = Tree::new(S::default());
    let panel = t
        .add_child(
            t.root(),
            S { flow: Flow::Column, padding: Insets::all(5.0), gap: 3.0, ..fit() },
        )
        .unwrap();
    let a = text(&mut t, panel, A, S { margin: Insets::all(2.0), ..fit() });
    let b = t.add_child(panel, fixed(30.0, 10.0)).unwrap();
    solve(&mut t, 400.0, 300.0);
    // width: the wider outer child (170 + 4) plus 10 of padding; height:
    // (20 + 4) + 3 + 10 plus 10 of padding.
    assert_eq!(rect(&t, panel), (0.0, 0.0, 184.0, 47.0));
    assert_eq!(rect(&t, a), (7.0, 7.0, 170.0, 20.0));
    assert_eq!(rect(&t, b), (5.0, 32.0, 30.0, 10.0));
}

#[test]
fn a_row_container_sums_its_children_on_the_main_axis() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, gap: 4.0, ..fit() }).unwrap();
    let a = text(&mut t, row, A, fit());
    let b = t.add_child(row, fixed(30.0, 50.0)).unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, row), (0.0, 0.0, 204.0, 50.0));
    assert_eq!(rect(&t, a), (0.0, 0.0, 170.0, 20.0));
    assert_eq!(rect(&t, b), (174.0, 0.0, 30.0, 50.0));
}

#[test]
fn a_fit_container_height_follows_the_text_wrapped_at_the_width_it_got() {
    let mut t = Tree::new(S::default());
    let panel = t
        .add_child(t.root(), S { flow: Flow::Column, width: Sizing::Fixed(100.0), ..fit() })
        .unwrap();
    let a = text(&mut t, panel, A, fit());
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 100.0, 60.0));
    assert_eq!(rect(&t, panel), (0.0, 0.0, 100.0, 60.0));
}

#[test]
fn a_node_with_visible_children_ignores_its_own_content() {
    let mut t = Tree::new(S::default());
    let n = text_root(&mut t, A, fit());
    let kid = t.add_child(n, fixed(30.0, 10.0)).unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, n), (0.0, 0.0, 30.0, 10.0));
    assert_eq!(rect(&t, kid), (0.0, 0.0, 30.0, 10.0));
    // Hide the child and the text comes back.
    t.style_mut(kid).unwrap().visible = false;
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, n), (0.0, 0.0, 170.0, 20.0));
}

// ── Flow with Fit, Fill, shrinking ────────────────────────────────────────────

#[test]
fn a_row_gives_fill_what_fit_and_fixed_leave() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(400.0, 50.0) }).unwrap();
    let a = text(&mut t, row, A, fit());
    let f = t.add_child(row, S::default()).unwrap();
    let x = t.add_child(row, fixed(30.0, 10.0)).unwrap();
    solve(&mut t, 500.0, 300.0);
    assert_eq!(rect(&t, a).2, 170.0);
    assert_eq!(rect(&t, f), (170.0, 0.0, 200.0, 50.0));
    assert_eq!(rect(&t, x), (370.0, 0.0, 30.0, 10.0));
}

#[test]
fn fit_children_that_overflow_shrink_toward_their_floors_in_proportion() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(200.0, 100.0) }).unwrap();
    let a1 = text(&mut t, row, A, fit());
    let a2 = text(&mut t, row, A, fit());
    solve(&mut t, 500.0, 300.0);
    // 340 wanted in 200: 140 short, each can give up 120, so each gives 70.
    assert_eq!(rect(&t, a1), (0.0, 0.0, 100.0, 60.0));
    assert_eq!(rect(&t, a2), (100.0, 0.0, 100.0, 60.0));
}

#[test]
fn a_child_already_at_its_floor_does_not_shrink() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(200.0, 100.0) }).unwrap();
    let a = text(&mut t, row, A, fit());
    let b = text(&mut t, row, B, fit());
    solve(&mut t, 500.0, 300.0);
    // A wants 170 (floor 50), B wants 100 (floor 100): 70 short, A pays it all.
    assert_eq!(rect(&t, a).2, 100.0);
    assert_eq!(rect(&t, b), (100.0, 0.0, 100.0, 20.0));
}

#[test]
fn when_even_the_floors_do_not_fit_the_row_overflows() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(120.0, 100.0) }).unwrap();
    let a = text(&mut t, row, A, fit());
    let b = text(&mut t, row, B, fit());
    solve(&mut t, 500.0, 300.0);
    // Floors are 50 and 100, 150 in 120.
    assert_eq!(rect(&t, a).2, 50.0);
    assert_eq!(rect(&t, b), (50.0, 0.0, 100.0, 20.0));
}

#[test]
fn a_fixed_child_never_shrinks() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(200.0, 100.0) }).unwrap();
    let f = t.add_child(row, fixed(150.0, 10.0)).unwrap();
    let a = text(&mut t, row, A, fit());
    solve(&mut t, 500.0, 300.0);
    assert_eq!(rect(&t, f).2, 150.0);
    // 170 wanted in the 50 that is left, floor 50.
    assert_eq!(rect(&t, a).2, 50.0);
}

// ── Minimum and maximum ───────────────────────────────────────────────────────

fn row_of_two_fills(a: S, b: S) -> (Tree<f32>, NodeId, NodeId) {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(300.0, 50.0) }).unwrap();
    let na = t.add_child(row, a).unwrap();
    let nb = t.add_child(row, b).unwrap();
    solve(&mut t, 500.0, 300.0);
    (t, na, nb)
}

#[test]
fn a_fill_with_a_maximum_hands_the_rest_to_its_sibling() {
    let (t, a, b) = row_of_two_fills(S { max_width: Some(50.0), ..S::default() }, S::default());
    assert_eq!(rect(&t, a), (0.0, 0.0, 50.0, 50.0));
    assert_eq!(rect(&t, b), (50.0, 0.0, 250.0, 50.0));
}

#[test]
fn a_fill_with_a_minimum_takes_it_from_its_sibling() {
    let (t, a, b) = row_of_two_fills(S { min_width: 200.0, ..S::default() }, S::default());
    assert_eq!(rect(&t, a), (0.0, 0.0, 200.0, 50.0));
    assert_eq!(rect(&t, b), (200.0, 0.0, 100.0, 50.0));
}

#[test]
fn weights_still_apply_among_the_siblings_that_are_not_clamped() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(400.0, 50.0) }).unwrap();
    let a = t
        .add_child(row, S { width: Sizing::Fill(1.0), max_width: Some(40.0), ..S::default() })
        .unwrap();
    let b = t.add_child(row, S { width: Sizing::Fill(1.0), ..S::default() }).unwrap();
    let c = t.add_child(row, S { width: Sizing::Fill(3.0), ..S::default() }).unwrap();
    solve(&mut t, 500.0, 300.0);
    // A is held at 40; the other 360 splits 1 : 3.
    assert_eq!(rect(&t, a).2, 40.0);
    assert_eq!(rect(&t, b).2, 90.0);
    assert_eq!(rect(&t, c).2, 270.0);
}

#[test]
fn minimum_beats_maximum_when_they_conflict() {
    let mut t = Tree::new(S::default());
    let n = t
        .add_child(
            t.root(),
            S { width: Sizing::Fixed(100.0), min_width: 80.0, max_width: Some(60.0), ..S::default() },
        )
        .unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, n).2, 80.0);
}

#[test]
fn limits_clamp_a_fixed_size_and_a_fill_on_the_cross_axis() {
    let mut t = Tree::new(S::default());
    let big = t
        .add_child(t.root(), S { max_width: Some(60.0), max_height: Some(40.0), ..fixed(100.0, 100.0) })
        .unwrap();
    let small = t.add_child(t.root(), S { min_width: 40.0, min_height: 30.0, ..fixed(10.0, 10.0) }).unwrap();
    let wide = t.add_child(t.root(), S { max_width: Some(120.0), ..S::default() }).unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, big).2, 60.0);
    assert_eq!(rect(&t, big).3, 40.0);
    assert_eq!(rect(&t, small).2, 40.0);
    assert_eq!(rect(&t, small).3, 30.0);
    assert_eq!(rect(&t, wide).2, 120.0);
}

#[test]
fn a_maximum_caps_a_fit_size_and_a_minimum_lifts_it() {
    let mut t = Tree::new(S::default());
    let capped = text_root(&mut t, A, S { max_width: Some(120.0), ..fit() });
    let lifted = text_root(&mut t, B, S { min_width: 130.0, ..fit() });
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, capped).2, 120.0);
    assert_eq!(rect(&t, capped).3, 40.0);
    assert_eq!(rect(&t, lifted).2, 130.0);
}

// ── Aspect ratio ──────────────────────────────────────────────────────────────

#[test]
fn aspect_derives_a_fit_height_from_the_width() {
    let mut t = Tree::new(S::default());
    let n = t
        .add_child(
            t.root(),
            S { width: Sizing::Fixed(100.0), height: Sizing::Fit, aspect: Some(2.0), ..S::default() },
        )
        .unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, n), (0.0, 0.0, 100.0, 50.0));
}

#[test]
fn aspect_derives_a_fit_width_from_the_height() {
    let mut t = Tree::new(S::default());
    let n = t
        .add_child(
            t.root(),
            S { width: Sizing::Fit, height: Sizing::Fixed(40.0), aspect: Some(2.0), ..S::default() },
        )
        .unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, n), (0.0, 0.0, 80.0, 40.0));
}

#[test]
fn a_zero_or_negative_aspect_is_ignored() {
    let mut t = Tree::new(S::default());
    let n = t
        .add_child(
            t.root(),
            S { width: Sizing::Fixed(100.0), height: Sizing::Fit, aspect: Some(0.0), ..S::default() },
        )
        .unwrap();
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, n), (0.0, 0.0, 100.0, 0.0));
}

// ── Cross axis and anchors ────────────────────────────────────────────────────

#[test]
fn a_fit_child_on_the_cross_axis_is_anchored_by_its_own_size() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(400.0, 100.0) }).unwrap();
    let a = text(&mut t, row, A, S { anchor: Anchor::Center, ..fit() });
    solve(&mut t, 500.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 40.0, 170.0, 20.0));
}

// ── The text loop ─────────────────────────────────────────────────────────────

#[test]
fn after_one_pass_every_text_already_has_the_height_of_its_own_width() {
    // The harness stack measures each text at the width it last got, solves
    // again, and repeats until no height moves. With the measure in reach of
    // the solver the first pass is already that fixed point: re-measuring at
    // the solved width changes nothing, and a second pass changes nothing.
    for width in [400.0_f32, 170.0, 110.0, 100.0, 60.0, 50.0, 40.0, 5.0] {
        let mut t = Tree::new(S::default());
        let col = t.add_child(t.root(), S { flow: Flow::Column, gap: 2.0, ..fit() }).unwrap();
        let a1 = text(&mut t, col, A, fit());
        let b1 = text(&mut t, col, B, fit());
        let a2 = text(&mut t, col, A, S { padding: Insets::all(1.0), ..fit() });
        solve(&mut t, width, 1000.0);
        let first: Vec<_> = [col, a1, b1, a2].iter().map(|&n| rect(&t, n)).collect();
        for (id, key) in [(a1, A), (b1, B), (a2, A)] {
            let r = t.rect(id).unwrap();
            let pad = if id == a2 { 2.0 } else { 0.0 };
            let again: f32 = Measure::<f32>::wrapped_height(&Fake::new(), key, r.w - pad);
            assert_eq!(r.h - pad, again, "width {width}: text height moved");
        }
        solve(&mut t, width, 1000.0);
        let second: Vec<_> = [col, a1, b1, a2].iter().map(|&n| rect(&t, n)).collect();
        assert_eq!(first, second, "width {width}: a second pass moved something");
    }
}

// ── Hidden nodes, depth, precision ────────────────────────────────────────────

#[test]
fn hidden_nodes_take_no_space_and_keep_their_old_rectangle() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(400.0, 50.0) }).unwrap();
    let a = text(&mut t, row, A, fit());
    let hidden = text(&mut t, row, B, S { visible: false, ..fit() });
    let b = t.add_child(row, fixed(30.0, 10.0)).unwrap();
    solve(&mut t, 500.0, 300.0);
    assert_eq!(rect(&t, hidden), (0.0, 0.0, 0.0, 0.0));
    assert_eq!(rect(&t, b), (170.0, 0.0, 30.0, 10.0));
    assert_eq!(rect(&t, a).2, 170.0);
}

#[test]
fn a_very_deep_fit_chain_does_not_use_the_call_stack() {
    let mut t = Tree::new(S::default());
    let mut parent = t.root();
    for _ in 0..20_000 {
        parent = t.add_child(parent, S { flow: Flow::Column, padding: Insets::all(0.5), ..fit() }).unwrap();
    }
    let leaf = t.add_child(parent, fixed(10.0, 6.0)).unwrap();
    solve(&mut t, 1.0e6, 1.0e6);
    // 20,000 levels each add one unit of padding on both sides.
    assert_eq!(rect(&t, leaf).2, 10.0);
    let top = t.children(t.root())[0];
    assert_eq!(rect(&t, top).2, 10.0 + 20_000.0);
    assert_eq!(rect(&t, top).3, 6.0 + 20_000.0);
}

#[test]
fn f64_gives_the_same_answers_as_f32() {
    fn scene<R: Real>(width: f64) -> Vec<(f64, f64, f64, f64)> {
        let r = R::from_f64;
        let mut t = Tree::<R>::new(Style::default());
        let row = t
            .add_child(
                t.root(),
                Style { flow: Flow::Row, width: Sizing::Fixed(r(200.0)), height: Sizing::Fixed(r(100.0)), ..Style::default() },
            )
            .unwrap();
        let fit_style = || Style::<R> { width: Sizing::Fit, height: Sizing::Fit, ..Style::default() };
        let a1 = t.add_child(row, fit_style()).unwrap();
        let a2 = t.add_child(row, fit_style()).unwrap();
        t.set_content(a1, Some(A));
        t.set_content(a2, Some(B));
        layout::compute_with(&mut t, Size::new(r(width), r(300.0)), &Fake::new());
        [row, a1, a2]
            .iter()
            .map(|&n| {
                let q = t.rect(n).unwrap();
                (q.x.to_f64(), q.y.to_f64(), q.w.to_f64(), q.h.to_f64())
            })
            .collect()
    }
    for w in [500.0, 200.0, 120.0] {
        assert_eq!(scene::<f32>(w), scene::<f64>(w), "viewport width {w}");
    }
}

// ── The content key on the tree ───────────────────────────────────────────────

#[test]
fn content_is_stored_per_node_and_goes_away_with_it() {
    let mut t = Tree::new(S::default());
    let n = t.add_child(t.root(), fit()).unwrap();
    assert_eq!(t.content(n), None);
    assert!(t.set_content(n, Some(A)));
    assert_eq!(t.content(n), Some(A));
    assert!(t.set_content(n, None));
    assert_eq!(t.content(n), None);
    assert!(t.remove(n));
    assert!(!t.set_content(n, Some(B)), "a stale id is refused");
    assert_eq!(t.content(n), None);
    // A new node in the reused slot starts without content.
    let m = t.add_child(t.root(), fit()).unwrap();
    assert_eq!(t.content(m), None);
}

// ── More edges ────────────────────────────────────────────────────────────────

struct Nonsense;

impl Measure<f32> for Nonsense {
    fn line_width(&self, _: ContentKey) -> f32 { f32::NAN }
    fn min_content_width(&self, _: ContentKey) -> f32 { -5.0 }
    fn wrapped_height(&self, _: ContentKey, _: f32) -> f32 { f32::INFINITY }
}

#[test]
fn a_measure_that_answers_nonsense_counts_as_zero() {
    let mut t = Tree::new(S::default());
    let a = text_root(&mut t, A, fit());
    layout::compute_with(&mut t, Size::new(400.0, 300.0), &Nonsense);
    assert_eq!(rect(&t, a), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn fills_that_are_all_held_at_their_maximum_leave_the_space_unused() {
    let (t, a, b) = row_of_two_fills(
        S { max_width: Some(50.0), ..S::default() },
        S { max_width: Some(50.0), ..S::default() },
    );
    assert_eq!(rect(&t, a), (0.0, 0.0, 50.0, 50.0));
    assert_eq!(rect(&t, b), (50.0, 0.0, 50.0, 50.0));
}

#[test]
fn a_fill_with_no_weight_keeps_its_minimum_and_the_rest_goes_on() {
    let (t, a, b) = row_of_two_fills(
        S { width: Sizing::Fill(0.0), min_width: 30.0, ..S::default() },
        S::default(),
    );
    assert_eq!(rect(&t, a), (0.0, 0.0, 30.0, 50.0));
    assert_eq!(rect(&t, b), (30.0, 0.0, 270.0, 50.0));
}

#[test]
fn in_an_overflowing_row_fit_shrinks_and_fill_gets_nothing() {
    let mut t = Tree::new(S::default());
    let row = t.add_child(t.root(), S { flow: Flow::Row, ..fixed(100.0, 50.0) }).unwrap();
    let a = text(&mut t, row, A, fit());
    let f = t.add_child(row, S::default()).unwrap();
    solve(&mut t, 500.0, 300.0);
    // The height of the wrapped text does not shrink, so it spills out of the 50.
    assert_eq!(rect(&t, a), (0.0, 0.0, 100.0, 60.0));
    assert_eq!(rect(&t, f), (100.0, 0.0, 0.0, 50.0));
}

#[test]
fn a_column_gives_fill_what_fit_leaves_on_the_vertical_axis() {
    let mut t = Tree::new(S::default());
    let col = t.add_child(t.root(), S { flow: Flow::Column, ..fixed(200.0, 300.0) }).unwrap();
    let a = text(&mut t, col, A, fit());
    let f = t.add_child(col, S::default()).unwrap();
    solve(&mut t, 500.0, 400.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 170.0, 20.0));
    assert_eq!(rect(&t, f), (0.0, 20.0, 200.0, 280.0));
}

#[test]
fn heights_do_not_shrink_below_their_content_in_a_short_column() {
    let mut t = Tree::new(S::default());
    let col = t.add_child(t.root(), S { flow: Flow::Column, ..fixed(100.0, 30.0) }).unwrap();
    let a1 = text(&mut t, col, A, fit());
    let a2 = text(&mut t, col, A, fit());
    solve(&mut t, 500.0, 400.0);
    assert_eq!(rect(&t, a1), (0.0, 0.0, 100.0, 60.0));
    assert_eq!(rect(&t, a2), (0.0, 60.0, 100.0, 60.0));
}

#[test]
fn padding_on_a_text_leaf_adds_to_both_sizes_and_the_text_wraps_inside_it() {
    let mut t = Tree::new(S::default());
    let a = text_root(&mut t, A, S { padding: Insets::all(3.0), ..fit() });
    solve(&mut t, 400.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 176.0, 26.0));
    // 106 wide leaves 100 for the text, which wraps to three lines.
    solve(&mut t, 106.0, 300.0);
    assert_eq!(rect(&t, a), (0.0, 0.0, 106.0, 66.0));
}
