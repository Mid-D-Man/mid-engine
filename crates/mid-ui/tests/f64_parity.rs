// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "real.rs"
// ============================================================================
//! f32/f64 agreement of the generic core.

use mid_ui::hit::hit_test;
use mid_ui::{layout, Anchor, Flow, Insets, NodeId, Point, Real, Size, Sizing, Style, Tree};

// One scenario written once against `Real`; run at both precisions.
fn scenario<R: Real>(weight_a: f64, weight_b: f64) -> (Vec<[f64; 4]>, Option<usize>) {
    let r = R::from_f64;
    let mut t = Tree::<R>::new(Style {
        flow: Flow::Column,
        padding: Insets::all(r(8.0)),
        gap: r(4.0),
        ..Style::default()
    });
    let header = t
        .add_child(t.root(), Style { height: Sizing::Fixed(r(24.0)), interactive: true, ..Style::default() })
        .unwrap();
    let body = t
        .add_child(
            t.root(),
            Style { flow: Flow::Row, gap: r(2.0), height: Sizing::Fill(r(1.0)), ..Style::default() },
        )
        .unwrap();
    let mut order = vec![t.root(), header, body];
    order.push(
        t.add_child(body, Style { width: Sizing::Fixed(r(50.0)), interactive: true, ..Style::default() })
            .unwrap(),
    );
    order.push(
        t.add_child(
            body,
            Style {
                width: Sizing::Fill(r(weight_a)),
                height: Sizing::Fixed(r(20.0)),
                anchor: Anchor::Center,
                margin: Insets::symmetric(r(2.0), r(2.0)),
                interactive: true,
                ..Style::default()
            },
        )
        .unwrap(),
    );
    order.push(
        t.add_child(body, Style { width: Sizing::Fill(r(weight_b)), interactive: true, ..Style::default() })
            .unwrap(),
    );

    layout::compute(&mut t, Size::new(r(320.0), r(200.0)));
    let rects = order
        .iter()
        .map(|id: &NodeId| {
            let q = t.rect(*id).unwrap();
            [q.x.to_f64(), q.y.to_f64(), q.w.to_f64(), q.h.to_f64()]
        })
        .collect();
    let hit = hit_test(&t, Point::new(r(150.0), r(100.0))).and_then(|h| order.iter().position(|id| *id == h));
    (rects, hit)
}

#[test]
fn dyadic_weights_give_bit_identical_results_at_both_precisions() {
    let (a, hit_a) = scenario::<f32>(1.0, 3.0);
    let (b, hit_b) = scenario::<f64>(1.0, 3.0);
    assert_eq!(a, b);
    assert_eq!(hit_a, hit_b);
    assert!(hit_a.is_some());
}

#[test]
fn non_dyadic_weights_agree_within_f32_precision() {
    let (a, hit_a) = scenario::<f32>(1.0, 2.0);
    let (b, hit_b) = scenario::<f64>(1.0, 2.0);
    for (ra, rb) in a.iter().zip(&b) {
        for k in 0..4 {
            assert!((ra[k] - rb[k]).abs() < 1e-3, "{ra:?} vs {rb:?}");
        }
    }
    assert_eq!(hit_a, hit_b);
}

#[test]
fn f64_keeps_sub_f32_precision_at_large_coordinates() {
    // 2^24 + 1 is not representable in f32 but is in f64.
    let big = 16_777_217.0_f64;
    let mut t = Tree::<f64>::new(Style::default());
    let c = t
        .add_child(
            t.root(),
            Style { width: Sizing::Fixed(1.0), height: Sizing::Fixed(1.0), anchor: Anchor::BottomRight, ..Style::default() },
        )
        .unwrap();
    layout::compute(&mut t, Size::new(big, big));
    assert_eq!(t.rect(c).unwrap().x, big - 1.0);
}
