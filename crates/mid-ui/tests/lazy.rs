// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "lazy.rs"
// ============================================================================
//! Lazy row ranges for long uniform lists.

use mid_ui::LazyRows;

fn rows() -> LazyRows<f32> {
    LazyRows::new(100, 20.0, 0.0)
}

#[test]
fn the_extent_is_all_rows_and_the_gaps_between_them() {
    assert_eq!(rows().extent(), 2000.0);
    assert_eq!(LazyRows::new(3, 20.0, 5.0).extent(), 70.0);
    assert_eq!(LazyRows::new(1, 20.0, 5.0).extent(), 20.0);
    assert_eq!(LazyRows::new(0, 20.0, 5.0).extent(), 0.0);
}

#[test]
fn row_tops_step_by_the_row_height_plus_the_gap() {
    let r = LazyRows::new(10, 20.0, 5.0);
    assert_eq!(r.row_top(0), 0.0);
    assert_eq!(r.row_top(4), 100.0);
}

#[test]
fn the_top_of_the_list_shows_the_first_rows() {
    assert_eq!(rows().visible(0.0, 100.0), 0..5);
}

#[test]
fn a_partly_visible_row_at_each_edge_counts() {
    // Window 30..130: row 1 (20..40) and row 6 (120..140) are cut by it.
    assert_eq!(rows().visible(30.0, 100.0), 1..7);
}

#[test]
fn a_row_that_ends_exactly_at_the_top_edge_does_not_count() {
    assert_eq!(rows().visible(40.0, 100.0), 2..7);
    // With a gap the first row can end exactly at the edge: row 0 is 0..20.
    let r = LazyRows::new(100, 20.0, 5.0);
    assert_eq!(r.visible(20.0, 50.0), 1..3);
}

#[test]
fn a_gap_alone_at_the_top_does_not_count() {
    // Rows 20 tall with 5 between: row 0 is 0..20, row 1 is 25..45.
    // The window starts at 22, in the gap, and is 50 tall: it ends at 72.
    let r = LazyRows::new(100, 20.0, 5.0);
    assert_eq!(r.visible(22.0, 50.0), 1..3);
}

#[test]
fn overscroll_shows_the_last_rows() {
    assert_eq!(rows().visible(1.0e9, 100.0), 95..100);
    assert_eq!(rows().visible(f32::INFINITY, 100.0), 0..5);
}

#[test]
fn a_negative_or_nan_scroll_counts_as_the_top() {
    assert_eq!(rows().visible(-50.0, 100.0), 0..5);
    assert_eq!(rows().visible(f32::NAN, 100.0), 0..5);
}

#[test]
fn a_window_taller_than_the_list_shows_everything() {
    assert_eq!(LazyRows::new(3, 20.0, 5.0).visible(0.0, 1000.0), 0..3);
    assert_eq!(LazyRows::new(3, 20.0, 5.0).visible(40.0, 1000.0), 0..3);
}

#[test]
fn an_empty_list_or_unusable_rows_show_nothing() {
    assert_eq!(LazyRows::new(0, 20.0, 0.0).visible(0.0, 100.0), 0..0);
    assert_eq!(LazyRows::new(10, 0.0, 0.0).visible(0.0, 100.0), 0..0);
    assert_eq!(LazyRows::new(10, f32::NAN, 0.0).visible(0.0, 100.0), 0..0);
    assert_eq!(LazyRows::new(10, -4.0, 0.0).visible(0.0, 100.0), 0..0);
}

#[test]
fn a_zero_height_window_shows_no_rows() {
    let r = rows().visible(30.0, 0.0);
    assert!(r.is_empty(), "{r:?}");
}

#[test]
fn a_hundred_thousand_rows_stay_cheap_and_exact() {
    let r = LazyRows::<f64>::new(100_000, 18.0, 2.0);
    assert_eq!(r.extent(), 100_000.0 * 18.0 + 99_999.0 * 2.0);
    let v = r.visible(1_000_000.0, 400.0);
    assert_eq!(v, 50_000..50_020);
    assert!(v.len() <= 21);
}

#[test]
fn f32_and_f64_agree() {
    for (scroll, view) in [(0.0, 100.0), (30.0, 100.0), (22.0, 50.0), (1.0e9, 100.0)] {
        let a = LazyRows::<f32>::new(100, 20.0, 5.0).visible(scroll as f32, view as f32);
        let b = LazyRows::<f64>::new(100, 20.0, 5.0).visible(scroll, view);
        assert_eq!(a, b, "scroll {scroll}, view {view}");
    }
}
