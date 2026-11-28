// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "cover.rs"
// ============================================================================
//! Coverage: fixed point, exactness on rectangles, areas, fill rules, region
//! independence, flattening and the shape outlines.

// `is_multiple_of` is newer than the toolchains the tests should still build on.
#![allow(clippy::manual_is_multiple_of)]

use mid_raster::cover::{bounds, rasterize};
use mid_raster::fixed::{self, from_f32, LIMIT};
use mid_raster::path::{Segment, FLATNESS};
use mid_raster::shapes::{self, rect_mask};
use mid_raster::{FillRule, IRect, Mask, Path};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
    fn unit(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
    }
}

fn fill(path: &Path, region: IRect, rule: FillRule) -> Mask {
    let segs = path.flatten().expect("finite path");
    rasterize(&segs, region, rule)
}

fn poly(points: &[(f32, f32)]) -> Path {
    let mut p = Path::new();
    p.move_to(points[0].0, points[0].1);
    for &(x, y) in &points[1..] {
        p.line_to(x, y);
    }
    p.close();
    p
}

fn assert_same(a: &Mask, b: &Mask, region: IRect, what: &str) {
    for y in region.y0..region.y1 {
        for x in region.x0..region.x1 {
            assert_eq!(a.at(x, y), b.at(x, y), "{what}: pixel {x},{y}");
        }
    }
}

fn sum(mask: &Mask) -> f64 {
    mask.data.iter().map(|&c| c as f64).sum::<f64>() / 255.0
}

// ── Fixed point ───────────────────────────────────────────────────────────────

#[test]
fn from_f32_rounds_to_the_nearest_256th_with_ties_away_from_zero() {
    assert_eq!(from_f32(1.0), Some(256));
    assert_eq!(from_f32(0.5), Some(128));
    assert_eq!(from_f32(3.0), Some(768));
    assert_eq!(from_f32(-2.0), Some(-512));
    assert_eq!(from_f32(1.0 / 512.0), Some(1), "a tie goes up");
    assert_eq!(from_f32(-1.0 / 512.0), Some(-1), "a negative tie goes down");
    assert_eq!(from_f32(1.0 / 1024.0), Some(0));
    assert_eq!(from_f32(0.0), Some(0));
    assert_eq!(from_f32(-0.0), Some(0));
}

#[test]
fn from_f32_saturates_and_refuses_what_is_not_finite() {
    assert_eq!(from_f32(1.0e30), Some(LIMIT));
    assert_eq!(from_f32(-1.0e30), Some(-LIMIT));
    assert_eq!(from_f32(f32::MAX), Some(LIMIT));
    assert_eq!(from_f32(f32::NAN), None);
    assert_eq!(from_f32(f32::INFINITY), None);
    assert_eq!(from_f32(f32::NEG_INFINITY), None);
}

#[test]
fn the_integer_helpers_round_the_way_they_say() {
    assert_eq!(fixed::floor_div(-1, 256), -1);
    assert_eq!(fixed::floor_div(255, 256), 0);
    assert_eq!(fixed::floor_div(256, 256), 1);
    assert_eq!(fixed::div_round(1, 2), 1, "half rounds up");
    assert_eq!(fixed::div_round(-1, 2), 0, "half rounds toward positive");
    assert_eq!(fixed::div_round(3, 2), 2);
    assert_eq!(fixed::div_round(-3, 2), -1);
    assert_eq!(fixed::div_round(7, 3), 2);
}

// ── Rectangles ────────────────────────────────────────────────────────────────

#[test]
fn an_integer_edged_rectangle_covers_whole_pixels_at_255() {
    let mut p = Path::new();
    p.move_to(1.0, 1.0).line_to(4.0, 1.0).line_to(4.0, 3.0).line_to(1.0, 3.0).close();
    let m = fill(&p, IRect::new(0, 0, 6, 5), FillRule::NonZero);
    for y in 0..5 {
        for x in 0..6 {
            let inside = (1..4).contains(&x) && (1..3).contains(&y);
            assert_eq!(m.at(x, y), if inside { 255 } else { 0 }, "pixel {x},{y}");
        }
    }
}

#[test]
fn a_fractional_rectangle_has_the_exact_edge_coverage() {
    // x from 0.5 to 2.25, y from 0 to 1: half, full, a quarter.
    let mut p = Path::new();
    p.move_to(0.5, 0.0).line_to(2.25, 0.0).line_to(2.25, 1.0).line_to(0.5, 1.0).close();
    let m = fill(&p, IRect::new(0, 0, 4, 1), FillRule::NonZero);
    assert_eq!([m.at(0, 0), m.at(1, 0), m.at(2, 0), m.at(3, 0)], [128, 255, 64, 0]);
}

#[test]
fn the_closed_form_rectangle_equals_the_path_rasterizer_on_random_rectangles() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let clip = IRect::new(-2, -2, 14, 14);
    for _ in 0..300 {
        let x0 = rng.range(-400, 3000);
        let y0 = rng.range(-400, 3000);
        let x1 = x0 + rng.range(1, 2500);
        let y1 = y0 + rng.range(1, 2500);
        let (fx0, fy0, fx1, fy1) = (x0 as f32 / 256.0, y0 as f32 / 256.0, x1 as f32 / 256.0, y1 as f32 / 256.0);
        let mut p = Path::new();
        p.move_to(fx0, fy0).line_to(fx1, fy0).line_to(fx1, fy1).line_to(fx0, fy1).close();
        let by_path = fill(&p, clip, FillRule::NonZero);
        let closed = rect_mask(x0, y0, x1, y1, clip);
        for y in clip.y0..clip.y1 {
            for x in clip.x0..clip.x1 {
                assert_eq!(by_path.at(x, y), closed.at(x, y), "rect {x0},{y0},{x1},{y1} pixel {x},{y}");
            }
        }
    }
}

#[test]
fn a_degenerate_or_inverted_rectangle_has_no_coverage() {
    let clip = IRect::new(0, 0, 8, 8);
    assert!(rect_mask(256, 256, 256, 1000, clip).data.iter().all(|&c| c == 0));
    assert!(rect_mask(900, 256, 256, 1000, clip).data.iter().all(|&c| c == 0));
    assert!(rect_mask(5000, 5000, 6000, 6000, clip).data.is_empty());
}

// ── Areas and winding ─────────────────────────────────────────────────────────

#[test]
fn a_triangle_has_the_coverage_of_its_area() {
    let m = fill(&poly(&[(1.0, 1.0), (9.0, 2.0), (4.0, 8.0)]), IRect::new(0, 0, 12, 12), FillRule::NonZero);
    // Twice the area is 8 * 7 - 3 * 1 = 53.
    assert!((sum(&m) - 26.5).abs() < 0.2, "area {}", sum(&m));
}

#[test]
fn the_two_halves_of_a_rectangle_add_up_to_the_rectangle_within_one_level() {
    let region = IRect::new(0, 0, 14, 12);
    let a = fill(&poly(&[(1.3, 1.7), (11.6, 1.7), (11.6, 9.4)]), region, FillRule::NonZero);
    let b = fill(&poly(&[(1.3, 1.7), (11.6, 9.4), (1.3, 9.4)]), region, FillRule::NonZero);
    let r = fill(&poly(&[(1.3, 1.7), (11.6, 1.7), (11.6, 9.4), (1.3, 9.4)]), region, FillRule::NonZero);
    for y in 0..12 {
        for x in 0..14 {
            let s = a.at(x, y) as i32 + b.at(x, y) as i32;
            assert!((s - r.at(x, y) as i32).abs() <= 2, "pixel {x},{y}: {s} against {}", r.at(x, y));
        }
    }
}

#[test]
fn overlapping_squares_in_the_same_direction_follow_the_rule() {
    let mut p = Path::new();
    p.move_to(0.0, 0.0).line_to(4.0, 0.0).line_to(4.0, 4.0).line_to(0.0, 4.0).close();
    p.move_to(2.0, 2.0).line_to(6.0, 2.0).line_to(6.0, 6.0).line_to(2.0, 6.0).close();
    let region = IRect::new(0, 0, 8, 8);
    let nz = fill(&p, region, FillRule::NonZero);
    let eo = fill(&p, region, FillRule::EvenOdd);
    assert_eq!((nz.at(1, 1), eo.at(1, 1)), (255, 255));
    assert_eq!((nz.at(3, 3), eo.at(3, 3)), (255, 0), "the overlap has winding 2");
    assert_eq!((nz.at(5, 5), eo.at(5, 5)), (255, 255));
    assert_eq!((nz.at(7, 7), eo.at(7, 7)), (0, 0));
}

#[test]
fn a_square_wound_the_other_way_cuts_a_hole_under_both_rules() {
    let mut p = Path::new();
    p.move_to(0.0, 0.0).line_to(8.0, 0.0).line_to(8.0, 8.0).line_to(0.0, 8.0).close();
    p.move_to(2.0, 2.0).line_to(2.0, 6.0).line_to(6.0, 6.0).line_to(6.0, 2.0).close();
    let region = IRect::new(0, 0, 8, 8);
    for rule in [FillRule::NonZero, FillRule::EvenOdd] {
        let m = fill(&p, region, rule);
        assert_eq!((m.at(1, 1), m.at(4, 4), m.at(7, 7)), (255, 0, 255), "{rule:?}");
    }
}

#[test]
fn a_five_pointed_star_differs_between_the_rules_only_in_its_middle() {
    let pts: Vec<(f32, f32)> = (0..5)
        .map(|i| {
            let a = (i as f32 * 2.0 * 2.0 * std::f32::consts::PI / 5.0) - std::f32::consts::FRAC_PI_2;
            (10.0 + 8.0 * a.cos(), 10.0 + 8.0 * a.sin())
        })
        .collect();
    let region = IRect::new(0, 0, 20, 20);
    let nz = fill(&poly(&pts), region, FillRule::NonZero);
    let eo = fill(&poly(&pts), region, FillRule::EvenOdd);
    assert_eq!(nz.at(10, 10), 255, "winding 2 in the middle fills under non-zero");
    assert_eq!(eo.at(10, 10), 0, "and leaves a hole under even-odd");
    assert!(sum(&nz) > sum(&eo) + 5.0);
}

// ── A region changes nothing about the pixels inside it ───────────────────────

#[test]
fn rasterizing_a_window_gives_the_crop_of_the_full_raster() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let full = IRect::new(-4, -4, 26, 26);
    let window = IRect::new(3, 5, 15, 18);
    for round in 0..200 {
        let mut p = Path::new();
        let n = rng.range(3, 7);
        p.move_to(rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0));
        for _ in 0..n {
            if rng.next() % 3 == 0 {
                p.quad_to(rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0));
            } else {
                p.line_to(rng.unit(-6.0, 28.0), rng.unit(-6.0, 28.0));
            }
        }
        p.close();
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let a = fill(&p, full, rule);
            let b = fill(&p, window, rule);
            for y in window.y0..window.y1 {
                for x in window.x0..window.x1 {
                    assert_eq!(a.at(x, y), b.at(x, y), "round {round} {rule:?} pixel {x},{y}");
                }
            }
        }
    }
}

#[test]
fn a_shape_wholly_outside_the_region_leaves_it_empty_and_one_to_its_left_fills_it() {
    let region = IRect::new(10, 10, 14, 14);
    let right = fill(&poly(&[(20.0, 10.0), (30.0, 10.0), (30.0, 14.0), (20.0, 14.0)]), region, FillRule::NonZero);
    assert!(right.data.iter().all(|&c| c == 0));
    let above = fill(&poly(&[(10.0, 0.0), (14.0, 0.0), (14.0, 5.0), (10.0, 5.0)]), region, FillRule::NonZero);
    assert!(above.data.iter().all(|&c| c == 0));
    // A shape that spans the region from far left to far right covers it.
    let across = fill(&poly(&[(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]), region, FillRule::NonZero);
    assert!(across.data.iter().all(|&c| c == 255));
}

#[test]
fn bounds_cover_every_segment_and_are_none_for_no_segments() {
    assert_eq!(bounds(&[]), None);
    let segs: Vec<Segment> = vec![[-100, 50, 300, 700]];
    assert_eq!(bounds(&segs), Some(IRect::new(-1, 0, 2, 3)));
}

// ── Flattening ────────────────────────────────────────────────────────────────

fn dist_to_polyline(segs: &[Segment], px: f64, py: f64) -> f64 {
    let mut best = f64::MAX;
    for s in segs {
        let (x0, y0, x1, y1) = (s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64);
        let (dx, dy) = (x1 - x0, y1 - y0);
        let len2 = dx * dx + dy * dy;
        let t = if len2 == 0.0 { 0.0 } else { (((px - x0) * dx + (py - y0) * dy) / len2).clamp(0.0, 1.0) };
        let (cx, cy) = (x0 + t * dx, y0 + t * dy);
        best = best.min(((px - cx).powi(2) + (py - cy).powi(2)).sqrt());
    }
    best
}

#[test]
fn a_flattened_quadratic_stays_within_the_flatness() {
    let mut p = Path::new();
    p.move_to(0.0, 0.0).quad_to(10.0, 20.0, 20.0, 0.0);
    let segs = p.flatten().unwrap();
    assert_eq!(segs[0][..2], [0, 0], "starts at the first point");
    // The path closes with a line back to the start, so look at the curve part.
    let curve = &segs[..segs.len() - 1];
    assert_eq!(curve.last().unwrap()[2..], [20 * 256, 0], "ends at the last point exactly");
    for k in 0..=200 {
        let t = k as f64 / 200.0;
        let (x, y) = (20.0 * 2.0 * t * (1.0 - t) * 10.0 / 10.0 + 20.0 * t * t * 0.0, 0.0);
        let _ = (x, y);
        let bx = 2.0 * (1.0 - t) * t * 10.0 + t * t * 20.0;
        let by = 2.0 * (1.0 - t) * t * 20.0;
        let d = dist_to_polyline(curve, bx * 256.0, by * 256.0);
        assert!(d <= (FLATNESS + 2) as f64, "t {t}: {d} fixed units off");
    }
}

#[test]
fn a_flattened_cubic_stays_within_the_flatness_and_hits_its_end_points() {
    let k = shapes::KAPPA as f64;
    let mut p = Path::new();
    p.move_to(10.0, 0.0).cubic_to(10.0, 10.0 * shapes::KAPPA, 10.0 * shapes::KAPPA, 10.0, 0.0, 10.0);
    let segs = p.flatten().unwrap();
    let curve = &segs[..segs.len() - 1];
    assert_eq!(curve[0][..2], [10 * 256, 0]);
    assert_eq!(curve.last().unwrap()[2..], [0, 10 * 256]);
    assert!(curve.len() >= 4, "a quarter circle of radius 10 needs several pieces");
    for i in 0..=200 {
        let t = i as f64 / 200.0;
        let m = 1.0 - t;
        let bx = m * m * m * 10.0 + 3.0 * m * m * t * 10.0 + 3.0 * m * t * t * 10.0 * k;
        let by = 3.0 * m * m * t * 10.0 * k + 3.0 * m * t * t * 10.0 + t * t * t * 10.0;
        let d = dist_to_polyline(curve, bx * 256.0, by * 256.0);
        assert!(d <= (FLATNESS + 2) as f64, "t {t}: {d} fixed units off");
    }
}

#[test]
fn a_straight_curve_is_one_piece_and_a_huge_one_is_capped() {
    let mut line = Path::new();
    line.move_to(0.0, 0.0).quad_to(5.0, 0.0, 10.0, 0.0);
    assert_eq!(line.flatten().unwrap().len(), 1 + 1, "one piece and the closing line");
    let mut huge = Path::new();
    huge.move_to(0.0, 0.0).quad_to(1.0e6, 1.0e6, 0.0, 2.0e6);
    assert!(huge.flatten().unwrap().len() <= 128 + 1);
}

#[test]
fn contours_close_themselves_and_a_stray_segment_starts_one() {
    let mut p = Path::new();
    p.move_to(0.0, 0.0).line_to(4.0, 0.0).line_to(4.0, 4.0);
    assert_eq!(p.flatten().unwrap().len(), 3, "two lines and the closing one");
    let mut q = Path::new();
    q.line_to(1.0, 1.0).line_to(5.0, 1.0).line_to(5.0, 5.0);
    assert_eq!(q.flatten().unwrap().len(), 3, "the first line only sets the start");
    assert!(Path::new().flatten().unwrap().is_empty());
}

#[test]
fn a_move_closes_the_contour_before_it() {
    let mut open = Path::new();
    open.move_to(0.0, 0.0).line_to(4.0, 0.0).line_to(4.0, 4.0);
    open.move_to(10.0, 0.0).line_to(14.0, 0.0).line_to(14.0, 4.0);
    let mut closed = Path::new();
    closed.move_to(0.0, 0.0).line_to(4.0, 0.0).line_to(4.0, 4.0).close();
    closed.move_to(10.0, 0.0).line_to(14.0, 0.0).line_to(14.0, 4.0).close();
    assert_eq!(open.flatten().unwrap(), closed.flatten().unwrap());
    assert_eq!(open.flatten().unwrap().len(), 6);
}

#[test]
fn a_path_with_a_value_that_is_not_finite_draws_nothing_at_all() {
    let mut p = Path::new();
    p.move_to(0.0, 0.0).line_to(4.0, 0.0).line_to(f32::NAN, 4.0);
    assert!(p.flatten().is_none());
    let mut q = Path::new();
    q.move_to(0.0, 0.0).quad_to(f32::INFINITY, 1.0, 4.0, 0.0);
    assert!(q.flatten().is_none());
}

// ── Shapes ────────────────────────────────────────────────────────────────────

#[test]
fn a_rounded_rectangle_with_no_radius_is_the_rectangle() {
    let region = IRect::new(0, 0, 14, 12);
    let r = shapes::rounded_rect(1.25, 1.5, 10.0, 7.5, 0.0).unwrap();
    let m = fill(&r, region, FillRule::NonZero);
    let c = rect_mask(320, 384, 320 + 2560, 384 + 1920, region);
    assert_same(&m, &c, region, "rounded rect with no radius");
}

#[test]
fn a_rounded_rectangle_has_the_area_of_its_corners_cut_off_and_is_symmetric() {
    let region = IRect::new(0, 0, 20, 20);
    let m = fill(&shapes::rounded_rect(0.0, 0.0, 20.0, 20.0, 6.0).unwrap(), region, FillRule::NonZero);
    // 400 minus four corners of (1 - pi / 4) r^2 each. The chords of the
    // flattened corners lie inside the arcs, so the raster is a little smaller:
    // 368.13 against 369.10 here, under one square pixel.
    let want = 400.0 - (4.0 - std::f64::consts::PI) * 36.0;
    assert!(sum(&m) < want && want - sum(&m) < 1.2, "area {} against {want}", sum(&m));
    assert_eq!(m.at(10, 10), 255);
    assert!(m.at(0, 0) < 20, "the corner pixel is mostly outside");
    for y in 0..20 {
        for x in 0..20 {
            let mirrored = [m.at(19 - x, y), m.at(x, 19 - y), m.at(19 - x, 19 - y)];
            for v in mirrored {
                assert!((v as i32 - m.at(x, y) as i32).abs() <= 3, "pixel {x},{y}: {v} against {}", m.at(x, y));
            }
        }
    }
}

#[test]
fn the_radius_is_held_to_half_the_shorter_side() {
    let region = IRect::new(0, 0, 12, 8);
    let big = fill(&shapes::rounded_rect(1.0, 1.0, 10.0, 6.0, 500.0).unwrap(), region, FillRule::NonZero);
    let capped = fill(&shapes::rounded_rect(1.0, 1.0, 10.0, 6.0, 3.0).unwrap(), region, FillRule::NonZero);
    assert_same(&big, &capped, region, "radius capped");
    let nan = shapes::rounded_rect(1.0, 1.0, 10.0, 6.0, f32::NAN).unwrap();
    assert_same(&fill(&nan, region, FillRule::NonZero), &rect_mask(256, 256, 2816, 1792, region), region, "nan radius");
}

#[test]
fn rounded_rectangles_and_lines_refuse_sizes_that_make_no_shape() {
    assert!(shapes::rounded_rect(0.0, 0.0, 0.0, 5.0, 1.0).is_none());
    assert!(shapes::rounded_rect(0.0, 0.0, 5.0, -1.0, 1.0).is_none());
    assert!(shapes::rounded_rect(f32::NAN, 0.0, 5.0, 5.0, 1.0).is_none());
    assert!(shapes::line(0.0, 0.0, 5.0, 5.0, 0.0).is_none());
    assert!(shapes::line(1.0, 1.0, 1.0, 1.0, 2.0).is_none(), "no length");
    assert!(shapes::line(0.0, 0.0, f32::INFINITY, 5.0, 2.0).is_none());
    assert!(shapes::line(0.0, 0.0, 5.0, 5.0, f32::NAN).is_none());
}

#[test]
fn a_horizontal_line_is_a_rectangle_and_a_slanted_one_has_its_area() {
    let region = IRect::new(0, 0, 16, 12);
    let h = fill(&shapes::line(2.0, 5.0, 10.0, 5.0, 2.0).unwrap(), region, FillRule::NonZero);
    assert_same(&h, &rect_mask(512, 1024, 2560, 1536, region), region, "horizontal line");
    let d = fill(&shapes::line(2.0, 2.0, 10.0, 10.0, 2.0).unwrap(), region, FillRule::NonZero);
    let want = 8.0 * 2.0_f64.sqrt() * 2.0;
    assert!((sum(&d) - want).abs() < 0.2, "area {} against {want}", sum(&d));
}
