// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "cover.rs"
// ============================================================================
//! The oracle: `tiny-skia` rasterizes the same shapes and the coverage is
//! compared within a stated tolerance. `tiny-skia` is a dev-dependency only.

use mid_raster::cover::rasterize;
use mid_raster::shapes::{self, KAPPA};
use mid_raster::{FillRule, IRect, Path};
use tiny_skia as ts;

#[derive(Clone, Copy)]
enum Cmd {
    Move(f32, f32),
    Line(f32, f32),
    Quad(f32, f32, f32, f32),
    Cubic(f32, f32, f32, f32, f32, f32),
    Close,
}

fn mine(cmds: &[Cmd]) -> Path {
    let mut p = Path::new();
    for &c in cmds {
        match c {
            Cmd::Move(x, y) => p.move_to(x, y),
            Cmd::Line(x, y) => p.line_to(x, y),
            Cmd::Quad(a, b, x, y) => p.quad_to(a, b, x, y),
            Cmd::Cubic(a, b, c, d, x, y) => p.cubic_to(a, b, c, d, x, y),
            Cmd::Close => p.close(),
        };
    }
    p
}

fn theirs(cmds: &[Cmd]) -> ts::Path {
    let mut b = ts::PathBuilder::new();
    for &c in cmds {
        match c {
            Cmd::Move(x, y) => b.move_to(x, y),
            Cmd::Line(x, y) => b.line_to(x, y),
            Cmd::Quad(a, bb, x, y) => b.quad_to(a, bb, x, y),
            Cmd::Cubic(a, bb, c, d, x, y) => b.cubic_to(a, bb, c, d, x, y),
            Cmd::Close => b.close(),
        }
    }
    b.finish().expect("a path with points")
}

const SIZE: u32 = 32;

/// Coverage of `cmds` from `tiny-skia`, as the alpha of white on transparent.
fn oracle(cmds: &[Cmd], rule: FillRule) -> Vec<u8> {
    let mut pix = ts::Pixmap::new(SIZE, SIZE).unwrap();
    let mut paint = ts::Paint::default();
    paint.set_color_rgba8(255, 255, 255, 255);
    paint.anti_alias = true;
    let fr = match rule {
        FillRule::NonZero => ts::FillRule::Winding,
        FillRule::EvenOdd => ts::FillRule::EvenOdd,
    };
    pix.fill_path(&theirs(cmds), &paint, fr, ts::Transform::identity(), None);
    pix.data().chunks_exact(4).map(|p| p[3]).collect()
}

fn ours(cmds: &[Cmd], rule: FillRule) -> Vec<u8> {
    let region = IRect::new(0, 0, SIZE as i32, SIZE as i32);
    let segs = mine(cmds).flatten().unwrap();
    rasterize(&segs, region, rule).data
}

struct Diff {
    max: i32,
    mean: f64,
    over8: usize,
    pixels: usize,
}

fn compare(name: &str, cmds: &[Cmd], rule: FillRule) -> Diff {
    let (a, b) = (ours(cmds, rule), oracle(cmds, rule));
    let (mut max, mut total, mut over8, mut pixels) = (0i32, 0i64, 0usize, 0usize);
    for (x, y) in a.iter().zip(&b) {
        if *x == 0 && *y == 0 {
            continue;
        }
        let d = (*x as i32 - *y as i32).abs();
        max = max.max(d);
        total += d as i64;
        over8 += (d > 8) as usize;
        pixels += 1;
    }
    let mean = if pixels == 0 { 0.0 } else { total as f64 / pixels as f64 };
    eprintln!("oracle {name:<22} max {max:>3}  mean {mean:>5.2}  over8 {over8:>3} of {pixels:>4} touched pixels");
    Diff { max, mean, over8, pixels }
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Cmd> {
    vec![Cmd::Move(x0, y0), Cmd::Line(x1, y0), Cmd::Line(x1, y1), Cmd::Line(x0, y1), Cmd::Close]
}

fn circle(cx: f32, cy: f32, r: f32, clockwise: bool) -> Vec<Cmd> {
    let k = r * KAPPA;
    if clockwise {
        vec![
            Cmd::Move(cx + r, cy),
            Cmd::Cubic(cx + r, cy + k, cx + k, cy + r, cx, cy + r),
            Cmd::Cubic(cx - k, cy + r, cx - r, cy + k, cx - r, cy),
            Cmd::Cubic(cx - r, cy - k, cx - k, cy - r, cx, cy - r),
            Cmd::Cubic(cx + k, cy - r, cx + r, cy - k, cx + r, cy),
            Cmd::Close,
        ]
    } else {
        vec![
            Cmd::Move(cx + r, cy),
            Cmd::Cubic(cx + r, cy - k, cx + k, cy - r, cx, cy - r),
            Cmd::Cubic(cx - k, cy - r, cx - r, cy - k, cx - r, cy),
            Cmd::Cubic(cx - r, cy + k, cx - k, cy + r, cx, cy + r),
            Cmd::Cubic(cx + k, cy + r, cx + r, cy + k, cx + r, cy),
            Cmd::Close,
        ]
    }
}

fn rounded(x: f32, y: f32, w: f32, h: f32, r: f32) -> Vec<Cmd> {
    let (right, bottom, k) = (x + w, y + h, r * KAPPA);
    vec![
        Cmd::Move(x + r, y),
        Cmd::Line(right - r, y),
        Cmd::Cubic(right - r + k, y, right, y + r - k, right, y + r),
        Cmd::Line(right, bottom - r),
        Cmd::Cubic(right, bottom - r + k, right - r + k, bottom, right - r, bottom),
        Cmd::Line(x + r, bottom),
        Cmd::Cubic(x + r - k, bottom, x, bottom - r + k, x, bottom - r),
        Cmd::Line(x, y + r),
        Cmd::Cubic(x, y + r - k, x + r - k, y, x + r, y),
        Cmd::Close,
    ]
}

fn polygon(pts: &[(f32, f32)]) -> Vec<Cmd> {
    let mut v = vec![Cmd::Move(pts[0].0, pts[0].1)];
    v.extend(pts[1..].iter().map(|&(x, y)| Cmd::Line(x, y)));
    v.push(Cmd::Close);
    v
}

fn star() -> Vec<Cmd> {
    let pts: Vec<(f32, f32)> = (0..5)
        .map(|i| {
            let a = i as f32 * 4.0 * std::f32::consts::PI / 5.0 - std::f32::consts::FRAC_PI_2;
            (16.0 + 13.0 * a.cos(), 16.5 + 13.0 * a.sin())
        })
        .collect();
    polygon(&pts)
}

/// The polygon `shapes::line` builds, written out for the oracle.
fn line_polygon(x0: f32, y0: f32, x1: f32, y1: f32, w: f32) -> Vec<Cmd> {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len = (dx * dx + dy * dy).sqrt();
    let (nx, ny) = (-dy / len * w * 0.5, dx / len * w * 0.5);
    polygon(&[(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)])
}

/// Compares one shape with the oracle and holds the difference under a ceiling
/// for its largest and its mean per-pixel difference (over the pixels either
/// side touches, on the 0..=255 coverage scale).
///
/// The ceilings sit a little above what was measured and say nothing about
/// shapes outside this set. Two causes explain the size of the differences:
/// `tiny-skia` is a port of Skia and quantizes coverage by supersampling,
/// four rows to a pixel, so a pixel that is 0.9 covered can come back as 0.8
/// (a difference of 26), while this rasterizer computes the exact area; and
/// where contours cross inside one pixel, a signed-area sum cannot tell which
/// part of that pixel is inside, while supersampling can.
fn check(name: &str, cmds: &[Cmd], rule: FillRule, max_ceiling: i32, mean_ceiling: f64) {
    let d = compare(name, cmds, rule);
    assert!(d.max <= max_ceiling, "{name}: max difference {} above {max_ceiling}", d.max);
    assert!(d.mean <= mean_ceiling, "{name}: mean difference {:.2} above {mean_ceiling}", d.mean);
}

#[test]
fn rectangles() {
    check("rect integer", &rect(2.0, 3.0, 20.0, 11.0), FillRule::NonZero, 0, 0.0);
    check("rect fractional", &rect(1.3, 1.7, 9.6, 7.4), FillRule::NonZero, 32, 10.0);
    check("rect thin", &rect(4.25, 4.1, 28.75, 5.0), FillRule::NonZero, 32, 28.0);
}

#[test]
fn curves_and_rounded_shapes() {
    check("rounded rect", &rounded(2.4, 3.6, 24.0, 18.0, 6.0), FillRule::NonZero, 32, 3.5);
    check("circle", &circle(16.2, 15.7, 11.5, true), FillRule::NonZero, 28, 2.0);
    check("small circle", &circle(8.5, 8.5, 3.0, true), FillRule::NonZero, 30, 5.5);
    let petal = [
        Cmd::Move(4.0, 16.3),
        Cmd::Quad(16.0, 1.5, 28.0, 16.3),
        Cmd::Quad(16.0, 31.0, 4.0, 16.3),
        Cmd::Close,
    ];
    check("quad petal", &petal, FillRule::NonZero, 36, 3.0);
    let mut ring = circle(16.0, 16.0, 13.0, true);
    ring.extend(circle(16.0, 16.0, 7.0, false));
    check("ring (non-zero)", &ring, FillRule::NonZero, 24, 3.0);
    check("ring (even-odd)", &ring, FillRule::EvenOdd, 24, 3.0);
}

#[test]
fn polygons_and_lines() {
    check("triangle", &polygon(&[(3.0, 3.0), (27.0, 8.0), (9.0, 28.0)]), FillRule::NonZero, 26, 2.0);
    // The star's edges cross each other, the case a signed-area sum is weakest at.
    check("star (non-zero)", &star(), FillRule::NonZero, 44, 3.5);
    check("star (even-odd)", &star(), FillRule::EvenOdd, 96, 6.0);
    check("line 1px diagonal", &line_polygon(3.0, 4.0, 27.0, 22.0, 1.0), FillRule::NonZero, 16, 6.0);
    check("line 3px shallow", &line_polygon(2.0, 10.0, 29.0, 14.0, 3.0), FillRule::NonZero, 24, 5.0);
    check("line 2px steep", &line_polygon(10.0, 2.0, 14.0, 29.0, 2.0), FillRule::NonZero, 26, 6.0);
}

#[test]
fn the_shape_helpers_build_what_the_oracle_is_given() {
    // shapes::line and shapes::rounded_rect are the paths the oracle shapes
    // above spell out; if they ever drift this fails.
    let region = IRect::new(0, 0, SIZE as i32, SIZE as i32);
    let a = rasterize(&shapes::line(3.0, 4.0, 27.0, 22.0, 1.0).unwrap().flatten().unwrap(), region, FillRule::NonZero);
    assert_eq!(a.data, ours(&line_polygon(3.0, 4.0, 27.0, 22.0, 1.0), FillRule::NonZero));
    let b = rasterize(&shapes::rounded_rect(2.4, 3.6, 24.0, 18.0, 6.0).unwrap().flatten().unwrap(), region, FillRule::NonZero);
    assert_eq!(b.data, ours(&rounded(2.4, 3.6, 24.0, 18.0, 6.0), FillRule::NonZero));
}

#[test]
fn random_polygons_in_bulk() {
    let mut state = 0xC0FF_EE12_3456_789Au64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
    };
    let (mut worst, mut mean_sum, mut n) = (0i32, 0.0f64, 0usize);
    for round in 0..40 {
        let pts: Vec<(f32, f32)> = (0..(3 + round % 4)).map(|_| (2.0 + next() * 28.0, 2.0 + next() * 28.0)).collect();
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let d = compare(&format!("random polygon {round}"), &polygon(&pts), rule);
            worst = worst.max(d.max);
            mean_sum += d.mean;
            n += 1;
            let _ = d.over8 + d.pixels;
        }
    }
    eprintln!("oracle random polygons: worst max {worst}, mean of means {:.2}", mean_sum / n as f64);
    // Random polygons cross themselves, so the worst pixel is a crossing one.
    assert!(worst <= 160, "worst difference {worst}");
    assert!(mean_sum / n as f64 <= 5.5, "mean of means {}", mean_sum / n as f64);
}
