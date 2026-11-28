// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "shapes.rs"
// ============================================================================
//! Shape outlines: the rounded rectangle and the thick line as paths.

use crate::clip::IRect;
use crate::cover::Mask;
use crate::fixed::{floor_div, ONE};
use crate::path::Path;

/// The control point offset of a cubic that approximates a quarter circle,
/// `4 * (sqrt(2) - 1) / 3`.
pub const KAPPA: f32 = 0.552_284_8;

/// A rectangle with rounded corners as a path. The radius is held between zero
/// and half of the shorter side; a radius that is not finite counts as zero.
/// Returns `None` for a size that is not finite or not positive.
pub fn rounded_rect(x: f32, y: f32, w: f32, h: f32, radius: f32) -> Option<Path> {
    if !(x.is_finite() && y.is_finite() && w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
        return None;
    }
    let r = if radius.is_finite() && radius > 0.0 { radius.min(w * 0.5).min(h * 0.5) } else { 0.0 };
    let (right, bottom) = (x + w, y + h);
    let mut p = Path::new();
    if r == 0.0 {
        p.move_to(x, y).line_to(right, y).line_to(right, bottom).line_to(x, bottom).close();
        return Some(p);
    }
    let k = r * KAPPA;
    p.move_to(x + r, y)
        .line_to(right - r, y)
        .cubic_to(right - r + k, y, right, y + r - k, right, y + r)
        .line_to(right, bottom - r)
        .cubic_to(right, bottom - r + k, right - r + k, bottom, right - r, bottom)
        .line_to(x + r, bottom)
        .cubic_to(x + r - k, bottom, x, bottom - r + k, x, bottom - r)
        .line_to(x, y + r)
        .cubic_to(x, y + r - k, x + r - k, y, x + r, y)
        .close();
    Some(p)
}

/// A line of the given width as a rectangle with butt caps, centered on the
/// segment from `(x0, y0)` to `(x1, y1)`. Returns `None` when the line has no
/// length, no width, or a value that is not finite.
pub fn line(x0: f32, y0: f32, x1: f32, y1: f32, width: f32) -> Option<Path> {
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite() && width.is_finite())
        || width <= 0.0
    {
        return None;
    }
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len = (dx * dx + dy * dy).sqrt();
    if !(len.is_finite() && len > 0.0) {
        return None;
    }
    let half = width * 0.5;
    let (nx, ny) = (-dy / len * half, dx / len * half);
    let mut p = Path::new();
    p.move_to(x0 + nx, y0 + ny)
        .line_to(x1 + nx, y1 + ny)
        .line_to(x1 - nx, y1 - ny)
        .line_to(x0 - nx, y0 - ny)
        .close();
    Some(p)
}

/// Exact coverage of pixel `(px, py)` by the fixed-point rectangle with the
/// given edges: the overlapping area in 1/65536 pixel, mapped to 0..=255 with
/// the same rounding as the path rasterizer. An integer-edged rectangle gives
/// 255 on its pixels and 0 outside.
pub fn rect_coverage(x0: i64, y0: i64, x1: i64, y1: i64, px: i32, py: i32) -> u8 {
    let overlap = |lo: i64, hi: i64, p: i32| -> i64 {
        let a = lo.max(p as i64 * ONE);
        let b = hi.min((p as i64 + 1) * ONE);
        (b - a).max(0)
    };
    let area = overlap(x0, x1, px) * overlap(y0, y1, py);
    ((area * 255 + 32768) >> 16) as u8
}

/// The pixels a fixed-point rectangle touches.
pub fn rect_bounds(x0: i64, y0: i64, x1: i64, y1: i64) -> IRect {
    IRect::new(
        floor_div(x0, ONE) as i32,
        floor_div(y0, ONE) as i32,
        floor_div(x1 + ONE - 1, ONE) as i32,
        floor_div(y1 + ONE - 1, ONE) as i32,
    )
}

/// The coverage mask of a fixed-point rectangle inside `clip`, by closed form.
pub fn rect_mask(x0: i64, y0: i64, x1: i64, y1: i64, clip: IRect) -> Mask {
    let region = rect_bounds(x0, y0, x1, y1).intersect(&clip);
    if region.is_empty() || x1 <= x0 || y1 <= y0 {
        return Mask { region: IRect::new(region.x0, region.y0, region.x0, region.y0), data: Vec::new() };
    }
    let mut data = Vec::with_capacity(region.width() as usize * region.height() as usize);
    for y in region.y0..region.y1 {
        for x in region.x0..region.x1 {
            data.push(rect_coverage(x0, y0, x1, y1, x, y));
        }
    }
    Mask { region, data }
}
