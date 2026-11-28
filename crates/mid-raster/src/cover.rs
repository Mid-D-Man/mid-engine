// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "cover.rs"
// ============================================================================
//! The coverage accumulator: line segments in, an 8-bit coverage mask out.
//!
//! Each segment adds signed area to a grid of cells, one cell per pixel. The
//! running sum along a row of cells is the winding area at that pixel, and the
//! fill rule turns it into coverage. All arithmetic is integer, in 1/256 pixel
//! units, so a segment's total contribution is conserved exactly: nothing
//! leaks, whatever the rounding does to where it lands.
//!
//! The accumulator covers one rectangular region. Segments are cut against its
//! top and bottom; the part of a segment left of the region counts as a vertical
//! edge at its left side and the part right of it is dropped, which keeps the
//! winding inside correct without splitting the segment in x first.

use crate::clip::IRect;
use crate::fixed::{div_round, floor_div, ONE};
use crate::path::Segment;

/// Which areas count as inside a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillRule {
    /// Inside where the winding number is not zero.
    #[default]
    NonZero,
    /// Inside where the winding number is odd.
    EvenOdd,
}

/// Coverage of a region: one byte per pixel, 255 fully covered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    /// The pixels the mask describes.
    pub region: IRect,
    /// Row-major coverage, `region.width() * region.height()` bytes.
    pub data: Vec<u8>,
}

impl Mask {
    /// Coverage at an absolute pixel, zero outside the region.
    pub fn at(&self, x: i32, y: i32) -> u8 {
        let r = &self.region;
        if x < r.x0 || x >= r.x1 || y < r.y0 || y >= r.y1 {
            return 0;
        }
        self.data[(y - r.y0) as usize * r.width() as usize + (x - r.x0) as usize]
    }
}

/// The pixel region the segments touch, before any clipping, or `None` when
/// there are no segments.
pub fn bounds(segs: &[Segment]) -> Option<IRect> {
    let first = segs.first()?;
    let (mut x0, mut y0, mut x1, mut y1) = (first[0], first[1], first[0], first[1]);
    for s in segs {
        x0 = x0.min(s[0]).min(s[2]);
        x1 = x1.max(s[0]).max(s[2]);
        y0 = y0.min(s[1]).min(s[3]);
        y1 = y1.max(s[1]).max(s[3]);
    }
    Some(IRect::new(
        floor_div(x0, ONE) as i32,
        floor_div(y0, ONE) as i32,
        floor_div(x1 + ONE - 1, ONE) as i32,
        floor_div(y1 + ONE - 1, ONE) as i32,
    ))
}

/// Rasterizes `segs` into a coverage mask of `region`.
pub fn rasterize(segs: &[Segment], region: IRect, rule: FillRule) -> Mask {
    if region.is_empty() {
        return Mask { region, data: Vec::new() };
    }
    let mut acc = Accum::new(region);
    for s in segs {
        acc.add(s[0], s[1], s[2], s[3]);
    }
    Mask { region, data: acc.resolve(rule) }
}

struct Accum {
    w: usize,
    h: usize,
    ox: i64,
    oy: i64,
    cells: Vec<i32>,
}

impl Accum {
    fn new(region: IRect) -> Self {
        let (w, h) = (region.width() as usize, region.height() as usize);
        Self {
            w,
            h,
            ox: region.x0 as i64 * ONE,
            oy: region.y0 as i64 * ONE,
            // One extra column takes whatever lands right of the region.
            cells: vec![0; (w + 1) * h],
        }
    }

    fn add(&mut self, x0: i64, y0: i64, x1: i64, y1: i64) {
        if y0 == y1 {
            return;
        }
        let (x0, y0, x1, y1) = (x0 - self.ox, y0 - self.oy, x1 - self.ox, y1 - self.oy);
        // Work top to bottom and remember which way the segment really runs.
        let (dir, xa, ya, xb, yb) = if y0 < y1 { (1i64, x0, y0, x1, y1) } else { (-1, x1, y1, x0, y0) };
        let bottom = (self.h as i64) << 8;
        if yb <= 0 || ya >= bottom {
            return;
        }
        let (dx, dy) = (xb - xa, yb - ya);
        let x_at = |y: i64| -> i64 {
            if y == ya {
                xa
            } else if y == yb {
                xb
            } else {
                xa + div_round((y - ya) * dx, dy)
            }
        };
        let (ys, ye) = (ya.max(0), yb.min(bottom));
        let (first, last) = (ys >> 8, (ye - 1) >> 8);
        let (mut py, mut px) = (ys, x_at(ys));
        for row in first..=last {
            let ny = ye.min((row + 1) << 8);
            let nx = x_at(ny);
            self.row(row as usize, dir, px, py, nx, ny);
            px = nx;
            py = ny;
        }
    }

    // A piece of a segment inside one pixel row, from (xp, yp) to (xn, yn)
    // with yn > yp.
    fn row(&mut self, row: usize, dir: i64, xp: i64, yp: i64, xn: i64, yn: i64) {
        let base = row * (self.w + 1);
        let wide = (self.w as i64) << 8;
        let (mut xl, mut yl, mut xr, mut yr) = if xp <= xn { (xp, yp, xn, yn) } else { (xn, yn, xp, yp) };

        if xl == xr {
            // Vertical piece.
            if xl >= wide {
                return;
            }
            let x = xl.max(0);
            self.span(base, dir, floor_div(x, ONE), x, x, yn - yp);
            return;
        }
        if xr <= 0 {
            // Wholly left of the region: it covers every pixel from column 0.
            self.span(base, dir, 0, 0, 0, yn - yp);
            return;
        }
        if xl >= wide {
            return;
        }
        let width = xr - xl;
        let ydiff = yr - yl;
        let (xl0, yl0) = (xl, yl);
        let y_at = move |x: i64| -> i64 { yl0 + div_round((x - xl0) * ydiff, width) };
        if xl < 0 {
            let y0 = y_at(0);
            self.span(base, dir, 0, 0, 0, (y0 - yl).abs());
            xl = 0;
            yl = y0;
        }
        if xr > wide {
            yr = y_at(wide);
            xr = wide;
        }
        let (mut px, mut py) = (xl, yl);
        let mut cell = floor_div(xl, ONE);
        let last = floor_div(xr - 1, ONE);
        while cell < last {
            let bx = (cell + 1) * ONE;
            let by = y_at(bx);
            self.span(base, dir, cell, px, bx, (by - py).abs());
            px = bx;
            py = by;
            cell += 1;
        }
        self.span(base, dir, cell, px, xr, (yr - py).abs());
    }

    // Adds a piece lying inside one cell: `d` rows of height at mean x between
    // `px` and `qx`. The left part of the cell gets `256 - f` and the cell to
    // its right gets `f`, where `f` is the mean x within the cell, so the pair
    // always adds up to `d * 256`.
    fn span(&mut self, base: usize, dir: i64, cell: i64, px: i64, qx: i64, d: i64) {
        if d == 0 {
            return;
        }
        let f = ((px + qx) >> 1) - cell * ONE;
        let s = dir * d;
        let i = base + cell as usize;
        self.cells[i] = self.cells[i].wrapping_add((s * (ONE - f)) as i32);
        self.cells[i + 1] = self.cells[i + 1].wrapping_add((s * f) as i32);
    }

    fn resolve(&self, rule: FillRule) -> Vec<u8> {
        let mut out = vec![0u8; self.w * self.h];
        for row in 0..self.h {
            let base = row * (self.w + 1);
            let mut sum = 0i32;
            for c in 0..self.w {
                sum = sum.wrapping_add(self.cells[base + c]);
                out[row * self.w + c] = coverage(sum, rule);
            }
        }
        out
    }
}

/// Winding area in 1/65536 pixel to an 8-bit coverage, rounding half up.
fn coverage(sum: i32, rule: FillRule) -> u8 {
    let m = sum.unsigned_abs();
    let m = match rule {
        FillRule::NonZero => m.min(65536),
        FillRule::EvenOdd => {
            let m = m % 131072;
            if m > 65536 { 131072 - m } else { m }
        }
    };
    ((m * 255 + 32768) >> 16) as u8
}
