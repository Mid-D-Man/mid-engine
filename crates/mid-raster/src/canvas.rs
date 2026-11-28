// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "canvas.rs"
// ============================================================================
//! The drawing surface: shapes, a clip stack and a blend space over a pixel
//! buffer.

use mid_math::Color32;

use crate::clip::{ClipStack, IRect};
use crate::cover::{self, FillRule, Mask};
use crate::fixed::from_f32;
use crate::path::Path;
use crate::pixel::{over, BlendSpace, SurfaceMut};
use crate::shapes::{self, rect_bounds, rect_coverage};

/// What a canvas had to ignore or repair, as `docs/mid-ui-contracts.md`
/// section 2.5 asks for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Pops with no clip open.
    pub ignored_pops: u32,
    /// Clips still open when [`Canvas::finish`] closed them.
    pub closed_clips: u32,
    /// Shapes that drew nothing because a value was not finite or a size was
    /// not positive.
    pub skipped_shapes: u32,
}

/// Draws into a surface through a clip stack, in one blend space.
#[derive(Debug)]
pub struct Canvas<'a> {
    surface: SurfaceMut<'a>,
    clip: ClipStack,
    space: BlendSpace,
    stats: Stats,
}

impl<'a> Canvas<'a> {
    /// A canvas over `surface` that blends in sRGB.
    pub fn new(surface: SurfaceMut<'a>) -> Self {
        let bounds = IRect::new(0, 0, surface.width() as i32, surface.height() as i32);
        Self { surface, clip: ClipStack::new(bounds), space: BlendSpace::Srgb, stats: Stats::default() }
    }

    /// Chooses the blend space for everything drawn afterwards.
    pub fn with_blend_space(mut self, space: BlendSpace) -> Self {
        self.space = space;
        self
    }

    /// The pixels drawn so far.
    pub fn surface(&self) -> &SurfaceMut<'a> {
        &self.surface
    }

    /// What was ignored or repaired so far.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The clip in force.
    pub fn clip(&self) -> IRect {
        self.clip.current()
    }

    /// How many clips are open.
    pub fn clip_depth(&self) -> usize {
        self.clip.depth()
    }

    /// Opens a clip rectangle. Its edges snap to the nearest pixel edges and it
    /// is intersected with the clip below.
    pub fn push_clip(&mut self, x: f32, y: f32, w: f32, h: f32) {
        let edges = (|| Some((from_f32(x)?, from_f32(y)?, from_f32(x + w)?, from_f32(y + h)?)))();
        self.clip.push(edges);
    }

    /// Closes the newest clip; counted and ignored when none is open.
    pub fn pop_clip(&mut self) {
        if !self.clip.pop() {
            self.stats.ignored_pops += 1;
        }
    }

    /// Closes any clips still open, counts them and returns the totals.
    pub fn finish(&mut self) -> Stats {
        self.stats.closed_clips += self.clip.close_all() as u32;
        self.stats
    }

    fn blend_at(&mut self, x: i32, y: i32, color: Color32, coverage: u8) {
        if coverage == 0 {
            return;
        }
        let src = color.scale_alpha_rounded(coverage);
        if src.a == 0 {
            return;
        }
        let (ux, uy) = (x as u32, y as u32);
        let dst = self.surface.pixel(ux, uy);
        self.surface.set(ux, uy, over(dst, src, self.space));
    }

    fn blend_mask(&mut self, mask: &Mask, color: Color32) {
        let r = mask.region;
        let w = r.width() as usize;
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let c = mask.data[(y - r.y0) as usize * w + (x - r.x0) as usize];
                self.blend_at(x, y, color, c);
            }
        }
    }

    /// Fills an axis-aligned rectangle with exact coverage on its edges.
    pub fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color32) {
        let edges = (|| {
            if !(w > 0.0 && h > 0.0) {
                return None;
            }
            Some((from_f32(x)?, from_f32(y)?, from_f32(x + w)?, from_f32(y + h)?))
        })();
        let Some((x0, y0, x1, y1)) = edges else {
            self.stats.skipped_shapes += 1;
            return;
        };
        let region = rect_bounds(x0, y0, x1, y1).intersect(&self.clip.current());
        if region.is_empty() {
            return;
        }
        for py in region.y0..region.y1 {
            for px in region.x0..region.x1 {
                let c = rect_coverage(x0, y0, x1, y1, px, py);
                self.blend_at(px, py, color, c);
            }
        }
    }

    /// Fills a rounded rectangle. A radius of zero is the plain rectangle; a
    /// larger one is held to half of the shorter side.
    pub fn fill_rounded_rect(&mut self, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Color32) {
        if radius.is_finite() && radius > 0.0 {
            match shapes::rounded_rect(x, y, w, h, radius) {
                Some(p) => self.fill_path(&p, FillRule::NonZero, color),
                None => self.stats.skipped_shapes += 1,
            }
        } else {
            self.fill_rect(x, y, w, h, color);
        }
    }

    /// Draws a line of the given width with butt caps.
    pub fn stroke_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, width: f32, color: Color32) {
        match shapes::line(x0, y0, x1, y1, width) {
            Some(p) => self.fill_path(&p, FillRule::NonZero, color),
            None => self.stats.skipped_shapes += 1,
        }
    }

    /// Fills a path under a fill rule.
    pub fn fill_path(&mut self, path: &Path, rule: FillRule, color: Color32) {
        let Some(segs) = path.flatten() else {
            self.stats.skipped_shapes += 1;
            return;
        };
        let Some(bounds) = cover::bounds(&segs) else {
            return;
        };
        let region = bounds.intersect(&self.clip.current());
        if region.is_empty() {
            return;
        }
        let mask = cover::rasterize(&segs, region, rule);
        self.blend_mask(&mask, color);
    }
}
