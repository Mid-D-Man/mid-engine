// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "pixel.rs"
// ============================================================================
//! Pixel surfaces and the blend step.
//!
//! A surface is premultiplied `RGBA8`, bytes in the order R, G, B, A, rows
//! `stride` bytes apart (`docs/mid-ui-contracts.md`, section 1.2). The window
//! surface is opaque in practice, where premultiplied and straight alpha agree,
//! so the blend treats a destination as opaque.

use mid_math::Color32;

use crate::linear;

/// The space the over operator mixes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendSpace {
    /// The encoded sRGB numbers are mixed directly. The default.
    #[default]
    Srgb,
    /// Colors are decoded to linear light, mixed and encoded back.
    Linear,
}

/// `src` over `dst`, rounding to the nearest level, in the chosen space.
#[inline]
pub fn over(dst: Color32, src: Color32, space: BlendSpace) -> Color32 {
    match space {
        BlendSpace::Srgb => dst.blend_over_rounded(src),
        BlendSpace::Linear => linear::over(dst, src),
    }
}

/// A borrowed pixel buffer.
#[derive(Debug)]
pub struct SurfaceMut<'a> {
    data: &'a mut [u8],
    width: u32,
    height: u32,
    stride: usize,
}

impl<'a> SurfaceMut<'a> {
    /// Wraps `data`. Returns `None` when `stride` is below `width * 4` or the
    /// slice is too short for `height` rows.
    pub fn new(data: &'a mut [u8], width: u32, height: u32, stride: usize) -> Option<Self> {
        let row = (width as usize).checked_mul(4)?;
        if stride < row {
            return None;
        }
        let need = if height == 0 {
            0
        } else {
            stride.checked_mul(height as usize - 1)?.checked_add(row)?
        };
        if data.len() < need {
            return None;
        }
        Some(Self { data, width, height, stride })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Distance in bytes between the starts of two rows.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The whole buffer.
    pub fn bytes(&self) -> &[u8] {
        self.data
    }

    fn index(&self, x: u32, y: u32) -> usize {
        y as usize * self.stride + x as usize * 4
    }

    /// The pixel at `(x, y)`, which must be inside the surface.
    pub fn pixel(&self, x: u32, y: u32) -> Color32 {
        let i = self.index(x, y);
        Color32 { r: self.data[i], g: self.data[i + 1], b: self.data[i + 2], a: self.data[i + 3] }
    }

    /// Stores a pixel at `(x, y)`, which must be inside the surface.
    pub fn set(&mut self, x: u32, y: u32, c: Color32) {
        let i = self.index(x, y);
        self.data[i] = c.r;
        self.data[i + 1] = c.g;
        self.data[i + 2] = c.b;
        self.data[i + 3] = c.a;
    }

    /// Sets every pixel to `c`.
    pub fn clear(&mut self, c: Color32) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set(x, y, c);
            }
        }
    }
}

/// A pixel buffer that owns its bytes, rows tightly packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Surface {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl Surface {
    /// A transparent black surface.
    pub fn new(width: u32, height: u32) -> Self {
        Self { data: vec![0; width as usize * height as usize * 4], width, height }
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The bytes, rows tightly packed.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// A mutable view for drawing.
    pub fn as_mut(&mut self) -> SurfaceMut<'_> {
        let stride = self.width as usize * 4;
        SurfaceMut { data: &mut self.data, width: self.width, height: self.height, stride }
    }

    /// The pixel at `(x, y)`, which must be inside the surface.
    pub fn pixel(&self, x: u32, y: u32) -> Color32 {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Color32 { r: self.data[i], g: self.data[i + 1], b: self.data[i + 2], a: self.data[i + 3] }
    }
}
