// crates/mid-math/src/ffi/color.rs
//! C-ABI types and #[no_mangle] exports for color types.
//!
//! All color types that are already #[repr(C)] map directly to their
//! C counterparts. Decoding functions use output pointer parameters
//! to avoid returning tuples across the C boundary.

use crate::color::{Color32, Hsl, Hsv, LogLuv32, Rgb, Rgba, Rgbe, YCbCr, YCbCrStandard};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

/// Packed sRGB u8 RGBA color. 4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CColor32 { pub r: u8, pub g: u8, pub b: u8, pub a: u8 }

/// Linear-light RGB. 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CRgb { pub r: f32, pub g: f32, pub b: f32 }

/// Linear-light RGBA. 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CRgba { pub r: f32, pub g: f32, pub b: f32, pub a: f32 }

/// HSV color in sRGB space. 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CHsv { pub h: f32, pub s: f32, pub v: f32 }

/// HSL color in sRGB space. 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CHsl { pub h: f32, pub s: f32, pub l: f32 }

/// Radiance HDR RGBE. 4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CRgbe { pub r: u8, pub g: u8, pub b: u8, pub e: u8 }

/// YCbCr color (full range). 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CYCbCr { pub y: f32, pub cb: f32, pub cr: f32 }

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<Color32> for CColor32 {
    #[inline] fn from(c: Color32) -> Self { Self { r: c.r, g: c.g, b: c.b, a: c.a } }
}
impl From<CColor32> for Color32 {
    #[inline] fn from(c: CColor32) -> Self { Color32::new(c.r, c.g, c.b, c.a) }
}

impl From<Rgb> for CRgb {
    #[inline] fn from(c: Rgb) -> Self { Self { r: c.r, g: c.g, b: c.b } }
}
impl From<CRgb> for Rgb {
    #[inline] fn from(c: CRgb) -> Self { Rgb::new(c.r, c.g, c.b) }
}

impl From<Rgba> for CRgba {
    #[inline] fn from(c: Rgba) -> Self { Self { r: c.r, g: c.g, b: c.b, a: c.a } }
}
impl From<CRgba> for Rgba {
    #[inline] fn from(c: CRgba) -> Self { Rgba::new(c.r, c.g, c.b, c.a) }
}

impl From<Hsv> for CHsv {
    #[inline] fn from(c: Hsv) -> Self { Self { h: c.h, s: c.s, v: c.v } }
}
impl From<CHsv> for Hsv {
    #[inline] fn from(c: CHsv) -> Self { Hsv::new(c.h, c.s, c.v) }
}

impl From<Hsl> for CHsl {
    #[inline] fn from(c: Hsl) -> Self { Self { h: c.h, s: c.s, l: c.l } }
}
impl From<CHsl> for Hsl {
    #[inline] fn from(c: CHsl) -> Self { Hsl::new(c.h, c.s, c.l) }
}

impl From<Rgbe> for CRgbe {
    #[inline] fn from(c: Rgbe) -> Self { Self { r: c.r, g: c.g, b: c.b, e: c.e } }
}
impl From<CRgbe> for Rgbe {
    #[inline] fn from(c: CRgbe) -> Self { Rgbe { r: c.r, g: c.g, b: c.b, e: c.e } }
}

impl From<YCbCr> for CYCbCr {
    #[inline] fn from(c: YCbCr) -> Self { Self { y: c.y, cb: c.cb, cr: c.cr } }
}
impl From<CYCbCr> for YCbCr {
    #[inline] fn from(c: CYCbCr) -> Self { YCbCr::new(c.y, c.cb, c.cr) }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Color32 exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_color32_new(r:u8,g:u8,b:u8,a:u8) -> CColor32 {
    Color32::new(r,g,b,a).into()
}
#[no_mangle] pub extern "C" fn mid_color32_new_opaque(r:u8,g:u8,b:u8) -> CColor32 {
    Color32::new_opaque(r,g,b).into()
}
#[no_mangle] pub extern "C" fn mid_color32_transparent() -> CColor32 { Color32::TRANSPARENT.into() }
#[no_mangle] pub extern "C" fn mid_color32_black()       -> CColor32 { Color32::BLACK.into() }
#[no_mangle] pub extern "C" fn mid_color32_white()       -> CColor32 { Color32::WHITE.into() }
#[no_mangle] pub extern "C" fn mid_color32_to_u32_rgba(c: CColor32) -> u32 {
    Color32::from(c).to_u32_rgba()
}
#[no_mangle] pub extern "C" fn mid_color32_to_u32_argb(c: CColor32) -> u32 {
    Color32::from(c).to_u32_argb()
}
#[no_mangle] pub extern "C" fn mid_color32_from_u32_rgba(v: u32) -> CColor32 {
    Color32::from_u32_rgba(v).into()
}
#[no_mangle] pub extern "C" fn mid_color32_blend_over(dst: CColor32, src: CColor32) -> CColor32 {
    Color32::from(dst).blend_over(Color32::from(src)).into()
}
#[no_mangle] pub extern "C" fn mid_color32_scale_alpha(c: CColor32, factor: u8) -> CColor32 {
    Color32::from(c).scale_alpha(factor).into()
}
#[no_mangle] pub extern "C" fn mid_color32_premultiply(c: CColor32) -> CColor32 {
    Color32::from(c).premultiply().into()
}
#[no_mangle] pub extern "C" fn mid_color32_with_alpha(c: CColor32, a: u8) -> CColor32 {
    Color32::from(c).with_alpha(a).into()
}
#[no_mangle] pub extern "C" fn mid_color32_is_opaque(c: CColor32) -> bool {
    Color32::from(c).is_opaque()
}
#[no_mangle] pub extern "C" fn mid_color32_to_rgba(c: CColor32) -> CRgba {
    Rgba::from_color32(Color32::from(c)).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Rgb exports (linear)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_rgb_new(r:f32,g:f32,b:f32) -> CRgb { Rgb::new(r,g,b).into() }
#[no_mangle] pub extern "C" fn mid_rgb_black() -> CRgb { Rgb::BLACK.into() }
#[no_mangle] pub extern "C" fn mid_rgb_white() -> CRgb { Rgb::WHITE.into() }
#[no_mangle] pub extern "C" fn mid_rgb_splat(v: f32) -> CRgb { Rgb::splat(v).into() }
#[no_mangle] pub extern "C" fn mid_rgb_add(a: CRgb, b: CRgb) -> CRgb {
    (Rgb::from(a) + Rgb::from(b)).into()
}
#[no_mangle] pub extern "C" fn mid_rgb_scale(c: CRgb, s: f32) -> CRgb {
    Rgb::from(c).scale(s).into()
}
#[no_mangle] pub extern "C" fn mid_rgb_lerp(a: CRgb, b: CRgb, t: f32) -> CRgb {
    Rgb::from(a).lerp(Rgb::from(b), t).into()
}
#[no_mangle] pub extern "C" fn mid_rgb_modulate(a: CRgb, b: CRgb) -> CRgb {
    Rgb::from(a).modulate(Rgb::from(b)).into()
}
#[no_mangle] pub extern "C" fn mid_rgb_luminance(c: CRgb) -> f32 {
    Rgb::from(c).luminance()
}
#[no_mangle] pub extern "C" fn mid_rgb_from_srgb(c: CRgb) -> CRgb {
    Rgb::from(c).from_srgb().into()
}
#[no_mangle] pub extern "C" fn mid_rgb_to_srgb(c: CRgb) -> CRgb {
    Rgb::from(c).to_srgb().into()
}
#[no_mangle] pub extern "C" fn mid_rgb_tone_map_reinhard(c: CRgb) -> CRgb {
    Rgb::from(c).tone_map_reinhard().into()
}
#[no_mangle] pub extern "C" fn mid_rgb_tone_map_reinhard_luminance(c: CRgb) -> CRgb {
    Rgb::from(c).tone_map_reinhard_luminance().into()
}
#[no_mangle] pub extern "C" fn mid_rgb_tone_map_aces(c: CRgb) -> CRgb {
    Rgb::from(c).tone_map_aces().into()
}
#[no_mangle] pub extern "C" fn mid_rgb_clamp(c: CRgb, lo: f32, hi: f32) -> CRgb {
    Rgb::from(c).clamp(lo, hi).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Rgba exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_rgba_new(r:f32,g:f32,b:f32,a:f32) -> CRgba {
    Rgba::new(r,g,b,a).into()
}
#[no_mangle] pub extern "C" fn mid_rgba_from_rgb(rgb: CRgb, a: f32) -> CRgba {
    Rgba::from_rgb(Rgb::from(rgb), a).into()
}
#[no_mangle] pub extern "C" fn mid_rgba_from_color32(c: CColor32) -> CRgba {
    Rgba::from_color32(Color32::from(c)).into()
}
#[no_mangle] pub extern "C" fn mid_rgba_to_color32(c: CRgba) -> CColor32 {
    Rgba::from(c).to_color32().into()
}
#[no_mangle] pub extern "C" fn mid_rgba_lerp(a: CRgba, b: CRgba, t: f32) -> CRgba {
    Rgba::from(a).lerp(Rgba::from(b), t).into()
}
#[no_mangle] pub extern "C" fn mid_rgba_premultiply_alpha(c: CRgba) -> CRgba {
    Rgba::from(c).premultiply_alpha().into()
}
#[no_mangle] pub extern "C" fn mid_rgba_unpremultiply_alpha(c: CRgba) -> CRgba {
    Rgba::from(c).unpremultiply_alpha().into()
}
#[no_mangle] pub extern "C" fn mid_rgba_with_alpha(c: CRgba, a: f32) -> CRgba {
    Rgba::from(c).with_alpha(a).into()
}
#[no_mangle] pub extern "C" fn mid_rgba_luminance(c: CRgba) -> f32 {
    Rgba::from(c).luminance()
}
#[no_mangle] pub extern "C" fn mid_rgba_tone_map_aces(c: CRgba) -> CRgba {
    Rgba::from(c).tone_map_aces().into()
}
#[no_mangle] pub extern "C" fn mid_rgba_from_srgb(c: CRgba) -> CRgba {
    Rgba::from(c).from_srgb().into()
}
#[no_mangle] pub extern "C" fn mid_rgba_to_srgb(c: CRgba) -> CRgba {
    Rgba::from(c).to_srgb().into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  HSV exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_hsv_new(h:f32,s:f32,v:f32) -> CHsv { Hsv::new(h,s,v).into() }
#[no_mangle] pub extern "C" fn mid_hsv_from_srgb(r:f32,g:f32,b:f32) -> CHsv {
    Hsv::from_srgb(r,g,b).into()
}
#[no_mangle] pub extern "C" fn mid_hsv_to_srgb(c: CHsv, r: *mut f32, g: *mut f32, b: *mut f32) {
    let (rv, gv, bv) = Hsv::from(c).to_srgb();
    unsafe { *r = rv; *g = gv; *b = bv; }
}
#[no_mangle] pub extern "C" fn mid_hsv_from_linear(c: CRgb) -> CHsv {
    Hsv::from_linear(Rgb::from(c)).into()
}
#[no_mangle] pub extern "C" fn mid_hsv_to_linear(c: CHsv) -> CRgb {
    Hsv::from(c).to_linear().into()
}
#[no_mangle] pub extern "C" fn mid_hsv_shift_hue(c: CHsv, delta: f32) -> CHsv {
    Hsv::from(c).shift_hue(delta).into()
}
#[no_mangle] pub extern "C" fn mid_hsv_scale_saturation(c: CHsv, f: f32) -> CHsv {
    Hsv::from(c).scale_saturation(f).into()
}
#[no_mangle] pub extern "C" fn mid_hsv_desaturate(c: CHsv, amount: f32) -> CHsv {
    Hsv::from(c).desaturate(amount).into()
}
#[no_mangle] pub extern "C" fn mid_hsv_scale_value(c: CHsv, f: f32) -> CHsv {
    Hsv::from(c).scale_value(f).into()
}
#[no_mangle] pub extern "C" fn mid_hsv_complement(c: CHsv) -> CHsv {
    Hsv::from(c).complement().into()
}
#[no_mangle] pub extern "C" fn mid_hsv_lerp(a: CHsv, b: CHsv, t: f32) -> CHsv {
    Hsv::from(a).lerp(Hsv::from(b), t).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  HSL exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_hsl_new(h:f32,s:f32,l:f32) -> CHsl { Hsl::new(h,s,l).into() }
#[no_mangle] pub extern "C" fn mid_hsl_from_srgb(r:f32,g:f32,b:f32) -> CHsl {
    Hsl::from_srgb(r,g,b).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_to_srgb(c: CHsl, r: *mut f32, g: *mut f32, b: *mut f32) {
    let (rv, gv, bv) = Hsl::from(c).to_srgb();
    unsafe { *r = rv; *g = gv; *b = bv; }
}
#[no_mangle] pub extern "C" fn mid_hsl_from_linear(c: CRgb) -> CHsl {
    Hsl::from_linear(Rgb::from(c)).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_to_linear(c: CHsl) -> CRgb {
    Hsl::from(c).to_linear().into()
}
#[no_mangle] pub extern "C" fn mid_hsl_shift_hue(c: CHsl, delta: f32) -> CHsl {
    Hsl::from(c).shift_hue(delta).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_lighten(c: CHsl, amount: f32) -> CHsl {
    Hsl::from(c).lighten(amount).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_darken(c: CHsl, amount: f32) -> CHsl {
    Hsl::from(c).darken(amount).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_desaturate(c: CHsl, amount: f32) -> CHsl {
    Hsl::from(c).desaturate(amount).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_complement(c: CHsl) -> CHsl {
    Hsl::from(c).complement().into()
}
#[no_mangle] pub extern "C" fn mid_hsl_lerp(a: CHsl, b: CHsl, t: f32) -> CHsl {
    Hsl::from(a).lerp(Hsl::from(b), t).into()
}
#[no_mangle] pub extern "C" fn mid_hsl_to_hsv(c: CHsl) -> CHsv {
    Hsl::from(c).to_hsv().into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  RGBE exports (HDR)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_rgbe_encode(r:f32,g:f32,b:f32) -> CRgbe {
    Rgbe::encode(r,g,b).into()
}
#[no_mangle] pub extern "C" fn mid_rgbe_encode_rgb(c: CRgb) -> CRgbe {
    Rgbe::encode_rgb(Rgb::from(c)).into()
}
#[no_mangle] pub unsafe extern "C" fn mid_rgbe_decode(
    rgbe: CRgbe,
    r: *mut f32, g: *mut f32, b: *mut f32,
) {
    let (rv, gv, bv) = Rgbe::from(rgbe).decode();
    *r = rv; *g = gv; *b = bv;
}
#[no_mangle] pub extern "C" fn mid_rgbe_decode_rgb(rgbe: CRgbe) -> CRgb {
    Rgbe::from(rgbe).decode_rgb().into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  LogLuv32 exports (perceptual HDR)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_logluv32_encode(r:f32,g:f32,b:f32) -> u32 {
    LogLuv32::encode(r,g,b).raw()
}
#[no_mangle] pub extern "C" fn mid_logluv32_encode_rgb(c: CRgb) -> u32 {
    LogLuv32::encode_rgb(Rgb::from(c)).raw()
}
#[no_mangle] pub unsafe extern "C" fn mid_logluv32_decode(
    v: u32,
    r: *mut f32, g: *mut f32, b: *mut f32,
) {
    let (rv, gv, bv) = LogLuv32(v).decode();
    *r = rv; *g = gv; *b = bv;
}
#[no_mangle] pub extern "C" fn mid_logluv32_decode_rgb(v: u32) -> CRgb {
    LogLuv32(v).decode_rgb().into()
}
#[no_mangle] pub extern "C" fn mid_logluv32_luminance_log2(v: u32) -> f32 {
    LogLuv32(v).luminance_log2()
}

// ═══════════════════════════════════════════════════════════════════════════
//  YCbCr exports
// ═══════════════════════════════════════════════════════════════════════════

/// Standard: 0 = BT.601 (JPEG/SD), 1 = BT.709 (HD video).
fn ycbcr_std(v: u32) -> YCbCrStandard {
    if v == 0 { YCbCrStandard::Bt601 } else { YCbCrStandard::Bt709 }
}

#[no_mangle] pub extern "C" fn mid_ycbcr_from_linear(c: CRgb, standard: u32) -> CYCbCr {
    YCbCr::from_linear(Rgb::from(c), ycbcr_std(standard)).into()
}
#[no_mangle] pub extern "C" fn mid_ycbcr_to_linear(c: CYCbCr, standard: u32) -> CRgb {
    YCbCr::from(c).to_linear(ycbcr_std(standard)).into()
}
#[no_mangle] pub extern "C" fn mid_ycbcr_from_srgb_bt601(r:f32,g:f32,b:f32) -> CYCbCr {
    YCbCr::from_srgb_bt601(r,g,b).into()
}
#[no_mangle] pub unsafe extern "C" fn mid_ycbcr_to_srgb_bt601(
    c: CYCbCr, r: *mut f32, g: *mut f32, b: *mut f32,
) {
    let (rv,gv,bv) = YCbCr::from(c).to_srgb_bt601();
    *r=rv; *g=gv; *b=bv;
}
#[no_mangle] pub extern "C" fn mid_ycbcr_from_srgb_bt709(r:f32,g:f32,b:f32) -> CYCbCr {
    YCbCr::from_srgb_bt709(r,g,b).into()
}
#[no_mangle] pub unsafe extern "C" fn mid_ycbcr_to_srgb_bt709(
    c: CYCbCr, r: *mut f32, g: *mut f32, b: *mut f32,
) {
    let (rv,gv,bv) = YCbCr::from(c).to_srgb_bt709();
    *r=rv; *g=gv; *b=bv;
}
#[no_mangle] pub unsafe extern "C" fn mid_ycbcr_to_u8(
    c: CYCbCr, y: *mut u8, cb: *mut u8, cr: *mut u8,
) {
    let (yv, cbv, crv) = YCbCr::from(c).to_u8();
    *y = yv; *cb = cbv; *cr = crv;
}
#[no_mangle] pub extern "C" fn mid_ycbcr_from_u8(y: u8, cb: u8, cr: u8) -> CYCbCr {
    YCbCr::from_u8(y, cb, cr).into()
}
#[no_mangle] pub extern "C" fn mid_ycbcr_to_greyscale(c: CYCbCr) -> CRgb {
    YCbCr::from(c).to_greyscale().into()
}
#[no_mangle] pub extern "C" fn mid_ycbcr_is_achromatic(c: CYCbCr) -> bool {
    YCbCr::from(c).is_achromatic()
}
