// crates/mid-math/benches/vs_color.rs
//! Color type benchmarks.
//!
//! Groups:
//!   color/srgb_linear  — sRGB ↔ linear conversion (per-channel and struct)
//!   color/hsv           — HSV encode/decode/ops
//!   color/hsl           — HSL encode/decode/ops
//!   color/rgbe          — Radiance HDR encode/decode
//!   color/logluv        — LogLuv32 encode/decode
//!   color/ycbcr         — YCbCr BT.601 / BT.709 encode/decode
//!   color/tone_map      — Reinhard / ACES tone mapping
//!   color/lerp          — Rgb and Rgba interpolation
//!   color/blend         — Color32 Porter-Duff blend
//!   color/premultiply   — Rgba premultiply / unpremultiply
//!   color/bulk_100k     — 100k-entity color conversion throughput

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mid_math::{
    Color32, Hsv, Hsl, LogLuv32, Rgb, Rgba, Rgbe, YCbCr, YCbCrStandard,
};

// ── sRGB ↔ Linear ─────────────────────────────────────────────────────────────

fn bench_srgb_linear(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/srgb_linear");

    let linear = Rgb::new(0.3, 0.6, 0.9);
    let srgb   = Rgb::new(0.5, 0.7, 0.4);

    g.bench_function("to_srgb", |b| {
        b.iter(|| black_box(black_box(linear).to_srgb()))
    });
    g.bench_function("from_srgb", |b| {
        b.iter(|| black_box(black_box(srgb).from_srgb()))
    });
    g.bench_function("single_channel_to_srgb", |b| {
        b.iter(|| black_box(Rgb::linear_to_srgb_ch(black_box(0.5_f32))))
    });
    g.bench_function("single_channel_from_srgb", |b| {
        b.iter(|| black_box(Rgb::srgb_to_linear_ch(black_box(0.5_f32))))
    });
    g.finish();
}

// ── HSV ───────────────────────────────────────────────────────────────────────

fn bench_hsv(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/hsv");

    let rgb = Rgb::new(0.8, 0.4, 0.2);
    let hsv = Hsv::new(210.0, 0.75, 0.9);

    g.bench_function("from_linear_rgb", |b| {
        b.iter(|| black_box(Hsv::from_linear(black_box(rgb))))
    });
    g.bench_function("to_linear_rgb", |b| {
        b.iter(|| black_box(black_box(hsv).to_linear()))
    });
    g.bench_function("shift_hue_90deg", |b| {
        b.iter(|| black_box(black_box(hsv).shift_hue(black_box(90.0_f32))))
    });
    g.bench_function("desaturate_half", |b| {
        b.iter(|| black_box(black_box(hsv).desaturate(black_box(0.5_f32))))
    });
    g.bench_function("lerp", |b| {
        let a = Hsv::new(0.0, 1.0, 1.0);
        let b_val = Hsv::new(240.0, 0.5, 0.8);
        b.iter(|| black_box(black_box(a).lerp(black_box(b_val), black_box(0.5_f32))))
    });
    g.finish();
}

// ── HSL ───────────────────────────────────────────────────────────────────────

fn bench_hsl(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/hsl");

    let rgb = Rgb::new(0.2, 0.6, 0.8);
    let hsl = Hsl::new(200.0, 0.6, 0.5);

    g.bench_function("from_linear_rgb", |b| {
        b.iter(|| black_box(Hsl::from_linear(black_box(rgb))))
    });
    g.bench_function("to_linear_rgb", |b| {
        b.iter(|| black_box(black_box(hsl).to_linear()))
    });
    g.bench_function("lighten", |b| {
        b.iter(|| black_box(black_box(hsl).lighten(black_box(0.2_f32))))
    });
    g.bench_function("darken", |b| {
        b.iter(|| black_box(black_box(hsl).darken(black_box(0.1_f32))))
    });
    g.bench_function("to_hsv", |b| {
        b.iter(|| black_box(black_box(hsl).to_hsv()))
    });
    g.bench_function("lerp", |b| {
        let a = Hsl::new(30.0, 0.8, 0.4);
        let b_val = Hsl::new(270.0, 0.3, 0.7);
        b.iter(|| black_box(black_box(a).lerp(black_box(b_val), black_box(0.5_f32))))
    });
    g.finish();
}

// ── RGBE (Radiance HDR) ───────────────────────────────────────────────────────

fn bench_rgbe(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/rgbe");

    let hdr  = Rgb::new(2.5, 8.1, 0.04);
    let rgbe = Rgbe::encode_rgb(hdr);

    g.bench_function("encode", |b| {
        b.iter(|| black_box(Rgbe::encode(
            black_box(2.5_f32), black_box(8.1_f32), black_box(0.04_f32)
        )))
    });
    g.bench_function("decode", |b| {
        b.iter(|| black_box(black_box(rgbe).decode()))
    });
    g.bench_function("encode_rgb", |b| {
        b.iter(|| black_box(Rgbe::encode_rgb(black_box(hdr))))
    });
    g.bench_function("decode_rgb", |b| {
        b.iter(|| black_box(black_box(rgbe).decode_rgb()))
    });
    g.finish();
}

// ── LogLuv32 ─────────────────────────────────────────────────────────────────

fn bench_logluv(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/logluv");

    let hdr    = Rgb::new(3.14, 0.5, 12.7);
    let packed = LogLuv32::encode_rgb(hdr);

    g.bench_function("encode", |b| {
        b.iter(|| black_box(LogLuv32::encode(
            black_box(3.14_f32), black_box(0.5_f32), black_box(12.7_f32)
        )))
    });
    g.bench_function("decode", |b| {
        b.iter(|| black_box(black_box(packed).decode()))
    });
    g.bench_function("encode_rgb", |b| {
        b.iter(|| black_box(LogLuv32::encode_rgb(black_box(hdr))))
    });
    g.bench_function("decode_rgb", |b| {
        b.iter(|| black_box(black_box(packed).decode_rgb()))
    });
    g.bench_function("luminance_log2_extract", |b| {
        b.iter(|| black_box(black_box(packed).luminance_log2()))
    });
    g.finish();
}

// ── YCbCr ─────────────────────────────────────────────────────────────────────

fn bench_ycbcr(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/ycbcr");

    let rgb  = Rgb::new(0.6, 0.3, 0.8);
    let ycc  = YCbCr::from_linear(rgb, YCbCrStandard::Bt709);

    g.bench_function("encode_bt601", |b| {
        b.iter(|| black_box(YCbCr::from_srgb_bt601(
            black_box(0.6_f32), black_box(0.3_f32), black_box(0.8_f32)
        )))
    });
    g.bench_function("decode_bt601", |b| {
        b.iter(|| black_box(black_box(ycc).to_srgb_bt601()))
    });
    g.bench_function("encode_bt709", |b| {
        b.iter(|| black_box(YCbCr::from_srgb_bt709(
            black_box(0.6_f32), black_box(0.3_f32), black_box(0.8_f32)
        )))
    });
    g.bench_function("decode_bt709", |b| {
        b.iter(|| black_box(black_box(ycc).to_srgb_bt709()))
    });
    g.bench_function("from_linear_bt709", |b| {
        b.iter(|| black_box(YCbCr::from_linear(black_box(rgb), YCbCrStandard::Bt709)))
    });
    g.bench_function("to_linear_bt709", |b| {
        b.iter(|| black_box(black_box(ycc).to_linear(YCbCrStandard::Bt709)))
    });
    g.bench_function("to_u8", |b| {
        b.iter(|| black_box(black_box(ycc).to_u8()))
    });
    g.finish();
}

// ── Tone mapping ──────────────────────────────────────────────────────────────

fn bench_tone_map(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/tone_map");

    let hdr = Rgb::new(2.0, 4.5, 0.8);

    g.bench_function("reinhard_per_channel", |b| {
        b.iter(|| black_box(black_box(hdr).tone_map_reinhard()))
    });
    g.bench_function("reinhard_luminance", |b| {
        b.iter(|| black_box(black_box(hdr).tone_map_reinhard_luminance()))
    });
    g.bench_function("aces", |b| {
        b.iter(|| black_box(black_box(hdr).tone_map_aces()))
    });
    g.bench_function("luminance", |b| {
        b.iter(|| black_box(black_box(hdr).luminance()))
    });
    g.finish();
}

// ── Lerp ─────────────────────────────────────────────────────────────────────

fn bench_lerp(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/lerp");

    let a = Rgb::new(0.1, 0.2, 0.3);
    let b = Rgb::new(0.9, 0.8, 0.7);
    let ra = Rgba::new(0.1, 0.2, 0.3, 0.5);
    let rb = Rgba::new(0.9, 0.8, 0.7, 1.0);

    g.bench_function("rgb_lerp", |b_| {
        b_.iter(|| black_box(black_box(a).lerp(black_box(b), black_box(0.5_f32))))
    });
    g.bench_function("rgba_lerp", |b_| {
        b_.iter(|| black_box(black_box(ra).lerp(black_box(rb), black_box(0.3_f32))))
    });
    g.finish();
}

// ── Color32 blend ─────────────────────────────────────────────────────────────

fn bench_blend(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/blend");

    let dst = Color32::new(80, 120, 200, 255);
    let src = Color32::new(255, 100, 50, 180);

    g.bench_function("blend_over_partial_alpha", |b| {
        b.iter(|| black_box(black_box(dst).blend_over(black_box(src))))
    });
    g.bench_function("blend_over_opaque", |b| {
        let opaque = Color32::new(200, 150, 50, 255);
        b.iter(|| black_box(black_box(dst).blend_over(black_box(opaque))))
    });
    g.bench_function("premultiply", |b| {
        b.iter(|| black_box(black_box(src).premultiply()))
    });
    g.finish();
}

// ── Rgba premultiply ─────────────────────────────────────────────────────────

fn bench_premultiply(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/premultiply");

    let c = Rgba::new(0.8, 0.5, 0.2, 0.7);

    g.bench_function("premultiply_alpha", |b| {
        b.iter(|| black_box(black_box(c).premultiply_alpha()))
    });
    g.bench_function("unpremultiply_alpha", |b| {
        let pm = c.premultiply_alpha();
        b.iter(|| black_box(black_box(pm).unpremultiply_alpha()))
    });
    g.bench_function("to_color32", |b| {
        b.iter(|| black_box(black_box(c).to_color32()))
    });
    g.bench_function("from_color32", |b| {
        let c32 = c.to_color32();
        b.iter(|| black_box(Rgba::from_color32(black_box(c32))))
    });
    g.finish();
}

// ── Bulk 100k throughput ──────────────────────────────────────────────────────

fn bench_bulk_100k(c: &mut Criterion) {
    let mut g = c.benchmark_group("color/bulk_100k");
    g.sample_size(20);

    let colors: Vec<Rgb> = (0..100_000)
        .map(|i| {
            let t = i as f32 / 100_000.0;
            Rgb::new(t, 1.0 - t, 0.5)
        })
        .collect();

    g.bench_function("to_srgb_100k", |b| {
        b.iter(|| {
            for &c in black_box(&colors) {
                let _ = black_box(c.to_srgb());
            }
        })
    });

    g.bench_function("to_logluv_100k", |b| {
        b.iter(|| {
            for &c in black_box(&colors) {
                let _ = black_box(LogLuv32::encode_rgb(c));
            }
        })
    });

    g.bench_function("to_rgbe_100k", |b| {
        b.iter(|| {
            for &c in black_box(&colors) {
                let _ = black_box(Rgbe::encode_rgb(c));
            }
        })
    });

    g.bench_function("tone_map_aces_100k", |b| {
        let hdr: Vec<Rgb> = colors.iter().map(|&c| c * 5.0).collect();
        b.iter(|| {
            for &c in black_box(&hdr) {
                let _ = black_box(c.tone_map_aces());
            }
        })
    });

    g.bench_function("hsv_round_trip_100k", |b| {
        b.iter(|| {
            for &c in black_box(&colors) {
                let hsv = black_box(Hsv::from_linear(c));
                let _ = black_box(hsv.to_linear());
            }
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_srgb_linear,
    bench_hsv,
    bench_hsl,
    bench_rgbe,
    bench_logluv,
    bench_ycbcr,
    bench_tone_map,
    bench_lerp,
    bench_blend,
    bench_premultiply,
    bench_bulk_100k,
);
criterion_main!(benches);
