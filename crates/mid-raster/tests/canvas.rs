// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-raster.md, section "canvas.rs"
// ============================================================================
//! The canvas: drawing, clips, blend spaces, counted repairs and robustness.

use mid_math::Color32;
use mid_raster::{BlendSpace, Canvas, ClipStack, FillRule, IRect, Path, Surface};

const RED: Color32 = Color32::new(255, 0, 0, 255);
const BLUE: Color32 = Color32::new(0, 0, 255, 255);
const BLACK: Color32 = Color32::new(0, 0, 0, 255);
const CLEAR: Color32 = Color32::new(0, 0, 0, 0);

fn surface(w: u32, h: u32, bg: Color32) -> Surface {
    let mut s = Surface::new(w, h);
    s.as_mut().clear(bg);
    s
}

fn count(s: &Surface, c: Color32) -> usize {
    let mut n = 0;
    for y in 0..s.height() {
        for x in 0..s.width() {
            if s.pixel(x, y) == c {
                n += 1;
            }
        }
    }
    n
}

// ── Rectangles ────────────────────────────────────────────────────────────────

#[test]
fn an_opaque_rectangle_replaces_pixels_and_leaves_the_rest() {
    let mut s = surface(8, 8, BLACK);
    Canvas::new(s.as_mut()).fill_rect(1.0, 2.0, 3.0, 4.0, RED);
    assert_eq!(count(&s, RED), 12);
    assert_eq!(s.pixel(1, 2), RED);
    assert_eq!(s.pixel(3, 5), RED);
    assert_eq!(s.pixel(4, 5), BLACK);
    assert_eq!(s.pixel(0, 0), BLACK);
}

#[test]
fn a_translucent_rectangle_blends_and_an_edge_pixel_blends_by_its_coverage() {
    let mut s = surface(4, 2, BLACK);
    // Half white over black, over x from 0.5 to 2: pixel 0 has half the coverage.
    Canvas::new(s.as_mut()).fill_rect(0.5, 0.0, 1.5, 1.0, Color32::new(255, 255, 255, 128));
    assert_eq!(s.pixel(1, 0), Color32::new(128, 128, 128, 255));
    // Coverage 128 scales alpha 128 to 64, which blends to 64.
    assert_eq!(s.pixel(0, 0), Color32::new(64, 64, 64, 255));
    assert_eq!(s.pixel(2, 0), BLACK);
    assert_eq!(s.pixel(0, 1), BLACK);
}

#[test]
fn drawing_onto_transparent_pixels_keeps_the_source_alpha() {
    let mut s = surface(2, 1, CLEAR);
    Canvas::new(s.as_mut()).fill_rect(0.0, 0.0, 1.0, 1.0, Color32::new(200, 100, 50, 128));
    assert_eq!(s.pixel(0, 0).a, 128);
}

#[test]
fn a_transparent_color_changes_nothing() {
    let mut s = surface(4, 4, BLACK);
    Canvas::new(s.as_mut()).fill_rect(0.0, 0.0, 4.0, 4.0, CLEAR);
    assert_eq!(count(&s, BLACK), 16);
}

#[test]
fn shapes_that_reach_beyond_the_surface_are_cut_to_it() {
    let mut s = surface(4, 4, BLACK);
    Canvas::new(s.as_mut()).fill_rect(-10.0, -10.0, 100.0, 100.0, RED);
    assert_eq!(count(&s, RED), 16);
    let mut t = surface(4, 4, BLACK);
    Canvas::new(t.as_mut()).fill_rect(100.0, 100.0, 5.0, 5.0, RED);
    assert_eq!(count(&t, BLACK), 16);
}

#[test]
fn a_view_with_a_stride_draws_into_the_right_bytes() {
    let mut buf = vec![0u8; 20 * 3];
    {
        let view = mid_raster::SurfaceMut::new(&mut buf, 4, 3, 20).unwrap();
        let mut c = Canvas::new(view);
        c.fill_rect(1.0, 1.0, 2.0, 1.0, RED);
    }
    assert_eq!(&buf[20 + 4..20 + 8], &[255, 0, 0, 255]);
    assert_eq!(&buf[20 + 8..20 + 12], &[255, 0, 0, 255]);
    assert_eq!(&buf[16..20], &[0, 0, 0, 0], "the padding stays untouched");
}

// ── Rounded rectangles, lines and paths ───────────────────────────────────────

#[test]
fn a_rounded_rectangle_leaves_its_corner_pixel_mostly_untouched() {
    let mut s = surface(20, 20, BLACK);
    Canvas::new(s.as_mut()).fill_rounded_rect(0.0, 0.0, 20.0, 20.0, 8.0, RED);
    assert_eq!(s.pixel(10, 10), RED);
    assert_eq!(s.pixel(10, 0), RED);
    assert!(s.pixel(0, 0).r < 20);
    let mut plain = surface(20, 20, BLACK);
    Canvas::new(plain.as_mut()).fill_rounded_rect(0.0, 0.0, 20.0, 20.0, 0.0, RED);
    assert_eq!(count(&plain, RED), 400, "radius zero is the plain rectangle");
}

#[test]
fn a_line_is_drawn_with_butt_caps() {
    let mut s = surface(16, 8, BLACK);
    Canvas::new(s.as_mut()).stroke_line(2.0, 4.0, 10.0, 4.0, 2.0, RED);
    assert_eq!(count(&s, RED), 16);
    assert_eq!(s.pixel(2, 3), RED);
    assert_eq!(s.pixel(9, 4), RED);
    assert_eq!(s.pixel(1, 4), BLACK);
    assert_eq!(s.pixel(10, 4), BLACK);
}

#[test]
fn a_path_fill_follows_its_rule() {
    let mut p = Path::new();
    p.move_to(0.0, 0.0).line_to(6.0, 0.0).line_to(6.0, 6.0).line_to(0.0, 6.0).close();
    p.move_to(2.0, 2.0).line_to(4.0, 2.0).line_to(4.0, 4.0).line_to(2.0, 4.0).close();
    let mut nz = surface(8, 8, BLACK);
    Canvas::new(nz.as_mut()).fill_path(&p, FillRule::NonZero, RED);
    let mut eo = surface(8, 8, BLACK);
    Canvas::new(eo.as_mut()).fill_path(&p, FillRule::EvenOdd, RED);
    assert_eq!(count(&nz, RED), 36);
    assert_eq!(count(&eo, RED), 32);
}

// ── Clips ─────────────────────────────────────────────────────────────────────

#[test]
fn a_clip_limits_drawing_and_a_pop_lifts_it() {
    let mut s = surface(8, 8, BLACK);
    {
        let mut c = Canvas::new(s.as_mut());
        c.push_clip(2.0, 2.0, 4.0, 4.0);
        c.fill_rect(0.0, 0.0, 8.0, 8.0, RED);
        assert_eq!(c.clip_depth(), 1);
        c.pop_clip();
        c.fill_rect(0.0, 0.0, 1.0, 1.0, BLUE);
    }
    assert_eq!(count(&s, RED), 16);
    assert_eq!(s.pixel(2, 2), RED);
    assert_eq!(s.pixel(5, 5), RED);
    assert_eq!(s.pixel(6, 6), BLACK);
    assert_eq!(s.pixel(0, 0), BLUE);
}

#[test]
fn nested_clips_intersect() {
    let mut s = surface(10, 10, BLACK);
    {
        let mut c = Canvas::new(s.as_mut());
        c.push_clip(1.0, 1.0, 6.0, 6.0);
        c.push_clip(4.0, 4.0, 6.0, 6.0);
        assert_eq!(c.clip(), IRect::new(4, 4, 7, 7));
        c.fill_rect(0.0, 0.0, 10.0, 10.0, RED);
    }
    assert_eq!(count(&s, RED), 9);
}

#[test]
fn clip_edges_snap_to_the_nearest_pixel_edge_with_ties_up() {
    let mut s = surface(8, 8, BLACK);
    let mut c = Canvas::new(s.as_mut());
    c.push_clip(1.5, 1.49, 2.0, 2.0);
    // 1.5 snaps to 2 (a tie goes up), 1.49 to 1, 3.5 to 4 and 3.49 to 3.
    assert_eq!(c.clip(), IRect::new(2, 1, 4, 3));
}

#[test]
fn a_clip_wholly_off_the_surface_hides_everything_until_its_pop() {
    let mut s = surface(8, 8, BLACK);
    {
        let mut c = Canvas::new(s.as_mut());
        c.push_clip(100.0, 100.0, 5.0, 5.0);
        c.fill_rect(0.0, 0.0, 8.0, 8.0, RED);
        c.pop_clip();
        c.fill_rect(0.0, 0.0, 1.0, 1.0, BLUE);
    }
    assert_eq!(count(&s, RED), 0);
    assert_eq!(s.pixel(0, 0), BLUE);
}

#[test]
fn a_clip_with_a_value_that_is_not_finite_opens_an_empty_clip_and_stays_paired() {
    let mut s = surface(8, 8, BLACK);
    {
        let mut c = Canvas::new(s.as_mut());
        c.push_clip(f32::NAN, 0.0, 4.0, 4.0);
        assert_eq!(c.clip_depth(), 1);
        c.fill_rect(0.0, 0.0, 8.0, 8.0, RED);
        c.pop_clip();
        assert_eq!(c.clip_depth(), 0);
        assert_eq!(c.stats().ignored_pops, 0);
    }
    assert_eq!(count(&s, RED), 0);
}

#[test]
fn a_pop_with_nothing_open_is_counted_and_open_clips_are_closed_and_counted() {
    let mut s = surface(8, 8, BLACK);
    let mut c = Canvas::new(s.as_mut());
    c.pop_clip();
    c.pop_clip();
    c.push_clip(0.0, 0.0, 4.0, 4.0);
    c.push_clip(0.0, 0.0, 2.0, 2.0);
    let stats = c.finish();
    assert_eq!((stats.ignored_pops, stats.closed_clips), (2, 2));
    assert_eq!(c.clip_depth(), 0);
    assert_eq!(c.finish().closed_clips, 2, "nothing more to close the second time");
}

#[test]
fn the_clip_stack_never_pops_its_bottom_entry() {
    let mut st = ClipStack::new(IRect::new(0, 0, 4, 4));
    assert!(!st.pop());
    st.push(Some((0, 0, 512, 512)));
    assert_eq!(st.current(), IRect::new(0, 0, 2, 2));
    assert!(st.pop());
    assert_eq!(st.current(), IRect::new(0, 0, 4, 4));
    assert_eq!(st.close_all(), 0);
}

// ── Blend space ───────────────────────────────────────────────────────────────

#[test]
fn the_canvas_blends_in_the_space_it_was_given() {
    let half_white = Color32::new(255, 255, 255, 128);
    let mut a = surface(2, 2, BLACK);
    Canvas::new(a.as_mut()).fill_rect(0.0, 0.0, 2.0, 2.0, half_white);
    let mut b = surface(2, 2, BLACK);
    Canvas::new(b.as_mut()).with_blend_space(BlendSpace::Linear).fill_rect(0.0, 0.0, 2.0, 2.0, half_white);
    assert_eq!(a.pixel(0, 0), Color32::new(128, 128, 128, 255));
    assert_eq!(b.pixel(0, 0), Color32::new(188, 188, 188, 255));
}

// ── What is refused, counted ──────────────────────────────────────────────────

#[test]
fn shapes_that_make_no_shape_are_counted_and_draw_nothing() {
    let mut s = surface(8, 8, BLACK);
    let stats = {
        let mut c = Canvas::new(s.as_mut());
        c.fill_rect(0.0, 0.0, -1.0, 4.0, RED);
        c.fill_rect(f32::NAN, 0.0, 4.0, 4.0, RED);
        c.fill_rect(0.0, 0.0, 0.0, 4.0, RED);
        c.fill_rounded_rect(0.0, 0.0, 4.0, 4.0, 1.0, RED);
        c.fill_rounded_rect(0.0, 0.0, f32::INFINITY, 4.0, 1.0, RED);
        c.stroke_line(1.0, 1.0, 1.0, 1.0, 2.0, RED);
        let mut bad = Path::new();
        bad.move_to(0.0, 0.0).line_to(f32::NAN, 1.0).line_to(4.0, 4.0);
        c.fill_path(&bad, FillRule::NonZero, RED);
        c.stats()
    };
    assert!(count(&s, RED) > 0, "the one valid rounded rectangle was drawn");
    assert_eq!(stats.skipped_shapes, 6);
}

#[test]
fn random_garbage_never_panics_or_hangs() {
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut s = surface(16, 16, BLACK);
    let mut c = Canvas::new(s.as_mut());
    let mut f = || f32::from_bits((next() >> 32) as u32);
    for round in 0..3000 {
        let (a, b, d, e, g, h) = (f(), f(), f(), f(), f(), f());
        match round % 6 {
            0 => c.fill_rect(a, b, d, e, RED),
            1 => c.fill_rounded_rect(a, b, d, e, g, BLUE),
            2 => c.stroke_line(a, b, d, e, g, RED),
            3 => {
                let mut p = Path::new();
                p.move_to(a, b).quad_to(d, e, g, h).cubic_to(a, e, d, h, b, g);
                c.fill_path(&p, FillRule::EvenOdd, BLUE);
            }
            4 => c.push_clip(a, b, d, e),
            _ => c.pop_clip(),
        }
    }
    c.finish();
}
