//! The editor through its public interface, as a host uses it.

use mid_editor::camera::{Area, Perspective};
use mid_editor::paint::{KIND_POP_CLIP, KIND_PUSH_CLIP, KIND_TEXT};
use mid_editor::text::TextMeasure;
use mid_editor::{Editor, FixedWidth, Mode, Phase, Tab};

const W: f32 = 1080.0;
const H: f32 = 2408.0;

fn editor(w: f32, h: f32) -> Editor {
    let mut e = Editor::new(w, h, Box::new(FixedWidth(0.5)));
    e.frame(0.016);
    e
}

/// The top-left corner and the size of the first text item that reads `s`.
fn find_text(e: &Editor, s: &str) -> Option<(f32, f32, f32)> {
    let p = e.paint();
    p.items
        .iter()
        .find(|i| i.kind == KIND_TEXT && p.text_of(i) == Some(s))
        .map(|i| (i.a, i.b, i.extra))
}

fn tap(e: &mut Editor, x: f32, y: f32) {
    e.pointer(Phase::Down, 0, x, y);
    e.frame(0.016);
    e.pointer(Phase::Up, 0, x, y);
    e.frame(0.016);
}

fn tap_text(e: &mut Editor, s: &str) {
    let (x, y, size) = find_text(e, s).unwrap_or_else(|| panic!("no text {s:?} on screen"));
    tap(e, x + 4.0, y + size * 0.5);
}

fn drag(e: &mut Editor, from: (f32, f32), to: (f32, f32)) {
    e.pointer(Phase::Down, 0, from.0, from.1);
    let steps = 6;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        e.pointer(
            Phase::Move,
            0,
            from.0 + (to.0 - from.0) * t,
            from.1 + (to.1 - from.1) * t,
        );
        e.frame(0.016);
    }
    e.pointer(Phase::Up, 0, to.0, to.1);
    e.frame(0.016);
}

fn texts(e: &Editor) -> Vec<String> {
    let p = e.paint();
    p.items
        .iter()
        .filter_map(|i| p.text_of(i).map(str::to_string))
        .collect()
}

#[test]
fn portrait_stacks_top_bar_viewport_tabs_and_panel_without_overlap() {
    let e = editor(W, H);
    let a = e.areas();
    assert_eq!(a.top.y, 0.0);
    assert!((a.top.y + a.top.h - a.viewport.y).abs() < 0.01);
    assert!((a.viewport.y + a.viewport.h - a.tabs.y).abs() < 0.01);
    assert!((a.tabs.y + a.tabs.h - a.panel.y).abs() < 0.01);
    assert!((a.panel.y + a.panel.h - H).abs() < 0.01);
    assert!(
        a.viewport.h > a.panel.h,
        "the viewport is the larger of the two"
    );
    for r in [a.top, a.viewport, a.tabs, a.panel] {
        assert_eq!((r.x, r.w), (0.0, W));
    }
    assert!((e.unit() - W / 411.0).abs() < 1e-4);
}

#[test]
fn landscape_puts_the_viewport_beside_the_panel() {
    let e = editor(H, W);
    let a = e.areas();
    assert!(a.viewport.x == 0.0 && a.panel.x >= a.viewport.x + a.viewport.w - 0.01);
    assert!(a.panel.x == a.tabs.x);
    assert!((a.tabs.y + a.tabs.h - a.panel.y).abs() < 0.01);
    assert!((a.viewport.y - a.top.h).abs() < 0.01);
}

#[test]
fn resizing_rebuilds_the_layout_and_bad_sizes_are_ignored() {
    let mut e = editor(W, H);
    e.resize(H, W);
    assert!(e.areas().viewport.w > e.areas().viewport.h * 0.5);
    let unit = e.unit();
    for bad in [
        (0.0, 10.0),
        (10.0, -1.0),
        (f32::NAN, 10.0),
        (10.0, f32::INFINITY),
    ] {
        e.resize(bad.0, bad.1);
    }
    assert_eq!(e.size(), (H, W));
    assert_eq!(e.unit(), unit);
}

#[test]
fn a_bad_initial_size_falls_back_to_a_phone_sized_screen() {
    let e = Editor::new(0.0, f32::NAN, Box::new(FixedWidth(0.5)));
    assert_eq!(e.size(), (411.0, 891.0));
}

#[test]
fn the_top_bar_and_tab_strip_show_their_labels() {
    let e = editor(W, H);
    let t = texts(&e);
    for want in [
        "Edit",
        "Play",
        "3D",
        "2D",
        "Hierarchy",
        "Inspector",
        "Project",
        "Console",
    ] {
        assert!(t.iter().any(|s| s == want), "missing {want}: {t:?}");
    }
}

#[test]
fn tapping_a_tab_switches_the_panel() {
    let mut e = editor(W, H);
    assert_eq!(e.tab(), Tab::Hierarchy);
    tap_text(&mut e, "Inspector");
    assert_eq!(e.tab(), Tab::Inspector);
    assert!(texts(&e).iter().any(|s| s == "Nothing selected."));
    tap_text(&mut e, "Project");
    assert_eq!(e.tab(), Tab::Project);
    assert!(texts(&e).iter().any(|s| s.contains("demo.scene")));
    tap_text(&mut e, "Console");
    assert_eq!(e.tab(), Tab::Console);
    assert!(texts(&e).iter().any(|s| s.starts_with("fps ")));
    tap_text(&mut e, "Hierarchy");
    assert_eq!(e.tab(), Tab::Hierarchy);
}

#[test]
fn the_hierarchy_lists_the_cubes_and_a_tap_on_a_row_selects_it() {
    let mut e = editor(W, H);
    assert!(texts(&e).iter().any(|s| s == "Cube.002"));
    tap_text(&mut e, "Cube.002");
    assert_eq!(e.selected(), Some(2));
    tap_text(&mut e, "Inspector");
    let t = texts(&e);
    assert!(t.iter().any(|s| s == "Cube.002"), "{t:?}");
    assert!(t.iter().any(|s| s.starts_with("Position")));
    assert!(t.iter().any(|s| s.starts_with("Size")));
}

#[test]
fn a_long_hierarchy_ends_with_a_count_of_the_rest() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "1000");
    assert_eq!(e.scene().len(), 1000);
    tap_text(&mut e, "Hierarchy");
    let more = texts(&e).into_iter().find(|s| s.starts_with("+ "));
    assert!(more.is_some_and(|s| s.ends_with(" more")));
}

#[test]
fn tapping_a_cube_in_the_viewport_selects_it_and_tapping_empty_space_clears_it() {
    let mut e = editor(W, H);
    let vp = e.areas().viewport;
    let cam = Perspective::new(&e.viewport().orbit, vp);
    let (c, _) = cam.project(e.scene().entities[5].pos).unwrap();
    tap(&mut e, c[0], c[1]);
    assert!(e.selected().is_some(), "a cube was under the finger");
    tap(&mut e, vp.x + 3.0, vp.y + vp.h - 3.0);
    assert_eq!(e.selected(), None);
}

#[test]
fn a_drag_in_the_viewport_orbits_and_a_drag_in_the_panel_does_not() {
    let mut e = editor(W, H);
    let yaw = e.viewport().orbit.yaw;
    let vp = e.areas().viewport;
    drag(
        &mut e,
        (vp.x + 100.0, vp.y + 300.0),
        (vp.x + 400.0, vp.y + 300.0),
    );
    assert_ne!(e.viewport().orbit.yaw, yaw);

    let yaw = e.viewport().orbit.yaw;
    let panel = e.areas().panel;
    drag(
        &mut e,
        (panel.x + 100.0, panel.y + 300.0),
        (panel.x + 500.0, panel.y + 300.0),
    );
    assert_eq!(e.viewport().orbit.yaw, yaw);
}

#[test]
fn a_drag_that_starts_in_the_viewport_keeps_orbiting_when_it_leaves_it() {
    let mut e = editor(W, H);
    let yaw = e.viewport().orbit.yaw;
    let vp = e.areas().viewport;
    drag(
        &mut e,
        (vp.x + 500.0, vp.y + vp.h - 20.0),
        (vp.x + 800.0, vp.y + vp.h + 400.0),
    );
    assert_ne!(e.viewport().orbit.yaw, yaw);
}

#[test]
fn two_fingers_zoom_the_viewport() {
    let mut e = editor(W, H);
    let d = e.viewport().orbit.dist;
    let vp = e.areas().viewport;
    let (cx, cy) = (vp.x + vp.w * 0.5, vp.y + vp.h * 0.5);
    e.pointer(Phase::Down, 0, cx - 100.0, cy);
    e.pointer(Phase::Down, 1, cx + 100.0, cy);
    e.pointer(Phase::Move, 1, cx + 300.0, cy);
    e.frame(0.016);
    e.pointer(Phase::Up, 1, cx + 300.0, cy);
    e.pointer(Phase::Up, 0, cx - 100.0, cy);
    assert!(e.viewport().orbit.dist < d, "fingers apart moves in");
}

#[test]
fn the_2d_button_switches_to_a_top_down_view_and_3d_switches_back() {
    let mut e = editor(W, H);
    let lines_3d = e.paint().items.len();
    tap_text(&mut e, "2D");
    assert!(e.viewport().mode2d);
    assert_ne!(e.paint().items.len(), lines_3d);
    let scale = e.viewport().ortho.scale;
    let vp = e.areas().viewport;
    let (cx, cy) = (vp.x + vp.w * 0.5, vp.y + vp.h * 0.5);
    e.pointer(Phase::Down, 0, cx - 100.0, cy);
    e.pointer(Phase::Down, 1, cx + 100.0, cy);
    e.pointer(Phase::Move, 1, cx + 300.0, cy);
    assert!(e.viewport().ortho.scale > scale);
    e.pointer(Phase::Up, 1, cx + 300.0, cy);
    e.pointer(Phase::Up, 0, cx - 100.0, cy);
    tap_text(&mut e, "3D");
    assert!(!e.viewport().mode2d);
}

#[test]
fn play_mode_moves_the_world_and_edit_mode_puts_it_back() {
    let mut e = editor(W, H);
    let rest: Vec<[f32; 3]> = e.scene().entities.iter().map(|x| x.pos).collect();
    tap_text(&mut e, "Play");
    assert_eq!(e.mode(), Mode::Play);
    for _ in 0..30 {
        e.frame(0.033);
    }
    let moved: Vec<[f32; 3]> = e.scene().entities.iter().map(|x| x.pos).collect();
    assert_ne!(rest, moved);
    tap_text(&mut e, "Edit");
    assert_eq!(e.mode(), Mode::Edit);
    let back: Vec<[f32; 3]> = e.scene().entities.iter().map(|x| x.pos).collect();
    assert_eq!(rest, back);
    e.frame(0.5);
    let still: Vec<[f32; 3]> = e.scene().entities.iter().map(|x| x.pos).collect();
    assert_eq!(rest, still, "edit mode does not step the world");
}

#[test]
fn the_console_chips_change_the_cube_count_and_clear_the_selection() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Cube.001");
    assert_eq!(e.selected(), Some(1));
    tap_text(&mut e, "Console");
    for n in [200usize, 5000, 12] {
        tap_text(&mut e, &n.to_string());
        assert_eq!(e.scene().len(), n);
        assert_eq!(e.selected(), None);
    }
}

#[test]
fn the_console_shows_statistics_and_the_log_after_frames() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Play");
    tap_text(&mut e, "Console");
    for _ in 0..60 {
        e.frame(0.020);
    }
    let t = texts(&e);
    assert!(t.iter().any(|s| s.starts_with("fps 50.0")), "{t:?}");
    assert!(t.iter().any(|s| s.starts_with("screen 1080 x 2408")));
    assert!(t.iter().any(|s| s.starts_with("touch: tap")));
    assert!(
        t.iter().any(|s| s == "mode Play"),
        "the log shows the last actions: {t:?}"
    );
}

#[test]
fn the_version_text_is_shown_when_it_fits_and_never_longer_than_24_characters() {
    let mut e = editor(W, H);
    e.set_version("1.2.3");
    e.frame(0.016);
    assert!(texts(&e).iter().any(|s| s == "v1.2.3"));

    e.set_version(&"x".repeat(40));
    e.frame(0.016);
    let long = texts(&e).into_iter().find(|s| s.starts_with("vx"));
    assert!(
        long.map(|s| s.len() <= 25).unwrap_or(true),
        "kept to 24 characters or left out"
    );
    assert!(!texts(&e).iter().any(|s| s.contains(&"x".repeat(25))));
}

#[test]
fn cancelling_pointers_lets_the_next_touch_tap() {
    let mut e = editor(W, H);
    let vp = e.areas().viewport;
    e.pointer(Phase::Down, 0, vp.x + 50.0, vp.y + 50.0);
    e.pointer(Phase::Down, 1, vp.x + 90.0, vp.y + 50.0);
    e.cancel_pointers();
    tap_text(&mut e, "Console");
    assert_eq!(e.tab(), Tab::Console);
}

#[test]
fn every_text_item_fits_on_the_screen_in_every_tab_and_orientation() {
    let m = FixedWidth(0.5);
    for (w, h) in [(W, H), (H, W), (411.0, 891.0), (320.0, 480.0)] {
        let mut e = editor(w, h);
        for tab in ["Hierarchy", "Inspector", "Project", "Console"] {
            tap_text(&mut e, tab);
            let p = e.paint();
            for i in p.items.iter().filter(|i| i.kind == KIND_TEXT) {
                let s = p.text_of(i).unwrap();
                assert!(
                    i.a >= -0.5 && i.a + m.width(s, i.extra) <= w + 0.5,
                    "{w}x{h} {tab}: {s:?} at {} width {}",
                    i.a,
                    m.width(s, i.extra)
                );
            }
        }
    }
}

#[test]
fn paint_lists_are_always_finite_balanced_and_valid_utf8() {
    for (w, h) in [
        (W, H),
        (H, W),
        (1.0, 1.0),
        (50.0, 3000.0),
        (10_000.0, 10_000.0),
    ] {
        let mut e = editor(w, h);
        let mut rng = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        for step in 0..400 {
            let phase = match (next() * 4.0) as u32 {
                0 => Phase::Down,
                1 => Phase::Move,
                2 => Phase::Up,
                _ => Phase::Cancel,
            };
            e.pointer(
                phase,
                (next() * 6.0) as i32,
                next() * w * 1.2 - 0.1 * w,
                next() * h * 1.2 - 0.1 * h,
            );
            if step % 50 == 0 {
                e.resize(w * (0.5 + next()), h * (0.5 + next()));
            }
            e.frame(next() * 0.05);
            let p = e.paint();
            let mut depth = 0i32;
            for i in &p.items {
                assert!(
                    [i.a, i.b, i.c, i.d, i.extra].iter().all(|v| v.is_finite()),
                    "{i:?}"
                );
                if i.kind == KIND_PUSH_CLIP {
                    depth += 1;
                }
                if i.kind == KIND_POP_CLIP {
                    depth -= 1;
                    assert!(depth >= 0, "a pop with no push");
                }
                if i.kind == KIND_TEXT {
                    assert!(p.text_of(i).is_some());
                }
            }
            assert_eq!(depth, 0, "clips are balanced");
        }
    }
}

#[test]
fn odd_frame_times_do_not_break_anything() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Play");
    for dt in [f32::NAN, -1.0, f32::INFINITY, 0.0, 1.0e9, 0.016] {
        e.frame(dt);
    }
    assert!(e
        .paint()
        .items
        .iter()
        .all(|i| i.a.is_finite() && i.b.is_finite()));
    assert!(e
        .scene()
        .entities
        .iter()
        .all(|x| x.pos.iter().all(|v| v.is_finite())));
}

#[test]
fn a_big_scene_still_paints_in_a_bounded_number_of_items() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "5000");
    e.frame(0.016);
    let n = e.paint().items.len();
    assert!(n < 5000 * 12 + 500, "{n}");
    assert!(e.stats().core_ms() >= 0.0);
}

#[test]
fn the_area_helper_matches_what_the_editor_reports() {
    let e = editor(W, H);
    let a = e.areas();
    let _ = Area {
        x: a.viewport.x,
        y: a.viewport.y,
        w: a.viewport.w,
        h: a.viewport.h,
    };
    assert!(a.viewport.contains(a.viewport.x + 1.0, a.viewport.y + 1.0));
}
