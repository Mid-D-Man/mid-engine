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

/// Every box's world position, in a fixed order.
fn positions(e: &Editor) -> Vec<(u64, [f32; 3])> {
    let mut v: Vec<(u64, [f32; 3])> = e
        .world()
        .drawables()
        .iter()
        .map(|d| (d.entity.as_ffi(), d.center()))
        .collect();
    v.sort_by_key(|(id, _)| *id);
    v
}

/// The numbers of the `Cube.NNN` rows on screen.
fn visible_cube_numbers(e: &Editor) -> Vec<usize> {
    texts(e)
        .iter()
        .filter_map(|s| s.strip_prefix("Cube."))
        .filter_map(|n| n.parse().ok())
        .collect()
}

/// Drags up (or down) in the panel `times` times, hard enough to cross many rows.
fn scroll_panel(e: &mut Editor, up: bool, times: usize) {
    let panel = e.areas().panel;
    let (a, b) = (panel.y + panel.h - 60.0, panel.y + 60.0);
    let (from, to) = if up { (a, b) } else { (b, a) };
    for _ in 0..times {
        drag(e, (panel.x + 200.0, from), (panel.x + 200.0, to));
    }
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
    assert_eq!(e.selected(), e.world().find("Cube.002"));
    tap_text(&mut e, "Inspector");
    let t = texts(&e);
    assert!(t.iter().any(|s| s.starts_with("Cube.002  #")), "{t:?}");
    for want in ["LocalTransform", "translation", "rotation", "scale"] {
        assert!(t.iter().any(|s| s == want), "{want} in {t:?}");
    }
}

#[test]
fn a_long_hierarchy_ends_with_a_count_of_the_rest() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "1000");
    assert_eq!(e.world().box_count(), 1000);
    tap_text(&mut e, "Hierarchy");
    let more = texts(&e).into_iter().find(|s| s.starts_with("+ "));
    assert!(more.is_some_and(|s| s.ends_with(" more")));
}

#[test]
fn tapping_a_cube_in_the_viewport_selects_it_and_tapping_empty_space_clears_it() {
    let mut e = editor(W, H);
    let vp = e.areas().viewport;
    let cam = Perspective::new(&e.viewport().orbit, vp);
    let target = e.world().find("Cube.005").unwrap();
    let (c, _) = cam.project(e.world().position_of(target).unwrap()).unwrap();
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
    let rest = positions(&e);
    tap_text(&mut e, "Play");
    assert_eq!(e.mode(), Mode::Play);
    for _ in 0..30 {
        e.frame(0.033);
    }
    let moved = positions(&e);
    assert_ne!(rest, moved);
    tap_text(&mut e, "Edit");
    assert_eq!(e.mode(), Mode::Edit);
    let back = positions(&e);
    assert_eq!(rest, back);
    e.frame(0.5);
    let still = positions(&e);
    assert_eq!(rest, still, "edit mode does not step the world");
}

#[test]
fn the_console_chips_change_the_cube_count_and_clear_the_selection() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Cube.001");
    assert_eq!(e.selected(), e.world().find("Cube.001"));
    tap_text(&mut e, "Console");
    for n in [200usize, 5000, 12] {
        tap_text(&mut e, &n.to_string());
        assert_eq!(e.world().box_count(), n);
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
    assert!(positions(&e)
        .iter()
        .all(|(_, p)| p.iter().all(|v| v.is_finite())));
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

#[test]
fn children_are_indented_under_their_parent_in_the_hierarchy() {
    let e = editor(W, H);
    let root = find_text(&e, "Cube").expect("the root row");
    let child = find_text(&e, "Cube.001").expect("a child row");
    let grandchild = find_text(&e, "Cube.003").expect("a grandchild row");
    assert!(child.0 > root.0, "{child:?} vs {root:?}");
    assert!(grandchild.0 > child.0, "{grandchild:?} vs {child:?}");
}

#[test]
fn dragging_in_the_hierarchy_scrolls_it_and_dragging_back_returns() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "1000");
    tap_text(&mut e, "Hierarchy");
    assert!(find_text(&e, "Cube").is_some());
    assert!(visible_cube_numbers(&e).iter().all(|n| *n < 20));
    let yaw = e.viewport().orbit.yaw;

    scroll_panel(&mut e, true, 12);
    assert!(
        find_text(&e, "Cube").is_none(),
        "the first row scrolled away"
    );
    assert!(
        visible_cube_numbers(&e).iter().any(|n| *n >= 30),
        "{:?}",
        visible_cube_numbers(&e)
    );
    assert_eq!(e.viewport().orbit.yaw, yaw, "a panel drag never orbits");
    let more = texts(&e).into_iter().find(|s| s.starts_with("+ "));
    assert!(more.is_some(), "rows remain below");

    scroll_panel(&mut e, false, 12);
    assert!(find_text(&e, "Cube").is_some(), "back at the top");
}

#[test]
fn the_hierarchy_stops_scrolling_at_the_last_row() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "200");
    tap_text(&mut e, "Hierarchy");
    scroll_panel(&mut e, true, 80);
    assert!(
        find_text(&e, "Cube.199").is_some(),
        "the last row is reachable"
    );
    assert!(
        texts(&e).iter().all(|s| !s.starts_with("+ ")),
        "nothing is left below the last row"
    );
    scroll_panel(&mut e, true, 5);
    assert!(
        find_text(&e, "Cube.199").is_some(),
        "scrolling further stays put"
    );
}

#[test]
fn tapping_a_row_after_scrolling_selects_that_row() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "200");
    tap_text(&mut e, "Hierarchy");
    scroll_panel(&mut e, true, 10);
    let n = *visible_cube_numbers(&e).iter().max().unwrap();
    let name = format!("Cube.{n:03}");
    tap_text(&mut e, &name);
    assert_eq!(e.selected(), e.world().find(&name));
}

#[test]
fn the_inspector_shows_the_selected_entitys_components_and_scrolls() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Cube.003");
    tap_text(&mut e, "Inspector");
    let t = texts(&e);
    assert!(t.iter().any(|s| s == "LocalTransform"), "{t:?}");
    assert!(
        t.iter().all(|s| s != "Bob"),
        "the last component starts off screen"
    );
    // Cube.003 hangs under Cube.002 and the inspector says so by name.
    scroll_panel(&mut e, true, 4);
    let t = texts(&e);
    assert!(t.iter().any(|s| s == "Name"), "{t:?}");
    assert!(t.iter().any(|s| s == "Cube.002"), "{t:?}");
    assert!(t.iter().any(|s| s == "Bob"), "{t:?}");
    scroll_panel(&mut e, false, 4);
    assert!(texts(&e).iter().any(|s| s == "LocalTransform"));
}

#[test]
fn switching_tabs_puts_the_scroll_back_at_the_top() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "200");
    tap_text(&mut e, "Hierarchy");
    scroll_panel(&mut e, true, 10);
    assert!(find_text(&e, "Cube").is_none());
    tap_text(&mut e, "Project");
    tap_text(&mut e, "Hierarchy");
    assert!(find_text(&e, "Cube").is_some());
}

#[test]
fn the_selection_survives_play_and_the_console_says_what_was_restored() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Cube.002");
    let chosen = e.selected();
    assert!(chosen.is_some());
    tap_text(&mut e, "Play");
    for _ in 0..20 {
        e.frame(0.033);
    }
    tap_text(&mut e, "Edit");
    assert_eq!(e.selected(), chosen);
    tap_text(&mut e, "Console");
    let t = texts(&e);
    assert!(
        t.iter().any(|s| s.starts_with("restored 12 entities")),
        "{t:?}"
    );
}

#[test]
fn changing_the_cube_count_while_playing_keeps_playing_and_edit_still_restores() {
    let mut rest: Vec<(u64, [f32; 3])> = mid_editor::world::EditorWorld::demo(200)
        .drawables()
        .iter()
        .map(|d| (d.entity.as_ffi(), d.center()))
        .collect();
    rest.sort_by_key(|(id, _)| *id);
    let mut e = editor(W, H);
    tap_text(&mut e, "Play");
    tap_text(&mut e, "Console");
    tap_text(&mut e, "200");
    assert!(e.world().is_playing());
    for _ in 0..20 {
        e.frame(0.033);
    }
    assert_ne!(rest, positions(&e), "the new world is being played");
    tap_text(&mut e, "Edit");
    assert!(!e.world().is_playing());
    assert_eq!(rest, positions(&e));
}

#[test]
fn an_empty_inspector_and_hierarchy_never_scroll_or_panic() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Inspector");
    scroll_panel(&mut e, true, 3);
    assert!(texts(&e).iter().any(|s| s == "Nothing selected."));
    tap_text(&mut e, "Hierarchy");
    scroll_panel(&mut e, true, 3);
    scroll_panel(&mut e, false, 3);
    assert!(find_text(&e, "Cube").is_some(), "back at the top");
}

#[test]
fn a_scroll_in_the_inspector_does_not_carry_over_to_the_hierarchy() {
    let mut e = editor(W, H);
    tap_text(&mut e, "Console");
    tap_text(&mut e, "200");
    tap_text(&mut e, "Hierarchy");
    tap_text(&mut e, "Cube.003");
    tap_text(&mut e, "Inspector");
    scroll_panel(&mut e, true, 4);
    assert!(texts(&e).iter().any(|s| s == "Bob"), "scrolled to the end");
    tap_text(&mut e, "Hierarchy");
    assert!(
        find_text(&e, "Cube").is_some(),
        "the hierarchy starts at its top"
    );
}
