// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "tree.rs"
// ============================================================================
//! Tree arena: ordering, removal, generational ids.

use mid_ui::{Rect, Style, Tree};

type S = Style<f32>;

#[test]
fn new_tree_has_only_the_root() {
    let t = Tree::<f32>::new(S::default());
    assert_eq!(t.len(), 1);
    assert!(!t.is_empty());
    assert!(t.contains(t.root()));
    assert_eq!(t.parent(t.root()), None);
    assert!(t.children(t.root()).is_empty());
}

#[test]
fn children_keep_insertion_order_and_know_their_parent() {
    let mut t = Tree::<f32>::new(S::default());
    let a = t.add_child(t.root(), S::default()).unwrap();
    let b = t.add_child(t.root(), S::default()).unwrap();
    let c = t.add_child(a, S::default()).unwrap();
    assert_eq!(t.children(t.root()), &[a, b]);
    assert_eq!(t.children(a), &[c]);
    assert_eq!(t.parent(c), Some(a));
    assert_eq!(t.parent(a), Some(t.root()));
    assert_eq!(t.len(), 4);
}

#[test]
fn add_child_to_a_stale_parent_fails() {
    let mut t = Tree::<f32>::new(S::default());
    let a = t.add_child(t.root(), S::default()).unwrap();
    assert!(t.remove(a));
    assert!(t.add_child(a, S::default()).is_none());
    assert_eq!(t.len(), 1);
}

#[test]
fn remove_takes_the_whole_subtree() {
    let mut t = Tree::<f32>::new(S::default());
    let a = t.add_child(t.root(), S::default()).unwrap();
    let a1 = t.add_child(a, S::default()).unwrap();
    let a2 = t.add_child(a1, S::default()).unwrap();
    let b = t.add_child(t.root(), S::default()).unwrap();
    assert!(t.remove(a));
    for gone in [a, a1, a2] {
        assert!(!t.contains(gone));
        assert!(t.style(gone).is_none());
        assert!(t.rect(gone).is_none());
        assert!(t.children(gone).is_empty());
    }
    assert!(t.contains(b));
    assert_eq!(t.children(t.root()), &[b]);
    assert_eq!(t.len(), 2);
}

#[test]
fn root_and_stale_ids_cannot_be_removed() {
    let mut t = Tree::<f32>::new(S::default());
    let root = t.root();
    assert!(!t.remove(root));
    let a = t.add_child(root, S::default()).unwrap();
    assert!(t.remove(a));
    assert!(!t.remove(a));
    assert_eq!(t.len(), 1);
}

#[test]
fn a_reused_slot_does_not_revive_the_old_id() {
    let mut t = Tree::<f32>::new(S::default());
    let a = t.add_child(t.root(), S::default()).unwrap();
    assert!(t.remove(a));
    let d = t.add_child(t.root(), S { gap: 7.0, ..S::default() }).unwrap();
    assert_ne!(a, d);
    assert!(!t.contains(a));
    assert!(t.style(a).is_none());
    assert_eq!(t.style(d).unwrap().gap, 7.0);
}

#[test]
fn style_mut_edits_in_place_and_rect_starts_zero() {
    let mut t = Tree::<f32>::new(S::default());
    let a = t.add_child(t.root(), S::default()).unwrap();
    t.style_mut(a).unwrap().gap = 3.0;
    assert_eq!(t.style(a).unwrap().gap, 3.0);
    assert_eq!(t.rect(a), Some(Rect::ZERO));
}

#[test]
fn many_add_remove_cycles_keep_len_consistent() {
    let mut t = Tree::<f64>::new(Style::default());
    let mut live = Vec::new();
    for round in 0..50 {
        for _ in 0..10 {
            live.push(t.add_child(t.root(), Style::default()).unwrap());
        }
        if round % 2 == 1 {
            for id in live.drain(..7) {
                assert!(t.remove(id));
            }
        }
        assert_eq!(t.len(), 1 + live.len());
        assert_eq!(t.children(t.root()).len(), live.len());
    }
}
