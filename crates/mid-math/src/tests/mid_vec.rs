use crate::MidVec;
use std::cell::RefCell;
use std::rc::Rc;

/// Drop-tracking element: records its `id` into a shared log every time an
/// instance is dropped, and owns a real heap allocation (`Box<u32>`) so
/// that a double-free/double-drop bug is very likely to abort the test
/// process outright (glibc's allocator will usually catch it) rather than
/// silently pass.
#[derive(Clone)]
struct Track {
    id: u32,
    payload: Box<u32>,
    log: Rc<RefCell<Vec<u32>>>,
}

impl Track {
    fn new(id: u32, log: &Rc<RefCell<Vec<u32>>>) -> Self {
        Track { id, payload: Box::new(id), log: log.clone() }
    }
}

impl Drop for Track {
    fn drop(&mut self) {
        assert_eq!(*self.payload, self.id, "payload corrupted before drop");
        self.log.borrow_mut().push(self.id);
    }
}

fn new_log() -> Rc<RefCell<Vec<u32>>> {
    Rc::new(RefCell::new(Vec::new()))
}

fn ids(v: &MidVec<Track, 3>) -> Vec<u32> {
    v.iter().map(|t| t.id).collect()
}

// ── basic construction / capacity ───────────────────────────────────────────

#[test]
fn new_is_empty_and_inline() {
    let v: MidVec<i32, 4> = MidVec::new();
    assert_eq!(v.len(), 0);
    assert!(v.is_empty());
    assert_eq!(v.capacity(), 4);
    assert!(!v.spilled());
}

#[test]
fn default_matches_new() {
    let v: MidVec<i32, 4> = MidVec::default();
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 4);
}

#[test]
fn with_capacity_under_n_stays_inline() {
    let v: MidVec<i32, 4> = MidVec::with_capacity(2);
    assert!(!v.spilled());
    assert_eq!(v.capacity(), 4);
}

#[test]
fn with_capacity_over_n_spills_immediately() {
    let v: MidVec<i32, 4> = MidVec::with_capacity(100);
    assert!(v.spilled());
    assert!(v.capacity() >= 100);
    assert_eq!(v.len(), 0);
}

// ── push / pop, inline and spilled ──────────────────────────────────────────

#[test]
fn push_within_inline_capacity() {
    let mut v: MidVec<i32, 4> = MidVec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    assert_eq!(v.len(), 3);
    assert!(!v.spilled());
    assert_eq!(&*v, &[1, 2, 3]);
}

#[test]
fn push_past_inline_capacity_spills() {
    let mut v: MidVec<i32, 2> = MidVec::new();
    v.push(1);
    v.push(2);
    assert!(!v.spilled());
    v.push(3); // must spill here
    assert!(v.spilled());
    assert_eq!(&*v, &[1, 2, 3]);
    assert!(v.capacity() >= 3);
}

#[test]
fn push_many_preserves_order_and_capacity_invariant() {
    let mut v: MidVec<i32, 2> = MidVec::new();
    for i in 0..1000 {
        v.push(i);
        assert!(v.capacity() >= v.len(), "capacity fell below len at i={i}");
    }
    assert_eq!(v.len(), 1000);
    for i in 0..1000 {
        assert_eq!(v[i as usize], i);
    }
}

#[test]
fn pop_returns_none_when_empty() {
    let mut v: MidVec<i32, 4> = MidVec::new();
    assert_eq!(v.pop(), None);
    v.push(5);
    assert_eq!(v.pop(), Some(5));
    assert_eq!(v.pop(), None);
}

#[test]
fn pop_after_spill_returns_correct_values_in_order() {
    let mut v: MidVec<i32, 2> = MidVec::new();
    for i in 0..10 {
        v.push(i);
    }
    for i in (0..10).rev() {
        assert_eq!(v.pop(), Some(i));
    }
    assert_eq!(v.pop(), None);
}

// ── drop correctness: this is the section that would actually catch a
//    double-free / double-drop / leak in the unsafe pointer logic ─────────

#[test]
fn drop_each_inline_element_exactly_once() {
    let log = new_log();
    {
        let mut v: MidVec<Track, 3> = MidVec::new();
        v.push(Track::new(0, &log));
        v.push(Track::new(1, &log));
        // v drops here
    }
    assert_eq!(*log.borrow(), vec![0, 1]);
}

#[test]
fn drop_each_spilled_element_exactly_once() {
    let log = new_log();
    {
        let mut v: MidVec<Track, 3> = MidVec::new();
        for i in 0..7 {
            v.push(Track::new(i, &log));
        }
        assert!(v.spilled());
        // v drops here
    }
    let mut got = log.borrow().clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn swap_remove_last_index_no_double_drop() {
    // Regression test: an earlier draft of swap_remove read the removed
    // slot AND the "last element" slot separately, which is a bug when
    // index == len - 1 (both reads hit the same memory, silently
    // duplicating a value that should have been moved exactly once).
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..4 {
        v.push(Track::new(i, &log));
    }
    let removed = v.swap_remove(3); // last index
    assert_eq!(removed.id, 3);
    assert_eq!(ids(&v), vec![0, 1, 2]);
    drop(removed);
    assert_eq!(*log.borrow(), vec![3]);
    drop(v);
    assert_eq!(*log.borrow(), vec![3, 0, 1, 2]);
}

#[test]
fn swap_remove_non_last_index_moves_last_into_gap() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..4 {
        v.push(Track::new(i, &log));
    }
    let removed = v.swap_remove(0);
    assert_eq!(removed.id, 0);
    assert_eq!(ids(&v), vec![3, 1, 2]);
    drop(removed);
    drop(v);
    let mut got = log.borrow().clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 2, 3]);
}

#[test]
fn remove_preserves_order() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..4 {
        v.push(Track::new(i, &log));
    }
    let removed = v.remove(1);
    assert_eq!(removed.id, 1);
    assert_eq!(ids(&v), vec![0, 2, 3]);
    drop(removed);
    drop(v);
}

#[test]
fn insert_within_capacity() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    v.push(Track::new(0, &log));
    v.push(Track::new(2, &log));
    v.insert(1, Track::new(1, &log));
    assert_eq!(ids(&v), vec![0, 1, 2]);
}

#[test]
fn insert_that_forces_a_spill() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    v.push(Track::new(0, &log));
    v.push(Track::new(1, &log));
    v.push(Track::new(2, &log));
    assert!(!v.spilled());
    v.insert(1, Track::new(99, &log));
    assert!(v.spilled());
    assert_eq!(ids(&v), vec![0, 99, 1, 2]);
}

#[test]
fn clear_drops_everything_exactly_once_and_resets_len() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..5 {
        v.push(Track::new(i, &log));
    }
    v.clear();
    assert_eq!(v.len(), 0);
    let mut got = log.borrow().clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
    // capacity survives a clear (we don't deallocate on clear)
    assert!(v.capacity() >= 5);
}

#[test]
fn truncate_drops_only_the_tail() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..6 {
        v.push(Track::new(i, &log));
    }
    v.truncate(2);
    assert_eq!(v.len(), 2);
    assert_eq!(ids(&v), vec![0, 1]);
    let mut got = log.borrow().clone();
    got.sort_unstable();
    assert_eq!(got, vec![2, 3, 4, 5]); // only the truncated tail so far
    drop(v);
    let mut got_all = log.borrow().clone();
    got_all.sort_unstable();
    assert_eq!(got_all, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn truncate_to_len_or_more_is_a_no_op() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    v.push(Track::new(0, &log));
    v.truncate(5);
    assert_eq!(v.len(), 1);
    assert!(log.borrow().is_empty());
}

// ── from_vec / into_vec: zero-copy adoption in both directions ─────────────

#[test]
fn from_vec_with_capacity_adopts_the_allocation() {
    let mut src: Vec<i32> = Vec::with_capacity(10);
    src.push(1);
    src.push(2);
    let original_ptr = src.as_ptr();
    let v: MidVec<i32, 3> = MidVec::from_vec(src);
    // len (2) is <= N (3), but from_vec must still report spilled, proving
    // it adopted the Vec's own heap buffer rather than copying into inline
    // storage.
    assert!(v.spilled());
    assert_eq!(v.capacity(), 10);
    assert_eq!(&*v, &[1, 2]);
    assert_eq!(v.as_ptr(), original_ptr, "from_vec must not reallocate/copy");
}

#[test]
fn from_vec_with_zero_capacity_is_inline_empty() {
    let v: MidVec<i32, 4> = MidVec::from_vec(Vec::new());
    assert!(!v.spilled());
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 4);
}

#[test]
fn into_vec_from_spilled_is_zero_copy_and_does_not_drop_elements() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..6 {
        v.push(Track::new(i, &log));
    }
    assert!(v.spilled());
    let vec = v.into_vec();
    // Ownership moved to `vec`; nothing should have been dropped yet.
    assert!(log.borrow().is_empty(), "into_vec must not drop elements itself");
    assert_eq!(vec.len(), 6);
    drop(vec);
    let mut got = log.borrow().clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn into_vec_from_inline_copies_without_dropping_originals() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    v.push(Track::new(0, &log));
    v.push(Track::new(1, &log));
    assert!(!v.spilled());
    let vec = v.into_vec();
    assert!(log.borrow().is_empty(), "into_vec must not drop elements itself");
    assert_eq!(vec.iter().map(|t| t.id).collect::<Vec<_>>(), vec![0, 1]);
    drop(vec);
    let mut got = log.borrow().clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1]);
}

#[test]
fn round_trip_vec_conversion_preserves_data() {
    let original = vec![10, 20, 30, 40, 50];
    let sv: MidVec<i32, 2> = MidVec::from(original.clone());
    let back: Vec<i32> = sv.into();
    assert_eq!(original, back);
}

// ── IntoIterator (by value): forward, backward, and partial-drop safety ────

#[test]
fn into_iter_yields_all_elements_in_order() {
    let mut v: MidVec<i32, 2> = MidVec::new();
    for i in 0..8 {
        v.push(i);
    }
    let collected: Vec<i32> = v.into_iter().collect();
    assert_eq!(collected, (0..8).collect::<Vec<_>>());
}

#[test]
fn into_iter_double_ended() {
    let v: MidVec<i32, 2> = (0..6).collect();
    let mut it = v.into_iter();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(5));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next_back(), Some(4));
    let rest: Vec<i32> = it.collect();
    assert_eq!(rest, vec![2, 3]);
}

#[test]
fn into_iter_partial_consumption_then_drop_is_leak_and_double_drop_free() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    for i in 0..5 {
        v.push(Track::new(i, &log));
    }
    let mut it = v.into_iter();
    it.next().unwrap(); // id 0, dropped immediately (temporary, not bound)
    it.next().unwrap(); // id 1, dropped immediately
    assert_eq!(*log.borrow(), vec![0, 1]);
    drop(it); // must drop remaining 2, 3, 4 exactly once each, in order
    assert_eq!(*log.borrow(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn into_iter_size_hint_and_len() {
    let v: MidVec<i32, 2> = (0..5).collect();
    let it = v.into_iter();
    assert_eq!(it.len(), 5);
    assert_eq!(it.size_hint(), (5, Some(5)));
}

// ── reference iteration ──────────────────────────────────────────────────

#[test]
fn iter_and_iter_mut() {
    let mut v: MidVec<i32, 2> = (0..5).collect();
    assert_eq!(v.iter().sum::<i32>(), 10);
    for x in v.iter_mut() {
        *x *= 2;
    }
    assert_eq!(&*v, &[0, 2, 4, 6, 8]);
    // `for x in &v` / `for x in &mut v` go through the reference
    // IntoIterator impls, not just `.iter()`/`.iter_mut()` directly.
    let mut sum = 0;
    for x in &v {
        sum += *x;
    }
    assert_eq!(sum, 20);
}

// ── Clone ────────────────────────────────────────────────────────────────

#[test]
fn clone_is_a_deep_independent_copy_inline() {
    let log = new_log();
    let mut v: MidVec<Track, 3> = MidVec::new();
    v.push(Track::new(0, &log));
    v.push(Track::new(1, &log));
    let mut cloned = v.clone();
    cloned.push(Track::new(2, &log));
    assert_eq!(ids(&v), vec![0, 1]);
    assert_eq!(ids(&cloned), vec![0, 1, 2]);
    drop(v);
    let mut first = log.borrow().clone();
    first.sort_unstable();
    assert_eq!(first, vec![0, 1]);
    drop(cloned);
    let mut all = log.borrow().clone();
    all.sort_unstable();
    assert_eq!(all, vec![0, 0, 1, 1, 2]);
}

#[test]
fn clone_is_a_deep_independent_copy_spilled() {
    let mut v: MidVec<i32, 2> = MidVec::new();
    for i in 0..10 {
        v.push(i);
    }
    let mut cloned = v.clone();
    cloned.push(999);
    assert_eq!(v.len(), 10);
    assert_eq!(cloned.len(), 11);
    assert_ne!(v.as_ptr(), cloned.as_ptr(), "clone must not alias storage");
}

// ── equality / debug / from-slice / collect ────────────────────────────────

#[test]
fn partial_eq_against_slice_vec_and_other_n() {
    let v: MidVec<i32, 4> = MidVec::from(vec![1, 2, 3]);
    assert_eq!(v, [1, 2, 3][..]);
    assert_eq!(v, vec![1, 2, 3]);
    let other: MidVec<i32, 8> = MidVec::from(vec![1, 2, 3]);
    assert_eq!(v, other);
    let different: MidVec<i32, 4> = MidVec::from(vec![1, 2, 4]);
    assert_ne!(v, different);
}

#[test]
fn debug_formatting_looks_like_a_list() {
    let v: MidVec<i32, 4> = MidVec::from(vec![1, 2, 3]);
    assert_eq!(format!("{v:?}"), "[1, 2, 3]");
}

#[test]
fn from_slice_and_extend_from_slice() {
    let v: MidVec<i32, 2> = MidVec::from_slice(&[1, 2, 3, 4]);
    assert_eq!(&*v, &[1, 2, 3, 4]);
    assert!(v.spilled());

    let mut w: MidVec<i32, 4> = MidVec::new();
    w.extend_from_slice(&[5, 6]);
    w.extend_from_slice(&[7, 8, 9]);
    assert_eq!(&*w, &[5, 6, 7, 8, 9]);
}

#[test]
fn from_iterator_and_extend() {
    let v: MidVec<i32, 4> = (0..10).collect();
    assert_eq!(v.len(), 10);
    assert!(v.spilled());
    assert_eq!(&*v, &(0..10).collect::<Vec<_>>()[..]);

    let mut w: MidVec<i32, 4> = MidVec::new();
    w.extend(0..3);
    w.extend(3..6);
    assert_eq!(&*w, &[0, 1, 2, 3, 4, 5]);
}

#[test]
fn from_fixed_array() {
    let v: MidVec<i32, 4> = MidVec::from([1, 2, 3]);
    assert_eq!(&*v, &[1, 2, 3]);
    assert!(!v.spilled());
}

#[test]
#[should_panic(expected = "zero-sized")]
fn zero_sized_type_panics_instead_of_silently_misbehaving() {
    let _v: MidVec<(), 4> = MidVec::new();
}

// ── alignment: mid-math's Vec3/Vec4 use SIMD types with real alignment
//    requirements (e.g. 16-byte for SSE2 __m128-backed storage) ────────────

#[test]
fn respects_over_alignment_inline_and_spilled() {
    #[repr(align(16))]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Aligned16(f32, f32, f32, f32);

    assert_eq!(std::mem::align_of::<Aligned16>(), 16);

    let mut v: MidVec<Aligned16, 2> = MidVec::new();
    for i in 0..8 {
        v.push(Aligned16(i as f32, 0.0, 0.0, 0.0));
        for item in v.iter() {
            let addr = item as *const Aligned16 as usize;
            assert_eq!(addr % 16, 0, "element misaligned at len={}", v.len());
        }
    }
    assert!(v.spilled());
}
