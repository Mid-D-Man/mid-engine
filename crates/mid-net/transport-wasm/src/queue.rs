//! A minimal single-threaded queue with an async "wait for the next item"
//! side. Hand-rolled rather than reaching for `tokio::sync::mpsc` (not
//! available here at all -- `mid-net-transport-quinn` uses it, but that's
//! native-only) or an extra dependency like `futures::channel::mpsc`:
//! wasm32 in a browser is single-threaded, so this doesn't need to be
//! `Send`/`Sync`, doesn't need atomics, and doesn't need a real channel's
//! multi-producer bookkeeping -- `Rc<RefCell<..>>` plus one stored
//! `Waker` is the whole implementation. In keeping with this project's
//! standing call on hand-rolling over pulling in weight that isn't
//! earning its keep (see `docs/architecture.md`'s dependency mandate).
//!
//! Two operations, on purpose, not a general-purpose channel:
//! - `push` — sync, never blocks, called from `Transport::send_reliable`
//!   (which must not block either).
//! - `next` — async, awaited by exactly one background task. Only one
//!   `Waker` slot is kept; this is not safe to await from two places at
//!   once. That's fine here — every queue in this crate has exactly one
//!   reader task by construction, never checked at runtime because
//!   nothing in this crate's own code ever violates it.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

pub struct WakeQueue<T> {
    inner: Rc<RefCell<Inner<T>>>,
}

struct Inner<T> {
    items: VecDeque<T>,
    waker: Option<Waker>,
}

impl<T> WakeQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(Inner { items: VecDeque::new(), waker: None })),
        }
    }

    /// A second handle to the same underlying queue -- `Rc::clone`, not a
    /// deep copy. Named distinctly from `Clone::clone` so a `#[derive(Clone)]`
    /// on a struct holding one of these doesn't silently do the right
    /// thing by accident without the reader noticing this is a shared
    /// handle, not an independent queue.
    pub fn handle(&self) -> Self {
        Self { inner: Rc::clone(&self.inner) }
    }

    /// Sync, non-blocking. Wakes the waiting `next()` future, if any.
    pub fn push(&self, item: T) {
        let mut inner = self.inner.borrow_mut();
        inner.items.push_back(item);
        if let Some(waker) = inner.waker.take() {
            waker.wake();
        }
    }

    /// Sync, non-blocking pop -- used by `Transport::poll_datagram`/
    /// `poll_reliable`, which must return immediately either way.
    pub fn try_pop(&self) -> Option<T> {
        self.inner.borrow_mut().items.pop_front()
    }

    /// Async pop -- awaited by this crate's single background writer
    /// task to wait for the next outgoing message without busy-polling.
    pub fn next(&self) -> Next<'_, T> {
        Next { queue: self }
    }
}

impl<T> Default for WakeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Next<'a, T> {
    queue: &'a WakeQueue<T>,
}

impl<T> Future for Next<'_, T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut inner = self.queue.inner.borrow_mut();
        match inner.items.pop_front() {
            Some(item) => Poll::Ready(item),
            None => {
                inner.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{RawWaker, RawWakerVTable};

    // A waker that does nothing when woken, just records that `wake` was
    // called at least once -- enough to test "next() actually resolves
    // after a push that arrives while it's pending" without needing a
    // real executor, which this crate deliberately doesn't depend on.
    fn noop_raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            noop_raw_waker()
        }
        fn no_op(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }

    fn noop_context() -> Context<'static> {
        // SAFETY: the vtable's functions are all valid no-ops that never
        // touch the null data pointer, for the lifetime of this waker.
        let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
        Context::from_waker(Box::leak(Box::new(waker)))
    }

    #[test]
    fn push_then_try_pop_round_trips_in_order() {
        let q: WakeQueue<i32> = WakeQueue::new();
        q.push(1);
        q.push(2);
        assert_eq!(q.try_pop(), Some(1));
        assert_eq!(q.try_pop(), Some(2));
        assert_eq!(q.try_pop(), None);
    }

    #[test]
    fn next_resolves_immediately_when_an_item_is_already_present() {
        let q: WakeQueue<&str> = WakeQueue::new();
        q.push("already here");

        let mut fut = q.next();
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
        let cx = noop_context();
        let mut cx = cx;
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(item) => assert_eq!(item, "already here"),
            Poll::Pending => panic!("expected Ready, item was already pushed"),
        }
    }

    #[test]
    fn next_is_pending_on_an_empty_queue_then_ready_after_a_push() {
        let q: WakeQueue<i32> = WakeQueue::new();

        let mut fut = q.next();
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
        let cx = noop_context();
        let mut cx = cx;

        assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Pending));

        q.push(42);
        // A real executor would be woken here (the no-op waker above
        // doesn't schedule a re-poll, it just proves `wake()` doesn't
        // panic) -- polling again directly is what the executor would
        // eventually do on our behalf, and this proves the item is
        // actually there afterward, which the wake alone doesn't.
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(item) => assert_eq!(item, 42),
            Poll::Pending => panic!("expected Ready after push"),
        }
    }

    #[test]
    fn two_handles_share_the_same_underlying_queue() {
        let a: WakeQueue<i32> = WakeQueue::new();
        let b = a.handle();
        a.push(1);
        assert_eq!(b.try_pop(), Some(1));
    }
}
