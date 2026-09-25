// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/barrier.rs"
// ============================================================================

//! Provides `Barrier` and `BarrierWaitResult`.

pub use implementation::{Barrier, BarrierWaitResult};

#[cfg(feature = "std")]
use std::sync as implementation;

#[cfg(not(feature = "std"))]
mod implementation {
    use crate::sync::Mutex;
    use core::fmt;

    /// `no_std` fallback implementation of `Barrier` from the standard
    /// library.
    ///
    /// Lowest priority of Phase 2's four primitives (`docs/mid-platform.md`,
    /// "Build order" -- "rarely used"), and the simplest to get right: real
    /// `spin::Barrier` (source read directly, `spin` 0.10.0's `barrier.rs`,
    /// from crates.io) is itself built on top of `spin::Mutex<BarrierState>`,
    /// not a bespoke atomic protocol. Ported onto this crate's own
    /// [`Mutex`](crate::sync::Mutex) instead of `spin::Mutex` -- same
    /// generation-counter algorithm, same reasoning `sync::mutex` and
    /// `sync::once` already applied: reuse an already-proven primitive from
    /// this same crate rather than inventing a new one.
    pub struct Barrier {
        lock: Mutex<BarrierState>,
        num_threads: usize,
    }

    struct BarrierState {
        count: usize,
        generation_id: usize,
    }

    impl Barrier {
        /// Creates a new barrier that can block a given number of threads.
        ///
        /// A barrier will block `n - 1` threads which call
        /// [`wait`](Self::wait), then wake all of them at once when the
        /// `n`th thread calls it. A barrier created with `n = 0` behaves
        /// identically to one created with `n = 1`.
        #[must_use]
        pub const fn new(n: usize) -> Self {
            Self {
                lock: Mutex::new(BarrierState { count: 0, generation_id: 0 }),
                num_threads: n,
            }
        }

        /// Blocks the current thread until all threads have rendezvoused
        /// here.
        ///
        /// Barriers are re-usable after all threads have rendezvoused once,
        /// and can be used continuously. A single (arbitrary) thread
        /// receives a [`BarrierWaitResult`] for which
        /// [`is_leader`](BarrierWaitResult::is_leader) returns `true`; every
        /// other thread's result returns `false`.
        pub fn wait(&self) -> BarrierWaitResult {
            let mut guard = self.lock.lock().expect("this crate's Mutex never poisons");
            guard.count += 1;

            if guard.count < self.num_threads {
                // Not the leader -- spin until either this generation
                // finishes (every thread has arrived) or a later generation
                // starts (this barrier got reused before we noticed).
                let local_gen = guard.generation_id;
                while local_gen == guard.generation_id && guard.count < self.num_threads {
                    drop(guard);
                    core::hint::spin_loop();
                    guard = self.lock.lock().expect("this crate's Mutex never poisons");
                }
                BarrierWaitResult(false)
            } else {
                // This thread is the last to arrive -- reset for reuse and
                // bump the generation so everyone else's wait loop above
                // notices and returns.
                guard.count = 0;
                guard.generation_id = guard.generation_id.wrapping_add(1);
                BarrierWaitResult(true)
            }
        }
    }

    impl fmt::Debug for Barrier {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Barrier").finish_non_exhaustive()
        }
    }

    /// Returned by [`Barrier::wait`] once every thread has rendezvoused.
    pub struct BarrierWaitResult(bool);

    impl BarrierWaitResult {
        /// Returns `true` if this thread is the "leader thread" for this
        /// call to [`Barrier::wait`]. Exactly one thread per rendezvous
        /// gets `true`; every other thread gets `false`.
        #[must_use]
        pub fn is_leader(&self) -> bool {
            self.0
        }
    }

    impl fmt::Debug for BarrierWaitResult {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("BarrierWaitResult").field("is_leader", &self.is_leader()).finish()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn barrier_of_one_is_immediately_its_own_leader() {
            let b = Barrier::new(1);
            assert!(b.wait().is_leader());
        }

        #[test]
        fn barrier_is_reusable_across_generations() {
            extern crate std;
            use std::sync::Arc;
            use std::sync::mpsc::channel;
            use std::thread;

            fn use_barrier(n: usize, barrier: &Arc<Barrier>) {
                let (tx, rx) = channel();
                let mut handles = std::vec::Vec::new();
                for _ in 0..n - 1 {
                    let barrier = Arc::clone(barrier);
                    let tx = tx.clone();
                    handles.push(thread::spawn(move || {
                        tx.send(barrier.wait().is_leader()).unwrap();
                    }));
                }

                let mut leader_found = barrier.wait().is_leader();
                for _ in 0..n - 1 {
                    if rx.recv().unwrap() {
                        assert!(!leader_found, "only one thread may be leader");
                        leader_found = true;
                    }
                }
                assert!(leader_found, "exactly one thread must be leader");

                for h in handles {
                    h.join().unwrap();
                }
            }

            let barrier = Arc::new(Barrier::new(10));
            use_barrier(10, &barrier);
            // Reused a second time to confirm the generation counter
            // actually resets `count` and unblocks the next round.
            use_barrier(10, &barrier);
        }
    }
}
