// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "resource.rs"
// ============================================================================
//! Resources: at most one value per Rust type, owned by the `World` but
//! attached to no entity (a clock, a configuration, an input snapshot).
//!
//! Rust callers use the typed API on `World` (`insert_resource`,
//! `get_resource`, `get_resource_mut`, `remove_resource`,
//! `contains_resource`). A type that must also be reachable from C is
//! registered once from Rust with `World::register_ffi_resource`, under a
//! name, and C then refers to it by the [`ResourceId`] that name resolves
//! to: read through a span, write by copying bytes in, remove by id.
//!
//! Resources are a separate namespace from components. The same Rust
//! type can be both a resource and a component without any conflict, and
//! nothing here is tracked by the storage-claim guard.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use mid_collections::FfiSpan;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::hash::TypeIdMap;

/// Identifies a resource type registered for FFI access. Issued by
/// `World::register_ffi_resource`, densely from 0, and only meaningful
/// for the `World` that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(u32);

impl ResourceId {
    /// Rebuilds an id from the plain `u32` that crossed the FFI boundary.
    /// Any `u32` is safe to pass to the FFI-facing `World` methods: an id
    /// that was never issued is reported as [`ResourceFfiError::UnknownId`].
    pub const fn from_u32(value: u32) -> Self {
        Self(value)
    }

    /// The plain `u32` to hand across the FFI boundary.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Why an id-based (FFI-facing) resource operation did not happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceFfiError {
    /// The id was never issued by `register_ffi_resource` on this world.
    UnknownId,
    /// `write_resource_bytes` was given a slice whose length isn't
    /// exactly the registered type's size.
    SizeMismatch,
    /// `remove_ffi_resource` on a registered type with no value inserted.
    Absent,
}

type Values = TypeIdMap<Box<dyn Any>>;

struct FfiEntry {
    type_id: TypeId,
    size: usize,
    read: fn(&dyn Any) -> FfiSpan,
    write: fn(&mut Values, &[u8]) -> bool,
}

fn read_one<T: 'static + IntoBytes + Immutable>(value: &dyn Any) -> FfiSpan {
    let value = value
        .downcast_ref::<T>()
        .expect("resource value type must match its registered TypeId");
    FfiSpan::from_slice(std::slice::from_ref(value))
}

/// Builds a `T` from `bytes` and stores it: in place if a `T` is already
/// there (its address does not change, so a span handed out earlier keeps
/// pointing at the live value), inserted otherwise. `false` only when
/// `bytes.len()` isn't `size_of::<T>()`.
fn write_one<T: 'static + FromBytes>(values: &mut Values, bytes: &[u8]) -> bool {
    let Ok(new_value) = T::read_from_bytes(bytes) else {
        return false;
    };
    match values.get_mut(&TypeId::of::<T>()) {
        Some(existing) => {
            *existing
                .downcast_mut::<T>()
                .expect("resource value type must match its registered TypeId") = new_value;
        }
        None => {
            values.insert(TypeId::of::<T>(), Box::new(new_value));
        }
    }
    true
}

#[derive(Default)]
pub(crate) struct Resources {
    values: Values,
    /// Indexed by `ResourceId`.
    ffi: Vec<FfiEntry>,
    ffi_names: HashMap<&'static str, ResourceId>,
}

impl Resources {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Stores `value`, returning the previous value of that type if any.
    pub(crate) fn insert<T: 'static>(&mut self, value: T) -> Option<T> {
        self.values
            .insert(TypeId::of::<T>(), Box::new(value))
            .map(|old| {
                *old.downcast::<T>()
                    .expect("resource value type must match its TypeId")
            })
    }

    pub(crate) fn get<T: 'static>(&self) -> Option<&T> {
        self.values.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    pub(crate) fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.values.get_mut(&TypeId::of::<T>())?.downcast_mut::<T>()
    }

    pub(crate) fn remove<T: 'static>(&mut self) -> Option<T> {
        self.values.remove(&TypeId::of::<T>()).map(|old| {
            *old.downcast::<T>()
                .expect("resource value type must match its TypeId")
        })
    }

    pub(crate) fn contains<T: 'static>(&self) -> bool {
        self.values.contains_key(&TypeId::of::<T>())
    }

    /// Opts `T` into FFI access under `name`. Idempotent for the same
    /// `T`.
    ///
    /// # Panics
    /// If `name` was already registered for a different resource type,
    /// the same condition and reasoning as the component registrations.
    pub(crate) fn register_ffi<T>(&mut self, name: &'static str) -> ResourceId
    where
        T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
    {
        let type_id = TypeId::of::<T>();
        let id = match self.ffi.iter().position(|e| e.type_id == type_id) {
            Some(index) => ResourceId(index as u32),
            None => {
                self.ffi.push(FfiEntry {
                    type_id,
                    size: std::mem::size_of::<T>(),
                    read: read_one::<T>,
                    write: write_one::<T>,
                });
                ResourceId((self.ffi.len() - 1) as u32)
            }
        };
        match self.ffi_names.get(name) {
            Some(&existing) if existing != id => panic!(
                "Resources::register_ffi: name {name:?} is already registered for a different resource type"
            ),
            _ => {
                self.ffi_names.insert(name, id);
            }
        }
        id
    }

    pub(crate) fn lookup_ffi_id(&self, name: &str) -> Option<ResourceId> {
        self.ffi_names.get(name).copied()
    }

    /// The registered resource's current value as a one-element span, or
    /// the empty span if it is registered but not inserted. `None` for an
    /// id that was never issued.
    pub(crate) fn ffi_span(&self, id: ResourceId) -> Option<FfiSpan> {
        let entry = self.ffi.get(id.0 as usize)?;
        Some(match self.values.get(&entry.type_id) {
            Some(value) => (entry.read)(value.as_ref()),
            None => FfiSpan::empty(),
        })
    }

    /// Copies `bytes` in as the resource's new value (inserting it if
    /// absent). See [`write_one`].
    pub(crate) fn ffi_write(
        &mut self,
        id: ResourceId,
        bytes: &[u8],
    ) -> Result<(), ResourceFfiError> {
        let entry = self
            .ffi
            .get(id.0 as usize)
            .ok_or(ResourceFfiError::UnknownId)?;
        if bytes.len() != entry.size {
            return Err(ResourceFfiError::SizeMismatch);
        }
        if (entry.write)(&mut self.values, bytes) {
            Ok(())
        } else {
            Err(ResourceFfiError::SizeMismatch)
        }
    }

    /// Removes and drops the registered resource's value.
    pub(crate) fn ffi_remove(&mut self, id: ResourceId) -> Result<(), ResourceFfiError> {
        let entry = self
            .ffi
            .get(id.0 as usize)
            .ok_or(ResourceFfiError::UnknownId)?;
        match self.values.remove(&entry.type_id) {
            Some(_) => Ok(()),
            None => Err(ResourceFfiError::Absent),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
    #[repr(C)]
    struct Time {
        delta: f32,
        frame: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
    #[repr(C)]
    struct Gravity {
        g: f32,
    }

    /// Not FFI-registerable on purpose: has no zerocopy derives.
    #[derive(Debug, PartialEq)]
    struct Config(&'static str);

    fn time_bytes(delta: f32, frame: u32) -> Vec<u8> {
        Time { delta, frame }.as_bytes().to_vec()
    }

    #[test]
    fn insert_get_and_get_mut() {
        let mut r = Resources::new();
        assert_eq!(r.get::<Config>(), None);
        assert!(!r.contains::<Config>());

        assert_eq!(r.insert(Config("a")), None);
        assert!(r.contains::<Config>());
        assert_eq!(r.get::<Config>(), Some(&Config("a")));

        *r.get_mut::<Config>().unwrap() = Config("b");
        assert_eq!(r.get::<Config>(), Some(&Config("b")));
    }

    #[test]
    fn insert_replaces_and_returns_the_previous_value() {
        let mut r = Resources::new();
        r.insert(Gravity { g: 9.8 });
        assert_eq!(r.insert(Gravity { g: 1.6 }), Some(Gravity { g: 9.8 }));
        assert_eq!(r.get::<Gravity>(), Some(&Gravity { g: 1.6 }));
    }

    #[test]
    fn remove_returns_the_value_and_clears_it() {
        let mut r = Resources::new();
        r.insert(Config("x"));
        assert_eq!(r.remove::<Config>(), Some(Config("x")));
        assert_eq!(r.remove::<Config>(), None);
        assert!(!r.contains::<Config>());
    }

    #[test]
    fn distinct_types_do_not_collide() {
        let mut r = Resources::new();
        r.insert(Gravity { g: 1.0 });
        r.insert(Time {
            delta: 0.5,
            frame: 3,
        });
        r.insert(Config("c"));
        assert_eq!(r.get::<Gravity>(), Some(&Gravity { g: 1.0 }));
        assert_eq!(
            r.get::<Time>(),
            Some(&Time {
                delta: 0.5,
                frame: 3
            })
        );
        assert_eq!(r.get::<Config>(), Some(&Config("c")));
    }

    struct Tracked(Rc<Cell<u32>>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn values_are_dropped_exactly_once() {
        let drops = Rc::new(Cell::new(0));
        {
            let mut r = Resources::new();
            r.insert(Tracked(drops.clone()));
            assert_eq!(drops.get(), 0);
            // Replacing hands the old value back; dropping it is ours.
            let old = r.insert(Tracked(drops.clone()));
            assert_eq!(drops.get(), 0);
            drop(old);
            assert_eq!(drops.get(), 1);
            // Removing hands it back too.
            drop(r.remove::<Tracked>());
            assert_eq!(drops.get(), 2);
            r.insert(Tracked(drops.clone()));
        }
        assert_eq!(drops.get(), 3, "the last one drops with the store");
    }

    #[test]
    fn register_ffi_is_idempotent_and_ids_are_dense() {
        let mut r = Resources::new();
        let time = r.register_ffi::<Time>("Time");
        let gravity = r.register_ffi::<Gravity>("Gravity");
        assert_eq!((time.as_u32(), gravity.as_u32()), (0, 1));
        assert_eq!(r.register_ffi::<Time>("Time"), time);
        assert_eq!(r.lookup_ffi_id("Time"), Some(time));
        assert_eq!(r.lookup_ffi_id("Gravity"), Some(gravity));
        assert_eq!(r.lookup_ffi_id("Nope"), None);
    }

    #[test]
    #[should_panic(expected = "already registered for a different resource type")]
    fn register_ffi_panics_on_a_name_reused_for_another_type() {
        let mut r = Resources::new();
        r.register_ffi::<Time>("Same");
        r.register_ffi::<Gravity>("Same");
    }

    #[test]
    fn ffi_span_reflects_the_typed_value() {
        let mut r = Resources::new();
        let id = r.register_ffi::<Time>("Time");

        let empty = r.ffi_span(id).expect("registered");
        assert_eq!(
            empty.count, 0,
            "registered but not inserted is empty, not an error"
        );

        r.insert(Time {
            delta: 0.25,
            frame: 9,
        });
        let span = r.ffi_span(id).unwrap();
        assert_eq!((span.count, span.stride), (1, std::mem::size_of::<Time>()));
        // SAFETY: the span points at the live `Time` inside `r`, which
        // isn't touched again until after this read.
        let seen = unsafe { *(span.ptr as *const Time) };
        assert_eq!(
            seen,
            Time {
                delta: 0.25,
                frame: 9
            }
        );

        assert_eq!(r.ffi_span(ResourceId::from_u32(99)), None);
    }

    #[test]
    fn ffi_write_inserts_when_absent_and_is_visible_to_the_typed_api() {
        let mut r = Resources::new();
        let id = r.register_ffi::<Time>("Time");
        assert_eq!(r.get::<Time>(), None);

        r.ffi_write(id, &time_bytes(0.5, 2)).unwrap();
        assert_eq!(
            r.get::<Time>(),
            Some(&Time {
                delta: 0.5,
                frame: 2
            })
        );
    }

    #[test]
    fn ffi_write_updates_in_place_so_an_earlier_span_stays_valid() {
        let mut r = Resources::new();
        let id = r.register_ffi::<Time>("Time");
        r.insert(Time {
            delta: 1.0,
            frame: 1,
        });
        let before = r.ffi_span(id).unwrap();

        r.ffi_write(id, &time_bytes(2.0, 5)).unwrap();
        let after = r.ffi_span(id).unwrap();
        assert_eq!(before.ptr, after.ptr, "same address: written in place");
        // SAFETY: `before.ptr` is the live value's address (checked
        // equal to `after.ptr`), and nothing has moved or removed it.
        let seen = unsafe { *(before.ptr as *const Time) };
        assert_eq!(
            seen,
            Time {
                delta: 2.0,
                frame: 5
            }
        );
    }

    #[test]
    fn ffi_write_rejects_a_wrong_size_and_an_unknown_id_without_changing_anything() {
        let mut r = Resources::new();
        let id = r.register_ffi::<Time>("Time");
        r.insert(Time {
            delta: 1.0,
            frame: 1,
        });

        let too_short = &time_bytes(9.0, 9)[..7];
        assert_eq!(
            r.ffi_write(id, too_short),
            Err(ResourceFfiError::SizeMismatch)
        );
        let mut too_long = time_bytes(9.0, 9);
        too_long.push(0);
        assert_eq!(
            r.ffi_write(id, &too_long),
            Err(ResourceFfiError::SizeMismatch)
        );
        assert_eq!(
            r.ffi_write(ResourceId::from_u32(7), &[]),
            Err(ResourceFfiError::UnknownId)
        );
        assert_eq!(
            r.get::<Time>(),
            Some(&Time {
                delta: 1.0,
                frame: 1
            })
        );
    }

    #[test]
    fn ffi_remove_reports_unknown_and_absent_distinctly() {
        let mut r = Resources::new();
        let id = r.register_ffi::<Time>("Time");
        assert_eq!(r.ffi_remove(id), Err(ResourceFfiError::Absent));
        assert_eq!(
            r.ffi_remove(ResourceId::from_u32(5)),
            Err(ResourceFfiError::UnknownId)
        );

        r.insert(Time {
            delta: 1.0,
            frame: 1,
        });
        assert_eq!(r.ffi_remove(id), Ok(()));
        assert!(!r.contains::<Time>());
        assert_eq!(r.ffi_span(id).unwrap().count, 0);
    }
}
