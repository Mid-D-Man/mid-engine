// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "label.rs"
// ============================================================================
//! Names for schedules and for groups of systems.
//!
//! A [`ScheduleLabel`] names a schedule (`Update`, `FixedUpdate`). It is
//! implemented for every type that is `Debug + Clone + Eq + Hash +
//! 'static`, so a unit struct or an enum with the usual `#[derive]` is a
//! label with no further code.
//!
//! A [`SystemSet`] names a group of systems (`Phase::Input`), so ordering
//! and run conditions can be stated once for the whole group. It has the
//! same bounds but is implemented by hand with an empty `impl`:
//!
//! ```
//! use mid_ecs::schedule::SystemSet;
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! enum Phase { Input, Simulate }
//! impl SystemSet for Phase {}
//! ```
//!
//! The opt-in is deliberate. The configuration builder accepts tuples of
//! sets, and a tuple of two enums is itself `Debug + Clone + Eq + Hash`:
//! with a blanket impl `(Phase::Input, Phase::Simulate)` would be a set
//! *and* a tuple of sets, and the call would be ambiguous.

use std::any::{type_name, Any, TypeId};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};

use crate::system::IntoSystem;

// ── Type-erased values ───────────────────────────────────────────────

/// A value that can be cloned, compared and hashed without knowing its
/// type. Equality requires the same concrete type on both sides.
trait DynValue: Any + Debug {
    fn dyn_clone(&self) -> Box<dyn DynValue>;
    fn dyn_eq(&self, other: &dyn DynValue) -> bool;
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn as_any(&self) -> &dyn Any;
}

impl<T: Debug + Clone + Eq + Hash + 'static> DynValue for T {
    fn dyn_clone(&self) -> Box<dyn DynValue> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn DynValue) -> bool {
        other
            .as_any()
            .downcast_ref::<T>()
            .is_some_and(|o| self == o)
    }

    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        TypeId::of::<T>().hash(&mut state);
        self.hash(&mut state);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A boxed label or set value: cloneable, comparable, hashable, usable as
/// a map key.
struct Interned(Box<dyn DynValue>);

impl Interned {
    fn new<T: Debug + Clone + Eq + Hash + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
}

impl Clone for Interned {
    fn clone(&self) -> Self {
        Self(self.0.dyn_clone())
    }
}

impl PartialEq for Interned {
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(&*other.0)
    }
}

impl Eq for Interned {}

impl Hash for Interned {
    fn hash<H: Hasher>(&self, mut state: &mut H) {
        self.0.dyn_hash(&mut state);
    }
}

impl Debug for Interned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

// ── Schedule labels ──────────────────────────────────────────────────

/// A schedule's identity, as stored: the label's value with its type
/// erased. Two keys are equal when they hold equal values of the same
/// type.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LabelKey(Interned);

impl Debug for LabelKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

/// Names a schedule. Implemented for every `Debug + Clone + Eq + Hash +
/// 'static` type; a [`LabelKey`] is a label too, and stays itself rather
/// than being wrapped a second time.
pub trait ScheduleLabel: Debug + 'static {
    /// The type-erased key for this label.
    fn label_key(&self) -> LabelKey;
}

impl<T: Debug + Clone + Eq + Hash + 'static> ScheduleLabel for T {
    fn label_key(&self) -> LabelKey {
        match (self as &dyn Any).downcast_ref::<LabelKey>() {
            Some(key) => key.clone(),
            None => LabelKey(Interned::new(self.clone())),
        }
    }
}

// ── System sets ──────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Hash)]
enum SetKind {
    /// A user type implementing [`SystemSet`].
    User(Interned),
    /// "The system made from this function": every system of that type.
    SystemType(TypeId, &'static str),
    /// A set made by the builder for a group that has its own run
    /// condition. The number is unique within one schedule.
    Anonymous(u32),
}

/// A set's identity, as stored.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SetKey(SetKind);

impl SetKey {
    pub(crate) fn system_type<S: 'static>() -> Self {
        Self(SetKind::SystemType(TypeId::of::<S>(), type_name::<S>()))
    }

    pub(crate) fn anonymous(n: u32) -> Self {
        Self(SetKind::Anonymous(n))
    }

    /// `true` for a set of the systems made from one function.
    pub fn is_system_type(&self) -> bool {
        matches!(self.0, SetKind::SystemType(..))
    }

    /// `true` for a set the builder made for a group with a condition.
    pub fn is_anonymous(&self) -> bool {
        matches!(self.0, SetKind::Anonymous(_))
    }
}

impl Debug for SetKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            SetKind::User(value) => Debug::fmt(value, f),
            SetKind::SystemType(_, name) => write!(f, "SystemTypeSet({name})"),
            SetKind::Anonymous(n) => write!(f, "AnonymousSet({n})"),
        }
    }
}

/// Names a group of systems. Implement it with an empty `impl` for a
/// `Debug + Clone + Eq + Hash + 'static` type; see the module docs for why
/// this is not automatic.
pub trait SystemSet: Debug + Clone + Eq + Hash + 'static {
    /// The type-erased key for this set.
    fn set_key(&self) -> SetKey {
        SetKey(SetKind::User(Interned::new(self.clone())))
    }
}

/// Marker for [`IntoSystemSet`] on a [`SystemSet`].
pub struct IsUserSet;
/// Marker for [`IntoSystemSet`] on something that becomes a system.
pub struct IsSystemTypeSet;

/// Something that names a set: a [`SystemSet`] value, or a system (a
/// function of system parameters, say), which names "the system made from
/// it". This is what `before`, `after` and `ambiguous_with` take.
pub trait IntoSystemSet<Marker>: Sized {
    /// The set's key.
    fn into_set(self) -> SetKey;
}

impl<S: SystemSet> IntoSystemSet<IsUserSet> for S {
    fn into_set(self) -> SetKey {
        self.set_key()
    }
}

impl<Marker, S: IntoSystem<Marker>> IntoSystemSet<(IsSystemTypeSet, Marker)> for S {
    fn into_set(self) -> SetKey {
        SetKey::system_type::<S::System>()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::system_param::Res;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Update;
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Other;
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Number(u32);
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct AlsoNumber(u32);
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum Phase {
        Input,
        Simulate,
    }
    impl SystemSet for Phase {}
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Same(u32);
    impl SystemSet for Same {}

    fn hash_of<T: Hash>(value: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        value.hash(&mut h);
        h.finish()
    }

    #[test]
    fn equal_values_of_one_type_are_one_label() {
        assert_eq!(Number(1).label_key(), Number(1).label_key());
        assert_ne!(Number(1).label_key(), Number(2).label_key());
        assert_eq!(
            hash_of(&Number(7).label_key()),
            hash_of(&Number(7).label_key())
        );
    }

    #[test]
    fn the_same_value_in_two_types_is_two_labels() {
        assert_ne!(Number(1).label_key(), AlsoNumber(1).label_key());
        assert_ne!(Update.label_key(), Other.label_key());
    }

    #[test]
    fn a_label_works_as_a_map_key() {
        let mut keys = HashSet::new();
        keys.insert(Number(1).label_key());
        keys.insert(Number(1).label_key());
        keys.insert(AlsoNumber(1).label_key());
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&Number(1).label_key()));
        assert!(!keys.contains(&Number(2).label_key()));
    }

    #[test]
    fn a_key_is_a_label_that_stays_itself() {
        let key = Number(3).label_key();
        let again = key.label_key();
        assert_eq!(key, again);
        assert_eq!(hash_of(&key), hash_of(&again));
        // not wrapped: it still equals a key made from the value
        assert_eq!(again, Number(3).label_key());
        assert_eq!(format!("{again:?}"), "Number(3)");
    }

    #[test]
    fn a_cloned_key_equals_the_original() {
        let key = Phase::Input.set_key();
        assert_eq!(key.clone(), key);
    }

    #[test]
    fn debug_prints_the_value() {
        assert_eq!(format!("{:?}", Update.label_key()), "Update");
        assert_eq!(format!("{:?}", Phase::Simulate.set_key()), "Simulate");
        assert_eq!(format!("{:?}", Number(5).label_key()), "Number(5)");
    }

    #[test]
    fn sets_of_one_enum_are_told_apart() {
        assert_eq!(Phase::Input.set_key(), Phase::Input.set_key());
        assert_ne!(Phase::Input.set_key(), Phase::Simulate.set_key());
        assert_ne!(Same(1).set_key(), Phase::Input.set_key());
    }

    fn first() {}
    fn second() {}
    fn reads(_r: Res<Number>) {}

    #[test]
    fn a_function_names_the_set_of_its_system() {
        let a: SetKey = IntoSystemSet::into_set(first);
        let b: SetKey = IntoSystemSet::into_set(first);
        let c: SetKey = IntoSystemSet::into_set(second);
        assert_eq!(a, b, "one function, one set");
        assert_ne!(a, c, "two functions, two sets");
        assert!(a.is_system_type());
        assert!(!a.is_anonymous());
        let d: SetKey = IntoSystemSet::into_set(reads);
        assert_ne!(a, d);
    }

    #[test]
    fn a_user_set_is_not_a_system_type_set() {
        let k: SetKey = IntoSystemSet::into_set(Phase::Input);
        assert_eq!(k, Phase::Input.set_key());
        assert!(!k.is_system_type());
        assert!(!k.is_anonymous());
    }

    #[test]
    fn system_type_sets_print_the_function_type() {
        let a: SetKey = IntoSystemSet::into_set(first);
        let text = format!("{a:?}");
        assert!(text.starts_with("SystemTypeSet("), "{text}");
        assert!(text.contains("first"), "{text}");
    }

    #[test]
    fn anonymous_sets_are_told_apart_by_number() {
        assert_eq!(SetKey::anonymous(1), SetKey::anonymous(1));
        assert_ne!(SetKey::anonymous(1), SetKey::anonymous(2));
        assert!(SetKey::anonymous(1).is_anonymous());
        assert_ne!(SetKey::anonymous(1), Phase::Input.set_key());
        assert_eq!(format!("{:?}", SetKey::anonymous(4)), "AnonymousSet(4)");
    }

    #[test]
    fn a_tuple_of_values_is_a_label_not_a_set() {
        // Tuples are labels through the blanket impl; they are not sets,
        // which is what keeps `(Phase::A, Phase::B).chain()` unambiguous.
        let k = (Phase::Input, Phase::Simulate).label_key();
        assert_eq!(k, (Phase::Input, Phase::Simulate).label_key());
        assert_ne!(k, (Phase::Simulate, Phase::Input).label_key());
    }
}
