// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "config.rs"
// ============================================================================
//! The builder behind `add_systems` and `configure_sets`: a system, a set
//! or a tuple of them, with ordering and conditions attached.
//!
//! ```
//! use mid_ecs::schedule::{IntoScheduleConfigs, Schedule, SystemSet};
//! use mid_ecs::World;
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! enum Phase { Input, Simulate }
//! impl SystemSet for Phase {}
//!
//! fn read_input() {}
//! fn move_units() {}
//! fn apply_damage() {}
//!
//! let mut schedule = Schedule::new("update");
//! schedule
//!     .configure_sets((Phase::Input, Phase::Simulate).chain())
//!     .add_systems((
//!         read_input.in_set(Phase::Input),
//!         (move_units, apply_damage).chain().in_set(Phase::Simulate),
//!     ));
//! let mut world = World::new();
//! schedule.run(&mut world);
//! ```
//!
//! A setting on a tuple applies to everything in it, to each member
//! separately, with two exceptions that make sense only for the whole
//! group: `chain` orders the members one after another, and `run_if` puts
//! the members in one anonymous set that carries the condition, so it is
//! evaluated once for the group rather than once per member.
//!
//! Systems and sets are different kinds, and each method takes only its
//! own: a set is not a system, a system is not a set, and only a type that
//! implements [`SystemSet`] can be named in `in_set`.
//!
//! ```compile_fail,E0277
//! use mid_ecs::schedule::{Schedule, SystemSet};
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! struct Phase;
//! impl SystemSet for Phase {}
//! Schedule::new("x").add_systems(Phase);
//! ```
//!
//! ```compile_fail,E0277
//! use mid_ecs::schedule::Schedule;
//! fn system() {}
//! Schedule::new("x").configure_sets(system);
//! ```
//!
//! ```compile_fail,E0277
//! use mid_ecs::schedule::{IntoScheduleConfigs, Schedule};
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! struct NotASet;
//! fn system() {}
//! Schedule::new("x").add_systems(system.in_set(NotASet));
//! ```

use crate::schedule::condition::{BoxedCondition, IntoCondition};
use crate::schedule::label::{IntoSystemSet, IsSystemTypeSet, IsUserSet, SetKey, SystemSet};
use crate::system::{IntoSystem, System};

// ── What a leaf is ───────────────────────────────────────────────────

/// A system on its way into a schedule, with the set that names "systems
/// made from this function".
pub struct ScheduleSystem {
    pub(crate) system: Box<dyn System>,
    pub(crate) type_set: SetKey,
}

// ── The builder ──────────────────────────────────────────────────────

#[derive(Default)]
struct ConfigMeta {
    in_sets: Vec<SetKey>,
    before: Vec<SetKey>,
    after: Vec<SetKey>,
    ambiguous_with: Vec<SetKey>,
    conditions: Vec<BoxedCondition>,
}

enum ConfigNode<T> {
    Leaf(T),
    Group {
        children: Vec<ScheduleConfigs<T>>,
        chained: bool,
    },
}

/// A system, a set, or a group of them, with the settings given so far.
/// `T` is [`ScheduleSystem`] for systems and [`SetKey`] for sets, which is
/// what keeps `add_systems` from taking a set and `configure_sets` from
/// taking a system.
pub struct ScheduleConfigs<T> {
    node: ConfigNode<T>,
    meta: ConfigMeta,
}

impl<T> ScheduleConfigs<T> {
    pub(crate) fn leaf(item: T) -> Self {
        Self {
            node: ConfigNode::Leaf(item),
            meta: ConfigMeta::default(),
        }
    }

    pub(crate) fn group(children: Vec<Self>) -> Self {
        Self {
            node: ConfigNode::Group {
                children,
                chained: false,
            },
            meta: ConfigMeta::default(),
        }
    }

    /// Puts everything here in `set`.
    pub fn in_set(mut self, set: impl SystemSet) -> Self {
        self.meta.in_sets.push(set.set_key());
        self
    }

    /// Runs everything here before `set` (or before the system, if a
    /// system is named).
    pub fn before<M>(mut self, set: impl IntoSystemSet<M>) -> Self {
        self.meta.before.push(set.into_set());
        self
    }

    /// Runs everything here after `set`.
    pub fn after<M>(mut self, set: impl IntoSystemSet<M>) -> Self {
        self.meta.after.push(set.into_set());
        self
    }

    /// Says that a conflict between anything here and anything in `set`
    /// is intended, so the ambiguity report leaves it out.
    pub fn ambiguous_with<M>(mut self, set: impl IntoSystemSet<M>) -> Self {
        self.meta.ambiguous_with.push(set.into_set());
        self
    }

    /// Runs everything here only when `condition` is `true`. On a single
    /// system the condition is the system's own; on a group it is one
    /// condition for the whole group, evaluated once per run.
    pub fn run_if<M>(mut self, condition: impl IntoCondition<M>) -> Self {
        self.meta
            .conditions
            .push(Box::new(condition.into_condition()));
        self
    }

    /// Orders the members of a group one after another (each member after
    /// the one before it, and every part of a nested group before every
    /// part of the next). Has no effect on a single system.
    pub fn chain(mut self) -> Self {
        if let ConfigNode::Group { chained, .. } = &mut self.node {
            *chained = true;
        }
        self
    }
}

// ── Into ─────────────────────────────────────────────────────────────

/// Marker for [`IntoScheduleConfigs`] on a [`ScheduleConfigs`].
pub struct IsConfigs;
/// Marker for [`IntoScheduleConfigs`] on a tuple.
pub struct IsTuple;

/// Something that becomes [`ScheduleConfigs`]: a system, a set, a tuple of
/// them, or configs already. The settings are the same as on
/// [`ScheduleConfigs`], available on anything that converts.
///
/// `Marker` only keeps the impls apart; it is inferred.
pub trait IntoScheduleConfigs<T, Marker>: Sized {
    /// Converts.
    fn into_configs(self) -> ScheduleConfigs<T>;

    /// See [`ScheduleConfigs::in_set`].
    fn in_set(self, set: impl SystemSet) -> ScheduleConfigs<T> {
        self.into_configs().in_set(set)
    }

    /// See [`ScheduleConfigs::before`].
    fn before<M>(self, set: impl IntoSystemSet<M>) -> ScheduleConfigs<T> {
        self.into_configs().before(set)
    }

    /// See [`ScheduleConfigs::after`].
    fn after<M>(self, set: impl IntoSystemSet<M>) -> ScheduleConfigs<T> {
        self.into_configs().after(set)
    }

    /// See [`ScheduleConfigs::ambiguous_with`].
    fn ambiguous_with<M>(self, set: impl IntoSystemSet<M>) -> ScheduleConfigs<T> {
        self.into_configs().ambiguous_with(set)
    }

    /// See [`ScheduleConfigs::run_if`].
    fn run_if<M>(self, condition: impl IntoCondition<M>) -> ScheduleConfigs<T> {
        self.into_configs().run_if(condition)
    }

    /// See [`ScheduleConfigs::chain`].
    fn chain(self) -> ScheduleConfigs<T> {
        self.into_configs().chain()
    }
}

impl<T> IntoScheduleConfigs<T, IsConfigs> for ScheduleConfigs<T> {
    fn into_configs(self) -> ScheduleConfigs<T> {
        self
    }
}

impl<S: SystemSet> IntoScheduleConfigs<SetKey, IsUserSet> for S {
    fn into_configs(self) -> ScheduleConfigs<SetKey> {
        ScheduleConfigs::leaf(self.set_key())
    }
}

impl<Marker, S: IntoSystem<Marker>> IntoScheduleConfigs<ScheduleSystem, (IsSystemTypeSet, Marker)>
    for S
{
    fn into_configs(self) -> ScheduleConfigs<ScheduleSystem> {
        let type_set = SetKey::system_type::<S::System>();
        ScheduleConfigs::leaf(ScheduleSystem {
            system: Box::new(self.into_system()),
            type_set,
        })
    }
}

macro_rules! impl_tuple_configs {
    ($(($A:ident, $M:ident, $v:ident)),+) => {
        impl<P, $($A: IntoScheduleConfigs<P, $M>, $M),+>
            IntoScheduleConfigs<P, (IsTuple, ($($M,)+))> for ($($A,)+)
        {
            fn into_configs(self) -> ScheduleConfigs<P> {
                let ($($v,)+) = self;
                ScheduleConfigs::group(vec![$($v.into_configs()),+])
            }
        }
    };
}

impl_tuple_configs!((A0, M0, a0));
impl_tuple_configs!((A0, M0, a0), (A1, M1, a1));
impl_tuple_configs!((A0, M0, a0), (A1, M1, a1), (A2, M2, a2));
impl_tuple_configs!((A0, M0, a0), (A1, M1, a1), (A2, M2, a2), (A3, M3, a3));
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5),
    (A6, M6, a6)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5),
    (A6, M6, a6),
    (A7, M7, a7)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5),
    (A6, M6, a6),
    (A7, M7, a7),
    (A8, M8, a8)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5),
    (A6, M6, a6),
    (A7, M7, a7),
    (A8, M8, a8),
    (A9, M9, a9)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5),
    (A6, M6, a6),
    (A7, M7, a7),
    (A8, M8, a8),
    (A9, M9, a9),
    (A10, M10, a10)
);
impl_tuple_configs!(
    (A0, M0, a0),
    (A1, M1, a1),
    (A2, M2, a2),
    (A3, M3, a3),
    (A4, M4, a4),
    (A5, M5, a5),
    (A6, M6, a6),
    (A7, M7, a7),
    (A8, M8, a8),
    (A9, M9, a9),
    (A10, M10, a10),
    (A11, M11, a11)
);

// ── Flattening ───────────────────────────────────────────────────────

/// Settings a group passes down to each member.
#[derive(Clone, Default)]
struct Inherited {
    in_sets: Vec<SetKey>,
    before: Vec<SetKey>,
    after: Vec<SetKey>,
    ambiguous_with: Vec<SetKey>,
}

/// One leaf with everything its enclosing groups said about it.
pub(crate) struct FlatLeaf<T> {
    pub(crate) item: T,
    pub(crate) in_sets: Vec<SetKey>,
    pub(crate) before: Vec<SetKey>,
    pub(crate) after: Vec<SetKey>,
    pub(crate) ambiguous_with: Vec<SetKey>,
    pub(crate) conditions: Vec<BoxedCondition>,
}

/// A set made for a group that has a run condition. Every member of the
/// group lists its key in `in_sets`.
pub(crate) struct FlatAnonSet {
    pub(crate) key: SetKey,
    pub(crate) conditions: Vec<BoxedCondition>,
}

/// A configuration tree reduced to a list of leaves, the orderings a
/// `chain` implies (as lists of leaf indices: every one of the first list
/// before every one of the second), and the anonymous sets it needed.
pub(crate) struct Flattened<T> {
    pub(crate) leaves: Vec<FlatLeaf<T>>,
    pub(crate) chains: Vec<(Vec<usize>, Vec<usize>)>,
    pub(crate) anon_sets: Vec<FlatAnonSet>,
}

impl<T> ScheduleConfigs<T> {
    /// Reduces the tree. `anon_counter` numbers the anonymous sets and is
    /// the schedule's, so two calls never reuse a number.
    pub(crate) fn flatten(self, anon_counter: &mut u32) -> Flattened<T> {
        let mut out = Flattened {
            leaves: Vec::new(),
            chains: Vec::new(),
            anon_sets: Vec::new(),
        };
        self.flatten_into(&Inherited::default(), anon_counter, &mut out);
        out
    }

    fn flatten_into(
        self,
        inherited: &Inherited,
        anon_counter: &mut u32,
        out: &mut Flattened<T>,
    ) -> Vec<usize> {
        let ScheduleConfigs { node, meta } = self;
        let mut inherit = inherited.clone();
        inherit.in_sets.extend(meta.in_sets);
        inherit.before.extend(meta.before);
        inherit.after.extend(meta.after);
        inherit.ambiguous_with.extend(meta.ambiguous_with);

        match node {
            ConfigNode::Leaf(item) => {
                out.leaves.push(FlatLeaf {
                    item,
                    in_sets: inherit.in_sets,
                    before: inherit.before,
                    after: inherit.after,
                    ambiguous_with: inherit.ambiguous_with,
                    conditions: meta.conditions,
                });
                vec![out.leaves.len() - 1]
            }
            ConfigNode::Group { children, chained } => {
                if !meta.conditions.is_empty() {
                    let key = SetKey::anonymous(*anon_counter);
                    *anon_counter += 1;
                    inherit.in_sets.push(key.clone());
                    out.anon_sets.push(FlatAnonSet {
                        key,
                        conditions: meta.conditions,
                    });
                }
                let mut all = Vec::new();
                let mut previous: Option<Vec<usize>> = None;
                for child in children {
                    let indices = child.flatten_into(&inherit, anon_counter, out);
                    if indices.is_empty() {
                        continue;
                    }
                    if chained {
                        if let Some(before) = previous.take() {
                            out.chains.push((before, indices.clone()));
                        }
                        previous = Some(indices.clone());
                    }
                    all.extend(indices);
                }
                all
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::condition::common_conditions::resource_exists;
    use crate::schedule::label::SystemSet;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum Phase {
        A,
        B,
        C,
    }
    impl SystemSet for Phase {}

    fn leaf(name: &'static str) -> ScheduleConfigs<&'static str> {
        ScheduleConfigs::leaf(name)
    }

    fn items(f: &Flattened<&'static str>) -> Vec<&'static str> {
        f.leaves.iter().map(|l| l.item).collect()
    }

    fn flatten(c: ScheduleConfigs<&'static str>) -> Flattened<&'static str> {
        c.flatten(&mut 0)
    }

    #[test]
    fn a_leaf_flattens_to_itself() {
        let f = flatten(leaf("a"));
        assert_eq!(items(&f), vec!["a"]);
        assert!(f.chains.is_empty() && f.anon_sets.is_empty());
        assert!(f.leaves[0].in_sets.is_empty());
        assert!(f.leaves[0].conditions.is_empty());
    }

    #[test]
    fn a_group_keeps_member_order() {
        let f = flatten(ScheduleConfigs::group(vec![
            leaf("a"),
            leaf("b"),
            leaf("c"),
        ]));
        assert_eq!(items(&f), vec!["a", "b", "c"]);
        assert!(f.chains.is_empty(), "no chain unless asked for");
    }

    #[test]
    fn settings_on_a_group_reach_each_member() {
        let f = flatten(
            ScheduleConfigs::group(vec![leaf("a"), leaf("b")])
                .in_set(Phase::A)
                .before(Phase::B)
                .after(Phase::C)
                .ambiguous_with(Phase::B),
        );
        for l in &f.leaves {
            assert_eq!(l.in_sets, vec![Phase::A.set_key()]);
            assert_eq!(l.before, vec![Phase::B.set_key()]);
            assert_eq!(l.after, vec![Phase::C.set_key()]);
            assert_eq!(l.ambiguous_with, vec![Phase::B.set_key()]);
        }
    }

    #[test]
    fn settings_on_a_leaf_stay_on_that_leaf() {
        let f = flatten(ScheduleConfigs::group(vec![
            leaf("a").in_set(Phase::A),
            leaf("b"),
        ]));
        assert_eq!(f.leaves[0].in_sets, vec![Phase::A.set_key()]);
        assert!(f.leaves[1].in_sets.is_empty());
    }

    #[test]
    fn nested_groups_accumulate_settings_outside_in() {
        let inner = ScheduleConfigs::group(vec![leaf("a"), leaf("b")]).in_set(Phase::B);
        let f = flatten(ScheduleConfigs::group(vec![inner, leaf("c")]).in_set(Phase::A));
        assert_eq!(
            f.leaves[0].in_sets,
            vec![Phase::A.set_key(), Phase::B.set_key()]
        );
        assert_eq!(f.leaves[1].in_sets, f.leaves[0].in_sets);
        assert_eq!(f.leaves[2].in_sets, vec![Phase::A.set_key()]);
    }

    #[test]
    fn chain_orders_consecutive_members() {
        let f = flatten(ScheduleConfigs::group(vec![leaf("a"), leaf("b"), leaf("c")]).chain());
        assert_eq!(f.chains, vec![(vec![0], vec![1]), (vec![1], vec![2])]);
    }

    #[test]
    fn chain_on_a_leaf_does_nothing() {
        let f = flatten(leaf("a").chain());
        assert!(f.chains.is_empty());
        assert_eq!(items(&f), vec!["a"]);
    }

    #[test]
    fn chain_puts_every_part_of_a_group_before_every_part_of_the_next() {
        let pair = ScheduleConfigs::group(vec![leaf("b"), leaf("c")]);
        let f = flatten(ScheduleConfigs::group(vec![leaf("a"), pair, leaf("d")]).chain());
        assert_eq!(f.chains, vec![(vec![0], vec![1, 2]), (vec![1, 2], vec![3])]);
    }

    #[test]
    fn a_chained_inner_group_orders_inside_only() {
        let inner = ScheduleConfigs::group(vec![leaf("b"), leaf("c")]).chain();
        let f = flatten(ScheduleConfigs::group(vec![leaf("a"), inner]));
        assert_eq!(f.chains, vec![(vec![1], vec![2])]);
    }

    #[test]
    fn an_empty_member_does_not_break_a_chain() {
        let empty: ScheduleConfigs<&'static str> = ScheduleConfigs::group(vec![]);
        let f = flatten(ScheduleConfigs::group(vec![leaf("a"), empty, leaf("b")]).chain());
        assert_eq!(f.chains, vec![(vec![0], vec![1])]);
    }

    #[test]
    fn a_condition_on_a_leaf_is_the_leafs_own() {
        let f = flatten(leaf("a").run_if(resource_exists::<u32>));
        assert_eq!(f.leaves[0].conditions.len(), 1);
        assert!(f.anon_sets.is_empty());
    }

    #[test]
    fn several_conditions_on_a_leaf_all_stay() {
        let f = flatten(
            leaf("a")
                .run_if(resource_exists::<u32>)
                .run_if(resource_exists::<u64>),
        );
        assert_eq!(f.leaves[0].conditions.len(), 2);
    }

    #[test]
    fn a_condition_on_a_group_becomes_one_anonymous_set() {
        let mut counter = 5;
        let f = ScheduleConfigs::group(vec![leaf("a"), leaf("b")])
            .run_if(resource_exists::<u32>)
            .flatten(&mut counter);
        assert_eq!(counter, 6);
        assert_eq!(f.anon_sets.len(), 1);
        assert_eq!(f.anon_sets[0].key, SetKey::anonymous(5));
        assert_eq!(f.anon_sets[0].conditions.len(), 1);
        for l in &f.leaves {
            assert!(l.conditions.is_empty(), "the members carry none");
            assert_eq!(l.in_sets, vec![SetKey::anonymous(5)]);
        }
    }

    #[test]
    fn nested_conditioned_groups_make_separate_sets_and_members_join_both() {
        let mut counter = 0;
        let inner = ScheduleConfigs::group(vec![leaf("a")]).run_if(resource_exists::<u64>);
        let f = ScheduleConfigs::group(vec![inner, leaf("b")])
            .run_if(resource_exists::<u32>)
            .flatten(&mut counter);
        assert_eq!(counter, 2);
        assert_eq!(f.anon_sets.len(), 2);
        // outer is numbered first, inner second
        assert_eq!(f.anon_sets[0].key, SetKey::anonymous(0));
        assert_eq!(f.anon_sets[1].key, SetKey::anonymous(1));
        assert_eq!(
            f.leaves[0].in_sets,
            vec![SetKey::anonymous(0), SetKey::anonymous(1)]
        );
        assert_eq!(f.leaves[1].in_sets, vec![SetKey::anonymous(0)]);
    }

    #[test]
    fn the_counter_continues_across_calls() {
        let mut counter = 0;
        let a = ScheduleConfigs::group(vec![leaf("a")])
            .run_if(resource_exists::<u32>)
            .flatten(&mut counter);
        let b = ScheduleConfigs::group(vec![leaf("b")])
            .run_if(resource_exists::<u32>)
            .flatten(&mut counter);
        assert_ne!(a.anon_sets[0].key, b.anon_sets[0].key);
    }

    #[test]
    fn a_condition_on_a_group_with_a_chain_keeps_both() {
        let f = ScheduleConfigs::group(vec![leaf("a"), leaf("b")])
            .chain()
            .run_if(resource_exists::<u32>)
            .flatten(&mut 0);
        assert_eq!(f.chains.len(), 1);
        assert_eq!(f.anon_sets.len(), 1);
    }

    // ── real conversions ─────────────────────────────────────────────

    fn sys_a() {}
    fn sys_b() {}

    #[test]
    fn systems_sets_and_tuples_convert() {
        let one: ScheduleConfigs<ScheduleSystem> = sys_a.into_configs();
        assert_eq!(one.flatten(&mut 0).leaves.len(), 1);

        let two = (sys_a, sys_b).into_configs();
        let f = two.flatten(&mut 0);
        assert_eq!(f.leaves.len(), 2);
        assert_ne!(f.leaves[0].item.type_set, f.leaves[1].item.type_set);

        let sets: ScheduleConfigs<SetKey> = (Phase::A, Phase::B, Phase::C).chain();
        let f = sets.flatten(&mut 0);
        assert_eq!(f.leaves.len(), 3);
        assert_eq!(f.chains.len(), 2);
        assert_eq!(f.leaves[1].item, Phase::B.set_key());
    }

    #[test]
    fn a_function_added_twice_shares_one_type_set() {
        let f = (sys_a, sys_a).into_configs().flatten(&mut 0);
        assert_eq!(f.leaves[0].item.type_set, f.leaves[1].item.type_set);
    }

    #[test]
    fn settings_chain_on_anything_that_converts() {
        let c = sys_a
            .in_set(Phase::A)
            .before(Phase::B)
            .after(sys_b)
            .ambiguous_with(Phase::C)
            .run_if(resource_exists::<u32>);
        let f = c.flatten(&mut 0);
        let leaf = &f.leaves[0];
        assert_eq!(leaf.in_sets, vec![Phase::A.set_key()]);
        assert_eq!(leaf.before, vec![Phase::B.set_key()]);
        assert_eq!(leaf.after.len(), 1);
        assert!(leaf.after[0].is_system_type());
        assert_eq!(leaf.ambiguous_with, vec![Phase::C.set_key()]);
        assert_eq!(leaf.conditions.len(), 1);
    }

    #[test]
    fn tuples_of_twelve_convert() {
        fn s() {}
        let c = (s, s, s, s, s, s, s, s, s, s, s, s).chain();
        let f = c.flatten(&mut 0);
        assert_eq!(f.leaves.len(), 12);
        assert_eq!(f.chains.len(), 11);
    }

    #[test]
    fn tuples_can_nest() {
        let c = (sys_a, (sys_b, sys_a).chain()).chain();
        let f = c.flatten(&mut 0);
        assert_eq!(f.leaves.len(), 3);
        assert_eq!(f.chains.len(), 2);
    }
}
