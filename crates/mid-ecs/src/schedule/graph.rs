// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "graph.rs"
// ============================================================================
//! Turning "this set contains these systems" and "this runs before that"
//! into one run order, or a list of what is wrong.
//!
//! The input knows nothing about systems beyond how many there are, what
//! they are called, which sets contain what, and which orderings were
//! asked for, so the whole thing is checked against brute force in the
//! tests without building a single system.
//!
//! Two graphs, as in `bevy_ecs`:
//!
//! - the *hierarchy*: a set contains systems and other sets. It must be
//!   acyclic. A set's *systems* are everything reachable below it.
//! - the *dependencies*: `a` before `b`, where `a` and `b` are systems or
//!   sets. A set stands for all of its systems, so an ordering between two
//!   sets becomes an ordering between every system of one and every system
//!   of the other.
//!
//! The run order is the topological order that always takes the
//! lowest-numbered ready system, so it is the same on every build.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

// ── Input ────────────────────────────────────────────────────────────

/// A system or a set, by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum NodeId {
    System(usize),
    Set(usize),
}

/// Everything the build needs to know.
pub(crate) struct GraphInput<'a> {
    pub(crate) system_names: &'a [String],
    pub(crate) set_names: &'a [String],
    /// `(set, member)`: `member` is directly in `set`.
    pub(crate) members: &'a [(usize, NodeId)],
    /// `(first, second)`: `first` runs before `second`.
    pub(crate) deps: &'a [(NodeId, NodeId)],
}

impl GraphInput<'_> {
    fn name(&self, node: NodeId) -> String {
        match node {
            NodeId::System(i) => self.system_names[i].clone(),
            NodeId::Set(i) => self.set_names[i].clone(),
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────

/// Systems that cannot be ordered because each must run before another in
/// the group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyCycle {
    /// The systems in the cycle, in the order they were added.
    pub systems: Vec<String>,
    /// The orderings, as asked for, that make it up: `(first, second)`.
    /// Either may be a set.
    pub orderings: Vec<(String, String)>,
}

/// An ordering that can never hold because both sides contain the same
/// system: a set ordered against one of its own members, or against
/// itself, or two sets that overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlappingOrder {
    /// The side that was to run first.
    pub before: String,
    /// The side that was to run second.
    pub after: String,
    /// The systems on both sides.
    pub shared: Vec<String>,
}

/// Two systems that touch the same data with nothing ordering them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ambiguity {
    /// The earlier-added system.
    pub first: String,
    /// The later-added system.
    pub second: String,
    /// The component and resource types they disagree about. Empty if
    /// only [`Self::whole_world`].
    pub conflicts: Vec<&'static str>,
    /// `true` if one of them takes everything (an exclusive system).
    pub whole_world: bool,
}

/// Why a schedule could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleBuildError {
    /// Sets that contain themselves, directly or through each other.
    /// Each entry is one cycle's set names.
    HierarchyCycle(Vec<Vec<String>>),
    /// Orderings that can never hold. See [`OverlappingOrder`].
    OverlappingOrder(Vec<OverlappingOrder>),
    /// Every group of systems that must each run before another.
    DependencyCycle(Vec<DependencyCycle>),
    /// Unordered systems that conflict, when the schedule is set to treat
    /// that as an error.
    Ambiguities(Vec<Ambiguity>),
}

impl fmt::Display for Ambiguity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` and `{}` conflict on ", self.first, self.second)?;
        if self.whole_world {
            f.write_str("the whole world")?;
            if !self.conflicts.is_empty() {
                f.write_str(", ")?;
            }
        }
        for (i, name) in self.conflicts.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "`{name}`")?;
        }
        f.write_str(" and nothing orders them")
    }
}

impl fmt::Display for ScheduleBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HierarchyCycle(cycles) => {
                f.write_str("system sets that contain themselves:")?;
                for cycle in cycles {
                    write!(f, "\n  {}", cycle.join(" -> "))?;
                }
                Ok(())
            }
            Self::OverlappingOrder(list) => {
                f.write_str("orderings that can never hold:")?;
                for o in list {
                    write!(
                        f,
                        "\n  `{}` before `{}`, but both contain {}",
                        o.before,
                        o.after,
                        o.shared
                            .iter()
                            .map(|s| format!("`{s}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                }
                Ok(())
            }
            Self::DependencyCycle(cycles) => {
                f.write_str("systems that cannot be ordered (each must run before another):")?;
                for c in cycles {
                    write!(f, "\n  cycle of {}", c.systems.join(", "))?;
                    for (first, second) in &c.orderings {
                        write!(f, "\n    `{first}` before `{second}`")?;
                    }
                }
                Ok(())
            }
            Self::Ambiguities(list) => {
                f.write_str("ambiguous systems:")?;
                for a in list {
                    write!(f, "\n  {a}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ScheduleBuildError {}

// ── Output ───────────────────────────────────────────────────────────

/// A square bit matrix: `contains(a, b)` is true when `a` must run before
/// `b`, directly or through others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reach {
    words: usize,
    bits: Vec<u64>,
}

impl Reach {
    fn new(n: usize) -> Self {
        let words = n.div_ceil(64);
        Self {
            words,
            bits: vec![0; n * words],
        }
    }

    pub(crate) fn contains(&self, from: usize, to: usize) -> bool {
        self.bits[from * self.words + to / 64] & (1 << (to % 64)) != 0
    }

    fn insert(&mut self, from: usize, to: usize) {
        self.bits[from * self.words + to / 64] |= 1 << (to % 64);
    }

    fn union_row(&mut self, into: usize, from: usize) {
        for k in 0..self.words {
            let word = self.bits[from * self.words + k];
            self.bits[into * self.words + k] |= word;
        }
    }
}

/// The result of a successful build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Built {
    /// Every system, once, in the order to run them.
    pub(crate) order: Vec<usize>,
    /// For each set, its systems (everything below it), sorted.
    pub(crate) systems_of_set: Vec<Vec<usize>>,
    /// For each system, every set above it, sorted.
    pub(crate) sets_of_system: Vec<Vec<usize>>,
    /// The orderings between systems that the dependencies amount to,
    /// without repeats, sorted.
    pub(crate) edges: Vec<(usize, usize)>,
}

impl Built {
    /// Which system must run before which, transitively. Worked out on
    /// request rather than during the build: it takes a bit per pair of
    /// systems, and only the ambiguity check wants it.
    pub(crate) fn reachability(&self) -> Reach {
        let n = self.order.len();
        let mut succ = vec![Vec::new(); n];
        for &(x, y) in &self.edges {
            succ[x].push(y);
        }
        let mut reach = Reach::new(n);
        // later systems first, so everything a successor reaches is known
        for &i in self.order.iter().rev() {
            for &j in &succ[i] {
                reach.insert(i, j);
                reach.union_row(i, j);
            }
        }
        reach
    }
}

/// The systems a node stands for: itself, or everything below a set.
pub(crate) fn systems_of<'a>(systems_of_set: &'a [Vec<usize>], node: &'a NodeId) -> &'a [usize] {
    match node {
        NodeId::System(i) => std::slice::from_ref(i),
        NodeId::Set(s) => &systems_of_set[*s],
    }
}

fn intersect(a: &[usize], b: &[usize]) -> Vec<usize> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

// ── Strongly connected components ────────────────────────────────────

/// The groups of nodes that can all reach each other and so form a cycle:
/// components of more than one node, and single nodes with an edge to
/// themselves. Each component is sorted; the list is in discovery order.
pub(crate) fn cyclic_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut next_index = 0;
    let mut out = Vec::new();
    // Tarjan's algorithm with an explicit call stack, so a long chain of
    // systems cannot overflow the real one.
    let mut call: Vec<(usize, usize)> = Vec::new();
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        index[root] = next_index;
        low[root] = next_index;
        next_index += 1;
        stack.push(root);
        on_stack[root] = true;
        call.push((root, 0));
        while let Some(&(v, edge)) = call.last() {
            if edge < adj[v].len() {
                call.last_mut().expect("just seen").1 += 1;
                let w = adj[v][edge];
                if index[w] == usize::MAX {
                    index[w] = next_index;
                    low[w] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut component = Vec::new();
                    loop {
                        let w = stack.pop().expect("v is on the stack");
                        on_stack[w] = false;
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    if component.len() > 1 || adj[v].contains(&v) {
                        component.sort_unstable();
                        out.push(component);
                    }
                }
            }
        }
    }
    out
}

// ── Hierarchy ────────────────────────────────────────────────────────

/// Each set's systems, or the cycles in the hierarchy.
fn hierarchy(input: &GraphInput) -> Result<Vec<Vec<usize>>, ScheduleBuildError> {
    let n_sets = input.set_names.len();
    let mut children = vec![Vec::new(); n_sets];
    let mut parents = vec![Vec::new(); n_sets];
    let mut systems = vec![Vec::new(); n_sets];
    for &(set, member) in input.members {
        match member {
            NodeId::System(s) => systems[set].push(s),
            NodeId::Set(c) => {
                children[set].push(c);
                parents[c].push(set);
            }
        }
    }
    for list in children.iter_mut().chain(parents.iter_mut()) {
        list.sort_unstable();
        list.dedup();
    }

    let cycles = cyclic_components(&children);
    if !cycles.is_empty() {
        return Err(ScheduleBuildError::HierarchyCycle(
            cycles
                .into_iter()
                .map(|c| c.into_iter().map(|s| input.set_names[s].clone()).collect())
                .collect(),
        ));
    }

    // Children before parents: a set is finished when all its child sets are.
    let mut waiting: Vec<usize> = children.iter().map(Vec::len).collect();
    let mut ready: Vec<usize> = (0..n_sets).filter(|&s| waiting[s] == 0).collect();
    while let Some(set) = ready.pop() {
        systems[set].sort_unstable();
        systems[set].dedup();
        for &parent in &parents[set] {
            let below = systems[set].clone();
            systems[parent].extend(below);
            waiting[parent] -= 1;
            if waiting[parent] == 0 {
                ready.push(parent);
            }
        }
    }
    Ok(systems)
}

// ── Build ────────────────────────────────────────────────────────────

/// Builds the run order, or says everything that is wrong, checking in
/// this order and stopping at the first that fails: the hierarchy,
/// orderings that overlap, then cycles.
pub(crate) fn build(input: &GraphInput) -> Result<Built, ScheduleBuildError> {
    let n = input.system_names.len();
    let systems_of_set = hierarchy(input)?;

    // Expand each ordering between nodes to orderings between systems.
    let mut overlaps = Vec::new();
    let mut raw: Vec<(usize, usize, usize)> = Vec::new();
    for (declared, (first, second)) in input.deps.iter().enumerate() {
        let a = systems_of(&systems_of_set, first);
        let b = systems_of(&systems_of_set, second);
        let shared = intersect(a, b);
        if !shared.is_empty() {
            overlaps.push(OverlappingOrder {
                before: input.name(*first),
                after: input.name(*second),
                shared: shared
                    .into_iter()
                    .map(|s| input.system_names[s].clone())
                    .collect(),
            });
            continue;
        }
        for &x in a {
            for &y in b {
                raw.push((x, y, declared));
            }
        }
    }
    if !overlaps.is_empty() {
        return Err(ScheduleBuildError::OverlappingOrder(overlaps));
    }
    // One entry per pair, remembering the first ordering that asked for it.
    raw.sort_unstable();
    raw.dedup_by_key(|e| (e.0, e.1));

    let mut succ = vec![Vec::new(); n];
    let mut waiting = vec![0usize; n];
    for &(x, y, _) in &raw {
        succ[x].push(y);
        waiting[y] += 1;
    }

    // Lowest-numbered ready system first.
    let mut ready: BinaryHeap<Reverse<usize>> =
        (0..n).filter(|&i| waiting[i] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(i)) = ready.pop() {
        order.push(i);
        for &j in &succ[i] {
            waiting[j] -= 1;
            if waiting[j] == 0 {
                ready.push(Reverse(j));
            }
        }
    }
    if order.len() < n {
        return Err(ScheduleBuildError::DependencyCycle(describe_cycles(
            input, &succ, &raw,
        )));
    }

    let mut sets_of_system = vec![Vec::new(); n];
    for (set, list) in systems_of_set.iter().enumerate() {
        for &s in list {
            sets_of_system[s].push(set);
        }
    }

    Ok(Built {
        order,
        systems_of_set,
        sets_of_system,
        edges: raw.iter().map(|&(x, y, _)| (x, y)).collect(),
    })
}

fn describe_cycles(
    input: &GraphInput,
    succ: &[Vec<usize>],
    raw: &[(usize, usize, usize)],
) -> Vec<DependencyCycle> {
    let components = cyclic_components(succ);
    let mut component_of = vec![usize::MAX; succ.len()];
    for (c, members) in components.iter().enumerate() {
        for &s in members {
            component_of[s] = c;
        }
    }
    let mut declared: Vec<Vec<usize>> = vec![Vec::new(); components.len()];
    for &(x, y, d) in raw {
        let c = component_of[x];
        if c != usize::MAX && c == component_of[y] && !declared[c].contains(&d) {
            declared[c].push(d);
        }
    }
    components
        .into_iter()
        .zip(declared)
        .map(|(members, mut asked)| {
            asked.sort_unstable();
            DependencyCycle {
                systems: members
                    .into_iter()
                    .map(|s| input.system_names[s].clone())
                    .collect(),
                orderings: asked
                    .into_iter()
                    .map(|d| {
                        let (first, second) = input.deps[d];
                        (input.name(first), input.name(second))
                    })
                    .collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    // ── a small deterministic generator ──────────────────────────────

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn chance(&mut self, num: u64, den: u64) -> bool {
            self.next() % den < num
        }
    }

    // ── fixtures ─────────────────────────────────────────────────────

    struct Spec {
        systems: Vec<String>,
        sets: Vec<String>,
        members: Vec<(usize, NodeId)>,
        deps: Vec<(NodeId, NodeId)>,
    }

    impl Spec {
        fn new(n_systems: usize, n_sets: usize) -> Self {
            Self {
                systems: (0..n_systems).map(|i| format!("s{i}")).collect(),
                sets: (0..n_sets).map(|i| format!("S{i}")).collect(),
                members: Vec::new(),
                deps: Vec::new(),
            }
        }

        fn build(&self) -> Result<Built, ScheduleBuildError> {
            build(&GraphInput {
                system_names: &self.systems,
                set_names: &self.sets,
                members: &self.members,
                deps: &self.deps,
            })
        }
    }

    fn s(i: usize) -> NodeId {
        NodeId::System(i)
    }

    fn set(i: usize) -> NodeId {
        NodeId::Set(i)
    }

    // ── oracles ──────────────────────────────────────────────────────

    /// Transitive closure of `edges` over `n` nodes, by Floyd-Warshall.
    fn closure(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<bool>> {
        let mut c = vec![vec![false; n]; n];
        for &(a, b) in edges {
            c[a][b] = true;
        }
        for k in 0..n {
            for i in 0..n {
                if c[i][k] {
                    for j in 0..n {
                        if c[k][j] {
                            c[i][j] = true;
                        }
                    }
                }
            }
        }
        c
    }

    /// The lowest-numbered-first topological order, the slow way.
    fn oracle_order(n: usize, edges: &[(usize, usize)]) -> Option<Vec<usize>> {
        let mut placed = vec![false; n];
        let mut order = Vec::new();
        for _ in 0..n {
            let next =
                (0..n).find(|&i| !placed[i] && edges.iter().all(|&(a, b)| b != i || placed[a]))?;
            placed[next] = true;
            order.push(next);
        }
        Some(order)
    }

    /// Each set's systems, by repeating until nothing changes.
    fn oracle_systems_of_set(n_sets: usize, members: &[(usize, NodeId)]) -> Vec<BTreeSet<usize>> {
        let mut out = vec![BTreeSet::new(); n_sets];
        loop {
            let mut changed = false;
            for &(set, member) in members {
                match member {
                    NodeId::System(x) => changed |= out[set].insert(x),
                    NodeId::Set(c) => {
                        let below: Vec<usize> = out[c].iter().copied().collect();
                        for x in below {
                            changed |= out[set].insert(x);
                        }
                    }
                }
            }
            if !changed {
                return out;
            }
        }
    }

    fn oracle_nodes(sets: &[BTreeSet<usize>], node: NodeId) -> BTreeSet<usize> {
        match node {
            NodeId::System(i) => BTreeSet::from([i]),
            NodeId::Set(i) => sets[i].clone(),
        }
    }

    fn random_dag(rng: &mut Rng, n: usize) -> Vec<(usize, usize)> {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                if rng.chance(1, 4) {
                    edges.push((perm[i], perm[j]));
                }
            }
        }
        edges
    }

    // ── the simple cases ─────────────────────────────────────────────

    #[test]
    fn nothing_builds_to_nothing() {
        let built = Spec::new(0, 0).build().unwrap();
        assert!(built.order.is_empty());
        assert!(built.edges.is_empty());
    }

    #[test]
    fn unordered_systems_run_in_the_order_they_were_added() {
        let built = Spec::new(5, 0).build().unwrap();
        assert_eq!(built.order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn a_dependency_moves_a_system_after_what_it_waits_for() {
        let mut spec = Spec::new(3, 0);
        spec.deps.push((s(2), s(0)));
        assert_eq!(spec.build().unwrap().order, vec![1, 2, 0]);
    }

    #[test]
    fn a_chain_runs_in_chain_order() {
        let mut spec = Spec::new(4, 0);
        spec.deps = vec![(s(3), s(2)), (s(2), s(1)), (s(1), s(0))];
        assert_eq!(spec.build().unwrap().order, vec![3, 2, 1, 0]);
    }

    #[test]
    fn ties_go_to_the_lowest_number_not_the_first_found() {
        let mut spec = Spec::new(4, 0);
        spec.deps = vec![(s(3), s(0))];
        assert_eq!(spec.build().unwrap().order, vec![1, 2, 3, 0]);
        let mut spec = Spec::new(4, 0);
        spec.deps = vec![(s(1), s(0)), (s(3), s(2))];
        assert_eq!(spec.build().unwrap().order, vec![1, 0, 3, 2]);
    }

    #[test]
    fn repeated_orderings_and_members_change_nothing() {
        let mut once = Spec::new(3, 1);
        once.members = vec![(0, s(0)), (0, s(1))];
        once.deps = vec![(set(0), s(2))];
        let mut twice = Spec::new(3, 1);
        twice.members = vec![(0, s(0)), (0, s(1)), (0, s(0)), (0, s(1))];
        twice.deps = vec![(set(0), s(2)), (set(0), s(2)), (s(0), s(2))];
        assert_eq!(once.build().unwrap(), twice.build().unwrap());
    }

    // ── sets ─────────────────────────────────────────────────────────

    #[test]
    fn ordering_two_sets_orders_every_pair_across_them() {
        let mut spec = Spec::new(4, 2);
        spec.members = vec![(0, s(2)), (0, s(3)), (1, s(0)), (1, s(1))];
        spec.deps = vec![(set(0), set(1))];
        let built = spec.build().unwrap();
        assert_eq!(built.edges, vec![(2, 0), (2, 1), (3, 0), (3, 1)]);
        assert_eq!(built.order, vec![2, 3, 0, 1]);
    }

    #[test]
    fn a_set_can_be_ordered_against_a_single_system() {
        let mut spec = Spec::new(3, 1);
        spec.members = vec![(0, s(1)), (0, s(2))];
        spec.deps = vec![(s(0), set(0))];
        assert_eq!(spec.build().unwrap().edges, vec![(0, 1), (0, 2)]);
        let mut spec = Spec::new(3, 1);
        spec.members = vec![(0, s(1)), (0, s(2))];
        spec.deps = vec![(set(0), s(0))];
        assert_eq!(spec.build().unwrap().edges, vec![(1, 0), (2, 0)]);
    }

    #[test]
    fn a_set_includes_the_systems_of_the_sets_inside_it() {
        let mut spec = Spec::new(3, 2);
        spec.members = vec![(0, set(1)), (0, s(2)), (1, s(0))];
        let built = spec.build().unwrap();
        assert_eq!(built.systems_of_set[0], vec![0, 2]);
        assert_eq!(built.systems_of_set[1], vec![0]);
        assert_eq!(built.sets_of_system[0], vec![0, 1]);
        assert_eq!(built.sets_of_system[1], Vec::<usize>::new());
        assert_eq!(built.sets_of_system[2], vec![0]);
    }

    #[test]
    fn a_system_in_two_sets_is_in_both() {
        let mut spec = Spec::new(2, 2);
        spec.members = vec![(0, s(0)), (1, s(0)), (1, s(1))];
        let built = spec.build().unwrap();
        assert_eq!(built.sets_of_system[0], vec![0, 1]);
        assert_eq!(built.sets_of_system[1], vec![1]);
    }

    #[test]
    fn an_empty_set_orders_nothing() {
        let mut spec = Spec::new(2, 1);
        spec.deps = vec![(set(0), s(0)), (s(1), set(0))];
        let built = spec.build().unwrap();
        assert!(built.edges.is_empty());
        assert_eq!(built.order, vec![0, 1]);
    }

    #[test]
    fn a_set_inside_another_orders_with_the_outer_one() {
        let mut spec = Spec::new(3, 2);
        spec.members = vec![(0, set(1)), (1, s(0)), (0, s(1))];
        spec.deps = vec![(s(2), set(0))];
        assert_eq!(spec.build().unwrap().edges, vec![(2, 0), (2, 1)]);
    }

    // ── errors ───────────────────────────────────────────────────────

    #[test]
    fn ordering_a_set_against_its_own_member_is_refused() {
        let mut spec = Spec::new(3, 1);
        spec.members = vec![(0, s(0)), (0, s(1))];
        spec.deps = vec![(set(0), s(0))];
        let Err(ScheduleBuildError::OverlappingOrder(list)) = spec.build() else {
            panic!("expected an overlapping order");
        };
        assert_eq!(
            list,
            vec![OverlappingOrder {
                before: "S0".into(),
                after: "s0".into(),
                shared: vec!["s0".into()],
            }]
        );
    }

    #[test]
    fn ordering_a_thing_against_itself_is_refused() {
        let mut spec = Spec::new(1, 1);
        spec.deps = vec![(s(0), s(0))];
        assert!(matches!(
            spec.build(),
            Err(ScheduleBuildError::OverlappingOrder(_))
        ));
        let mut spec = Spec::new(1, 1);
        spec.members = vec![(0, s(0))];
        spec.deps = vec![(set(0), set(0))];
        assert!(matches!(
            spec.build(),
            Err(ScheduleBuildError::OverlappingOrder(_))
        ));
    }

    #[test]
    fn two_sets_that_overlap_cannot_be_ordered() {
        let mut spec = Spec::new(3, 2);
        spec.members = vec![(0, s(0)), (0, s(1)), (1, s(1)), (1, s(2))];
        spec.deps = vec![(set(0), set(1))];
        let Err(ScheduleBuildError::OverlappingOrder(list)) = spec.build() else {
            panic!("expected an overlapping order");
        };
        assert_eq!(list[0].shared, vec!["s1".to_string()]);
    }

    #[test]
    fn every_overlapping_order_is_reported() {
        let mut spec = Spec::new(2, 1);
        spec.members = vec![(0, s(0)), (0, s(1))];
        spec.deps = vec![(set(0), s(0)), (s(1), set(0)), (s(0), s(1))];
        let Err(ScheduleBuildError::OverlappingOrder(list)) = spec.build() else {
            panic!("expected an overlapping order");
        };
        assert_eq!(list.len(), 2, "the plain ordering is fine");
    }

    #[test]
    fn a_dependency_cycle_names_the_systems_and_the_orderings() {
        let mut spec = Spec::new(4, 0);
        spec.deps = vec![(s(0), s(1)), (s(1), s(2)), (s(2), s(0)), (s(2), s(3))];
        let Err(ScheduleBuildError::DependencyCycle(cycles)) = spec.build() else {
            panic!("expected a cycle");
        };
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].systems, vec!["s0", "s1", "s2"]);
        assert_eq!(
            cycles[0].orderings,
            vec![
                ("s0".to_string(), "s1".to_string()),
                ("s1".to_string(), "s2".to_string()),
                ("s2".to_string(), "s0".to_string()),
            ],
            "the way out to s3 is not part of it"
        );
    }

    #[test]
    fn two_separate_cycles_are_both_reported() {
        let mut spec = Spec::new(5, 0);
        spec.deps = vec![
            (s(0), s(1)),
            (s(1), s(0)),
            (s(3), s(4)),
            (s(4), s(3)),
            (s(1), s(2)),
        ];
        let Err(ScheduleBuildError::DependencyCycle(cycles)) = spec.build() else {
            panic!("expected cycles");
        };
        assert_eq!(cycles.len(), 2);
        let all: BTreeSet<Vec<String>> = cycles.iter().map(|c| c.systems.clone()).collect();
        assert!(all.contains(&vec!["s0".to_string(), "s1".to_string()]));
        assert!(all.contains(&vec!["s3".to_string(), "s4".to_string()]));
    }

    #[test]
    fn a_cycle_through_sets_names_the_sets() {
        let mut spec = Spec::new(2, 2);
        spec.members = vec![(0, s(0)), (1, s(1))];
        spec.deps = vec![(set(0), set(1)), (set(1), set(0))];
        let Err(ScheduleBuildError::DependencyCycle(cycles)) = spec.build() else {
            panic!("expected a cycle");
        };
        assert_eq!(cycles[0].systems, vec!["s0", "s1"]);
        assert_eq!(
            cycles[0].orderings,
            vec![
                ("S0".to_string(), "S1".to_string()),
                ("S1".to_string(), "S0".to_string())
            ]
        );
    }

    #[test]
    fn a_set_that_contains_itself_is_a_hierarchy_cycle() {
        let mut spec = Spec::new(0, 1);
        spec.members = vec![(0, set(0))];
        assert_eq!(
            spec.build().unwrap_err(),
            ScheduleBuildError::HierarchyCycle(vec![vec!["S0".to_string()]])
        );
    }

    #[test]
    fn sets_that_contain_each_other_are_a_hierarchy_cycle() {
        let mut spec = Spec::new(1, 3);
        spec.members = vec![(0, set(1)), (1, set(2)), (2, set(0)), (2, s(0))];
        let Err(ScheduleBuildError::HierarchyCycle(cycles)) = spec.build() else {
            panic!("expected a hierarchy cycle");
        };
        assert_eq!(cycles, vec![vec!["S0", "S1", "S2"]]);
    }

    #[test]
    fn the_hierarchy_is_checked_before_the_orderings() {
        let mut spec = Spec::new(2, 1);
        spec.members = vec![(0, set(0))];
        spec.deps = vec![(s(0), s(1)), (s(1), s(0))];
        assert!(matches!(
            spec.build(),
            Err(ScheduleBuildError::HierarchyCycle(_))
        ));
    }

    #[test]
    fn errors_print_what_to_fix() {
        let mut spec = Spec::new(2, 0);
        spec.deps = vec![(s(0), s(1)), (s(1), s(0))];
        let text = spec.build().unwrap_err().to_string();
        assert!(text.contains("s0") && text.contains("s1"), "{text}");
        assert!(text.contains("before"), "{text}");

        let mut spec = Spec::new(1, 1);
        spec.members = vec![(0, s(0))];
        spec.deps = vec![(set(0), s(0))];
        let text = spec.build().unwrap_err().to_string();
        assert!(text.contains("S0") && text.contains("`s0`"), "{text}");

        let mut spec = Spec::new(0, 1);
        spec.members = vec![(0, set(0))];
        assert!(spec.build().unwrap_err().to_string().contains("S0"));

        let a = Ambiguity {
            first: "x".into(),
            second: "y".into(),
            conflicts: vec!["Pos", "Vel"],
            whole_world: false,
        };
        assert_eq!(
            a.to_string(),
            "`x` and `y` conflict on `Pos`, `Vel` and nothing orders them"
        );
        let a = Ambiguity {
            first: "x".into(),
            second: "y".into(),
            conflicts: vec![],
            whole_world: true,
        };
        assert!(a.to_string().contains("the whole world and nothing"), "{a}");
        let a = Ambiguity {
            first: "x".into(),
            second: "y".into(),
            conflicts: vec!["Pos"],
            whole_world: true,
        };
        assert!(a.to_string().contains("the whole world, `Pos`"), "{a}");
        let e = ScheduleBuildError::Ambiguities(vec![a]);
        assert!(e.to_string().starts_with("ambiguous systems:"));
    }

    // ── reachability ─────────────────────────────────────────────────

    #[test]
    fn reach_is_transitive_and_one_way() {
        let mut spec = Spec::new(4, 0);
        spec.deps = vec![(s(0), s(1)), (s(1), s(2))];
        let reach = spec.build().unwrap().reachability();
        assert!(reach.contains(0, 1));
        assert!(reach.contains(1, 2));
        assert!(reach.contains(0, 2), "through 1");
        assert!(!reach.contains(2, 0));
        assert!(!reach.contains(0, 3));
        assert!(!reach.contains(3, 1));
        assert!(!reach.contains(0, 0), "never before itself");
    }

    #[test]
    fn reach_works_past_one_word_of_bits() {
        let n = 200;
        let mut spec = Spec::new(n, 0);
        spec.deps = (0..n - 1).map(|i| (s(i), s(i + 1))).collect();
        let built = spec.build().unwrap();
        let reach = built.reachability();
        assert!(reach.contains(0, 199));
        assert!(reach.contains(63, 64));
        assert!(reach.contains(10, 130));
        assert!(!reach.contains(130, 10));
        assert!(!reach.contains(199, 0));
        assert_eq!(built.order, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn reach_matches_the_closure_on_graphs_wider_than_a_word() {
        // Sparse random graphs of 70 to 140 systems, so every bit position
        // of a word and more than one word per row are used.
        let mut rng = Rng(0xA5A5_5A5A_1357_9BDF);
        for case in 0..12 {
            let n = 70 + rng.below(71);
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                perm.swap(i, rng.below(i + 1));
            }
            let mut edges = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    if rng.chance(1, 40) {
                        edges.push((perm[i], perm[j]));
                    }
                }
            }
            let mut spec = Spec::new(n, 0);
            spec.deps = edges.iter().map(|&(a, b)| (s(a), s(b))).collect();
            let reach = spec.build().unwrap().reachability();
            let c = closure(n, &edges);
            let mut reached = 0;
            for a in 0..n {
                for b in 0..n {
                    assert_eq!(reach.contains(a, b), c[a][b], "case {case}: {a}->{b}");
                    reached += usize::from(c[a][b]);
                }
            }
            assert!(
                reached > n,
                "case {case}: the graph is too sparse to mean anything"
            );
        }
    }

    #[test]
    fn a_single_edge_sets_exactly_one_bit_wherever_it_is() {
        let n = 200;
        for to in [0usize, 1, 31, 32, 33, 63, 64, 65, 100, 127, 128, 199] {
            let from = (to + 7) % n;
            if from == to {
                continue;
            }
            let mut spec = Spec::new(n, 0);
            spec.deps = vec![(s(from), s(to))];
            let reach = spec.build().unwrap().reachability();
            for a in 0..n {
                for b in 0..n {
                    assert_eq!(
                        reach.contains(a, b),
                        (a, b) == (from, to),
                        "edge {from}->{to}, asked {a}->{b}"
                    );
                }
            }
        }
    }

    // ── against brute force ──────────────────────────────────────────

    #[test]
    fn random_acyclic_orderings_match_the_slow_oracle() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..300 {
            let n = 1 + rng.below(14);
            let edges = random_dag(&mut rng, n);
            let mut spec = Spec::new(n, 0);
            spec.deps = edges.iter().map(|&(a, b)| (s(a), s(b))).collect();
            let built = spec.build().unwrap_or_else(|e| panic!("case {case}: {e}"));

            // every declared ordering holds
            let mut position = vec![0; n];
            for (i, &sys) in built.order.iter().enumerate() {
                position[sys] = i;
            }
            for &(a, b) in &edges {
                assert!(position[a] < position[b], "case {case}: {a} before {b}");
            }
            // the exact order, and again for determinism
            assert_eq!(
                Some(built.order.clone()),
                oracle_order(n, &edges),
                "case {case}"
            );
            assert_eq!(spec.build().unwrap(), built, "case {case}: deterministic");
            // reachability is the closure
            let c = closure(n, &edges);
            let reach = built.reachability();
            for a in 0..n {
                for b in 0..n {
                    assert_eq!(reach.contains(a, b), c[a][b], "case {case}: {a}->{b}");
                }
            }
            // the edges are exactly the declared ones, once each
            let want: BTreeSet<(usize, usize)> = edges.iter().copied().collect();
            assert_eq!(built.edges.iter().copied().collect::<BTreeSet<_>>(), want);
            assert_eq!(built.edges.len(), want.len());
        }
    }

    #[test]
    fn an_injected_back_edge_is_always_reported_with_the_right_systems() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let mut checked = 0;
        for case in 0..300 {
            let n = 2 + rng.below(12);
            let mut edges = random_dag(&mut rng, n);
            let c = closure(n, &edges);
            let pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|a| (0..n).map(move |b| (a, b)))
                .filter(|&(a, b)| c[a][b])
                .collect();
            if pairs.is_empty() {
                continue;
            }
            let (a, b) = pairs[rng.below(pairs.len())];
            edges.push((b, a));
            let mut spec = Spec::new(n, 0);
            spec.deps = edges.iter().map(|&(x, y)| (s(x), s(y))).collect();
            let Err(ScheduleBuildError::DependencyCycle(cycles)) = spec.build() else {
                panic!("case {case}: the cycle was not found");
            };
            // oracle components: nodes that reach themselves, grouped by mutual reach
            let c2 = closure(n, &edges);
            let mut want: BTreeSet<Vec<String>> = BTreeSet::new();
            for i in 0..n {
                if c2[i][i] {
                    let group: Vec<String> = (0..n)
                        .filter(|&j| c2[i][j] && c2[j][i])
                        .map(|j| format!("s{j}"))
                        .collect();
                    want.insert(group);
                }
            }
            let got: BTreeSet<Vec<String>> = cycles.iter().map(|c| c.systems.clone()).collect();
            assert_eq!(got, want, "case {case}");
            assert!(got
                .iter()
                .any(|g| g.contains(&format!("s{a}")) && g.contains(&format!("s{b}"))));
            // every reported ordering joins two systems of its own cycle
            for cycle in &cycles {
                assert!(!cycle.orderings.is_empty());
                for (x, y) in &cycle.orderings {
                    assert!(
                        cycle.systems.contains(x) && cycle.systems.contains(y),
                        "case {case}"
                    );
                }
            }
            checked += 1;
        }
        assert!(
            checked > 100,
            "the generator made too few cycles: {checked}"
        );
    }

    #[test]
    fn random_sets_and_orderings_match_the_slow_oracle() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let (mut ok, mut overlap, mut cyclic) = (0, 0, 0);
        for case in 0..600 {
            let n_sys = 1 + rng.below(8);
            let n_set = 1 + rng.below(5);
            let mut spec = Spec::new(n_sys, n_set);
            // a set may hold systems and sets of lower number: acyclic
            for set_i in 0..n_set {
                for x in 0..n_sys {
                    if rng.chance(1, 3) {
                        spec.members.push((set_i, s(x)));
                    }
                }
                for child in 0..set_i {
                    if rng.chance(1, 4) {
                        spec.members.push((set_i, set(child)));
                    }
                }
            }
            let node = |rng: &mut Rng| {
                if rng.chance(1, 2) {
                    s(rng.below(n_sys))
                } else {
                    set(rng.below(n_set))
                }
            };
            for _ in 0..rng.below(7) {
                let (a, b) = (node(&mut rng), node(&mut rng));
                spec.deps.push((a, b));
            }

            let sets = oracle_systems_of_set(n_set, &spec.members);
            // expected overlaps
            let mut want_overlaps = Vec::new();
            let mut edges: Vec<(usize, usize)> = Vec::new();
            for &(a, b) in &spec.deps {
                let (xa, xb) = (oracle_nodes(&sets, a), oracle_nodes(&sets, b));
                let shared: Vec<String> = xa.intersection(&xb).map(|i| format!("s{i}")).collect();
                if shared.is_empty() {
                    for &x in &xa {
                        for &y in &xb {
                            edges.push((x, y));
                        }
                    }
                } else {
                    want_overlaps.push(shared);
                }
            }
            let result = spec.build();
            if !want_overlaps.is_empty() {
                let Err(ScheduleBuildError::OverlappingOrder(list)) = result else {
                    panic!("case {case}: expected overlaps");
                };
                let got: Vec<Vec<String>> = list.into_iter().map(|o| o.shared).collect();
                assert_eq!(got, want_overlaps, "case {case}");
                overlap += 1;
                continue;
            }
            let c = closure(n_sys, &edges);
            if (0..n_sys).any(|i| c[i][i]) {
                assert!(
                    matches!(result, Err(ScheduleBuildError::DependencyCycle(_))),
                    "case {case}: expected a cycle"
                );
                cyclic += 1;
                continue;
            }
            let built = result.unwrap_or_else(|e| panic!("case {case}: {e}"));
            ok += 1;
            assert_eq!(
                Some(built.order.clone()),
                oracle_order(n_sys, &edges),
                "case {case}"
            );
            let want: BTreeSet<(usize, usize)> = edges.iter().copied().collect();
            assert_eq!(
                built.edges.iter().copied().collect::<BTreeSet<_>>(),
                want,
                "case {case}"
            );
            for (set_i, want_systems) in sets.iter().enumerate() {
                assert_eq!(
                    built.systems_of_set[set_i],
                    want_systems.iter().copied().collect::<Vec<_>>(),
                    "case {case}: set {set_i}"
                );
            }
            for x in 0..n_sys {
                let want_sets: Vec<usize> = (0..n_set).filter(|&i| sets[i].contains(&x)).collect();
                assert_eq!(
                    built.sets_of_system[x], want_sets,
                    "case {case}: system {x}"
                );
            }
            let reach = built.reachability();
            for a in 0..n_sys {
                for b in 0..n_sys {
                    assert_eq!(reach.contains(a, b), c[a][b], "case {case}");
                }
            }
        }
        assert!(
            ok > 100 && overlap > 20 && cyclic > 5,
            "{ok} ok, {overlap} overlap, {cyclic} cyclic"
        );
    }

    #[test]
    fn random_hierarchy_cycles_are_found_and_acyclic_hierarchies_are_not_flagged() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for case in 0..400 {
            let n_set = 1 + rng.below(7);
            let mut spec = Spec::new(0, n_set);
            let mut edges = Vec::new();
            for p in 0..n_set {
                for c in 0..n_set {
                    if rng.chance(1, 6) {
                        spec.members.push((p, set(c)));
                        edges.push((p, c));
                    }
                }
            }
            let c = closure(n_set, &edges);
            let cyclic = (0..n_set).any(|i| c[i][i]);
            match spec.build() {
                Err(ScheduleBuildError::HierarchyCycle(list)) => {
                    assert!(cyclic, "case {case}: flagged an acyclic hierarchy");
                    let mut want: BTreeSet<Vec<String>> = BTreeSet::new();
                    for i in 0..n_set {
                        if c[i][i] {
                            want.insert(
                                (0..n_set)
                                    .filter(|&j| c[i][j] && c[j][i])
                                    .map(|j| format!("S{j}"))
                                    .collect(),
                            );
                        }
                    }
                    assert_eq!(
                        list.into_iter().collect::<BTreeSet<_>>(),
                        want,
                        "case {case}"
                    );
                }
                Ok(_) => assert!(!cyclic, "case {case}: missed a cycle"),
                Err(e) => panic!("case {case}: {e}"),
            }
        }
    }

    // ── size ─────────────────────────────────────────────────────────

    #[test]
    fn a_long_chain_and_a_deep_hierarchy_do_not_overflow_the_stack() {
        let n = 20_000;
        let mut spec = Spec::new(n, 0);
        spec.deps = (0..n - 1).map(|i| (s(i + 1), s(i))).collect();
        let built = spec.build().unwrap();
        assert_eq!(built.order[0], n - 1);
        assert_eq!(built.order[n - 1], 0);

        let depth = 20_000;
        let mut spec = Spec::new(1, depth);
        spec.members = (0..depth - 1).map(|i| (i + 1, set(i))).collect();
        spec.members.push((0, s(0)));
        let built = spec.build().unwrap();
        assert_eq!(built.systems_of_set[depth - 1], vec![0]);
        assert_eq!(built.sets_of_system[0].len(), depth);

        let mut spec = Spec::new(0, depth);
        spec.members = (0..depth - 1).map(|i| (i + 1, set(i))).collect();
        spec.members.push((0, set(depth - 1)));
        assert!(matches!(
            spec.build(),
            Err(ScheduleBuildError::HierarchyCycle(_))
        ));

        let mut spec = Spec::new(n, 0);
        spec.deps = (0..n).map(|i| (s(i), s((i + 1) % n))).collect();
        let Err(ScheduleBuildError::DependencyCycle(c)) = spec.build() else {
            panic!("expected one big cycle");
        };
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].systems.len(), n);
    }

    #[test]
    fn two_big_sets_ordered_against_each_other_expand_to_a_product() {
        let mut spec = Spec::new(60, 2);
        spec.members = (0..30)
            .map(|i| (0, s(i)))
            .chain((30..60).map(|i| (1, s(i))))
            .collect();
        spec.deps = vec![(set(0), set(1))];
        let built = spec.build().unwrap();
        assert_eq!(built.edges.len(), 900);
        assert_eq!(built.order, (0..60).collect::<Vec<_>>());
    }
}
