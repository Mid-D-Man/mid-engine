// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule-parallel.md, section "tests_parallel.rs"
// ============================================================================
//! The multi-threaded executor: order, mutual exclusion, real concurrency,
//! deferred work, conditions, exclusive systems, panics, small pools, and a
//! randomized comparison with the single-threaded executor.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::*;
use crate::commands::Commands;
use crate::system_param::{Query, Res, ResMut};
use crate::tick::Tick;
use crate::world::Entity;

// ── fixtures ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Update;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Node(usize);
impl SystemSet for Node {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Group(u32);
impl SystemSet for Group {}

fn pool(threads: usize) -> Arc<rayon::ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap(),
    )
}

fn multi(threads: usize) -> Schedule {
    let mut s = Schedule::new(Update);
    s.set_executor(ExecutorKind::MultiThreaded)
        .set_thread_pool(Some(pool(threads)));
    s
}

struct Clock(AtomicUsize);

/// Per-system records of one run.
struct Trace {
    start: Vec<AtomicUsize>,
    end: Vec<AtomicUsize>,
    runs: Vec<AtomicUsize>,
    active: AtomicUsize,
    max_active: AtomicUsize,
}

impl Trace {
    fn new(n: usize) -> Self {
        let v = |n| (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
        Self {
            start: v(n),
            end: v(n),
            runs: (0..n).map(|_| AtomicUsize::new(0)).collect(),
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        for a in self.start.iter().chain(&self.end) {
            a.store(usize::MAX, SeqCst);
        }
    }
}

trait Pushes {
    fn push(&mut self, id: usize);
    fn get(&self) -> &[usize];
}

macro_rules! resources {
    ($($name:ident),*) => {
        $(
            struct $name(Vec<usize>);
            impl Pushes for $name {
                fn push(&mut self, id: usize) { self.0.push(id); }
                fn get(&self) -> &[usize] { &self.0 }
            }
        )*
    };
}
resources!(R0, R1, R2, R3);

fn begin(clock: &Clock, trace: &Trace, id: usize) {
    trace.start[id].store(clock.0.fetch_add(1, SeqCst), SeqCst);
    let now = trace.active.fetch_add(1, SeqCst) + 1;
    trace.max_active.fetch_max(now, SeqCst);
    std::thread::yield_now();
}

fn finish(clock: &Clock, trace: &Trace, id: usize) {
    std::thread::yield_now();
    trace.runs[id].fetch_add(1, SeqCst);
    trace.active.fetch_sub(1, SeqCst);
    trace.end[id].store(clock.0.fetch_add(1, SeqCst), SeqCst);
}

fn probe_rw<Rd: Sync + 'static, Wr: Pushes + Send + 'static>(
    id: usize,
) -> impl FnMut(Res<Clock>, Res<Trace>, Res<Rd>, ResMut<Wr>) + Send + 'static {
    move |clock: Res<Clock>, trace: Res<Trace>, _r: Res<Rd>, mut w: ResMut<Wr>| {
        begin(&clock, &trace, id);
        w.push(id);
        finish(&clock, &trace, id);
    }
}

fn probe_r<Rd: Sync + 'static>(
    id: usize,
) -> impl FnMut(Res<Clock>, Res<Trace>, Res<Rd>) + Send + 'static {
    move |clock: Res<Clock>, trace: Res<Trace>, _r: Res<Rd>| {
        begin(&clock, &trace, id);
        finish(&clock, &trace, id);
    }
}

fn probe_w<Wr: Pushes + Send + 'static>(
    id: usize,
) -> impl FnMut(Res<Clock>, Res<Trace>, ResMut<Wr>) + Send + 'static {
    move |clock: Res<Clock>, trace: Res<Trace>, mut w: ResMut<Wr>| {
        begin(&clock, &trace, id);
        w.push(id);
        finish(&clock, &trace, id);
    }
}

fn add_probe(s: &mut Schedule, id: usize, read: Option<usize>, write: Option<usize>) {
    macro_rules! rw {
        ($r:ident, $w:ident) => {
            s.add_systems(probe_rw::<$r, $w>(id).in_set(Node(id)))
        };
    }
    macro_rules! r {
        ($r:ident) => {
            s.add_systems(probe_r::<$r>(id).in_set(Node(id)))
        };
    }
    macro_rules! w {
        ($w:ident) => {
            s.add_systems(probe_w::<$w>(id).in_set(Node(id)))
        };
    }
    match (read, write) {
        (Some(0), None) => r!(R0),
        (Some(1), None) => r!(R1),
        (Some(2), None) => r!(R2),
        (Some(3), None) => r!(R3),
        (None, Some(0)) => w!(R0),
        (None, Some(1)) => w!(R1),
        (None, Some(2)) => w!(R2),
        (None, Some(3)) => w!(R3),
        (Some(0), Some(1)) => rw!(R0, R1),
        (Some(0), Some(2)) => rw!(R0, R2),
        (Some(0), Some(3)) => rw!(R0, R3),
        (Some(1), Some(0)) => rw!(R1, R0),
        (Some(1), Some(2)) => rw!(R1, R2),
        (Some(1), Some(3)) => rw!(R1, R3),
        (Some(2), Some(0)) => rw!(R2, R0),
        (Some(2), Some(1)) => rw!(R2, R1),
        (Some(2), Some(3)) => rw!(R2, R3),
        (Some(3), Some(0)) => rw!(R3, R0),
        (Some(3), Some(1)) => rw!(R3, R1),
        (Some(3), Some(2)) => rw!(R3, R2),
        other => panic!("not a probe: {other:?}"),
    };
}

fn probe_world(n: usize) -> World {
    let mut w = World::new();
    w.insert_resource(Clock(AtomicUsize::new(0)));
    w.insert_resource(Trace::new(n));
    w.insert_resource(R0(Vec::new()));
    w.insert_resource(R1(Vec::new()));
    w.insert_resource(R2(Vec::new()));
    w.insert_resource(R3(Vec::new()));
    w
}

fn sequences(w: &World) -> [Vec<usize>; 4] {
    [
        w.get_resource::<R0>().unwrap().get().to_vec(),
        w.get_resource::<R1>().unwrap().get().to_vec(),
        w.get_resource::<R2>().unwrap().get().to_vec(),
        w.get_resource::<R3>().unwrap().get().to_vec(),
    ]
}

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

/// How long a test waits for something that should happen at once.
const PATIENCE: Duration = Duration::from_secs(20);

fn wait_for(flag: &AtomicBool, what: &str) {
    let start = Instant::now();
    while !flag.load(SeqCst) {
        assert!(start.elapsed() < PATIENCE, "gave up waiting for {what}");
        std::thread::yield_now();
    }
}

// ── the random comparison ────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Spec {
    read: Option<usize>,
    write: Option<usize>,
}

fn conflict(a: Spec, b: Spec) -> bool {
    let hit = |x: Option<usize>, y: Option<usize>| x.is_some() && x == y;
    hit(a.write, b.write) || hit(a.write, b.read) || hit(a.read, b.write)
}

struct Case {
    specs: Vec<Spec>,
    /// `(a, b)`: `a` runs before `b`.
    orders: Vec<(usize, usize)>,
}

fn random_case(rng: &mut Rng, fully_ordered: bool) -> Case {
    let n = 2 + rng.below(8);
    let specs: Vec<Spec> = (0..n)
        .map(|_| loop {
            let read = rng.chance(1, 2).then(|| rng.below(4));
            let write = rng.chance(2, 3).then(|| rng.below(4));
            if read.is_none() && write.is_none() {
                continue;
            }
            if read.is_some() && read == write {
                continue;
            }
            break Spec { read, write };
        })
        .collect();
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    let mut orders = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if rng.chance(1, 4) {
                orders.push((perm[i], perm[j]));
            }
        }
    }
    if fully_ordered {
        // transitive closure of what is there, then order every conflicting
        // pair that is still unordered, lower number first
        let mut c = vec![vec![false; n]; n];
        for &(a, b) in &orders {
            c[a][b] = true;
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if c[i][k] && c[k][j] {
                        c[i][j] = true;
                    }
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                if conflict(specs[i], specs[j]) && !c[i][j] && !c[j][i] {
                    orders.push((i, j));
                    c[i][j] = true;
                    for a in 0..n {
                        for b in 0..n {
                            if (a == i || c[a][i]) && (b == j || c[j][b]) {
                                c[a][b] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    Case { specs, orders }
}

fn build_case(case: &Case, kind: ExecutorKind, threads: usize) -> Schedule {
    let mut s = Schedule::new(Update);
    s.set_executor(kind).set_thread_pool(Some(pool(threads)));
    for (id, spec) in case.specs.iter().enumerate() {
        add_probe(&mut s, id, spec.read, spec.write);
    }
    for &(a, b) in &case.orders {
        s.configure_sets(Node(a).before(Node(b)));
    }
    s
}

fn check_run(case: &Case, trace: &Trace, label: &str) {
    for (id, _) in case.specs.iter().enumerate() {
        assert_eq!(
            trace.runs[id].load(SeqCst),
            1,
            "{label}: system {id} ran once"
        );
    }
    let (s, e) = (&trace.start, &trace.end);
    for &(a, b) in &case.orders {
        assert!(
            e[a].load(SeqCst) < s[b].load(SeqCst),
            "{label}: {a} must finish before {b} starts"
        );
    }
    for x in 0..case.specs.len() {
        for y in x + 1..case.specs.len() {
            if conflict(case.specs[x], case.specs[y]) {
                let apart =
                    e[x].load(SeqCst) < s[y].load(SeqCst) || e[y].load(SeqCst) < s[x].load(SeqCst);
                assert!(apart, "{label}: {x} and {y} conflict but overlapped");
            }
        }
    }
}

#[test]
fn random_schedules_respect_order_and_exclusion_on_a_pool() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut widest = 0;
    for case_no in 0..60 {
        let case = random_case(&mut rng, false);
        let mut schedule = build_case(&case, ExecutorKind::MultiThreaded, 4);
        let mut world = probe_world(case.specs.len());
        for run in 0..3 {
            world.get_resource::<Trace>().unwrap().reset();
            schedule.run(&mut world);
            let trace = world.get_resource::<Trace>().unwrap();
            check_run(&case, trace, &format!("case {case_no} run {run}"));
            widest = widest.max(trace.max_active.load(SeqCst));
            for r in trace.runs.iter() {
                r.store(0, SeqCst);
            }
        }
    }
    assert!(widest >= 2, "nothing ever ran side by side: {widest}");
}

#[test]
fn fully_ordered_random_schedules_match_the_single_threaded_executor_exactly() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for case_no in 0..60 {
        let case = random_case(&mut rng, true);
        let mut single = build_case(&case, ExecutorKind::SingleThreaded, 1);
        let mut parallel = build_case(&case, ExecutorKind::MultiThreaded, 4);
        let mut w1 = probe_world(case.specs.len());
        let mut w2 = probe_world(case.specs.len());
        for run in 0..3 {
            single.run(&mut w1);
            w2.get_resource::<Trace>().unwrap().reset();
            parallel.run(&mut w2);
            let trace = w2.get_resource::<Trace>().unwrap();
            check_run(&case, trace, &format!("case {case_no} run {run}"));
            for r in trace.runs.iter() {
                r.store(0, SeqCst);
            }
            assert_eq!(sequences(&w1), sequences(&w2), "case {case_no} run {run}");
        }
    }
}

// ── real concurrency ─────────────────────────────────────────────────

struct Meet {
    a: AtomicBool,
    b: AtomicBool,
}

fn side_a(m: Res<Meet>) {
    m.a.store(true, SeqCst);
    wait_for(&m.b, "the other side to start");
}

fn side_b(m: Res<Meet>) {
    m.b.store(true, SeqCst);
    wait_for(&m.a, "the other side to start");
}

fn meet_world() -> World {
    let mut w = World::new();
    w.insert_resource(Meet {
        a: AtomicBool::new(false),
        b: AtomicBool::new(false),
    });
    w
}

#[test]
fn two_systems_that_do_not_conflict_really_run_at_the_same_time() {
    // each can only finish once the other has started
    let mut w = meet_world();
    let mut s = multi(4);
    s.add_systems((side_a, side_b));
    s.run(&mut w);
}

#[test]
fn the_single_threaded_executor_cannot_do_that() {
    // the same pair on one thread would wait forever for the other, so
    // this checks the test above tests concurrency: with a short patience
    // the first one gives up
    fn impatient(m: Res<Meet>) {
        let start = Instant::now();
        m.a.store(true, SeqCst);
        while !m.b.load(SeqCst) {
            if start.elapsed() > Duration::from_millis(200) {
                return;
            }
            std::thread::yield_now();
        }
        m.b.store(false, SeqCst); // would mean something else ran meanwhile
    }
    let mut w = meet_world();
    let mut s = Schedule::new(Update);
    s.add_systems(impatient);
    s.run(&mut w);
    assert!(w.get_resource::<Meet>().unwrap().a.load(SeqCst));
}

#[test]
fn many_independent_systems_overlap_on_a_wide_pool() {
    let n = 12;
    let mut w = probe_world(n);
    let mut s = multi(4);
    for id in 0..n {
        add_probe(&mut s, id, Some(id % 4), None);
    }
    for _ in 0..5 {
        s.run(&mut w);
    }
    let t = w.get_resource::<Trace>().unwrap();
    assert!(t.max_active.load(SeqCst) >= 2);
    assert!(t.runs.iter().all(|r| r.load(SeqCst) == 5));
}

#[test]
fn writers_to_one_resource_never_overlap() {
    let n = 10;
    let mut w = probe_world(n);
    let mut s = multi(4);
    for id in 0..n {
        add_probe(&mut s, id, None, Some(0));
    }
    for _ in 0..5 {
        s.run(&mut w);
    }
    let t = w.get_resource::<Trace>().unwrap();
    assert_eq!(t.max_active.load(SeqCst), 1);
    assert_eq!(w.get_resource::<R0>().unwrap().get().len(), n * 5);
}

#[test]
fn a_pool_of_one_thread_and_running_inside_the_pool_both_work() {
    // a hang here would be a deadlock: the test runs in a thread that is
    // given a deadline
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let one = pool(1);
        let mut w = probe_world(6);
        let mut s = Schedule::new(Update);
        s.set_executor(ExecutorKind::MultiThreaded)
            .set_thread_pool(Some(one.clone()));
        add_probe(&mut s, 0, None, Some(0));
        add_probe(&mut s, 1, Some(0), None);
        add_probe(&mut s, 2, None, Some(1));
        add_probe(&mut s, 3, Some(2), None);
        add_probe(&mut s, 4, None, Some(3));
        add_probe(&mut s, 5, Some(3), Some(2));
        s.configure_sets(Node(0).before(Node(1)));
        s.run(&mut w);
        // and from inside a task of that same pool
        let counted = one.install(|| {
            let mut w = probe_world(6);
            let mut s2 = Schedule::new(Update);
            s2.set_executor(ExecutorKind::MultiThreaded)
                .set_thread_pool(Some(one.clone()));
            for id in 0..6 {
                add_probe(&mut s2, id, None, Some(id % 4));
            }
            s2.run(&mut w);
            let t = w.get_resource::<Trace>().unwrap();
            t.runs.iter().map(|r| r.load(SeqCst)).sum::<usize>()
        });
        tx.send((
            w.get_resource::<Trace>()
                .unwrap()
                .runs
                .iter()
                .map(|r| r.load(SeqCst))
                .sum::<usize>(),
            counted,
        ))
        .unwrap();
    });
    let (outer, inner) = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the run deadlocked");
    assert_eq!((outer, inner), (6, 6));
}

#[test]
fn the_global_pool_is_used_when_none_is_given() {
    let mut w = probe_world(3);
    let mut s = Schedule::new(Update);
    s.set_executor(ExecutorKind::MultiThreaded);
    for id in 0..3 {
        add_probe(&mut s, id, None, Some(id));
    }
    s.run(&mut w);
    assert!(w
        .get_resource::<Trace>()
        .unwrap()
        .runs
        .iter()
        .all(|r| r.load(SeqCst) == 1));
}

// ── deferred work ────────────────────────────────────────────────────

fn spawn_one(mut commands: Commands) {
    commands.spawn();
}

struct Counts(Vec<usize>);

fn count_entities(q: Query<Entity>, mut counts: ResMut<Counts>) {
    counts.0.push(q.iter().count());
}

#[test]
fn what_a_system_defers_is_applied_before_anything_ordered_after_it() {
    for _ in 0..20 {
        let mut w = World::new();
        w.insert_resource(Counts(Vec::new()));
        let mut s = multi(4);
        s.add_systems((spawn_one, count_entities, spawn_one, count_entities).chain());
        s.run(&mut w);
        assert_eq!(w.get_resource::<Counts>().unwrap().0, [1, 2]);
        assert_eq!(w.entity_count(), 2);
    }
}

#[test]
fn a_diamond_after_a_deferring_system_sees_its_work_in_every_branch() {
    for _ in 0..20 {
        let mut w = World::new();
        w.insert_resource(Counts(Vec::new()));
        let mut s = multi(4);
        s.configure_sets((Group(0), Group(1)).chain());
        s.add_systems((spawn_one.in_set(Group(0)), count_entities.in_set(Group(1))));
        s.run(&mut w);
        assert_eq!(w.get_resource::<Counts>().unwrap().0, [1]);
    }
}

#[test]
fn the_last_systems_deferred_work_is_applied_before_the_run_returns() {
    let mut w = World::new();
    let mut s = multi(4);
    s.add_systems((spawn_one, spawn_one, spawn_one));
    s.run(&mut w);
    assert_eq!(w.entity_count(), 3);
}

struct Applied(Vec<&'static str>);

fn defer_a(mut commands: Commands) {
    commands.queue(|w: &mut World| w.get_resource_mut::<Applied>().unwrap().0.push("a"));
}

fn defer_b(mut commands: Commands) {
    commands.queue(|w: &mut World| w.get_resource_mut::<Applied>().unwrap().0.push("b"));
}

fn defer_c(mut commands: Commands) {
    commands.queue(|w: &mut World| w.get_resource_mut::<Applied>().unwrap().0.push("c"));
}

#[test]
fn deferred_work_is_applied_in_the_schedules_order_whichever_system_finished_first() {
    for _ in 0..30 {
        let mut w = World::new();
        w.insert_resource(Applied(Vec::new()));
        let mut s = multi(4);
        s.add_systems((defer_c, defer_a, defer_b));
        s.run(&mut w);
        assert_eq!(w.get_resource::<Applied>().unwrap().0, ["c", "a", "b"]);
    }
}

struct Flags {
    s_done: AtomicBool,
}

fn x_slow(t: Res<Trace>, f: Res<Flags>) {
    t.active.fetch_add(1, SeqCst);
    wait_for(&f.s_done, "the deferring system");
    for _ in 0..3000 {
        std::thread::yield_now();
    }
    t.active.fetch_sub(1, SeqCst);
}

fn s_defer(mut commands: Commands, f: Res<Flags>) {
    f.s_done.store(true, SeqCst);
    commands.queue(|w: &mut World| {
        let t = w.get_resource::<Trace>().unwrap();
        assert_eq!(
            t.active.load(SeqCst),
            0,
            "applied while a system was running"
        );
    });
}

#[test]
fn deferred_work_is_never_applied_while_another_system_is_running() {
    for _ in 0..5 {
        let mut w = probe_world(1);
        w.insert_resource(Flags {
            s_done: AtomicBool::new(false),
        });
        let mut s = multi(4);
        s.add_systems((x_slow, s_defer, other.after(s_defer)));
        s.run(&mut w);
    }
}

fn defer_v_slow(mut commands: Commands) {
    for _ in 0..3000 {
        std::thread::yield_now();
    }
    commands.queue(|w: &mut World| w.get_resource_mut::<Applied>().unwrap().0.push("v"));
}

fn defer_u(mut commands: Commands) {
    commands.queue(|w: &mut World| w.get_resource_mut::<Applied>().unwrap().0.push("u"));
}

fn quick() {}

#[test]
fn deferred_work_is_applied_in_run_order_even_when_it_finished_in_another() {
    // run order: v, quick, u (u waits for quick); u is added first. u finishes
    // long before the slow v, and its work still comes after v's
    for _ in 0..10 {
        let mut w = World::new();
        w.insert_resource(Applied(Vec::new()));
        let mut s = multi(4);
        s.add_systems((defer_u.after(quick), defer_v_slow, quick));
        s.run(&mut w);
        assert_eq!(w.get_resource::<Applied>().unwrap().0, ["v", "u"]);
    }
}

static EMPTY_ACCESS_ACTIVE: AtomicUsize = AtomicUsize::new(0);

fn touches_nothing() {
    EMPTY_ACCESS_ACTIVE.fetch_add(1, SeqCst);
    for _ in 0..2000 {
        std::thread::yield_now();
    }
    EMPTY_ACCESS_ACTIVE.fetch_sub(1, SeqCst);
}

fn exclusive_checks_nothing_else_runs(_w: &mut World) {
    assert_eq!(EMPTY_ACCESS_ACTIVE.load(SeqCst), 0, "a system was running");
    for _ in 0..500 {
        std::thread::yield_now();
        assert_eq!(
            EMPTY_ACCESS_ACTIVE.load(SeqCst),
            0,
            "a system started meanwhile"
        );
    }
}

#[test]
fn an_exclusive_system_does_not_overlap_even_a_system_that_touches_nothing() {
    for _ in 0..5 {
        let mut w = World::new();
        let mut s = multi(4);
        s.add_systems((exclusive_checks_nothing_else_runs, touches_nothing));
        s.run(&mut w);
    }
}

#[test]
fn an_exclusive_system_after_a_deferring_one_sees_its_work() {
    fn count_world(world: &mut World) {
        let n = world.entity_count();
        world.get_resource_mut::<Counts>().unwrap().0.push(n);
    }
    let mut w = World::new();
    w.insert_resource(Counts(Vec::new()));
    let mut s = multi(4);
    s.add_systems((spawn_one, count_world).chain());
    s.run(&mut w);
    assert_eq!(w.get_resource::<Counts>().unwrap().0, [1]);
}

struct Marker(u32);

fn spawn_marked(mut commands: Commands) {
    commands.spawn_bundle((Marker(1),));
}

fn count_added(q: Query<Entity, crate::query_data::Added<Marker>>, mut counts: ResMut<Counts>) {
    counts.0.push(q.iter().count());
}

#[test]
fn entities_made_by_deferred_work_count_as_added_to_the_systems_after_it_on_every_run() {
    // the work is applied with the tick the run has counted to, so a later
    // system whose last run was in the previous run still sees it
    let run = |kind: ExecutorKind| {
        let mut w = World::new();
        w.insert_resource(Counts(Vec::new()));
        let mut s = Schedule::new(Update);
        s.set_executor(kind).set_thread_pool(Some(pool(2)));
        s.add_systems((spawn_marked, count_added).chain());
        for _ in 0..4 {
            s.run(&mut w);
        }
        w.get_resource::<Counts>().unwrap().0.clone()
    };
    assert_eq!(run(ExecutorKind::SingleThreaded), [1, 1, 1, 1]);
    assert_eq!(run(ExecutorKind::MultiThreaded), [1, 1, 1, 1]);
}

#[test]
fn deferred_work_follows_the_run_order_not_the_order_systems_were_added_in() {
    for _ in 0..20 {
        let mut w = World::new();
        w.insert_resource(Applied(Vec::new()));
        let mut s = multi(4);
        // added a, b, c; run order is b, a, c (a waits for b)
        s.add_systems((defer_a.after(defer_b), defer_b, defer_c));
        s.run(&mut w);
        let applied = &w.get_resource::<Applied>().unwrap().0;
        let position = |name| applied.iter().position(|&n| n == name).unwrap();
        assert!(position("b") < position("a"), "{applied:?}");
    }
}

// ── exclusive systems ────────────────────────────────────────────────

#[test]
fn an_exclusive_system_never_overlaps_anything() {
    fn exclusive(world: &mut World) {
        let clock = world
            .get_resource::<Clock>()
            .unwrap()
            .0
            .fetch_add(1, SeqCst);
        let trace = world.get_resource::<Trace>().unwrap();
        assert_eq!(trace.active.load(SeqCst), 0, "something was running");
        trace.start[9].store(clock, SeqCst);
        for _ in 0..50 {
            std::thread::yield_now();
            assert_eq!(trace.active.load(SeqCst), 0, "something started meanwhile");
        }
        trace.runs[9].fetch_add(1, SeqCst);
    }
    for _ in 0..10 {
        let mut w = probe_world(10);
        let mut s = multi(4);
        for id in 0..9 {
            add_probe(&mut s, id, Some(id % 4), None);
        }
        s.add_systems(exclusive);
        s.run(&mut w);
        let t = w.get_resource::<Trace>().unwrap();
        assert_eq!(t.runs[9].load(SeqCst), 1);
        assert!((0..9).all(|i| t.runs[i].load(SeqCst) == 1));
    }
}

// ── conditions ───────────────────────────────────────────────────────

struct Switch(bool);
struct Evaluated(AtomicUsize);

fn is_on(s: Res<Switch>, e: Res<Evaluated>) -> bool {
    e.0.fetch_add(1, SeqCst);
    s.0
}

#[test]
fn a_false_condition_skips_the_system_and_the_systems_after_it_still_run() {
    let mut w = probe_world(3);
    w.insert_resource(Switch(false));
    w.insert_resource(Evaluated(AtomicUsize::new(0)));
    let mut s = multi(4);
    add_probe(&mut s, 0, None, Some(0));
    add_probe(&mut s, 1, None, Some(1));
    add_probe(&mut s, 2, None, Some(2));
    s.configure_sets((Node(0), Node(1), Node(2)).chain());
    s.configure_sets(Node(1).run_if(is_on));
    s.run(&mut w);
    let t = w.get_resource::<Trace>().unwrap();
    assert_eq!(
        [0, 1, 2].map(|i| t.runs[i].load(SeqCst)),
        [1, 0, 1],
        "the middle one was skipped and the third still ran"
    );
    assert_eq!(w.get_resource::<Evaluated>().unwrap().0.load(SeqCst), 1);
}

#[test]
fn a_set_condition_is_evaluated_once_however_many_members_race_for_it() {
    for _ in 0..20 {
        let mut w = probe_world(8);
        w.insert_resource(Switch(true));
        w.insert_resource(Evaluated(AtomicUsize::new(0)));
        let mut s = multi(4);
        for id in 0..8 {
            add_probe(&mut s, id, Some(id % 4), None);
            s.configure_sets(Node(id).in_set(Group(1)));
        }
        s.configure_sets(Group(1).run_if(is_on));
        s.run(&mut w);
        assert_eq!(w.get_resource::<Evaluated>().unwrap().0.load(SeqCst), 1);
        let t = w.get_resource::<Trace>().unwrap();
        assert!(t.runs.iter().take(8).all(|r| r.load(SeqCst) == 1));
    }
}

struct Verdict(bool, usize);

fn says_yes(v: Res<Verdict>, e: Res<Evaluated>) -> bool {
    e.0.fetch_add(1, SeqCst);
    v.1 == 0 || v.0
}

fn says_no(_v: Res<Verdict>, e: Res<Evaluated>) -> bool {
    e.0.fetch_add(1, SeqCst);
    false
}

#[test]
fn a_system_in_two_conditioned_sets_needs_both_and_both_are_asked() {
    // declared "no" then "yes", and "yes" then "no": the answer is the same
    for no_first in [true, false] {
        let mut w = probe_world(2);
        w.insert_resource(Verdict(true, 1));
        w.insert_resource(Evaluated(AtomicUsize::new(0)));
        let mut s = multi(4);
        add_probe(&mut s, 0, None, Some(0));
        add_probe(&mut s, 1, None, Some(1));
        if no_first {
            s.configure_sets(Group(1).run_if(says_no));
            s.configure_sets(Group(2).run_if(says_yes));
        } else {
            s.configure_sets(Group(2).run_if(says_yes));
            s.configure_sets(Group(1).run_if(says_no));
        }
        s.configure_sets(Node(0).in_set(Group(1)).in_set(Group(2)));
        s.run(&mut w);
        let t = w.get_resource::<Trace>().unwrap();
        assert_eq!(
            t.runs[0].load(SeqCst),
            0,
            "no_first = {no_first}: turned away"
        );
        assert_eq!(t.runs[1].load(SeqCst), 1, "the other system is not in them");
        assert_eq!(
            w.get_resource::<Evaluated>().unwrap().0.load(SeqCst),
            2,
            "both asked"
        );
    }
}

#[test]
fn a_system_turned_away_by_its_set_does_not_ask_its_own_conditions() {
    let mut w = probe_world(1);
    w.insert_resource(Verdict(true, 1));
    w.insert_resource(Evaluated(AtomicUsize::new(0)));
    let mut s = multi(4);
    s.add_systems(other.in_set(Group(1)).run_if(says_yes));
    s.configure_sets(Group(1).run_if(says_no));
    s.run(&mut w);
    assert_eq!(
        w.get_resource::<Evaluated>().unwrap().0.load(SeqCst),
        1,
        "only the set was asked"
    );
    // and let back in, it is asked
    let mut w = probe_world(1);
    w.insert_resource(Verdict(true, 1));
    w.insert_resource(Evaluated(AtomicUsize::new(0)));
    let mut s = multi(4);
    s.add_systems(other.in_set(Group(1)).run_if(says_yes));
    s.configure_sets(Group(1).run_if(says_yes));
    s.run(&mut w);
    assert_eq!(w.get_resource::<Evaluated>().unwrap().0.load(SeqCst), 2);
}

struct Gate(AtomicUsize);
struct Seen(AtomicUsize);

fn slow_writer(clock: Res<Clock>, mut gate: ResMut<Gate>, trace: Res<Trace>) {
    trace.start[0].store(clock.0.fetch_add(1, SeqCst), SeqCst);
    for _ in 0..200 {
        std::thread::yield_now();
    }
    gate.0.store(7, SeqCst);
    trace.end[0].store(clock.0.fetch_add(1, SeqCst), SeqCst);
}

fn reads_gate(g: Res<Gate>, clock: Res<Clock>, seen: Res<Seen>) -> bool {
    seen.0.store(clock.0.fetch_add(1, SeqCst), SeqCst);
    g.0.load(SeqCst) == 7
}

fn other(_c: Res<Clock>) {}

#[test]
fn a_condition_does_not_read_what_a_running_system_is_writing() {
    for _ in 0..10 {
        let mut w = probe_world(2);
        w.insert_resource(Gate(AtomicUsize::new(0)));
        w.insert_resource(Seen(AtomicUsize::new(0)));
        let mut s = multi(4);
        s.add_systems((slow_writer, other.run_if(reads_gate)));
        s.run(&mut w);
        let t = w.get_resource::<Trace>().unwrap();
        let seen = w.get_resource::<Seen>().unwrap().0.load(SeqCst);
        assert!(
            seen > t.end[0].load(SeqCst),
            "the condition read ({seen}) before the writer was done ({})",
            t.end[0].load(SeqCst)
        );
    }
}

fn reads_gate_group(g: Res<Gate>, clock: Res<Clock>, seen: Res<Seen>) -> bool {
    seen.0.store(clock.0.fetch_add(1, SeqCst), SeqCst);
    g.0.load(SeqCst) == 7
}

#[test]
fn a_set_condition_does_not_read_what_a_running_system_is_writing() {
    for _ in 0..10 {
        let mut w = probe_world(2);
        w.insert_resource(Gate(AtomicUsize::new(0)));
        w.insert_resource(Seen(AtomicUsize::new(0)));
        let mut s = multi(4);
        s.add_systems((slow_writer, other.in_set(Group(3))));
        s.configure_sets(Group(3).run_if(reads_gate_group));
        s.run(&mut w);
        let t = w.get_resource::<Trace>().unwrap();
        let seen = w.get_resource::<Seen>().unwrap().0.load(SeqCst);
        assert!(
            seen > t.end[0].load(SeqCst),
            "the condition read ({seen}) before the writer was done ({})",
            t.end[0].load(SeqCst)
        );
    }
}

fn touch_gauge_exclusive(world: &mut World) {
    world.get_resource_mut::<Gauge>().unwrap().0 += 1;
}

#[test]
fn what_an_exclusive_system_writes_counts_as_changed_to_the_systems_after_it() {
    let run = |kind: ExecutorKind| {
        let mut w = World::new();
        w.insert_resource(Gauge(0));
        w.insert_resource(Observed(Vec::new()));
        let mut s = Schedule::new(Update);
        s.set_executor(kind).set_thread_pool(Some(pool(2)));
        s.add_systems((touch_gauge_exclusive, observe).chain());
        for _ in 0..4 {
            s.run(&mut w);
        }
        w.get_resource::<Observed>().unwrap().0.clone()
    };
    assert_eq!(run(ExecutorKind::SingleThreaded), [true; 4]);
    assert_eq!(run(ExecutorKind::MultiThreaded), [true; 4]);
}

#[test]
fn what_an_exclusive_system_writes_is_seen_by_a_system_that_ran_before_it_on_the_next_run() {
    // the reader's last run is later than any stale tick, so an exclusive
    // system that stamped its write with the world's old tick would be missed
    let run = |kind: ExecutorKind| {
        let mut w = World::new();
        w.insert_resource(Gauge(0));
        w.insert_resource(Observed(Vec::new()));
        let mut s = Schedule::new(Update);
        s.set_executor(kind).set_thread_pool(Some(pool(2)));
        s.add_systems((observe, touch_gauge_exclusive).chain());
        for _ in 0..4 {
            s.run(&mut w);
        }
        w.get_resource::<Observed>().unwrap().0.clone()
    };
    assert_eq!(run(ExecutorKind::SingleThreaded), [true; 4]);
    assert_eq!(run(ExecutorKind::MultiThreaded), [true; 4]);
}

#[test]
fn what_deferred_work_changes_is_seen_by_a_system_that_ran_before_it_on_the_next_run() {
    struct Level(u32);
    fn bump_level(mut commands: Commands) {
        commands.queue(|w: &mut World| w.get_resource_mut::<Level>().unwrap().0 += 1);
    }
    fn watch_level(l: Res<Level>, mut o: ResMut<Observed>) {
        o.0.push(l.is_changed());
    }
    let run = |kind: ExecutorKind| {
        let mut w = World::new();
        w.insert_resource(Level(0));
        w.insert_resource(Observed(Vec::new()));
        let mut s = Schedule::new(Update);
        s.set_executor(kind).set_thread_pool(Some(pool(2)));
        s.add_systems((watch_level, bump_level).chain());
        for _ in 0..4 {
            s.run(&mut w);
        }
        w.get_resource::<Observed>().unwrap().0.clone()
    };
    assert_eq!(run(ExecutorKind::SingleThreaded), [true; 4]);
    assert_eq!(run(ExecutorKind::MultiThreaded), [true; 4]);
}

// ── ticks ────────────────────────────────────────────────────────────

struct Gauge(u32);
struct Observed(Vec<bool>);

fn bump(mut g: ResMut<Gauge>) {
    g.0 += 1;
}

fn observe(g: Res<Gauge>, mut o: ResMut<Observed>) {
    o.0.push(g.is_changed());
}

fn every_other(mut n: crate::system_param::Local<u32>) -> bool {
    *n += 1;
    *n % 2 == 1
}

#[test]
fn change_detection_across_ordered_systems_matches_the_single_threaded_executor() {
    let run = |kind: ExecutorKind| {
        let mut w = World::new();
        w.insert_resource(Gauge(0));
        w.insert_resource(Observed(Vec::new()));
        let mut s = Schedule::new(Update);
        s.set_executor(kind).set_thread_pool(Some(pool(4)));
        s.add_systems((bump.run_if(every_other), observe, observe).chain());
        for _ in 0..6 {
            s.run(&mut w);
        }
        (
            w.get_resource::<Observed>().unwrap().0.clone(),
            w.change_tick(),
        )
    };
    let (single_seen, single_tick) = run(ExecutorKind::SingleThreaded);
    let (multi_seen, multi_tick) = run(ExecutorKind::MultiThreaded);
    assert_eq!(single_seen, multi_seen);
    assert_eq!(single_tick, multi_tick, "the same ticks were handed out");
    assert!(single_seen.iter().any(|&b| b) && single_seen.iter().any(|&b| !b));
}

#[test]
fn the_world_tick_advances_by_one_per_condition_and_per_system() {
    let mut w = probe_world(3);
    w.insert_resource(Switch(true));
    w.insert_resource(Evaluated(AtomicUsize::new(0)));
    let mut s = multi(4);
    add_probe(&mut s, 0, None, Some(0));
    add_probe(&mut s, 1, None, Some(1));
    add_probe(&mut s, 2, None, Some(2));
    s.configure_sets(Node(1).run_if(is_on));
    let before = w.change_tick().get();
    s.run(&mut w);
    assert_eq!(
        w.change_tick().get(),
        before + 4,
        "three systems and one condition"
    );
}

#[test]
fn stored_ticks_are_swept_after_a_parallel_run() {
    use crate::tick::MAX_CHANGE_AGE;
    let mut w = probe_world(2);
    w.insert_resource(Switch(false));
    w.insert_resource(Evaluated(AtomicUsize::new(0)));
    let mut s = multi(2);
    add_probe(&mut s, 0, None, Some(0));
    add_probe(&mut s, 1, None, Some(1));
    s.configure_sets(Node(1).run_if(is_on));
    s.run(&mut w);
    w.set_change_tick(Tick::new(MAX_CHANGE_AGE + 900));
    s.run(&mut w);
    let now = w.change_tick();
    let age = now.relative_to(s.oldest_last_run(now).unwrap()).get();
    assert!(age <= MAX_CHANGE_AGE, "{age}");
}

#[test]
fn the_sweep_reaches_conditions_that_were_not_asked_in_a_parallel_run() {
    use crate::tick::MAX_CHANGE_AGE;
    // a system turned away by its set never consults its own conditions, and
    // a set with no members is never evaluated: both sit while the tick runs on
    let mut w = probe_world(2);
    w.insert_resource(Verdict(true, 1));
    w.insert_resource(Evaluated(AtomicUsize::new(0)));
    let mut s = multi(2);
    s.add_systems(other.in_set(Group(1)).run_if(says_yes));
    s.configure_sets(Group(1).run_if(says_no));
    s.configure_sets(Group(2).run_if(says_yes));
    add_probe(&mut s, 1, None, Some(1));
    s.run(&mut w);
    w.set_change_tick(Tick::new(MAX_CHANGE_AGE + 900));
    s.run(&mut w);
    let now = w.change_tick();
    let age = now.relative_to(s.oldest_last_run(now).unwrap()).get();
    assert!(age <= MAX_CHANGE_AGE, "{age} > {MAX_CHANGE_AGE}");
}

// ── panics ───────────────────────────────────────────────────────────

struct Armed(AtomicBool);

fn boom(a: Res<Armed>) {
    if a.0.load(SeqCst) {
        panic!("a system failed");
    }
}

#[test]
fn a_panicking_system_lets_the_others_finish_and_the_schedule_run_again() {
    let mut w = probe_world(4);
    w.insert_resource(Armed(AtomicBool::new(true)));
    let mut s = multi(4);
    s.add_systems(boom);
    for id in 0..4 {
        add_probe(&mut s, id, Some(id % 4), None);
    }
    let before = w.change_tick().get();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)));
    let payload = result.expect_err("the panic must come out");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"a system failed"));
    assert!(
        w.change_tick().get() > before,
        "the ticks handed out were kept"
    );
    w.get_resource::<Armed>().unwrap().0.store(false, SeqCst);
    for r in w.get_resource::<Trace>().unwrap().runs.iter() {
        r.store(0, SeqCst);
    }
    s.run(&mut w);
    let t = w.get_resource::<Trace>().unwrap();
    assert!(
        t.runs.iter().all(|r| r.load(SeqCst) == 1),
        "everything runs again"
    );
}

fn slow_x(t: Res<Trace>) {
    for _ in 0..3000 {
        std::thread::yield_now();
    }
    t.runs[0].fetch_add(1, SeqCst);
}

fn y_after_x(t: Res<Trace>) {
    t.runs[1].fetch_add(1, SeqCst);
}

#[test]
fn nothing_new_starts_once_a_system_has_panicked() {
    let mut w = probe_world(2);
    w.insert_resource(Armed(AtomicBool::new(true)));
    let mut s = multi(4);
    s.add_systems((slow_x, y_after_x.after(slow_x), boom));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)));
    assert!(r.is_err());
    let t = w.get_resource::<Trace>().unwrap();
    assert_eq!(t.runs[0].load(SeqCst), 1, "what was running finished");
    assert_eq!(
        t.runs[1].load(SeqCst),
        0,
        "what was waiting for it did not start"
    );
}

fn slow_writer_x(t: Res<Trace>, mut c: ResMut<Counts>) {
    for _ in 0..3000 {
        std::thread::yield_now();
    }
    c.0.push(0);
    t.runs[0].fetch_add(1, SeqCst);
}

fn writer_y(t: Res<Trace>, mut c: ResMut<Counts>) {
    c.0.push(1);
    t.runs[1].fetch_add(1, SeqCst);
}

#[test]
fn nothing_that_was_waiting_for_room_starts_once_a_system_has_panicked() {
    // y conflicts with x, so it waits for x to finish; x is still running
    // when boom fails, and y must not be let in when x is done
    let mut w = probe_world(2);
    w.insert_resource(Counts(Vec::new()));
    w.insert_resource(Armed(AtomicBool::new(true)));
    let mut s = multi(4);
    s.add_systems((slow_writer_x, writer_y, boom));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)));
    assert!(r.is_err());
    let t = w.get_resource::<Trace>().unwrap();
    assert_eq!(t.runs[0].load(SeqCst), 1, "what was running finished");
    assert_eq!(
        t.runs[1].load(SeqCst),
        0,
        "what was waiting for room did not start"
    );
}

#[test]
fn a_panicking_condition_is_reported_the_same_way() {
    fn bad(_c: Res<Clock>) -> bool {
        panic!("a condition failed");
    }
    let mut w = probe_world(1);
    let mut s = multi(2);
    s.add_systems(other.run_if(bad));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)));
    assert_eq!(
        r.expect_err("it must panic").downcast_ref::<&str>(),
        Some(&"a condition failed")
    );
}

#[test]
fn a_panic_in_deferred_work_comes_out_of_run_too() {
    fn queue_a_panic(mut commands: Commands) {
        commands.queue(|_w: &mut World| panic!("a command failed"));
    }
    let mut w = World::new();
    let mut s = multi(2);
    s.add_systems(queue_a_panic);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)));
    assert_eq!(
        r.expect_err("it must panic").downcast_ref::<&str>(),
        Some(&"a command failed")
    );
}

#[test]
fn a_panic_in_an_exclusive_system_comes_out_of_run_too() {
    fn exclusive_boom(_w: &mut World) {
        panic!("an exclusive system failed");
    }
    let mut w = probe_world(1);
    let mut s = multi(2);
    s.add_systems(exclusive_boom);
    add_probe(&mut s, 0, Some(0), None);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)));
    assert_eq!(
        r.expect_err("it must panic").downcast_ref::<&str>(),
        Some(&"an exclusive system failed")
    );
}

// ── the executor setting ─────────────────────────────────────────────

#[test]
fn the_default_is_single_threaded_and_switching_rebuilds() {
    let mut w = probe_world(2);
    let mut s = Schedule::new(Update);
    assert_eq!(s.executor(), ExecutorKind::SingleThreaded);
    add_probe(&mut s, 0, None, Some(0));
    add_probe(&mut s, 1, None, Some(1));
    s.run(&mut w);
    s.set_executor(ExecutorKind::MultiThreaded);
    assert_eq!(s.executor(), ExecutorKind::MultiThreaded);
    assert!(s.run_order().is_none(), "switching discards the build");
    s.run(&mut w);
    s.set_executor(ExecutorKind::SingleThreaded);
    s.run(&mut w);
    let t = w.get_resource::<Trace>().unwrap();
    assert!(t.runs.iter().all(|r| r.load(SeqCst) == 3));
}

#[test]
fn an_empty_multi_threaded_schedule_runs() {
    let mut w = World::new();
    let mut s = multi(2);
    s.run(&mut w);
}

#[test]
fn a_multi_threaded_schedule_is_still_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Schedule>();
}

#[test]
fn a_few_hundred_systems_in_phases_run_in_parallel_many_times() {
    let n = 240;
    let mut w = probe_world(n);
    let mut s = multi(4);
    s.configure_sets((Group(0), Group(1), Group(2)).chain());
    for id in 0..n {
        add_probe(
            &mut s,
            id,
            Some(id % 4),
            if id % 3 == 0 {
                Some((id + 1) % 4)
            } else {
                None
            },
        );
        s.configure_sets(Node(id).in_set(Group((id % 3) as u32)));
    }
    for _ in 0..5 {
        w.get_resource::<Trace>().unwrap().reset();
        s.run(&mut w);
        let t = w.get_resource::<Trace>().unwrap();
        // phase order holds
        for id in 0..n {
            for other in 0..n {
                if id % 3 < other % 3 {
                    assert!(t.end[id].load(SeqCst) < t.start[other].load(SeqCst));
                }
            }
        }
    }
}
