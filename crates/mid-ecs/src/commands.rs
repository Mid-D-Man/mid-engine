//! Deferred structural changes: queue them while a query borrows the
//! [`World`], apply them afterwards.
//!
//! A query holds `&World` for as long as its iterator lives, so nothing
//! that spawns, despawns, inserts or removes can run inside that loop.
//! [`CommandQueue`] records those changes instead and
//! [`CommandQueue::apply`] runs them, in the order they were queued,
//! once the borrow has ended:
//!
//! ```
//! use mid_ecs::{CommandQueue, World};
//!
//! struct Health(i32);
//!
//! let mut world = World::new();
//! let doomed = world.spawn();
//! world.insert_static(doomed, Health(0));
//!
//! let mut queue = CommandQueue::new();
//! {
//!     let mut commands = queue.commands();
//!     for (entity, health) in world.query_static::<Health>() {
//!         if health.0 <= 0 {
//!             commands.despawn(entity);
//!         }
//!     }
//! }
//! queue.apply(&mut world);
//! assert!(!world.is_alive(doomed));
//! ```
//!
//! Design decisions and their history live in `docs/mid-ecs.md`
//! ("Commands"); this comment only says what the code does.
//!
//! - **Order.** Commands run first-queued-first.
//! - **Semantics.** Each command does exactly what the `World` method it
//!   names does when called at that point, no more. A command aimed at an
//!   entity that is no longer alive does nothing, and `insert_static` of a
//!   component the entity already has is a no-op (the first value stays).
//! - **Spawning.** [`Commands::spawn`] cannot hand back a real [`Entity`]:
//!   nothing is allocated until `apply`. It returns an [`EntityCommands`]
//!   that targets the entity the spawn will produce, so
//!   `commands.spawn().insert_static(a).insert_static(b)` works. The real
//!   entities come back from [`CommandQueue::apply`], in spawn order.
//! - **Generic escape hatch.** [`Commands::queue`] and
//!   [`EntityCommands::queue`] take any closure, which is how Sparse Shell
//!   components (`World::insert`) and anything else without a dedicated
//!   method are queued.
//! - **Cost.** Removals and despawns store no payload. Inserts and
//!   bundles box one closure each (the boxed value is the component), so
//!   an insert allocates.
//! - **`Send`.** Not required of the stored closures yet: `World` itself
//!   does not require `Send` of its components.

use crate::archetype::Bundle;
use crate::world::{Entity, World};

/// What a queued command acts on.
#[derive(Clone, Copy)]
enum Target {
    /// An entity that already exists.
    Entity(Entity),
    /// The `n`th entity spawned by this queue since it was last applied
    /// or cleared. Only ever issued by a spawn queued earlier, so it
    /// always resolves.
    Slot(usize),
}

enum Command {
    /// Spawns an empty entity; pushes it onto the slot list.
    Spawn,
    /// Spawns an entity with a bundle; pushes it onto the slot list.
    SpawnWith(Box<dyn FnOnce(&mut World) -> Entity>),
    Despawn(Target),
    /// Runs against one entity (inserts, custom per-entity commands).
    OnEntity(Target, Box<dyn FnOnce(&mut World, Entity)>),
    /// A removal: a monomorphised function pointer, so no allocation.
    Remove(Target, fn(&mut World, Entity)),
    OnWorld(Box<dyn FnOnce(&mut World)>),
}

/// An ordered buffer of deferred [`World`] changes. See the module docs.
#[derive(Default)]
pub struct CommandQueue {
    commands: Vec<Command>,
    /// Spawns queued so far; the next spawn's slot index.
    spawns: usize,
}

impl CommandQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty queue with room for `capacity` commands before it grows.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            spawns: 0,
        }
    }

    /// Number of commands waiting to be applied.
    #[inline]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether no commands are waiting.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Number of spawn commands waiting, which is exactly how many
    /// entities the next [`CommandQueue::apply`] will return.
    #[inline]
    pub fn spawn_count(&self) -> usize {
        self.spawns
    }

    /// Drops every waiting command without running it.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.spawns = 0;
    }

    /// The recording interface onto this queue.
    #[inline]
    pub fn commands(&mut self) -> Commands<'_> {
        Commands { queue: self }
    }

    /// Runs every waiting command against `world`, first-queued first,
    /// and leaves the queue empty (capacity kept) and reusable.
    ///
    /// Returns the entities spawned by this call, in the order their
    /// spawn commands ran. The vector allocates only if something was
    /// spawned.
    ///
    /// If a command panics, the commands after it are dropped unrun, the
    /// queue is left empty and reusable, and the panic propagates.
    pub fn apply(&mut self, world: &mut World) -> Vec<Entity> {
        let mut spawned = Vec::with_capacity(self.spawns);
        self.spawns = 0;
        for command in self.commands.drain(..) {
            match command {
                Command::Spawn => spawned.push(world.spawn()),
                Command::SpawnWith(spawn) => spawned.push(spawn(world)),
                Command::Despawn(target) => {
                    world.despawn(resolve(target, &spawned));
                }
                Command::OnEntity(target, run) => run(world, resolve(target, &spawned)),
                Command::Remove(target, run) => run(world, resolve(target, &spawned)),
                Command::OnWorld(run) => run(world),
            }
        }
        spawned
    }

    fn push(&mut self, command: Command) {
        self.commands.push(command);
    }

    /// Queues a spawn-type command and returns the slot it will fill.
    fn push_spawn(&mut self, command: Command) -> Target {
        let slot = self.spawns;
        self.spawns += 1;
        self.commands.push(command);
        Target::Slot(slot)
    }
}

impl std::fmt::Debug for CommandQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandQueue")
            .field("len", &self.commands.len())
            .finish()
    }
}

#[inline]
fn resolve(target: Target, spawned: &[Entity]) -> Entity {
    match target {
        Target::Entity(entity) => entity,
        Target::Slot(slot) => spawned[slot],
    }
}

/// Records [`World`] changes onto a [`CommandQueue`]. Get one from
/// [`CommandQueue::commands`].
pub struct Commands<'q> {
    queue: &'q mut CommandQueue,
}

impl Commands<'_> {
    /// Queues a spawn. The returned [`EntityCommands`] targets the entity
    /// that spawn will produce.
    pub fn spawn(&mut self) -> EntityCommands<'_> {
        let target = self.queue.push_spawn(Command::Spawn);
        EntityCommands {
            queue: &mut *self.queue,
            target,
        }
    }

    /// Queues a spawn that attaches `bundle` in the same step
    /// ([`World::spawn_bundle`]).
    #[allow(private_bounds)]
    pub fn spawn_bundle<B: Bundle>(&mut self, bundle: B) -> EntityCommands<'_> {
        let target = self
            .queue
            .push_spawn(Command::SpawnWith(Box::new(move |world| {
                world.spawn_bundle(bundle)
            })));
        EntityCommands {
            queue: &mut *self.queue,
            target,
        }
    }

    /// Targets an entity that already exists.
    pub fn entity(&mut self, entity: Entity) -> EntityCommands<'_> {
        EntityCommands {
            queue: &mut *self.queue,
            target: Target::Entity(entity),
        }
    }

    /// Queues [`World::despawn`].
    pub fn despawn(&mut self, entity: Entity) {
        self.queue.push(Command::Despawn(Target::Entity(entity)));
    }

    /// Queues [`World::insert_resource`].
    pub fn insert_resource<T: 'static>(&mut self, value: T) {
        self.queue.push(Command::OnWorld(Box::new(move |world| {
            world.insert_resource(value);
        })));
    }

    /// Queues [`World::remove_resource`].
    pub fn remove_resource<T: 'static>(&mut self) {
        self.queue.push(Command::OnWorld(Box::new(|world| {
            world.remove_resource::<T>();
        })));
    }

    /// Queues any closure over the whole world. The way to queue Sparse
    /// Shell inserts (`World::insert`) or anything else without a
    /// dedicated method.
    pub fn queue(&mut self, command: impl FnOnce(&mut World) + 'static) {
        self.queue.push(Command::OnWorld(Box::new(command)));
    }
}

/// Records changes to one entity: either one that exists
/// ([`Commands::entity`]) or the one a queued spawn will produce
/// ([`Commands::spawn`]). Methods chain.
pub struct EntityCommands<'q> {
    queue: &'q mut CommandQueue,
    target: Target,
}

impl EntityCommands<'_> {
    /// Queues [`World::insert_static`].
    pub fn insert_static<T: 'static>(&mut self, component: T) -> &mut Self {
        self.queue.push(Command::OnEntity(
            self.target,
            Box::new(move |world, entity| {
                world.insert_static(entity, component);
            }),
        ));
        self
    }

    /// Queues [`World::remove_static`].
    pub fn remove_static<T: 'static>(&mut self) -> &mut Self {
        self.queue
            .push(Command::Remove(self.target, |world, entity| {
                world.remove_static::<T>(entity);
            }));
        self
    }

    /// Queues [`World::insert_bundle`].
    #[allow(private_bounds)]
    pub fn insert_bundle<B: Bundle>(&mut self, bundle: B) -> &mut Self {
        self.queue.push(Command::OnEntity(
            self.target,
            Box::new(move |world, entity| {
                world.insert_bundle(entity, bundle);
            }),
        ));
        self
    }

    /// Queues [`World::remove_bundle`].
    #[allow(private_bounds)]
    pub fn remove_bundle<B: Bundle>(&mut self) -> &mut Self {
        self.queue
            .push(Command::Remove(self.target, |world, entity| {
                world.remove_bundle::<B>(entity);
            }));
        self
    }

    /// Queues any closure over the world and this entity. The way to
    /// queue Sparse Shell inserts and removals for it.
    pub fn queue(&mut self, command: impl FnOnce(&mut World, Entity) + 'static) -> &mut Self {
        self.queue
            .push(Command::OnEntity(self.target, Box::new(command)));
        self
    }

    /// Queues [`World::despawn`] for this entity.
    pub fn despawn(self) {
        self.queue.push(Command::Despawn(self.target));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Pos(f32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Vel(f32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Health(i32);
    #[derive(Debug, PartialEq)]
    struct Score(u32);
    #[derive(Debug, PartialEq)]
    struct Tag(&'static str);

    #[test]
    fn new_queue_is_empty() {
        let q = CommandQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn spawn_count_matches_what_apply_returns() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        assert_eq!(q.spawn_count(), 0);
        {
            let mut c = q.commands();
            c.spawn();
            c.spawn_bundle((Pos(1.0), Vel(1.0)));
            c.despawn(world.spawn());
            c.insert_resource(Score(1));
        }
        assert_eq!(q.len(), 4);
        assert_eq!(q.spawn_count(), 2, "only spawns count, not other commands");
        let spawned = q.apply(&mut world);
        assert_eq!(spawned.len(), 2);
        assert_eq!(q.spawn_count(), 0);
    }

    #[test]
    fn spawn_count_resets_on_clear() {
        let mut q = CommandQueue::new();
        q.commands().spawn();
        q.clear();
        assert_eq!(q.spawn_count(), 0);
    }

    #[test]
    fn nothing_happens_until_apply() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands().spawn().insert_static(Pos(1.0));
        assert_eq!(world.entity_count(), 0);
        assert_eq!(q.len(), 2);
        q.apply(&mut world);
        assert_eq!(world.entity_count(), 1);
    }

    #[test]
    fn spawn_returns_the_real_entities_in_order() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.spawn().insert_static(Pos(1.0));
            c.spawn().insert_static(Pos(2.0));
            c.spawn().insert_static(Pos(3.0));
        }
        let spawned = q.apply(&mut world);
        assert_eq!(spawned.len(), 3);
        assert_eq!(world.get_static::<Pos>(spawned[0]), Some(&Pos(1.0)));
        assert_eq!(world.get_static::<Pos>(spawned[1]), Some(&Pos(2.0)));
        assert_eq!(world.get_static::<Pos>(spawned[2]), Some(&Pos(3.0)));
    }

    #[test]
    fn apply_with_no_spawns_returns_an_empty_vec() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands().insert_resource(Score(1));
        assert!(q.apply(&mut world).is_empty());
    }

    #[test]
    fn chained_inserts_reach_the_spawned_entity() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands()
            .spawn()
            .insert_static(Pos(1.0))
            .insert_static(Vel(2.0))
            .insert_static(Health(3));
        let e = q.apply(&mut world)[0];
        assert_eq!(world.get_static::<Pos>(e), Some(&Pos(1.0)));
        assert_eq!(world.get_static::<Vel>(e), Some(&Vel(2.0)));
        assert_eq!(world.get_static::<Health>(e), Some(&Health(3)));
    }

    #[test]
    fn interleaved_spawns_each_get_their_own_inserts() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            let mut a = c.spawn();
            a.insert_static(Health(1));
            drop(a);
            let mut b = c.spawn();
            b.insert_static(Health(2));
            drop(b);
        }
        let s = q.apply(&mut world);
        assert_eq!(world.get_static::<Health>(s[0]), Some(&Health(1)));
        assert_eq!(world.get_static::<Health>(s[1]), Some(&Health(2)));
    }

    #[test]
    fn spawn_bundle_attaches_every_component() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands().spawn_bundle((Pos(1.0), Vel(2.0)));
        let e = q.apply(&mut world)[0];
        assert_eq!(world.get_static::<Pos>(e), Some(&Pos(1.0)));
        assert_eq!(world.get_static::<Vel>(e), Some(&Vel(2.0)));
    }

    #[test]
    fn spawn_bundle_target_accepts_further_commands() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands()
            .spawn_bundle((Pos(1.0), Vel(2.0)))
            .insert_static(Health(9));
        let e = q.apply(&mut world)[0];
        assert_eq!(world.get_static::<Health>(e), Some(&Health(9)));
        assert_eq!(world.get_static::<Pos>(e), Some(&Pos(1.0)));
    }

    #[test]
    fn insert_and_remove_on_an_existing_entity() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert_static(e, Pos(1.0));
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.entity(e).insert_static(Vel(5.0)).remove_static::<Pos>();
        }
        q.apply(&mut world);
        assert!(!world.has_static::<Pos>(e));
        assert_eq!(world.get_static::<Vel>(e), Some(&Vel(5.0)));
    }

    #[test]
    fn bundle_insert_and_remove_on_an_existing_entity() {
        let mut world = World::new();
        let e = world.spawn();
        let mut q = CommandQueue::new();
        q.commands().entity(e).insert_bundle((Pos(1.0), Vel(2.0)));
        q.apply(&mut world);
        assert!(world.has_static::<Pos>(e) && world.has_static::<Vel>(e));
        q.commands().entity(e).remove_bundle::<(Pos, Vel)>();
        q.apply(&mut world);
        assert!(!world.has_static::<Pos>(e) && !world.has_static::<Vel>(e));
    }

    #[test]
    fn despawn_an_existing_entity() {
        let mut world = World::new();
        let e = world.spawn();
        let mut q = CommandQueue::new();
        q.commands().despawn(e);
        assert!(world.is_alive(e));
        q.apply(&mut world);
        assert!(!world.is_alive(e));
    }

    #[test]
    fn entity_commands_despawn_works_on_existing_and_spawned_targets() {
        let mut world = World::new();
        let e = world.spawn();
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.entity(e).despawn();
            c.spawn().despawn();
        }
        let spawned = q.apply(&mut world);
        assert!(!world.is_alive(e));
        assert!(!world.is_alive(spawned[0]));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn commands_run_first_queued_first() {
        let mut world = World::new();
        let e = world.spawn();
        let mut q = CommandQueue::new();
        // Insert then remove: gone.
        q.commands()
            .entity(e)
            .insert_static(Pos(1.0))
            .remove_static::<Pos>();
        q.apply(&mut world);
        assert!(!world.has_static::<Pos>(e));
        // Remove then insert: present.
        q.commands()
            .entity(e)
            .remove_static::<Pos>()
            .insert_static(Pos(2.0));
        q.apply(&mut world);
        assert_eq!(world.get_static::<Pos>(e), Some(&Pos(2.0)));
    }

    #[test]
    fn inserting_a_component_the_entity_already_has_keeps_the_first_value() {
        // Each command has exactly the semantics of the World method it
        // queues, and World::insert_static is a no-op on a component the
        // entity already carries.
        let mut world = World::new();
        let e = world.spawn();
        let mut q = CommandQueue::new();
        q.commands()
            .entity(e)
            .insert_static(Health(1))
            .insert_static(Health(2));
        q.apply(&mut world);
        assert_eq!(world.get_static::<Health>(e), Some(&Health(1)));
    }

    #[test]
    fn commands_on_a_dead_entity_do_nothing() {
        let mut world = World::new();
        let e = world.spawn();
        world.despawn(e);
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.entity(e)
                .insert_static(Pos(1.0))
                .insert_bundle((Vel(1.0), Health(1)))
                .remove_static::<Pos>();
            c.despawn(e);
        }
        q.apply(&mut world);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn a_command_after_an_earlier_despawn_does_nothing() {
        let mut world = World::new();
        let e = world.spawn();
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.despawn(e);
            c.entity(e).insert_static(Pos(1.0));
        }
        q.apply(&mut world);
        assert!(!world.is_alive(e));
        // The recycled slot must not have picked up the dead entity's insert.
        let fresh = world.spawn();
        assert!(!world.has_static::<Pos>(fresh));
    }

    #[test]
    fn despawn_while_iterating_a_query() {
        let mut world = World::new();
        let mut alive = Vec::new();
        for hp in [5, 0, 3, -1, 8, 0] {
            let e = world.spawn();
            world.insert_static(e, Health(hp));
            if hp > 0 {
                alive.push(e);
            }
        }
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            for (e, h) in world.query_static::<Health>() {
                if h.0 <= 0 {
                    c.despawn(e);
                }
            }
        }
        assert_eq!(world.entity_count(), 6);
        q.apply(&mut world);
        assert_eq!(world.entity_count(), 3);
        for e in alive {
            assert!(world.is_alive(e));
        }
    }

    #[test]
    fn spawn_while_iterating_a_query() {
        let mut world = World::new();
        for i in 0..4 {
            let e = world.spawn();
            world.insert_static(e, Pos(i as f32));
        }
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            for (_, p) in world.query_static::<Pos>() {
                c.spawn().insert_static(Vel(p.0 * 10.0));
            }
        }
        q.apply(&mut world);
        let mut vels: Vec<f32> = world.query_static::<Vel>().map(|(_, v)| v.0).collect();
        vels.sort_by(f32::total_cmp);
        assert_eq!(vels, vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn apply_leaves_the_queue_empty_and_reusable() {
        let mut world = World::new();
        let mut q = CommandQueue::with_capacity(16);
        q.commands().spawn().insert_static(Health(1));
        q.apply(&mut world);
        assert!(q.is_empty());
        q.commands().spawn().insert_static(Health(2));
        let second = q.apply(&mut world);
        assert_eq!(second.len(), 1);
        assert_eq!(world.get_static::<Health>(second[0]), Some(&Health(2)));
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn slot_numbering_restarts_after_apply() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands().spawn();
        q.commands().spawn();
        let first = q.apply(&mut world);
        // A fresh spawn after apply is slot 0 again; its insert must land
        // on the new entity, not on either entity from the first apply.
        q.commands().spawn().insert_static(Health(7));
        let second = q.apply(&mut world);
        assert_eq!(world.get_static::<Health>(second[0]), Some(&Health(7)));
        assert!(!world.has_static::<Health>(first[0]));
        assert!(!world.has_static::<Health>(first[1]));
    }

    #[test]
    fn clear_drops_pending_commands_unrun_and_restarts_slot_numbering() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands().spawn().insert_static(Pos(1.0));
        q.clear();
        assert!(q.is_empty());
        // Queued straight after clear, with no apply in between: the new
        // spawn must be slot 0 again, and the cleared spawn must not run.
        q.commands().spawn().insert_static(Pos(2.0));
        let spawned = q.apply(&mut world);
        assert_eq!(spawned.len(), 1);
        assert_eq!(world.entity_count(), 1);
        assert_eq!(world.get_static::<Pos>(spawned[0]), Some(&Pos(2.0)));
    }

    #[test]
    fn dropping_a_queue_does_not_apply_it() {
        let mut world = World::new();
        {
            let mut q = CommandQueue::new();
            q.commands().spawn();
        }
        assert_eq!(world.entity_count(), 0);
        world.spawn();
        assert_eq!(world.entity_count(), 1);
    }

    #[test]
    fn resource_commands() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands().insert_resource(Score(10));
        q.apply(&mut world);
        assert_eq!(world.get_resource::<Score>(), Some(&Score(10)));
        q.commands().remove_resource::<Score>();
        q.apply(&mut world);
        assert_eq!(world.get_resource::<Score>(), None);
    }

    #[test]
    fn custom_world_command_sees_earlier_effects() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.insert_resource(Score(1));
            c.queue(|w| {
                let s = w.get_resource::<Score>().map_or(0, |s| s.0);
                w.insert_resource(Score(s + 100));
            });
        }
        q.apply(&mut world);
        assert_eq!(world.get_resource::<Score>(), Some(&Score(101)));
    }

    #[test]
    fn custom_entity_command_reaches_sparse_shell() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        q.commands()
            .spawn()
            .queue(|w, e| {
                w.insert(e, Tag("sparse"));
            })
            .insert_static(Pos(1.0));
        let e = q.apply(&mut world)[0];
        assert_eq!(world.get::<Tag>(e), Some(&Tag("sparse")));
        assert_eq!(world.get_static::<Pos>(e), Some(&Pos(1.0)));
    }

    #[test]
    fn a_panicking_command_drops_the_rest_and_leaves_the_queue_usable() {
        let mut world = World::new();
        let mut q = CommandQueue::new();
        {
            let mut c = q.commands();
            c.spawn().insert_static(Health(1));
            c.queue(|_| panic!("boom"));
            c.spawn().insert_static(Health(2));
        }
        let result = catch_unwind(AssertUnwindSafe(|| q.apply(&mut world)));
        assert!(result.is_err());
        assert!(q.is_empty());
        assert_eq!(
            world.entity_count(),
            1,
            "only the spawn before the panic ran"
        );
        // Reusable, with slot numbering reset.
        q.commands().spawn().insert_static(Health(3));
        let e = q.apply(&mut world)[0];
        assert_eq!(world.get_static::<Health>(e), Some(&Health(3)));
    }

    #[test]
    fn inserted_components_count_as_added_to_a_tracker_that_predates_apply() {
        use crate::tick::ChangeTracker;
        let mut world = World::new();
        let e = world.spawn();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        let mut q = CommandQueue::new();
        q.commands().entity(e).insert_static(Health(1));
        q.apply(&mut world);
        let added: Vec<Entity> = world
            .query_added::<Health>(&tracker)
            .map(|(e, _)| e)
            .collect();
        assert_eq!(added, vec![e]);
        // And once the tracker catches up, it is no longer new.
        world.increment_change_tick();
        tracker.update(&world);
        assert_eq!(world.query_added::<Health>(&tracker).count(), 0);
    }
}
