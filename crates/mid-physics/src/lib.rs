// crates/mid-physics/src/lib.rs
//! mid-physics — rigid body dynamics for Mid Engine.
//!
//! New this pass (`docs/roadmap.md`, "What can be built in parallel"):
//! built directly against `mid-math` vectors/quaternions and `mid-geom`
//! shapes — deliberately **not** wired into `mid-ecs` yet. `mid-ecs`'s
//! archetype/component storage is still evolving; wrapping [`RigidBody`]
//! as a component is a thin follow-up once that settles, not something
//! worth blocking this crate's actual physics logic on.
//!
//! ## What's real in this pass
//!
//! [`RigidBody`] plus [`integrate`]: semi-implicit (symplectic) Euler
//! integration of linear motion under a constant acceleration (gravity).
//! Correct, if minimal — this is the same integrator Box2D/Bullet/Rapier
//! all default to (update velocity first, then use the *updated*
//! velocity to update position — unlike naive/explicit Euler, this is
//! unconditionally stable for a constant force, no accumulating energy
//! gain).
//!
//! ## What's not built yet, on purpose
//!
//! - **Angular integration** (quaternion derivative / angular velocity)
//!   — linear-only for this pass.
//! - **Collision detection** (broadphase/narrowphase) — `mid-geom`'s own
//!   known gaps (OBB, capsule-vs-AABB, a real broadphase) are explicitly
//!   "driven by `mid-physics`'s actual requirements rather than built
//!   speculatively" per `docs/architecture.md`. Building collision
//!   detection here is exactly the trigger that finally gives those gaps
//!   real requirements to build against — deliberately left for the next
//!   real pass on this crate, not this one.
//! - **Constraint solving / contact resolution, sleeping** — needs
//!   collision detection first.
//! - **A fixed-timestep driver** — [`integrate`] takes a plain `dt:
//!   f32`; wiring `mid-time`'s `FixedTimestep`/`Clock` in is the natural
//!   next step once both crates exist side by side (`docs/roadmap.md`),
//!   not duplicated here ahead of that.

use mid_math::{Quat, Vec3};

/// A single rigid body: position/rotation plus linear/angular velocity
/// and mass. `inv_mass == 0.0` marks a static or kinematic body — never
/// moved by [`integrate`], matching the sentinel convention Box2D/
/// Bullet/Rapier all use (an exact `0.0`, set by construction, not a
/// computed near-zero value from dividing by a very large mass).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBody {
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
}

impl RigidBody {
    /// A movable body with finite mass. Panics on non-positive mass —
    /// that's an authoring bug (use [`RigidBody::new_static`] for
    /// infinite-mass bodies), not a runtime state to silently coerce.
    pub fn new_dynamic(position: Vec3, mass: f32) -> Self {
        assert!(mass > 0.0, "RigidBody::new_dynamic requires mass > 0.0 — use new_static for infinite mass");
        Self {
            position,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass: 1.0 / mass,
        }
    }

    /// An immovable body (infinite mass). [`integrate`] never touches
    /// static bodies at all.
    pub fn new_static(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: f32::INFINITY,
            inv_mass: 0.0,
        }
    }

    #[inline]
    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
}

/// Advances every dynamic body in `bodies` by `dt` seconds under
/// constant acceleration `gravity`, via semi-implicit Euler:
///
/// ```text
/// velocity += gravity * dt   // update velocity first...
/// position += velocity * dt  // ...then use the *updated* velocity
/// ```
///
/// Static bodies ([`RigidBody::is_static`]) are skipped entirely.
/// Angular velocity is carried but not yet integrated into `rotation` —
/// see the module-level "not built yet" list.
pub fn integrate(bodies: &mut [RigidBody], gravity: Vec3, dt: f32) {
    for body in bodies.iter_mut() {
        if body.is_static() {
            continue;
        }
        body.linear_velocity += gravity * dt;
        body.position += body.linear_velocity * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrate_applies_gravity_to_dynamic_body() {
        let mut bodies = [RigidBody::new_dynamic(Vec3::ZERO, 1.0)];
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        integrate(&mut bodies, gravity, 1.0);

        // Semi-implicit Euler: velocity updates first, so after one
        // dt=1.0 step, velocity == gravity * dt, and position ==
        // (updated) velocity * dt == gravity * dt * dt.
        assert_eq!(bodies[0].linear_velocity, gravity);
        assert_eq!(bodies[0].position, gravity);
    }

    #[test]
    fn integrate_never_moves_static_body() {
        let start = Vec3::new(1.0, 2.0, 3.0);
        let mut bodies = [RigidBody::new_static(start)];
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        integrate(&mut bodies, gravity, 1.0 / 60.0);

        assert_eq!(bodies[0].position, start);
        assert_eq!(bodies[0].linear_velocity, Vec3::ZERO);
    }

    #[test]
    fn integrate_leaves_static_bodies_untouched_among_dynamic_ones() {
        let mut bodies = [
            RigidBody::new_dynamic(Vec3::ZERO, 2.0),
            RigidBody::new_static(Vec3::new(5.0, 0.0, 0.0)),
        ];
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        integrate(&mut bodies, gravity, 1.0 / 60.0);

        assert_ne!(bodies[0].position, Vec3::ZERO);
        assert_eq!(bodies[1].position, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "mass > 0.0")]
    fn new_dynamic_rejects_non_positive_mass() {
        let _ = RigidBody::new_dynamic(Vec3::ZERO, 0.0);
    }
}
