# mid-ecs

Data-oriented ECS using Structure of Arrays (SoA) layout.

## Target

100 000+ entities at 60 Hz physics on a single core.
Parallelised queries via rayon.

## Archetype Model

Entities with the same component set share an archetype.
Components stored in contiguous arrays — cache-friendly iteration.

## Network Sync

The `sync` module marks components for mid-net replication.
This is the Multiplayer-First mandate in practice:
networking is baked into the ECS from day one, not bolted on later.
