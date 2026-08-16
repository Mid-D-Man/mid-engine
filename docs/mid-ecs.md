# mid-ecs

Data-oriented ECS using Structure of Arrays (SoA) layout.

## Target

100 000+ entities at 60 Hz physics on a single core.
Parallelised queries via rayon.

## The Hybrid ECS Architecture: Static Core, Dynamic Shell

Mid Engine completely avoids the traditional Object-Oriented memory traps by splitting entity data into two highly optimized zones:

### 1. The Archetype Core (Heavy Logic)
* Components that remain static throughout an entity's lifecycle—like `Transform`, `Velocity`, or `PhysicsBody`—are packed into rigid Archetype tables.
* This guarantees perfect CPU cache locality.
* It allows our `mid-math` wide SIMD vectors to blast through positional updates without jumping around in memory, forming our high-performance "Inner Loops".

### 2. The Sparse Shell (Volatile Logic)
* Status effects or states that flicker on and off constantly—like `IsPoisoned`, `Disabled`, or `Hidden`—are managed using Sparse Sets or highly efficient Bitsets.
* The Sparse Set this shell will be built on is real now — `mid_collections::SparseSet` (`crates/mid-collections`, see `docs/mid-collections.md`), 18/18 real tests passing. Not wired into `World` yet — that needs the generational-arena piece (the entity handle type) to exist first, so `SparseSet` stays generic over any `SparseSetIndex` key rather than assuming one.
* **The "Stutter" Fix:** If you poison 1,000 goblins, the engine just flips a bitmask or adds a tiny entry in a sparse set. 
* Result: Zero memory is physically moved between archetype tables. The engine stays fast, and we avoid the memory-copying lag spikes that plague pure archetype architectures during massive state changes.
* For lightning-fast entity querying, the engine utilizes a `BitVec` layout (1 boolean into 1 bit), allowing us to filter hundreds of thousands of entities in microseconds using simple bitwise AND operations.

## The Ubel Stratum Bridge (The OOP Illusion)

While the Rust core handles the raw, flat memory arrays, the gameplay programmer never has to think about Archetypes or Bitsets.
* **HIGH Tier:** Developers interact with what looks like standard OOP classes (e.g., an `Actor` or `Entity` object).
* **LOW Tier:** The Ubel compiler acts as the "Middle Man," secretly lowering high-level code (e.g., `player.health -= 10`) into raw, memory-safe array accesses in the `mid-ecs` core.

## Network Sync (Multiplayer-First)

The `sync` module marks components for `mid-net` replication.
This is the Multiplayer-First mandate in practice: networking is baked into the ECS from day one, not bolted on later.
* Components can be explicitly flagged for synchronization (e.g., `@net Transform`). 
* The engine automatically handles serialization via DixScript (`.mdix`) to sync state across the wire.
* Because data is stored contiguously in the Archetype Core, the network system can simply request a memory block and run a single SIMD pass over that memory to detect deltas, compress with MBFA-lite, encrypt, and ship the UDP packet.
