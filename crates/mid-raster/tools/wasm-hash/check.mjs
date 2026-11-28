// Runs the wasm32 module built from tools/wasm-hash and compares its hashes
// with the values pinned in tests/golden/scenes.rs.
//
//   cargo build --release --target wasm32-unknown-unknown \
//       --manifest-path crates/mid-raster/tools/wasm-hash/Cargo.toml
//   node crates/mid-raster/tools/wasm-hash/check.mjs \
//       crates/mid-raster/tools/wasm-hash/target/wasm32-unknown-unknown/release/mid_raster_wasm_hash.wasm

import { readFileSync } from 'node:fs';

const path = process.argv[2];
if (!path) {
  console.error('usage: node check.mjs <module.wasm>');
  process.exit(2);
}

const { instance } = await WebAssembly.instantiate(readFileSync(path), {});
const hex = (v) => '0x' + BigInt.asUintN(64, v).toString(16).padStart(16, '0');
let bad = 0;
for (const name of ['srgb', 'linear', 'random']) {
  const got = instance.exports[name]();
  const want = instance.exports['expected_' + name]();
  const ok = got === want;
  console.log(`${name.padEnd(7)} ${hex(got)} ${ok ? 'matches' : 'DIFFERS from ' + hex(want)}`);
  if (!ok) bad += 1;
}
process.exit(bad === 0 ? 0 : 1);
