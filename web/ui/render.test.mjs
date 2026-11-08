// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-wasm.md, section "render.js"
// ============================================================================
// Run with: node --test web/ui/render.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { cssColor, paint, paintDrawList, paintLabels, paintOutline, roundedRectPath, toViewport } from './render.js';

// A canvas context that records every call and property assignment.
function fakeCtx() {
  const calls = [];
  const ctx = new Proxy({ calls }, {
    get(t, k) {
      if (k === 'calls') return t.calls;
      if (k in t) return t[k];
      return (...args) => { t.calls.push([k, ...args]); };
    },
    set(t, k, v) { t.calls.push(['set:' + k, v]); t[k] = v; return true; },
  });
  return ctx;
}
const names = (ctx) => ctx.calls.map((c) => c[0]);

test('colors become rgba with alpha scaled to 0..1', () => {
  assert.equal(cssColor([255, 0, 128, 255]), 'rgba(255,0,128,1)');
  assert.equal(cssColor([0, 0, 0, 0]), 'rgba(0,0,0,0)');
  assert.equal(cssColor([10, 20, 30, 128]), 'rgba(10,20,30,0.502)');
});

test('a square fill is a plain rect', () => {
  const ctx = fakeCtx();
  roundedRectPath(ctx, 1, 2, 30, 40, 0);
  assert.deepEqual(ctx.calls, [['beginPath'], ['rect', 1, 2, 30, 40]]);
});

test('a rounded fill uses four arcs with the radius', () => {
  const ctx = fakeCtx();
  roundedRectPath(ctx, 0, 0, 100, 50, 8);
  assert.equal(names(ctx).filter((n) => n === 'arcTo').length, 4);
  assert.deepEqual(ctx.calls[1], ['moveTo', 8, 0]);
  assert.ok(ctx.calls.filter((c) => c[0] === 'arcTo').every((c) => c[5] === 8));
});

test('the radius is clamped to half the shorter side', () => {
  const ctx = fakeCtx();
  roundedRectPath(ctx, 0, 0, 100, 20, 999);
  assert.ok(ctx.calls.filter((c) => c[0] === 'arcTo').every((c) => c[5] === 10));
  const neg = fakeCtx();
  roundedRectPath(neg, 0, 0, 10, 10, -5);
  assert.deepEqual(neg.calls, [['beginPath'], ['rect', 0, 0, 10, 10]]);
});

test('fill sets the color then fills', () => {
  const ctx = fakeCtx();
  paintDrawList(ctx, [{ t: 'fill', x: 0, y: 0, w: 10, h: 10, r: 0, c: [1, 2, 3, 255] }]);
  assert.deepEqual(ctx.calls[0], ['set:fillStyle', 'rgba(1,2,3,1)']);
  assert.equal(names(ctx).at(-1), 'fill');
});

test('clip pushes a save and clip, pop restores', () => {
  const ctx = fakeCtx();
  paintDrawList(ctx, [{ t: 'clip', x: 5, y: 6, w: 7, h: 8 }, { t: 'pop' }]);
  assert.deepEqual(ctx.calls, [['save'], ['beginPath'], ['rect', 5, 6, 7, 8], ['clip'], ['restore']]);
});

test('an unbalanced list is balanced on exit and a stray pop is ignored', () => {
  const open = fakeCtx();
  paintDrawList(open, [{ t: 'clip', x: 0, y: 0, w: 1, h: 1 }, { t: 'clip', x: 0, y: 0, w: 1, h: 1 }]);
  assert.equal(names(open).filter((n) => n === 'save').length, 2);
  assert.equal(names(open).filter((n) => n === 'restore').length, 2);
  const stray = fakeCtx();
  paintDrawList(stray, [{ t: 'pop' }, { t: 'pop' }]);
  assert.deepEqual(stray.calls, []);
});

test('unknown commands are skipped', () => {
  const ctx = fakeCtx();
  paintDrawList(ctx, [{ t: 'unknown' }, { t: 'mystery' }]);
  assert.deepEqual(ctx.calls, []);
});

test('labels: buttons centered, labels left, hidden and empty text skipped', () => {
  const ctx = fakeCtx();
  paintLabels(ctx, [
    { visible: true, kind: 'UiKind.Button', text: 'Go', x: 10, y: 20, w: 100, h: 40 },
    { visible: true, kind: 'UiKind.Label', text: 'HP', x: 0, y: 0, w: 50, h: 30 },
    { visible: false, kind: 'UiKind.Label', text: 'hidden', x: 0, y: 0, w: 1, h: 1 },
    { visible: true, kind: 'UiKind.Label', text: null, x: 0, y: 0, w: 1, h: 1 },
  ]);
  const texts = ctx.calls.filter((c) => c[0] === 'fillText');
  assert.deepEqual(texts, [['fillText', 'Go', 60, 40], ['fillText', 'HP', 8, 15]]);
  const aligns = ctx.calls.filter((c) => c[0] === 'set:textAlign').map((c) => c[1]);
  assert.deepEqual(aligns, ['center', 'left']);
  assert.equal(names(ctx).filter((n) => n === 'save').length, names(ctx).filter((n) => n === 'restore').length);
});

test('the hover outline strokes the node rectangle and does nothing for null', () => {
  const ctx = fakeCtx();
  paintOutline(ctx, { x: 10, y: 10, w: 50, h: 20 });
  assert.deepEqual(ctx.calls.find((c) => c[0] === 'strokeRect'), ['strokeRect', 11, 11, 48, 18]);
  const none = fakeCtx();
  paintOutline(none, null);
  assert.deepEqual(none.calls, []);
});

test('paint clears first, then draw list, labels and outline in that order', () => {
  const ctx = fakeCtx();
  paint(
    ctx, 200, 100,
    [{ t: 'fill', x: 0, y: 0, w: 10, h: 10, r: 0, c: [0, 0, 0, 255] }],
    [{ visible: true, kind: 'UiKind.Label', text: 'x', x: 0, y: 0, w: 10, h: 10 }],
    { x: 0, y: 0, w: 10, h: 10 },
  );
  const n = names(ctx);
  assert.deepEqual(ctx.calls[0], ['clearRect', 0, 0, 200, 100]);
  assert.ok(n.indexOf('fill') < n.indexOf('fillText'));
  assert.ok(n.indexOf('fillText') < n.indexOf('strokeRect'));
});

test('pointer positions scale from the displayed size to the viewport', () => {
  assert.deepEqual(toViewport(200, 150, 400, 300, 800, 600), { x: 400, y: 300 });
  assert.deepEqual(toViewport(5, 5, 0, 0, 800, 600), { x: 0, y: 0 });
});
