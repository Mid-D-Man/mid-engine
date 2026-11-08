// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-wasm.md, section "render.js"
// ============================================================================
// Canvas 2D backend for the mid-ui draw list. Pure functions over a canvas
// context, so web/ui/render.test.mjs can run them against a recording fake.

// Rounded-rectangle path; the radius is clamped so it never exceeds half the
// shorter side (the same rule CSS border-radius uses).
export function roundedRectPath(ctx, x, y, w, h, r) {
  const rad = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  if (rad === 0) {
    ctx.rect(x, y, w, h);
    return;
  }
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

export function cssColor(c) {
  return `rgba(${c[0]},${c[1]},${c[2]},${+(c[3] / 255).toFixed(3)})`;
}

// Paints the draw-command list. Clip pushes and pops are kept balanced even if
// the list is not, so one bad list cannot leave the context clipped.
export function paintDrawList(ctx, list) {
  let depth = 0;
  for (const c of list) {
    if (c.t === 'fill') {
      ctx.fillStyle = cssColor(c.c);
      roundedRectPath(ctx, c.x, c.y, c.w, c.h, c.r);
      ctx.fill();
    } else if (c.t === 'clip') {
      ctx.save();
      ctx.beginPath();
      ctx.rect(c.x, c.y, c.w, c.h);
      ctx.clip();
      depth++;
    } else if (c.t === 'pop') {
      if (depth > 0) {
        ctx.restore();
        depth--;
      }
    }
  }
  while (depth > 0) {
    ctx.restore();
    depth--;
  }
}

// Text is NOT part of the draw list yet (there is no text backend), so the
// playground draws each node's `text` itself: buttons centered, labels
// left-aligned. This is a stand-in, clipped to the node's own rectangle.
export function paintLabels(ctx, nodes) {
  ctx.font = '14px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (const n of nodes) {
    if (!n.visible || !n.text) continue;
    const button = n.kind === 'UiKind.Button';
    ctx.save();
    ctx.beginPath();
    ctx.rect(n.x, n.y, n.w, n.h);
    ctx.clip();
    ctx.fillStyle = '#e6edf3';
    ctx.textAlign = button ? 'center' : 'left';
    ctx.fillText(n.text, button ? n.x + n.w / 2 : n.x + 8, n.y + n.h / 2);
    ctx.restore();
  }
}

export function paintOutline(ctx, node, color = '#857fff') {
  if (!node) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(node.x + 1, node.y + 1, Math.max(0, node.w - 2), Math.max(0, node.h - 2));
  ctx.restore();
}

// Everything in one call: background, draw list, labels, hover outline.
export function paint(ctx, width, height, list, nodes, hoverNode) {
  ctx.clearRect(0, 0, width, height);
  paintDrawList(ctx, list);
  paintLabels(ctx, nodes);
  paintOutline(ctx, hoverNode);
}

// Maps a pointer position on the displayed canvas to viewport coordinates.
export function toViewport(offsetX, offsetY, displayW, displayH, width, height) {
  if (displayW <= 0 || displayH <= 0) return { x: 0, y: 0 };
  return { x: (offsetX / displayW) * width, y: (offsetY / displayH) * height };
}
