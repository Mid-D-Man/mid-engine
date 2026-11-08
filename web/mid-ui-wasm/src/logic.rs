// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-wasm.md, section "logic.rs"
// ============================================================================
//! The playground's logic, written without any wasm-bindgen types so it can
//! be tested natively.

use mid_ui::{draw, hit, layout, DrawCmd, Point, Size};
use mid_ui_mxui::Loaded;

/// A loaded `.mxui` tree plus the operations the page calls on it.
pub struct Core {
    loaded: Loaded<f32>,
}

impl Core {
    /// Wraps a loaded file.
    pub fn new(loaded: Loaded<f32>) -> Self {
        Self { loaded }
    }

    /// Computes rectangles for a viewport.
    pub fn layout(&mut self, width: f32, height: f32) {
        layout::compute(&mut self.loaded.tree, Size::new(width, height));
    }

    /// The draw list as a JSON array. See [`draw_list_json`].
    pub fn draw_list_json(&self) -> String {
        let mut list = Vec::new();
        draw::build_draw_list(&self.loaded.tree, &mut list);
        draw_list_json(&list)
    }

    /// Every node in paint order as a JSON array. See [`describe_json`].
    pub fn describe_json(&self) -> String {
        describe_json(&self.loaded)
    }

    /// Name of the topmost interactive node at a point: `None` when nothing
    /// is hit, `"(unnamed)"` when the node has no `id`.
    pub fn hit_name(&self, x: f32, y: f32) -> Option<String> {
        let id = hit::hit_test(&self.loaded.tree, Point::new(x, y))?;
        Some(
            self.loaded
                .meta
                .get(&id)
                .and_then(|m| m.name.clone())
                .unwrap_or_else(|| "(unnamed)".to_string()),
        )
    }

    /// Whether the UI wants the pointer at a point.
    pub fn claims_pointer(&self, x: f32, y: f32) -> bool {
        hit::claims_pointer(&self.loaded.tree, Point::new(x, y))
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.loaded.tree.len()
    }
}

/// Appends `s` as a JSON string literal, escaping what JSON requires.
pub fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

// JSON has no NaN or infinity; they are written as 0 so the page never gets
// unparseable output.
fn num(out: &mut String, v: f32) {
    if v.is_finite() {
        out.push_str(&v.to_string());
    } else {
        out.push('0');
    }
}

fn rect_fields(out: &mut String, r: mid_ui::Rect<f32>) {
    for (key, v) in [("x", r.x), ("y", r.y), ("w", r.w), ("h", r.h)] {
        out.push_str(",\"");
        out.push_str(key);
        out.push_str("\":");
        num(out, v);
    }
}

/// Serializes a draw list: `{"t":"fill","x","y","w","h","r","c":[r,g,b,a]}`,
/// `{"t":"clip","x","y","w","h"}` and `{"t":"pop"}`. A command this wrapper
/// does not know serializes as `{"t":"unknown"}`.
pub fn draw_list_json(list: &[DrawCmd<f32>]) -> String {
    let mut out = String::from("[");
    for (i, cmd) in list.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match cmd {
            DrawCmd::FillRect { rect, color, radius } => {
                out.push_str("{\"t\":\"fill\"");
                rect_fields(&mut out, *rect);
                out.push_str(",\"r\":");
                num(&mut out, *radius);
                out.push_str(&format!(",\"c\":[{},{},{},{}]}}", color.r, color.g, color.b, color.a));
            }
            DrawCmd::PushClip(rect) => {
                out.push_str("{\"t\":\"clip\"");
                rect_fields(&mut out, *rect);
                out.push('}');
            }
            DrawCmd::PopClip => out.push_str("{\"t\":\"pop\"}"),
            _ => out.push_str("{\"t\":\"unknown\"}"),
        }
    }
    out.push(']');
    out
}

/// Serializes every node in paint order:
/// `{"depth","id","kind","text","bind","x","y","w","h","visible","interactive"}`.
/// `id`, `kind` (as `"Type.Variant"`), `text` and `bind` are `null` when the
/// file did not declare them.
pub fn describe_json(loaded: &Loaded<f32>) -> String {
    let tree = &loaded.tree;
    let mut out = String::from("[");
    let mut first = true;
    let mut stack = vec![(tree.root(), 0_usize)];
    while let Some((id, depth)) = stack.pop() {
        let (Some(style), Some(rect)) = (tree.style(id), tree.rect(id)) else { continue };
        if !first {
            out.push(',');
        }
        first = false;
        let meta = loaded.meta.get(&id);
        out.push_str(&format!("{{\"depth\":{depth}"));
        let optional = |out: &mut String, key: &str, value: Option<String>| {
            out.push_str(",\"");
            out.push_str(key);
            out.push_str("\":");
            match value {
                Some(v) => json_string(out, &v),
                None => out.push_str("null"),
            }
        };
        optional(&mut out, "id", meta.and_then(|m| m.name.clone()));
        optional(&mut out, "kind", meta.and_then(|m| m.kind.as_ref()).map(|k| format!("{}.{}", k.type_name, k.field)));
        optional(&mut out, "text", meta.and_then(|m| m.text.clone()));
        optional(&mut out, "bind", meta.and_then(|m| m.bind.as_ref()).map(|b| b.name.clone()));
        rect_fields(&mut out, rect);
        out.push_str(&format!(",\"visible\":{},\"interactive\":{}}}", style.visible, style.interactive));
        for child in tree.children(id).iter().rev() {
            stack.push((*child, depth + 1));
        }
    }
    out.push(']');
    out
}
