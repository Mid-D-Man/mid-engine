// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-wasm.md, section "lib.rs"
// ============================================================================
//! WASM bindings for the mid-ui playground. The page (`web/ui/index.html`)
//! loads a `.mxui` source into a [`Playground`], asks it for a layout at a
//! viewport size, and paints the JSON draw list on a canvas.

pub mod logic;

use wasm_bindgen::prelude::*;

use logic::Core;

/// Installs a panic hook that prints Rust panics to the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// A loaded `.mxui` file.
#[wasm_bindgen]
pub struct Playground {
    core: Core,
}

#[wasm_bindgen]
impl Playground {
    /// Parses `.mxui` source and builds its tree. Throws a message naming the
    /// path of the problem on failure.
    #[cfg(feature = "dixscript")]
    #[wasm_bindgen(constructor)]
    pub fn new(source: &str) -> Result<Playground, JsError> {
        let loaded = mid_ui_mxui::load_str::<f32>(source).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Playground { core: Core::new(loaded) })
    }

    /// Computes rectangles for a viewport of `width` x `height`.
    pub fn layout(&mut self, width: f32, height: f32) {
        self.core.layout(width, height);
    }

    /// The draw list for the last layout, as a JSON array.
    pub fn draw_list_json(&self) -> String {
        self.core.draw_list_json()
    }

    /// Every node with its metadata and rectangle, as a JSON array.
    pub fn describe_json(&self) -> String {
        self.core.describe_json()
    }

    /// Name of the topmost interactive node at a point, if any.
    pub fn hit_name(&self, x: f32, y: f32) -> Option<String> {
        self.core.hit_name(x, y)
    }

    /// Whether the UI claims the pointer at a point.
    pub fn claims_pointer(&self, x: f32, y: f32) -> bool {
        self.core.claims_pointer(x, y)
    }

    /// Number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.core.node_count()
    }
}
