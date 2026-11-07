// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui.md, section "tree.rs"
// ============================================================================
//! The widget tree: an arena of nodes addressed by generational ids.

use crate::geom::Rect;
use crate::real::Real;
use crate::style::Style;

/// Handle to a node in a [`Tree`]. Becomes invalid when the node is removed;
/// a stale id is never mistaken for a different node that reused its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

struct Node<R: Real> {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    style: Style<R>,
    rect: Rect<R>,
}

struct Slot<R: Real> {
    generation: u32,
    node: Option<Node<R>>,
}

/// An ordered tree of nodes with one permanent root.
///
/// Child order is paint order: later siblings are drawn on top of earlier
/// ones and win hit tests.
pub struct Tree<R: Real = f32> {
    slots: Vec<Slot<R>>,
    free: Vec<u32>,
    root: NodeId,
    len: usize,
}

impl<R: Real> Tree<R> {
    /// Creates a tree containing only a root node with the given style.
    pub fn new(root_style: Style<R>) -> Self {
        let root = NodeId { index: 0, generation: 0 };
        let node = Node { parent: None, children: Vec::new(), style: root_style, rect: Rect::ZERO };
        Self { slots: vec![Slot { generation: 0, node: Some(node) }], free: Vec::new(), root, len: 1 }
    }

    /// The root node. It always exists and cannot be removed.
    #[inline]
    pub fn root(&self) -> NodeId { self.root }

    /// Number of live nodes, root included.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// Always `false`: the root is permanent. Present for API symmetry with
    /// [`Tree::len`].
    #[inline]
    pub fn is_empty(&self) -> bool { false }

    /// `true` when `id` refers to a live node.
    #[inline]
    pub fn contains(&self, id: NodeId) -> bool { self.node(id).is_some() }

    /// Appends a new node as the last child of `parent`. Returns `None` when
    /// `parent` is stale.
    pub fn add_child(&mut self, parent: NodeId, style: Style<R>) -> Option<NodeId> {
        if !self.contains(parent) {
            return None;
        }
        let node = Node { parent: Some(parent), children: Vec::new(), style, rect: Rect::ZERO };
        let id = match self.free.pop() {
            Some(index) => {
                let slot = &mut self.slots[index as usize];
                slot.node = Some(node);
                NodeId { index, generation: slot.generation }
            }
            None => {
                let index = u32::try_from(self.slots.len()).ok()?;
                self.slots.push(Slot { generation: 0, node: Some(node) });
                NodeId { index, generation: 0 }
            }
        };
        if let Some(p) = self.node_mut(parent) {
            p.children.push(id);
        }
        self.len += 1;
        Some(id)
    }

    /// Removes `id` and its whole subtree. Returns `false` for a stale id or
    /// the root, in which case nothing changes.
    pub fn remove(&mut self, id: NodeId) -> bool {
        if id == self.root || !self.contains(id) {
            return false;
        }
        if let Some(parent) = self.node(id).and_then(|n| n.parent) {
            if let Some(p) = self.node_mut(parent) {
                p.children.retain(|c| *c != id);
            }
        }
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            let slot = &mut self.slots[cur.index as usize];
            if let Some(node) = slot.node.take() {
                stack.extend(node.children);
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(cur.index);
                self.len -= 1;
            }
        }
        true
    }

    /// Parent of `id`; `None` for the root and for stale ids.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.node(id).and_then(|n| n.parent)
    }

    /// Children of `id` in paint order; empty for stale ids.
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        self.node(id).map_or(&[], |n| n.children.as_slice())
    }

    /// Style of `id`.
    pub fn style(&self, id: NodeId) -> Option<&Style<R>> {
        self.node(id).map(|n| &n.style)
    }

    /// Mutable style of `id`. Changes take effect on the next
    /// [`layout::compute`](crate::layout::compute).
    pub fn style_mut(&mut self, id: NodeId) -> Option<&mut Style<R>> {
        self.node_mut(id).map(|n| &mut n.style)
    }

    /// Rectangle computed by the last layout pass; [`Rect::ZERO`] before the
    /// first pass. Hidden nodes keep whatever they had before they were hidden.
    pub fn rect(&self, id: NodeId) -> Option<Rect<R>> {
        self.node(id).map(|n| n.rect)
    }

    pub(crate) fn set_rect(&mut self, id: NodeId, rect: Rect<R>) {
        if let Some(n) = self.node_mut(id) {
            n.rect = rect;
        }
    }

    fn node(&self, id: NodeId) -> Option<&Node<R>> {
        self.slots
            .get(id.index as usize)
            .filter(|s| s.generation == id.generation)
            .and_then(|s| s.node.as_ref())
    }

    fn node_mut(&mut self, id: NodeId) -> Option<&mut Node<R>> {
        self.slots
            .get_mut(id.index as usize)
            .filter(|s| s.generation == id.generation)
            .and_then(|s| s.node.as_mut())
    }
}
