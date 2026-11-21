// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "paint.rs"
// ============================================================================
//! The paint list: the flat, C-compatible list of shapes a host draws.

/// A filled rectangle, with rounded corners when `extra` is above zero.
pub const KIND_RECT: u32 = 0;
/// Start clipping to a rectangle.
pub const KIND_PUSH_CLIP: u32 = 1;
/// End the most recent clip.
pub const KIND_POP_CLIP: u32 = 2;
/// A line from `(a, b)` to `(c, d)`, `extra` pixels wide.
pub const KIND_LINE: u32 = 3;
/// A run of text with its top-left corner at `(a, b)`, `extra` pixels tall.
pub const KIND_TEXT: u32 = 4;

/// One shape. The layout is fixed (`repr(C)`) so a C host can read the list
/// directly. Colors are `0xRRGGBBAA`.
///
/// | kind | `a`, `b` | `c`, `d` | `extra` | text |
/// |---|---|---|---|---|
/// | rect | x, y | width, height | corner radius | none |
/// | push clip | x, y | width, height | unused | none |
/// | pop clip | unused | unused | unused | none |
/// | line | start x, y | end x, y | width | none |
/// | text | x, y | unused | size | bytes `text_start..text_start + text_len` |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PaintItem {
    /// One of the `KIND_*` constants.
    pub kind: u32,
    /// `0xRRGGBBAA`.
    pub color: u32,
    /// First coordinate, see the table above.
    pub a: f32,
    /// Second coordinate.
    pub b: f32,
    /// Third coordinate.
    pub c: f32,
    /// Fourth coordinate.
    pub d: f32,
    /// Radius, width or size, by kind.
    pub extra: f32,
    /// Start of this item's text in the list's text pool.
    pub text_start: u32,
    /// Length in bytes of this item's text.
    pub text_len: u32,
}

/// The shapes for one frame, plus the text they refer to.
#[derive(Clone, Debug, Default)]
pub struct PaintList {
    /// The shapes in paint order.
    pub items: Vec<PaintItem>,
    /// UTF-8 text for the text items. Not NUL-terminated.
    pub text: Vec<u8>,
}

/// Packs a color as `0xRRGGBBAA`.
pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32
}

impl PaintList {
    /// An empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Empties the list and keeps its allocations for the next frame.
    pub fn clear(&mut self) {
        self.items.clear();
        self.text.clear();
    }

    /// Number of shapes.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True when there are no shapes.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Appends a filled rectangle.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32, radius: f32) {
        self.items.push(PaintItem {
            kind: KIND_RECT,
            color,
            a: x,
            b: y,
            c: w,
            d: h,
            extra: radius,
            ..PaintItem::default()
        });
    }

    /// Starts clipping to a rectangle.
    pub fn push_clip(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.items.push(PaintItem {
            kind: KIND_PUSH_CLIP,
            a: x,
            b: y,
            c: w,
            d: h,
            ..PaintItem::default()
        });
    }

    /// Ends the most recent clip.
    pub fn pop_clip(&mut self) {
        self.items.push(PaintItem {
            kind: KIND_POP_CLIP,
            ..PaintItem::default()
        });
    }

    /// Appends a line.
    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, width: f32, color: u32) {
        self.items.push(PaintItem {
            kind: KIND_LINE,
            color,
            a: x0,
            b: y0,
            c: x1,
            d: y1,
            extra: width,
            ..PaintItem::default()
        });
    }

    /// Appends a run of text. Empty text appends nothing.
    pub fn text(&mut self, x: f32, y: f32, size: f32, color: u32, s: &str) {
        if s.is_empty() {
            return;
        }
        let start = self.text.len() as u32;
        self.text.extend_from_slice(s.as_bytes());
        self.items.push(PaintItem {
            kind: KIND_TEXT,
            color,
            a: x,
            b: y,
            extra: size,
            text_start: start,
            text_len: s.len() as u32,
            ..PaintItem::default()
        });
    }

    /// The text of a text item, or `None` for any other kind or a bad range.
    pub fn text_of(&self, item: &PaintItem) -> Option<&str> {
        if item.kind != KIND_TEXT {
            return None;
        }
        let start = item.text_start as usize;
        let end = start.checked_add(item.text_len as usize)?;
        std::str::from_utf8(self.text.get(start..end)?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_pack_as_rrggbbaa() {
        assert_eq!(rgba(0x12, 0x34, 0x56, 0x78), 0x1234_5678);
        assert_eq!(rgba(255, 0, 0, 255), 0xFF00_00FF);
    }

    #[test]
    fn text_is_stored_in_the_pool_and_read_back() {
        let mut p = PaintList::new();
        p.text(1.0, 2.0, 16.0, 0xFFFF_FFFF, "one");
        p.text(3.0, 4.0, 16.0, 0xFFFF_FFFF, "two é");
        p.text(0.0, 0.0, 16.0, 0xFFFF_FFFF, "");
        assert_eq!(p.len(), 2, "empty text adds nothing");
        assert_eq!(p.text_of(&p.items[0]), Some("one"));
        assert_eq!(p.text_of(&p.items[1]), Some("two é"));
        assert_eq!(p.items[1].text_start, 3);
    }

    #[test]
    fn text_of_refuses_other_kinds_and_bad_ranges() {
        let mut p = PaintList::new();
        p.rect(0.0, 0.0, 1.0, 1.0, 0, 0.0);
        assert_eq!(p.text_of(&p.items[0]), None);
        let bad = PaintItem {
            kind: KIND_TEXT,
            text_start: 99,
            text_len: 5,
            ..PaintItem::default()
        };
        assert_eq!(p.text_of(&bad), None);
        let overflow = PaintItem {
            kind: KIND_TEXT,
            text_start: u32::MAX,
            text_len: u32::MAX,
            ..PaintItem::default()
        };
        assert_eq!(p.text_of(&overflow), None);
    }

    #[test]
    fn clear_keeps_capacity_and_empties_both_buffers() {
        let mut p = PaintList::new();
        p.text(0.0, 0.0, 10.0, 0, "abc");
        p.line(0.0, 0.0, 1.0, 1.0, 2.0, 0);
        let cap = p.items.capacity();
        p.clear();
        assert!(p.is_empty() && p.text.is_empty());
        assert_eq!(p.items.capacity(), cap);
    }

    #[test]
    fn the_item_layout_is_the_one_the_c_header_declares() {
        assert_eq!(std::mem::size_of::<PaintItem>(), 36);
        assert_eq!(std::mem::align_of::<PaintItem>(), 4);
    }
}
