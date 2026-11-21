// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-editor.md, section "text.rs"
// ============================================================================
//! Text measuring: the seam a host fills in so layout can size text.

/// Measures text. The host implements it with its own font code.
pub trait TextMeasure {
    /// The width in pixels of `text` drawn at `size` pixels tall.
    fn width(&self, text: &str, size: f32) -> f32;
}

/// A measure for tests and for hosts with no font: every character is
/// `factor * size` wide.
#[derive(Clone, Copy, Debug)]
pub struct FixedWidth(pub f32);

impl TextMeasure for FixedWidth {
    fn width(&self, text: &str, size: f32) -> f32 {
        text.chars().count() as f32 * size * self.0
    }
}

/// Returns `text`, shortened with `...` on the right when it is wider than
/// `max_w`. Returns an empty string when not even the dots fit.
pub fn fit(m: &dyn TextMeasure, text: &str, size: f32, max_w: f32) -> String {
    if m.width(text, size) <= max_w {
        return text.to_string();
    }
    let dots = "...";
    if m.width(dots, size) > max_w {
        return String::new();
    }
    let mut out: String = text.to_string();
    while !out.is_empty() {
        out.pop();
        let candidate = format!("{out}{dots}");
        if m.width(&candidate, size) <= max_w {
            return candidate;
        }
    }
    dots.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_counts_characters_not_bytes() {
        let m = FixedWidth(0.5);
        assert_eq!(m.width("abcd", 10.0), 20.0);
        assert_eq!(m.width("é", 10.0), 5.0);
    }

    #[test]
    fn fit_keeps_text_that_fits() {
        let m = FixedWidth(0.5);
        assert_eq!(fit(&m, "hello", 10.0, 25.0), "hello");
    }

    #[test]
    fn fit_shortens_with_dots_and_never_exceeds_the_width() {
        let m = FixedWidth(0.5);
        let s = fit(&m, "hello world", 10.0, 40.0);
        assert!(s.ends_with("..."));
        assert!(m.width(&s, 10.0) <= 40.0);
        assert_eq!(s, "hello...");
    }

    #[test]
    fn fit_returns_nothing_when_the_dots_do_not_fit() {
        let m = FixedWidth(0.5);
        assert_eq!(fit(&m, "hello", 10.0, 10.0), "");
        assert_eq!(fit(&m, "hello", 10.0, 15.0), "...");
    }
}
