// crates/mid-common/src/string/search.rs
//! Fuzzy string search — ported from Blender's BLI_string_search.hh.
//!
//! Core algorithm: Damerau-Levenshtein distance (edit distance with transposition)
//! operating at Unicode codepoint level, not byte level.
//!
//! `StringSearch<T>` filters and ranks items by a query string.
//! Items that don't match are filtered out. Remaining items are sorted by
//! match quality (fewer errors = better), then by user-supplied weight.
//!
//! Engine uses: dev console command search, asset browser, entity inspector,
//! DixScript packet type lookup.
//!
//! Blender reference: BLI_string_search.hh / intern/string_search.cc

use alloc::string::String;
use alloc::vec::Vec;

// ─────────────────────────────────────────────────────────────────────────────
// Damerau-Levenshtein distance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Damerau-Levenshtein distance between two strings.
///
/// Operates at Unicode codepoint level (not bytes). Supports:
///   - Deletion
///   - Insertion
///   - Substitution
///   - Transposition of adjacent characters
///
/// Returns the minimum number of single-character edits.
///
/// ```rust
/// use mid_common::string::damerau_levenshtein_distance;
/// assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(damerau_levenshtein_distance("ca", "abc"), 2);
/// assert_eq!(damerau_levenshtein_distance("", "abc"), 3);
/// ```
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> usize {
    // Collect into char vecs so we index by codepoint, not byte.
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let len_a = a.len();
    let len_b = b.len();

    if len_a == 0 { return len_b; }
    if len_b == 0 { return len_a; }

    // Use Optimal String Alignment (restricted edit distance).
    // True DL requires a more complex matrix, but OSA is sufficient for
    // fuzzy search and matches Blender's implementation intent.
    //
    // d[i][j] = edit distance between a[0..i] and b[0..j].
    let mut d = vec![vec![0usize; len_b + 1]; len_a + 1];

    for i in 0..=len_a { d[i][0] = i; }
    for j in 0..=len_b { d[0][j] = j; }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = usize::from(a[i - 1] != b[j - 1]);

            d[i][j] = (d[i - 1][j] + 1)              // deletion
                .min(d[i][j - 1] + 1)                 // insertion
                .min(d[i - 1][j - 1] + cost);          // substitution

            // Transposition
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }

    d[len_a][len_b]
}

// ─────────────────────────────────────────────────────────────────────────────
// Fuzzy match scoring
// ─────────────────────────────────────────────────────────────────────────────

/// Test whether `query` fuzzy-matches `full`.
///
/// Returns `Some(errors)` where errors is the match cost (lower = better).
/// Returns `None` if the match is too poor to be useful.
///
/// Matching rules (from Blender's `get_fuzzy_match_errors`):
///   - `query` must be a subsequence of `full` (all query chars appear in order)
///   - Edit distance is computed per query-word against matching regions of full
///   - Short queries get tighter tolerance
///
/// ```rust
/// use mid_common::string::fuzzy_match_score;
/// assert!(fuzzy_match_score("pos", "position").is_some());
/// assert!(fuzzy_match_score("xyz123", "position").is_none());
/// ```
pub fn fuzzy_match_score(query: &str, full: &str) -> Option<usize> {
    if query.is_empty() { return Some(0); }

    let query_lower = query.to_lowercase();
    let full_lower  = full.to_lowercase();

    // Fast path: exact prefix match
    if full_lower.starts_with(&query_lower) {
        return Some(0);
    }

    // Fast path: exact substring match
    if full_lower.contains(&query_lower) {
        return Some(0);
    }

    // Subsequence check: all query chars must appear in order in full
    if !is_subsequence(&query_lower, &full_lower) {
        return None;
    }

    // Compute edit distance against the best matching window in full
    let q_chars: Vec<char> = query_lower.chars().collect();
    let f_chars: Vec<char> = full_lower.chars().collect();
    let q_len = q_chars.len();
    let f_len = f_chars.len();

    // Tolerance: allow errors proportional to query length, minimum 0
    let max_errors = (q_len / 4).max(1);

    // Slide a window of query_len chars over full, find best DL distance
    let window = (q_len + max_errors).min(f_len);
    let mut best = usize::MAX;

    for start in 0..=(f_len.saturating_sub(q_len)) {
        let end = (start + window).min(f_len);
        let window_str: String = f_chars[start..end].iter().collect();
        let q_str: String = q_chars.iter().collect();
        let dist = damerau_levenshtein_distance(&q_str, &window_str);
        best = best.min(dist);
        if best == 0 { break; }
    }

    if best <= max_errors {
        Some(best)
    } else {
        None
    }
}

/// Returns true if all characters of `needle` appear in `haystack` in order.
fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut needle_chars = needle.chars();
    let mut current = match needle_chars.next() {
        Some(c) => c,
        None => return true,
    };
    for c in haystack.chars() {
        if c == current {
            match needle_chars.next() {
                Some(next) => current = next,
                None => return true,
            }
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// Word extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Split a string into lowercase words, removing punctuation.
///
/// Delimiters: whitespace, `.`, `_`, `-`, `/`, `\`, `(`, `)`.
/// Empty words are skipped.
///
/// ```rust
/// use mid_common::string::search::extract_words;
/// let words = extract_words("Player_Left.Arm");
/// assert_eq!(words, vec!["player", "left", "arm"]);
/// ```
pub fn extract_words(s: &str) -> Vec<String> {
    s.split(|c: char| c.is_whitespace() || matches!(c, '.' | '_' | '-' | '/' | '\\' | '(' | ')'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchItem
// ─────────────────────────────────────────────────────────────────────────────

/// A single searchable item stored in `StringSearch<T>`.
pub struct SearchItem<T> {
    /// The data payload returned when this item matches.
    pub data: T,
    /// Original display name.
    pub name: String,
    /// Pre-split, lowercased words for fast matching.
    words: Vec<String>,
    /// User-supplied priority weight. Higher = preferred on equal score.
    pub weight: f32,
    /// Logical recency timestamp. Higher = more recently used.
    pub recent_time: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// StringSearch<T>
// ─────────────────────────────────────────────────────────────────────────────

/// Fuzzy string search over a set of items.
///
/// Ported from Blender's `StringSearch<T>` (BLI_string_search.hh).
///
/// # Usage
/// ```rust
/// use mid_common::string::StringSearch;
///
/// let mut search: StringSearch<u32> = StringSearch::new();
/// search.add("Player Health",  0, 1.0);
/// search.add("Player Speed",   1, 1.0);
/// search.add("Enemy Position", 2, 0.5);
///
/// let results = search.query("player");
/// // results contains items 0 and 1, sorted by match quality
/// ```
pub struct StringSearch<T> {
    items: Vec<SearchItem<T>>,
    next_recent: u32,
}

impl<T> StringSearch<T> {
    /// Create an empty `StringSearch`.
    pub fn new() -> Self {
        Self { items: Vec::new(), next_recent: 0 }
    }

    /// Add an item with a display name and optional priority weight.
    ///
    /// Items with higher `weight` are sorted first when scores are equal.
    /// Call `mark_recent(name)` after the user selects an item to boost it.
    pub fn add(&mut self, name: &str, data: T, weight: f32) {
        self.items.push(SearchItem {
            data,
            name: name.to_string(),
            words: extract_words(name),
            weight,
            recent_time: 0,
        });
    }

    /// Bump the recency of the item with this name.
    /// Recent items sort above same-score matches.
    pub fn mark_recent(&mut self, name: &str) {
        self.next_recent += 1;
        let t = self.next_recent;
        for item in &mut self.items {
            if item.name == name {
                item.recent_time = t;
            }
        }
    }

    /// Filter and rank all items against `query`.
    ///
    /// Returns references to matching items, sorted best-first.
    /// Items that don't match at all are excluded.
    pub fn query(&self, query: &str) -> Vec<QueryResult<'_, T>> {
        if query.is_empty() {
            // Return all items sorted by weight + recency
            let mut results: Vec<QueryResult<'_, T>> = self.items.iter()
                .map(|item| QueryResult { item, score: 0 })
                .collect();
            results.sort_by(|a, b| {
                b.item.recent_time.cmp(&a.item.recent_time)
                    .then(b.item.weight.partial_cmp(&a.item.weight).unwrap_or(core::cmp::Ordering::Equal))
            });
            return results;
        }

        let query_words = extract_words(query);
        let mut results: Vec<QueryResult<'_, T>> = Vec::new();

        'item: for item in &self.items {
            // Every query word must match at least one item word
            let mut total_score = 0usize;

            for q_word in &query_words {
                // Find best match for this query word across item words
                let best = item.words.iter()
                    .filter_map(|w| fuzzy_match_score(q_word, w))
                    .min();

                match best {
                    Some(score) => total_score += score,
                    // Also try against the full item name for short queries
                    None => {
                        match fuzzy_match_score(q_word, &item.name.to_lowercase()) {
                            Some(score) => total_score += score + 1, // slight penalty
                            None => continue 'item,
                        }
                    }
                }
            }

            results.push(QueryResult { item, score: total_score });
        }

        // Sort: lower score = better match; break ties by recency, then weight
        results.sort_by(|a, b| {
            a.score.cmp(&b.score)
                .then(b.item.recent_time.cmp(&a.item.recent_time))
                .then(b.item.weight.partial_cmp(&a.item.weight)
                    .unwrap_or(core::cmp::Ordering::Equal))
        });

        results
    }

    /// Number of items in the search set.
    #[inline]
    pub fn len(&self) -> usize { self.items.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.items.is_empty() }

    /// Remove all items.
    #[inline]
    pub fn clear(&mut self) { self.items.clear(); }
}

impl<T> Default for StringSearch<T> {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────
// QueryResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single result from `StringSearch::query`.
pub struct QueryResult<'a, T> {
    /// The matched item.
    pub item: &'a SearchItem<T>,
    /// Match quality score. 0 = perfect. Higher = worse.
    pub score: usize,
}

impl<'a, T> QueryResult<'a, T> {
    /// Shorthand to get the item's data directly.
    #[inline]
    pub fn data(&self) -> &T { &self.item.data }

    /// Shorthand to get the item's display name.
    #[inline]
    pub fn name(&self) -> &str { &self.item.name }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dl_distance_basics() {
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("a", ""), 1);
        assert_eq!(damerau_levenshtein_distance("", "a"), 1);
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(damerau_levenshtein_distance("ca", "abc"), 2);
        // Transposition
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
    }

    #[test]
    fn fuzzy_exact() {
        assert_eq!(fuzzy_match_score("pos", "position"), Some(0));
        assert_eq!(fuzzy_match_score("Position", "Position"), Some(0));
    }

    #[test]
    fn fuzzy_no_match() {
        assert!(fuzzy_match_score("xyz999", "position").is_none());
    }

    #[test]
    fn subsequence_check() {
        assert!(is_subsequence("pos", "position"));
        assert!(is_subsequence("", "anything"));
        assert!(!is_subsequence("xyz", "position"));
    }

    #[test]
    fn word_extraction() {
        let words = extract_words("Player_Left.Arm");
        assert_eq!(words, vec!["player", "left", "arm"]);

        let words = extract_words("  hello  world  ");
        assert_eq!(words, vec!["hello", "world"]);
    }

    #[test]
    fn search_basic() {
        let mut s: StringSearch<u32> = StringSearch::new();
        s.add("Player Health", 0, 1.0);
        s.add("Player Speed",  1, 1.0);
        s.add("Enemy Count",   2, 1.0);

        let results = s.query("player");
        assert_eq!(results.len(), 2);
        assert!(*results[0].data() == 0 || *results[0].data() == 1);
    }

    #[test]
    fn search_empty_query_returns_all() {
        let mut s: StringSearch<u32> = StringSearch::new();
        s.add("A", 0, 1.0);
        s.add("B", 1, 1.0);
        assert_eq!(s.query("").len(), 2);
    }

    #[test]
    fn search_weight_tiebreak() {
        let mut s: StringSearch<u32> = StringSearch::new();
        s.add("transform", 0, 0.5);
        s.add("transform", 1, 2.0); // same name, higher weight

        let results = s.query("transform");
        assert_eq!(results.len(), 2);
        // Higher weight should sort first on equal score
        assert_eq!(*results[0].data(), 1);
    }
}
