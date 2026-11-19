// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "json.rs"
// ============================================================================
//! JSON text for [`Value`]: strict RFC 8259 in, compact or indented out.
//!
//! Non-finite floats have no JSON form, so they are written as the strings
//! `"NaN"`, `"Infinity"` and `"-Infinity"`, which the float types read back.

use std::fmt;

use crate::value::Value;

/// Nesting depth allowed when parsing, unless asked otherwise.
pub const DEFAULT_MAX_DEPTH: usize = 128;

/// What was wrong with a piece of JSON text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonErrorKind {
    /// The text is empty or only whitespace.
    EmptyInput,
    /// The text ended in the middle of a value.
    UnexpectedEnd,
    /// A character that cannot start or continue what was being read.
    UnexpectedChar(char),
    /// A value was read, then something other than whitespace followed.
    TrailingCharacters,
    /// A malformed number (`01`, `1.`, `+1`, `.5`, `1e`).
    InvalidNumber,
    /// A backslash followed by something that is not an escape.
    InvalidEscape,
    /// A bad `\u` escape, or a surrogate without its pair.
    InvalidUnicode,
    /// A raw control character inside a string.
    ControlCharInString,
    /// An object with the same key twice.
    DuplicateKey(String),
    /// Arrays and objects nested deeper than the limit.
    TooDeep,
}

/// A JSON error with the byte offset where it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    /// Byte offset into the text.
    pub offset: usize,
    /// What was wrong.
    pub kind: JsonErrorKind,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match &self.kind {
            JsonErrorKind::EmptyInput => "empty input".to_string(),
            JsonErrorKind::UnexpectedEnd => "unexpected end of input".to_string(),
            JsonErrorKind::UnexpectedChar(c) => format!("unexpected character {c:?}"),
            JsonErrorKind::TrailingCharacters => "characters after the value".to_string(),
            JsonErrorKind::InvalidNumber => "invalid number".to_string(),
            JsonErrorKind::InvalidEscape => "invalid escape".to_string(),
            JsonErrorKind::InvalidUnicode => "invalid unicode escape".to_string(),
            JsonErrorKind::ControlCharInString => "control character in string".to_string(),
            JsonErrorKind::DuplicateKey(k) => format!("duplicate key {k:?}"),
            JsonErrorKind::TooDeep => "nesting too deep".to_string(),
        };
        write!(f, "JSON error at byte {}: {what}", self.offset)
    }
}

impl std::error::Error for JsonError {}

impl Value {
    /// Compact JSON text, no whitespace.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        write_value(&mut out, self, None, 0);
        out
    }

    /// JSON text indented by two spaces per level.
    pub fn to_json_pretty(&self) -> String {
        let mut out = String::new();
        write_value(&mut out, self, Some(2), 0);
        out
    }

    /// Parses JSON text, allowing [`DEFAULT_MAX_DEPTH`] levels of nesting.
    pub fn from_json(text: &str) -> Result<Value, JsonError> {
        Value::from_json_with_depth(text, DEFAULT_MAX_DEPTH)
    }

    /// Parses JSON text, allowing `max_depth` levels of nesting.
    pub fn from_json_with_depth(text: &str, max_depth: usize) -> Result<Value, JsonError> {
        let mut parser = Parser {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            max_depth,
        };
        parser.skip_whitespace();
        if parser.pos >= parser.bytes.len() {
            return Err(parser.error(JsonErrorKind::EmptyInput));
        }
        let value = parser.parse_value(0)?;
        parser.skip_whitespace();
        if parser.pos < parser.bytes.len() {
            return Err(parser.error(JsonErrorKind::TrailingCharacters));
        }
        Ok(value)
    }
}

fn write_value(out: &mut String, value: &Value, indent: Option<usize>, level: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::UInt(u) => out.push_str(&u.to_string()),
        Value::Float(f) => write_float(out, *f),
        Value::Str(s) => write_string(out, s),
        Value::List(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, indent, level + 1);
                write_value(out, item, indent, level + 1);
            }
            newline(out, indent, level);
            out.push(']');
        }
        Value::Map(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, indent, level + 1);
                write_string(out, key);
                out.push(':');
                if indent.is_some() {
                    out.push(' ');
                }
                write_value(out, item, indent, level + 1);
            }
            newline(out, indent, level);
            out.push('}');
        }
    }
}

fn newline(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(width) = indent {
        out.push('\n');
        for _ in 0..width * level {
            out.push(' ');
        }
    }
}

fn write_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("\"NaN\"");
    } else if f == f64::INFINITY {
        out.push_str("\"Infinity\"");
    } else if f == f64::NEG_INFINITY {
        out.push_str("\"-Infinity\"");
    } else {
        // `{:?}` is the shortest text that reads back as the same f64, keeps
        // a `.0` on whole numbers (so a float stays a float) and switches to
        // an exponent for very large and very small magnitudes; all of it is
        // valid JSON.
        out.push_str(&format!("{f:?}"));
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    max_depth: usize,
}

impl Parser<'_> {
    fn error(&self, kind: JsonErrorKind) -> JsonError {
        JsonError {
            offset: self.pos,
            kind,
        }
    }

    fn unexpected(&self) -> JsonError {
        match self.text[self.pos..].chars().next() {
            Some(c) => self.error(JsonErrorKind::UnexpectedChar(c)),
            None => self.error(JsonErrorKind::UnexpectedEnd),
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect_literal(&mut self, word: &str, value: Value) -> Result<Value, JsonError> {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(value)
        } else {
            // Report the first byte that differs from the word.
            let rest = &self.bytes[self.pos..];
            let same = rest
                .iter()
                .zip(word.as_bytes())
                .take_while(|(a, b)| a == b)
                .count();
            self.pos += same;
            Err(self.unexpected())
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value, JsonError> {
        match self.bytes.get(self.pos) {
            None => Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(b'n') => self.expect_literal("null", Value::Null),
            Some(b't') => self.expect_literal("true", Value::Bool(true)),
            Some(b'f') => self.expect_literal("false", Value::Bool(false)),
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'{') => self.parse_object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.unexpected()),
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value, JsonError> {
        if depth > self.max_depth {
            return Err(self.error(JsonErrorKind::TooDeep));
        }
        self.pos += 1; // [
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Value::List(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.parse_value(depth)?);
            self.skip_whitespace();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::List(items));
                }
                _ => return Err(self.unexpected()),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value, JsonError> {
        if depth > self.max_depth {
            return Err(self.error(JsonErrorKind::TooDeep));
        }
        self.pos += 1; // {
        let mut entries: Vec<(String, Value)> = Vec::new();
        let mut key_offsets: Vec<usize> = Vec::new();
        self.skip_whitespace();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Value::Map(entries));
        }
        loop {
            self.skip_whitespace();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return Err(self.unexpected());
            }
            key_offsets.push(self.pos);
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.bytes.get(self.pos) != Some(&b':') {
                return Err(self.unexpected());
            }
            self.pos += 1;
            self.skip_whitespace();
            let value = self.parse_value(depth)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.unexpected()),
            }
        }
        // Duplicate keys, found by sorting, so a hostile object with many
        // keys costs n log n and not n squared.
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|a, b| entries[*a].0.cmp(&entries[*b].0));
        for pair in order.windows(2) {
            if entries[pair[0]].0 == entries[pair[1]].0 {
                let later = pair[0].max(pair[1]);
                return Err(JsonError {
                    offset: key_offsets[later],
                    kind: JsonErrorKind::DuplicateKey(entries[later].0.clone()),
                });
            }
        }
        Ok(Value::Map(entries))
    }

    fn parse_number(&mut self) -> Result<Value, JsonError> {
        let start = self.pos;
        let invalid = |p: &Parser| JsonError {
            offset: p.pos,
            kind: JsonErrorKind::InvalidNumber,
        };
        let negative = self.bytes[self.pos] == b'-';
        if negative {
            self.pos += 1;
        }
        match self.bytes.get(self.pos) {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    return Err(invalid(self));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(invalid(self)),
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            is_float = true;
            self.pos += 1;
            let digits_from = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits_from {
                return Err(invalid(self));
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let digits_from = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits_from {
                return Err(invalid(self));
            }
        }
        let text = &self.text[start..self.pos];
        if !is_float {
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Value::Int(i));
            }
            if !negative {
                if let Ok(u) = text.parse::<u64>() {
                    return Ok(Value::UInt(u));
                }
            }
        }
        // A number too large for an f64 parses to infinity; JSON has no
        // infinity, and the value would not survive being written out again.
        match text.parse::<f64>() {
            Ok(f) if f.is_finite() => Ok(Value::Float(f)),
            _ => Err(invalid(self)),
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, JsonError> {
        let mut value = 0u32;
        for _ in 0..4 {
            let Some(&b) = self.bytes.get(self.pos) else {
                return Err(self.error(JsonErrorKind::InvalidUnicode));
            };
            let digit = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(self.error(JsonErrorKind::InvalidUnicode)),
            };
            value = value * 16 + u32::from(digit);
            self.pos += 1;
        }
        Ok(value)
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        let mut run_start = self.pos;
        loop {
            let Some(&b) = self.bytes.get(self.pos) else {
                return Err(self.error(JsonErrorKind::UnexpectedEnd));
            };
            match b {
                b'"' => {
                    out.push_str(&self.text[run_start..self.pos]);
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    out.push_str(&self.text[run_start..self.pos]);
                    self.pos += 1;
                    let Some(&e) = self.bytes.get(self.pos) else {
                        return Err(self.error(JsonErrorKind::UnexpectedEnd));
                    };
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.parse_hex4()?;
                            let c = if (0xD800..0xDC00).contains(&first) {
                                if self.bytes.get(self.pos) != Some(&b'\\')
                                    || self.bytes.get(self.pos + 1) != Some(&b'u')
                                {
                                    return Err(self.error(JsonErrorKind::InvalidUnicode));
                                }
                                self.pos += 2;
                                let second = self.parse_hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err(self.error(JsonErrorKind::InvalidUnicode));
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&first) {
                                return Err(self.error(JsonErrorKind::InvalidUnicode));
                            } else {
                                first
                            };
                            match char::from_u32(c) {
                                Some(c) => out.push(c),
                                None => return Err(self.error(JsonErrorKind::InvalidUnicode)),
                            }
                        }
                        _ => {
                            self.pos -= 1;
                            return Err(self.error(JsonErrorKind::InvalidEscape));
                        }
                    }
                    run_start = self.pos;
                }
                0..=0x1f => return Err(self.error(JsonErrorKind::ControlCharInString)),
                _ => self.pos += 1,
            }
        }
    }
}
