// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "load.rs"
// ============================================================================
//! The `.mxui` schema reader: turns an [`MxValue`] node tree into a
//! [`mid_ui::Tree`] plus the names, bindings and metadata the file declared.

use std::collections::HashMap;

use mid_ui::{Anchor, Color32, Flow, Insets, NodeId, Real, Sizing, Style, Tree};

use crate::bind::BindKey;
use crate::error::{MxError, MxErrorKind};
use crate::value::MxValue;

/// The enums a file declares: enum name, then each variant with its integer
/// value. DixScript turns an enum value that passes through a QuickFunc
/// argument into a plain integer, so a loader needs these tables to turn it
/// back into a variant name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnumTables {
    tables: HashMap<String, Vec<(String, i64)>>,
}

impl EnumTables {
    /// No enums declared.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares an enum with its `(variant, value)` pairs.
    pub fn insert<I, N>(&mut self, enum_name: impl Into<String>, variants: I)
    where
        I: IntoIterator<Item = (N, i64)>,
        N: Into<String>,
    {
        let mut v: Vec<(String, i64)> = variants.into_iter().map(|(n, x)| (n.into(), x)).collect();
        // Sorted so that two variants sharing a value always decode the same way.
        v.sort();
        self.tables.insert(enum_name.into(), v);
    }

    /// Whether an enum with this name is declared.
    pub fn declares(&self, enum_name: &str) -> bool {
        self.tables.contains_key(enum_name)
    }

    /// The value of a variant of `enum_name`, if declared.
    pub fn variant_value(&self, enum_name: &str, variant: &str) -> Option<i64> {
        self.tables.get(enum_name)?.iter().find(|(n, _)| n == variant).map(|(_, x)| *x)
    }

    /// The variant of `enum_name` with this value, if any. When several
    /// variants share a value the first by name is returned.
    pub fn variant_name(&self, enum_name: &str, value: i64) -> Option<&str> {
        self.tables.get(enum_name)?.iter().find(|(_, x)| *x == value).map(|(n, _)| n.as_str())
    }
}

/// An enum value as written in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumRef {
    /// Enum type name, for example `UiKind`.
    pub type_name: String,
    /// Variant name, for example `Button`.
    pub field: String,
}

/// A data-binding declared on a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bind {
    /// Hashed key.
    pub key: BindKey,
    /// The name as written.
    pub name: String,
}

/// File-declared information about a node that the layout and paint core does
/// not use itself. Later layers (text, widgets, binding) read it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeMeta {
    /// The node's `id`.
    pub name: Option<String>,
    /// The node's `kind`.
    pub kind: Option<EnumRef>,
    /// The node's `text`.
    pub text: Option<String>,
    /// The node's `bind`.
    pub bind: Option<Bind>,
}

impl NodeMeta {
    fn is_empty(&self) -> bool {
        self.name.is_none() && self.kind.is_none() && self.text.is_none() && self.bind.is_none()
    }
}

/// The result of loading a `.mxui` file.
pub struct Loaded<R: Real = f32> {
    /// The widget tree. Run [`mid_ui::layout::compute`] before using rectangles.
    pub tree: Tree<R>,
    /// Nodes by their `id`.
    pub names: HashMap<String, NodeId>,
    /// Metadata for every node that declared any.
    pub meta: HashMap<NodeId, NodeMeta>,
}

struct Parsed<'a, R: Real> {
    style: Style<R>,
    meta: NodeMeta,
    children: &'a [MxValue],
}

/// Reads one root node value (an object) into a tree.
///
/// Fields of a node object, all optional: `id`, `kind`, `text`, `bind`,
/// `anchor` (`UiAnchor`), `flow` (`UiFlow`), `width`, `height` (a number, or
/// `{ fill = weight }`), `margin`, `padding` (a number, `[h, v]` or
/// `[left, top, right, bottom]`), `gap`, `visible`, `interactive`, `clip`,
/// `background` (a hex color), `radius` and `children` (an array of nodes).
/// Any other field is an error, so a typo is caught instead of ignored.
/// Integer enum values cannot be decoded here; see [`build_with`].
pub fn build<R: Real>(root: &MxValue) -> Result<Loaded<R>, MxError> {
    build_with(root, &EnumTables::new())
}

/// Like [`build`], decoding integer enum values through `tables`. Fields that
/// take an enum (`anchor`, `flow`, `kind`) accept either a real enum value or
/// an integer that names a variant of `UiAnchor`, `UiFlow` or `UiKind`.
pub fn build_with<R: Real>(root: &MxValue, tables: &EnumTables) -> Result<Loaded<R>, MxError> {
    let mut paths = Paths::new();
    let first = parse_node::<R>(root, tables).map_err(|e| e.prefixed("root"))?;
    let mut tree = Tree::new(first.style);
    let mut names = HashMap::new();
    let mut meta = HashMap::new();
    let mut binds: HashMap<u32, String> = HashMap::new();
    register(tree.root(), first.meta, &mut names, &mut meta, &mut binds)
        .map_err(|e| e.prefixed("root"))?;

    // Explicit stack: nesting depth in the file never grows the call stack.
    let mut stack: Vec<(NodeId, &MxValue, usize)> = Vec::new();
    push_children(&mut stack, &mut paths, tree.root(), first.children, Paths::ROOT);
    while let Some((parent, value, seg)) = stack.pop() {
        let node = parse_node::<R>(value, tables).map_err(|e| e.prefixed(&paths.render(seg)))?;
        let id = tree
            .add_child(parent, node.style)
            .ok_or_else(|| MxError::new(paths.render(seg), MxErrorKind::TreeFull))?;
        register(id, node.meta, &mut names, &mut meta, &mut binds)
            .map_err(|e| e.prefixed(&paths.render(seg)))?;
        push_children(&mut stack, &mut paths, id, node.children, seg);
    }
    Ok(Loaded { tree, names, meta })
}

// Node positions as parent links, so a path string is only built when an error
// needs one. Building it per node would cost time quadratic in nesting depth.
struct Paths {
    links: Vec<(usize, usize)>,
}

impl Paths {
    const ROOT: usize = 0;

    fn new() -> Self {
        Self { links: vec![(usize::MAX, 0)] }
    }

    fn child(&mut self, parent: usize, index: usize) -> usize {
        self.links.push((parent, index));
        self.links.len() - 1
    }

    fn render(&self, mut seg: usize) -> String {
        let mut indices = Vec::new();
        while seg != Self::ROOT {
            let (parent, index) = self.links[seg];
            indices.push(index);
            seg = parent;
        }
        let mut out = String::from("root");
        for index in indices.iter().rev() {
            out.push_str(&format!(".children[{index}]"));
        }
        out
    }
}

fn push_children<'a>(
    stack: &mut Vec<(NodeId, &'a MxValue, usize)>,
    paths: &mut Paths,
    parent: NodeId,
    children: &'a [MxValue],
    parent_seg: usize,
) {
    // Reversed so the first child is popped, and therefore added, first.
    for (i, child) in children.iter().enumerate().rev() {
        let seg = paths.child(parent_seg, i);
        stack.push((parent, child, seg));
    }
}

fn register(
    id: NodeId,
    node_meta: NodeMeta,
    names: &mut HashMap<String, NodeId>,
    meta: &mut HashMap<NodeId, NodeMeta>,
    binds: &mut HashMap<u32, String>,
) -> Result<(), MxError> {
    if let Some(name) = &node_meta.name {
        if names.insert(name.clone(), id).is_some() {
            return Err(MxError::field("id", MxErrorKind::DuplicateName(name.clone())));
        }
    }
    if let Some(bind) = &node_meta.bind {
        match binds.get(&bind.key.0) {
            Some(first) if *first != bind.name => {
                return Err(MxError::field(
                    "bind",
                    MxErrorKind::BindCollision { first: first.clone(), second: bind.name.clone() },
                ));
            }
            Some(_) => {}
            None => {
                binds.insert(bind.key.0, bind.name.clone());
            }
        }
    }
    if !node_meta.is_empty() {
        meta.insert(id, node_meta);
    }
    Ok(())
}

fn parse_node<'a, R: Real>(v: &'a MxValue, tables: &EnumTables) -> Result<Parsed<'a, R>, MxError> {
    let entries = match v {
        MxValue::Object(entries) => entries,
        other => {
            return Err(MxError::new(
                "",
                MxErrorKind::WrongType { expected: "object", found: other.type_name() },
            ))
        }
    };
    let mut style = Style::<R>::default();
    let mut meta = NodeMeta::default();
    let mut children: &[MxValue] = &[];
    for (key, val) in entries {
        let f = key.as_str();
        match f {
            "id" => meta.name = Some(string(val, f)?),
            "kind" => meta.kind = Some(kind_ref(val, tables, f)?),
            "text" => meta.text = Some(string(val, f)?),
            "bind" => {
                let name = string(val, f)?;
                meta.bind = Some(Bind { key: BindKey::of(&name), name });
            }
            "anchor" => style.anchor = anchor(val, tables, f)?,
            "flow" => style.flow = flow(val, tables, f)?,
            "width" => style.width = sizing(val, f)?,
            "height" => style.height = sizing(val, f)?,
            "margin" => style.margin = insets(val, f)?,
            "padding" => style.padding = insets(val, f)?,
            "gap" => style.gap = number(val, f)?,
            "visible" => style.visible = boolean(val, f)?,
            "interactive" => style.interactive = boolean(val, f)?,
            "clip" => style.clip = boolean(val, f)?,
            "background" => style.background = color(val, f)?,
            "radius" => style.radius = number(val, f)?,
            "children" => match val {
                MxValue::Array(items) => children = items,
                other => return Err(wrong(f, "array", other)),
            },
            _ => return Err(MxError::field(f, MxErrorKind::UnknownField(f.to_string()))),
        }
    }
    Ok(Parsed { style, meta, children })
}

fn wrong(field: &str, expected: &'static str, found: &MxValue) -> MxError {
    MxError::field(field, MxErrorKind::WrongType { expected, found: found.type_name() })
}

fn string(v: &MxValue, field: &str) -> Result<String, MxError> {
    match v {
        MxValue::Str(s) => Ok(s.clone()),
        other => Err(wrong(field, "string", other)),
    }
}

fn boolean(v: &MxValue, field: &str) -> Result<bool, MxError> {
    match v {
        MxValue::Bool(b) => Ok(*b),
        other => Err(wrong(field, "bool", other)),
    }
}

fn number<R: Real>(v: &MxValue, field: &str) -> Result<R, MxError> {
    match v.as_f64() {
        Some(n) => Ok(R::from_f64(n)),
        None => Err(wrong(field, "number", v)),
    }
}

// `kind` is metadata: a real enum value of any type is kept as written. An
// integer (the form a QuickFunc argument takes) is decoded as `UiKind`.
fn kind_ref(v: &MxValue, tables: &EnumTables, field: &str) -> Result<EnumRef, MxError> {
    match v {
        MxValue::Enum { type_name, field: variant } => {
            Ok(EnumRef { type_name: type_name.clone(), field: variant.clone() })
        }
        MxValue::Int(_) => Ok(EnumRef {
            type_name: "UiKind".to_string(),
            field: decode(v, "UiKind", tables, field)?.to_string(),
        }),
        other => Err(wrong(field, "enum", other)),
    }
}

// The variant name of a value for the enum called `expected`: either a real
// enum value of that type, or an integer decoded through the file's enums.
fn variant_of<'a>(
    v: &'a MxValue,
    expected: &'static str,
    tables: &'a EnumTables,
    field: &str,
) -> Result<&'a str, MxError> {
    match v {
        MxValue::Enum { type_name, field: variant } if type_name == expected => Ok(variant),
        MxValue::Enum { type_name, .. } => Err(MxError::field(
            field,
            MxErrorKind::WrongEnum { expected, found: type_name.clone() },
        )),
        MxValue::Int(_) => decode(v, expected, tables, field),
        other => Err(wrong(field, "enum", other)),
    }
}

fn decode<'a>(
    v: &MxValue,
    expected: &'static str,
    tables: &'a EnumTables,
    field: &str,
) -> Result<&'a str, MxError> {
    let value = match v {
        MxValue::Int(i) => *i,
        other => return Err(wrong(field, "enum", other)),
    };
    if !tables.declares(expected) {
        return Err(MxError::field(field, MxErrorKind::EnumNotDeclared(expected)));
    }
    tables
        .variant_name(expected, value)
        .ok_or_else(|| unknown_variant(field, expected, &value.to_string()))
}

fn unknown_variant(field: &str, enum_name: &'static str, found: &str) -> MxError {
    MxError::field(field, MxErrorKind::UnknownVariant { enum_name, found: found.to_string() })
}

fn anchor(v: &MxValue, tables: &EnumTables, field: &str) -> Result<Anchor, MxError> {
    Ok(match variant_of(v, "UiAnchor", tables, field)? {
        "TopLeft" => Anchor::TopLeft,
        "Top" => Anchor::Top,
        "TopRight" => Anchor::TopRight,
        "Left" => Anchor::Left,
        "Center" => Anchor::Center,
        "Right" => Anchor::Right,
        "BottomLeft" => Anchor::BottomLeft,
        "Bottom" => Anchor::Bottom,
        "BottomRight" => Anchor::BottomRight,
        other => return Err(unknown_variant(field, "UiAnchor", other)),
    })
}

fn flow(v: &MxValue, tables: &EnumTables, field: &str) -> Result<Flow, MxError> {
    Ok(match variant_of(v, "UiFlow", tables, field)? {
        "Stack" => Flow::Stack,
        "Row" => Flow::Row,
        "Column" => Flow::Column,
        other => return Err(unknown_variant(field, "UiFlow", other)),
    })
}

fn sizing<R: Real>(v: &MxValue, field: &str) -> Result<Sizing<R>, MxError> {
    if let Some(n) = v.as_f64() {
        return Ok(Sizing::Fixed(R::from_f64(n)));
    }
    match v {
        MxValue::Object(entries) => match entries.as_slice() {
            [(key, weight)] if key == "fill" => Ok(Sizing::Fill(number(weight, field)?)),
            _ => Err(MxError::field(field, MxErrorKind::BadSizing)),
        },
        other => Err(wrong(field, "number or fill object", other)),
    }
}

fn insets<R: Real>(v: &MxValue, field: &str) -> Result<Insets<R>, MxError> {
    if let Some(n) = v.as_f64() {
        return Ok(Insets::all(R::from_f64(n)));
    }
    match v {
        MxValue::Array(items) => match items.as_slice() {
            [h, vert] => Ok(Insets::symmetric(number(h, field)?, number(vert, field)?)),
            [l, t, r, b] => Ok(Insets::new(
                number(l, field)?,
                number(t, field)?,
                number(r, field)?,
                number(b, field)?,
            )),
            other => Err(MxError::field(field, MxErrorKind::BadInsets(other.len()))),
        },
        other => Err(wrong(field, "number or array", other)),
    }
}

fn color(v: &MxValue, field: &str) -> Result<Color32, MxError> {
    match v {
        MxValue::Color(s) => parse_color(s)
            .ok_or_else(|| MxError::field(field, MxErrorKind::BadColor(s.clone()))),
        other => Err(wrong(field, "color", other)),
    }
}

// `#RGB`, `#RGBA`, `#RRGGBB` or `#RRGGBBAA`. The short forms double each digit.
fn parse_color(s: &str) -> Option<Color32> {
    let hex = s.strip_prefix('#')?;
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    match hex.len() {
        3 | 4 => {
            let mut full = String::with_capacity(1 + hex.len() * 2);
            full.push('#');
            for c in hex.chars() {
                full.push(c);
                full.push(c);
            }
            Color32::from_hex(&full)
        }
        6 | 8 => Color32::from_hex(s),
        _ => None,
    }
}
