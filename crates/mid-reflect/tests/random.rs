//! Randomized checks: every value survives text, arbitrary text never panics
//! the parser, every f32 survives, random structs survive.

use mid_reflect::{impl_reflect_enum, impl_reflect_struct, Reflect, Value};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_string(rng: &mut Lcg) -> String {
    const PIECES: &[&str] = &[
        "a", "Z", "0", " ", "\"", "\\", "/", "\n", "\t", "\u{1}", "\u{1f}", "é", "😀", "\u{2028}",
        "\u{7f}", "key", "{", "[", ",",
    ];
    (0..rng.below(8))
        .map(|_| PIECES[rng.below(PIECES.len())])
        .collect()
}

fn random_value(rng: &mut Lcg, depth: usize) -> Value {
    let kinds = if depth >= 4 { 6 } else { 8 };
    match rng.below(kinds) {
        0 => Value::Null,
        1 => Value::Bool(rng.below(2) == 0),
        2 => Value::Int(rng.next() as i64 - (1 << 52)),
        3 => Value::from_u64(rng.next() << 11 | rng.next() & 0x7ff),
        4 => {
            let f = f64::from_bits(rng.next() << 11 | rng.next() & 0x7ff);
            Value::Float(if f.is_finite() { f } else { 0.5 })
        }
        5 => Value::Str(random_string(rng)),
        6 => Value::List(
            (0..rng.below(4))
                .map(|_| random_value(rng, depth + 1))
                .collect(),
        ),
        _ => {
            let mut entries: Vec<(String, Value)> = Vec::new();
            for i in 0..rng.below(4) {
                // Keys made unique by their position.
                let key = format!("{}#{i}", random_string(rng));
                entries.push((key, random_value(rng, depth + 1)));
            }
            Value::Map(entries)
        }
    }
}

/// `Value` equality with floats compared by bits, so -0.0 and 0.0 differ.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        (Value::Map(x), Value::Map(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.0 == q.0 && same(&p.1, &q.1))
        }
        _ => a == b,
    }
}

#[test]
fn random_values_survive_compact_and_pretty_text() {
    let mut rng = Lcg(1);
    let (mut deep, mut big_ints, mut strings_with_escapes) = (0, 0, 0);
    for case in 0..3000 {
        let v = random_value(&mut rng, 0);
        for text in [v.to_json(), v.to_json_pretty()] {
            let back =
                Value::from_json(&text).unwrap_or_else(|e| panic!("case {case}: {e}\n{text}"));
            assert!(same(&v, &back), "case {case}:\n{text}\n{v:?}\n{back:?}");
        }
        let text = v.to_json();
        deep += usize::from(text.matches('[').count() + text.matches('{').count() >= 4);
        big_ints += usize::from(text.contains("e") || text.len() > 40);
        strings_with_escapes += usize::from(text.contains("\\u") || text.contains("\\\""));
    }
    assert!(deep >= 60, "too few nested values: {deep}");
    assert!(big_ints >= 600, "too few large values: {big_ints}");
    assert!(
        strings_with_escapes >= 200,
        "too few escaped strings: {strings_with_escapes}"
    );
}

#[test]
fn arbitrary_text_never_panics_the_parser_and_what_parses_re_emits() {
    const ALPHABET: &[&str] = &[
        "{", "}", "[", "]", "\"", ",", ":", "0", "1", "9", "-", "+", ".", "e", "E", "\\", "u", "d",
        "8", "n", "t", "f", "a", "l", "s", "r", " ", "\n", "é", "😀", "\u{0}", "\u{1f}", "/",
    ];
    let mut rng = Lcg(7);
    let (mut parsed, mut refused) = (0, 0);
    for _ in 0..60_000 {
        let text: String = (0..rng.below(14))
            .map(|_| ALPHABET[rng.below(ALPHABET.len())])
            .collect();
        match Value::from_json(&text) {
            Ok(v) => {
                parsed += 1;
                let again = Value::from_json(&v.to_json())
                    .unwrap_or_else(|e| panic!("{text:?} parsed, but its output did not: {e}"));
                assert!(same(&v, &again), "{text:?}");
            }
            Err(e) => {
                refused += 1;
                assert!(e.offset <= text.len(), "{text:?}: offset {}", e.offset);
            }
        }
    }
    assert!(parsed >= 300, "the fuzz parsed too little: {parsed}");
    assert!(refused >= 30_000, "the fuzz refused too little: {refused}");
}

#[test]
fn mutated_valid_json_never_panics_the_parser_and_what_parses_re_emits() {
    const INSERTS: &[&str] = &[
        "{", "}", "[", "]", "\"", ",", ":", "0", "-", ".", "e", "\\", "u", " ", "n", "é", "\u{0}",
    ];
    let mut rng = Lcg(21);
    let (mut parsed, mut refused, mut changed_value) = (0, 0, 0);
    for _ in 0..40_000 {
        let original = random_value(&mut rng, 0);
        let mut text: Vec<char> = original.to_json().chars().collect();
        for _ in 0..1 + rng.below(3) {
            if text.is_empty() {
                break;
            }
            let at = rng.below(text.len());
            match rng.below(3) {
                0 => {
                    text.remove(at);
                }
                1 => text.insert(
                    at,
                    INSERTS[rng.below(INSERTS.len())].chars().next().unwrap(),
                ),
                _ => text[at] = INSERTS[rng.below(INSERTS.len())].chars().next().unwrap(),
            }
        }
        let text: String = text.into_iter().collect();
        match Value::from_json(&text) {
            Ok(v) => {
                parsed += 1;
                changed_value += usize::from(!same(&v, &original));
                let again = Value::from_json(&v.to_json())
                    .unwrap_or_else(|e| panic!("{text:?} parsed, but its output did not: {e}"));
                assert!(same(&v, &again), "{text:?}");
            }
            Err(e) => {
                refused += 1;
                assert!(e.offset <= text.len(), "{text:?}: offset {}", e.offset);
            }
        }
    }
    assert!(parsed >= 3_000, "too few mutations still parsed: {parsed}");
    assert!(refused >= 15_000, "too few mutations refused: {refused}");
    assert!(
        changed_value >= 2_500,
        "too few parsed mutations differ: {changed_value}"
    );
}

#[test]
fn every_kind_of_truncation_of_valid_json_is_refused_or_valid() {
    let mut rng = Lcg(11);
    for _ in 0..300 {
        let text = random_value(&mut rng, 0).to_json();
        for end in 0..text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            // Either it parses (a shorter valid document, such as a number)
            // or it is refused; it never panics.
            let _ = Value::from_json(&text[..end]);
        }
    }
}

#[test]
fn every_f32_value_survives_both_directly_and_through_text() {
    let mut rng = Lcg(3);
    let (mut tiny, mut huge, mut checked) = (0, 0, 0);
    // The edges, then random bit patterns.
    let mut bits: Vec<u32> = vec![
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0x3f80_0000,
    ];
    bits.extend((0..200_000).map(|_| (rng.next() >> 8) as u32 ^ ((rng.next() as u32) << 16)));
    for b in bits {
        let x = f32::from_bits(b);
        if x.is_nan() {
            continue;
        }
        checked += 1;
        tiny += usize::from(x != 0.0 && x.abs() < 1e-20);
        huge += usize::from(x.abs() > 1e20 && x.is_finite());
        let v = x.to_value();
        assert_eq!(f32::from_value(&v).unwrap().to_bits(), b, "direct {x:e}");
        let back = Value::from_json(&v.to_json()).unwrap();
        assert_eq!(f32::from_value(&back).unwrap().to_bits(), b, "text {x:e}");
    }
    assert!(
        checked > 190_000 && tiny > 1_000 && huge > 1_000,
        "{checked} {tiny} {huge}"
    );
}

#[test]
fn every_f64_value_survives_text() {
    let mut rng = Lcg(5);
    let mut checked = 0;
    for _ in 0..200_000 {
        let x = f64::from_bits(rng.next() << 11 | rng.next() & 0x7ff);
        if x.is_nan() {
            continue;
        }
        checked += 1;
        let v = x.to_value();
        let back = Value::from_json(&v.to_json()).unwrap();
        assert_eq!(
            f64::from_value(&back).unwrap().to_bits(),
            x.to_bits(),
            "{x:e}"
        );
    }
    assert!(checked > 190_000);
}

// `u64::try_from(u64)` cannot fail, which is the point of listing every type.
#[allow(irrefutable_let_patterns)]
#[test]
fn integers_survive_in_every_type_that_can_hold_them() {
    let mut rng = Lcg(9);
    for _ in 0..50_000 {
        let n = rng.next() << 11 | rng.next() & 0x7ff;
        macro_rules! check_unsigned {
            ($($t:ty),*) => {$(
                if let Ok(x) = <$t>::try_from(n) {
                    let v = Value::from_json(&x.to_value().to_json()).unwrap();
                    assert_eq!(<$t>::from_value(&v).unwrap(), x);
                } else {
                    assert!(<$t>::from_value(&Value::from_u64(n)).is_err());
                }
            )*};
        }
        check_unsigned!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
        let s = n as i64;
        macro_rules! check_signed {
            ($($t:ty),*) => {$(
                if let Ok(x) = <$t>::try_from(s) {
                    let v = Value::from_json(&x.to_value().to_json()).unwrap();
                    assert_eq!(<$t>::from_value(&v).unwrap(), x);
                } else {
                    assert!(<$t>::from_value(&Value::Int(s)).is_err());
                }
            )*};
        }
        check_signed!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Kind {
    A,
    B,
    C,
}
impl_reflect_enum!("t::Kind" => Kind { A, B, C });

#[derive(Debug, PartialEq, Clone)]
struct Leaf {
    n: i32,
    f: f32,
    k: Kind,
}
impl_reflect_struct!("t::Leaf" => Leaf { n: i32, f: f32, k: Kind });

#[derive(Debug, PartialEq, Clone)]
struct Node {
    id: u64,
    name: String,
    leaves: Vec<Leaf>,
    extra: Option<Leaf>,
    grid: [i16; 3],
    ok: bool,
}
impl_reflect_struct!("t::Node" => Node {
    id: u64,
    name: String,
    leaves: Vec<Leaf>,
    extra: Option<Leaf>,
    grid: [i16; 3],
    ok: bool,
});

fn random_leaf(rng: &mut Lcg) -> Leaf {
    Leaf {
        n: rng.next() as i32,
        f: loop {
            let f = f32::from_bits(rng.next() as u32);
            if f.is_finite() {
                break f;
            }
        },
        k: [Kind::A, Kind::B, Kind::C][rng.below(3)],
    }
}

#[test]
fn random_structs_survive_text() {
    let mut rng = Lcg(13);
    for case in 0..3000 {
        let node = Node {
            id: rng.next() << 11 | rng.next() & 0x7ff,
            name: random_string(&mut rng),
            leaves: (0..rng.below(4)).map(|_| random_leaf(&mut rng)).collect(),
            extra: (rng.below(2) == 0).then(|| random_leaf(&mut rng)),
            grid: [rng.next() as i16, rng.next() as i16, rng.next() as i16],
            ok: rng.below(2) == 0,
        };
        let text = node.to_value().to_json();
        let back = Node::from_value(&Value::from_json(&text).unwrap())
            .unwrap_or_else(|e| panic!("case {case}: {e}\n{text}"));
        assert_eq!(back, node, "case {case}:\n{text}");
    }
}
