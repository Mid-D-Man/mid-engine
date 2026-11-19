//! `Value`, its paths, and JSON text in and out.

use mid_reflect::{ErrorKind, JsonErrorKind, PathSeg, Value};

fn parse(text: &str) -> Value {
    Value::from_json(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
}

fn parse_err(text: &str) -> mid_reflect::JsonError {
    match Value::from_json(text) {
        Ok(v) => panic!("{text:?} parsed as {v:?}"),
        Err(e) => e,
    }
}

#[test]
fn literals_and_whitespace() {
    assert_eq!(parse("null"), Value::Null);
    assert_eq!(parse("true"), Value::Bool(true));
    assert_eq!(parse("false"), Value::Bool(false));
    assert_eq!(parse(" \t\r\n null \n"), Value::Null);
}

#[test]
fn integers_floats_and_their_limits() {
    assert_eq!(parse("0"), Value::Int(0));
    assert_eq!(parse("-0"), Value::Int(0));
    assert_eq!(parse("42"), Value::Int(42));
    assert_eq!(parse("-9223372036854775808"), Value::Int(i64::MIN));
    assert_eq!(parse("9223372036854775807"), Value::Int(i64::MAX));
    assert_eq!(parse("9223372036854775808"), Value::UInt(1 << 63));
    assert_eq!(parse("18446744073709551615"), Value::UInt(u64::MAX));
    assert_eq!(
        parse("18446744073709551616"),
        Value::Float(18446744073709551616.0)
    );
    assert_eq!(
        parse("-9223372036854775809"),
        Value::Float(-9223372036854775809.0)
    );
    assert_eq!(parse("1.5"), Value::Float(1.5));
    assert_eq!(parse("1e3"), Value::Float(1000.0));
    assert_eq!(parse("1E+3"), Value::Float(1000.0));
    assert_eq!(parse("2.5e-2"), Value::Float(0.025));
    match parse("-0.0") {
        Value::Float(f) => assert!(f == 0.0 && f.is_sign_negative()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_number_too_large_for_an_f64_is_refused_and_one_too_small_is_zero() {
    for text in ["1e999", "-1e999", "1958e3538141297", "1e309", "2e308"] {
        let e = parse_err(text);
        assert_eq!(e.kind, JsonErrorKind::InvalidNumber, "{text}");
    }
    assert_eq!(parse("1.7976931348623157e308"), Value::Float(f64::MAX));
    assert_eq!(parse("1e-999"), Value::Float(0.0));
    assert_eq!(
        parse("123456789012345678901234567890"),
        Value::Float(1.2345678901234568e29)
    );
    assert!(Value::from_json("[1e999]").is_err());
}

#[test]
fn malformed_numbers_are_refused() {
    for text in [
        "01", "1.", ".5", "+1", "-", "1e", "1e+", "--1", "0x10", "1.2.3", "1 2", "-a", "00", "1.e3",
    ] {
        assert!(Value::from_json(text).is_err(), "{text:?} should not parse");
    }
}

#[test]
fn string_escapes() {
    assert_eq!(parse(r#""a\"b\\c\/d""#), Value::from("a\"b\\c/d"));
    assert_eq!(parse(r#""\b\f\n\r\t""#), Value::from("\u{8}\u{c}\n\r\t"));
    assert_eq!(parse(r#""\u00e9\u0041""#), Value::from("éA"));
    assert_eq!(parse(r#""\ud83d\ude00""#), Value::from("😀"));
    assert_eq!(parse("\"héllo 😀\""), Value::from("héllo 😀"));
    assert_eq!(parse(r#""""#), Value::from(""));
}

#[test]
fn bad_strings_are_refused() {
    assert_eq!(
        parse_err(r#""\ud83d""#).kind,
        JsonErrorKind::InvalidUnicode,
        "high surrogate alone"
    );
    assert_eq!(
        parse_err(r#""\ude00""#).kind,
        JsonErrorKind::InvalidUnicode,
        "low surrogate alone"
    );
    assert_eq!(
        parse_err(r#""\ud83d\u0041""#).kind,
        JsonErrorKind::InvalidUnicode,
        "high surrogate then not a low one"
    );
    assert_eq!(parse_err(r#""\x""#).kind, JsonErrorKind::InvalidEscape);
    assert_eq!(parse_err(r#""\u12""#).kind, JsonErrorKind::InvalidUnicode);
    assert_eq!(
        parse_err("\"a\nb\"").kind,
        JsonErrorKind::ControlCharInString
    );
    assert_eq!(parse_err("\"abc").kind, JsonErrorKind::UnexpectedEnd);
    assert_eq!(parse_err("\"abc\\").kind, JsonErrorKind::UnexpectedEnd);
}

#[test]
fn containers() {
    assert_eq!(parse("[]"), Value::List(vec![]));
    assert_eq!(parse("{}"), Value::Map(vec![]));
    assert_eq!(
        parse(r#" [1, [2, 3], {"a": null}] "#),
        Value::List(vec![
            Value::Int(1),
            Value::List(vec![Value::Int(2), Value::Int(3)]),
            Value::map([("a", Value::Null)]),
        ])
    );
    assert_eq!(
        parse(r#"{"b":1,"a":2}"#),
        Value::map([("b", Value::Int(1)), ("a", Value::Int(2))]),
        "order is kept"
    );
}

#[test]
fn malformed_containers_are_refused() {
    for text in [
        "[1,]",
        "[,1]",
        "[1 2]",
        "[1",
        "{",
        r#"{"a":1,}"#,
        r#"{"a" 1}"#,
        r#"{a:1}"#,
        r#"{"a":}"#,
        r#"{1:2}"#,
        "[}",
        "{]",
    ] {
        assert!(Value::from_json(text).is_err(), "{text:?} should not parse");
    }
}

#[test]
fn duplicate_keys_are_refused_with_the_offset_of_the_second() {
    let text = r#"{"a":1,"b":2,"a":3}"#;
    let e = parse_err(text);
    assert_eq!(e.kind, JsonErrorKind::DuplicateKey("a".to_string()));
    assert_eq!(e.offset, text.rfind("\"a\"").unwrap());
    assert!(Value::from_json(r#"{"a":{"x":1},"b":{"x":2}}"#).is_ok());
}

#[test]
fn empty_input_trailing_text_and_unexpected_characters() {
    assert_eq!(parse_err("").kind, JsonErrorKind::EmptyInput);
    assert_eq!(parse_err("  \n ").kind, JsonErrorKind::EmptyInput);
    assert_eq!(parse_err("1 x").kind, JsonErrorKind::TrailingCharacters);
    assert_eq!(parse_err("[] []").kind, JsonErrorKind::TrailingCharacters);
    assert_eq!(parse_err("nul").kind, JsonErrorKind::UnexpectedEnd);
    assert_eq!(parse_err("nulx").kind, JsonErrorKind::UnexpectedChar('x'));
    assert_eq!(parse_err("@").kind, JsonErrorKind::UnexpectedChar('@'));
    assert_eq!(parse_err("[1,@]").offset, 3);
}

#[test]
fn nesting_depth_is_limited() {
    let nested = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
    assert!(Value::from_json(&nested(128)).is_ok());
    assert_eq!(parse_err(&nested(129)).kind, JsonErrorKind::TooDeep);
    let objects = |n: usize| format!("{}1{}", "{\"a\":".repeat(n), "}".repeat(n));
    assert!(Value::from_json(&objects(128)).is_ok());
    assert_eq!(parse_err(&objects(129)).kind, JsonErrorKind::TooDeep);
    assert!(Value::from_json_with_depth("[[1]]", 2).is_ok());
    assert_eq!(
        Value::from_json_with_depth("[[[1]]]", 2).unwrap_err().kind,
        JsonErrorKind::TooDeep
    );
    // A hostile megabyte of brackets is refused, not recursed into.
    let huge = "[".repeat(1_000_000);
    assert_eq!(parse_err(&huge).kind, JsonErrorKind::TooDeep);
}

#[test]
fn compact_output() {
    let v = Value::map([
        ("a", Value::Int(1)),
        ("b", Value::List(vec![Value::Bool(true), Value::Null])),
        ("c", Value::from("x\"y")),
        ("d", Value::Map(vec![])),
        ("e", Value::List(vec![])),
    ]);
    assert_eq!(
        v.to_json(),
        r#"{"a":1,"b":[true,null],"c":"x\"y","d":{},"e":[]}"#
    );
}

#[test]
fn pretty_output_is_indented_and_reads_back() {
    let v = Value::map([
        ("a", Value::Int(1)),
        ("b", Value::List(vec![Value::Int(2), Value::Int(3)])),
        ("c", Value::Map(vec![])),
    ]);
    let expected = "{\n  \"a\": 1,\n  \"b\": [\n    2,\n    3\n  ],\n  \"c\": {}\n}";
    assert_eq!(v.to_json_pretty(), expected);
    assert_eq!(parse(expected), v);
}

#[test]
fn string_output_escapes() {
    let v = Value::from("q\"b\\n\n\r\t\u{8}\u{c}\u{1}\u{1f}é😀");
    assert_eq!(
        v.to_json(),
        "\"q\\\"b\\\\n\\n\\r\\t\\b\\f\\u0001\\u001fé😀\""
    );
    assert_eq!(parse(&v.to_json()), v);
}

#[test]
fn float_output_keeps_floats_floats_and_non_finite_become_strings() {
    assert_eq!(Value::Float(1.0).to_json(), "1.0");
    assert_eq!(Value::Float(0.1).to_json(), "0.1");
    assert_eq!(Value::Float(-0.0).to_json(), "-0.0");
    assert_eq!(Value::Float(1e21).to_json(), "1e21");
    assert_eq!(Value::Float(1.5e-7).to_json(), "1.5e-7");
    assert_eq!(Value::Float(f64::NAN).to_json(), "\"NaN\"");
    assert_eq!(Value::Float(f64::INFINITY).to_json(), "\"Infinity\"");
    assert_eq!(Value::Float(f64::NEG_INFINITY).to_json(), "\"-Infinity\"");
    for f in [
        0.1,
        1.0 / 3.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,
        -123456.789e10,
    ] {
        assert_eq!(parse(&Value::Float(f).to_json()), Value::Float(f), "{f}");
    }
    match parse(&Value::Float(-0.0).to_json()) {
        Value::Float(f) => assert!(f.is_sign_negative()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn integers_round_trip_through_text() {
    for v in [
        Value::Int(0),
        Value::Int(-1),
        Value::Int(i64::MIN),
        Value::Int(i64::MAX),
        Value::UInt(1 << 63),
        Value::UInt(u64::MAX),
    ] {
        assert_eq!(parse(&v.to_json()), v);
    }
    assert_eq!(Value::from_u64(5), Value::Int(5));
    assert_eq!(Value::from_u64(u64::MAX), Value::UInt(u64::MAX));
    assert_eq!(Value::from_u64(1 << 63), Value::UInt(1 << 63));
    assert_eq!(Value::from_u64(i64::MAX as u64), Value::Int(i64::MAX));
}

#[test]
fn accessors() {
    assert_eq!(Value::Bool(true).as_bool(), Some(true));
    assert_eq!(Value::Int(-3).as_i64(), Some(-3));
    assert_eq!(Value::UInt(u64::MAX).as_i64(), None);
    assert_eq!(Value::Int(3).as_u64(), Some(3));
    assert_eq!(Value::Int(-3).as_u64(), None);
    assert_eq!(Value::UInt(u64::MAX).as_u64(), Some(u64::MAX));
    assert_eq!(Value::Int(2).as_f64(), Some(2.0));
    assert_eq!(Value::Float(2.5).as_f64(), Some(2.5));
    assert_eq!(Value::from("x").as_str(), Some("x"));
    assert_eq!(Value::List(vec![]).as_list(), Some(&[][..]));
    assert!(Value::Null.as_map().is_none());
    let m = Value::map([("a", Value::Int(1))]);
    assert_eq!(m.get("a"), Some(&Value::Int(1)));
    assert_eq!(m.get("b"), None);
    assert_eq!(Value::Int(1).get("a"), None);
}

fn sample() -> Value {
    parse(r#"{"t":{"x":1.0,"y":2.0},"items":[{"name":"a"},{"name":"b"}],"n":5}"#)
}

#[test]
fn get_path_follows_fields_and_indexes() {
    let v = sample();
    assert_eq!(v.get_path("n").unwrap(), &Value::Int(5));
    assert_eq!(v.get_path("t.y").unwrap(), &Value::Float(2.0));
    assert_eq!(v.get_path("items[1].name").unwrap(), &Value::from("b"));
    assert_eq!(v.get_path("items[0]").unwrap(), &parse(r#"{"name":"a"}"#));
    assert_eq!(v.get_path("").unwrap(), &v);
    assert_eq!(
        parse("[[1,2],[3]]").get_path("[0][1]").unwrap(),
        &Value::Int(2)
    );
}

#[test]
fn get_path_errors_say_where() {
    let v = sample();
    let e = v.get_path("t.z").unwrap_err();
    assert_eq!(e.kind, ErrorKind::BadPath("no field `z`".to_string()));
    assert_eq!(e.path, vec![PathSeg::Field("t".to_string())]);
    assert_eq!(e.path_string(), "t");

    let e = v.get_path("items[5]").unwrap_err();
    assert!(e
        .to_string()
        .contains("index 5 is out of range for 2 elements"));
    assert_eq!(e.path_string(), "items");

    let e = v.get_path("n.x").unwrap_err();
    assert!(e.to_string().contains("cannot read field `x` of a integer"));
    let e = v.get_path("t[0]").unwrap_err();
    assert!(e.to_string().contains("cannot read index 0 of a map"));
    let e = v.get_path("items.x").unwrap_err();
    assert!(e.to_string().contains("cannot read field `x` of a list"));
}

#[test]
fn set_path_replaces_what_is_there() {
    let mut v = sample();
    v.set_path("t.x", Value::Float(9.0)).unwrap();
    v.set_path("items[1].name", Value::from("z")).unwrap();
    v.set_path("n", Value::Null).unwrap();
    assert_eq!(
        v,
        parse(r#"{"t":{"x":9.0,"y":2.0},"items":[{"name":"a"},{"name":"z"}],"n":null}"#)
    );
    v.set_path("", Value::Int(1)).unwrap();
    assert_eq!(v, Value::Int(1));
}

#[test]
fn set_path_never_makes_a_new_field_or_element() {
    let mut v = sample();
    let before = v.clone();
    assert!(v.set_path("t.w", Value::Int(1)).is_err());
    assert!(v.set_path("items[2]", Value::Int(1)).is_err());
    assert!(v.set_path("n.x", Value::Int(1)).is_err());
    assert_eq!(v, before, "a failed set changes nothing");
}

#[test]
fn bad_path_text_is_refused() {
    let v = sample();
    for path in [
        ".a",
        "a.",
        "a..b",
        "a[",
        "a[]",
        "a[x]",
        "a[-1]",
        "a]b",
        "a[0]b",
        "[0",
        "a[ 1 ]",
        "a[99999999999999999999999]",
    ] {
        let e = v.get_path(path).unwrap_err();
        assert!(
            matches!(e.kind, ErrorKind::BadPath(_)),
            "{path:?} gave {:?}",
            e.kind
        );
    }
}

#[test]
fn path_text_round_trips() {
    let segs = [
        PathSeg::Field("a".to_string()),
        PathSeg::Index(2),
        PathSeg::Field("b".to_string()),
        PathSeg::Index(0),
    ];
    assert_eq!(mid_reflect::path_string(&segs), "a[2].b[0]");
    assert_eq!(mid_reflect::path_string(&[]), "");
}

#[test]
fn json_errors_display_with_their_offset() {
    let e = parse_err("[1,@]");
    assert_eq!(
        e.to_string(),
        "JSON error at byte 3: unexpected character '@'"
    );
    let e = parse_err("");
    assert_eq!(e.to_string(), "JSON error at byte 0: empty input");
}
