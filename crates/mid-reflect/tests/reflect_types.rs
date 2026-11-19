//! `Reflect` for the primitives, containers, and the three macros.

use mid_reflect::{
    impl_reflect_enum, impl_reflect_newtype, impl_reflect_struct, ErrorKind, Field, Kind, PathSeg,
    Reflect, Ty, TypeInfo, Value,
};

fn round<T: Reflect + PartialEq + std::fmt::Debug>(x: T) {
    let v = x.to_value();
    assert_eq!(T::from_value(&v).unwrap(), x, "direct: {v:?}");
    let text = v.to_json();
    let back = Value::from_json(&text).unwrap();
    assert_eq!(T::from_value(&back).unwrap(), x, "through {text}");
}

fn err<T: Reflect + std::fmt::Debug>(v: &Value) -> mid_reflect::ReflectError {
    T::from_value(v).unwrap_err()
}

macro_rules! int_edges {
    ($($t:ty),*) => {$(
        round(<$t>::MIN);
        round(<$t>::MAX);
        round::<$t>(0);
        round::<$t>(1);
    )*};
}

#[test]
fn every_integer_type_round_trips_at_its_edges() {
    int_edges!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
}

#[test]
fn integers_outside_the_type_are_refused() {
    assert!(matches!(
        err::<u8>(&Value::Int(256)).kind,
        ErrorKind::OutOfRange { expected: "u8", .. }
    ));
    assert!(matches!(
        err::<u8>(&Value::Int(-1)).kind,
        ErrorKind::OutOfRange { .. }
    ));
    assert!(matches!(
        err::<i8>(&Value::Int(128)).kind,
        ErrorKind::OutOfRange { .. }
    ));
    assert!(matches!(
        err::<i8>(&Value::Int(-129)).kind,
        ErrorKind::OutOfRange { .. }
    ));
    assert!(matches!(
        err::<i64>(&Value::UInt(u64::MAX)).kind,
        ErrorKind::OutOfRange { .. }
    ));
    assert!(matches!(
        err::<u32>(&Value::UInt(u64::MAX)).kind,
        ErrorKind::OutOfRange { .. }
    ));
    assert_eq!(u64::from_value(&Value::UInt(u64::MAX)).unwrap(), u64::MAX);
    assert_eq!(
        err::<u8>(&Value::Int(300)).to_string(),
        "300 does not fit u8"
    );
}

#[test]
fn integers_accept_whole_floats_and_nothing_else() {
    assert_eq!(u8::from_value(&Value::Float(7.0)).unwrap(), 7);
    assert_eq!(i32::from_value(&Value::Float(-3.0)).unwrap(), -3);
    assert!(matches!(
        err::<i32>(&Value::Float(1.5)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert!(matches!(
        err::<i32>(&Value::Float(f64::NAN)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert!(matches!(
        err::<i32>(&Value::Float(f64::INFINITY)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert!(matches!(
        err::<u8>(&Value::Float(300.0)).kind,
        ErrorKind::OutOfRange { .. }
    ));
    assert!(matches!(
        err::<u8>(&Value::Float(1e300)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    for v in [Value::from("1"), Value::Bool(true), Value::Null] {
        assert!(matches!(err::<u8>(&v).kind, ErrorKind::TypeMismatch { .. }));
    }
}

#[test]
fn unsigned_values_use_the_canonical_form() {
    assert_eq!(5u64.to_value(), Value::Int(5));
    assert_eq!(u64::MAX.to_value(), Value::UInt(u64::MAX));
    assert_eq!(u8::MAX.to_value(), Value::Int(255));
    assert_eq!((-5i32).to_value(), Value::Int(-5));
}

#[test]
fn floats_round_trip_including_the_special_ones() {
    round(1.5f64);
    round(-0.0f64);
    round(f64::MAX);
    round(f64::MIN_POSITIVE);
    round(f64::INFINITY);
    round(f64::NEG_INFINITY);
    round(0.1f32);
    round(f32::MAX);
    round(f32::MIN_POSITIVE);
    round(f32::INFINITY);
    let nan = f64::from_value(&Value::from_json("\"NaN\"").unwrap()).unwrap();
    assert!(nan.is_nan());
    let nan32 = f32::from_value(&Value::Float(f64::NAN)).unwrap();
    assert!(nan32.is_nan());
    assert_eq!(f64::NAN.to_value().to_json(), "\"NaN\"");
}

#[test]
fn an_f32_is_stored_as_its_short_decimal() {
    assert_eq!(0.1f32.to_value(), Value::Float(0.1));
    assert_eq!(0.3f32.to_value().to_json(), "0.3");
    assert_eq!(1.0f32.to_value().to_json(), "1.0");
    assert_eq!(16777216.0f32.to_value().to_json(), "16777216.0");
    assert_eq!((0.1f32 as f64).to_value().to_json(), "0.10000000149011612");
}

#[test]
fn floats_read_integers_and_special_strings_and_refuse_the_rest() {
    assert_eq!(f32::from_value(&Value::Int(3)).unwrap(), 3.0);
    assert_eq!(
        f64::from_value(&Value::UInt(u64::MAX)).unwrap(),
        1.8446744073709552e19
    );
    assert_eq!(
        f64::from_value(&Value::from("-Infinity")).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(matches!(
        err::<f64>(&Value::from("abc")).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert!(matches!(
        err::<f64>(&Value::Bool(true)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert!(matches!(
        err::<f32>(&Value::Float(1e39)).kind,
        ErrorKind::OutOfRange {
            expected: "f32",
            ..
        }
    ));
    assert_eq!(
        f32::from_value(&Value::Float(1e-50)).unwrap(),
        0.0,
        "underflow is not an error"
    );
}

#[test]
fn bool_and_string() {
    round(true);
    round(false);
    round(String::new());
    round("héllo \"q\" \n 😀".to_string());
    assert!(matches!(
        err::<bool>(&Value::Int(1)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert!(matches!(
        err::<String>(&Value::Int(1)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
    assert_eq!(
        err::<bool>(&Value::from("x")).to_string(),
        "expected bool, found string"
    );
}

#[test]
fn option_is_null_or_the_inner_value() {
    round(None::<u32>);
    round(Some(5u32));
    round(Some("x".to_string()));
    assert_eq!(None::<u32>.to_value(), Value::Null);
    assert_eq!(Some(5u32).to_value(), Value::Int(5));
    let e = err::<Option<u8>>(&Value::Int(999));
    assert!(matches!(e.kind, ErrorKind::OutOfRange { .. }));
}

#[test]
fn vec_and_array() {
    round(vec![1u8, 2, 3]);
    round(Vec::<u8>::new());
    round([1.5f32, 2.5, 3.5]);
    round([[1u8, 2], [3, 4]]);
    round(vec![Some(1u8), None]);
    let e = err::<Vec<u8>>(&Value::from_json("[1,2,300]").unwrap());
    assert_eq!(e.path, vec![PathSeg::Index(2)]);
    assert_eq!(e.to_string(), "at [2]: 300 does not fit u8");
    let e = err::<[u8; 3]>(&Value::from_json("[1,2]").unwrap());
    assert_eq!(
        e.kind,
        ErrorKind::BadLength {
            expected: 3,
            found: 2
        }
    );
    let e = err::<[u8; 2]>(&Value::from_json("[1,2,3]").unwrap());
    assert_eq!(
        e.kind,
        ErrorKind::BadLength {
            expected: 2,
            found: 3
        }
    );
    let e = err::<Vec<u8>>(&Value::Int(1));
    assert_eq!(e.to_string(), "expected Vec<u8>, found integer");
    let e = err::<[[u8; 2]; 2]>(&Value::from_json("[[1,2],[3,999]]").unwrap());
    assert_eq!(e.path_string(), "[1][1]");
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Stats {
    hp: u32,
    speed: f32,
}
impl_reflect_struct!("game::Stats" => Stats { hp: u32, speed: f32 });

#[derive(Debug, PartialEq, Clone)]
struct Name(String);
impl_reflect_newtype!("game::Name" => Name(String));

#[derive(Debug, PartialEq, Clone, Copy)]
enum Mode {
    Idle,
    Run,
    Jump,
}
impl_reflect_enum!("game::Mode" => Mode { Idle, Run, Jump });

#[derive(Debug, PartialEq, Clone)]
struct Actor {
    name: Name,
    stats: Stats,
    mode: Mode,
    tags: Vec<Name>,
    target: Option<Name>,
    grid: [u8; 2],
}
impl_reflect_struct!("game::Actor" => Actor {
    name: Name,
    stats: Stats,
    mode: Mode,
    tags: Vec<Name>,
    target: Option<Name>,
    grid: [u8; 2],
});

#[derive(Debug, PartialEq)]
struct Marker {}
impl_reflect_struct!("game::Marker" => Marker {});

fn actor() -> Actor {
    Actor {
        name: Name("orc".into()),
        stats: Stats { hp: 30, speed: 1.5 },
        mode: Mode::Run,
        tags: vec![Name("a".into()), Name("b".into())],
        target: None,
        grid: [3, 4],
    }
}

#[test]
fn a_struct_is_a_map_in_field_order() {
    assert_eq!(
        Stats { hp: 3, speed: 0.5 }.to_value().to_json(),
        r#"{"hp":3,"speed":0.5}"#
    );
    assert_eq!(
        actor().to_value().to_json(),
        r#"{"name":"orc","stats":{"hp":30,"speed":1.5},"mode":"Run","tags":["a","b"],"target":null,"grid":[3,4]}"#
    );
    round(actor());
    round(Marker {});
    assert_eq!(Marker {}.to_value().to_json(), "{}");
}

#[test]
fn the_order_of_fields_in_the_input_does_not_matter() {
    let v = Value::from_json(r#"{"speed":0.5,"hp":3}"#).unwrap();
    assert_eq!(Stats::from_value(&v).unwrap(), Stats { hp: 3, speed: 0.5 });
}

#[test]
fn a_struct_needs_exactly_its_fields() {
    let v = Value::from_json(r#"{"hp":3}"#).unwrap();
    assert_eq!(
        err::<Stats>(&v).kind,
        ErrorKind::MissingField("speed".to_string())
    );
    let v = Value::from_json(r#"{"hp":3,"speed":1.0,"extra":1}"#).unwrap();
    assert_eq!(
        err::<Stats>(&v).kind,
        ErrorKind::UnknownField("extra".to_string())
    );
    let v = Value::from_json(r#"{"extra":1}"#).unwrap();
    assert!(matches!(err::<Stats>(&v).kind, ErrorKind::UnknownField(_)));
    assert_eq!(
        err::<Stats>(&Value::Int(1)).to_string(),
        "expected game::Stats, found integer"
    );
    assert!(matches!(
        err::<Marker>(&Value::from_json(r#"{"a":1}"#).unwrap()).kind,
        ErrorKind::UnknownField(_)
    ));
}

#[test]
fn errors_name_the_path_into_nested_values() {
    let mut v = actor().to_value();
    v.set_path("stats.hp", Value::Int(-1)).unwrap();
    let e = err::<Actor>(&v);
    assert_eq!(e.path_string(), "stats.hp");
    assert_eq!(e.to_string(), "at stats.hp: -1 does not fit u32");

    let mut v = actor().to_value();
    v.set_path("tags[1]", Value::Int(5)).unwrap();
    let e = err::<Actor>(&v);
    assert_eq!(e.path_string(), "tags[1]");
    assert_eq!(e.to_string(), "at tags[1]: expected String, found integer");

    let mut v = actor().to_value();
    v.set_path("mode", Value::from("Fly")).unwrap();
    let e = err::<Actor>(&v);
    assert_eq!(
        e.to_string(),
        "at mode: `Fly` is not a variant of game::Mode"
    );

    let mut v = actor().to_value();
    v.set_path("grid", Value::from_json("[1,2,3]").unwrap())
        .unwrap();
    assert_eq!(
        err::<Actor>(&v).to_string(),
        "at grid: expected 2 elements, found 3"
    );
}

#[test]
fn a_newtype_is_transparent() {
    assert_eq!(Name("x".into()).to_value(), Value::from("x"));
    round(Name("x".into()));
    assert!(matches!(
        err::<Name>(&Value::Int(1)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn an_enum_is_its_variant_name() {
    for m in [Mode::Idle, Mode::Run, Mode::Jump] {
        round(m);
    }
    assert_eq!(Mode::Jump.to_value(), Value::from("Jump"));
    assert_eq!(
        err::<Mode>(&Value::from("jump")).kind,
        ErrorKind::UnknownVariant {
            ty: "game::Mode".to_string(),
            found: "jump".to_string()
        },
        "variant names are case sensitive"
    );
    assert!(matches!(
        err::<Mode>(&Value::Int(0)).kind,
        ErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn types_describe_themselves() {
    assert_eq!(Vec::<Option<f32>>::ty().to_string(), "Vec<Option<f32>>");
    assert_eq!(<[u8; 4]>::ty().to_string(), "[u8; 4]");
    assert_eq!(i64::ty().to_string(), "i64");
    assert_eq!(usize::ty().to_string(), "u64");
    assert_eq!(String::ty().to_string(), "String");
    assert_eq!(bool::ty(), Ty::Bool);
    assert_eq!(Stats::ty(), Ty::Named("game::Stats".to_string()));
    assert!(f32::info().is_none() && Vec::<Stats>::info().is_none());
    assert_eq!(
        Stats::info().unwrap(),
        TypeInfo {
            name: "game::Stats".to_string(),
            kind: Kind::Struct(vec![
                Field {
                    name: "hp".to_string(),
                    ty: Ty::Int {
                        bits: 32,
                        signed: false
                    }
                },
                Field {
                    name: "speed".to_string(),
                    ty: Ty::Float { bits: 32 }
                },
            ]),
        }
    );
    assert_eq!(Name::info().unwrap().kind, Kind::Newtype(Ty::Str));
    assert_eq!(
        Mode::info().unwrap().kind,
        Kind::Enum(vec![
            "Idle".to_string(),
            "Run".to_string(),
            "Jump".to_string()
        ])
    );
}

#[test]
fn type_info_exports_as_a_value() {
    assert_eq!(
        Stats::info().unwrap().to_value().to_json(),
        r#"{"kind":"struct","fields":[{"name":"hp","type":"u32"},{"name":"speed","type":"f32"}]}"#
    );
    assert_eq!(
        Name::info().unwrap().to_value().to_json(),
        r#"{"kind":"newtype","type":"String"}"#
    );
    assert_eq!(
        Mode::info().unwrap().to_value().to_json(),
        r#"{"kind":"enum","variants":["Idle","Run","Jump"]}"#
    );
}
