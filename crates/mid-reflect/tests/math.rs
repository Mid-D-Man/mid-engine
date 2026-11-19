//! `mid-math` types (feature `math`).
#![cfg(feature = "math")]

use mid_math::{Affine3, DAffine3, DQuat, DVec2, DVec3, DVec4, Quat, Vec2, Vec3, Vec4};
use mid_reflect::{ErrorKind, Reflect, TypeRegistry, Value};

fn round<T: Reflect + PartialEq + std::fmt::Debug>(x: T) {
    let text = x.to_value().to_json();
    let back = T::from_value(&Value::from_json(&text).unwrap()).unwrap();
    assert_eq!(back, x, "{text}");
}

#[test]
fn vectors_and_quaternions_are_maps_of_components() {
    assert_eq!(
        Vec2::new(1.0, 2.0).to_value().to_json(),
        r#"{"x":1.0,"y":2.0}"#
    );
    assert_eq!(
        Vec3::new(1.0, 2.0, 3.0).to_value().to_json(),
        r#"{"x":1.0,"y":2.0,"z":3.0}"#
    );
    assert_eq!(
        Vec4::new(1.0, 2.0, 3.0, 4.0).to_value().to_json(),
        r#"{"x":1.0,"y":2.0,"z":3.0,"w":4.0}"#
    );
    assert_eq!(
        Quat::from_xyzw(0.0, 0.0, 0.0, 1.0).to_value().to_json(),
        r#"{"x":0.0,"y":0.0,"z":0.0,"w":1.0}"#
    );
    assert_eq!(
        Vec3::new(0.1, -2.5, 1e-3).to_value().to_json(),
        r#"{"x":0.1,"y":-2.5,"z":0.001}"#
    );
}

#[test]
fn they_round_trip() {
    round(Vec2::new(1.5, -2.0));
    round(Vec3::new(0.1, 0.2, 0.3));
    round(Vec4::new(1.0, 2.0, 3.0, 4.0));
    round(Quat::from_xyzw(0.1, 0.2, 0.3, 0.9));
    round(DVec2::new(0.1, 0.2));
    round(DVec3::new(0.1, 1e300, -5e-324));
    round(DVec4::new(1.0, 2.0, 3.0, 4.0));
    round(DQuat::from_xyzw(0.1, 0.2, 0.3, 0.9));
    round(Affine3::from_translation(Vec3::new(1.0, 2.0, 3.0)));
    round(DAffine3::from_translation(DVec3::new(
        100_000_000.5,
        0.0,
        0.0,
    )));
}

#[test]
fn an_f64_vector_keeps_every_digit_and_an_f32_one_its_short_form() {
    let wide = DVec3::new(0.1, 0.2, 0.3).to_value().to_json();
    assert_eq!(wide, r#"{"x":0.1,"y":0.2,"z":0.3}"#);
    let narrow = Vec3::new(0.1, 0.2, 0.3).to_value().to_json();
    assert_eq!(
        narrow, wide,
        "an f32 0.1 is written as 0.1, not 0.10000000149011612"
    );
}

#[test]
fn components_are_strict() {
    let v = Value::from_json(r#"{"x":1.0,"y":2.0,"z":3.0,"w":4.0}"#).unwrap();
    assert!(matches!(
        Vec3::from_value(&v).unwrap_err().kind,
        ErrorKind::UnknownField(_)
    ));
    let v = Value::from_json(r#"{"x":1.0,"y":2.0}"#).unwrap();
    assert_eq!(
        Vec3::from_value(&v).unwrap_err().kind,
        ErrorKind::MissingField("z".to_string())
    );
    let v = Value::from_json(r#"{"x":1.0,"y":"a","z":3.0}"#).unwrap();
    assert_eq!(Vec3::from_value(&v).unwrap_err().path_string(), "y");
    assert!(Vec3::from_value(&Value::Int(1)).is_err());
}

#[test]
fn registering_an_affine_registers_its_vector_and_a_path_edits_through_both() {
    let mut r = TypeRegistry::new();
    r.register::<Affine3>().unwrap();
    let names: Vec<&str> = r.iter().map(|x| x.name()).collect();
    assert_eq!(names, ["mid_math::Affine3", "mid_math::Vec3"]);
    let mut a = Affine3::IDENTITY;
    r.mutate(
        "mid_math::Affine3",
        &mut a,
        "translation.y",
        Value::Float(7.5),
    )
    .unwrap();
    assert_eq!(a.translation.y, 7.5);
    assert_eq!(a.translation.x, 0.0);
    assert_eq!(a.x_axis.x, 1.0);
    assert!(r
        .mutate(
            "mid_math::Affine3",
            &mut a,
            "translation.w",
            Value::Float(1.0)
        )
        .is_err());
}

#[test]
fn the_schema_describes_the_math_types() {
    let mut r = TypeRegistry::new();
    r.register::<DAffine3>().unwrap();
    let schema = r.schema();
    assert_eq!(
        schema.get("mid_math::DVec3").unwrap().to_json(),
        r#"{"kind":"struct","fields":[{"name":"x","type":"f64"},{"name":"y","type":"f64"},{"name":"z","type":"f64"}]}"#
    );
    assert_eq!(
        schema
            .get("mid_math::DAffine3")
            .unwrap()
            .get("fields")
            .unwrap()
            .as_list()
            .unwrap()
            .len(),
        4
    );
}
