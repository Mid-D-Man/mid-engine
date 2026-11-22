//! `TypeRegistry`: registration, dynamic access, type data, the schema.

use std::any::Any;

use mid_reflect::{
    impl_reflect_enum, impl_reflect_newtype, impl_reflect_struct, ErrorKind, ReflectError,
    RegistryError, TypeRegistry, Value,
};

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
}
impl_reflect_enum!("game::Mode" => Mode { Idle, Run });

#[derive(Debug, PartialEq, Clone)]
struct Actor {
    name: Name,
    stats: Stats,
    modes: Vec<Mode>,
    last: Option<Stats>,
}
impl_reflect_struct!("game::Actor" => Actor {
    name: Name,
    stats: Stats,
    modes: Vec<Mode>,
    last: Option<Stats>,
});

#[derive(Debug, PartialEq)]
struct Tree {
    label: u8,
    children: Vec<Tree>,
}
impl_reflect_struct!("game::Tree" => Tree { label: u8, children: Vec<Tree> });

#[derive(Debug, PartialEq)]
struct Impostor {
    x: u8,
}
impl_reflect_struct!("game::Stats" => Impostor { x: u8 });

fn actor() -> Actor {
    Actor {
        name: Name("orc".into()),
        stats: Stats { hp: 30, speed: 1.5 },
        modes: vec![Mode::Idle, Mode::Run],
        last: None,
    }
}

fn names(r: &TypeRegistry) -> Vec<String> {
    r.iter().map(|x| x.name().to_string()).collect()
}

#[test]
fn registering_a_type_registers_what_it_contains_in_field_order() {
    let mut r = TypeRegistry::new();
    assert!(r.is_empty());
    r.register::<Actor>().unwrap();
    assert_eq!(
        names(&r),
        ["game::Actor", "game::Name", "game::Stats", "game::Mode"]
    );
    assert_eq!(r.len(), 4);
    assert!(!r.is_empty());
}

#[test]
fn registering_twice_changes_nothing() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    r.register::<Actor>().unwrap();
    r.register::<Stats>().unwrap();
    assert_eq!(r.len(), 4);
}

#[test]
fn containers_register_their_elements_and_primitives_register_nothing() {
    let mut r = TypeRegistry::new();
    r.register::<Vec<Option<[Mode; 2]>>>().unwrap();
    assert_eq!(names(&r), ["game::Mode"]);
    r.register::<u32>().unwrap();
    r.register::<String>().unwrap();
    assert_eq!(r.len(), 1);
}

#[test]
fn a_type_that_contains_itself_registers_and_works() {
    let mut r = TypeRegistry::new();
    r.register::<Tree>().unwrap();
    assert_eq!(names(&r), ["game::Tree"]);
    let t = Tree {
        label: 1,
        children: vec![
            Tree {
                label: 2,
                children: vec![],
            },
            Tree {
                label: 3,
                children: vec![Tree {
                    label: 4,
                    children: vec![],
                }],
            },
        ],
    };
    let v = r.to_value("game::Tree", &t).unwrap();
    assert_eq!(
        v.to_json(),
        r#"{"label":1,"children":[{"label":2,"children":[]},{"label":3,"children":[{"label":4,"children":[]}]}]}"#
    );
    let built = r.construct("game::Tree", &v).unwrap();
    assert_eq!(*built.downcast::<Tree>().unwrap(), t);
}

#[test]
fn two_types_under_one_name_is_an_error_and_the_first_stays() {
    let mut r = TypeRegistry::new();
    r.register::<Stats>().unwrap();
    let e = r.register::<Impostor>().unwrap_err();
    assert_eq!(e, RegistryError::NameConflict("game::Stats".to_string()));
    assert_eq!(
        e.to_string(),
        "two different types are registered as `game::Stats`"
    );
    assert_eq!(r.len(), 1);
    assert!(r.get_of::<Stats>().is_some());
    assert!(r.get_of::<Impostor>().is_none());
    // The same conflict found through a containing type is reported too.
    #[derive(Debug)]
    struct Holder {
        i: Impostor,
    }
    impl_reflect_struct!("game::Holder" => Holder { i: Impostor });
    assert!(r.register::<Holder>().is_err());
}

#[test]
fn lookups() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    let reg = r.get("game::Stats").unwrap();
    assert_eq!(reg.name(), "game::Stats");
    assert_eq!(reg.type_id(), std::any::TypeId::of::<Stats>());
    assert!(reg.rust_name().ends_with("Stats"));
    assert_eq!(reg.info().name, "game::Stats");
    assert!(r.get("game::Nope").is_none());
    assert!(r.get_of::<Stats>().is_some());
    assert!(r.get_of::<Tree>().is_none());
    assert_eq!(
        r.get_by_type_id(std::any::TypeId::of::<Mode>())
            .unwrap()
            .name(),
        "game::Mode"
    );
    assert!(r.get_by_type_id(std::any::TypeId::of::<u8>()).is_none());
}

#[test]
fn reading_writing_and_building_by_name() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    let mut a = actor();
    let v = r.to_value("game::Actor", &a).unwrap();
    assert_eq!(v, mid_reflect::Reflect::to_value(&a));

    let mut changed = v.clone();
    changed.set_path("stats.hp", Value::Int(99)).unwrap();
    r.apply("game::Actor", &mut a, &changed).unwrap();
    assert_eq!(a.stats.hp, 99);

    let built = r
        .construct(
            "game::Stats",
            &Value::from_json(r#"{"hp":1,"speed":2.0}"#).unwrap(),
        )
        .unwrap();
    assert_eq!(
        *built.downcast::<Stats>().unwrap(),
        Stats { hp: 1, speed: 2.0 }
    );
}

#[test]
fn mutate_sets_one_part_by_path() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    let mut a = actor();
    r.mutate("game::Actor", &mut a, "stats.speed", Value::Float(0.25))
        .unwrap();
    r.mutate("game::Actor", &mut a, "name", Value::from("elf"))
        .unwrap();
    r.mutate("game::Actor", &mut a, "modes[0]", Value::from("Run"))
        .unwrap();
    r.mutate(
        "game::Actor",
        &mut a,
        "last",
        Value::from_json(r#"{"hp":5,"speed":1.0}"#).unwrap(),
    )
    .unwrap();
    assert_eq!(a.stats.speed, 0.25);
    assert_eq!(a.name, Name("elf".into()));
    assert_eq!(a.modes, vec![Mode::Run, Mode::Run]);
    assert_eq!(a.last, Some(Stats { hp: 5, speed: 1.0 }));
}

#[test]
fn a_failed_write_changes_nothing() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    let mut a = actor();
    let before = a.clone();
    // A path that does not exist.
    assert!(r
        .mutate("game::Actor", &mut a, "stats.mana", Value::Int(1))
        .is_err());
    // A value that does not fit: u32 cannot be negative.
    let e = r
        .mutate("game::Actor", &mut a, "stats.hp", Value::Int(-1))
        .unwrap_err();
    assert_eq!(e.path_string(), "stats.hp");
    // A whole value that is half right.
    let half = Value::from_json(
        r#"{"name":"x","stats":{"hp":1,"speed":1.0},"modes":["Run","Fly"],"last":null}"#,
    )
    .unwrap();
    assert!(r.apply("game::Actor", &mut a, &half).is_err());
    assert_eq!(a, before);
}

#[test]
fn the_wrong_rust_type_or_an_unknown_name_is_an_error_not_a_panic() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    let e = r.to_value("game::Actor", &5u32).unwrap_err();
    assert_eq!(e.kind, ErrorKind::WrongType("game::Actor".to_string()));
    let mut s = Stats { hp: 1, speed: 1.0 };
    let e = r.apply("game::Actor", &mut s, &Value::Null).unwrap_err();
    assert_eq!(e.kind, ErrorKind::WrongType("game::Actor".to_string()));
    let e = r.to_value("game::Nope", &s).unwrap_err();
    assert_eq!(e.kind, ErrorKind::UnknownType("game::Nope".to_string()));
    assert_eq!(e.to_string(), "no type registered as `game::Nope`");
    assert!(r.construct("game::Nope", &Value::Null).is_err());
    assert!(r.mutate("game::Nope", &mut s, "", Value::Null).is_err());
}

#[test]
fn registered_values_can_be_any_through_a_box() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    let boxed: Box<dyn Any + Send + Sync> = Box::new(actor());
    let v = r.to_value("game::Actor", boxed.as_ref()).unwrap();
    assert_eq!(v, mid_reflect::Reflect::to_value(&actor()));
}

struct Tag(&'static str);
struct Other(u32);

#[test]
fn other_crates_can_attach_data_to_a_registration() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    assert!(r.insert_data::<Actor, _>(Tag("component")));
    assert!(r.insert_data::<Actor, _>(Other(7)));
    assert_eq!(r.data_of::<Actor, Tag>().unwrap().0, "component");
    assert_eq!(r.data_of::<Actor, Other>().unwrap().0, 7);
    assert!(r.data_of::<Stats, Tag>().is_none(), "per type");
    assert_eq!(r.data_by_name::<Tag>("game::Actor").unwrap().0, "component");
    assert!(r.data_by_name::<Tag>("game::Nope").is_none());
    assert!(r.insert_data::<Actor, _>(Tag("replaced")));
    assert_eq!(r.data_of::<Actor, Tag>().unwrap().0, "replaced");
    assert_eq!(
        r.data_of::<Actor, Other>().unwrap().0,
        7,
        "the other is untouched"
    );
    assert!(
        !r.insert_data::<Tree, _>(Tag("x")),
        "an unregistered type takes none"
    );
}

#[test]
fn the_schema_lists_every_type_in_registration_order() {
    let mut r = TypeRegistry::new();
    r.register::<Actor>().unwrap();
    assert_eq!(
        r.schema().to_json(),
        concat!(
            r#"{"game::Actor":{"kind":"struct","fields":["#,
            r#"{"name":"name","type":"game::Name"},"#,
            r#"{"name":"stats","type":"game::Stats"},"#,
            r#"{"name":"modes","type":"Vec<game::Mode>"},"#,
            r#"{"name":"last","type":"Option<game::Stats>"}]},"#,
            r#""game::Name":{"kind":"newtype","type":"String"},"#,
            r#""game::Stats":{"kind":"struct","fields":[{"name":"hp","type":"u32"},{"name":"speed","type":"f32"}]},"#,
            r#""game::Mode":{"kind":"enum","variants":["Idle","Run"]}}"#,
        )
    );
    assert_eq!(TypeRegistry::new().schema(), Value::Map(vec![]));
}

#[test]
fn unsupported_error_displays_its_message() {
    let error = ReflectError::new(ErrorKind::Unsupported("read only".to_string()));
    assert_eq!(error.to_string(), "not supported: read only");
}
