// crates/mid-math/src/tests/mint_tests.rs
//
// Tests for the `mint` feature's conversions (features/mint_conversions/).
// Only compiled with `--features mint`; see docs/mid-math.md, section
// "features/mint_conversions/f32.rs".

use crate::{DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4, IVec3, Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

// Matrices are compared through plain arrays so the tests do not depend on
// which backend (scalar, SSE2, NEON, wasm) provides the type.
fn mat4_cols(m: Mat4) -> [[f32; 4]; 4] {
    [m.x_axis.to_array(), m.y_axis.to_array(), m.z_axis.to_array(), m.w_axis.to_array()]
}

// ── vectors ──────────────────────────────────────────────────────────────────

#[test]
fn vec2_vec3_vec4_map_components_in_order() {
    let v2: mint::Vector2<f32> = Vec2::new(1.0, 2.0).into();
    assert_eq!(v2, mint::Vector2 { x: 1.0, y: 2.0 });
    let v3: mint::Vector3<f32> = Vec3::new(1.0, 2.0, 3.0).into();
    assert_eq!(v3, mint::Vector3 { x: 1.0, y: 2.0, z: 3.0 });
    let v4: mint::Vector4<f32> = Vec4::new(1.0, 2.0, 3.0, 4.0).into();
    assert_eq!(v4, mint::Vector4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 });
}

#[test]
fn vectors_round_trip() {
    let a = Vec3::from(mint::Vector3 { x: 5.0_f32, y: 6.0, z: 7.0 });
    assert_eq!(a.to_array(), [5.0, 6.0, 7.0]);
    let b = Vec4::from(mint::Vector4 { x: 1.0_f32, y: 2.0, z: 3.0, w: 4.0 });
    assert_eq!(mint::Vector4::from(b), mint::Vector4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 });
}

#[test]
fn f64_and_integer_vectors_convert_too() {
    let d: mint::Vector3<f64> = DVec3::new(1.5, 2.5, 3.5).into();
    assert_eq!(d, mint::Vector3 { x: 1.5, y: 2.5, z: 3.5 });
    let back = DVec2::from(mint::Vector2 { x: 9.0_f64, y: 8.0 });
    assert_eq!((back.x, back.y), (9.0, 8.0));
    let d4 = DVec4::from(mint::Vector4 { x: 1.0_f64, y: 2.0, z: 3.0, w: 4.0 });
    assert_eq!((d4.x, d4.w), (1.0, 4.0));
    let i: mint::Vector3<i32> = IVec3::new(-1, 2, -3).into();
    assert_eq!(i, mint::Vector3 { x: -1, y: 2, z: -3 });
}

// ── quaternions ──────────────────────────────────────────────────────────────

#[test]
fn quat_maps_the_scalar_part_to_w() {
    let q = Quat::from(mint::Quaternion { s: 4.0_f32, v: mint::Vector3 { x: 1.0, y: 2.0, z: 3.0 } });
    assert_eq!((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0));
    let m: mint::Quaternion<f32> = Quat::new(1.0, 2.0, 3.0, 4.0).into();
    assert_eq!(m, mint::Quaternion { s: 4.0, v: mint::Vector3 { x: 1.0, y: 2.0, z: 3.0 } });
}

#[test]
fn dquat_maps_the_scalar_part_to_w() {
    let q = DQuat::from(mint::Quaternion { s: 4.0_f64, v: mint::Vector3 { x: 1.0, y: 2.0, z: 3.0 } });
    assert_eq!((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0));
}

// ── Mat2 (the conversions that did not compile before) ───────────────────────

#[test]
fn mat2_column_matrix_keeps_its_columns() {
    let m = Mat2::from(mint::ColumnMatrix2 {
        x: mint::Vector2 { x: 1.0_f32, y: 2.0 },
        y: mint::Vector2 { x: 3.0, y: 4.0 },
    });
    assert_eq!(m.to_cols_array(), [1.0, 2.0, 3.0, 4.0]);
    let back: mint::ColumnMatrix2<f32> = m.into();
    assert_eq!(back.x, mint::Vector2 { x: 1.0, y: 2.0 });
    assert_eq!(back.y, mint::Vector2 { x: 3.0, y: 4.0 });
}

#[test]
fn mat2_row_matrix_is_transposed_into_columns() {
    // Rows [1 2] and [3 4] make the matrix 1 2 / 3 4, whose columns are
    // (1, 3) and (2, 4).
    let m = Mat2::from(mint::RowMatrix2 {
        x: mint::Vector2 { x: 1.0_f32, y: 2.0 },
        y: mint::Vector2 { x: 3.0, y: 4.0 },
    });
    assert_eq!(m.to_cols_array(), [1.0, 3.0, 2.0, 4.0]);
    let back: mint::RowMatrix2<f32> = m.into();
    assert_eq!(back.x, mint::Vector2 { x: 1.0, y: 2.0 });
    assert_eq!(back.y, mint::Vector2 { x: 3.0, y: 4.0 });
}

#[test]
fn mat2_column_and_row_forms_of_a_transpose_agree() {
    let col = Mat2::from(mint::ColumnMatrix2 {
        x: mint::Vector2 { x: 1.0_f32, y: 3.0 },
        y: mint::Vector2 { x: 2.0, y: 4.0 },
    });
    let row = Mat2::from(mint::RowMatrix2 {
        x: mint::Vector2 { x: 1.0_f32, y: 2.0 },
        y: mint::Vector2 { x: 3.0, y: 4.0 },
    });
    assert_eq!(col.to_cols_array(), row.to_cols_array());
}

// ── Mat3 / Mat4 ──────────────────────────────────────────────────────────────

fn v3(x: f32, y: f32, z: f32) -> mint::Vector3<f32> {
    mint::Vector3 { x, y, z }
}

#[test]
fn mat3_column_matrix_keeps_its_columns() {
    let m = Mat3::from(mint::ColumnMatrix3 { x: v3(1.0, 2.0, 3.0), y: v3(4.0, 5.0, 6.0), z: v3(7.0, 8.0, 9.0) });
    assert_eq!(m.to_cols_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let back: mint::ColumnMatrix3<f32> = m.into();
    assert_eq!((back.x, back.y, back.z), (v3(1.0, 2.0, 3.0), v3(4.0, 5.0, 6.0), v3(7.0, 8.0, 9.0)));
}

#[test]
fn mat3_row_matrix_is_transposed_into_columns() {
    let m = Mat3::from(mint::RowMatrix3 { x: v3(1.0, 2.0, 3.0), y: v3(4.0, 5.0, 6.0), z: v3(7.0, 8.0, 9.0) });
    assert_eq!(m.to_cols_array(), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    let back: mint::RowMatrix3<f32> = m.into();
    assert_eq!((back.x, back.y, back.z), (v3(1.0, 2.0, 3.0), v3(4.0, 5.0, 6.0), v3(7.0, 8.0, 9.0)));
}

fn v4(x: f32, y: f32, z: f32, w: f32) -> mint::Vector4<f32> {
    mint::Vector4 { x, y, z, w }
}

#[test]
fn mat4_column_matrix_keeps_its_columns() {
    let m = Mat4::from(mint::ColumnMatrix4 {
        x: v4(1.0, 2.0, 3.0, 4.0),
        y: v4(5.0, 6.0, 7.0, 8.0),
        z: v4(9.0, 10.0, 11.0, 12.0),
        w: v4(13.0, 14.0, 15.0, 16.0),
    });
    assert_eq!(
        mat4_cols(m),
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    );
    let back: mint::ColumnMatrix4<f32> = m.into();
    assert_eq!(back.w, v4(13.0, 14.0, 15.0, 16.0));
}

#[test]
fn mat4_row_matrix_is_transposed_into_columns() {
    let m = Mat4::from(mint::RowMatrix4 {
        x: v4(1.0, 2.0, 3.0, 4.0),
        y: v4(5.0, 6.0, 7.0, 8.0),
        z: v4(9.0, 10.0, 11.0, 12.0),
        w: v4(13.0, 14.0, 15.0, 16.0),
    });
    assert_eq!(
        mat4_cols(m),
        [[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]
    );
    let back: mint::RowMatrix4<f32> = m.into();
    assert_eq!(back.x, v4(1.0, 2.0, 3.0, 4.0));
    assert_eq!(back.w, v4(13.0, 14.0, 15.0, 16.0));
}

// ── f64 matrices ─────────────────────────────────────────────────────────────

#[test]
fn dmat2_column_and_row_forms() {
    let col = DMat2::from(mint::ColumnMatrix2 {
        x: mint::Vector2 { x: 1.0_f64, y: 2.0 },
        y: mint::Vector2 { x: 3.0, y: 4.0 },
    });
    assert_eq!((col.x_axis.x, col.x_axis.y, col.y_axis.x, col.y_axis.y), (1.0, 2.0, 3.0, 4.0));
    let row = DMat2::from(mint::RowMatrix2 {
        x: mint::Vector2 { x: 1.0_f64, y: 2.0 },
        y: mint::Vector2 { x: 3.0, y: 4.0 },
    });
    assert_eq!((row.x_axis.x, row.x_axis.y, row.y_axis.x, row.y_axis.y), (1.0, 3.0, 2.0, 4.0));
}

#[test]
fn dmat3_and_dmat4_column_forms_keep_their_columns() {
    let m3 = DMat3::from(mint::ColumnMatrix3 {
        x: mint::Vector3 { x: 1.0_f64, y: 2.0, z: 3.0 },
        y: mint::Vector3 { x: 4.0, y: 5.0, z: 6.0 },
        z: mint::Vector3 { x: 7.0, y: 8.0, z: 9.0 },
    });
    assert_eq!(m3.cols, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m4 = DMat4::from(mint::ColumnMatrix4 {
        x: mint::Vector4 { x: 1.0_f64, y: 2.0, z: 3.0, w: 4.0 },
        y: mint::Vector4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
        z: mint::Vector4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
        w: mint::Vector4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 },
    });
    assert_eq!(m4.cols[3], [13.0, 14.0, 15.0, 16.0]);
}

// ── IntoMint ─────────────────────────────────────────────────────────────────

fn through_mint<T>(t: T) -> T::MintType
where
    T: mint::IntoMint + Into<T::MintType>,
{
    t.into()
}

#[test]
fn into_mint_names_the_matching_mint_type() {
    let v: mint::Vector3<f32> = through_mint(Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(v, v3(1.0, 2.0, 3.0));
    let m: mint::ColumnMatrix2<f32> = through_mint(Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0)));
    assert_eq!(m.y, mint::Vector2 { x: 3.0, y: 4.0 });
}
