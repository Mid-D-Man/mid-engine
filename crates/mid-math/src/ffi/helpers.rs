// crates/mid-math/src/ffi/helpers.rs
//! C-ABI types and exports for helpers: DualQuat, Rotor3, angle,
//! TangentFrame, SpatialVelocity/Force/Inertia.

use crate::{
    Degrees, DualQuat, PackedTangent, Quat, Radians, Rotor3,
    SpatialForce, SpatialInertia, SpatialVelocity, TangentFrame, Vec2, Vec3,
};
use crate::ffi::float32::{CQuat, CVec2, CVec3};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

/// Dual quaternion — 8 floats (real xyzw + dual xyzw). 32 bytes, align 16.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CDualQuat {
    pub real_x: f32, pub real_y: f32, pub real_z: f32, pub real_w: f32,
    pub dual_x: f32, pub dual_y: f32, pub dual_z: f32, pub dual_w: f32,
}

/// Rotor3 — 4 floats (s, b_yz, b_xz, b_xy). 16 bytes.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CRotor3 {
    pub s: f32, pub b_yz: f32, pub b_xz: f32, pub b_xy: f32,
}

/// TangentFrame — 3 × Vec3 (normal, tangent, bitangent). 36 bytes.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CTangentFrame {
    pub nx: f32, pub ny: f32, pub nz: f32,
    pub tx: f32, pub ty: f32, pub tz: f32,
    pub bx: f32, pub by: f32, pub bz: f32,
}

/// PackedTangent — tangent xyz + handedness. 16 bytes.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CPackedTangent {
    pub tx: f32, pub ty: f32, pub tz: f32,
    pub handedness: f32,
}

/// SpatialVelocity — angular + linear Vec3. 24 bytes.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CSpatialVelocity {
    pub ax: f32, pub ay: f32, pub az: f32,
    pub vx: f32, pub vy: f32, pub vz: f32,
}

/// SpatialForce — torque + force Vec3. 24 bytes.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CSpatialForce {
    pub tx: f32, pub ty: f32, pub tz: f32,
    pub fx: f32, pub fy: f32, pub fz: f32,
}

/// SpatialInertia — mass, com (3 floats), inertia tensor (6 floats). 40 bytes.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CSpatialInertia {
    pub mass: f32,
    pub com_x: f32, pub com_y: f32, pub com_z: f32,
    pub inertia: [f32; 6],
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline(always)]
fn dq_to_c(dq: DualQuat) -> CDualQuat {
    CDualQuat {
        real_x: dq.real.x, real_y: dq.real.y, real_z: dq.real.z, real_w: dq.real.w,
        dual_x: dq.dual.x, dual_y: dq.dual.y, dual_z: dq.dual.z, dual_w: dq.dual.w,
    }
}

#[inline(always)]
fn dq_from_c(c: CDualQuat) -> DualQuat {
    DualQuat {
        real: Quat::new(c.real_x, c.real_y, c.real_z, c.real_w),
        dual: Quat::new(c.dual_x, c.dual_y, c.dual_z, c.dual_w),
    }
}

#[inline(always)]
fn tf_to_c(t: TangentFrame) -> CTangentFrame {
    CTangentFrame {
        nx: t.normal.x,    ny: t.normal.y,    nz: t.normal.z,
        tx: t.tangent.x,   ty: t.tangent.y,   tz: t.tangent.z,
        bx: t.bitangent.x, by: t.bitangent.y, bz: t.bitangent.z,
    }
}

#[inline(always)]
fn tf_from_c(c: CTangentFrame) -> TangentFrame {
    TangentFrame {
        normal:    Vec3::new(c.nx, c.ny, c.nz),
        tangent:   Vec3::new(c.tx, c.ty, c.tz),
        bitangent: Vec3::new(c.bx, c.by, c.bz),
    }
}

#[inline(always)]
fn sv_to_c(v: SpatialVelocity) -> CSpatialVelocity {
    CSpatialVelocity {
        ax: v.angular.x, ay: v.angular.y, az: v.angular.z,
        vx: v.linear.x,  vy: v.linear.y,  vz: v.linear.z,
    }
}
#[inline(always)]
fn sv_from_c(c: CSpatialVelocity) -> SpatialVelocity {
    SpatialVelocity::new(Vec3::new(c.ax,c.ay,c.az), Vec3::new(c.vx,c.vy,c.vz))
}

#[inline(always)]
fn sf_to_c(f: SpatialForce) -> CSpatialForce {
    CSpatialForce {
        tx: f.torque.x, ty: f.torque.y, tz: f.torque.z,
        fx: f.force.x,  fy: f.force.y,  fz: f.force.z,
    }
}
#[inline(always)]
fn sf_from_c(c: CSpatialForce) -> SpatialForce {
    SpatialForce::new(Vec3::new(c.tx,c.ty,c.tz), Vec3::new(c.fx,c.fy,c.fz))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Angle exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_to_radians(deg: f32) -> f32 {
    Degrees::new(deg).to_radians().value()
}
#[no_mangle] pub extern "C" fn mid_to_degrees(rad: f32) -> f32 {
    Radians::new(rad).to_degrees().value()
}
#[no_mangle] pub extern "C" fn mid_radians_sin(rad: f32) -> f32  { Radians::new(rad).sin() }
#[no_mangle] pub extern "C" fn mid_radians_cos(rad: f32) -> f32  { Radians::new(rad).cos() }
#[no_mangle] pub extern "C" fn mid_radians_tan(rad: f32) -> f32  { Radians::new(rad).tan() }
#[no_mangle] pub extern "C" fn mid_radians_wrap(rad: f32) -> f32 { Radians::new(rad).wrap().value() }
#[no_mangle] pub extern "C" fn mid_radians_wrap_positive(rad: f32) -> f32 {
    Radians::new(rad).wrap_positive().value()
}
#[no_mangle] pub extern "C" fn mid_radians_lerp(a: f32, b: f32, t: f32) -> f32 {
    Radians::new(a).lerp(Radians::new(b), t).value()
}
#[no_mangle] pub extern "C" fn mid_degrees_wrap(deg: f32) -> f32 {
    Degrees::new(deg).wrap().value()
}
#[no_mangle] pub extern "C" fn mid_degrees_lerp(a: f32, b: f32, t: f32) -> f32 {
    Degrees::new(a).lerp(Degrees::new(b), t).value()
}

// ═══════════════════════════════════════════════════════════════════════════
//  DualQuat exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_dual_quat_identity() -> CDualQuat {
    dq_to_c(DualQuat::IDENTITY)
}
#[no_mangle] pub extern "C" fn mid_dual_quat_from_rotation_translation(
    q: CQuat, tx: f32, ty: f32, tz: f32,
) -> CDualQuat {
    dq_to_c(DualQuat::from_rotation_translation(
        Quat::new(q.x, q.y, q.z, q.w),
        Vec3::new(tx, ty, tz),
    ))
}
#[no_mangle] pub extern "C" fn mid_dual_quat_from_rotation(q: CQuat) -> CDualQuat {
    dq_to_c(DualQuat::from_rotation(Quat::new(q.x, q.y, q.z, q.w)))
}
#[no_mangle] pub extern "C" fn mid_dual_quat_from_translation(tx:f32,ty:f32,tz:f32) -> CDualQuat {
    dq_to_c(DualQuat::from_translation(Vec3::new(tx,ty,tz)))
}
#[no_mangle] pub extern "C" fn mid_dual_quat_rotation(dq: CDualQuat) -> CQuat {
    let q = dq_from_c(dq).rotation();
    CQuat { x: q.x, y: q.y, z: q.z, w: q.w }
}
#[no_mangle] pub extern "C" fn mid_dual_quat_translation(dq: CDualQuat) -> CVec3 {
    let t = dq_from_c(dq).translation();
    CVec3::new(t.x, t.y, t.z)
}
#[no_mangle] pub extern "C" fn mid_dual_quat_transform_point(dq: CDualQuat, p: CVec3) -> CVec3 {
    let t = dq_from_c(dq).transform_point(Vec3::new(p.x, p.y, p.z));
    CVec3::new(t.x, t.y, t.z)
}
#[no_mangle] pub extern "C" fn mid_dual_quat_transform_vector(dq: CDualQuat, v: CVec3) -> CVec3 {
    let t = dq_from_c(dq).transform_vector(Vec3::new(v.x, v.y, v.z));
    CVec3::new(t.x, t.y, t.z)
}
#[no_mangle] pub extern "C" fn mid_dual_quat_normalize(dq: CDualQuat) -> CDualQuat {
    dq_to_c(dq_from_c(dq).normalize())
}
#[no_mangle] pub extern "C" fn mid_dual_quat_mul(a: CDualQuat, b: CDualQuat) -> CDualQuat {
    dq_to_c(dq_from_c(a) * dq_from_c(b))
}
#[no_mangle] pub extern "C" fn mid_dual_quat_conjugate(dq: CDualQuat) -> CDualQuat {
    dq_to_c(dq_from_c(dq).conjugate())
}
#[no_mangle] pub extern "C" fn mid_dual_quat_blend2(
    dq0: CDualQuat, w0: f32,
    dq1: CDualQuat, w1: f32,
) -> CDualQuat {
    dq_to_c(DualQuat::blend2(dq_from_c(dq0), w0, dq_from_c(dq1), w1))
}
#[no_mangle] pub extern "C" fn mid_dual_quat_is_finite(dq: CDualQuat) -> bool {
    dq_from_c(dq).is_finite()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Rotor3 exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_rotor3_identity() -> CRotor3 {
    let r = Rotor3::IDENTITY;
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}
#[no_mangle] pub extern "C" fn mid_rotor3_from_axis_angle(
    ax: f32, ay: f32, az: f32, angle: f32,
) -> CRotor3 {
    let r = Rotor3::from_axis_angle(Vec3::new(ax,ay,az), angle);
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}
#[no_mangle] pub extern "C" fn mid_rotor3_from_vec_to_vec(
    fx: f32, fy: f32, fz: f32,
    tx: f32, ty: f32, tz: f32,
) -> CRotor3 {
    let r = Rotor3::from_vec_to_vec(Vec3::new(fx,fy,fz), Vec3::new(tx,ty,tz));
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}
#[no_mangle] pub extern "C" fn mid_rotor3_rotate(cr: CRotor3, vx:f32, vy:f32, vz:f32) -> CVec3 {
    let r = Rotor3::new(cr.s, cr.b_yz, cr.b_xz, cr.b_xy);
    let out = r.rotate(Vec3::new(vx,vy,vz));
    CVec3::new(out.x, out.y, out.z)
}
#[no_mangle] pub extern "C" fn mid_rotor3_normalize(cr: CRotor3) -> CRotor3 {
    let r = Rotor3::new(cr.s, cr.b_yz, cr.b_xz, cr.b_xy).normalize();
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}
#[no_mangle] pub extern "C" fn mid_rotor3_nlerp(a: CRotor3, b: CRotor3, t: f32) -> CRotor3 {
    let ra = Rotor3::new(a.s, a.b_yz, a.b_xz, a.b_xy);
    let rb = Rotor3::new(b.s, b.b_yz, b.b_xz, b.b_xy);
    let r = ra.nlerp(rb, t);
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}
#[no_mangle] pub extern "C" fn mid_rotor3_mul(a: CRotor3, b: CRotor3) -> CRotor3 {
    let r = (Rotor3::new(a.s,a.b_yz,a.b_xz,a.b_xy)
           * Rotor3::new(b.s,b.b_yz,b.b_xz,b.b_xy));
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}
#[no_mangle] pub extern "C" fn mid_rotor3_to_quat(cr: CRotor3) -> CQuat {
    let q = Rotor3::new(cr.s, cr.b_yz, cr.b_xz, cr.b_xy).to_quat();
    CQuat { x: q.x, y: q.y, z: q.z, w: q.w }
}
#[no_mangle] pub extern "C" fn mid_rotor3_from_quat(q: CQuat) -> CRotor3 {
    let r = Rotor3::from_quat(Quat::new(q.x, q.y, q.z, q.w));
    CRotor3 { s: r.s, b_yz: r.b_yz, b_xz: r.b_xz, b_xy: r.b_xy }
}

// ═══════════════════════════════════════════════════════════════════════════
//  TangentFrame exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_tangent_frame_from_normal_tangent(
    nx:f32,ny:f32,nz:f32,
    tx:f32,ty:f32,tz:f32,
    handedness: f32,
) -> CTangentFrame {
    tf_to_c(TangentFrame::from_normal_tangent(Vec3::new(nx,ny,nz), Vec3::new(tx,ty,tz), handedness))
}
/// Returns 0 on degenerate triangle, 1 on success. Writes result to `out`.
#[no_mangle] pub unsafe extern "C" fn mid_tangent_frame_from_triangle(
    p0: CVec3, p1: CVec3, p2: CVec3,
    uv0x: f32, uv0y: f32, uv1x: f32, uv1y: f32, uv2x: f32, uv2y: f32,
    nx: f32, ny: f32, nz: f32,
    out: *mut CTangentFrame,
) -> i32 {
    match TangentFrame::from_triangle(
        Vec3::new(p0.x,p0.y,p0.z), Vec3::new(p1.x,p1.y,p1.z), Vec3::new(p2.x,p2.y,p2.z),
        Vec2::new(uv0x,uv0y), Vec2::new(uv1x,uv1y), Vec2::new(uv2x,uv2y),
        Vec3::new(nx,ny,nz),
    ) {
        Some(tf) => { *out = tf_to_c(tf); 1 }
        None => 0,
    }
}
#[no_mangle] pub extern "C" fn mid_tangent_frame_transform_normal(
    tf: CTangentFrame, nx: f32, ny: f32, nz: f32,
) -> CVec3 {
    let t = tf_from_c(tf).transform_normal(Vec3::new(nx,ny,nz));
    CVec3::new(t.x, t.y, t.z)
}
#[no_mangle] pub extern "C" fn mid_tangent_frame_to_tangent_space(
    tf: CTangentFrame, wx: f32, wy: f32, wz: f32,
) -> CVec3 {
    let t = tf_from_c(tf).to_tangent_space(Vec3::new(wx,wy,wz));
    CVec3::new(t.x, t.y, t.z)
}
#[no_mangle] pub extern "C" fn mid_tangent_frame_orthogonalise(tf: CTangentFrame) -> CTangentFrame {
    tf_to_c(tf_from_c(tf).orthogonalise())
}
#[no_mangle] pub extern "C" fn mid_tangent_frame_handedness(tf: CTangentFrame) -> f32 {
    tf_from_c(tf).handedness()
}
#[no_mangle] pub extern "C" fn mid_tangent_frame_pack(tf: CTangentFrame) -> CPackedTangent {
    let p = tf_from_c(tf).pack();
    CPackedTangent { tx: p.tangent.x, ty: p.tangent.y, tz: p.tangent.z, handedness: p.handedness }
}
#[no_mangle] pub extern "C" fn mid_tangent_frame_unpack(
    pt: CPackedTangent, nx: f32, ny: f32, nz: f32,
) -> CTangentFrame {
    let tf = TangentFrame::unpack(
        PackedTangent { tangent: Vec3::new(pt.tx,pt.ty,pt.tz), handedness: pt.handedness },
        Vec3::new(nx, ny, nz),
    );
    tf_to_c(tf)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Spatial exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_spatial_velocity_zero() -> CSpatialVelocity {
    sv_to_c(SpatialVelocity::ZERO)
}
#[no_mangle] pub extern "C" fn mid_spatial_velocity_new(
    ax:f32,ay:f32,az:f32, vx:f32,vy:f32,vz:f32,
) -> CSpatialVelocity {
    sv_to_c(SpatialVelocity::new(Vec3::new(ax,ay,az),Vec3::new(vx,vy,vz)))
}
#[no_mangle] pub extern "C" fn mid_spatial_velocity_add(
    a: CSpatialVelocity, b: CSpatialVelocity,
) -> CSpatialVelocity {
    sv_to_c(sv_from_c(a) + sv_from_c(b))
}
#[no_mangle] pub extern "C" fn mid_spatial_velocity_scale(v: CSpatialVelocity, s: f32) -> CSpatialVelocity {
    sv_to_c(sv_from_c(v).scale(s))
}
#[no_mangle] pub extern "C" fn mid_spatial_velocity_cross_vel(
    a: CSpatialVelocity, b: CSpatialVelocity,
) -> CSpatialVelocity {
    sv_to_c(sv_from_c(a).cross_vel(sv_from_c(b)))
}
#[no_mangle] pub extern "C" fn mid_spatial_velocity_cross_force(
    v: CSpatialVelocity, f: CSpatialForce,
) -> CSpatialForce {
    sf_to_c(sv_from_c(v).cross_force(sf_from_c(f)))
}
#[no_mangle] pub extern "C" fn mid_spatial_velocity_dot_force(
    v: CSpatialVelocity, f: CSpatialForce,
) -> f32 {
    sv_from_c(v).dot_force(sf_from_c(f))
}

#[no_mangle] pub extern "C" fn mid_spatial_force_zero() -> CSpatialForce {
    sf_to_c(SpatialForce::ZERO)
}
#[no_mangle] pub extern "C" fn mid_spatial_force_new(
    tx:f32,ty:f32,tz:f32, fx:f32,fy:f32,fz:f32,
) -> CSpatialForce {
    sf_to_c(SpatialForce::new(Vec3::new(tx,ty,tz),Vec3::new(fx,fy,fz)))
}
#[no_mangle] pub extern "C" fn mid_spatial_force_add(a: CSpatialForce, b: CSpatialForce) -> CSpatialForce {
    sf_to_c(sf_from_c(a) + sf_from_c(b))
}
#[no_mangle] pub extern "C" fn mid_spatial_force_scale(f: CSpatialForce, s: f32) -> CSpatialForce {
    sf_to_c(sf_from_c(f).scale(s))
}

#[no_mangle] pub extern "C" fn mid_spatial_inertia_mul_vel(
    inertia: CSpatialInertia, vel: CSpatialVelocity,
) -> CSpatialForce {
    let si = SpatialInertia {
        mass: inertia.mass,
        com:  Vec3::new(inertia.com_x, inertia.com_y, inertia.com_z),
        inertia: inertia.inertia,
    };
    sf_to_c(si.mul_vel(sv_from_c(vel)))
      }
