/* crates/mid-math/benches/handmademath_bench.c
 * HandmadeMath v2 benchmark — popular single-header indie game math library.
 * Widely used in the Handmade Hero / Casey Muratori community.
 *
 * NOTE: HMM v2 does not expose a general 4x4 matrix inverse.
 * We implement it manually via Cramer's rule (same algorithm used internally
 * by most math libraries). This benchmarks the HMM Mat4 data layout overhead.
 *
 * Build: see bench-vs-handmademath.yml
 */

/* HMM v2 uses HANDMADE_MATH_IMPLEMENTATION to include the implementation.
 * SIMD is auto-detected by HMM v2; the old HANDMADE_MATH_USE_SSE define
 * from v1 is harmless if present but not required. */
#define HANDMADE_MATH_IMPLEMENTATION
#include "HandmadeMath.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ── timing ────────────────────────────────────────────────────────────────── */
static inline long long ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

#define ITERS 1000000
#define BARRIER(x) __asm__ volatile("" : "+m"(x))

volatile float sink = 0.0f;

static void report(const char *label, long long ns, int iters) {
    printf("  %-52s %7.2f ns/op\n", label, (double)ns / iters);
    fflush(stdout);
}

/* ── Manual 4×4 general inverse via Cramer's rule ────────────────────────────
 * HMM v2 does not provide HMM_InvM4 / HMM_InvGeneralM4.
 * This matches the algorithm in mid-math's scalar fallback exactly.
 * Layout: HMM_Mat4.Elements[col][row] — column-major, same as mid-math.
 */
static HMM_Mat4 hmm_mat4_inv(HMM_Mat4 m) {
    float a[16];
    int c, r;
    for (c = 0; c < 4; c++)
        for (r = 0; r < 4; r++)
            a[c * 4 + r] = m.Elements[c][r];

    float inv[16];
    inv[ 0] =  a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10];
    inv[ 4] = -a[4]*a[10]*a[15]+a[4]*a[11]*a[14]+a[8]*a[6]*a[15]-a[8]*a[7]*a[14]-a[12]*a[6]*a[11]+a[12]*a[7]*a[10];
    inv[ 8] =  a[4]*a[9]*a[15]-a[4]*a[11]*a[13]-a[8]*a[5]*a[15]+a[8]*a[7]*a[13]+a[12]*a[5]*a[11]-a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]+a[4]*a[10]*a[13]+a[8]*a[5]*a[14]-a[8]*a[6]*a[13]-a[12]*a[5]*a[10]+a[12]*a[6]*a[9];
    inv[ 1] = -a[1]*a[10]*a[15]+a[1]*a[11]*a[14]+a[9]*a[2]*a[15]-a[9]*a[3]*a[14]-a[13]*a[2]*a[11]+a[13]*a[3]*a[10];
    inv[ 5] =  a[0]*a[10]*a[15]-a[0]*a[11]*a[14]-a[8]*a[2]*a[15]+a[8]*a[3]*a[14]+a[12]*a[2]*a[11]-a[12]*a[3]*a[10];
    inv[ 9] = -a[0]*a[9]*a[15]+a[0]*a[11]*a[13]+a[8]*a[1]*a[15]-a[8]*a[3]*a[13]-a[12]*a[1]*a[11]+a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14]-a[0]*a[10]*a[13]-a[8]*a[1]*a[14]+a[8]*a[2]*a[13]+a[12]*a[1]*a[10]-a[12]*a[2]*a[9];
    inv[ 2] =  a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6];
    inv[ 6] = -a[0]*a[6]*a[15]+a[0]*a[7]*a[14]+a[4]*a[2]*a[15]-a[4]*a[3]*a[14]-a[12]*a[2]*a[7]+a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15]-a[0]*a[7]*a[13]-a[4]*a[1]*a[15]+a[4]*a[3]*a[13]+a[12]*a[1]*a[7]-a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]+a[0]*a[6]*a[13]+a[4]*a[1]*a[14]-a[4]*a[2]*a[13]-a[12]*a[1]*a[6]+a[12]*a[2]*a[5];
    inv[ 3] = -a[1]*a[6]*a[11]+a[1]*a[7]*a[10]+a[5]*a[2]*a[11]-a[5]*a[3]*a[10]-a[9]*a[2]*a[7]+a[9]*a[3]*a[6];
    inv[ 7] =  a[0]*a[6]*a[11]-a[0]*a[7]*a[10]-a[4]*a[2]*a[11]+a[4]*a[3]*a[10]+a[8]*a[2]*a[7]-a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]+a[0]*a[7]*a[9]+a[4]*a[1]*a[11]-a[4]*a[3]*a[9]-a[8]*a[1]*a[7]+a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10]-a[0]*a[6]*a[9]-a[4]*a[1]*a[10]+a[4]*a[2]*a[9]+a[8]*a[1]*a[6]-a[8]*a[2]*a[5];

    float det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    if (fabsf(det) < 1e-6f) {
        HMM_Mat4 z;
        int i;
        for (i = 0; i < 16; i++) ((float*)&z)[i] = 0.0f;
        return z;
    }
    float id = 1.0f / det;
    HMM_Mat4 result;
    for (c = 0; c < 4; c++)
        for (r = 0; r < 4; r++)
            result.Elements[c][r] = inv[c * 4 + r] * id;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Vec3                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_vec3(void) {
    printf("vec3 (4-way interleaved):\n");

    /* — add — */
    {
        HMM_Vec3 a0=HMM_V3(1,2,3), a1=HMM_V3(1.1f,2.1f,3.1f),
                 a2=HMM_V3(1.2f,2.2f,3.2f), a3=HMM_V3(1.3f,2.3f,3.3f);
        HMM_Vec3 b=HMM_V3(4,5,6);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            a0 = HMM_AddV3(a0, b);
            a1 = HMM_AddV3(a1, b);
            a2 = HMM_AddV3(a2, b);
            a3 = HMM_AddV3(a3, b);
        }
        BARRIER(a0); sink += a0.X;
        report("add/handmademath", ns_now()-t, ITERS);
    }

    /* — dot — */
    {
        HMM_Vec3 a=HMM_V3(1,2,3), b=HMM_V3(4,5,6);
        float d=0;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            d = HMM_DotV3(a, b);
            a.X += d * 1e-30f;
            BARRIER(a);
        }
        sink += d;
        report("dot/handmademath", ns_now()-t, ITERS);
    }

    /* — cross — */
    {
        HMM_Vec3 a0=HMM_V3(1,2,3), a1=HMM_V3(1.1f,2.1f,3.1f),
                 a2=HMM_V3(1.2f,2.2f,3.2f), a3=HMM_V3(1.3f,2.3f,3.3f);
        HMM_Vec3 b=HMM_V3(4,5,6);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            a0 = HMM_Cross(a0, b);
            a1 = HMM_Cross(a1, b);
            a2 = HMM_Cross(a2, b);
            a3 = HMM_Cross(a3, b);
        }
        BARRIER(a0); sink += a0.X;
        report("cross/handmademath", ns_now()-t, ITERS);
    }

    /* — normalize — */
    {
        HMM_Vec3 a0=HMM_V3(1,2,3), a1=HMM_V3(1.1f,2.1f,3.1f),
                 a2=HMM_V3(1.2f,2.2f,3.2f), a3=HMM_V3(1.3f,2.3f,3.3f);
        HMM_Vec3 eps=HMM_V3(1e-30f,0,0);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            a0 = HMM_NormV3(a0); a0 = HMM_AddV3(a0,eps); BARRIER(a0);
            a1 = HMM_NormV3(a1); a1 = HMM_AddV3(a1,eps); BARRIER(a1);
            a2 = HMM_NormV3(a2); a2 = HMM_AddV3(a2,eps); BARRIER(a2);
            a3 = HMM_NormV3(a3); a3 = HMM_AddV3(a3,eps); BARRIER(a3);
        }
        sink += a0.X;
        report("normalize/handmademath", ns_now()-t, ITERS);
    }

    /* — lerp — */
    {
        HMM_Vec3 a=HMM_V3(0,0,0), e=HMM_V3(1,1,1);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = HMM_LerpV3(a, 0.5f, e);
        }
        BARRIER(a); sink += a.X;
        report("lerp/handmademath", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Quaternion                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_quat(void) {
    printf("\nquat:\n");

    HMM_Vec3 axis_y=HMM_V3(0,1,0);
    HMM_Vec3 axis_d=HMM_NormV3(HMM_V3(1,1,0));
    HMM_Quat q1=HMM_QFromAxisAngle_RH(axis_y, HMM_AngleDeg(45.0f));
    HMM_Quat q2=HMM_QFromAxisAngle_RH(axis_d, HMM_AngleDeg(30.0f));

    /* — mul — */
    {
        HMM_Quat a=q1, b=q2;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = HMM_MulQ(a, b);
        }
        BARRIER(a); sink += a.X;
        report("mul/handmademath", ns_now()-t, ITERS);
    }

    /* — rotate — */
    {
        HMM_Vec3 v=HMM_V3(1,0,0);
        HMM_Quat q=q1;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            v = HMM_RotateV3Q(v, q);
            BARRIER(v);
        }
        sink += v.X;
        report("rotate/handmademath", ns_now()-t, ITERS);
    }

    /* — slerp — */
    {
        HMM_Quat a=q1, b=q2;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = HMM_SLerp(a, 0.5f, b);
        }
        BARRIER(a); sink += a.X;
        report("slerp/handmademath", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Mat4                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_mat4(void) {
    printf("\nmat4:\n");

    HMM_Vec3 axis_y=HMM_V3(0,1,0);
    HMM_Quat q1=HMM_QFromAxisAngle_RH(axis_y, HMM_AngleDeg(45.0f));
    HMM_Quat q2=HMM_QFromAxisAngle_RH(axis_y, HMM_AngleDeg(30.0f));

    /* TRS: Scale * Rotate * Translate (HMM is column-major, right-multiply) */
    HMM_Mat4 ma = HMM_MulM4(
        HMM_MulM4(HMM_Scale(HMM_V3(2,2,2)), HMM_QToM4(q1)),
        HMM_Translate(HMM_V3(1,0,0)));
    HMM_Mat4 mb = HMM_MulM4(
        HMM_MulM4(HMM_Scale(HMM_V3(1.5f,1.5f,1.5f)), HMM_QToM4(q2)),
        HMM_Translate(HMM_V3(0.5f,0,0)));

    /* — mul — */
    {
        HMM_Mat4 a=ma, b=mb;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = HMM_MulM4(a, b);
        }
        BARRIER(a); sink += a.Elements[0][0];
        report("mul/handmademath", ns_now()-t, ITERS);
    }

    /* — transform point — */
    {
        HMM_Vec4 p=HMM_V4(1,2,3,1);
        HMM_Mat4 m=ma;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            p = HMM_MulM4V4(m, p);
            BARRIER(p);
        }
        sink += p.X;
        report("transform_point/handmademath", ns_now()-t, ITERS);
    }

    /* — inverse general (manual Cramer — HMM v2 has no builtin) — */
    {
        HMM_Mat4 m=ma;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            m = hmm_mat4_inv(m);
        }
        BARRIER(m); sink += m.Elements[0][0];
        report("inverse_general/handmademath", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Bulk                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_bulk(void) {
    printf("\n100k entity transforms (single run):\n");
    const int N = 100000;

    HMM_Vec3 axis_y=HMM_V3(0,1,0);
    HMM_Quat q=HMM_QFromAxisAngle_RH(axis_y, HMM_AngleDeg(45.0f));
    HMM_Mat4 trs=HMM_MulM4(HMM_QToM4(q), HMM_Translate(HMM_V3(1,0,0)));

    HMM_Vec4 *pos = (HMM_Vec4*)aligned_alloc(16, (size_t)N * sizeof(HMM_Vec4));
    if (!pos) { fprintf(stderr, "alloc failed\n"); return; }
    for (int i = 0; i < N; i++) pos[i] = HMM_V4(i*0.01f, 0, 0, 1);

    long long t0 = ns_now();
    for (int i = 0; i < N; i++) pos[i] = HMM_MulM4V4(trs, pos[i]);
    long long dt = ns_now() - t0;

    printf("  %-52s %7.1f µs  (%.2f ns/entity)\n",
        "transform_point/handmademath", (double)dt/1000.0, (double)dt/N);
    fflush(stdout);
    sink += pos[0].X;
    free(pos);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("HandmadeMath v2 benchmark — compiled %s\n", __DATE__);
    printf("SIMD: auto-detected by HMM v2\n\n");

    bench_vec3();
    bench_quat();
    bench_mat4();
    bench_bulk();

    printf("\ndone. (sink=%.6f)\n", (float)sink);
    fflush(stdout);
    return 0;
}
