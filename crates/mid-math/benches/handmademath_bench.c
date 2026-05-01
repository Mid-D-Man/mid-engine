/* crates/mid-math/benches/handmademath_bench.c
 * HandmadeMath v2 benchmark — popular single-header indie game math library.
 * Widely used in the Handmade Hero / Casey Muratori community.
 * Scalar by default; SIMD enabled when HMM_USE_SSE is defined.
 *
 * Build: see bench-vs-handmademath.yml
 */

/* Enable SSE2 intrinsics in HandmadeMath */
#define HANDMADE_MATH_USE_SSE
#define HANDMADE_MATH_IMPLEMENTATION
#include "HandmadeMath.h"

#include <stdio.h>
#include <stdlib.h>
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

    /* TRS = Scale * Rotate * Translate (HMM is column-major) */
    HMM_Mat4 ma = HMM_MulM4(
        HMM_MulM4(HMM_Scale(HMM_V3(2,2,2)),
                  HMM_QToM4(q1)),
        HMM_Translate(HMM_V3(1,0,0)));
    HMM_Mat4 mb = HMM_MulM4(
        HMM_MulM4(HMM_Scale(HMM_V3(1.5f,1.5f,1.5f)),
                  HMM_QToM4(q2)),
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

    /* — inverse general — */
    {
        HMM_Mat4 m=ma;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            m = HMM_InvM4(m);
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

    HMM_Vec4 *pos = (HMM_Vec4*)aligned_alloc(16, N * sizeof(HMM_Vec4));
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
    printf("SSE2 intrinsics enabled (HANDMADE_MATH_USE_SSE)\n\n");

    bench_vec3();
    bench_quat();
    bench_mat4();
    bench_bulk();

    printf("\ndone. (sink=%.6f)\n", (float)sink);
    fflush(stdout);
    return 0;
}
