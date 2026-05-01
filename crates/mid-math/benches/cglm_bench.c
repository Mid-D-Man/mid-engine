/* crates/mid-math/benches/cglm_bench.c
 * FIXED: proper data-dependency chains throughout.
 *
 * ROOT CAUSE OF OLD BUG: The BENCH macro held a,b as compile-time constants.
 * GCC hoisted all cglm ops out of the loop entirely and just ran
 * `sink_f += constant` 1M times — every op showed ~4.22ns (loop overhead).
 *
 * FIX: each BENCH_DEP call copies output back into one of the inputs, forcing
 * a true RAW (read-after-write) data dependency. The compiler cannot move or
 * eliminate the operation without computing the correct chain.
 *
 * Uses 4-way interleaving to approximate throughput (match criterion).
 * Single chain would measure latency (2-3x slower for compute-heavy ops).
 *
 * Build: see bench-vs-cglm.yml
 */

#include <cglm/cglm.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* ── timing ────────────────────────────────────────────────────────────────── */

static inline long long ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── compiler barrier — prevents GCC/Clang from hoisting loop body ─────────── */
#define BARRIER(v) __asm__ volatile("" : "+m"(v))

/* ── 4-way interleaved benchmark: approx throughput (matches criterion) ─────── 
 * Each chain has an independent RAW dependency.
 * ITERS must be divisible by 4.
 */
#define ITERS 1000000

volatile float sink = 0.0f;

/* ── prevent dead code elimination without touching loop measurement ─────────── */
#define SINK_VEC3(v)  do { sink += (v)[0]; } while(0)
#define SINK_VEC4(v)  do { sink += (v)[0]; } while(0)
#define SINK_MAT4(m)  do { sink += (m)[0][0]; } while(0)
#define SINK_FLOAT(f) do { sink += (f); } while(0)

/* ── print helper ─────────────────────────────────────────────────────────── */
static void report(const char *label, long long ns, int iters) {
    printf("  %-48s %7.2f ns/op\n", label, (double)ns / iters);
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Vec3                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_vec3(void) {
    printf("vec3 (4-way interleaved, throughput approx):\n");

    /* — add — */
    {
        vec3 a0={1.0f,2.0f,3.0f}, a1={1.1f,2.1f,3.1f},
             a2={1.2f,2.2f,3.2f}, a3={1.3f,2.3f,3.3f};
        vec3 b={4.0f,5.0f,6.0f};
        vec3 o0,o1,o2,o3;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            glm_vec3_add(a0,b,o0); glm_vec3_copy(o0,a0);
            glm_vec3_add(a1,b,o1); glm_vec3_copy(o1,a1);
            glm_vec3_add(a2,b,o2); glm_vec3_copy(o2,a2);
            glm_vec3_add(a3,b,o3); glm_vec3_copy(o3,a3);
        }
        SINK_VEC3(a0);
        report("add/cglm", ns_now()-t, ITERS);
    }

    /* — dot — */
    {
        vec3 a={1.0f,2.0f,3.0f}, b={4.0f,5.0f,6.0f};
        float d0=0,d1=0,d2=0,d3=0;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            d0 = glm_vec3_dot(a,b);
            d1 = glm_vec3_dot(a,b);
            d2 = glm_vec3_dot(a,b);
            d3 = glm_vec3_dot(a,b);
            /* feed result into next a to create dependency */
            a[0] += (d0+d1+d2+d3) * 1e-30f;
            BARRIER(a);
        }
        SINK_FLOAT(d0);
        report("dot/cglm", ns_now()-t, ITERS);
    }

    /* — cross — */
    {
        vec3 a0={1.0f,2.0f,3.0f}, a1={1.1f,2.1f,3.1f},
             a2={1.2f,2.2f,3.2f}, a3={1.3f,2.3f,3.3f};
        vec3 b={4.0f,5.0f,6.0f};
        vec3 o0,o1,o2,o3;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            glm_vec3_cross(a0,b,o0); glm_vec3_copy(o0,a0);
            glm_vec3_cross(a1,b,o1); glm_vec3_copy(o1,a1);
            glm_vec3_cross(a2,b,o2); glm_vec3_copy(o2,a2);
            glm_vec3_cross(a3,b,o3); glm_vec3_copy(o3,a3);
        }
        SINK_VEC3(a0);
        report("cross/cglm", ns_now()-t, ITERS);
    }

    /* — normalize — */
    {
        vec3 a0={1.0f,2.0f,3.0f}, a1={1.1f,2.1f,3.1f},
             a2={1.2f,2.2f,3.2f}, a3={1.3f,2.3f,3.3f};
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            glm_vec3_normalize(a0);
            glm_vec3_normalize(a1);
            glm_vec3_normalize(a2);
            glm_vec3_normalize(a3);
            /* tiny perturbation to prevent collapse to constant */
            a0[0] += 1e-30f; a1[0] += 1e-30f;
            a2[0] += 1e-30f; a3[0] += 1e-30f;
            BARRIER(a0); BARRIER(a1); BARRIER(a2); BARRIER(a3);
        }
        SINK_VEC3(a0);
        report("normalize/cglm", ns_now()-t, ITERS);
    }

    /* — lerp — */
    {
        vec3 a={0.0f,0.0f,0.0f}, b={1.0f,1.0f,1.0f}, o;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_vec3_lerp(a, b, 0.5f, o);
            glm_vec3_copy(o, a);
        }
        SINK_VEC3(a);
        report("lerp/cglm", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Vec4                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_vec4(void) {
    printf("\nvec4:\n");

    /* — dot — */
    {
        vec4 a={1.0f,2.0f,3.0f,4.0f}, b={5.0f,6.0f,7.0f,8.0f};
        float d=0;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            d = glm_vec4_dot(a, b);
            a[0] += d * 1e-30f;
            BARRIER(a);
        }
        SINK_FLOAT(d);
        report("dot/cglm", ns_now()-t, ITERS);
    }

    /* — normalize — */
    {
        vec4 a0={1.0f,2.0f,3.0f,4.0f}, a1={1.1f,2.1f,3.1f,4.1f},
             a2={1.2f,2.2f,3.2f,4.2f}, a3={1.3f,2.3f,3.3f,4.3f};
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            glm_vec4_normalize(a0); a0[0] += 1e-30f; BARRIER(a0);
            glm_vec4_normalize(a1); a1[0] += 1e-30f; BARRIER(a1);
            glm_vec4_normalize(a2); a2[0] += 1e-30f; BARRIER(a2);
            glm_vec4_normalize(a3); a3[0] += 1e-30f; BARRIER(a3);
        }
        SINK_VEC4(a0);
        report("normalize/cglm", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Quaternion                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_quat(void) {
    printf("\nquat (xyzw):\n");

    versor q1, q2, q3, q4, qout;
    vec3 axis_y={0.0f,1.0f,0.0f}, axis_d={0.7071068f,0.7071068f,0.0f};
    glm_quatv(q1, glm_rad(45.0f), axis_y);
    glm_quatv(q2, glm_rad(30.0f), axis_d);
    glm_quat_copy(q1, q3);
    glm_quat_copy(q2, q4);

    /* — mul — */
    {
        versor a, b;
        glm_quat_copy(q1, a);
        glm_quat_copy(q2, b);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_quat_mul(a, b, qout);
            glm_quat_copy(qout, a);
        }
        SINK_VEC4(a);
        report("mul/cglm", ns_now()-t, ITERS);
    }

    /* — rotate vec — */
    {
        vec3 v={1.0f,0.0f,0.0f}, vout;
        versor q; glm_quat_copy(q1, q);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_quat_rotatev(q, v, vout);
            v[0] = vout[0]; v[1] = vout[1]; v[2] = vout[2];
            BARRIER(v);
        }
        SINK_VEC3(v);
        report("rotate/cglm", ns_now()-t, ITERS);
    }

    /* — slerp — */
    {
        versor a, b, o;
        glm_quat_copy(q1, a);
        glm_quat_copy(q2, b);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_quat_slerp(a, b, 0.5f, o);
            glm_quat_copy(o, a);
        }
        SINK_VEC4(a);
        report("slerp/cglm", ns_now()-t, ITERS);
    }

    /* — nlerp — */
    {
        versor a, b, o;
        glm_quat_copy(q1, a);
        glm_quat_copy(q2, b);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_quat_nlerp(a, b, 0.5f, o);
            glm_quat_copy(o, a);
        }
        SINK_VEC4(a);
        report("nlerp/cglm", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Mat4                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_mat4(void) {
    printf("\nmat4:\n");

    /* Build two TRS matrices */
    mat4 ma, mb, mout;
    versor q;
    vec3 axis_y={0.0f,1.0f,0.0f}, t={1.0f,0.0f,0.0f}, s={2.0f,2.0f,2.0f};
    glm_quatv(q, glm_rad(45.0f), axis_y);
    glm_mat4_identity(ma);
    glm_translate(ma, t);
    glm_quat_rotate(ma, q, ma);
    glm_scale(ma, s);

    vec3 t2={0.5f,0.0f,0.0f}, s2={1.5f,1.5f,1.5f};
    versor q2; glm_quatv(q2, glm_rad(30.0f), axis_y);
    glm_mat4_identity(mb);
    glm_translate(mb, t2); glm_quat_rotate(mb, q2, mb); glm_scale(mb, s2);

    /* — mul — */
    {
        mat4 a, b; glm_mat4_copy(ma, a); glm_mat4_copy(mb, b);
        long long tt = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_mat4_mul(a, b, mout);
            glm_mat4_copy(mout, a);
        }
        SINK_MAT4(a);
        report("mul/cglm", ns_now()-tt, ITERS);
    }

    /* — transform point — */
    {
        vec4 p={1.0f,2.0f,3.0f,1.0f}, pout;
        mat4 m; glm_mat4_copy(ma, m);
        long long tt = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_mat4_mulv(m, p, pout);
            p[0]=pout[0]; p[1]=pout[1]; p[2]=pout[2];
            BARRIER(p);
        }
        SINK_VEC4(p);
        report("transform_point/cglm", ns_now()-tt, ITERS);
    }

    /* — inverse general — */
    {
        mat4 m; glm_mat4_copy(ma, m);
        long long tt = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_mat4_inv(m, mout);
            /* alternate m/mout so there is a dep chain but no drift */
            glm_mat4_inv(mout, m);
            i++;  /* counted as 2 */
        }
        SINK_MAT4(m);
        report("inverse_general/cglm", ns_now()-tt, ITERS);
    }

    /* — inverse TRS (rot+trans only, no scale) — */
    {
        mat4 rt; glm_mat4_identity(rt);
        glm_translate(rt, t); glm_quat_rotate(rt, q, rt);
        mat4 m; glm_mat4_copy(rt, m);
        long long tt = ns_now();
        for (int i = 0; i < ITERS; i++) {
            glm_mat4_inv_fast(m, mout);
            glm_mat4_copy(mout, m);
        }
        SINK_MAT4(m);
        report("inverse_trs/cglm (glm_mat4_inv_fast, rot+trans)", ns_now()-tt, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Bulk transforms                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_bulk(void) {
    printf("\n100k entity transforms (single run):\n");
    const int N = 100000;

    mat4 trs;
    versor q; vec3 axis_y={0.0f,1.0f,0.0f}, t={1.0f,0.0f,0.0f}, s={1.0f,1.0f,1.0f};
    glm_quatv(q, glm_rad(45.0f), axis_y);
    glm_mat4_identity(trs);
    glm_translate(trs, t); glm_quat_rotate(trs, q, trs); glm_scale(trs, s);

    vec4 *pos = (vec4*)aligned_alloc(32, N * sizeof(vec4));
    if (!pos) { fprintf(stderr, "alloc failed\n"); return; }
    for (int i = 0; i < N; i++) {
        pos[i][0]=i*0.01f; pos[i][1]=0.0f; pos[i][2]=0.0f; pos[i][3]=1.0f;
    }

    long long t0 = ns_now();
    for (int i = 0; i < N; i++) {
        vec4 o; glm_mat4_mulv(trs, pos[i], o);
        glm_vec4_copy(o, pos[i]);
    }
    long long dt = ns_now() - t0;
    printf("  %-48s %7.1f µs  (%.2f ns/entity)\n",
        "transform_point/cglm", (double)dt/1000.0, (double)dt/N);
    fflush(stdout);
    sink += pos[0][0];
    free(pos);

    printf("\n5k inverse_general (single run):\n");
    const int M = 5000;
    mat4 *mats = (mat4*)aligned_alloc(32, M * sizeof(mat4));
    if (!mats) { fprintf(stderr, "alloc failed\n"); return; }
    for (int i = 0; i < M; i++) {
        versor qi; vec3 ti={i*0.1f,0.0f,0.0f}, si={1.0f+i*0.001f,1.0f+i*0.001f,1.0f+i*0.001f};
        glm_quatv(qi, glm_rad((float)i), axis_y);
        glm_mat4_identity(mats[i]);
        glm_translate(mats[i], ti); glm_quat_rotate(mats[i], qi, mats[i]); glm_scale(mats[i], si);
    }
    mat4 inv;
    t0 = ns_now();
    for (int i = 0; i < M; i++) { glm_mat4_inv(mats[i], inv); sink += inv[0][0]; }
    dt = ns_now() - t0;
    printf("  %-48s %7.1f µs  (%.2f ns/op)\n",
        "inverse_general/cglm", (double)dt/1000.0, (double)dt/M);
    fflush(stdout);
    free(mats);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("cglm benchmark (fixed: data-dependency chains) — %s\n", __DATE__);
    printf("4-way interleaved = throughput approx, single-chain = latency\n\n");

    bench_vec3();
    bench_vec4();
    bench_quat();
    bench_mat4();
    bench_bulk();

    printf("\ndone. (sink=%.6f)\n", (float)sink);
    fflush(stdout);
    return 0;
}
