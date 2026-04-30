/* crates/mid-math/benches/cglm_bench.c
 * Standalone C benchmark for cglm.
 * Compiled and timed by bench-vs-cglm.yml.
 *
 * Mirrors the same operations as vs_all.rs so numbers are comparable.
 * Uses clock_gettime(CLOCK_MONOTONIC) — nanosecond wall time.
 *
 * Build:
 *   gcc -O3 -march=native -o cglm_bench cglm_bench.c -lcglm -lm
 * Run:
 *   ./cglm_bench
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

#define BENCH(label, iters, body)                                  \
    do {                                                           \
        long long _t0 = ns_now();                                  \
        for (int _i = 0; _i < (iters); _i++) { body; }            \
        long long _dt = ns_now() - _t0;                            \
        printf("  %-45s %6.2f ns/op\n",                           \
               label, (double)_dt / (iters));                      \
        fflush(stdout);                                            \
    } while(0)

/* ── prevent dead-code elimination ─────────────────────────────────────────── */

volatile float sink_f = 0.0f;
volatile int   sink_i = 0;

static void consume_vec3(vec3 v) { sink_f += v[0]; }
static void consume_vec4(vec4 v) { sink_f += v[0]; }
static void consume_mat4(mat4 m) { sink_f += m[0][0]; }

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(void) {
    /* Force line-buffered stdout so output survives a crash or pipe.
     * Without this, block-buffering means a segfault silently discards
     * all printf output when piped through tee. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("cglm benchmark — compiled %s\n", __DATE__);
    printf("Operations run 1 000 000 iterations unless noted.\n\n");
    fflush(stdout);

    /* ── Vec3 ─────────────────────────────────────────────────────────────── */
    printf("vec3:\n");
    fflush(stdout);
    {
        vec3 a = {1.0f, 2.0f, 3.0f};
        vec3 b = {4.0f, 5.0f, 6.0f};
        vec3 out;

        BENCH("add/cglm", 1000000, {
            glm_vec3_add(a, b, out);
            consume_vec3(out);
        });

        float dot_result;
        BENCH("dot/cglm", 1000000, {
            dot_result = glm_vec3_dot(a, b);
            sink_f += dot_result;
        });

        BENCH("cross/cglm", 1000000, {
            glm_vec3_cross(a, b, out);
            consume_vec3(out);
        });

        BENCH("normalize/cglm", 1000000, {
            glm_vec3_copy(a, out);
            glm_vec3_normalize(out);
            consume_vec3(out);
        });

        BENCH("lerp/cglm", 1000000, {
            glm_vec3_lerp(a, b, 0.5f, out);
            consume_vec3(out);
        });
    }

    /* ── Vec4 ─────────────────────────────────────────────────────────────── */
    printf("\nvec4:\n");
    fflush(stdout);
    {
        vec4 a = {1.0f, 2.0f, 3.0f, 4.0f};
        vec4 b = {5.0f, 6.0f, 7.0f, 8.0f};
        vec4 out;

        BENCH("dot/cglm", 1000000, {
            float d = glm_vec4_dot(a, b);
            sink_f += d;
        });

        BENCH("normalize/cglm", 1000000, {
            glm_vec4_copy(a, out);
            glm_vec4_normalize(out);
            consume_vec4(out);
        });
    }

    /* ── Quaternion ───────────────────────────────────────────────────────── */
    printf("\nquat (xyzw):\n");
    fflush(stdout);
    {
        /* cglm quat layout: [x, y, z, w] */
        versor q1, q2, qout;
        vec3 axis_y = {0.0f, 1.0f, 0.0f};
        vec3 axis_d = {0.7071068f, 0.7071068f, 0.0f}; /* normalized (1,1,0) */
        glm_quatv(q1, glm_rad(45.0f), axis_y);
        glm_quatv(q2, glm_rad(30.0f), axis_d);

        BENCH("mul/cglm", 1000000, {
            glm_quat_mul(q1, q2, qout);
            consume_vec4(qout);
        });

        vec3 v = {1.0f, 0.0f, 0.0f};
        vec3 vout;
        BENCH("rotate/cglm", 1000000, {
            glm_quat_rotatev(q1, v, vout);
            consume_vec3(vout);
        });

        BENCH("slerp/cglm", 1000000, {
            glm_quat_slerp(q1, q2, 0.5f, qout);
            consume_vec4(qout);
        });

        BENCH("nlerp/cglm", 1000000, {
            glm_quat_nlerp(q1, q2, 0.5f, qout);
            consume_vec4(qout);
        });
    }

    /* ── Mat4 ─────────────────────────────────────────────────────────────── */
    printf("\nmat4:\n");
    fflush(stdout);
    {
        /* Build a TRS matrix: translate(1,0,0) * rotateY(45deg) * scale(2) */
        mat4 ma, mb, mout;
        versor q;
        vec3 axis_y = {0.0f, 1.0f, 0.0f};
        vec3 t      = {1.0f, 0.0f, 0.0f};
        vec3 s      = {2.0f, 2.0f, 2.0f};

        glm_quatv(q, glm_rad(45.0f), axis_y);
        glm_mat4_identity(ma);
        glm_translate(ma, t);
        glm_quat_rotate(ma, q, ma);
        glm_scale(ma, s);

        vec3 t2 = {0.5f, 0.0f, 0.0f};
        vec3 s2 = {1.5f, 1.5f, 1.5f};
        versor q2;
        glm_quatv(q2, glm_rad(30.0f), axis_y);
        glm_mat4_identity(mb);
        glm_translate(mb, t2);
        glm_quat_rotate(mb, q2, mb);
        glm_scale(mb, s2);

        BENCH("mul/cglm", 1000000, {
            glm_mat4_mul(ma, mb, mout);
            consume_mat4(mout);
        });

        vec4 p_in  = {1.0f, 2.0f, 3.0f, 1.0f};
        vec4 p_out;
        BENCH("transform_point/cglm", 1000000, {
            glm_mat4_mulv(ma, p_in, p_out);
            consume_vec4(p_out);
        });

        BENCH("inverse_general/cglm", 1000000, {
            glm_mat4_inv(ma, mout);
            consume_mat4(mout);
        });

        /* cglm inverse_trs equivalent: glm_mat4_inv_fast is only valid for
         * pure rotation+translation (no scale). For TRS we use glm_mat4_inv. */
        BENCH("inverse_trs/cglm (glm_mat4_inv_fast, rot+trans only)", 1000000, {
            /* Remove scale first for a fair comparison */
            mat4 rot_trans;
            glm_mat4_identity(rot_trans);
            glm_translate(rot_trans, t);
            glm_quat_rotate(rot_trans, q, rot_trans);
            glm_mat4_inv_fast(rot_trans, mout);
            consume_mat4(mout);
        });
    }

    /* ── 100k entity bulk transform ──────────────────────────────────────── */
    printf("\n100k entity transforms (single run, us):\n");
    fflush(stdout);
    {
        const int N = 100000;
        mat4 trs;
        versor q;
        vec3 axis_y = {0.0f, 1.0f, 0.0f};
        vec3 t      = {1.0f, 0.0f, 0.0f};
        vec3 s      = {1.0f, 1.0f, 1.0f};
        glm_quatv(q, glm_rad(45.0f), axis_y);
        glm_mat4_identity(trs);
        glm_translate(trs, t);
        glm_quat_rotate(trs, q, trs);
        glm_scale(trs, s);

        /* aligned_alloc: 32-byte alignment satisfies AVX intrinsics in cglm.
         * Standard malloc only guarantees 16 bytes. With -march=native GCC
         * emits AVX2 loads (_mm256_load_ps) which require 32-byte alignment.
         * Passing a 16-byte-aligned pointer causes SIGSEGV with no error output
         * because block-buffered stdout discards the printf buffer on crash. */
        vec4 *positions = (vec4*)aligned_alloc(32, N * sizeof(vec4));
        if (!positions) { fprintf(stderr, "aligned_alloc failed\n"); return 1; }

        for (int i = 0; i < N; i++) {
            positions[i][0] = i * 0.01f;
            positions[i][1] = 0.0f;
            positions[i][2] = 0.0f;
            positions[i][3] = 1.0f;
        }

        long long t0 = ns_now();
        for (int i = 0; i < N; i++) {
            vec4 out;
            glm_mat4_mulv(trs, positions[i], out);
            glm_vec4_copy(out, positions[i]);
        }
        long long dt = ns_now() - t0;
        printf("  %-45s %6.1f us  (%4.1f ns/entity)\n",
               "transform_point/cglm",
               (double)dt / 1000.0,
               (double)dt / N);
        fflush(stdout);

        sink_f += positions[0][0];
        free(positions);
    }

    /* ── 5k bulk inverse ─────────────────────────────────────────────────── */
    printf("\n5k inverse_general (single run, us):\n");
    fflush(stdout);
    {
        const int N = 5000;
        /* Same aligned_alloc fix — mat4 is 64 bytes, AVX wants 32-byte
         * alignment on the base pointer for _mm256 loads inside glm_mat4_inv. */
        mat4 *mats = (mat4*)aligned_alloc(32, N * sizeof(mat4));
        if (!mats) { fprintf(stderr, "aligned_alloc failed\n"); return 1; }
        mat4 mout;

        for (int i = 0; i < N; i++) {
            versor q;
            vec3 axis_y = {0.0f, 1.0f, 0.0f};
            vec3 ti = {i * 0.1f, 0.0f, 0.0f};
            vec3 si = {1.0f + i * 0.001f, 1.0f + i * 0.001f, 1.0f + i * 0.001f};
            glm_quatv(q, glm_rad((float)i), axis_y);
            glm_mat4_identity(mats[i]);
            glm_translate(mats[i], ti);
            glm_quat_rotate(mats[i], q, mats[i]);
            glm_scale(mats[i], si);
        }

        long long t0 = ns_now();
        for (int i = 0; i < N; i++) {
            glm_mat4_inv(mats[i], mout);
            consume_mat4(mout);
        }
        long long dt = ns_now() - t0;
        printf("  %-45s %6.1f us  (%5.1f ns/op)\n",
               "inverse_general/cglm",
               (double)dt / 1000.0,
               (double)dt / N);
        fflush(stdout);

        free(mats);
    }

    printf("\ndone. (sink=%f)\n", (float)sink_f);
    fflush(stdout);
    return 0;
                }
