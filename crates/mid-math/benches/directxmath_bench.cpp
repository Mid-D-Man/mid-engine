/* crates/mid-math/benches/directxmath_bench.cpp
 * DirectXMath benchmark — the AAA industry standard.
 * Demonstrates: XMVECTOR (register type) vs XMFLOAT3 (storage type) philosophy.
 * We benchmark with data kept IN registers between ops (no store/load per iter)
 * which is how both DirectXMath and our Rust types (SIMD register = the type) work.
 *
 * Build: see bench-vs-directxmath.yml
 */

// DirectXMath Linux compatibility
#ifndef _WIN32
  #ifndef __cdecl
    #define __cdecl
  #endif
#endif

#include <DirectXMath.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

using namespace DirectX;

/* ── timing ────────────────────────────────────────────────────────────────── */
static inline long long ns_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

#define ITERS 1000000

volatile float sink = 0.0f;

static void report(const char* label, long long ns, int iters) {
    printf("  %-52s %7.2f ns/op\n", label, (double)ns / iters);
    fflush(stdout);
}

/* ── barrier: prevents GCC from reordering across ─────────────────────────── */
#define BARRIER(x) asm volatile("" : "+x"(x))

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Vec3 operations (stored in XMVECTOR — no load/store overhead)             */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_vec3() {
    printf("vec3 (XMVECTOR register type, 4-way interleaved):\n");

    XMVECTOR a0=XMVectorSet(1,2,3,0), a1=XMVectorSet(1.1f,2.1f,3.1f,0),
             a2=XMVectorSet(1.2f,2.2f,3.2f,0), a3=XMVectorSet(1.3f,2.3f,3.3f,0);
    XMVECTOR b =XMVectorSet(4,5,6,0);

    /* — add — */
    {
        XMVECTOR v0=a0,v1=a1,v2=a2,v3=a3;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            v0 = XMVectorAdd(v0, b);
            v1 = XMVectorAdd(v1, b);
            v2 = XMVectorAdd(v2, b);
            v3 = XMVectorAdd(v3, b);
        }
        BARRIER(v0);
        sink += XMVectorGetX(v0);
        report("add/directxmath", ns_now()-t, ITERS);
    }

    /* — dot — */
    {
        XMVECTOR v=a0, bv=b;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            XMVECTOR d = XMVector3Dot(v, bv);
            /* feed dot scalar back into v to create dependency */
            v = XMVectorSetX(v, XMVectorGetX(v) + XMVectorGetX(d) * 1e-30f);
            BARRIER(v);
        }
        sink += XMVectorGetX(v);
        report("dot/directxmath", ns_now()-t, ITERS);
    }

    /* — cross — */
    {
        XMVECTOR v0=a0,v1=a1,v2=a2,v3=a3;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            v0 = XMVector3Cross(v0, b);
            v1 = XMVector3Cross(v1, b);
            v2 = XMVector3Cross(v2, b);
            v3 = XMVector3Cross(v3, b);
        }
        BARRIER(v0);
        sink += XMVectorGetX(v0);
        report("cross/directxmath", ns_now()-t, ITERS);
    }

    /* — normalize — */
    {
        XMVECTOR v0=a0,v1=a1,v2=a2,v3=a3;
        XMVECTOR eps = XMVectorSet(1e-30f,0,0,0);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i += 4) {
            v0 = XMVector3Normalize(v0); v0 = XMVectorAdd(v0, eps); BARRIER(v0);
            v1 = XMVector3Normalize(v1); v1 = XMVectorAdd(v1, eps); BARRIER(v1);
            v2 = XMVector3Normalize(v2); v2 = XMVectorAdd(v2, eps); BARRIER(v2);
            v3 = XMVector3Normalize(v3); v3 = XMVectorAdd(v3, eps); BARRIER(v3);
        }
        sink += XMVectorGetX(v0);
        report("normalize/directxmath", ns_now()-t, ITERS);
    }

    /* — lerp — */
    {
        XMVECTOR v=a0, e=XMVectorSet(1,1,1,0);
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            v = XMVectorLerp(v, e, 0.5f);
        }
        BARRIER(v);
        sink += XMVectorGetX(v);
        report("lerp/directxmath", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Quaternion                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_quat() {
    printf("\nquat (XMVECTOR, 4-component):\n");

    XMVECTOR q1 = XMQuaternionRotationAxis(XMVectorSet(0,1,0,0), XMConvertToRadians(45.0f));
    XMVECTOR q2 = XMQuaternionRotationAxis(
        XMVector3Normalize(XMVectorSet(1,1,0,0)), XMConvertToRadians(30.0f));

    /* — mul — */
    {
        XMVECTOR a=q1, b=q2;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = XMQuaternionMultiply(a, b);
        }
        BARRIER(a);
        sink += XMVectorGetX(a);
        report("mul/directxmath", ns_now()-t, ITERS);
    }

    /* — rotate vec — */
    {
        XMVECTOR v=XMVectorSet(1,0,0,0), q=q1;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            v = XMVector3Rotate(v, q);
            BARRIER(v);
        }
        sink += XMVectorGetX(v);
        report("rotate/directxmath", ns_now()-t, ITERS);
    }

    /* — slerp — */
    {
        XMVECTOR a=q1, b=q2;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = XMQuaternionSlerp(a, b, 0.5f);
        }
        BARRIER(a);
        sink += XMVectorGetX(a);
        report("slerp/directxmath", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Mat4                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_mat4() {
    printf("\nmat4:\n");

    XMVECTOR q1 = XMQuaternionRotationAxis(XMVectorSet(0,1,0,0), XMConvertToRadians(45.0f));
    XMVECTOR q2 = XMQuaternionRotationAxis(XMVectorSet(0,1,0,0), XMConvertToRadians(30.0f));

    XMMATRIX ma = XMMatrixScaling(2,2,2)
                * XMMatrixRotationQuaternion(q1)
                * XMMatrixTranslation(1,0,0);
    XMMATRIX mb = XMMatrixScaling(1.5f,1.5f,1.5f)
                * XMMatrixRotationQuaternion(q2)
                * XMMatrixTranslation(0.5f,0,0);

    /* — mul — */
    {
        XMMATRIX a=ma, b=mb;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            a = XMMatrixMultiply(a, b);
        }
        /* access to prevent elimination */
        XMFLOAT4X4 tmp; XMStoreFloat4x4(&tmp, a);
        sink += tmp._11;
        report("mul/directxmath", ns_now()-t, ITERS);
    }

    /* — transform point — */
    {
        XMVECTOR p = XMVectorSet(1,2,3,1);
        XMMATRIX m = ma;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            p = XMVector3Transform(p, m);
            BARRIER(p);
        }
        sink += XMVectorGetX(p);
        report("transform_point/directxmath", ns_now()-t, ITERS);
    }

    /* — inverse general — */
    {
        XMMATRIX m=ma;
        XMVECTOR det;
        long long t = ns_now();
        for (int i = 0; i < ITERS; i++) {
            m = XMMatrixInverse(&det, m);
        }
        XMFLOAT4X4 tmp; XMStoreFloat4x4(&tmp, m);
        sink += tmp._11;
        report("inverse_general/directxmath", ns_now()-t, ITERS);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Bulk 100k                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void bench_bulk() {
    printf("\n100k entity transforms (single run):\n");
    const int N = 100000;

    XMVECTOR q = XMQuaternionRotationAxis(XMVectorSet(0,1,0,0), XMConvertToRadians(45.0f));
    XMMATRIX trs = XMMatrixScaling(1,1,1) * XMMatrixRotationQuaternion(q)
                 * XMMatrixTranslation(1,0,0);

    // Aligned storage for XMVECTOR arrays
    XMFLOAT4 *pos_store = (XMFLOAT4*)_mm_malloc(N * sizeof(XMFLOAT4), 16);
    if (!pos_store) { fprintf(stderr, "alloc failed\n"); return; }

    for (int i = 0; i < N; i++) {
        pos_store[i] = {i * 0.01f, 0.0f, 0.0f, 1.0f};
    }

    long long t0 = ns_now();
    for (int i = 0; i < N; i++) {
        XMVECTOR p = XMLoadFloat4(&pos_store[i]);
        p = XMVector3Transform(p, trs);
        XMStoreFloat4(&pos_store[i], p);
    }
    long long dt = ns_now() - t0;

    printf("  %-52s %7.1f µs  (%.2f ns/entity)\n",
        "transform_point/directxmath", (double)dt/1000.0, (double)dt/N);
    fflush(stdout);
    sink += pos_store[0].x;
    _mm_free(pos_store);
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main() {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("DirectXMath benchmark — compiled %s\n", __DATE__);
    printf("Note: XMVECTOR = register type (no load/store per op, like our Vec3)\n\n");

    bench_vec3();
    bench_quat();
    bench_mat4();
    bench_bulk();

    printf("\ndone. (sink=%.6f)\n", (float)sink);
    fflush(stdout);
    return 0;
}
