/**
 * @file avx.cppm
 * @brief SSE/AVX SIMD implementations of all vector, matrix, quaternion, box, and geometric operations.
 */
module;

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

export module sgl:simd;

import :types;
import :scalar;

export namespace sgl {

/* Load vec2 into lower 2 lanes of __m128, upper lanes zeroed. */
inline __m128 load(const vec2& v) noexcept {
    return _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&v.x)));
}

inline void store(const __m128 vals, vec2& dst) noexcept {
    _mm_store_sd(reinterpret_cast<double*>(&dst.x), _mm_castps_pd(vals));
}

inline __m128 load(const vec3& v) noexcept {
    return _mm_load_ps(&v.x);
}

inline void store(const __m128 vals, vec3& dst) noexcept {
    _mm_store_ps(&dst.x, vals);
}

inline __m128 load(const vec4& v) noexcept {
    return _mm_load_ps(&v.x);
}

inline void store(const __m128 vals, vec4& dst) noexcept {
    _mm_store_ps(&dst.x, vals);
}

inline __m128 load(const box2d& b) noexcept {
    return _mm_load_ps(&b.min.x);
}

inline void store(const __m128 v, box2d& b) noexcept {
    _mm_store_ps(&b.min.x, v);
}

inline __m128 load(const quat& q) noexcept {
    return _mm_load_ps(&q.x);
}

inline quat store_quat(const __m128 v) noexcept {
    quat r{};
    _mm_store_ps(&r.x, v);
    return r;
}

template <typename Box> struct box_traits;

template <> struct box_traits<box2d> {
    using vec_type = vec2;
    static constexpr std::int32_t lane_mask = 0x3;
    static constexpr std::int32_t dp_mask = 0x31;
};

template <> struct box_traits<box3d> {
    using vec_type = vec3;
    static constexpr std::int32_t lane_mask = 0x7;
    static constexpr std::int32_t dp_mask = 0x71;
};

template <typename Box> using box_vec_t = typename box_traits<Box>::vec_type;

inline void load_min_max(const box2d& b, __m128& out_min, __m128& out_max) noexcept {
    const __m128 v = _mm_load_ps(&b.min.x);
    /* {minX, minY, ?, ?} */
    out_min = v;
    /* {maxX, maxY, ?, ?} */
    out_max = _mm_movehl_ps(v, v);
}

inline box2d make_box_from_min_max(const __m128 lo, const __m128 hi, box_traits<box2d>) noexcept {
    box2d b{};
    _mm_store_ps(&b.min.x, _mm_movelh_ps(lo, hi));
    return b;
}

inline void load_min_max(const box3d& b, __m128& out_min, __m128& out_max) noexcept {
    /* {minX, minY, minZ, pad} */
    out_min = _mm_load_ps(&b.min.x);
    /* {maxX, maxY, maxZ, pad} */
    out_max = _mm_load_ps(&b.max.x);
}

inline box3d make_box_from_min_max(const __m128 lo, const __m128 hi, box_traits<box3d>) noexcept {
    box3d b{};
    _mm_store_ps(&b.min.x, lo);
    _mm_store_ps(&b.max.x, hi);
    return b;
}

/* Convenience wrapper — deduces box tag from type. */
template <typename Box> Box make_box(const __m128 lo, const __m128 hi) noexcept {
    return make_box_from_min_max(lo, hi, box_traits<Box>{});
}

template <typename Vec> __m128 safe_divisor(__m128 v) noexcept {
    return v;
}

template <> inline __m128 safe_divisor<vec3>(__m128 v) noexcept {
    /* Replace lane 3 (the pad) with 1.0f so we don't divide 0/0 */
    return _mm_blend_ps(v, _mm_set1_ps(1.0f), 0b1000);
}

template <> inline __m128 safe_divisor<vec2>(__m128 v) noexcept {
    /* Replace lanes 2 and 3 with 1.0f */
    return _mm_blend_ps(v, _mm_set1_ps(1.0f), 0b1100);
}

template <typename T> consteval std::int32_t lane_mask() noexcept {
    using vec = std::remove_cvref_t<T>;

    if constexpr (std::is_same_v<vec, vec2>) {
        return 0b0011;
    } else if constexpr (std::is_same_v<vec, vec3>) {
        return 0b0111;
    } else if constexpr (std::is_same_v<vec, vec4>) {
        return 0b1111;
    } else {
        static_assert(false);
    }
}

/* result in lane 0 only, for scalar extraction */
template <typename T> consteval std::int32_t dp_scalar_mask() noexcept {
    using vec = std::remove_cvref_t<T>;

    if constexpr (std::is_same_v<vec, vec2>) {
        return 0x31;
    } else if constexpr (std::is_same_v<vec, vec3>) {
        return 0x71;
    } else if constexpr (std::is_same_v<vec, vec4>) {
        return 0xF1;
    } else {
        static_assert(false);
    }
}

/* result broadcast to all active lanes, for normalize, etc. */
template <typename T> consteval std::int32_t dp_broadcast_mask() noexcept {
    using vec = std::remove_cvref_t<T>;

    if constexpr (std::is_same_v<vec, vec2>) {
        return 0x3F;
    } else if constexpr (std::is_same_v<vec, vec3>) {
        return 0x7F;
    } else if constexpr (std::is_same_v<vec, vec4>) {
        return 0xFF;
    } else {
        static_assert(false);
    }
}

/* Newton-Raphson refined rsqrt: ~23-bit accuracy vs ~12 for _mm_rsqrt_ps alone.
 * x' = x * (3 - v * x²) * 0.5 */
inline __m128 rsqrt_nr(const __m128 v) noexcept {
    const auto approx{_mm_rsqrt_ps(v)};
    const auto vxx{_mm_mul_ps(v, _mm_mul_ps(approx, approx))};
    return _mm_mul_ps(_mm_mul_ps(approx, _mm_sub_ps(_mm_set1_ps(3.0f), vxx)), _mm_set1_ps(0.5f));
}

/* ============================================================
 * mat4 detail helpers
 * ============================================================ */

struct mat4_cols {
    __m128 c[4];
};

inline mat4_cols load_cols(const mat4& m) noexcept {
    return {{_mm_load_ps(&m.cols[0].x), _mm_load_ps(&m.cols[1].x), _mm_load_ps(&m.cols[2].x), _mm_load_ps(&m.cols[3].x)}};
}

inline mat4 store_cols(const __m128 c0, const __m128 c1, const __m128 c2, const __m128 c3) noexcept {
    mat4 r{};
    _mm_store_ps(&r.cols[0].x, c0);
    _mm_store_ps(&r.cols[1].x, c1);
    _mm_store_ps(&r.cols[2].x, c2);
    _mm_store_ps(&r.cols[3].x, c3);
    return r;
}

/* M * v as a linear combination of columns (used for mat4 * vec and mat4 * mat4). */
inline __m128 linear_combine(const mat4_cols& m, const __m128 v) noexcept {
    auto r{_mm_mul_ps(m.c[0], _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)))};
    r = _mm_fmadd_ps(m.c[1], _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)), r);
    r = _mm_fmadd_ps(m.c[2], _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)), r);
    r = _mm_fmadd_ps(m.c[3], _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)), r);
    return r;
}

/* The 12 sub-determinants shared by determinant and inverse. */
struct sub_dets {
    float s0, s1, s2, s3, s4, s5;
    float t0, t1, t2, t3, t4, t5;
};

inline sub_dets compute_sub_dets(const mat4& m) noexcept {
    return {
        m.cols[0].x * m.cols[1].y - m.cols[0].y * m.cols[1].x,
        m.cols[0].x * m.cols[1].z - m.cols[0].z * m.cols[1].x,
        m.cols[0].x * m.cols[1].w - m.cols[0].w * m.cols[1].x,
        m.cols[0].y * m.cols[1].z - m.cols[0].z * m.cols[1].y,
        m.cols[0].y * m.cols[1].w - m.cols[0].w * m.cols[1].y,
        m.cols[0].z * m.cols[1].w - m.cols[0].w * m.cols[1].z,
        m.cols[2].x * m.cols[3].y - m.cols[2].y * m.cols[3].x,
        m.cols[2].x * m.cols[3].z - m.cols[2].z * m.cols[3].x,
        m.cols[2].x * m.cols[3].w - m.cols[2].w * m.cols[3].x,
        m.cols[2].y * m.cols[3].z - m.cols[2].z * m.cols[3].y,
        m.cols[2].y * m.cols[3].w - m.cols[2].w * m.cols[3].y,
        m.cols[2].z * m.cols[3].w - m.cols[2].w * m.cols[3].z,
    };
}

} // namespace sgl

export namespace sgl {

/* non-constexpr but simd and branchless */
template <vector Vec> bool nearly_equal(const Vec& lhs, const Vec& rhs, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    const __m128 a = load(lhs);
    const __m128 b = load(rhs);

    /* abs(a-b) */
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    /* Exact equality (handles ±inf) */
    const __m128 exact = _mm_cmpeq_ps(a, b);

    /* absolute check */
    const __m128 diff = _mm_andnot_ps(sign_mask, _mm_sub_ps(a, b));
    const __m128 abs_ok = _mm_cmple_ps(diff, _mm_set1_ps(abs_tol));

    /* relative check */
    const __m128 abs_a = _mm_andnot_ps(sign_mask, a);
    const __m128 abs_b = _mm_andnot_ps(sign_mask, b);
    const __m128 largest = _mm_max_ps(abs_a, abs_b);
    const __m128 rel_ok = _mm_cmple_ps(diff, _mm_mul_ps(largest, _mm_set1_ps(rel_tol)));

    const __m128 ok = _mm_or_ps(exact, _mm_or_ps(abs_ok, rel_ok));

    constexpr auto mask{lane_mask<Vec>()};
    return (_mm_movemask_ps(ok) & mask) == mask;
}

template <vector Vec> bool nearly_zero(const Vec& val, const float tol = fp32_abs_tol) noexcept {
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    const __m128 abs_val = _mm_andnot_ps(sign_mask, load(val));
    const __m128 ok = _mm_cmplt_ps(abs_val, _mm_set1_ps(tol));

    constexpr auto mask{lane_mask<Vec>()};
    return (_mm_movemask_ps(ok) & mask) == mask;
}

template <vector Vec> bool any_nearly_zero(const Vec& val, const float tol = fp32_abs_tol) noexcept {
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    const __m128 abs_val = _mm_andnot_ps(sign_mask, load(val));
    const __m128 ok = _mm_cmplt_ps(abs_val, _mm_set1_ps(tol));

    constexpr auto mask{lane_mask<Vec>()};
    return (_mm_movemask_ps(ok) & mask) != 0;
}

/* unary - operator */
template <vector Vec> Vec operator-(const Vec& v) noexcept {
    Vec out{};
    store(_mm_xor_ps(load(v), _mm_set1_ps(-0.0f)), out);
    return out;
}

template <vector Vec> Vec operator+(const Vec& lhs, const Vec& rhs) noexcept {
    Vec out{};
    store(_mm_add_ps(load(lhs), load(rhs)), out);
    return out;
}

template <vector Vec> Vec operator-(const Vec& lhs, const Vec& rhs) noexcept {
    Vec out{};
    store(_mm_sub_ps(load(lhs), load(rhs)), out);
    return out;
}

template <vector Vec> Vec operator*(const Vec& lhs, const Vec& rhs) noexcept {
    Vec out{};
    store(_mm_mul_ps(load(lhs), load(rhs)), out);
    return out;
}

template <vector Vec> Vec operator/(const Vec& lhs, const Vec& rhs) noexcept {
    assert(!any_nearly_zero(rhs));
    Vec out{};
    store(_mm_div_ps(load(lhs), safe_divisor<Vec>(load(rhs))), out);
    return out;
}

template <vector Vec> Vec& operator+=(Vec& lhs, const Vec& rhs) noexcept {
    store(_mm_add_ps(load(lhs), load(rhs)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator-=(Vec& lhs, const Vec& rhs) noexcept {
    store(_mm_sub_ps(load(lhs), load(rhs)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator*=(Vec& lhs, const Vec& rhs) noexcept {
    store(_mm_mul_ps(load(lhs), load(rhs)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator/=(Vec& lhs, const Vec& rhs) noexcept {
    assert(!any_nearly_zero(rhs));
    store(_mm_div_ps(load(lhs), safe_divisor<Vec>(load(rhs))), lhs);
    return lhs;
}

template <vector Vec> Vec operator+(const Vec& lhs, const float scalar) noexcept {
    Vec out{};
    store(_mm_add_ps(load(lhs), _mm_set1_ps(scalar)), out);
    return out;
}

template <vector Vec> Vec operator-(const Vec& lhs, const float scalar) noexcept {
    Vec out{};
    store(_mm_sub_ps(load(lhs), _mm_set1_ps(scalar)), out);
    return out;
}

template <vector Vec> Vec operator*(const Vec& lhs, const float scalar) noexcept {
    Vec out{};
    store(_mm_mul_ps(load(lhs), _mm_set1_ps(scalar)), out);
    return out;
}

template <vector Vec> Vec operator/(const Vec& lhs, const float scalar) noexcept {
    assert(!nearly_zero(scalar));
    Vec out{};
    store(_mm_div_ps(load(lhs), _mm_set1_ps(scalar)), out);
    return out;
}

template <vector Vec> Vec& operator+=(Vec& lhs, const float scalar) noexcept {
    store(_mm_add_ps(load(lhs), _mm_set1_ps(scalar)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator-=(Vec& lhs, const float scalar) noexcept {
    store(_mm_sub_ps(load(lhs), _mm_set1_ps(scalar)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator*=(Vec& lhs, const float scalar) noexcept {
    store(_mm_mul_ps(load(lhs), _mm_set1_ps(scalar)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator/=(Vec& lhs, const float scalar) noexcept {
    assert(!nearly_zero(scalar));
    store(_mm_div_ps(load(lhs), _mm_set1_ps(scalar)), lhs);
    return lhs;
}

/* commutative */
template <vector Vec> Vec operator+(const float scalar, const Vec& rhs) noexcept {
    return rhs + scalar;
}

template <vector Vec> Vec operator*(const float scalar, const Vec& rhs) noexcept {
    return rhs * scalar;
}

/* non-commutative */
template <vector Vec> Vec operator-(const float scalar, const Vec& rhs) noexcept {
    Vec out{};
    store(_mm_sub_ps(_mm_set1_ps(scalar), load(rhs)), out);
    return out;
}

template <vector Vec> Vec operator/(const float scalar, const Vec& rhs) noexcept {
    assert(!any_nearly_zero(rhs));
    Vec out{};
    store(_mm_div_ps(_mm_set1_ps(scalar), safe_divisor<Vec>(load(rhs))), out);
    return out;
}

template <vector Vec> float dot(const Vec& a, const Vec& b) noexcept {
    return _mm_cvtss_f32(_mm_dp_ps(load(a), load(b), dp_scalar_mask<Vec>()));
}

template <vector Vec> float length_squared(const Vec& v) noexcept {
    return dot(v, v);
}

template <vector Vec> float length(const Vec& v) noexcept {
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(load(v), load(v), dp_scalar_mask<Vec>())));
}

/* Fast in-place normalize. _mm_rsqrt_ps gives ~12 bits (~3.5 decimal digits) — fine for directions, not for physics. */
template <vector Vec> void normalize_fast(Vec& vec) noexcept {
    const __m128 v = load(vec);
    const __m128 inv_len = _mm_rsqrt_ps(_mm_dp_ps(v, v, dp_broadcast_mask<Vec>()));
    store(_mm_mul_ps(v, inv_len), vec);
}

/* Precise in-place normalize using full sqrt. Slower than normalize_fast but accurate to full float precision. */
template <vector Vec> void normalize(Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    const __m128 v = load(vec);
    const __m128 len = _mm_sqrt_ps(_mm_dp_ps(v, v, dp_broadcast_mask<Vec>()));
    store(_mm_div_ps(v, len), vec);
}

template <vector Vec> void normalize_nr(Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    const __m128 v = load(vec);
    const __m128 dot = _mm_dp_ps(v, v, dp_broadcast_mask<Vec>());
    const __m128 inv_len = rsqrt_nr(dot);
    store(_mm_mul_ps(v, inv_len), vec);
}

template <vector Vec> Vec normalized_fast(const Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    Vec result{};
    const __m128 v = load(vec);
    const __m128 inv_len = _mm_rsqrt_ps(_mm_dp_ps(v, v, dp_broadcast_mask<Vec>()));
    store(_mm_mul_ps(v, inv_len), result);
    return result;
}

template <vector Vec> Vec normalized_nr(const Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    Vec result{};
    const __m128 v = load(vec);
    const __m128 dot = _mm_dp_ps(v, v, dp_broadcast_mask<Vec>());
    const __m128 inv_len = rsqrt_nr(dot);
    store(_mm_mul_ps(v, inv_len), result);
    return result;
}

template <vector Vec> Vec normalized(const Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    Vec result{};
    const __m128 v = load(vec);
    const __m128 len = _mm_sqrt_ps(_mm_dp_ps(v, v, dp_broadcast_mask<Vec>()));
    store(_mm_div_ps(v, len), result);
    return result;
}

/* branchless, _mm_blendv_ps selects per-lane based on the sign bit of the mask. */
template <vector Vec> void normalize_safe(Vec& vec) noexcept {
    const __m128 v = load(vec);
    const __m128 dot = _mm_dp_ps(v, v, dp_broadcast_mask<Vec>());

    /* If dot near 0, return zero vector instead of inf/NaN */
    const __m128 zero = _mm_setzero_ps();
    const __m128 is_zero = _mm_cmple_ps(dot, _mm_set1_ps(fp32_abs_tol * fp32_abs_tol));

    const __m128 inv_len = rsqrt_nr(dot);
    const __m128 result = _mm_mul_ps(v, inv_len);

    /* blend: zero where is_zero is true, result otherwise */
    store(_mm_blendv_ps(result, zero, is_zero), vec);
}

template <vector Vec> Vec normalized_safe(const Vec& vec) noexcept {
    const __m128 v = load(vec);
    const __m128 dot = _mm_dp_ps(v, v, dp_broadcast_mask<Vec>());

    /* If dot near 0, return zero vector instead of inf/NaN */
    const __m128 zero = _mm_setzero_ps();
    const __m128 is_zero = _mm_cmple_ps(dot, _mm_set1_ps(fp32_abs_tol * fp32_abs_tol));

    const __m128 inv_len = rsqrt_nr(dot);
    const __m128 result = _mm_mul_ps(v, inv_len);

    /* blend: zero where is_zero is true, result otherwise */
    Vec out{};
    store(_mm_blendv_ps(result, zero, is_zero), out);
    return out;
}

/* Project v onto `onto` (assumes `onto` is already normalised). */
template <vector Vec> Vec project(const Vec& v, const Vec& onto) noexcept {
    const __m128 vv{load(v)};
    const __m128 nn{load(onto)};

    const __m128 d{_mm_dp_ps(vv, nn, dp_broadcast_mask<Vec>())};
    Vec out{};
    store(_mm_mul_ps(d, nn), out);
    return out;
}

/* Reflect v about normal (assumes normal is unit length): v - 2 * dot(v, n) * n. */
template <vector Vec> Vec reflect(const Vec& v, const Vec& normal) noexcept {
    const __m128 vv{load(v)};
    const __m128 nn{load(normal)};

    const __m128 d{_mm_dp_ps(vv, nn, dp_broadcast_mask<Vec>())};
    /* d + d = 2 * dot; avoids loading an extra 2.0f constant */
    const __m128 two_d_n{_mm_mul_ps(_mm_add_ps(d, d), nn)};
    Vec out{};
    store(_mm_sub_ps(vv, two_d_n), out);
    return out;
}

template <vector Vec> Vec min(const Vec& a, const Vec& b) noexcept {
    Vec out{};
    store(_mm_min_ps(load(a), load(b)), out);
    return out;
}

template <vector Vec> Vec max(const Vec& a, const Vec& b) noexcept {
    Vec out{};
    store(_mm_max_ps(load(a), load(b)), out);
    return out;
}

template <vector Vec> Vec abs(const Vec& v) noexcept {
    Vec out{};
    store(_mm_andnot_ps(_mm_set1_ps(-0.0f), load(v)), out);
    return out;
}

template <vector Vec> Vec clamp(const Vec& v, const Vec& lo, const Vec& hi) noexcept {
    Vec out{};
    store(_mm_min_ps(_mm_max_ps(load(v), load(lo)), load(hi)), out);
    return out;
}

template <vector Vec> Vec clamp(const Vec& v, const float lo, const float hi) noexcept {
    Vec out{};
    store(_mm_min_ps(_mm_max_ps(load(v), _mm_set1_ps(lo)), _mm_set1_ps(hi)), out);
    return out;
}

template <vector Vec> Vec lerp(const Vec& a, const Vec& b, const float t) noexcept {
    const __m128 va = load(a);
    Vec out{};
    store(_mm_fmadd_ps(_mm_set1_ps(t), _mm_sub_ps(load(b), va), va), out);
    return out;
}

template <vector Vec> float distance_squared(const Vec& a, const Vec& b) noexcept {
    const __m128 d = _mm_sub_ps(load(a), load(b));
    return _mm_cvtss_f32(_mm_dp_ps(d, d, dp_scalar_mask<Vec>()));
}

template <vector Vec> float distance(const Vec& a, const Vec& b) noexcept {
    const auto d = _mm_sub_ps(load(a), load(b));
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(d, d, dp_scalar_mask<Vec>())));
}

template <vector Vec> bool is_perpendicular(const Vec& a, const Vec& b, float tol = fp32_abs_tol) noexcept {
    /* Two vectors are perpendicular when their dot product is ~0 */
    const auto d{_mm_cvtss_f32(_mm_dp_ps(load(a), load(b), dp_scalar_mask<Vec>()))};
    return std::abs(d) <= tol;
}

template <vector Vec> bool is_parallel(const Vec& a, const Vec& b, float tol = fp32_rel_tol) noexcept {
    constexpr auto mask{dp_scalar_mask<Vec>()};

    /* Cauchy-Schwarz equality
     * dot(a,b)^2 ≈ dot(a,a) · dot(b,b) */
    const auto va = load(a);
    const auto vb = load(b);

    const auto ab{_mm_cvtss_f32(_mm_dp_ps(va, vb, mask))};
    const auto aa{_mm_cvtss_f32(_mm_dp_ps(va, va, mask))};
    const auto bb{_mm_cvtss_f32(_mm_dp_ps(vb, vb, mask))};

    return std::abs(ab * ab - aa * bb) <= tol * aa * bb;
}

template <vector Vec> bool all_greater_than(const Vec& a, const Vec& b) noexcept {
    constexpr auto mask{lane_mask<Vec>()};
    return (_mm_movemask_ps(_mm_cmpgt_ps(load(a), load(b))) & mask) == mask;
}

template <vector Vec> bool all_less_than(const Vec& a, const Vec& b) noexcept {
    constexpr auto mask{lane_mask<Vec>()};
    return (_mm_movemask_ps(_mm_cmplt_ps(load(a), load(b))) & mask) == mask;
}

template <vector Vec> bool any_greater_than(const Vec& a, const Vec& b) noexcept {
    constexpr auto mask{lane_mask<Vec>()};
    return (_mm_movemask_ps(_mm_cmpgt_ps(load(a), load(b))) & mask) != 0;
}

/* AABB containment: point inside box */
inline bool contains(const box3d& box, const vec3& point) {
    return all_greater_than(point, box.min) && all_less_than(point, box.max);
}

/* vec2: returns scalar (the z-component of the implied 3D cross) */
inline float cross(const vec2& a, const vec2& b) noexcept {
    const __m128 va = load(a);
    const __m128 vb = load(b);

    /* shuffle b to {b.y, b.x, ...} */
    const __m128 b_yx = _mm_shuffle_ps(vb, vb, _MM_SHUFFLE(3, 2, 0, 1));
    const __m128 mul = _mm_mul_ps(va, b_yx);

    /* mul = {a.x*b.y, a.y*b.x, ...}
     * subtract lane 1 from lane 0
     * movehdup: {[1],[1],[3],[3]} */
    return _mm_cvtss_f32(_mm_sub_ss(mul, _mm_movehdup_ps(mul)));
}

/* vec3, returns vec3 */
inline vec3 cross(const vec3& a, const vec3& b) noexcept {
    const __m128 va = load(a);
    const __m128 vb = load(b);

    /* result = {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x} */
    const __m128 a_yzx = _mm_shuffle_ps(va, va, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 b_yzx = _mm_shuffle_ps(vb, vb, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 result = _mm_sub_ps(_mm_mul_ps(va, b_yzx), _mm_mul_ps(a_yzx, vb));

    /* result is in zxy order, shuffle back to xyz */
    vec3 out{};
    store(_mm_shuffle_ps(result, result, _MM_SHUFFLE(3, 0, 2, 1)), out);
    return out;
}

/* Specializations, faster */
template <> inline bool is_parallel<vec2>(const vec2& a, const vec2& b, float tol) noexcept {
    return std::abs(cross(a, b)) <= tol;
}

template <> inline bool is_parallel<vec3>(const vec3& a, const vec3& b, float tol) noexcept {
    return nearly_zero(cross(a, b), tol);
}

/* vec2: {-y, x} — 90° counter-clockwise rotation */
inline vec2 perpendicular(const vec2& v) noexcept {
    const __m128 shuffled = _mm_shuffle_ps(load(v), load(v), _MM_SHUFFLE(3, 2, 0, 1));
    /* Negate lane 0 only: XOR sign bit */
    vec2 out{};
    store(_mm_xor_ps(shuffled, _mm_set_ss(-0.0f)), out);
    return out;
}

/* vec3 find an arbitrary perpendicular vector */
inline vec3 perpendicular(const vec3& v) noexcept {
    /*  Pick the axis least aligned with v, cross with it */
    const __m128 abs_mask = _mm_set1_ps(-0.0f);
    const __m128 av = _mm_andnot_ps(abs_mask, load(v));

    /* extract components to find the smallest */
    const auto ax{_mm_cvtss_f32(av)};
    const auto ay{_mm_cvtss_f32(_mm_movehdup_ps(av))};
    const auto az{_mm_cvtss_f32(_mm_movehl_ps(av, av))};

    /* cross with the axis corresponding to the smallest component
     * this maximizes the resulting vector's length (numerical stability) */
    if (ax <= ay && ax <= az) {
        return cross(v, vec3{1.0f, 0.0f, 0.0f});
    }

    if (ay <= az) {
        return cross(v, vec3{0.0f, 1.0f, 0.0f});
    }

    return cross(v, vec3{0.0f, 0.0f, 1.0f});
}

template <vector Vec> float angle_between(const Vec& a, const Vec& b) noexcept {
    const auto va = load(a);
    const auto vb = load(b);

    /* cos(θ) = dot(a,b) / sqrt(dot(a,a) · dot(b,b)) */
    const auto ab = _mm_dp_ps(va, vb, dp_scalar_mask<Vec>());
    const auto aa = _mm_dp_ps(va, va, dp_scalar_mask<Vec>());
    const auto bb = _mm_dp_ps(vb, vb, dp_scalar_mask<Vec>());

    const auto cos_theta = _mm_div_ss(ab, _mm_sqrt_ss(_mm_mul_ss(aa, bb)));

    /* clamp to [-1,1] — guards against fp overshoot before acos */
    const auto ct{_mm_cvtss_f32(_mm_min_ss(_mm_max_ss(cos_theta, _mm_set_ss(-1.0f)), _mm_set_ss(1.0f)))};

    return std::acos(ct);
}

template <box_type Box> bool is_valid(const Box& b) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    constexpr std::int32_t mask = box_traits<Box>::lane_mask;
    return (_mm_movemask_ps(_mm_cmple_ps(lo, hi)) & mask) == mask;
}

template <box_type Box> box_vec_t<Box> center(const Box& b) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    box_vec_t<Box> out{};
    store(_mm_mul_ps(_mm_add_ps(lo, hi), _mm_set1_ps(0.5f)), out);
    return out;
}

template <box_type Box> box_vec_t<Box> half_extents(const Box& b) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    box_vec_t<Box> out{};
    store(_mm_mul_ps(_mm_sub_ps(hi, lo), _mm_set1_ps(0.5f)), out);
    return out;
}

template <box_type Box> box_vec_t<Box> extents(const Box& b) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);

    box_vec_t<Box> out{};
    store(_mm_sub_ps(hi, lo), out);
    return out;
}

template <box_type Box> Box translate(const Box& b, const box_vec_t<Box>& offset) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 o = load(offset);
    return make_box<Box>(_mm_add_ps(lo, o), _mm_add_ps(hi, o));
}

template <box_type Box> Box scale_about_origin(const Box& b, const box_vec_t<Box>& s) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 sv = load(s);
    return make_box<Box>(_mm_mul_ps(lo, sv), _mm_mul_ps(hi, sv));
}

template <box_type Box> Box scale_uniform(const Box& b, const float s) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 sv = _mm_set1_ps(s);
    return make_box<Box>(_mm_mul_ps(lo, sv), _mm_mul_ps(hi, sv));
}

template <box_type Box> Box dilate(const Box& b, const float amount) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 a = _mm_set1_ps(amount);
    return make_box<Box>(_mm_sub_ps(lo, a), _mm_add_ps(hi, a));
}

template <box_type Box> Box dilate(const Box& b, const box_vec_t<Box>& amount) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 a = load(amount);
    return make_box<Box>(_mm_sub_ps(lo, a), _mm_add_ps(hi, a));
}

template <box_type Box> bool overlaps(const Box& a, const Box& b) noexcept {
    __m128 a_lo, a_hi, b_lo, b_hi;
    load_min_max(a, a_lo, a_hi);
    load_min_max(b, b_lo, b_hi);

    constexpr auto mask{box_traits<Box>::lane_mask};
    const auto sep{_mm_movemask_ps(_mm_cmpgt_ps(a_lo, b_hi)) | _mm_movemask_ps(_mm_cmpgt_ps(b_lo, a_hi))};

    return (sep & mask) == 0;
}

template <box_type Box> bool contains(const Box& outer, const Box& inner) noexcept {
    __m128 o_lo, o_hi, i_lo, i_hi;
    load_min_max(outer, o_lo, o_hi);
    load_min_max(inner, i_lo, i_hi);

    constexpr auto mask{box_traits<Box>::lane_mask};
    const auto ok{_mm_movemask_ps(_mm_cmple_ps(o_lo, i_lo)) & _mm_movemask_ps(_mm_cmple_ps(i_hi, o_hi))};

    return (ok & mask) == mask;
}

template <box_type Box> bool contains_point(const Box& b, const box_vec_t<Box>& p) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 pv = load(p);

    constexpr auto mask = box_traits<Box>::lane_mask;
    const std::int32_t ok = _mm_movemask_ps(_mm_cmple_ps(lo, pv)) & _mm_movemask_ps(_mm_cmple_ps(pv, hi));
    return (ok & mask) == mask;
}

template <box_type Box> Box intersection(const Box& a, const Box& b) noexcept {
    __m128 a_lo, a_hi, b_lo, b_hi;
    load_min_max(a, a_lo, a_hi);
    load_min_max(b, b_lo, b_hi);
    return make_box<Box>(_mm_max_ps(a_lo, b_lo), _mm_min_ps(a_hi, b_hi));
}

template <box_type Box> Box merge(const Box& a, const Box& b) noexcept {
    __m128 a_lo, a_hi, b_lo, b_hi;
    load_min_max(a, a_lo, a_hi);
    load_min_max(b, b_lo, b_hi);
    return make_box<Box>(_mm_min_ps(a_lo, b_lo), _mm_max_ps(a_hi, b_hi));
}

template <box_type Box> Box expand(const Box& b, const box_vec_t<Box>& p) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 pv = load(p);
    return make_box<Box>(_mm_min_ps(lo, pv), _mm_max_ps(hi, pv));
}

template <box_type Box> float distance_squared(const Box& b, const box_vec_t<Box>& p) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 pv = load(p);
    const __m128 nearest = _mm_min_ps(_mm_max_ps(pv, lo), hi);
    const __m128 diff = _mm_sub_ps(pv, nearest);
    constexpr std::int32_t dp = box_traits<Box>::dp_mask;
    return _mm_cvtss_f32(_mm_dp_ps(diff, diff, dp));
}

template <box_type Box> float distance(const Box& b, const box_vec_t<Box>& p) noexcept {
    return std::sqrt(distance_squared(b, p));
}

template <box_type Box> box_vec_t<Box> closest_point(const Box& b, const box_vec_t<Box>& p) noexcept {
    __m128 lo, hi;
    load_min_max(b, lo, hi);
    const __m128 pv = load(p);
    box_vec_t<Box> out{};
    store(_mm_min_ps(_mm_max_ps(pv, lo), hi), out);
    return out;
}

inline box2d box2d_from_center_half(const vec2& c, const vec2& half_ext) noexcept {
    const __m128 cv{load(c)};
    const __m128 hv{load(half_ext)};
    const __m128 cc{_mm_movelh_ps(cv, cv)};
    const __m128 hh{_mm_movelh_ps(hv, hv)};
    /* layout: {min.x, min.y, max.x, max.y} = center ∓ half_ext, packed into one 128-bit store */
    const __m128 signed_h{_mm_xor_ps(hh, _mm_setr_ps(-0.0f, -0.0f, 0.0f, 0.0f))};
    box2d b{};
    _mm_store_ps(&b.min.x, _mm_add_ps(cc, signed_h));
    return b;
}

inline box2d box2d_from_point(const vec2& p) noexcept {
    const __m128 v{load(p)};
    box2d b{};
    _mm_store_ps(&b.min.x, _mm_movelh_ps(v, v));
    return b;
}

inline float area(const box2d& b) noexcept {
    const __m128 v = load(b);
    const __m128 lo = v;
    const __m128 hi = _mm_movehl_ps(v, v);

    /* {w, h, ?, ?} */
    const __m128 diff = _mm_sub_ps(hi, lo);
    return _mm_cvtss_f32(_mm_mul_ss(diff, _mm_movehdup_ps(diff)));
}

inline float perimeter(const box2d& b) noexcept {
    const __m128 v = load(b);
    const __m128 lo = v;
    const __m128 hi = _mm_movehl_ps(v, v);
    /* {w, h, ?, ?} */
    const __m128 diff = _mm_sub_ps(hi, lo);
    const __m128 sum = _mm_add_ss(diff, _mm_movehdup_ps(diff));

    /* 2*(w+h) */
    return _mm_cvtss_f32(_mm_add_ss(sum, sum));
}

inline box3d box3d_from_center_half(const vec3& c, const vec3& half_ext) noexcept {
    const __m128 cv{load(c)};
    const __m128 hv{load(half_ext)};
    box3d b{};
    _mm_store_ps(&b.min.x, _mm_sub_ps(cv, hv));
    _mm_store_ps(&b.max.x, _mm_add_ps(cv, hv));
    return b;
}

inline box3d box3d_from_point(const vec3& p) noexcept {
    const __m128 v{load(p)};
    box3d b{};
    _mm_store_ps(&b.min.x, v);
    _mm_store_ps(&b.max.x, v);
    return b;
}

inline float volume(const box3d& b) noexcept {
    const __m128 lo = _mm_load_ps(&b.min.x);
    const __m128 hi = _mm_load_ps(&b.max.x);

    /* {w, h, d, ?} */
    const __m128 diff = _mm_sub_ps(hi, lo);

    /* === w * h * d === */

    /* {h, h, ?, ?} */
    const __m128 y = _mm_movehdup_ps(diff);

    /* {w*h, ...} */
    const __m128 wh = _mm_mul_ss(diff, y);
    /* {d, ?, ?, ?} */
    const __m128 z = _mm_movehl_ps(diff, diff);

    /* w*h*d */
    return _mm_cvtss_f32(_mm_mul_ss(wh, z));
}

inline float surface_area(const box3d& b) noexcept {
    const __m128 lo = _mm_load_ps(&b.min.x);
    const __m128 hi = _mm_load_ps(&b.max.x);

    /* {w, h, d, ?} */
    const __m128 d = _mm_sub_ps(hi, lo);

    /* SA = 2(wh + wd + hd)
     * we need {w, w, h} · {h, d, d} */

    /* {w, w, h, ?} */
    const __m128 lhs = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 1, 0, 0));
    /* {h, d, d, ?} */
    const __m128 rhs = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 2, 2, 1));
    /* {wh, wd, hd, ?} */
    const __m128 products = _mm_mul_ps(lhs, rhs);

    /* sum 3 lanes */
    const __m128 sum = _mm_dp_ps(products, _mm_set1_ps(1.0f), 0x71);

    /* x2 */
    return _mm_cvtss_f32(_mm_add_ss(sum, sum));
}

/* ============================================================
 * mat4 operations
 * ============================================================ */
inline mat4 mat4_diagonal(const float d) noexcept {
    const __m128 z = _mm_setzero_ps();
    const __m128 dv = _mm_set1_ps(d);
    return store_cols(_mm_blend_ps(z, dv, 0b0001), _mm_blend_ps(z, dv, 0b0010), _mm_blend_ps(z, dv, 0b0100), _mm_blend_ps(z, dv, 0b1000));
}

inline mat4 add(const mat4& a, const mat4& b) noexcept {
    return store_cols(_mm_add_ps(_mm_load_ps(&a.cols[0].x), _mm_load_ps(&b.cols[0].x)), _mm_add_ps(_mm_load_ps(&a.cols[1].x), _mm_load_ps(&b.cols[1].x)),
        _mm_add_ps(_mm_load_ps(&a.cols[2].x), _mm_load_ps(&b.cols[2].x)), _mm_add_ps(_mm_load_ps(&a.cols[3].x), _mm_load_ps(&b.cols[3].x)));
}

inline mat4 sub(const mat4& a, const mat4& b) noexcept {
    return store_cols(_mm_sub_ps(_mm_load_ps(&a.cols[0].x), _mm_load_ps(&b.cols[0].x)), _mm_sub_ps(_mm_load_ps(&a.cols[1].x), _mm_load_ps(&b.cols[1].x)),
        _mm_sub_ps(_mm_load_ps(&a.cols[2].x), _mm_load_ps(&b.cols[2].x)), _mm_sub_ps(_mm_load_ps(&a.cols[3].x), _mm_load_ps(&b.cols[3].x)));
}

inline mat4 scale(const mat4& m, const float s) noexcept {
    const __m128 sv = _mm_set1_ps(s);
    return store_cols(_mm_mul_ps(_mm_load_ps(&m.cols[0].x), sv), _mm_mul_ps(_mm_load_ps(&m.cols[1].x), sv), _mm_mul_ps(_mm_load_ps(&m.cols[2].x), sv),
        _mm_mul_ps(_mm_load_ps(&m.cols[3].x), sv));
}

inline mat4 negate(const mat4& m) noexcept {
    const __m128 sign = _mm_set1_ps(-0.0f);
    return store_cols(_mm_xor_ps(_mm_load_ps(&m.cols[0].x), sign), _mm_xor_ps(_mm_load_ps(&m.cols[1].x), sign), _mm_xor_ps(_mm_load_ps(&m.cols[2].x), sign),
        _mm_xor_ps(_mm_load_ps(&m.cols[3].x), sign));
}

/* ============================================================
 * mat4 * mat4
 * ============================================================ */
inline mat4 operator*(const mat4& a, const mat4& b) noexcept {
    const auto lhs = load_cols(a);
    const auto rhs = load_cols(b);
    return store_cols(linear_combine(lhs, rhs.c[0]), linear_combine(lhs, rhs.c[1]), linear_combine(lhs, rhs.c[2]), linear_combine(lhs, rhs.c[3]));
}

/* ============================================================
 * mat4 × vector transforms
 * ============================================================ */

/** Transform a vec4 by a mat4: M * v. */
inline vec4 operator*(const mat4& m, const vec4& v) noexcept {
    const auto cols{load_cols(m)};
    vec4 out{};
    _mm_store_ps(&out.x, linear_combine(cols, _mm_load_ps(&v.x)));
    return out;
}

/**
 * @brief Transform a vec3 point by a mat4: M * (p, 1).
 *
 * Promotes @c p to homogeneous coordinates with w = 1 before multiplying,
 * which applies the full affine transform (rotation, scale, and translation).
 * Use @c transform_dir for direction vectors (w = 0).
 */
inline vec3 operator*(const mat4& m, const vec3& p) noexcept {
    const auto cols{load_cols(m)};
    auto v{_mm_load_ps(&p.x)};
    v = _mm_blend_ps(v, _mm_set1_ps(1.0f), 0b1000); /* {x, y, z, 1} */
    vec3 out{};
    _mm_store_ps(&out.x, linear_combine(cols, v));
    return out;
}

inline vec4 transform_vec4(const mat4& m, const vec4& v) noexcept {
    const auto cols{load_cols(m)};
    vec4 out{};
    _mm_store_ps(&out.x, linear_combine(cols, _mm_load_ps(&v.x)));
    return out;
}

inline vec3 transform_point(const mat4& m, const vec3& p) noexcept {
    const auto cols{load_cols(m)};
    auto v{_mm_load_ps(&p.x)};
    v = _mm_blend_ps(v, _mm_set1_ps(1.0f), 0b1000); /* {x, y, z, 1} */
    vec3 out{};
    _mm_store_ps(&out.x, linear_combine(cols, v));
    return out;
}

inline vec3 transform_dir(const mat4& m, const vec3& d) noexcept {
    const auto cols{load_cols(m)};
    vec3 out{};
    _mm_store_ps(&out.x, linear_combine(cols, _mm_load_ps(&d.x)));
    return out; /* w lanes are zero in vec3 padding, so col[3] contributes nothing */
}

/* ============================================================
 * mat4 transpose
 * ============================================================ */
inline mat4 transpose(const mat4& m) noexcept {
    const __m128 c0 = _mm_load_ps(&m.cols[0].x);
    const __m128 c1 = _mm_load_ps(&m.cols[1].x);
    const __m128 c2 = _mm_load_ps(&m.cols[2].x);
    const __m128 c3 = _mm_load_ps(&m.cols[3].x);

    const __m128 lo01 = _mm_unpacklo_ps(c0, c1);
    const __m128 hi01 = _mm_unpackhi_ps(c0, c1);
    const __m128 lo23 = _mm_unpacklo_ps(c2, c3);
    const __m128 hi23 = _mm_unpackhi_ps(c2, c3);

    return store_cols(_mm_movelh_ps(lo01, lo23), _mm_movehl_ps(lo23, lo01), _mm_movelh_ps(hi01, hi23), _mm_movehl_ps(hi23, hi01));
}

/* ============================================================
 * mat4 trace, determinant, inverse
 * ============================================================ */
inline float trace(const mat4& m) noexcept {
    return m.cols[0].x + m.cols[1].y + m.cols[2].z + m.cols[3].w;
}

inline float determinant(const mat4& m) noexcept {
    const auto [s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5] = compute_sub_dets(m);
    return s0 * t5 - s1 * t4 + s2 * t3 + s3 * t2 - s4 * t1 + s5 * t0;
}

inline mat4 inverse(const mat4& m) noexcept {
    const auto [s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5] = compute_sub_dets(m);

    const auto det{s0 * t5 - s1 * t4 + s2 * t3 + s3 * t2 - s4 * t1 + s5 * t0};
    const auto inv_det{1.0f / det};

    /* Scaled adjugate: adj(M) / det where adj(M) = cofactor(M)^T. */
    const auto& c{m.cols};

    mat4 r{};

    r.cols[0].x = (c[1].y * t5 - c[1].z * t4 + c[1].w * t3) * inv_det;
    r.cols[0].y = (-c[0].y * t5 + c[0].z * t4 - c[0].w * t3) * inv_det;
    r.cols[0].z = (c[3].y * s5 - c[3].z * s4 + c[3].w * s3) * inv_det;
    r.cols[0].w = (-c[2].y * s5 + c[2].z * s4 - c[2].w * s3) * inv_det;

    r.cols[1].x = (-c[1].x * t5 + c[1].z * t2 - c[1].w * t1) * inv_det;
    r.cols[1].y = (c[0].x * t5 - c[0].z * t2 + c[0].w * t1) * inv_det;
    r.cols[1].z = (-c[3].x * s5 + c[3].z * s2 - c[3].w * s1) * inv_det;
    r.cols[1].w = (c[2].x * s5 - c[2].z * s2 + c[2].w * s1) * inv_det;

    r.cols[2].x = (c[1].x * t4 - c[1].y * t2 + c[1].w * t0) * inv_det;
    r.cols[2].y = (-c[0].x * t4 + c[0].y * t2 - c[0].w * t0) * inv_det;
    r.cols[2].z = (c[3].x * s4 - c[3].y * s2 + c[3].w * s0) * inv_det;
    r.cols[2].w = (-c[2].x * s4 + c[2].y * s2 - c[2].w * s0) * inv_det;

    r.cols[3].x = (-c[1].x * t3 + c[1].y * t1 - c[1].z * t0) * inv_det;
    r.cols[3].y = (c[0].x * t3 - c[0].y * t1 + c[0].z * t0) * inv_det;
    r.cols[3].z = (-c[3].x * s3 + c[3].y * s1 - c[3].z * s0) * inv_det;
    r.cols[3].w = (c[2].x * s3 - c[2].y * s1 + c[2].z * s0) * inv_det;

    return r;
}

/**
 * Inverse of a rigid transform (rotation + translation only, no scale).
 *
 * For M = [R | t; 0 | 1], M⁻¹ = [Rᵀ | -Rᵀt; 0 | 1].
 * Cheaper than the full inverse since we can exploit orthogonality.
 *
 * Input column layout: c0={r00,r10,r20,0}, c1={r01,r11,r21,0},
 *                      c2={r02,r12,r22,0}, c3={tx,ty,tz,1}
 */
inline mat4 inverse_rigid(const mat4& m) noexcept {
    const __m128 c0{_mm_load_ps(&m.cols[0].x)};
    const __m128 c1{_mm_load_ps(&m.cols[1].x)};
    const __m128 c2{_mm_load_ps(&m.cols[2].x)};
    const __m128 c3{_mm_load_ps(&m.cols[3].x)};

    /* Transpose the 3×3 rotation block.
     * r0 = {r00, r01, r02, 0}, r1 = {r10, r11, r12, 0}, r2 = {r20, r21, r22, 0} */
    const __m128 lo01{_mm_unpacklo_ps(c0, c1)};               /* {r00, r01, r10, r11} */
    const __m128 lo2x{_mm_unpacklo_ps(c2, _mm_setzero_ps())}; /* {r02, 0, r12, 0} */
    const __m128 hi01{_mm_unpackhi_ps(c0, c1)};               /* {r20, r21, 0, 0} */

    const __m128 r0{_mm_movelh_ps(lo01, lo2x)};                                  /* {r00, r01, r02, 0} */
    const __m128 r1{_mm_movehl_ps(lo2x, lo01)};                                  /* {r10, r11, r12, 0} */
    const __m128 r2{_mm_movelh_ps(hi01, _mm_unpackhi_ps(c2, _mm_setzero_ps()))}; /* {r20, r21, r22, 0} */

    /* Compute -Rᵀ * t so the new translation column becomes {-r0·t, -r1·t, -r2·t, 1}. */
    const __m128 neg_t{_mm_xor_ps(c3, _mm_set1_ps(-0.0f))};

    const __m128 d0{_mm_dp_ps(r0, neg_t, 0x71)}; /* {-(r0·t), 0, 0, 0} */
    const __m128 d1{_mm_dp_ps(r1, neg_t, 0x72)}; /* {0, -(r1·t), 0, 0} */
    const __m128 d2{_mm_dp_ps(r2, neg_t, 0x74)}; /* {0, 0, -(r2·t), 0} */

    /* Combine: {-r0·t, -r1·t, -r2·t, 1} */
    const __m128 new_t = _mm_blend_ps(_mm_or_ps(_mm_or_ps(d0, d1), d2), _mm_set1_ps(1.0f), 0b1000);

    return store_cols(r0, r1, r2, new_t);
}

/**
 * Inverse of an affine transform with uniform scale and no shear: M = sR + t.
 *
 *    M⁻¹ = [ (1/s)Rᵀ  | -(1/s)Rᵀt ]
 *          [    0     |      1    ]
 *
 * s² = dot(col0, col0). Divide the transposed rotation by s² to get (1/s)Rᵀ.
 */
inline mat4 inverse_affine_uniform(const mat4& m) noexcept {
    const __m128 c0{_mm_load_ps(&m.cols[0].x)};
    const __m128 c1{_mm_load_ps(&m.cols[1].x)};
    const __m128 c2{_mm_load_ps(&m.cols[2].x)};
    const __m128 c3{_mm_load_ps(&m.cols[3].x)};

    const __m128 s2{_mm_dp_ps(c0, c0, 0x7F)}; /* broadcast s² */
    const __m128 inv_s2{_mm_div_ps(_mm_set1_ps(1.0f), s2)};

    /* Transpose 3×3 rotation block and scale by 1/s² */
    const __m128 lo01{_mm_unpacklo_ps(c0, c1)};
    const __m128 lo2x{_mm_unpacklo_ps(c2, _mm_setzero_ps())};
    const __m128 hi01{_mm_unpackhi_ps(c0, c1)};

    auto r0{_mm_mul_ps(_mm_movelh_ps(lo01, lo2x), inv_s2)};
    auto r1{_mm_mul_ps(_mm_movehl_ps(lo2x, lo01), inv_s2)};
    auto r2{_mm_mul_ps(_mm_movelh_ps(hi01, _mm_unpackhi_ps(c2, _mm_setzero_ps())), inv_s2)};

    /* -Rᵀ/s² · t */
    const __m128 neg_t{_mm_xor_ps(c3, _mm_set1_ps(-0.0f))};
    const __m128 d0{_mm_dp_ps(r0, neg_t, 0x71)};
    const __m128 d1{_mm_dp_ps(r1, neg_t, 0x72)};
    const __m128 d2{_mm_dp_ps(r2, neg_t, 0x74)};
    const __m128 new_t = _mm_blend_ps(_mm_or_ps(_mm_or_ps(d0, d1), d2), _mm_set1_ps(1.0f), 0b1000);

    return store_cols(r0, r1, r2, new_t);
}

/* ============================================================
 * mat2 operations
 * ============================================================ */

/* Loads the full mat2 as a single __m128: {col0.x, col0.y, col1.x, col1.y} = {a, c, b, d}. */
inline __m128 load(const mat2& m) noexcept {
    return _mm_load_ps(&m.cols[0].x);
}

inline mat2 add(const mat2& a, const mat2& b) noexcept {
    mat2 r{};
    _mm_store_ps(&r.cols[0].x, _mm_add_ps(load(a), load(b)));
    return r;
}

inline mat2 sub(const mat2& a, const mat2& b) noexcept {
    mat2 r{};
    _mm_store_ps(&r.cols[0].x, _mm_sub_ps(load(a), load(b)));
    return r;
}

inline mat2 scale(const mat2& m, const float s) noexcept {
    mat2 r{};
    _mm_store_ps(&r.cols[0].x, _mm_mul_ps(load(m), _mm_set1_ps(s)));
    return r;
}

inline mat2 negate(const mat2& m) noexcept {
    mat2 r{};
    _mm_store_ps(&r.cols[0].x, _mm_xor_ps(load(m), _mm_set1_ps(-0.0f)));
    return r;
}

/* mat2 × mat2
 * | a  b |   | e  f |   | ae+bg  af+bh |
 * | c  d | × | g  h | = | ce+dg  cf+dh |
 *
 * Memory layout: lhs = {a, c, b, d}, rhs = {e, g, f, h}
 */
inline mat2 operator*(const mat2& lhs, const mat2& rhs) noexcept {
    const __m128 l{load(lhs)}; /* {a, c, b, d} */
    const __m128 r{load(rhs)}; /* {e, g, f, h} */

    /* {a, c, a, c} * {e, e, f, f} + {b, d, b, d} * {g, g, h, h} */
    const __m128 ac_ac{_mm_movelh_ps(l, l)};
    const __m128 bd_bd{_mm_movehl_ps(l, l)};
    const __m128 ee_ff{_mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 0, 0))};
    const __m128 gg_hh{_mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 3, 1, 1))};

    mat2 result{};
    _mm_store_ps(&result.cols[0].x, _mm_fmadd_ps(ac_ac, ee_ff, _mm_mul_ps(bd_bd, gg_hh)));
    return result;
}

/* mat2 × vec2: result = {a,c}*x + {b,d}*y */
inline vec2 transform_vec2(const mat2& m, const vec2& v) noexcept {
    const __m128 mv{load(m)}; /* {a, c, b, d} */
    const __m128 vv{load(v)}; /* {x, y, 0, 0} */

    const __m128 ac{mv};
    const __m128 bd{_mm_movehl_ps(mv, mv)};
    const __m128 xx{_mm_shuffle_ps(vv, vv, _MM_SHUFFLE(0, 0, 0, 0))};
    const __m128 yy{_mm_shuffle_ps(vv, vv, _MM_SHUFFLE(1, 1, 1, 1))};

    vec2 out{};
    store(_mm_fmadd_ps(ac, xx, _mm_mul_ps(bd, yy)), out);
    return out;
}

/* Transpose mat2: {a, c, b, d} → {a, b, c, d}. */
inline mat2 transpose(const mat2& m) noexcept {
    mat2 r{};
    _mm_store_ps(&r.cols[0].x, _mm_shuffle_ps(load(m), load(m), _MM_SHUFFLE(3, 1, 2, 0)));
    return r;
}

/* det(mat2) = ad - bc. Memory: {a, c, b, d}. */
inline float determinant(const mat2& m) noexcept {
    const __m128 v{load(m)};
    /* {a, c, b, d} * {d, b, c, a} → {ad, cb, bc, da}; lane0 - lane2 = ad - bc */
    const __m128 swz{_mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3))};
    const __m128 prod{_mm_mul_ps(v, swz)};
    return _mm_cvtss_f32(_mm_sub_ss(prod, _mm_movehl_ps(prod, prod)));
}

inline mat2 inverse(const mat2& m) noexcept {
    const __m128 v{load(m)}; /* {a, c, b, d} */

    const __m128 d_lane{_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3))};
    const __m128 b_lane{_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))};
    const __m128 c_lane{_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))};

    const __m128 ad{_mm_mul_ss(v, d_lane)};
    const __m128 bc{_mm_mul_ss(b_lane, c_lane)};
    const __m128 det{_mm_sub_ss(ad, bc)};

    const __m128 inv_det{_mm_div_ss(_mm_set_ss(1.0f), det)};
    const __m128 inv_det4{_mm_shuffle_ps(inv_det, inv_det, 0)}; /* broadcast */

    /* Adjugate: {d, -c, -b, a} */
    const __m128 adj{_mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 2, 1, 3))};
    const __m128 sign{_mm_setr_ps(0.0f, -0.0f, -0.0f, 0.0f)};

    mat2 r{};
    _mm_store_ps(&r.cols[0].x, _mm_mul_ps(_mm_xor_ps(adj, sign), inv_det4));
    return r;
}

inline float trace(const mat2& m) noexcept {
    return m.cols[0].x + m.cols[1].y;
}

/* ============================================================
 * mat3 <-> mat4 conversions
 * ============================================================ */

inline mat3 to_mat3(const mat4& m) noexcept {
    mat3 r{};
    /* vec3 store writes to the pad lane — that's fine, it's unused */
    _mm_store_ps(&r.cols[0].x, _mm_load_ps(&m.cols[0].x));
    _mm_store_ps(&r.cols[1].x, _mm_load_ps(&m.cols[1].x));
    _mm_store_ps(&r.cols[2].x, _mm_load_ps(&m.cols[2].x));
    return r;
}

inline mat4 to_mat4(const mat3& m) noexcept {
    return store_cols(_mm_load_ps(&m.cols[0].x), /* w=0 from vec3 padding */
        _mm_load_ps(&m.cols[1].x), _mm_load_ps(&m.cols[2].x), _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f));
}

/* Embed mat3 + translation into mat4. */
inline mat4 to_mat4(const mat3& m, const vec3& translation) noexcept {
    auto t{_mm_load_ps(&translation.x)};
    t = _mm_blend_ps(t, _mm_set1_ps(1.0f), 0b1000); /* {tx, ty, tz, 1} */
    return store_cols(_mm_load_ps(&m.cols[0].x), _mm_load_ps(&m.cols[1].x), _mm_load_ps(&m.cols[2].x), t);
}

/* Extract translation from mat4 */
inline vec3 get_translation(const mat4& m) noexcept {
    vec3 out{};
    store(_mm_load_ps(&m.cols[3].x), out);
    return out;
}

/* Set translation column in mat4, preserving w. */
inline mat4 set_translation(const mat4& m, const vec3& t) noexcept {
    auto r{m};
    auto tv{_mm_load_ps(&t.x)};
    tv = _mm_blend_ps(tv, _mm_load_ps(&m.cols[3].x), 0b1000); /* preserve w */
    _mm_store_ps(&r.cols[3].x, tv);
    return r;
}

/* ============================================================
 * Quaternion operations
 * ============================================================ */

/**
 * Construct a unit quaternion that rotates by `angle` (radians) around `axis`.
 * The axis need not be unit length — it is normalized internally. Identity is returned for a zero axis.
 *
 *   q = (sin(θ/2) * normalize(axis), cos(θ/2))
 */
inline quat quat_from_axis_angle(const vec3& axis, const float angle) noexcept {
    const auto half{angle * 0.5f};
    const auto n{normalized_safe(axis)};
    const auto s{std::sin(half)};
    return {n.x * s, n.y * s, n.z * s, std::cos(half)};
}

inline float dot(const quat& a, const quat& b) noexcept {
    return _mm_cvtss_f32(_mm_dp_ps(load(a), load(b), 0xFF));
}

inline float norm_sq(const quat& q) noexcept {
    return dot(q, q);
}

/* more expensive and not needed for unit quats of unit length (which should always be the case). */
inline float norm(const quat& q) noexcept {
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(load(q), load(q), 0xFF)));
}

/* Negate xyz, keep w */
inline quat conjugate(const quat& q) noexcept {
    const __m128 sign = _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f);
    return store_quat(_mm_xor_ps(load(q), sign));
}

inline quat negate(const quat& q) noexcept {
    return store_quat(_mm_xor_ps(load(q), _mm_set1_ps(-0.0f)));
}

/* Quaternions should always be unit length, but floating-point drift accumulates
 * over many chained multiplications. Sanitize after deserialization:
 *   if (std::abs(norm_sq(q) - 1.0f) > tolerance) { q = normalized(q); }
 */
inline quat normalized(const quat& q) noexcept {
    const __m128 v = load(q);
    const __m128 len = _mm_sqrt_ps(_mm_dp_ps(v, v, 0xFF));
    return store_quat(_mm_div_ps(v, len));
}

/* For unit quaternions inverse == conjugate; for general quaternions: conjugate / norm². */
inline quat inverse(const quat& q) noexcept {
    const __m128 v = load(q);
    const __m128 sign = _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f);
    const __m128 conj = _mm_xor_ps(v, sign);
    const __m128 n2 = _mm_dp_ps(v, v, 0xFF);
    return store_quat(_mm_div_ps(conj, n2));
}

/* just the conjugate, much cheaper, assumes unit quaternion */
inline quat inverse_unit(const quat& q) noexcept {
    return conjugate(q);
}

/**
 * Quaternion multiplication (Hamilton product): q1 * q2 composes rotations.
 *
 * result.x =  w1*x2 + x1*w2 + y1*z2 - z1*y2
 * result.y =  w1*y2 - x1*z2 + y1*w2 + z1*x2
 * result.z =  w1*z2 + x1*y2 - y1*x2 + z1*w2
 * result.w =  w1*w2 - x1*x2 - y1*y2 - z1*z2
 *
 * Implemented as four FMA rows, each broadcasting one component of q1 against
 * a sign-flipped permutation of q2.
 */

inline quat operator*(const quat& q1, const quat& q2) noexcept {
    const __m128 a = load(q1);
    const __m128 b = load(q2);

    /* broadcast each component of q1 */
    const __m128 a_xxxx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 a_yyyy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 a_zzzz = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 a_wwww = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3));

    /* each row: q1.component * (shuffled q2 with sign flips)
     * r = w1 * { x2,  y2,  z2,  w2}    →  a_wwww * b
     *   + x1 * { w2, -z2,  y2, -x2}
     *   + y1 * { z2,  w2, -x2, -y2}
     *   + z1 * {-y2,  x2,  w2, -z2}
     */

    const __m128 b_wzyx{_mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3))}; /* {w,z,y,x} */
    const __m128 b_zwxy{_mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 0, 3, 2))}; /* {z,w,x,y} */
    const __m128 b_yxwz{_mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1))}; /* {y,x,w,z} */

    const __m128 sign_x{_mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f)};
    const __m128 sign_y{_mm_setr_ps(1.0f, 1.0f, -1.0f, -1.0f)};
    const __m128 sign_z{_mm_setr_ps(-1.0f, 1.0f, 1.0f, -1.0f)};

    auto r{_mm_mul_ps(a_wwww, b)};
    r = _mm_fmadd_ps(a_xxxx, _mm_mul_ps(b_wzyx, sign_x), r);
    r = _mm_fmadd_ps(a_yyyy, _mm_mul_ps(b_zwxy, sign_y), r);
    r = _mm_fmadd_ps(a_zzzz, _mm_mul_ps(b_yxwz, sign_z), r);

    return store_quat(r);
}

/**
 * Rotate a vector by a unit quaternion.
 *
 * Full sandwich product q * v * q⁻¹ would require two quat multiplies.
 * Equivalent but cheaper form (Rodrigues):
 *   t  = 2 * cross(q.xyz, v)
 *   v' = v + q.w * t + cross(q.xyz, t)
 */

inline vec3 rotate(const quat& q, const vec3& v) noexcept {
    const __m128 qv{load(q)};
    const __m128 vv{_mm_load_ps(&v.x)};
    const __m128 q_xyz{qv}; /* w lane is ignored in cross products */
    const __m128 qw{_mm_shuffle_ps(qv, qv, _MM_SHUFFLE(3, 3, 3, 3))};

    /* t = 2 * cross(q.xyz, v) */
    const __m128 q_yzx = _mm_shuffle_ps(q_xyz, q_xyz, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 q_zxy = _mm_shuffle_ps(q_xyz, q_xyz, _MM_SHUFFLE(3, 1, 0, 2));
    const __m128 v_yzx = _mm_shuffle_ps(vv, vv, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 v_zxy = _mm_shuffle_ps(vv, vv, _MM_SHUFFLE(3, 1, 0, 2));

    /* 2 * cross */
    const __m128 cross1 = _mm_fmsub_ps(q_yzx, v_zxy, _mm_mul_ps(q_zxy, v_yzx));
    const __m128 t = _mm_add_ps(cross1, cross1);

    /* cross(q.xyz, t) */
    const __m128 t_yzx = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 t_zxy = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 1, 0, 2));

    const __m128 cross2 = _mm_fmsub_ps(q_yzx, t_zxy, _mm_mul_ps(q_zxy, t_yzx));

    /* v' = v + qw * t + cross(q.xyz, t) */
    const __m128 result{_mm_add_ps(_mm_fmadd_ps(qw, t, vv), cross2)};

    vec3 out{};
    _mm_store_ps(&out.x, result);
    return out;
}

/* ============================================================
 * Quaternion to Matrix Conversions
 * ============================================================ */

/* Quaternion to mat3 (pure rotation, no translation) */
inline mat3 to_mat3(const quat& q) noexcept {
    const auto x2{2.0f * q.x}, y2{2.0f * q.y}, z2{2.0f * q.z};
    const auto xx2{x2 * q.x}, xy2{x2 * q.y}, xz2{x2 * q.z};
    const auto yy2{y2 * q.y}, yz2{y2 * q.z}, zz2{z2 * q.z};
    const auto wx2{x2 * q.w}, wy2{y2 * q.w}, wz2{z2 * q.w};

    mat3 result{};
    _mm_store_ps(&result.cols[0].x, _mm_setr_ps(1.0f - yy2 - zz2, xy2 + wz2, xz2 - wy2, 0.0f));
    _mm_store_ps(&result.cols[1].x, _mm_setr_ps(xy2 - wz2, 1.0f - xx2 - zz2, yz2 + wx2, 0.0f));
    _mm_store_ps(&result.cols[2].x, _mm_setr_ps(xz2 + wy2, yz2 - wx2, 1.0f - xx2 - yy2, 0.0f));

    return result;
}

/* Quaternion to mat4 (rotation only, translation = 0) */
inline mat4 to_mat4(const quat& q) noexcept {
    return to_mat4(to_mat3(q));
}

/* Quaternion to mat4 with translation */
inline mat4 to_mat4(const quat& q, const vec3& translation) noexcept {
    return to_mat4(to_mat3(q), translation);
}

inline bool nearly_equal(const quat& a, const quat& b, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    /* quaternions q and -q represent the same rotation,c heck both orientations */
    const __m128 av = load(a);
    const __m128 bv = load(b);
    const __m128 neg_bv = _mm_xor_ps(bv, _mm_set1_ps(-0.0f));

    /* |a - b| */
    const __m128 diff1 = _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_sub_ps(av, bv));
    /* |a + b| */
    const __m128 diff2 = _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_sub_ps(av, neg_bv));
    /* min(|a-b|, |a+b|) */
    const __m128 diff = _mm_min_ps(diff1, diff2);

    /* Combined tolerance: abs_tol + rel_tol * max(|a|, |b|) */
    const __m128 abs_a = _mm_andnot_ps(_mm_set1_ps(-0.0f), av);
    const __m128 abs_b = _mm_andnot_ps(_mm_set1_ps(-0.0f), bv);
    const __m128 tol = _mm_add_ps(_mm_set1_ps(abs_tol), _mm_mul_ps(_mm_set1_ps(rel_tol), _mm_max_ps(abs_a, abs_b)));

    const __m128 cmp = _mm_cmple_ps(diff, tol);
    return _mm_movemask_ps(cmp) == 0xF;
}

/* Returns the shortest rotation angle (radians) to go from orientation a to b:
 *   angle = 2 * acos(|dot(a, b)|) */
inline float angle_between(const quat& a, const quat& b) noexcept {
    /* note that we clamp for acos safety */
    return 2.0f * std::acos(std::min(std::abs(dot(a, b)), 1.0f));
}

/* Returns true when `direction` is within `tolerance` of a cardinal axis. */
inline bool is_axis_aligned(const vec3& direction, const float tolerance = fp32_rel_tol) noexcept {
    const auto snapped{snap_to_axis(direction)};
    const auto dir_norm{normalized(direction)};
    /* |dot(normalized, snapped)| ≈ 1 if axis-aligned */
    const auto d{std::abs(_mm_cvtss_f32(_mm_dp_ps(load(dir_norm), load(snapped), dp_scalar_mask<vec3>())))};
    return d >= (1.0f - tolerance);
}

/**
 * Closest point on line segment [seg_start, seg_end] to point.
 *
 *   t = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0, 1)
 *   result = a + t * (b - a)
 *
 * Uses load/store helpers to handle vec2 (8-byte aligned) correctly.
 */
template <spatial_vector Vec> Vec closest_point_on_segment(const Vec& point, const Vec& seg_start, const Vec& seg_end) noexcept {
    constexpr auto mask{dp_broadcast_mask<Vec>()};

    const __m128 p{load(point)};
    const __m128 a{load(seg_start)};
    const __m128 b{load(seg_end)};

    const __m128 ab{_mm_sub_ps(b, a)};
    const __m128 ap{_mm_sub_ps(p, a)};

    const __m128 ab_dot_ap{_mm_dp_ps(ab, ap, mask)};
    const __m128 ab_dot_ab{_mm_dp_ps(ab, ab, mask)};

    /* t = dot(ap, ab) / dot(ab, ab), clamped to [0, 1] */
    auto t{_mm_div_ps(ab_dot_ap, ab_dot_ab)};
    t = _mm_max_ps(t, _mm_setzero_ps());
    t = _mm_min_ps(t, _mm_set1_ps(1.0f));

    Vec out{};
    store(_mm_fmadd_ps(t, ab, a), out);
    return out;
}

/* Squared distance from point to line segment */
template <spatial_vector Vec> float distance_point_to_segment_sq(const Vec& point, const Vec& seg_start, const Vec& seg_end) noexcept {
    const Vec closest = closest_point_on_segment(point, seg_start, seg_end);
    return distance_squared(point, closest);
}

/* Distance from point to line segment */
template <spatial_vector Vec> float distance_point_to_segment(const Vec& point, const Vec& seg_start, const Vec& seg_end) noexcept {
    const Vec closest = closest_point_on_segment(point, seg_start, seg_end);
    return distance(point, closest);
}

/* ============================================================
 * Plane operations
 * ============================================================ */

/* Construct plane from point on plane + unit normal. */
inline plane plane_from_point_normal(const vec3& point, const vec3& normal) noexcept {
    return {normal, dot(normal, point)};
}

/* Construct plane from three CCW points. */
inline plane plane_from_points(const vec3& a, const vec3& b, const vec3& c) noexcept {
    const auto n{normalized(cross(b - a, c - a))};
    return {n, dot(n, a)};
}

/* Signed distance from point to plane (positive = same side as normal) */
inline float signed_distance_to_plane(const vec3& point, const plane& pl) noexcept {
    return dot(pl.normal, point) - pl.d;
}

/* Absolute distance from point to plane */
inline float distance_to_plane(const vec3& point, const plane& pl) noexcept {
    return std::abs(signed_distance_to_plane(point, pl));
}

/* Project point onto plane (perpendicular projection) */
inline vec3 project_onto_plane(const vec3& point, const plane& pl) noexcept {
    const float dist = signed_distance_to_plane(point, pl);
    const __m128 p = _mm_load_ps(&point.x);
    const __m128 n = _mm_load_ps(&pl.normal.x);
    const __m128 result = _mm_fnmadd_ps(_mm_set1_ps(dist), n, p); /* p - dist * n */

    vec3 out{};
    _mm_store_ps(&out.x, result);
    return out;
}

/* Project a 3D point onto the 2D coordinate system defined by a plane basis.
 * result = { dot(p - origin, u), dot(p - origin, v) }
 */
inline vec2 project_to_2d(const vec3& point, const plane_basis& basis) noexcept {
    const __m128 p = _mm_sub_ps(_mm_load_ps(&point.x), _mm_load_ps(&basis.origin.x));
    const __m128 u = _mm_load_ps(&basis.u.x);
    const __m128 v = _mm_load_ps(&basis.v.x);

    const float pu = _mm_cvtss_f32(_mm_dp_ps(p, u, 0x71));
    const float pv = _mm_cvtss_f32(_mm_dp_ps(p, v, 0x71));

    return {pu, pv};
}

/* Unproject a 2D point back to 3D on the plane.
 * result = origin + point2d.x * u + point2d.y * v
 */
inline vec3 unproject_to_3d(const vec2& point2d, const plane_basis& basis) noexcept {
    const __m128 origin = _mm_load_ps(&basis.origin.x);
    const __m128 u = _mm_load_ps(&basis.u.x);
    const __m128 v = _mm_load_ps(&basis.v.x);
    const __m128 px = _mm_set1_ps(point2d.x);
    const __m128 py = _mm_set1_ps(point2d.y);

    const __m128 result = _mm_fmadd_ps(py, v, _mm_fmadd_ps(px, u, origin));

    vec3 out{};
    _mm_store_ps(&out.x, result);
    return out;
}

/* Batch project: convert an array of 3D points to 2D.
 * This is the hot path for wall carving — worth optimizing.
 */
inline void project_to_2d_batch(const vec3* points_3d, vec2* points_2d, const std::int32_t count, const plane_basis& basis) noexcept {
    const __m128 origin = _mm_load_ps(&basis.origin.x);
    const __m128 u = _mm_load_ps(&basis.u.x);
    const __m128 v = _mm_load_ps(&basis.v.x);

    for (int i{}; i < count; ++i) {
        const __m128 p = _mm_sub_ps(_mm_load_ps(&points_3d[i].x), origin);
        const float pu = _mm_cvtss_f32(_mm_dp_ps(p, u, 0x71));
        const float pv = _mm_cvtss_f32(_mm_dp_ps(p, v, 0x71));
        points_2d[i] = {pu, pv};
    }
}

/* Batch unproject: convert an array of 2D points back to 3D */
inline void unproject_to_3d_batch(const vec2* points_2d, vec3* points_3d, const std::int32_t count, const plane_basis& basis) noexcept {
    const __m128 origin = _mm_load_ps(&basis.origin.x);
    const __m128 u = _mm_load_ps(&basis.u.x);
    const __m128 v = _mm_load_ps(&basis.v.x);

    for (int i{}; i < count; ++i) {
        const __m128 px = _mm_set1_ps(points_2d[i].x);
        const __m128 py = _mm_set1_ps(points_2d[i].y);
        const __m128 result = _mm_fmadd_ps(py, v, _mm_fmadd_ps(px, u, origin));
        _mm_store_ps(&points_3d[i].x, result);
    }
}

/* ============================================================
 * Intersection Tests
 * ============================================================ */

/* 2D segment-segment intersection.
 *
 * Segments: P = a0 + t*(a1 - a0), Q = b0 + u*(b1 - b0)
 * Solves:   a0 + t*(a1-a0) = b0 + u*(b1-b0)
 *
 * Uses the cross-product form:
 *   t = cross(b0 - a0, db) / cross(da, db)
 *   u = cross(b0 - a0, da) / cross(da, db)
 * where cross for vec2 returns the scalar z-component.
 */
inline segment_intersection_2d intersect_segments_2d(const vec2& a0, const vec2& a1, const vec2& b0, const vec2& b1, const float epsilon = 1e-7f) noexcept {
    const vec2 da = a1 - a0;
    const vec2 db = b1 - b0;
    const vec2 d_origin = b0 - a0;

    const float denom = cross(da, db);

    /* Parallel or degenerate — no single intersection point */
    if (std::abs(denom) < epsilon) {
        return {};
    }

    const float inv_denom = 1.0f / denom;
    const float t = cross(d_origin, db) * inv_denom;
    const float u = cross(d_origin, da) * inv_denom;

    if (t < 0.0f || t > 1.0f || u < 0.0f || u > 1.0f) {
        return {};
    }

    vec2 point{};
    store(_mm_fmadd_ps(_mm_set1_ps(t), load(da), load(a0)), point);

    return {point, t, u, true};
}

/* Ray-AABB intersection using the slab method (2D).
 *
 * Ray: origin + t * direction, t ∈ [0, ∞)
 * Computes t_entry and t_exit for all slab pairs simultaneously.
 *
 * Uses reciprocal direction to turn division into multiplication.
 * Handles infinities correctly when direction component is 0
 * (IEEE 754: 1/0 = ±inf, then min/max propagate correctly).
 */
inline ray_box_hit intersect_ray_box(const vec2& origin, const vec2& direction, const box2d& box) noexcept {
    /* For vec2 we only care about lanes 0 and 1.
     * Use load() for origin/direction (alignas(8)), load_min_max for the box. */
    const __m128 orig{load(origin)};
    const __m128 inv_dir{_mm_div_ps(_mm_set1_ps(1.0f), load(direction))};
    __m128 bmin, bmax;
    load_min_max(box, bmin, bmax);

    /* t values for near and far slabs */
    const __m128 t1 = _mm_mul_ps(_mm_sub_ps(bmin, orig), inv_dir);
    const __m128 t2 = _mm_mul_ps(_mm_sub_ps(bmax, orig), inv_dir);

    const __m128 t_near = _mm_min_ps(t1, t2);
    const __m128 t_far = _mm_max_ps(t1, t2);

    /* t_entry = max of all near values, t_exit = min of all far values */
    /* For vec2: max(t_near.x, t_near.y) and min(t_far.x, t_far.y) */
    const __m128 t_near_yx = _mm_shuffle_ps(t_near, t_near, _MM_SHUFFLE(0, 0, 0, 1));
    const __m128 t_far_yx = _mm_shuffle_ps(t_far, t_far, _MM_SHUFFLE(0, 0, 0, 1));

    const float t_entry = _mm_cvtss_f32(_mm_max_ps(t_near, t_near_yx));
    const float t_exit = _mm_cvtss_f32(_mm_min_ps(t_far, t_far_yx));

    /* Hit if t_entry <= t_exit and t_exit >= 0 */
    if (t_entry > t_exit || t_exit < 0.0f) {
        return {};
    }
    return {t_entry, t_exit, true};
}

/* Ray-AABB intersection (3D slab method) */
inline ray_box_hit intersect_ray_box(const vec3& origin, const vec3& direction, const box3d& box) noexcept {
    const __m128 orig = _mm_load_ps(&origin.x);
    const __m128 inv_dir = _mm_div_ps(_mm_set1_ps(1.0f), _mm_load_ps(&direction.x));
    const __m128 bmin = _mm_load_ps(&box.min.x);
    const __m128 bmax = _mm_load_ps(&box.max.x);

    const __m128 t1 = _mm_mul_ps(_mm_sub_ps(bmin, orig), inv_dir);
    const __m128 t2 = _mm_mul_ps(_mm_sub_ps(bmax, orig), inv_dir);

    const __m128 t_near = _mm_min_ps(t1, t2);
    const __m128 t_far = _mm_max_ps(t1, t2);

    /* Horizontal max of t_near lanes 0,1,2 and horizontal min of t_far lanes 0,1,2 */
    /* Shuffle: {y, z, x, _} */
    const __m128 near_yzx = _mm_shuffle_ps(t_near, t_near, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 near_zxy = _mm_shuffle_ps(t_near, t_near, _MM_SHUFFLE(3, 1, 0, 2));
    const __m128 far_yzx = _mm_shuffle_ps(t_far, t_far, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 far_zxy = _mm_shuffle_ps(t_far, t_far, _MM_SHUFFLE(3, 1, 0, 2));

    const float t_entry = _mm_cvtss_f32(_mm_max_ps(_mm_max_ps(t_near, near_yzx), near_zxy));
    const float t_exit = _mm_cvtss_f32(_mm_min_ps(_mm_min_ps(t_far, far_yzx), far_zxy));

    if (t_entry > t_exit || t_exit < 0.0f) {
        return {};
    }
    return {t_entry, t_exit, true};
}

/* Segment-AABB intersection: does the segment [a, b] intersect the box?
 * Implemented as a ray cast with t clamped to [0, 1].
 */
template <typename Vec, typename Box>
    requires((std::is_same_v<Vec, vec2> && std::is_same_v<Box, box2d>) || (std::is_same_v<Vec, vec3> && std::is_same_v<Box, box3d>))
bool segment_intersects_box(const Vec& a, const Vec& b, const Box& box) noexcept {
    const Vec direction = b - a;
    const ray_box_hit hit = intersect_ray_box(a, direction, box);
    /* Ray hit is parameterized by segment length, so t ∈ [0, 1] means within segment */
    return hit.hit && hit.t_entry <= 1.0f && hit.t_exit >= 0.0f;
}

/* Point-in-polygon (2D) using winding number.
 *
 * Robust for concave polygons, self-intersecting polygons, and points on edges.
 * Vertices are expected in order (CW or CCW), forming a closed polygon.
 *
 * Uses the crossing-number variant of winding number:
 * For each edge, check if a horizontal ray from p crosses the edge,
 * and whether the crossing is upward or downward.
 *
 * Returns the winding number. Non-zero means inside (standard rule).
 */
inline int winding_number(const vec2& point, const vec2* vertices, const std::int32_t count) noexcept {
    std::int32_t winding{};

    for (int i{}; i < count; ++i) {
        const vec2& v0 = vertices[i];
        const vec2& v1 = vertices[(i + 1) % count];

        if (v0.y <= point.y) {
            if (v1.y > point.y) {
                /* Upward crossing — check if point is left of edge */
                const float side = cross(v1 - v0, point - v0);
                if (side > 0.0f) {
                    ++winding;
                }
            }
        } else {
            if (v1.y <= point.y) {
                /* Downward crossing — check if point is right of edge */
                const float side = cross(v1 - v0, point - v0);
                if (side < 0.0f) {
                    --winding;
                }
            }
        }
    }

    return winding;
}

inline bool point_in_polygon(const vec2& point, const vec2* vertices, const std::int32_t count) noexcept {
    return winding_number(point, vertices, count) != 0;
}

inline segment_polygon_hit intersect_segment_polygon(const vec2& a, const vec2& b, const vec2* vertices, const std::int32_t count) noexcept {
    segment_polygon_hit nearest{};
    float nearest_t{std::numeric_limits<float>::max()};

    for (int i{}; i < count; ++i) {
        const vec2& v0 = vertices[i];
        const vec2& v1 = vertices[(i + 1) % count];

        const segment_intersection_2d result = intersect_segments_2d(a, b, v0, v1);
        if (result.intersects && result.t < nearest_t) {
            nearest_t = result.t;
            nearest = {result.point, result.t, i, true};
        }
    }

    return nearest;
}

inline ivec2 grid_cell(const vec2& point, const vec2& grid_origin, const float cell_size) noexcept {
    const __m128 p{_mm_sub_ps(load(point), load(grid_origin))};
    const __m128 inv_size = _mm_set1_ps(1.0f / cell_size);
    const __m128 cell_f = _mm_floor_ps(_mm_mul_ps(p, inv_size));

    /* Convert to int — _mm_cvttps_epi32 truncates, but we already floored */
    const __m128i cell_i = _mm_cvtps_epi32(cell_f); /* rounds to nearest, but value is already integer after floor */

    alignas(16) int cells[4]{};
    _mm_store_si128(reinterpret_cast<__m128i*>(cells), cell_i);

    return {cells[0], cells[1]};
}

inline ivec3 grid_cell(const vec3& point, const vec3& grid_origin, const float cell_size) noexcept {
    const __m128 p = _mm_sub_ps(_mm_load_ps(&point.x), _mm_load_ps(&grid_origin.x));
    const __m128 inv_size = _mm_set1_ps(1.0f / cell_size);
    const __m128 cell_f = _mm_floor_ps(_mm_mul_ps(p, inv_size));

    const __m128i cell_i = _mm_cvtps_epi32(cell_f);

    alignas(16) int cells[4]{};
    _mm_store_si128(reinterpret_cast<__m128i*>(cells), cell_i);

    return {cells[0], cells[1], cells[2]};
}

inline grid_cell_range_2d grid_cell_range(const box2d& box, const vec2& grid_origin, const float cell_size) noexcept {
    return {grid_cell(box.min, grid_origin, cell_size), grid_cell(box.max, grid_origin, cell_size)};
}

inline grid_cell_range_3d grid_cell_range(const box3d& box, const vec3& grid_origin, const float cell_size) noexcept {
    return {grid_cell(box.min, grid_origin, cell_size), grid_cell(box.max, grid_origin, cell_size)};
}

/* Convert transform to mat4 for point transformation.
 *
 * The resulting matrix applies: M * p = R * diag(scale) * p + translation
 * This is the matrix to use to transform local-space geometry to world-space.
 */
inline mat4 to_mat4(const transform& tf) noexcept {
    /* Start with rotation matrix */
    mat4 result = to_mat4(tf.rotation);

    /* Scale each column (rotation axis) by the corresponding scale component */
    const __m128 sx = _mm_set1_ps(tf.scale.x);
    const __m128 sy = _mm_set1_ps(tf.scale.y);
    const __m128 sz = _mm_set1_ps(tf.scale.z);

    _mm_store_ps(&result.cols[0].x, _mm_mul_ps(_mm_load_ps(&result.cols[0].x), sx));
    _mm_store_ps(&result.cols[1].x, _mm_mul_ps(_mm_load_ps(&result.cols[1].x), sy));
    _mm_store_ps(&result.cols[2].x, _mm_mul_ps(_mm_load_ps(&result.cols[2].x), sz));

    /* Set translation in column 3 */
    result.cols[3] = {tf.position.x, tf.position.y, tf.position.z, 1.0f};

    return result;
}

/* Inverse of a uniform-scale transform.
 *
 * If scale is uniform (sx == sy == sz == s):
 *   inv_scale = 1/s
 *   inv_rotation = conjugate(rotation)
 *   inv_position = inv_rotation * (-position * inv_scale)
 *
 * Much cheaper than a general mat4 inverse.
 */
inline transform inverse_uniform(const transform& tf) noexcept {
    const float inv_s = 1.0f / tf.scale.x; /* assumes uniform scale */

    const quat inv_rot = conjugate(tf.rotation);

    /* -position * inv_scale, then rotate by inverse rotation */
    const vec3 scaled_neg_pos = {-tf.position.x * inv_s, -tf.position.y * inv_s, -tf.position.z * inv_s};
    const vec3 inv_pos = rotate(inv_rot, scaled_neg_pos);

    return {inv_pos, inv_rot, {inv_s, inv_s, inv_s}};
}

/* Inverse of a non-uniform-scale transform.
 *
 * inv_scale = { 1/sx, 1/sy, 1/sz }
 * inv_rotation = conjugate(rotation)
 * inv_position = inv_rotation * (-position / scale)
 *
 * Note: non-uniform scale + rotation doesn't compose cleanly.
 * The inverse is correct for a single transform, but chaining
 * non-uniform-scale transforms is not equivalent to multiplying
 * these structs (requires a full mat4 for that).
 */
inline transform inverse(const transform& tf) noexcept {
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 s = _mm_load_ps(&tf.scale.x);
    const __m128 inv_s = _mm_div_ps(one, s);

    vec3 inv_scale{};
    _mm_store_ps(&inv_scale.x, inv_s);

    const quat inv_rot = conjugate(tf.rotation);

    /* -position / scale */
    const __m128 neg_pos = _mm_xor_ps(_mm_load_ps(&tf.position.x), _mm_set1_ps(-0.0f));
    const __m128 scaled_neg_pos = _mm_mul_ps(neg_pos, inv_s);

    vec3 snp{};
    _mm_store_ps(&snp.x, scaled_neg_pos);

    const vec3 inv_pos = rotate(inv_rot, snp);

    return {inv_pos, inv_rot, inv_scale};
}

/* Compose two transforms: result applies b first, then a.
 *
 * Equivalent to: to_mat4(a) * to_mat4(b)
 * But stays in transform representation.
 *
 * result.scale    = a.scale * b.scale
 * result.rotation = a.rotation * b.rotation
 * result.position = a.position + a.rotation * (a.scale * b.position)
 */
inline transform compose(const transform& a, const transform& b) noexcept {
    /* Scale: component-wise multiply */
    const __m128 sa = _mm_load_ps(&a.scale.x);
    const __m128 sb = _mm_load_ps(&b.scale.x);
    vec3 result_scale{};
    _mm_store_ps(&result_scale.x, _mm_mul_ps(sa, sb));

    /* Rotation: quaternion multiply */
    const quat result_rot = a.rotation * b.rotation;

    /* Position: a.pos + rotate(a.rot, a.scale * b.pos) */
    const __m128 bp = _mm_load_ps(&b.position.x);
    const __m128 scaled_bp = _mm_mul_ps(sa, bp);

    vec3 sbp{};
    _mm_store_ps(&sbp.x, scaled_bp);

    const vec3 rotated = rotate(a.rotation, sbp);

    const __m128 result_pos = _mm_add_ps(_mm_load_ps(&a.position.x), _mm_load_ps(&rotated.x));
    vec3 rp{};
    _mm_store_ps(&rp.x, result_pos);

    return {rp, result_rot, result_scale};
}

/* Transform a point: applies scale, then rotation, then translation.
 * Convenience function that avoids building a full mat4 for single points.
 * For batches, prefer converting to mat4 first.
 */
inline vec3 transform_point(const transform& tf, const vec3& point) noexcept {
    /* scale * point */
    const __m128 sp = _mm_mul_ps(_mm_load_ps(&tf.scale.x), _mm_load_ps(&point.x));

    vec3 scaled{};
    _mm_store_ps(&scaled.x, sp);

    /* rotate(scaled) + position */
    const vec3 rotated = rotate(tf.rotation, scaled);

    const __m128 result = _mm_add_ps(_mm_load_ps(&rotated.x), _mm_load_ps(&tf.position.x));

    vec3 out{};
    _mm_store_ps(&out.x, result);
    return out;
}

/* Transform a direction: applies scale and rotation, but NOT translation.
 * Use for normals/directions that shouldn't be affected by position.
 */
inline vec3 transform_direction(const transform& tf, const vec3& direction) noexcept {
    const __m128 sd = _mm_mul_ps(_mm_load_ps(&tf.scale.x), _mm_load_ps(&direction.x));

    vec3 scaled{};
    _mm_store_ps(&scaled.x, sd);

    return rotate(tf.rotation, scaled);
}

} // namespace sgl
