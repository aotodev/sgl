/**
 * @file neon.cppm
 * @brief NEON SIMD implementations of all vector, matrix, quaternion, box, and geometric operations.
 *        Targets Apple Silicon (M1–M5), ARMv8.2-A+.
 *
 * Uses native float32x2_t (D-form) for vec2 and box2d corners, and float32x4_t (Q-form)
 * for vec3, vec4, mat4 columns, quaternions. On Apple Silicon, D-form and Q-form arithmetic
 * have identical latency and throughput — both execute on the same SIMD pipes.
 */
module;

#include <arm_neon.h>

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

/* ============================================================
 * Overloaded D-form / Q-form NEON primitives
 *
 * By overloading on the register type, template operators using
 * load()/store() dispatch to the correct width automatically.
 * ============================================================ */

/* --- Arithmetic --- */
inline float32x2_t n_add(float32x2_t a, float32x2_t b) noexcept {
    return vadd_f32(a, b);
}
inline float32x4_t n_add(float32x4_t a, float32x4_t b) noexcept {
    return vaddq_f32(a, b);
}
inline float32x2_t n_sub(float32x2_t a, float32x2_t b) noexcept {
    return vsub_f32(a, b);
}
inline float32x4_t n_sub(float32x4_t a, float32x4_t b) noexcept {
    return vsubq_f32(a, b);
}
inline float32x2_t n_mul(float32x2_t a, float32x2_t b) noexcept {
    return vmul_f32(a, b);
}
inline float32x4_t n_mul(float32x4_t a, float32x4_t b) noexcept {
    return vmulq_f32(a, b);
}
inline float32x2_t n_div(float32x2_t a, float32x2_t b) noexcept {
    return vdiv_f32(a, b);
}
inline float32x4_t n_div(float32x4_t a, float32x4_t b) noexcept {
    return vdivq_f32(a, b);
}
inline float32x2_t n_neg(float32x2_t a) noexcept {
    return vneg_f32(a);
}
inline float32x4_t n_neg(float32x4_t a) noexcept {
    return vnegq_f32(a);
}
inline float32x2_t n_abs(float32x2_t a) noexcept {
    return vabs_f32(a);
}
inline float32x4_t n_abs(float32x4_t a) noexcept {
    return vabsq_f32(a);
}
inline float32x2_t n_min(float32x2_t a, float32x2_t b) noexcept {
    return vmin_f32(a, b);
}
inline float32x4_t n_min(float32x4_t a, float32x4_t b) noexcept {
    return vminq_f32(a, b);
}
inline float32x2_t n_max(float32x2_t a, float32x2_t b) noexcept {
    return vmax_f32(a, b);
}
inline float32x4_t n_max(float32x4_t a, float32x4_t b) noexcept {
    return vmaxq_f32(a, b);
}
inline float32x2_t n_sqrt(float32x2_t a) noexcept {
    return vsqrt_f32(a);
}
inline float32x4_t n_sqrt(float32x4_t a) noexcept {
    return vsqrtq_f32(a);
}

/* --- Scalar broadcast --- */
inline float32x2_t n_dup(float s, float32x2_t) noexcept {
    return vdup_n_f32(s);
}
inline float32x4_t n_dup(float s, float32x4_t) noexcept {
    return vdupq_n_f32(s);
}

/* --- Scalar multiply --- */
inline float32x2_t n_mul_n(float32x2_t a, float s) noexcept {
    return vmul_n_f32(a, s);
}
inline float32x4_t n_mul_n(float32x4_t a, float s) noexcept {
    return vmulq_n_f32(a, s);
}

/* --- FMA: acc + a*b --- */
inline float32x2_t n_fma(float32x2_t acc, float32x2_t a, float32x2_t b) noexcept {
    return vfma_f32(acc, a, b);
}
inline float32x4_t n_fma(float32x4_t acc, float32x4_t a, float32x4_t b) noexcept {
    return vfmaq_f32(acc, a, b);
}
inline float32x2_t n_fma_n(float32x2_t acc, float32x2_t a, float s) noexcept {
    return vfma_n_f32(acc, a, s);
}
inline float32x4_t n_fma_n(float32x4_t acc, float32x4_t a, float s) noexcept {
    return vfmaq_n_f32(acc, a, s);
}

/* --- FMS: acc - a*b --- */
inline float32x2_t n_fms(float32x2_t acc, float32x2_t a, float32x2_t b) noexcept {
    return vfms_f32(acc, a, b);
}
inline float32x4_t n_fms(float32x4_t acc, float32x4_t a, float32x4_t b) noexcept {
    return vfmsq_f32(acc, a, b);
}
inline float32x2_t n_fms_n(float32x2_t acc, float32x2_t a, float s) noexcept {
    return vfms_n_f32(acc, a, s);
}
inline float32x4_t n_fms_n(float32x4_t acc, float32x4_t a, float s) noexcept {
    return vfmsq_n_f32(acc, a, s);
}

/* --- Comparisons (return uint mask) --- */
inline uint32x2_t n_ceq(float32x2_t a, float32x2_t b) noexcept {
    return vceq_f32(a, b);
}
inline uint32x4_t n_ceq(float32x4_t a, float32x4_t b) noexcept {
    return vceqq_f32(a, b);
}
inline uint32x2_t n_cle(float32x2_t a, float32x2_t b) noexcept {
    return vcle_f32(a, b);
}
inline uint32x4_t n_cle(float32x4_t a, float32x4_t b) noexcept {
    return vcleq_f32(a, b);
}
inline uint32x2_t n_clt(float32x2_t a, float32x2_t b) noexcept {
    return vclt_f32(a, b);
}
inline uint32x4_t n_clt(float32x4_t a, float32x4_t b) noexcept {
    return vcltq_f32(a, b);
}
inline uint32x2_t n_cgt(float32x2_t a, float32x2_t b) noexcept {
    return vcgt_f32(a, b);
}
inline uint32x4_t n_cgt(float32x4_t a, float32x4_t b) noexcept {
    return vcgtq_f32(a, b);
}

/* --- Mask OR --- */
inline uint32x2_t n_or(uint32x2_t a, uint32x2_t b) noexcept {
    return vorr_u32(a, b);
}
inline uint32x4_t n_or(uint32x4_t a, uint32x4_t b) noexcept {
    return vorrq_u32(a, b);
}

/* --- Blend (bitwise select) --- */
inline float32x2_t n_bsl(uint32x2_t m, float32x2_t a, float32x2_t b) noexcept {
    return vbsl_f32(m, a, b);
}
inline float32x4_t n_bsl(uint32x4_t m, float32x4_t a, float32x4_t b) noexcept {
    return vbslq_f32(m, a, b);
}

/* --- Rsqrt estimate + NR step helper --- */
inline float32x2_t n_rsqrte(float32x2_t a) noexcept {
    return vrsqrte_f32(a);
}
inline float32x4_t n_rsqrte(float32x4_t a) noexcept {
    return vrsqrteq_f32(a);
}
inline float32x2_t n_rsqrts(float32x2_t a, float32x2_t b) noexcept {
    return vrsqrts_f32(a, b);
}
inline float32x4_t n_rsqrts(float32x4_t a, float32x4_t b) noexcept {
    return vrsqrtsq_f32(a, b);
}

/* --- Movemask: extract sign bits as integer mask --- */
inline std::int32_t n_movemask(uint32x2_t v) noexcept {
    uint32x2_t bits = vshr_n_u32(v, 31);
    return static_cast<std::int32_t>(vget_lane_u32(bits, 0) | (vget_lane_u32(bits, 1) << 1));
}

inline std::int32_t n_movemask(uint32x4_t v) noexcept {
    static const uint32x4_t shift = {0, 1, 2, 3};
    const uint32x4_t bits = vshrq_n_u32(v, 31);
    const uint32x4_t shifted = vshlq_u32(bits, vreinterpretq_s32_u32(shift));
    return static_cast<std::int32_t>(vaddvq_u32(shifted));
}

/* ============================================================
 * Load / Store — vec2 uses D-form natively
 * ============================================================ */

inline float32x2_t load(const vec2& v) noexcept {
    return vld1_f32(&v.x);
}
inline void store(const float32x2_t vals, vec2& dst) noexcept {
    vst1_f32(&dst.x, vals);
}

inline float32x4_t load(const vec3& v) noexcept {
    return vld1q_f32(&v.x);
}
inline void store(const float32x4_t vals, vec3& dst) noexcept {
    vst1q_f32(&dst.x, vals);
}

inline float32x4_t load(const vec4& v) noexcept {
    return vld1q_f32(&v.x);
}
inline void store(const float32x4_t vals, vec4& dst) noexcept {
    vst1q_f32(&dst.x, vals);
}

inline float32x4_t load(const quat& q) noexcept {
    return vld1q_f32(&q.x);
}

inline quat store_quat(const float32x4_t v) noexcept {
    quat r{};
    vst1q_f32(&r.x, v);
    return r;
}

/* ============================================================
 * Box traits
 * ============================================================ */

template <typename Box> struct box_traits;

template <> struct box_traits<box2d> {
    using vec_type = vec2;
    using reg_type = float32x2_t;
    using ureg_type = uint32x2_t;
    static constexpr std::int32_t lane_mask = 0x3;
};

template <> struct box_traits<box3d> {
    using vec_type = vec3;
    using reg_type = float32x4_t;
    using ureg_type = uint32x4_t;
    static constexpr std::int32_t lane_mask = 0x7;
};

template <typename Box> using box_vec_t = typename box_traits<Box>::vec_type;
template <typename Box> using box_reg_t = typename box_traits<Box>::reg_type;

/* ============================================================
 * Box load/store — box2d uses D-form pairs natively
 * ============================================================ */

inline void load_min_max(const box2d& b, float32x2_t& out_min, float32x2_t& out_max) noexcept {
    out_min = vld1_f32(&b.min.x);
    out_max = vld1_f32(&b.max.x);
}

inline box2d make_box_from_min_max(const float32x2_t lo, const float32x2_t hi, box_traits<box2d>) noexcept {
    box2d b{};
    vst1_f32(&b.min.x, lo);
    vst1_f32(&b.max.x, hi);
    return b;
}

inline void load_min_max(const box3d& b, float32x4_t& out_min, float32x4_t& out_max) noexcept {
    out_min = vld1q_f32(&b.min.x);
    out_max = vld1q_f32(&b.max.x);
}

inline box3d make_box_from_min_max(const float32x4_t lo, const float32x4_t hi, box_traits<box3d>) noexcept {
    box3d b{};
    vst1q_f32(&b.min.x, lo);
    vst1q_f32(&b.max.x, hi);
    return b;
}

template <typename Box> Box make_box(const box_reg_t<Box> lo, const box_reg_t<Box> hi) noexcept {
    return make_box_from_min_max(lo, hi, box_traits<Box>{});
}

/* ============================================================
 * Safe divisor — only needed for vec3 (pad lane)
 * vec2 has no padding, vec4 uses all lanes.
 * ============================================================ */

template <typename Vec, typename Reg> Reg safe_divisor(Reg v) noexcept {
    return v;
}
template <> inline float32x4_t safe_divisor<vec3, float32x4_t>(float32x4_t v) noexcept {
    return vsetq_lane_f32(1.0f, v, 3);
}

/* ============================================================
 * Lane mask
 * ============================================================ */

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

/* ============================================================
 * Dot product — native D-form for vec2, Q-form for vec3/vec4
 * ============================================================ */

/* Scalar result */
template <typename Vec> float dot_scalar(auto a, auto b) noexcept;

template <> inline float dot_scalar<vec2>(float32x2_t a, float32x2_t b) noexcept {
    float32x2_t prod = vmul_f32(a, b);
    return vget_lane_f32(vpadd_f32(prod, prod), 0);
}

template <> inline float dot_scalar<vec3>(float32x4_t a, float32x4_t b) noexcept {
    float32x4_t mul = vmulq_f32(a, b);
    return vgetq_lane_f32(mul, 0) + vgetq_lane_f32(mul, 1) + vgetq_lane_f32(mul, 2);
}

template <> inline float dot_scalar<vec4>(float32x4_t a, float32x4_t b) noexcept {
    return vaddvq_f32(vmulq_f32(a, b));
}

/* Broadcast result to all lanes (same width as input) */
inline float32x2_t dot_broadcast_2(float32x2_t a, float32x2_t b) noexcept {
    float32x2_t prod = vmul_f32(a, b);
    return vpadd_f32(prod, prod); /* {sum, sum} */
}

inline float32x4_t dot_broadcast_3(float32x4_t a, float32x4_t b) noexcept {
    return vdupq_n_f32(dot_scalar<vec3>(a, b));
}

inline float32x4_t dot_broadcast_4(float32x4_t a, float32x4_t b) noexcept {
    return vdupq_n_f32(vaddvq_f32(vmulq_f32(a, b)));
}

template <typename Vec> auto dot_broadcast(auto a, auto b) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<Vec>, vec2>) {
        return dot_broadcast_2(a, b);
    } else if constexpr (std::is_same_v<std::remove_cvref_t<Vec>, vec3>) {
        return dot_broadcast_3(a, b);
    } else {
        return dot_broadcast_4(a, b);
    }
}

/* 3-component dot (for internal use with box3d, rigid inverse, etc.) */
inline float dot_scalar_3(float32x4_t a, float32x4_t b) noexcept {
    return dot_scalar<vec3>(a, b);
}

/* Box distance-squared dot */
template <typename Box> float box_dot_scalar(box_reg_t<Box> a, box_reg_t<Box> b) noexcept {
    if constexpr (std::is_same_v<Box, box2d>) {
        return dot_scalar<vec2>(a, b);
    } else {
        return dot_scalar<vec3>(a, b);
    }
}

/* ============================================================
 * Newton-Raphson refined rsqrt — both D-form and Q-form
 * ============================================================ */

inline float32x2_t rsqrt_nr(float32x2_t v) noexcept {
    float32x2_t approx = vrsqrte_f32(v);
    float32x2_t step1 = vmul_f32(approx, vrsqrts_f32(vmul_f32(approx, approx), v));
    return vmul_f32(step1, vrsqrts_f32(vmul_f32(step1, step1), v));
}

inline float32x4_t rsqrt_nr(float32x4_t v) noexcept {
    float32x4_t approx = vrsqrteq_f32(v);
    float32x4_t step1 = vmulq_f32(approx, vrsqrtsq_f32(vmulq_f32(approx, approx), v));
    return vmulq_f32(step1, vrsqrtsq_f32(vmulq_f32(step1, step1), v));
}

/* ============================================================
 * mat4 detail helpers
 * ============================================================ */

struct mat4_cols {
    float32x4_t c[4];
};

inline mat4_cols load_cols(const mat4& m) noexcept {
    return {{vld1q_f32(&m.cols[0].x), vld1q_f32(&m.cols[1].x), vld1q_f32(&m.cols[2].x), vld1q_f32(&m.cols[3].x)}};
}

inline mat4 store_cols(float32x4_t c0, float32x4_t c1, float32x4_t c2, float32x4_t c3) noexcept {
    mat4 r{};
    vst1q_f32(&r.cols[0].x, c0);
    vst1q_f32(&r.cols[1].x, c1);
    vst1q_f32(&r.cols[2].x, c2);
    vst1q_f32(&r.cols[3].x, c3);
    return r;
}

inline float32x4_t linear_combine(const mat4_cols& m, float32x4_t v) noexcept {
    float32x4_t r = vmulq_laneq_f32(m.c[0], v, 0);
    r = vfmaq_laneq_f32(r, m.c[1], v, 1);
    r = vfmaq_laneq_f32(r, m.c[2], v, 2);
    r = vfmaq_laneq_f32(r, m.c[3], v, 3);
    return r;
}

struct sub_dets {
    float s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5;
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

/* Q-form shuffle helpers */
inline float32x4_t combine_low(float32x4_t a, float32x4_t b) noexcept {
    return vcombine_f32(vget_low_f32(a), vget_low_f32(b));
}
inline float32x4_t combine_high(float32x4_t a, float32x4_t b) noexcept {
    return vcombine_f32(vget_high_f32(a), vget_high_f32(b));
}

} // namespace sgl

export namespace sgl {

/* ============================================================
 * Comparisons
 * ============================================================ */

template <vector Vec> bool nearly_equal(const Vec& lhs, const Vec& rhs, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    const auto a = load(lhs);
    const auto b = load(rhs);
    const auto diff = n_abs(n_sub(a, b));
    const auto exact = n_ceq(a, b);
    const auto abs_ok = n_cle(diff, n_dup(abs_tol, a));
    const auto largest = n_max(n_abs(a), n_abs(b));
    const auto rel_ok = n_cle(diff, n_mul(largest, n_dup(rel_tol, a)));
    const auto ok = n_or(exact, n_or(abs_ok, rel_ok));
    constexpr auto mask{lane_mask<Vec>()};
    return (n_movemask(ok) & mask) == mask;
}

template <vector Vec> bool nearly_zero(const Vec& val, const float tol = fp32_abs_tol) noexcept {
    const auto v = load(val);
    const auto ok = n_clt(n_abs(v), n_dup(tol, v));
    constexpr auto mask{lane_mask<Vec>()};
    return (n_movemask(ok) & mask) == mask;
}

template <vector Vec> bool any_nearly_zero(const Vec& val, const float tol = fp32_abs_tol) noexcept {
    const auto v = load(val);
    const auto ok = n_clt(n_abs(v), n_dup(tol, v));
    constexpr auto mask{lane_mask<Vec>()};
    return (n_movemask(ok) & mask) != 0;
}

/* ============================================================
 * Vector-vector operators
 * ============================================================ */

template <vector Vec> Vec operator-(const Vec& v) noexcept {
    Vec out{};
    store(n_neg(load(v)), out);
    return out;
}
template <vector Vec> Vec operator+(const Vec& lhs, const Vec& rhs) noexcept {
    Vec out{};
    store(n_add(load(lhs), load(rhs)), out);
    return out;
}
template <vector Vec> Vec operator-(const Vec& lhs, const Vec& rhs) noexcept {
    Vec out{};
    store(n_sub(load(lhs), load(rhs)), out);
    return out;
}
template <vector Vec> Vec operator*(const Vec& lhs, const Vec& rhs) noexcept {
    Vec out{};
    store(n_mul(load(lhs), load(rhs)), out);
    return out;
}

template <vector Vec> Vec operator/(const Vec& lhs, const Vec& rhs) noexcept {
    assert(!any_nearly_zero(rhs));
    Vec out{};
    store(n_div(load(lhs), safe_divisor<Vec>(load(rhs))), out);
    return out;
}

template <vector Vec> Vec& operator+=(Vec& lhs, const Vec& rhs) noexcept {
    store(n_add(load(lhs), load(rhs)), lhs);
    return lhs;
}
template <vector Vec> Vec& operator-=(Vec& lhs, const Vec& rhs) noexcept {
    store(n_sub(load(lhs), load(rhs)), lhs);
    return lhs;
}
template <vector Vec> Vec& operator*=(Vec& lhs, const Vec& rhs) noexcept {
    store(n_mul(load(lhs), load(rhs)), lhs);
    return lhs;
}

template <vector Vec> Vec& operator/=(Vec& lhs, const Vec& rhs) noexcept {
    assert(!any_nearly_zero(rhs));
    store(n_div(load(lhs), safe_divisor<Vec>(load(rhs))), lhs);
    return lhs;
}

/* ============================================================
 * Vector-scalar operators
 * ============================================================ */

template <vector Vec> Vec operator+(const Vec& lhs, float s) noexcept {
    Vec out{};
    const auto v = load(lhs);
    store(n_add(v, n_dup(s, v)), out);
    return out;
}
template <vector Vec> Vec operator-(const Vec& lhs, float s) noexcept {
    Vec out{};
    const auto v = load(lhs);
    store(n_sub(v, n_dup(s, v)), out);
    return out;
}
template <vector Vec> Vec operator*(const Vec& lhs, float s) noexcept {
    Vec out{};
    store(n_mul_n(load(lhs), s), out);
    return out;
}

template <vector Vec> Vec operator/(const Vec& lhs, float s) noexcept {
    assert(!math::nearly_zero(s));
    Vec out{};
    const auto v = load(lhs);
    store(n_div(v, n_dup(s, v)), out);
    return out;
}

template <vector Vec> Vec& operator+=(Vec& lhs, float s) noexcept {
    const auto v = load(lhs);
    store(n_add(v, n_dup(s, v)), lhs);
    return lhs;
}
template <vector Vec> Vec& operator-=(Vec& lhs, float s) noexcept {
    const auto v = load(lhs);
    store(n_sub(v, n_dup(s, v)), lhs);
    return lhs;
}
template <vector Vec> Vec& operator*=(Vec& lhs, float s) noexcept {
    store(n_mul_n(load(lhs), s), lhs);
    return lhs;
}

template <vector Vec> Vec& operator/=(Vec& lhs, float s) noexcept {
    assert(!math::nearly_zero(s));
    const auto v = load(lhs);
    store(n_div(v, n_dup(s, v)), lhs);
    return lhs;
}

template <vector Vec> Vec operator+(float s, const Vec& rhs) noexcept {
    return rhs + s;
}
template <vector Vec> Vec operator*(float s, const Vec& rhs) noexcept {
    return rhs * s;
}

template <vector Vec> Vec operator-(float s, const Vec& rhs) noexcept {
    Vec out{};
    const auto v = load(rhs);
    store(n_sub(n_dup(s, v), v), out);
    return out;
}

template <vector Vec> Vec operator/(float s, const Vec& rhs) noexcept {
    assert(!any_nearly_zero(rhs));
    Vec out{};
    const auto v = load(rhs);
    store(n_div(n_dup(s, v), safe_divisor<Vec>(v)), out);
    return out;
}

/* ============================================================
 * Dot / length / distance
 * ============================================================ */

template <vector Vec> float dot(const Vec& a, const Vec& b) noexcept {
    return dot_scalar<Vec>(load(a), load(b));
}
template <vector Vec> float length_squared(const Vec& v) noexcept {
    return dot(v, v);
}
template <vector Vec> float length(const Vec& v) noexcept {
    return std::sqrt(dot_scalar<Vec>(load(v), load(v)));
}

/* ============================================================
 * Normalize variants
 * ============================================================ */

template <vector Vec> void normalize_fast(Vec& vec) noexcept {
    const auto v = load(vec);
    store(n_mul(v, n_rsqrte(dot_broadcast<Vec>(v, v))), vec);
}

template <vector Vec> void normalize(Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    const auto v = load(vec);
    store(n_div(v, n_sqrt(dot_broadcast<Vec>(v, v))), vec);
}

template <vector Vec> void normalize_nr(Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    const auto v = load(vec);
    store(n_mul(v, rsqrt_nr(dot_broadcast<Vec>(v, v))), vec);
}

template <vector Vec> Vec normalized_fast(const Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    Vec r{};
    const auto v = load(vec);
    store(n_mul(v, n_rsqrte(dot_broadcast<Vec>(v, v))), r);
    return r;
}

template <vector Vec> Vec normalized_nr(const Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    Vec r{};
    const auto v = load(vec);
    store(n_mul(v, rsqrt_nr(dot_broadcast<Vec>(v, v))), r);
    return r;
}

template <vector Vec> Vec normalized(const Vec& vec) noexcept {
    assert(!nearly_zero(vec) && "normalizing zero-length vector");
    Vec r{};
    const auto v = load(vec);
    store(n_div(v, n_sqrt(dot_broadcast<Vec>(v, v))), r);
    return r;
}

template <vector Vec> void normalize_safe(Vec& vec) noexcept {
    const auto v = load(vec);
    const auto d = dot_broadcast<Vec>(v, v);
    const auto zero = n_dup(0.0f, v);
    const auto is_zero = n_cle(d, n_dup(fp32_abs_tol * fp32_abs_tol, v));
    store(n_bsl(is_zero, zero, n_mul(v, rsqrt_nr(d))), vec);
}

template <vector Vec> Vec normalized_safe(const Vec& vec) noexcept {
    const auto v = load(vec);
    const auto d = dot_broadcast<Vec>(v, v);
    const auto zero = n_dup(0.0f, v);
    const auto is_zero = n_cle(d, n_dup(fp32_abs_tol * fp32_abs_tol, v));
    Vec out{};
    store(n_bsl(is_zero, zero, n_mul(v, rsqrt_nr(d))), out);
    return out;
}

/* ============================================================
 * Geometric helpers
 * ============================================================ */

template <vector Vec> Vec project(const Vec& v, const Vec& onto) noexcept {
    const auto vv = load(v);
    const auto nn = load(onto);
    Vec out{};
    store(n_mul(dot_broadcast<Vec>(vv, nn), nn), out);
    return out;
}

template <vector Vec> Vec reflect(const Vec& v, const Vec& normal) noexcept {
    const auto vv = load(v);
    const auto nn = load(normal);
    const auto d = dot_broadcast<Vec>(vv, nn);
    Vec out{};
    store(n_sub(vv, n_mul(n_add(d, d), nn)), out);
    return out;
}

template <vector Vec> Vec min(const Vec& a, const Vec& b) noexcept {
    Vec out{};
    store(n_min(load(a), load(b)), out);
    return out;
}
template <vector Vec> Vec max(const Vec& a, const Vec& b) noexcept {
    Vec out{};
    store(n_max(load(a), load(b)), out);
    return out;
}
template <vector Vec> Vec abs(const Vec& v) noexcept {
    Vec out{};
    store(n_abs(load(v)), out);
    return out;
}

template <vector Vec> Vec clamp(const Vec& v, const Vec& lo, const Vec& hi) noexcept {
    Vec out{};
    store(n_min(n_max(load(v), load(lo)), load(hi)), out);
    return out;
}

template <vector Vec> Vec clamp(const Vec& v, float lo, float hi) noexcept {
    const auto lv = load(v);
    Vec out{};
    store(n_min(n_max(lv, n_dup(lo, lv)), n_dup(hi, lv)), out);
    return out;
}

template <vector Vec> Vec lerp(const Vec& a, const Vec& b, float t) noexcept {
    const auto va = load(a);
    Vec out{};
    store(n_fma_n(va, n_sub(load(b), va), t), out);
    return out;
}

template <vector Vec> float distance_squared(const Vec& a, const Vec& b) noexcept {
    const auto d = n_sub(load(a), load(b));
    return dot_scalar<Vec>(d, d);
}

template <vector Vec> float distance(const Vec& a, const Vec& b) noexcept {
    const auto d = n_sub(load(a), load(b));
    return std::sqrt(dot_scalar<Vec>(d, d));
}

template <vector Vec> bool is_perpendicular(const Vec& a, const Vec& b, float tol = fp32_abs_tol) noexcept {
    return std::abs(dot_scalar<Vec>(load(a), load(b))) <= tol;
}

template <vector Vec> bool is_parallel(const Vec& a, const Vec& b, float tol = fp32_rel_tol) noexcept {
    const auto va = load(a);
    const auto vb = load(b);
    const auto ab = dot_scalar<Vec>(va, vb);
    const auto aa = dot_scalar<Vec>(va, va);
    const auto bb = dot_scalar<Vec>(vb, vb);
    return std::abs(ab * ab - aa * bb) <= tol * aa * bb;
}

template <vector Vec> bool all_greater_than(const Vec& a, const Vec& b) noexcept {
    constexpr auto mask{lane_mask<Vec>()};
    return (n_movemask(n_cgt(load(a), load(b))) & mask) == mask;
}

template <vector Vec> bool all_less_than(const Vec& a, const Vec& b) noexcept {
    constexpr auto mask{lane_mask<Vec>()};
    return (n_movemask(n_clt(load(a), load(b))) & mask) == mask;
}

template <vector Vec> bool any_greater_than(const Vec& a, const Vec& b) noexcept {
    constexpr auto mask{lane_mask<Vec>()};
    return (n_movemask(n_cgt(load(a), load(b))) & mask) != 0;
}

inline bool contains(const box3d& box, const vec3& point) {
    return all_greater_than(point, box.min) && all_less_than(point, box.max);
}

/* ============================================================
 * Cross products
 * ============================================================ */

/* vec2: a.x*b.y - a.y*b.x — native D-form */
inline float cross(const vec2& a, const vec2& b) noexcept {
    float32x2_t va = load(a);
    float32x2_t vb = load(b);
    float32x2_t b_yx = vrev64_f32(vb);    /* {b.y, b.x} */
    float32x2_t mul = vmul_f32(va, b_yx); /* {a.x*b.y, a.y*b.x} */
    return vget_lane_f32(mul, 0) - vget_lane_f32(mul, 1);
}

/* vec3 cross product */
inline vec3 cross(const vec3& a, const vec3& b) noexcept {
    const float32x4_t va = load(a);
    const float32x4_t vb = load(b);
    const float32x4_t a_yzx = __builtin_shufflevector(va, va, 1, 2, 0, 3);
    const float32x4_t b_yzx = __builtin_shufflevector(vb, vb, 1, 2, 0, 3);
    const float32x4_t result = vfmsq_f32(vmulq_f32(va, b_yzx), a_yzx, vb);
    vec3 out{};
    store(__builtin_shufflevector(result, result, 1, 2, 0, 3), out);
    return out;
}

template <> inline bool is_parallel<vec2>(const vec2& a, const vec2& b, float tol) noexcept {
    return std::abs(cross(a, b)) <= tol;
}
template <> inline bool is_parallel<vec3>(const vec3& a, const vec3& b, float tol) noexcept {
    return nearly_zero(cross(a, b), tol);
}

/* vec2 perpendicular: {-y, x} — D-form */
inline vec2 perpendicular(const vec2& v) noexcept {
    float32x2_t loaded = load(v);
    float32x2_t yx = vrev64_f32(loaded); /* {y, x} */
    vec2 out{};
    store(vset_lane_f32(-vget_lane_f32(yx, 0), yx, 0), out);
    return out;
}

inline vec3 perpendicular(const vec3& v) noexcept {
    float32x4_t av = vabsq_f32(load(v));
    const float ax{vgetq_lane_f32(av, 0)};
    const float ay{vgetq_lane_f32(av, 1)};
    const float az{vgetq_lane_f32(av, 2)};
    if (ax <= ay && ax <= az) {
        return cross(v, vec3{1, 0, 0});
    }
    if (ay <= az) {
        return cross(v, vec3{0, 1, 0});
    }
    return cross(v, vec3{0, 0, 1});
}

template <vector Vec> float angle_between(const Vec& a, const Vec& b) noexcept {
    const auto va = load(a);
    const auto vb = load(b);
    auto ct = dot_scalar<Vec>(va, vb) / std::sqrt(dot_scalar<Vec>(va, va) * dot_scalar<Vec>(vb, vb));
    return std::acos(std::max(-1.0f, std::min(ct, 1.0f)));
}

/* ============================================================
 * Box operations — generic over box2d (D-form) and box3d (Q-form)
 * ============================================================ */

template <box_type Box> bool is_valid(const Box& b) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    constexpr auto mask = box_traits<Box>::lane_mask;
    return (n_movemask(n_cle(lo, hi)) & mask) == mask;
}

template <box_type Box> box_vec_t<Box> center(const Box& b) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    box_vec_t<Box> out{};
    store(n_mul_n(n_add(lo, hi), 0.5f), out);
    return out;
}

template <box_type Box> box_vec_t<Box> half_extents(const Box& b) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    box_vec_t<Box> out{};
    store(n_mul_n(n_sub(hi, lo), 0.5f), out);
    return out;
}

template <box_type Box> box_vec_t<Box> extents(const Box& b) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    box_vec_t<Box> out{};
    store(n_sub(hi, lo), out);
    return out;
}

template <box_type Box> Box translate(const Box& b, const box_vec_t<Box>& offset) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto o = load(offset);
    return make_box<Box>(n_add(lo, o), n_add(hi, o));
}

template <box_type Box> Box scale_about_origin(const Box& b, const box_vec_t<Box>& s) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto sv = load(s);
    return make_box<Box>(n_mul(lo, sv), n_mul(hi, sv));
}

template <box_type Box> Box scale_uniform(const Box& b, float s) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    return make_box<Box>(n_mul_n(lo, s), n_mul_n(hi, s));
}

template <box_type Box> Box dilate(const Box& b, float amount) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto a = n_dup(amount, lo);
    return make_box<Box>(n_sub(lo, a), n_add(hi, a));
}

template <box_type Box> Box dilate(const Box& b, const box_vec_t<Box>& amount) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto a = load(amount);
    return make_box<Box>(n_sub(lo, a), n_add(hi, a));
}

template <box_type Box> bool overlaps(const Box& a, const Box& b) noexcept {
    box_reg_t<Box> a_lo, a_hi, b_lo, b_hi;
    load_min_max(a, a_lo, a_hi);
    load_min_max(b, b_lo, b_hi);
    constexpr auto mask = box_traits<Box>::lane_mask;
    return ((n_movemask(n_cgt(a_lo, b_hi)) | n_movemask(n_cgt(b_lo, a_hi))) & mask) == 0;
}

template <box_type Box> bool contains(const Box& outer, const Box& inner) noexcept {
    box_reg_t<Box> o_lo, o_hi, i_lo, i_hi;
    load_min_max(outer, o_lo, o_hi);
    load_min_max(inner, i_lo, i_hi);
    constexpr auto mask = box_traits<Box>::lane_mask;
    return ((n_movemask(n_cle(o_lo, i_lo)) & n_movemask(n_cle(i_hi, o_hi))) & mask) == mask;
}

template <box_type Box> bool contains_point(const Box& b, const box_vec_t<Box>& p) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto pv = load(p);
    constexpr auto mask = box_traits<Box>::lane_mask;
    return ((n_movemask(n_cle(lo, pv)) & n_movemask(n_cle(pv, hi))) & mask) == mask;
}

template <box_type Box> Box intersection(const Box& a, const Box& b) noexcept {
    box_reg_t<Box> a_lo, a_hi, b_lo, b_hi;
    load_min_max(a, a_lo, a_hi);
    load_min_max(b, b_lo, b_hi);
    return make_box<Box>(n_max(a_lo, b_lo), n_min(a_hi, b_hi));
}

template <box_type Box> Box merge(const Box& a, const Box& b) noexcept {
    box_reg_t<Box> a_lo, a_hi, b_lo, b_hi;
    load_min_max(a, a_lo, a_hi);
    load_min_max(b, b_lo, b_hi);
    return make_box<Box>(n_min(a_lo, b_lo), n_max(a_hi, b_hi));
}

template <box_type Box> Box expand(const Box& b, const box_vec_t<Box>& p) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto pv = load(p);
    return make_box<Box>(n_min(lo, pv), n_max(hi, pv));
}

template <box_type Box> float distance_squared(const Box& b, const box_vec_t<Box>& p) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    const auto pv = load(p);
    const auto diff = n_sub(pv, n_min(n_max(pv, lo), hi));
    return box_dot_scalar<Box>(diff, diff);
}

template <box_type Box> float distance(const Box& b, const box_vec_t<Box>& p) noexcept {
    return std::sqrt(distance_squared(b, p));
}

template <box_type Box> box_vec_t<Box> closest_point(const Box& b, const box_vec_t<Box>& p) noexcept {
    box_reg_t<Box> lo, hi;
    load_min_max(b, lo, hi);
    box_vec_t<Box> out{};
    store(n_min(n_max(load(p), lo), hi), out);
    return out;
}

/* ============================================================
 * box2d specific — native D-form
 * ============================================================ */

inline box2d box2d_from_center_half(const vec2& c, const vec2& half_ext) noexcept {
    float32x2_t cv = load(c), hv = load(half_ext);
    box2d b{};
    vst1_f32(&b.min.x, vsub_f32(cv, hv));
    vst1_f32(&b.max.x, vadd_f32(cv, hv));
    return b;
}

inline box2d box2d_from_point(const vec2& p) noexcept {
    float32x2_t v = load(p);
    box2d b{};
    vst1_f32(&b.min.x, v);
    vst1_f32(&b.max.x, v);
    return b;
}

inline float area(const box2d& b) noexcept {
    float32x2_t diff = vsub_f32(vld1_f32(&b.max.x), vld1_f32(&b.min.x));
    return vget_lane_f32(diff, 0) * vget_lane_f32(diff, 1);
}

inline float perimeter(const box2d& b) noexcept {
    float32x2_t diff = vsub_f32(vld1_f32(&b.max.x), vld1_f32(&b.min.x));
    float sum = vget_lane_f32(vpadd_f32(diff, diff), 0);
    return sum + sum;
}

/* ============================================================
 * box3d specific
 * ============================================================ */

inline box3d box3d_from_center_half(const vec3& c, const vec3& half_ext) noexcept {
    float32x4_t cv = load(c), hv = load(half_ext);
    box3d b{};
    vst1q_f32(&b.min.x, vsubq_f32(cv, hv));
    vst1q_f32(&b.max.x, vaddq_f32(cv, hv));
    return b;
}

inline box3d box3d_from_point(const vec3& p) noexcept {
    float32x4_t v = load(p);
    box3d b{};
    vst1q_f32(&b.min.x, v);
    vst1q_f32(&b.max.x, v);
    return b;
}

inline float volume(const box3d& b) noexcept {
    float32x4_t d = vsubq_f32(vld1q_f32(&b.max.x), vld1q_f32(&b.min.x));
    return vgetq_lane_f32(d, 0) * vgetq_lane_f32(d, 1) * vgetq_lane_f32(d, 2);
}

inline float surface_area(const box3d& b) noexcept {
    float32x4_t d = vsubq_f32(vld1q_f32(&b.max.x), vld1q_f32(&b.min.x));
    float w = vgetq_lane_f32(d, 0), h = vgetq_lane_f32(d, 1), dp = vgetq_lane_f32(d, 2);
    float sum = w * h + w * dp + h * dp;
    return sum + sum;
}

/* ============================================================
 * mat4 operations
 * ============================================================ */

inline mat4 mat4_diagonal(float d) noexcept {
    float32x4_t z = vdupq_n_f32(0.0f);
    return store_cols(vsetq_lane_f32(d, z, 0), vsetq_lane_f32(d, z, 1), vsetq_lane_f32(d, z, 2), vsetq_lane_f32(d, z, 3));
}

inline mat4 add(const mat4& a, const mat4& b) noexcept {
    return store_cols(vaddq_f32(vld1q_f32(&a.cols[0].x), vld1q_f32(&b.cols[0].x)), vaddq_f32(vld1q_f32(&a.cols[1].x), vld1q_f32(&b.cols[1].x)),
        vaddq_f32(vld1q_f32(&a.cols[2].x), vld1q_f32(&b.cols[2].x)), vaddq_f32(vld1q_f32(&a.cols[3].x), vld1q_f32(&b.cols[3].x)));
}

inline mat4 sub(const mat4& a, const mat4& b) noexcept {
    return store_cols(vsubq_f32(vld1q_f32(&a.cols[0].x), vld1q_f32(&b.cols[0].x)), vsubq_f32(vld1q_f32(&a.cols[1].x), vld1q_f32(&b.cols[1].x)),
        vsubq_f32(vld1q_f32(&a.cols[2].x), vld1q_f32(&b.cols[2].x)), vsubq_f32(vld1q_f32(&a.cols[3].x), vld1q_f32(&b.cols[3].x)));
}

inline mat4 scale(const mat4& m, float s) noexcept {
    float32x4_t sv = vdupq_n_f32(s);
    return store_cols(vmulq_f32(vld1q_f32(&m.cols[0].x), sv), vmulq_f32(vld1q_f32(&m.cols[1].x), sv), vmulq_f32(vld1q_f32(&m.cols[2].x), sv),
        vmulq_f32(vld1q_f32(&m.cols[3].x), sv));
}

inline mat4 negate(const mat4& m) noexcept {
    return store_cols(
        vnegq_f32(vld1q_f32(&m.cols[0].x)), vnegq_f32(vld1q_f32(&m.cols[1].x)), vnegq_f32(vld1q_f32(&m.cols[2].x)), vnegq_f32(vld1q_f32(&m.cols[3].x)));
}

inline mat4 operator*(const mat4& a, const mat4& b) noexcept {
    const auto l = load_cols(a);
    const auto r = load_cols(b);
    return store_cols(linear_combine(l, r.c[0]), linear_combine(l, r.c[1]), linear_combine(l, r.c[2]), linear_combine(l, r.c[3]));
}

inline vec4 operator*(const mat4& m, const vec4& v) noexcept {
    vec4 out{};
    vst1q_f32(&out.x, linear_combine(load_cols(m), vld1q_f32(&v.x)));
    return out;
}

inline vec3 operator*(const mat4& m, const vec3& p) noexcept {
    float32x4_t v = vsetq_lane_f32(1.0f, vld1q_f32(&p.x), 3);
    vec3 out{};
    vst1q_f32(&out.x, linear_combine(load_cols(m), v));
    return out;
}

inline vec4 transform_vec4(const mat4& m, const vec4& v) noexcept {
    return m * v;
}

inline vec3 transform_point(const mat4& m, const vec3& p) noexcept {
    return m * p;
}

inline vec3 transform_dir(const mat4& m, const vec3& d) noexcept {
    vec3 out{};
    vst1q_f32(&out.x, linear_combine(load_cols(m), vld1q_f32(&d.x)));
    return out;
}

inline mat4 transpose(const mat4& m) noexcept {
    float32x4_t c0 = vld1q_f32(&m.cols[0].x), c1 = vld1q_f32(&m.cols[1].x), c2 = vld1q_f32(&m.cols[2].x), c3 = vld1q_f32(&m.cols[3].x);
    float32x4_t lo01 = vzip1q_f32(c0, c1), hi01 = vzip2q_f32(c0, c1), lo23 = vzip1q_f32(c2, c3), hi23 = vzip2q_f32(c2, c3);
    return store_cols(combine_low(lo01, lo23), combine_high(lo01, lo23), combine_low(hi01, hi23), combine_high(hi01, hi23));
}

inline float trace(const mat4& m) noexcept {
    return m.cols[0].x + m.cols[1].y + m.cols[2].z + m.cols[3].w;
}

inline float determinant(const mat4& m) noexcept {
    const auto [s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5] = compute_sub_dets(m);
    return s0 * t5 - s1 * t4 + s2 * t3 + s3 * t2 - s4 * t1 + s5 * t0;
}

inline mat4 inverse(const mat4& m) noexcept {
    const auto [s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5] = compute_sub_dets(m);
    const auto det = s0 * t5 - s1 * t4 + s2 * t3 + s3 * t2 - s4 * t1 + s5 * t0;
    const auto id = 1.0f / det;
    const auto& c = m.cols;
    mat4 r{};
    r.cols[0] = {(c[1].y * t5 - c[1].z * t4 + c[1].w * t3) * id, (-c[0].y * t5 + c[0].z * t4 - c[0].w * t3) * id,
        (c[3].y * s5 - c[3].z * s4 + c[3].w * s3) * id, (-c[2].y * s5 + c[2].z * s4 - c[2].w * s3) * id};
    r.cols[1] = {(-c[1].x * t5 + c[1].z * t2 - c[1].w * t1) * id, (c[0].x * t5 - c[0].z * t2 + c[0].w * t1) * id,
        (-c[3].x * s5 + c[3].z * s2 - c[3].w * s1) * id, (c[2].x * s5 - c[2].z * s2 + c[2].w * s1) * id};
    r.cols[2] = {(c[1].x * t4 - c[1].y * t2 + c[1].w * t0) * id, (-c[0].x * t4 + c[0].y * t2 - c[0].w * t0) * id,
        (c[3].x * s4 - c[3].y * s2 + c[3].w * s0) * id, (-c[2].x * s4 + c[2].y * s2 - c[2].w * s0) * id};
    r.cols[3] = {(-c[1].x * t3 + c[1].y * t1 - c[1].z * t0) * id, (c[0].x * t3 - c[0].y * t1 + c[0].z * t0) * id,
        (-c[3].x * s3 + c[3].y * s1 - c[3].z * s0) * id, (c[2].x * s3 - c[2].y * s1 + c[2].z * s0) * id};
    return r;
}

inline mat4 inverse_rigid(const mat4& m) noexcept {
    float32x4_t c0 = vld1q_f32(&m.cols[0].x), c1 = vld1q_f32(&m.cols[1].x), c2 = vld1q_f32(&m.cols[2].x), c3 = vld1q_f32(&m.cols[3].x);
    float32x4_t z = vdupq_n_f32(0.0f);
    float32x4_t lo01 = vzip1q_f32(c0, c1), lo2z = vzip1q_f32(c2, z), hi01 = vzip2q_f32(c0, c1);
    float32x4_t r0 = combine_low(lo01, lo2z), r1 = combine_high(lo01, lo2z), r2 = combine_low(hi01, vzip2q_f32(c2, z));
    float32x4_t neg_t = vnegq_f32(c3);
    float32x4_t new_t = {dot_scalar_3(r0, neg_t), dot_scalar_3(r1, neg_t), dot_scalar_3(r2, neg_t), 1.0f};
    return store_cols(r0, r1, r2, new_t);
}

inline mat4 inverse_affine_uniform(const mat4& m) noexcept {
    float32x4_t c0 = vld1q_f32(&m.cols[0].x), c1 = vld1q_f32(&m.cols[1].x), c2 = vld1q_f32(&m.cols[2].x), c3 = vld1q_f32(&m.cols[3].x);
    float32x4_t inv_s2 = vdupq_n_f32(1.0f / dot_scalar_3(c0, c0)), z = vdupq_n_f32(0.0f);
    float32x4_t lo01 = vzip1q_f32(c0, c1), lo2z = vzip1q_f32(c2, z), hi01 = vzip2q_f32(c0, c1);
    float32x4_t r0 = vmulq_f32(combine_low(lo01, lo2z), inv_s2), r1 = vmulq_f32(combine_high(lo01, lo2z), inv_s2),
                r2 = vmulq_f32(combine_low(hi01, vzip2q_f32(c2, z)), inv_s2);
    float32x4_t neg_t = vnegq_f32(c3);
    float32x4_t new_t = {dot_scalar_3(r0, neg_t), dot_scalar_3(r1, neg_t), dot_scalar_3(r2, neg_t), 1.0f};
    return store_cols(r0, r1, r2, new_t);
}

/* ============================================================
 * mat2 operations
 * ============================================================ */

inline float32x4_t load(const mat2& m) noexcept {
    return vld1q_f32(&m.cols[0].x);
}

inline mat2 add(const mat2& a, const mat2& b) noexcept {
    mat2 r{};
    vst1q_f32(&r.cols[0].x, vaddq_f32(load(a), load(b)));
    return r;
}
inline mat2 sub(const mat2& a, const mat2& b) noexcept {
    mat2 r{};
    vst1q_f32(&r.cols[0].x, vsubq_f32(load(a), load(b)));
    return r;
}
inline mat2 scale(const mat2& m, float s) noexcept {
    mat2 r{};
    vst1q_f32(&r.cols[0].x, vmulq_n_f32(load(m), s));
    return r;
}
inline mat2 negate(const mat2& m) noexcept {
    mat2 r{};
    vst1q_f32(&r.cols[0].x, vnegq_f32(load(m)));
    return r;
}

inline mat2 operator*(const mat2& lhs, const mat2& rhs) noexcept {
    float32x4_t l = load(lhs), r = load(rhs);
    float32x4_t ac_ac = vcombine_f32(vget_low_f32(l), vget_low_f32(l));
    float32x4_t bd_bd = vcombine_f32(vget_high_f32(l), vget_high_f32(l));
    mat2 result{};
    vst1q_f32(&result.cols[0].x, vfmaq_f32(vmulq_f32(ac_ac, vtrn1q_f32(r, r)), bd_bd, vtrn2q_f32(r, r)));
    return result;
}

inline vec2 transform_vec2(const mat2& m, const vec2& v) noexcept {
    float32x4_t mv = load(m);
    float32x2_t vv = load(v);
    float32x2_t ac = vget_low_f32(mv), bd = vget_high_f32(mv);
    vec2 out{};
    store(vfma_f32(vmul_f32(ac, vdup_lane_f32(vv, 0)), bd, vdup_lane_f32(vv, 1)), out);
    return out;
}

inline mat2 transpose(const mat2& m) noexcept {
    mat2 r{};
    vst1q_f32(&r.cols[0].x, __builtin_shufflevector(load(m), load(m), 0, 2, 1, 3));
    return r;
}

inline float determinant(const mat2& m) noexcept {
    float32x4_t v = load(m);
    return vgetq_lane_f32(v, 0) * vgetq_lane_f32(v, 3) - vgetq_lane_f32(v, 2) * vgetq_lane_f32(v, 1);
}

inline mat2 inverse(const mat2& m) noexcept {
    float32x4_t v = load(m);
    float a = vgetq_lane_f32(v, 0), c = vgetq_lane_f32(v, 1), b = vgetq_lane_f32(v, 2), d = vgetq_lane_f32(v, 3);
    float id = 1.0f / (a * d - b * c);
    float32x4_t adj = {d * id, -c * id, -b * id, a * id};
    mat2 r{};
    vst1q_f32(&r.cols[0].x, adj);
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
    vst1q_f32(&r.cols[0].x, vld1q_f32(&m.cols[0].x));
    vst1q_f32(&r.cols[1].x, vld1q_f32(&m.cols[1].x));
    vst1q_f32(&r.cols[2].x, vld1q_f32(&m.cols[2].x));
    return r;
}

inline mat4 to_mat4(const mat3& m) noexcept {
    return store_cols(vld1q_f32(&m.cols[0].x), vld1q_f32(&m.cols[1].x), vld1q_f32(&m.cols[2].x), vsetq_lane_f32(1.0f, vdupq_n_f32(0.0f), 3));
}

inline mat4 to_mat4(const mat3& m, const vec3& translation) noexcept {
    return store_cols(vld1q_f32(&m.cols[0].x), vld1q_f32(&m.cols[1].x), vld1q_f32(&m.cols[2].x), vsetq_lane_f32(1.0f, vld1q_f32(&translation.x), 3));
}

inline vec3 get_translation(const mat4& m) noexcept {
    vec3 out{};
    store(vld1q_f32(&m.cols[3].x), out);
    return out;
}

inline mat4 set_translation(const mat4& m, const vec3& t) noexcept {
    auto r = m;
    float32x4_t tv = vsetq_lane_f32(vgetq_lane_f32(vld1q_f32(&m.cols[3].x), 3), vld1q_f32(&t.x), 3);
    vst1q_f32(&r.cols[3].x, tv);
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
    return vaddvq_f32(vmulq_f32(load(a), load(b)));
}
inline float norm_sq(const quat& q) noexcept {
    return dot(q, q);
}
inline float norm(const quat& q) noexcept {
    return std::sqrt(dot(q, q));
}

inline quat conjugate(const quat& q) noexcept {
    float32x4_t v = load(q);
    const uint32x4_t mask = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000};
    return store_quat(vbslq_f32(mask, vnegq_f32(v), v));
}

inline quat negate(const quat& q) noexcept {
    return store_quat(vnegq_f32(load(q)));
}

inline quat normalized(const quat& q) noexcept {
    float32x4_t v = load(q);
    return store_quat(vdivq_f32(v, vsqrtq_f32(dot_broadcast_4(v, v))));
}

inline quat inverse(const quat& q) noexcept {
    float32x4_t v = load(q);
    const uint32x4_t mask = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000};
    return store_quat(vdivq_f32(vbslq_f32(mask, vnegq_f32(v), v), dot_broadcast_4(v, v)));
}

inline quat inverse_unit(const quat& q) noexcept {
    return conjugate(q);
}

inline quat operator*(const quat& q1, const quat& q2) noexcept {
    float32x4_t a = load(q1), b = load(q2);
    float32x4_t aw = vdupq_laneq_f32(a, 3), ax = vdupq_laneq_f32(a, 0), ay = vdupq_laneq_f32(a, 1), az = vdupq_laneq_f32(a, 2);
    float32x4_t b_wzyx = __builtin_shufflevector(b, b, 3, 2, 1, 0);
    float32x4_t b_zwxy = __builtin_shufflevector(b, b, 2, 3, 0, 1);
    float32x4_t b_yxwz = __builtin_shufflevector(b, b, 1, 0, 3, 2);
    const float32x4_t sx = {1, -1, 1, -1}, sy = {1, 1, -1, -1}, sz = {-1, 1, 1, -1};
    float32x4_t r = vmulq_f32(aw, b);
    r = vfmaq_f32(r, ax, vmulq_f32(b_wzyx, sx));
    r = vfmaq_f32(r, ay, vmulq_f32(b_zwxy, sy));
    r = vfmaq_f32(r, az, vmulq_f32(b_yxwz, sz));
    return store_quat(r);
}

inline vec3 rotate(const quat& q, const vec3& v) noexcept {
    float32x4_t qv = load(q), vv = vld1q_f32(&v.x), qw = vdupq_laneq_f32(qv, 3);
    float32x4_t q_yzx = __builtin_shufflevector(qv, qv, 1, 2, 0, 3), q_zxy = __builtin_shufflevector(qv, qv, 2, 0, 1, 3);
    float32x4_t v_yzx = __builtin_shufflevector(vv, vv, 1, 2, 0, 3), v_zxy = __builtin_shufflevector(vv, vv, 2, 0, 1, 3);
    float32x4_t t = vaddq_f32(vfmsq_f32(vmulq_f32(q_yzx, v_zxy), q_zxy, v_yzx), vfmsq_f32(vmulq_f32(q_yzx, v_zxy), q_zxy, v_yzx));
    float32x4_t t_yzx = __builtin_shufflevector(t, t, 1, 2, 0, 3), t_zxy = __builtin_shufflevector(t, t, 2, 0, 1, 3);
    vec3 out{};
    vst1q_f32(&out.x, vaddq_f32(vfmaq_f32(vv, qw, t), vfmsq_f32(vmulq_f32(q_yzx, t_zxy), q_zxy, t_yzx)));
    return out;
}

inline mat3 to_mat3(const quat& q) noexcept {
    float x2 = 2 * q.x, y2 = 2 * q.y, z2 = 2 * q.z;
    float xx2 = x2 * q.x, xy2 = x2 * q.y, xz2 = x2 * q.z, yy2 = y2 * q.y, yz2 = y2 * q.z, zz2 = z2 * q.z, wx2 = x2 * q.w, wy2 = y2 * q.w, wz2 = z2 * q.w;
    mat3 r{};
    const float32x4_t c0 = {1 - yy2 - zz2, xy2 + wz2, xz2 - wy2, 0};
    const float32x4_t c1 = {xy2 - wz2, 1 - xx2 - zz2, yz2 + wx2, 0};
    const float32x4_t c2 = {xz2 + wy2, yz2 - wx2, 1 - xx2 - yy2, 0};
    vst1q_f32(&r.cols[0].x, c0);
    vst1q_f32(&r.cols[1].x, c1);
    vst1q_f32(&r.cols[2].x, c2);
    return r;
}

inline mat4 to_mat4(const quat& q) noexcept {
    return to_mat4(to_mat3(q));
}
inline mat4 to_mat4(const quat& q, const vec3& t) noexcept {
    return to_mat4(to_mat3(q), t);
}

inline bool nearly_equal(const quat& a, const quat& b, float abs_tol = fp32_abs_tol, float rel_tol = fp32_rel_tol) noexcept {
    float32x4_t av = load(a), bv = load(b);
    float32x4_t diff = vminq_f32(vabsq_f32(vsubq_f32(av, bv)), vabsq_f32(vaddq_f32(av, bv)));
    float32x4_t tol = vaddq_f32(vdupq_n_f32(abs_tol), vmulq_f32(vdupq_n_f32(rel_tol), vmaxq_f32(vabsq_f32(av), vabsq_f32(bv))));
    return n_movemask(vcleq_f32(diff, tol)) == 0xF;
}

inline float angle_between(const quat& a, const quat& b) noexcept {
    return 2.0f * std::acos(std::min(std::abs(dot(a, b)), 1.0f));
}

inline bool is_axis_aligned(const vec3& direction, float tolerance = fp32_rel_tol) noexcept {
    return std::abs(dot_scalar<vec3>(load(normalized(direction)), load(snap_to_axis(direction)))) >= (1.0f - tolerance);
}

/* ============================================================
 * Segment helpers
 * ============================================================ */

template <spatial_vector Vec> Vec closest_point_on_segment(const Vec& point, const Vec& seg_start, const Vec& seg_end) noexcept {
    const auto p = load(point), a = load(seg_start), b = load(seg_end);
    const auto ab = n_sub(b, a), ap = n_sub(p, a);
    auto t = n_div(dot_broadcast<Vec>(ab, ap), dot_broadcast<Vec>(ab, ab));
    t = n_min(n_max(t, n_dup(0.0f, t)), n_dup(1.0f, t));
    Vec out{};
    store(n_fma(a, t, ab), out);
    return out;
}

template <spatial_vector Vec> float distance_point_to_segment_sq(const Vec& p, const Vec& a, const Vec& b) noexcept {
    return distance_squared(p, closest_point_on_segment(p, a, b));
}
template <spatial_vector Vec> float distance_point_to_segment(const Vec& p, const Vec& a, const Vec& b) noexcept {
    return distance(p, closest_point_on_segment(p, a, b));
}

/**
 * Closest points between two 3D segments [a0, a1] and [b0, b1].
 *
 * Follows Ericson, "Real-Time Collision Detection": solve for the closest
 * points on the two infinite lines, then clamp the parameters into [0, 1] and
 * re-solve as endpoints fall outside, so segment ends are handled exactly.
 * Degenerate (zero-length) segments collapse to the point-segment / point-point
 * cases. The result carries the squared distance, so capsule/clearance tests can
 * compare against a squared radius without a square root.
 */
inline segment_closest_result closest_points_between_segments(const vec3& a0, const vec3& a1, const vec3& b0, const vec3& b1) noexcept {
    const float32x4_t p1 = vld1q_f32(&a0.x);
    const float32x4_t p2 = vld1q_f32(&b0.x);
    const float32x4_t d1 = vsubq_f32(vld1q_f32(&a1.x), p1); /* direction of segment A */
    const float32x4_t d2 = vsubq_f32(vld1q_f32(&b1.x), p2); /* direction of segment B */
    const float32x4_t r = vsubq_f32(p1, p2);

    const auto a{dot_scalar_3(d1, d1)}; /* squared length of A */
    const auto e{dot_scalar_3(d2, d2)}; /* squared length of B */
    const auto f {
        dot_scalar_3(d2, r);

        /* squared, since a and e are squared lengths */
        constexpr auto eps{fp32_abs_tol * fp32_abs_tol};

        float s{};
        float t{};
        if (a <= eps && e <= eps) {
            /* both segments degenerate to points */
        } else if (a <= eps) {
            /* segment A is a point */
            t = std::clamp(f / e, 0.0f, 1.0f);
        } else {
            const auto c{dot_scalar_3(d1, r)};
            if (e <= eps) {
                /* segment B is a point */
                s = std::clamp(-c / a, 0.0f, 1.0f);
            } else {
                const auto b{dot_scalar_3(d1, d2)};
                const auto denom{a * e - b * b}; /* >= 0; zero when segments are parallel */

                /* closest point on line A to line B, clamped; parallel -> pick s = 0 */
                s = denom > 0.0f ? std::clamp((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;

                /* closest point on line B to S1(s); re-clamp s if t leaves [0, 1] */
                t = (b * s + f) / e;
                if (t < 0.0f) {
                    t = 0.0f;
                    s = std::clamp(-c / a, 0.0f, 1.0f);
                } else if (t > 1.0f) {
                    t = 1.0f;
                    s = std::clamp((b - c) / a, 0.0f, 1.0f);
                }
            }
        }

        const float32x4_t c1 = vfmaq_n_f32(p1, d1, s); /* a0 + s * d1 */
        const float32x4_t c2 = vfmaq_n_f32(p2, d2, t); /* b0 + t * d2 */
        const float32x4_t diff = vsubq_f32(c2, c1);

        segment_closest_result out{};
        vst1q_f32(&out.point_a.x, c1);
        vst1q_f32(&out.point_b.x, c2);
        out.s = s;
        out.t = t;
        out.distance_squared = dot_scalar_3(diff, diff);
        return out;
    }

    /* Squared distance between two 3D segments (closest approach). */
    inline float distance_between_segments_sq(const vec3& a0, const vec3& a1, const vec3& b0, const vec3& b1) noexcept {
        return closest_points_between_segments(a0, a1, b0, b1).distance_squared;
    }

    /* Distance between two 3D segments (closest approach). */
    inline float distance_between_segments(const vec3& a0, const vec3& a1, const vec3& b0, const vec3& b1) noexcept {
        return std::sqrt(distance_between_segments_sq(a0, a1, b0, b1));
    }

    /* ============================================================
     * Plane operations
     * ============================================================ */

    inline plane plane_from_point_normal(const vec3& point, const vec3& normal) noexcept {
        return {normal, dot(normal, point)};
    }

    inline plane plane_from_points(const vec3& a, const vec3& b, const vec3& c) noexcept {
        auto n = normalized(cross(b - a, c - a));
        return {n, dot(n, a)};
    }

    inline float signed_distance_to_plane(const vec3& point, const plane& pl) noexcept {
        return dot(pl.normal, point) - pl.d;
    }
    inline float distance_to_plane(const vec3& point, const plane& pl) noexcept {
        return std::abs(signed_distance_to_plane(point, pl));
    }

    inline vec3 project_onto_plane(const vec3& point, const plane& pl) noexcept {
        float dist = signed_distance_to_plane(point, pl);
        vec3 out{};
        vst1q_f32(&out.x, vfmsq_n_f32(vld1q_f32(&point.x), vld1q_f32(&pl.normal.x), dist));
        return out;
    }

    inline vec2 project_to_2d(const vec3& point, const plane_basis& basis) noexcept {
        float32x4_t p = vsubq_f32(vld1q_f32(&point.x), vld1q_f32(&basis.origin.x));
        return {dot_scalar_3(p, vld1q_f32(&basis.u.x)), dot_scalar_3(p, vld1q_f32(&basis.v.x))};
    }

    inline vec3 unproject_to_3d(const vec2& p2, const plane_basis& basis) noexcept {
        vec3 out{};
        vst1q_f32(&out.x, vfmaq_n_f32(vfmaq_n_f32(vld1q_f32(&basis.origin.x), vld1q_f32(&basis.u.x), p2.x), vld1q_f32(&basis.v.x), p2.y));
        return out;
    }

    inline void project_to_2d_batch(const vec3* pts3, vec2* pts2, std::int32_t count, const plane_basis& basis) noexcept {
        float32x4_t o = vld1q_f32(&basis.origin.x), u = vld1q_f32(&basis.u.x), v = vld1q_f32(&basis.v.x);
        for (int i = 0; i < count; ++i) {
            float32x4_t p = vsubq_f32(vld1q_f32(&pts3[i].x), o);
            pts2[i] = {dot_scalar_3(p, u), dot_scalar_3(p, v)};
        }
    }

    inline void unproject_to_3d_batch(const vec2* pts2, vec3* pts3, std::int32_t count, const plane_basis& basis) noexcept {
        float32x4_t o = vld1q_f32(&basis.origin.x), u = vld1q_f32(&basis.u.x), v = vld1q_f32(&basis.v.x);
        for (std::int32_t i{0}; i < count; ++i) {
            vst1q_f32(&pts3[i].x, vfmaq_n_f32(vfmaq_n_f32(o, u, pts2[i].x), v, pts2[i].y));
        }
    }

    /* ============================================================
     * Intersection tests
     * ============================================================ */

    inline segment_intersection_2d intersect_segments_2d(const vec2& a0, const vec2& a1, const vec2& b0, const vec2& b1, float epsilon = 1e-7f) noexcept {
        vec2 da = a1 - a0, db = b1 - b0, d0 = b0 - a0;
        const float denom{cross(da, db)};
        if (std::abs(denom) < epsilon) {
            return {};
        }
        const float inv{1.0f / denom};
        const float t{cross(d0, db) * inv};
        const float u{cross(d0, da) * inv};
        if (t < 0 || t > 1 || u < 0 || u > 1) {
            return {};
        }
        vec2 pt{};
        store(vfma_n_f32(load(a0), load(da), t), pt);
        return {pt, t, u, true};
    }

    inline ray_box_hit intersect_ray_box(const vec2& origin, const vec2& direction, const box2d& box) noexcept {
        float32x2_t orig = load(origin), inv_dir = vdiv_f32(vdup_n_f32(1.0f), load(direction));
        float32x2_t bmin, bmax;
        load_min_max(box, bmin, bmax);
        float32x2_t t1 = vmul_f32(vsub_f32(bmin, orig), inv_dir), t2 = vmul_f32(vsub_f32(bmax, orig), inv_dir);
        float32x2_t t_near = vmin_f32(t1, t2), t_far = vmax_f32(t1, t2);
        const float te{std::max(vget_lane_f32(t_near, 0), vget_lane_f32(t_near, 1))};
        const float tx{std::min(vget_lane_f32(t_far, 0), vget_lane_f32(t_far, 1))};
        if (te > tx || tx < 0) {
            return {};
        }
        return {te, tx, true};
    }

    inline ray_box_hit intersect_ray_box(const vec3& origin, const vec3& direction, const box3d& box) noexcept {
        float32x4_t orig = vld1q_f32(&origin.x), inv_dir = vdivq_f32(vdupq_n_f32(1.0f), vld1q_f32(&direction.x));
        float32x4_t t1 = vmulq_f32(vsubq_f32(vld1q_f32(&box.min.x), orig), inv_dir), t2 = vmulq_f32(vsubq_f32(vld1q_f32(&box.max.x), orig), inv_dir);
        float32x4_t tn = vminq_f32(t1, t2), tf = vmaxq_f32(t1, t2);
        const float te{std::max({vgetq_lane_f32(tn, 0), vgetq_lane_f32(tn, 1), vgetq_lane_f32(tn, 2)})};
        const float tx{std::min({vgetq_lane_f32(tf, 0), vgetq_lane_f32(tf, 1), vgetq_lane_f32(tf, 2)})};
        if (te > tx || tx < 0) {
            return {};
        }
        return {te, tx, true};
    }

    template <typename Vec, typename Box>
        requires((std::is_same_v<Vec, vec2> && std::is_same_v<Box, box2d>) || (std::is_same_v<Vec, vec3> && std::is_same_v<Box, box3d>))
    bool segment_intersects_box(const Vec& a, const Vec& b, const Box& box) noexcept {
        auto hit = intersect_ray_box(a, b - a, box);
        return hit.hit && hit.t_entry <= 1.0f && hit.t_exit >= 0.0f;
    }

    inline int winding_number(const vec2& point, const vec2* verts, std::int32_t count) noexcept {
        std::int32_t w{};
        for (std::int32_t i{0}; i < count; ++i) {
            const auto& v0 = verts[i];
            const auto& v1 = verts[(i + 1) % count];
            if (v0.y <= point.y) {
                if (v1.y > point.y && cross(v1 - v0, point - v0) > 0) {
                    ++w;
                }
            } else {
                if (v1.y <= point.y && cross(v1 - v0, point - v0) < 0) {
                    --w;
                }
            }
        }
        return w;
    }

    inline bool point_in_polygon(const vec2& p, const vec2* v, std::int32_t n) noexcept {
        return winding_number(p, v, n) != 0;
    }

    inline segment_polygon_hit intersect_segment_polygon(const vec2& a, const vec2& b, const vec2* verts, std::int32_t count) noexcept {
        segment_polygon_hit nearest{};
        float best = std::numeric_limits<float>::max();
        for (int i = 0; i < count; ++i) {
            auto r = intersect_segments_2d(a, b, verts[i], verts[(i + 1) % count]);
            if (r.intersects && r.t < best) {
                best = r.t;
                nearest = {r.point, r.t, i, true};
            }
        }
        return nearest;
    }

    /* ============================================================
     * Grid operations
     * ============================================================ */

    inline ivec2 grid_cell(const vec2& point, const vec2& grid_origin, float cell_size) noexcept {
        float32x2_t p = vmul_n_f32(vsub_f32(load(point), load(grid_origin)), 1.0f / cell_size);
        int32x2_t ci = vcvt_s32_f32(vrndm_f32(p));
        return {vget_lane_s32(ci, 0), vget_lane_s32(ci, 1)};
    }

    inline ivec3 grid_cell(const vec3& point, const vec3& grid_origin, float cell_size) noexcept {
        float32x4_t p = vmulq_n_f32(vsubq_f32(vld1q_f32(&point.x), vld1q_f32(&grid_origin.x)), 1.0f / cell_size);
        int32x4_t ci = vcvtq_s32_f32(vrndmq_f32(p));
        return {vgetq_lane_s32(ci, 0), vgetq_lane_s32(ci, 1), vgetq_lane_s32(ci, 2)};
    }

    inline grid_cell_range_2d grid_cell_range(const box2d& b, const vec2& o, float s) noexcept {
        return {grid_cell(b.min, o, s), grid_cell(b.max, o, s)};
    }
    inline grid_cell_range_3d grid_cell_range(const box3d& b, const vec3& o, float s) noexcept {
        return {grid_cell(b.min, o, s), grid_cell(b.max, o, s)};
    }

    /* ============================================================
     * Transform operations
     * ============================================================ */

    inline mat4 to_mat4(const transform& tf) noexcept {
        mat4 r = to_mat4(tf.rotation);
        vst1q_f32(&r.cols[0].x, vmulq_n_f32(vld1q_f32(&r.cols[0].x), tf.scale.x));
        vst1q_f32(&r.cols[1].x, vmulq_n_f32(vld1q_f32(&r.cols[1].x), tf.scale.y));
        vst1q_f32(&r.cols[2].x, vmulq_n_f32(vld1q_f32(&r.cols[2].x), tf.scale.z));
        r.cols[3] = {tf.position.x, tf.position.y, tf.position.z, 1.0f};
        return r;
    }

    inline transform inverse_uniform(const transform& tf) noexcept {
        float is = 1.0f / tf.scale.x;
        quat ir = conjugate(tf.rotation);
        return {rotate(ir, {-tf.position.x * is, -tf.position.y * is, -tf.position.z * is}), ir, {is, is, is}};
    }

    inline transform inverse(const transform& tf) noexcept {
        float32x4_t inv_s = vdivq_f32(vdupq_n_f32(1.0f), vld1q_f32(&tf.scale.x));
        vec3 is{};
        vst1q_f32(&is.x, inv_s);
        quat ir = conjugate(tf.rotation);
        vec3 snp{};
        vst1q_f32(&snp.x, vmulq_f32(vnegq_f32(vld1q_f32(&tf.position.x)), inv_s));
        return {rotate(ir, snp), ir, is};
    }

    inline transform compose(const transform& a, const transform& b) noexcept {
        float32x4_t sa = vld1q_f32(&a.scale.x);
        vec3 rs{};
        vst1q_f32(&rs.x, vmulq_f32(sa, vld1q_f32(&b.scale.x)));
        quat rr = a.rotation * b.rotation;
        vec3 sbp{};
        vst1q_f32(&sbp.x, vmulq_f32(sa, vld1q_f32(&b.position.x)));
        vec3 rot = rotate(a.rotation, sbp);
        vec3 rp{};
        vst1q_f32(&rp.x, vaddq_f32(vld1q_f32(&a.position.x), vld1q_f32(&rot.x)));
        return {rp, rr, rs};
    }

    inline vec3 transform_point(const transform& tf, const vec3& pt) noexcept {
        vec3 sc{};
        vst1q_f32(&sc.x, vmulq_f32(vld1q_f32(&tf.scale.x), vld1q_f32(&pt.x)));
        vec3 rot = rotate(tf.rotation, sc);
        vec3 out{};
        vst1q_f32(&out.x, vaddq_f32(vld1q_f32(&rot.x), vld1q_f32(&tf.position.x)));
        return out;
    }

    inline vec3 transform_direction(const transform& tf, const vec3& dir) noexcept {
        vec3 sc{};
        vst1q_f32(&sc.x, vmulq_f32(vld1q_f32(&tf.scale.x), vld1q_f32(&dir.x)));
        return rotate(tf.rotation, sc);
    }

} // namespace sgl
