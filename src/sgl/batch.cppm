/**
 * @file batch.cppm
 * @brief Internal fixed-width SIMD wrapper the SoA kernels are written against. Nothing here is exported.
 *
 * Semantics shared by every backend, so a kernel gives the same answer wherever it runs:
 *   - min(a, b) is `a < b ? a : b` and max(a, b) is `a > b ? a : b`: a NaN in either operand yields b.
 *     Kernels keep accumulators in b, so a NaN input never reaches them.
 *   - comparisons are ordered: false when either side is NaN.
 *   - loads and stores are unaligned.
 *
 * The native members are marked inline explicitly: in a module purview an in-class definition
 * is not implicitly inline, and importers do not inline a non-inline function (constexpr ones
 * already are).
 */
module;

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#define SGL_BATCH_AVX2 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SGL_BATCH_NEON 1
#endif

export module sgl:batch;

namespace sgl::detail {

struct scalar_isa {};
struct avx2_isa {};
struct neon_isa {};

#if defined(SGL_BATCH_AVX2)
using native_isa = avx2_isa;
#elif defined(SGL_BATCH_NEON)
using native_isa = neon_isa;
#else
using native_isa = scalar_isa;
#endif

template <class T, class Isa> struct batch;
template <class T, class Isa> struct batch_mask;

/* ============================================================
 * scalar: width 1, any T; also runs the tail of every native loop
 * ============================================================ */

template <class T> struct batch_mask<T, scalar_isa> {
    bool m;

    friend constexpr batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {a.m && b.m}; }
    friend constexpr batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {a.m || b.m}; }
    friend constexpr batch_mask operator~(const batch_mask a) noexcept { return {!a.m}; }
    constexpr std::uint32_t bits() const noexcept { return m ? 1u : 0u; }
};

template <class T> struct batch<T, scalar_isa> {
    using value_type = T;
    using mask = batch_mask<T, scalar_isa>;
    static constexpr std::size_t width = 1;

    T v;

    static constexpr batch load(const T* p) noexcept { return {*p}; }
    static constexpr batch broadcast(const T s) noexcept { return {s}; }
    /* Lanes [0, k) set; k <= width. */
    static constexpr mask first_lanes(const std::size_t k) noexcept { return {k > 0}; }
    constexpr void store(T* p) const noexcept { *p = v; }

    friend constexpr batch operator-(const batch a, const batch b) noexcept { return {a.v - b.v}; }
    friend constexpr batch operator*(const batch a, const batch b) noexcept { return {a.v * b.v}; }
    friend constexpr mask operator<=(const batch a, const batch b) noexcept { return {a.v <= b.v}; }
    friend constexpr mask operator<(const batch a, const batch b) noexcept { return {a.v < b.v}; }
    friend constexpr batch min(const batch a, const batch b) noexcept { return {a.v < b.v ? a.v : b.v}; }
    friend constexpr batch max(const batch a, const batch b) noexcept { return {a.v > b.v ? a.v : b.v}; }
    friend constexpr batch select(const mask m, const batch a, const batch b) noexcept { return m.m ? a : b; }
    friend constexpr T reduce_min(const batch a) noexcept { return a.v; }
    friend constexpr T reduce_max(const batch a) noexcept { return a.v; }
};

/* ============================================================
 * AVX2: 8 x float
 * ============================================================ */

#if defined(SGL_BATCH_AVX2)

template <> struct batch_mask<float, avx2_isa> {
    __m256 m;

    friend inline batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {_mm256_and_ps(a.m, b.m)}; }
    friend inline batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {_mm256_or_ps(a.m, b.m)}; }
    friend inline batch_mask operator~(const batch_mask a) noexcept { return {_mm256_xor_ps(a.m, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))}; }
    inline std::uint32_t bits() const noexcept { return static_cast<std::uint32_t>(_mm256_movemask_ps(m)); }
};

template <> struct batch<float, avx2_isa> {
    using value_type = float;
    using mask = batch_mask<float, avx2_isa>;
    static constexpr std::size_t width = 8;

    __m256 v;

    /* memcpy, not _mm256_loadu_ps: GCC 16 drops the aligned(1) of __m256_u when an importer
     * inlines the intrinsic from the module, and emits vmovaps. */
    static inline batch load(const float* p) noexcept {
        batch r;
        std::memcpy(&r.v, p, sizeof r.v);
        return r;
    }
    static inline batch broadcast(const float s) noexcept { return {_mm256_set1_ps(s)}; }
    static inline mask first_lanes(const std::size_t k) noexcept {
        return {_mm256_cmp_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_ps(static_cast<float>(k)), _CMP_LT_OQ)};
    }
    inline void store(float* p) const noexcept { std::memcpy(p, &v, sizeof v); }

    friend inline batch operator-(const batch a, const batch b) noexcept { return {_mm256_sub_ps(a.v, b.v)}; }
    friend inline batch operator*(const batch a, const batch b) noexcept { return {_mm256_mul_ps(a.v, b.v)}; }
    friend inline mask operator<=(const batch a, const batch b) noexcept { return {_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)}; }
    friend inline mask operator<(const batch a, const batch b) noexcept { return {_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)}; }
    /* vminps / vmaxps return the second operand on NaN, which is the contract as is. */
    friend inline batch min(const batch a, const batch b) noexcept { return {_mm256_min_ps(a.v, b.v)}; }
    friend inline batch max(const batch a, const batch b) noexcept { return {_mm256_max_ps(a.v, b.v)}; }
    friend inline batch select(const mask m, const batch a, const batch b) noexcept { return {_mm256_blendv_ps(b.v, a.v, m.m)}; }
    friend inline float reduce_min(const batch a) noexcept {
        __m128 r = _mm_min_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
        r = _mm_min_ps(r, _mm_movehl_ps(r, r));
        return _mm_cvtss_f32(_mm_min_ss(r, _mm_movehdup_ps(r)));
    }
    friend inline float reduce_max(const batch a) noexcept {
        __m128 r = _mm_max_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
        r = _mm_max_ps(r, _mm_movehl_ps(r, r));
        return _mm_cvtss_f32(_mm_max_ss(r, _mm_movehdup_ps(r)));
    }
};

/* ============================================================
 * AVX2: 4 x double
 * ============================================================ */

template <> struct batch_mask<double, avx2_isa> {
    __m256d m;

    friend inline batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {_mm256_and_pd(a.m, b.m)}; }
    friend inline batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {_mm256_or_pd(a.m, b.m)}; }
    friend inline batch_mask operator~(const batch_mask a) noexcept { return {_mm256_xor_pd(a.m, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)))}; }
    inline std::uint32_t bits() const noexcept { return static_cast<std::uint32_t>(_mm256_movemask_pd(m)); }
};

template <> struct batch<double, avx2_isa> {
    using value_type = double;
    using mask = batch_mask<double, avx2_isa>;
    static constexpr std::size_t width = 4;

    __m256d v;

    /* memcpy for the same reason as the float batch. */
    static inline batch load(const double* p) noexcept {
        batch r;
        std::memcpy(&r.v, p, sizeof r.v);
        return r;
    }
    static inline batch broadcast(const double s) noexcept { return {_mm256_set1_pd(s)}; }
    static inline mask first_lanes(const std::size_t k) noexcept {
        return {_mm256_cmp_pd(_mm256_setr_pd(0, 1, 2, 3), _mm256_set1_pd(static_cast<double>(k)), _CMP_LT_OQ)};
    }
    inline void store(double* p) const noexcept { std::memcpy(p, &v, sizeof v); }

    friend inline batch operator-(const batch a, const batch b) noexcept { return {_mm256_sub_pd(a.v, b.v)}; }
    friend inline batch operator*(const batch a, const batch b) noexcept { return {_mm256_mul_pd(a.v, b.v)}; }
    friend inline mask operator<=(const batch a, const batch b) noexcept { return {_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)}; }
    friend inline mask operator<(const batch a, const batch b) noexcept { return {_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)}; }
    friend inline batch min(const batch a, const batch b) noexcept { return {_mm256_min_pd(a.v, b.v)}; }
    friend inline batch max(const batch a, const batch b) noexcept { return {_mm256_max_pd(a.v, b.v)}; }
    friend inline batch select(const mask m, const batch a, const batch b) noexcept { return {_mm256_blendv_pd(b.v, a.v, m.m)}; }
    friend inline double reduce_min(const batch a) noexcept {
        const __m128d r{_mm_min_pd(_mm256_castpd256_pd128(a.v), _mm256_extractf128_pd(a.v, 1))};
        return _mm_cvtsd_f64(_mm_min_sd(r, _mm_unpackhi_pd(r, r)));
    }
    friend inline double reduce_max(const batch a) noexcept {
        const __m128d r{_mm_max_pd(_mm256_castpd256_pd128(a.v), _mm256_extractf128_pd(a.v, 1))};
        return _mm_cvtsd_f64(_mm_max_sd(r, _mm_unpackhi_pd(r, r)));
    }
};

/* ============================================================
 * AVX2: 8 x int32. No arithmetic: nothing integral needs it yet.
 * ============================================================ */

template <> struct batch_mask<std::int32_t, avx2_isa> {
    __m256i m;

    friend inline batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {_mm256_and_si256(a.m, b.m)}; }
    friend inline batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {_mm256_or_si256(a.m, b.m)}; }
    friend inline batch_mask operator~(const batch_mask a) noexcept { return {_mm256_xor_si256(a.m, _mm256_set1_epi32(-1))}; }
    inline std::uint32_t bits() const noexcept { return static_cast<std::uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(m))); }
};

template <> struct batch<std::int32_t, avx2_isa> {
    using value_type = std::int32_t;
    using mask = batch_mask<std::int32_t, avx2_isa>;
    static constexpr std::size_t width = 8;

    __m256i v;

    static inline batch load(const std::int32_t* p) noexcept {
        batch r;
        std::memcpy(&r.v, p, sizeof r.v);
        return r;
    }
    static inline batch broadcast(const std::int32_t s) noexcept { return {_mm256_set1_epi32(s)}; }
    static inline mask first_lanes(const std::size_t k) noexcept {
        return {_mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<std::int32_t>(k)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7))};
    }
    inline void store(std::int32_t* p) const noexcept { std::memcpy(p, &v, sizeof v); }

    /* AVX2 compares integers only for > and ==. */
    friend inline mask operator<=(const batch a, const batch b) noexcept { return ~mask{_mm256_cmpgt_epi32(a.v, b.v)}; }
    friend inline mask operator<(const batch a, const batch b) noexcept { return {_mm256_cmpgt_epi32(b.v, a.v)}; }
    friend inline batch min(const batch a, const batch b) noexcept { return {_mm256_min_epi32(a.v, b.v)}; }
    friend inline batch max(const batch a, const batch b) noexcept { return {_mm256_max_epi32(a.v, b.v)}; }
    friend inline batch select(const mask m, const batch a, const batch b) noexcept { return {_mm256_blendv_epi8(b.v, a.v, m.m)}; }
    friend inline std::int32_t reduce_min(const batch a) noexcept {
        __m128i r{_mm_min_epi32(_mm256_castsi256_si128(a.v), _mm256_extracti128_si256(a.v, 1))};
        r = _mm_min_epi32(r, _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtsi128_si32(_mm_min_epi32(r, _mm_shuffle_epi32(r, _MM_SHUFFLE(2, 3, 0, 1))));
    }
    friend inline std::int32_t reduce_max(const batch a) noexcept {
        __m128i r{_mm_max_epi32(_mm256_castsi256_si128(a.v), _mm256_extracti128_si256(a.v, 1))};
        r = _mm_max_epi32(r, _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtsi128_si32(_mm_max_epi32(r, _mm_shuffle_epi32(r, _MM_SHUFFLE(2, 3, 0, 1))));
    }
};

#endif

/* ============================================================
 * NEON: 4 x float
 * ============================================================ */

#if defined(SGL_BATCH_NEON)

template <> struct batch_mask<float, neon_isa> {
    uint32x4_t m;

    friend inline batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {vandq_u32(a.m, b.m)}; }
    friend inline batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {vorrq_u32(a.m, b.m)}; }
    friend inline batch_mask operator~(const batch_mask a) noexcept { return {vmvnq_u32(a.m)}; }
    inline std::uint32_t bits() const noexcept {
        const uint32x4_t weights = {1, 2, 4, 8};
        return vaddvq_u32(vandq_u32(m, weights));
    }
};

template <> struct batch<float, neon_isa> {
    using value_type = float;
    using mask = batch_mask<float, neon_isa>;
    static constexpr std::size_t width = 4;

    float32x4_t v;

    static inline batch load(const float* p) noexcept { return {vld1q_f32(p)}; }
    static inline batch broadcast(const float s) noexcept { return {vdupq_n_f32(s)}; }
    static inline mask first_lanes(const std::size_t k) noexcept {
        const float32x4_t iota = {0, 1, 2, 3};
        return {vcltq_f32(iota, vdupq_n_f32(static_cast<float>(k)))};
    }
    inline void store(float* p) const noexcept { vst1q_f32(p, v); }

    friend inline batch operator-(const batch a, const batch b) noexcept { return {vsubq_f32(a.v, b.v)}; }
    friend inline batch operator*(const batch a, const batch b) noexcept { return {vmulq_f32(a.v, b.v)}; }
    friend inline mask operator<=(const batch a, const batch b) noexcept { return {vcleq_f32(a.v, b.v)}; }
    friend inline mask operator<(const batch a, const batch b) noexcept { return {vcltq_f32(a.v, b.v)}; }
    /* Not vminq/vmaxq: FMIN/FMAX propagate NaN, and vminnm returns whichever side is a number. */
    friend inline batch min(const batch a, const batch b) noexcept { return {vbslq_f32(vcltq_f32(a.v, b.v), a.v, b.v)}; }
    friend inline batch max(const batch a, const batch b) noexcept { return {vbslq_f32(vcgtq_f32(a.v, b.v), a.v, b.v)}; }
    friend inline batch select(const mask m, const batch a, const batch b) noexcept { return {vbslq_f32(m.m, a.v, b.v)}; }
    friend inline float reduce_min(const batch a) noexcept { return vminvq_f32(a.v); }
    friend inline float reduce_max(const batch a) noexcept { return vmaxvq_f32(a.v); }
};

/* ============================================================
 * NEON: 2 x double
 * ============================================================ */

template <> struct batch_mask<double, neon_isa> {
    uint64x2_t m;

    friend inline batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {vandq_u64(a.m, b.m)}; }
    friend inline batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {vorrq_u64(a.m, b.m)}; }
    friend inline batch_mask operator~(const batch_mask a) noexcept { return {veorq_u64(a.m, vdupq_n_u64(~std::uint64_t{}))}; }
    inline std::uint32_t bits() const noexcept { return static_cast<std::uint32_t>((vgetq_lane_u64(m, 0) & 1u) | ((vgetq_lane_u64(m, 1) & 1u) << 1)); }
};

template <> struct batch<double, neon_isa> {
    using value_type = double;
    using mask = batch_mask<double, neon_isa>;
    static constexpr std::size_t width = 2;

    float64x2_t v;

    static inline batch load(const double* p) noexcept { return {vld1q_f64(p)}; }
    static inline batch broadcast(const double s) noexcept { return {vdupq_n_f64(s)}; }
    static inline mask first_lanes(const std::size_t k) noexcept {
        const float64x2_t iota = {0, 1};
        return {vcltq_f64(iota, vdupq_n_f64(static_cast<double>(k)))};
    }
    inline void store(double* p) const noexcept { vst1q_f64(p, v); }

    friend inline batch operator-(const batch a, const batch b) noexcept { return {vsubq_f64(a.v, b.v)}; }
    friend inline batch operator*(const batch a, const batch b) noexcept { return {vmulq_f64(a.v, b.v)}; }
    friend inline mask operator<=(const batch a, const batch b) noexcept { return {vcleq_f64(a.v, b.v)}; }
    friend inline mask operator<(const batch a, const batch b) noexcept { return {vcltq_f64(a.v, b.v)}; }
    /* bsl for the NaN contract, as in the float batch. */
    friend inline batch min(const batch a, const batch b) noexcept { return {vbslq_f64(vcltq_f64(a.v, b.v), a.v, b.v)}; }
    friend inline batch max(const batch a, const batch b) noexcept { return {vbslq_f64(vcgtq_f64(a.v, b.v), a.v, b.v)}; }
    friend inline batch select(const mask m, const batch a, const batch b) noexcept { return {vbslq_f64(m.m, a.v, b.v)}; }
    friend inline double reduce_min(const batch a) noexcept { return vminvq_f64(a.v); }
    friend inline double reduce_max(const batch a) noexcept { return vmaxvq_f64(a.v); }
};

/* ============================================================
 * NEON: 4 x int32. No arithmetic: nothing integral needs it yet.
 * ============================================================ */

template <> struct batch_mask<std::int32_t, neon_isa> {
    uint32x4_t m;

    friend inline batch_mask operator&(const batch_mask a, const batch_mask b) noexcept { return {vandq_u32(a.m, b.m)}; }
    friend inline batch_mask operator|(const batch_mask a, const batch_mask b) noexcept { return {vorrq_u32(a.m, b.m)}; }
    friend inline batch_mask operator~(const batch_mask a) noexcept { return {vmvnq_u32(a.m)}; }
    inline std::uint32_t bits() const noexcept {
        const uint32x4_t weights = {1, 2, 4, 8};
        return vaddvq_u32(vandq_u32(m, weights));
    }
};

template <> struct batch<std::int32_t, neon_isa> {
    using value_type = std::int32_t;
    using mask = batch_mask<std::int32_t, neon_isa>;
    static constexpr std::size_t width = 4;

    int32x4_t v;

    static inline batch load(const std::int32_t* p) noexcept { return {vld1q_s32(p)}; }
    static inline batch broadcast(const std::int32_t s) noexcept { return {vdupq_n_s32(s)}; }
    static inline mask first_lanes(const std::size_t k) noexcept {
        const int32x4_t iota = {0, 1, 2, 3};
        return {vcltq_s32(iota, vdupq_n_s32(static_cast<std::int32_t>(k)))};
    }
    inline void store(std::int32_t* p) const noexcept { vst1q_s32(p, v); }

    friend inline mask operator<=(const batch a, const batch b) noexcept { return {vcleq_s32(a.v, b.v)}; }
    friend inline mask operator<(const batch a, const batch b) noexcept { return {vcltq_s32(a.v, b.v)}; }
    friend inline batch min(const batch a, const batch b) noexcept { return {vminq_s32(a.v, b.v)}; }
    friend inline batch max(const batch a, const batch b) noexcept { return {vmaxq_s32(a.v, b.v)}; }
    friend inline batch select(const mask m, const batch a, const batch b) noexcept { return {vbslq_s32(m.m, a.v, b.v)}; }
    friend inline std::int32_t reduce_min(const batch a) noexcept { return vminvq_s32(a.v); }
    friend inline std::int32_t reduce_max(const batch a) noexcept { return vmaxvq_s32(a.v); }
};

#endif

/* ============================================================
 * Index compaction: store(out, first, m) writes first + k for every set bit k of the
 * width-bit mask m, packed at out, and returns how many. It always stores a full width
 * of entries, so out needs room for width past the current count. Its width is that of
 * the 32-bit batches, independent of the element type being queried.
 * ============================================================ */

template <class Isa> struct index_compactor;

template <> struct index_compactor<scalar_isa> {
    static constexpr std::size_t width{1};

    static constexpr std::size_t store(std::uint32_t* out, const std::uint32_t first, const std::uint32_t m) noexcept {
        *out = first;
        return m;
    }
};

#if defined(SGL_BATCH_AVX2)

/* Entry m packs the positions of the set bits of m, one per byte, lowest first. */
inline constexpr std::array<std::uint64_t, 256> set_bit_positions8{[] {
    std::array<std::uint64_t, 256> table{};
    for (std::uint32_t m{}; m < 256; ++m) {
        std::uint32_t k{};
        for (std::uint32_t b{}; b < 8; ++b) {
            if ((m >> b) & 1u) {
                table[m] |= std::uint64_t{b} << (8 * k++);
            }
        }
    }
    return table;
}()};

template <> struct index_compactor<avx2_isa> {
    static constexpr std::size_t width{8};

    static inline std::size_t store(std::uint32_t* out, const std::uint32_t first, const std::uint32_t m) noexcept {
        const __m256i positions{_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(set_bit_positions8[m])))};
        const __m256i indices{_mm256_add_epi32(positions, _mm256_set1_epi32(static_cast<int>(first)))};
        std::memcpy(out, &indices, sizeof indices);
        return static_cast<std::size_t>(std::popcount(m));
    }
};

#endif

#if defined(SGL_BATCH_NEON)

/* Entry m lists the positions of the set bits of m, lowest first. */
inline constexpr std::array<std::array<std::uint32_t, 4>, 16> set_bit_positions4{[] {
    std::array<std::array<std::uint32_t, 4>, 16> table{};
    for (std::uint32_t m{}; m < 16; ++m) {
        std::uint32_t k{};
        for (std::uint32_t b{}; b < 4; ++b) {
            if ((m >> b) & 1u) {
                table[m][k++] = b;
            }
        }
    }
    return table;
}()};

template <> struct index_compactor<neon_isa> {
    static constexpr std::size_t width{4};

    static inline std::size_t store(std::uint32_t* out, const std::uint32_t first, const std::uint32_t m) noexcept {
        vst1q_u32(out, vaddq_u32(vld1q_u32(set_bit_positions4[m].data()), vdupq_n_u32(first)));
        return static_cast<std::size_t>(std::popcount(m));
    }
};

#endif

} // namespace sgl::detail

#undef SGL_BATCH_AVX2
#undef SGL_BATCH_NEON
