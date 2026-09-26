/**
 * @file soa_types_test.cpp
 * @brief The SoA kernels for every element type (float, double, int32) against scalar references.
 *
 * Mirrors soa_test.cpp in shape: each check runs on an unpadded view, an owning buffer, and a
 * padded view whose slack is poisoned with values chosen to leak into a wrong result.
 */
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

import sgl;

namespace {

constexpr std::array<std::size_t, 10> sizes{0, 1, 3, 7, 8, 9, 63, 64, 65, 1001};

template <class T> using axes = std::array<std::vector<T>, 3>;
template <class T> using point = std::array<T, 3>;

/* Coordinates span [0, 10) for floating point and [0, 1000) for int32, so the same query
 * shapes select similar fractions. */
template <class T> constexpr T scale{std::is_floating_point_v<T> ? T{1} : T{100}};

template <class T> T at_scale(const double x) {
    return static_cast<T>(x * static_cast<double>(scale<T>));
}

template <class T> const sgl::soa::aabb<T, 3> query{{at_scale<T>(2), at_scale<T>(3), at_scale<T>(1)}, {at_scale<T>(7), at_scale<T>(8), at_scale<T>(6)}};
template <class T> const point<T> query_center{at_scale<T>(4.5), at_scale<T>(5.5), at_scale<T>(3.5)};
template <class T> constexpr point<T> far_point{std::numeric_limits<T>::max() / 2, std::numeric_limits<T>::max() / 2, std::numeric_limits<T>::max() / 2};
template <class T> constexpr point<T> lowest_point{std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()};
template <class T> constexpr point<T> highest_point{std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};

template <class T> T random_coord(std::mt19937& rng, const double hi) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::uniform_real_distribution<T>{T{0}, static_cast<T>(hi)}(rng);
    } else {
        return std::uniform_int_distribution<T>{0, static_cast<T>(hi * 100) - 1}(rng);
    }
}

template <class T> axes<T> random_points(const std::size_t n, const unsigned seed) {
    std::mt19937 rng{seed};
    axes<T> a;
    for (auto& axis : a) {
        axis.resize(n);
        for (auto& x : axis) {
            x = random_coord<T>(rng, 10.0);
        }
    }
    return a;
}

template <class T> struct box_axes {
    axes<T> lo;
    axes<T> hi;
};

template <class T> box_axes<T> random_boxes(const std::size_t n, const unsigned seed) {
    std::mt19937 rng{seed};
    box_axes<T> b;
    for (std::size_t a{}; a < 3; ++a) {
        b.lo[a].resize(n);
        b.hi[a].resize(n);
        for (std::size_t i{}; i < n; ++i) {
            b.lo[a][i] = random_coord<T>(rng, 10.0);
            b.hi[a][i] = static_cast<T>(b.lo[a][i] + random_coord<T>(rng, 2.0));
        }
    }
    return b;
}

template <class T> point<T> at(const axes<T>& a, const std::size_t i) {
    return {a[0][i], a[1][i], a[2][i]};
}

template <class T> axes<T> poisoned(const axes<T>& a, const point<T>& poison) {
    axes<T> out{a};
    for (std::size_t k{}; k < 3; ++k) {
        out[k].resize((a[k].size() + 63) / 64 * 64, poison[k]);
    }
    return out;
}

template <class T> sgl::soa::points<T, 3, sgl::soa::block_size> padded_view(const axes<T>& a, const std::size_t n) {
    return {{a[0].data(), a[1].data(), a[2].data()}, n};
}

template <class T, class F> void for_each_layout(const axes<T>& a, const point<T>& poison, F&& check) {
    const std::size_t n{a[0].size()};
    check(sgl::soa::make_points(a[0], a[1], a[2]), "view");

    sgl::soa::point_buffer<T, 3> buf;
    for (std::size_t i{}; i < n; ++i) {
        buf.push_back(at(a, i));
    }
    check(buf, "buffer");

    const auto p{poisoned(a, poison)};
    check(padded_view(p, n), "padded");
}

template <class T, class F> void for_each_layout(const box_axes<T>& b, const point<T>& poison_lo, const point<T>& poison_hi, F&& check) {
    const std::size_t n{b.lo[0].size()};
    check(sgl::soa::make_boxes(sgl::soa::make_points(b.lo[0], b.lo[1], b.lo[2]), sgl::soa::make_points(b.hi[0], b.hi[1], b.hi[2])), "view");

    sgl::soa::box_buffer<T, 3> buf;
    for (std::size_t i{}; i < n; ++i) {
        buf.push_back(sgl::soa::aabb<T, 3>{at(b.lo, i), at(b.hi, i)});
    }
    check(buf, "buffer");

    const auto lo{poisoned(b.lo, poison_lo)};
    const auto hi{poisoned(b.hi, poison_hi)};
    check(sgl::soa::make_boxes(padded_view(lo, n), padded_view(hi, n)), "padded");
}

template <class T> bool ref_inside(const point<T>& p, const point<T>& lo, const point<T>& hi) {
    return lo[0] <= p[0] && p[0] <= hi[0] && lo[1] <= p[1] && p[1] <= hi[1] && lo[2] <= p[2] && p[2] <= hi[2];
}

template <class T> bool ref_overlaps(const point<T>& lo, const point<T>& hi, const sgl::soa::aabb<T, 3>& q) {
    return lo[0] <= q.max[0] && q.min[0] <= hi[0] && lo[1] <= q.max[1] && q.min[1] <= hi[1] && lo[2] <= q.max[2] && q.min[2] <= hi[2];
}

/* bounds returns box3d for float and aabb otherwise; compare either as arrays. */
template <class T> std::array<point<T>, 2> corners(const sgl::box3d& b) {
    return {{{b.min.x, b.min.y, b.min.z}, {b.max.x, b.max.y, b.max.z}}};
}
template <class T> std::array<point<T>, 2> corners(const sgl::soa::aabb<T, 3>& b) {
    return {b.min, b.max};
}

template <class T> struct ref_hit {
    bool hit;
    T t;
};

/* The slab test with parallel axes handled explicitly, in T. */
template <class T> ref_hit<T> ref_ray(const point<T>& lo, const point<T>& hi, const point<T>& o, const point<T>& d, const T t_min, const T t_max) {
    T t_near{t_min};
    T t_far{t_max};
    bool inside{true};
    for (std::size_t a{}; a < 3; ++a) {
        const T inv{T{1} / d[a]};
        if (std::isinf(inv)) {
            inside = inside && lo[a] <= o[a] && o[a] <= hi[a];
            continue;
        }
        const T t1{((std::signbit(inv) ? hi[a] : lo[a]) - o[a]) * inv};
        const T t2{((std::signbit(inv) ? lo[a] : hi[a]) - o[a]) * inv};
        t_near = t1 > t_near ? t1 : t_near;
        t_far = t2 < t_far ? t2 : t_far;
    }
    const bool hit{inside && t_near <= t_far};
    return {hit, hit ? t_near : std::numeric_limits<T>::infinity()};
}

std::string trace(const std::string& layout, const std::size_t n) {
    return layout + " n=" + std::to_string(n);
}

} // namespace

template <class T> class SoaTyped : public ::testing::Test {};
using element_types = ::testing::Types<float, double, std::int32_t>;
TYPED_TEST_SUITE(SoaTyped, element_types);

TYPED_TEST(SoaTyped, InsideMatchesReference) {
    using T = TypeParam;
    const auto& q{query<T>};
    for (const auto n : sizes) {
        const auto a{random_points<T>(n, 3u + static_cast<unsigned>(n))};
        std::vector<std::uint32_t> expected;
        for (std::size_t i{}; i < n; ++i) {
            if (ref_inside(at(a, i), q.min, q.max)) {
                expected.push_back(static_cast<std::uint32_t>(i));
            }
        }
        const std::size_t outside{n - expected.size()};

        /* Poison inside the query: a leaked tail lane would add a match. */
        for_each_layout(a, query_center<T>, [&](const auto& pts, const std::string& layout) {
            SCOPED_TRACE(trace(layout, n));
            const auto pred{sgl::soa::inside(q)};

            std::vector<std::uint64_t> bits((n + 63) / 64);
            sgl::soa::mask(pts, pred, bits);
            std::vector<std::uint32_t> from_bits;
            for (std::size_t i{}; i < bits.size() * 64; ++i) {
                if ((bits[i / 64] >> (i % 64)) & 1u) {
                    from_bits.push_back(static_cast<std::uint32_t>(i));
                }
            }
            EXPECT_EQ(from_bits, expected);

            std::vector<std::uint32_t> indices(n);
            indices.resize(sgl::soa::match_indices(pts, pred, indices));
            EXPECT_EQ(indices, expected);

            EXPECT_EQ(sgl::soa::count(pts, pred), expected.size());
            EXPECT_EQ(sgl::soa::any(pts, pred), !expected.empty());
        });

        /* Poison far away: a leaked tail lane would pass the negation. */
        for_each_layout(a, far_point<T>, [&](const auto& pts, const std::string& layout) {
            SCOPED_TRACE(trace(layout, n));
            EXPECT_EQ(sgl::soa::count(pts, !sgl::soa::inside(q)), outside);
        });
    }
}

TYPED_TEST(SoaTyped, OverlapsAndContainsMatchReference) {
    using T = TypeParam;
    const auto& q{query<T>};
    const auto& probe{query_center<T>};
    for (const auto n : sizes) {
        const auto b{random_boxes<T>(n, 17u + static_cast<unsigned>(n))};
        std::size_t over{};
        std::size_t cont{};
        for (std::size_t i{}; i < n; ++i) {
            over += ref_overlaps(at(b.lo, i), at(b.hi, i), q) ? 1u : 0u;
            cont += ref_inside(probe, at(b.lo, i), at(b.hi, i)) ? 1u : 0u;
        }
        /* Poison: a box covering the whole range, which both predicates would match. */
        for_each_layout(b, lowest_point<T>, highest_point<T>, [&](const auto& view, const std::string& layout) {
            SCOPED_TRACE(trace(layout, n));
            EXPECT_EQ(sgl::soa::count(view, sgl::soa::overlaps(q)), over);
            EXPECT_EQ(sgl::soa::count(view, sgl::soa::contains(probe)), cont);
        });
    }
}

TYPED_TEST(SoaTyped, BoundsMatchReference) {
    using T = TypeParam;
    for (const auto n : sizes) {
        if (!n) {
            continue;
        }
        const auto a{random_points<T>(n, 29u + static_cast<unsigned>(n))};
        point<T> lo{highest_point<T>};
        point<T> hi{lowest_point<T>};
        for (std::size_t k{}; k < 3; ++k) {
            for (const T x : a[k]) {
                lo[k] = x < lo[k] ? x : lo[k];
                hi[k] = x > hi[k] ? x : hi[k];
            }
        }
        /* Poison at the lowest value, so a leaked lane would lower the min. */
        for_each_layout(a, lowest_point<T>, [&](const auto& pts, const std::string& layout) {
            SCOPED_TRACE(trace(layout, n));
            const auto c{corners<T>(sgl::soa::bounds(pts))};
            EXPECT_EQ(c[0], lo);
            EXPECT_EQ(c[1], hi);
        });
    }
}

TYPED_TEST(SoaTyped, BoundsOfEmptyIsTheIdentity) {
    using T = TypeParam;
    const std::vector<T> none;
    const auto c{corners<T>(sgl::soa::bounds(sgl::soa::make_points(none, none, none)))};
    constexpr T big{std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max()};
    constexpr T small{std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest()};
    EXPECT_EQ(c[0], (point<T>{big, big, big}));
    EXPECT_EQ(c[1], (point<T>{small, small, small}));
}

TYPED_TEST(SoaTyped, RayMatchesReference) {
    using T = TypeParam;
    if constexpr (!std::floating_point<T>) {
        GTEST_SKIP() << "rays are floating point only";
    } else {
        constexpr T inf{std::numeric_limits<T>::infinity()};
        const std::array<point<T>, 4> dirs{{{1, T(0.7), T(0.3)}, {T(-0.4), 1, T(-0.2)}, {0, 1, T(0.5)}, {T(-0.0), T(0.3), 0}}};
        const point<T> o{T(0.5), T(0.25), T(9)};
        for (const auto n : sizes) {
            const auto b{random_boxes<T>(n, 41u + static_cast<unsigned>(n))};
            for (const auto& d : dirs) {
                const auto ray{sgl::soa::hit_by(o, d)};
                std::vector<T> expected(n);
                std::size_t hits{};
                for (std::size_t i{}; i < n; ++i) {
                    const auto r{ref_ray(at(b.lo, i), at(b.hi, i), o, d, T{0}, inf)};
                    expected[i] = r.t;
                    hits += r.hit ? 1u : 0u;
                }
                for_each_layout(b, lowest_point<T>, highest_point<T>, [&](const auto& view, const std::string& layout) {
                    SCOPED_TRACE(trace(layout, n));
                    std::vector<T> t(n);
                    EXPECT_EQ(sgl::soa::hit_distances(view, ray, t), hits);
                    EXPECT_EQ(t, expected);
                    EXPECT_EQ(sgl::soa::count(view, ray), hits);
                });
            }
        }
    }
}

TYPED_TEST(SoaTyped, RayAlongBoxEdgesHits) {
    using T = TypeParam;
    if constexpr (!std::floating_point<T>) {
        GTEST_SKIP() << "rays are floating point only";
    } else {
        const std::vector<T> lo{0};
        const std::vector<T> hi{1};
        const auto view{sgl::soa::make_boxes(sgl::soa::make_points(lo, lo), sgl::soa::make_points(hi, hi))};
        for (const T y : {T{0}, T{1}}) {
            std::vector<T> t(1);
            EXPECT_EQ(sgl::soa::hit_distances(view, sgl::soa::hit_by(std::array<T, 2>{-1, y}, std::array<T, 2>{1, 0}), t), 1u) << "y=" << y;
            EXPECT_EQ(t[0], T{1});
        }
        EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(std::array<T, 2>{-1, T(-0.0001)}, std::array<T, 2>{1, 0})), 0u);
    }
}

TYPED_TEST(SoaTyped, NaNNeverMatchesAndBoundsSkipIt) {
    using T = TypeParam;
    if constexpr (!std::floating_point<T>) {
        GTEST_SKIP() << "no NaN in integers";
    } else {
        constexpr T qnan{std::numeric_limits<T>::quiet_NaN()};
        std::vector<T> x(20, T{4});
        std::vector<T> y(20, T{5});
        std::vector<T> z(20, T{3});
        x[3] = qnan;
        y[11] = qnan;
        x[7] = T{-2};
        const auto pts{sgl::soa::make_points(x, y, z)};
        EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(query<T>)), 17u);
        const auto c{corners<T>(sgl::soa::bounds(pts))};
        EXPECT_EQ(c[0], (point<T>{-2, 5, 3}));
        EXPECT_EQ(c[1], (point<T>{4, 5, 3}));
    }
}

TEST(SoaInt32, ExtremeValuesAndCoreVocabulary) {
    constexpr std::int32_t lo{std::numeric_limits<std::int32_t>::lowest()};
    constexpr std::int32_t hi{std::numeric_limits<std::int32_t>::max()};

    sgl::soa::point_buffer<std::int32_t, 3> pts;
    pts.push_back(sgl::ivec3{lo, 0, 0});
    pts.push_back(sgl::ivec3{hi, 0, 0});
    pts.push_back(sgl::ivec3{0, lo, hi});
    pts.push_back(sgl::ivec3{-1, 1, 0});

    const auto b{sgl::soa::bounds(pts)};
    EXPECT_EQ(b.min, (std::array<std::int32_t, 3>{lo, lo, 0}));
    EXPECT_EQ(b.max, (std::array<std::int32_t, 3>{hi, 1, hi}));

    EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(sgl::soa::aabb<std::int32_t, 3>{{lo, lo, lo}, {hi, hi, hi}})), 4u);
    EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(sgl::soa::aabb<std::int32_t, 3>{{hi, 0, 0}, {hi, 0, 0}})), 1u);
    EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(sgl::soa::aabb<std::int32_t, 3>{{-1, -1, -1}, {0, 1, 0}})), 1u);

    sgl::soa::box_buffer<std::int32_t, 2> cells;
    cells.push_back(sgl::soa::aabb<std::int32_t, 2>{{0, 0}, {9, 9}});
    cells.push_back(sgl::soa::aabb<std::int32_t, 2>{{10, 0}, {19, 9}});
    EXPECT_EQ(sgl::soa::count(cells, sgl::soa::contains(sgl::ivec2{10, 5})), 1u);
    EXPECT_EQ(sgl::soa::count(cells, sgl::soa::overlaps(sgl::soa::aabb<std::int32_t, 2>{{9, 9}, {10, 9}})), 2u);
}

TEST(SoaDouble, PrecisionBeyondFloat) {
    /* Two points 1e-9 apart near 1e6: float cannot tell them apart, double can. */
    const std::vector<double> x{1e6, 1e6 + 1e-9};
    const std::vector<double> zero{0.0, 0.0};
    const auto pts{sgl::soa::make_points(x, zero)};
    EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(sgl::soa::aabb<double, 2>{{1e6 + 5e-10, -1.0}, {2e6, 1.0}})), 1u);
}
