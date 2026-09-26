/**
 * @file soa_test.cpp
 * @brief Unit tests for the SoA batch queries, checked against scalar references.
 */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

import sgl;

namespace {

constexpr float inf{std::numeric_limits<float>::infinity()};
constexpr float qnan{std::numeric_limits<float>::quiet_NaN()};

/* Sizes that exercise every path of the drivers: scalar-only tails, partial and full
 * native batches, exact and off-by-one blocks of 64. */
constexpr std::array<std::size_t, 12> sizes{0, 1, 3, 4, 7, 8, 9, 63, 64, 65, 130, 1001};

struct cloud {
    std::array<std::vector<float>, 3> axis;
    [[nodiscard]] sgl::soa::points3f view() const { return sgl::soa::make_points(axis[0], axis[1], axis[2]); }
};

struct box_set {
    std::array<std::vector<float>, 3> lo;
    std::array<std::vector<float>, 3> hi;
    [[nodiscard]] sgl::soa::boxes3f view() const {
        return sgl::soa::make_boxes(sgl::soa::make_points(lo[0], lo[1], lo[2]), sgl::soa::make_points(hi[0], hi[1], hi[2]));
    }
};

cloud random_cloud(const std::size_t n, const unsigned seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> u{0.0f, 10.0f};
    cloud c;
    for (auto& a : c.axis) {
        a.resize(n);
        for (auto& x : a) {
            x = u(rng);
        }
    }
    return c;
}

box_set random_boxes(const std::size_t n, const unsigned seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> pos{0.0f, 10.0f};
    std::uniform_real_distribution<float> ext{0.0f, 2.0f};
    box_set b;
    for (std::size_t a{}; a < 3; ++a) {
        b.lo[a].resize(n);
        b.hi[a].resize(n);
        for (std::size_t i{}; i < n; ++i) {
            b.lo[a][i] = pos(rng);
            b.hi[a][i] = b.lo[a][i] + ext(rng);
        }
    }
    return b;
}

std::array<float, 3> at(const std::array<std::vector<float>, 3>& axis, const std::size_t i) {
    return {axis[0][i], axis[1][i], axis[2][i]};
}

bool ref_inside(const std::array<float, 3>& p, const sgl::box3d& b) {
    return b.min.x <= p[0] && p[0] <= b.max.x && b.min.y <= p[1] && p[1] <= b.max.y && b.min.z <= p[2] && p[2] <= b.max.z;
}

bool ref_overlaps(const std::array<float, 3>& lo, const std::array<float, 3>& hi, const sgl::box3d& q) {
    return lo[0] <= q.max.x && q.min.x <= hi[0] && lo[1] <= q.max.y && q.min.y <= hi[1] && lo[2] <= q.max.z && q.min.z <= hi[2];
}

/* Same slab formulation as the library, written out per box. */
struct ref_hit {
    bool hit;
    float t;
};

ref_hit ref_ray(const std::array<float, 3>& lo, const std::array<float, 3>& hi, const std::array<float, 3>& o, const std::array<float, 3>& d, const float t_min,
    const float t_max) {
    float t_near{t_min};
    float t_far{t_max};
    bool inside{true};
    for (std::size_t a{}; a < 3; ++a) {
        const float inv{1.0f / d[a]};
        if (std::isinf(inv)) {
            inside = inside && lo[a] <= o[a] && o[a] <= hi[a];
            continue;
        }
        const float t1{((std::signbit(inv) ? hi[a] : lo[a]) - o[a]) * inv};
        const float t2{((std::signbit(inv) ? lo[a] : hi[a]) - o[a]) * inv};
        t_near = t1 > t_near ? t1 : t_near;
        t_far = t2 < t_far ? t2 : t_far;
    }
    const bool hit{inside && t_near <= t_far};
    return {hit, hit ? t_near : inf};
}

const sgl::box3d query_box{{2.0f, 3.0f, 1.0f}, {7.0f, 8.0f, 6.0f}};

} // namespace

/* ------------------------------------------------------------
 * Points inside a box
 * ------------------------------------------------------------ */

TEST(Soa, InsideMatchesReference) {
    for (const auto n : sizes) {
        const auto c{random_cloud(n, static_cast<unsigned>(n) + 1)};
        const auto pts{c.view()};
        const auto pred{sgl::soa::inside(query_box)};

        std::vector<std::uint64_t> bits((n + 63) / 64 + 1, ~std::uint64_t{});
        sgl::soa::mask(pts, pred, bits);

        std::vector<std::size_t> matches;
        sgl::soa::for_each_match(pts, pred, [&](const std::size_t i) { matches.push_back(i); });

        std::vector<std::size_t> expected;
        for (std::size_t i{}; i < n; ++i) {
            const bool in{ref_inside(at(c.axis, i), query_box)};
            EXPECT_EQ(((bits[i / 64] >> (i % 64)) & 1u) != 0, in) << "n=" << n << " i=" << i;
            if (in) {
                expected.push_back(i);
            }
        }
        for (std::size_t i{n}; i < (n + 63) / 64 * 64; ++i) {
            EXPECT_EQ((bits[i / 64] >> (i % 64)) & 1u, 0u) << "tail bit set, n=" << n << " i=" << i;
        }
        EXPECT_EQ(bits.back(), ~std::uint64_t{}) << "wrote past the required words, n=" << n;
        EXPECT_EQ(matches, expected) << "n=" << n;
        EXPECT_EQ(sgl::soa::count(pts, pred), expected.size()) << "n=" << n;
        EXPECT_EQ(sgl::soa::any(pts, pred), !expected.empty()) << "n=" << n;
    }
}

TEST(Soa, InsideIsClosedAndRejectsNaN) {
    const std::vector<float> x{2.0f, 7.0f, 1.999f, qnan, 4.0f};
    const std::vector<float> y{3.0f, 8.0f, 5.0f, 5.0f, qnan};
    const std::vector<float> z{1.0f, 6.0f, 3.0f, 3.0f, 3.0f};
    const auto pts{sgl::soa::make_points(x, y, z)};

    std::vector<std::size_t> hits;
    sgl::soa::for_each_match(pts, sgl::soa::inside(query_box), [&](const std::size_t i) { hits.push_back(i); });
    EXPECT_EQ(hits, (std::vector<std::size_t>{0, 1}));
}

TEST(Soa, Inside2D) {
    const std::vector<float> x{0.0f, 1.0f, 2.0f, 0.5f};
    const std::vector<float> y{0.0f, 1.0f, 0.5f, 1.5f};
    const auto pts{sgl::soa::make_points(x, y)};
    EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(sgl::box2d{{0.0f, 0.0f}, {1.0f, 1.0f}})), 2u);
}

TEST(Soa, AnyFindsOnlyTheLastElement) {
    const std::size_t n{200};
    cloud c;
    for (auto& a : c.axis) {
        a.assign(n, 100.0f);
    }
    EXPECT_FALSE(sgl::soa::any(c.view(), sgl::soa::inside(query_box)));
    c.axis[0].back() = 4.0f;
    c.axis[1].back() = 4.0f;
    c.axis[2].back() = 4.0f;
    EXPECT_TRUE(sgl::soa::any(c.view(), sgl::soa::inside(query_box)));
}

/* ------------------------------------------------------------
 * Composition
 * ------------------------------------------------------------ */

TEST(Soa, ComposedPredicatesMatchReference) {
    const sgl::box3d hole{{4.0f, 4.0f, 2.0f}, {5.0f, 6.0f, 4.0f}};
    const sgl::box3d other{{0.0f, 0.0f, 0.0f}, {1.0f, 10.0f, 10.0f}};
    for (const auto n : sizes) {
        const auto c{random_cloud(n, 77u + static_cast<unsigned>(n))};
        const auto pts{c.view()};

        std::size_t ring{};
        std::size_t either{};
        for (std::size_t i{}; i < n; ++i) {
            const auto p{at(c.axis, i)};
            ring += (ref_inside(p, query_box) && !ref_inside(p, hole)) ? 1u : 0u;
            either += (ref_inside(p, query_box) || ref_inside(p, other)) ? 1u : 0u;
        }
        EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(query_box) && !sgl::soa::inside(hole)), ring) << "n=" << n;
        EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(query_box) || sgl::soa::inside(other)), either) << "n=" << n;
    }
}

/* ------------------------------------------------------------
 * Boxes against a box or a point
 * ------------------------------------------------------------ */

TEST(Soa, OverlapsAndContainsMatchReference) {
    const std::array<float, 3> probe{4.5f, 5.0f, 3.5f};
    for (const auto n : sizes) {
        const auto b{random_boxes(n, 9u + static_cast<unsigned>(n))};
        const auto view{b.view()};

        std::vector<std::size_t> over;
        std::vector<std::size_t> cont;
        for (std::size_t i{}; i < n; ++i) {
            const auto lo{at(b.lo, i)};
            const auto hi{at(b.hi, i)};
            if (ref_overlaps(lo, hi, query_box)) {
                over.push_back(i);
            }
            if (ref_inside(probe, sgl::box3d{{lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}})) {
                cont.push_back(i);
            }
        }

        std::vector<std::size_t> got_over;
        std::vector<std::size_t> got_cont;
        sgl::soa::for_each_match(view, sgl::soa::overlaps(query_box), [&](const std::size_t i) { got_over.push_back(i); });
        sgl::soa::for_each_match(view, sgl::soa::contains(sgl::vec3{probe[0], probe[1], probe[2]}), [&](const std::size_t i) { got_cont.push_back(i); });
        EXPECT_EQ(got_over, over) << "n=" << n;
        EXPECT_EQ(got_cont, cont) << "n=" << n;
    }
}

TEST(Soa, OverlapsTouchingCountsEmptyNever) {
    /* box 0 touches the query at x = 7, box 1 misses it by 0.001, box 2 is the empty box */
    const std::vector<float> lo_x{7.0f, 7.001f, inf};
    const std::vector<float> hi_x{9.0f, 9.0f, -inf};
    const std::vector<float> lo_y{4.0f, 4.0f, inf};
    const std::vector<float> hi_y{5.0f, 5.0f, -inf};
    const std::vector<float> lo_z{2.0f, 2.0f, inf};
    const std::vector<float> hi_z{3.0f, 3.0f, -inf};
    const auto view{sgl::soa::make_boxes(sgl::soa::make_points(lo_x, lo_y, lo_z), sgl::soa::make_points(hi_x, hi_y, hi_z))};

    std::vector<std::size_t> hits;
    sgl::soa::for_each_match(view, sgl::soa::overlaps(query_box), [&](const std::size_t i) { hits.push_back(i); });
    EXPECT_EQ(hits, (std::vector<std::size_t>{0}));
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::contains(sgl::vec3{7.0f, 4.5f, 2.5f})), 1u);
}

/* ------------------------------------------------------------
 * Ray against boxes
 * ------------------------------------------------------------ */

TEST(Soa, RayMatchesReference) {
    const std::array<std::array<float, 3>, 4> dirs{{{1.0f, 0.7f, 0.3f}, {-0.4f, 1.0f, -0.2f}, {0.0f, 1.0f, 0.5f}, {0.0f, 0.0f, -1.0f}}};
    const std::array<float, 3> o{0.5f, 0.25f, 9.0f};

    for (const auto n : sizes) {
        const auto b{random_boxes(n, 31u + static_cast<unsigned>(n))};
        const auto view{b.view()};
        for (const auto& d : dirs) {
            for (const auto [t_min, t_max] : std::array<std::array<float, 2>, 2>{{{0.0f, inf}, {1.0f, 6.0f}}}) {
                const auto ray{sgl::soa::hit_by(sgl::vec3{o[0], o[1], o[2]}, sgl::vec3{d[0], d[1], d[2]}, t_min, t_max)};

                std::vector<float> t(n);
                const auto hits{sgl::soa::hit_distances(view, ray, t)};

                std::size_t expected_hits{};
                for (std::size_t i{}; i < n; ++i) {
                    const auto r{ref_ray(at(b.lo, i), at(b.hi, i), o, d, t_min, t_max)};
                    expected_hits += r.hit ? 1u : 0u;
                    EXPECT_EQ(t[i], r.t) << "n=" << n << " i=" << i;
                }
                EXPECT_EQ(hits, expected_hits);
                EXPECT_EQ(sgl::soa::count(view, ray), expected_hits);
            }
        }
    }
}

/* The cases the single-ray AVX and NEON paths used to disagree on. */
TEST(Soa, RayAlongBoxEdgesHits) {
    const std::vector<float> lo_x{0.0f};
    const std::vector<float> lo_y{0.0f};
    const std::vector<float> hi_x{1.0f};
    const std::vector<float> hi_y{1.0f};
    const auto view{sgl::soa::make_boxes(sgl::soa::make_points(lo_x, lo_y), sgl::soa::make_points(hi_x, hi_y))};

    for (const float y : {0.0f, 1.0f}) {
        const auto ray{sgl::soa::hit_by(sgl::vec2{-1.0f, y}, sgl::vec2{1.0f, 0.0f})};
        std::vector<float> t(1);
        EXPECT_EQ(sgl::soa::hit_distances(view, ray, t), 1u) << "y=" << y;
        EXPECT_FLOAT_EQ(t[0], 1.0f);
    }
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(sgl::vec2{-1.0f, 1.0001f}, sgl::vec2{1.0f, 0.0f})), 0u);
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(sgl::vec2{-1.0f, 0.5f}, sgl::vec2{1.0f, -0.0f})), 1u);
}

TEST(Soa, RayRangeAndDegenerateCases) {
    /* box 0: ahead on +x, box 1: behind, box 2: around the origin, box 3: inverted */
    const std::vector<float> lo_x{2.0f, -3.0f, -1.0f, 5.0f};
    const std::vector<float> hi_x{3.0f, -2.0f, 1.0f, 4.0f};
    const std::vector<float> lo_y{-1.0f, -1.0f, -1.0f, -1.0f};
    const std::vector<float> hi_y{1.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> lo_z{-1.0f, -1.0f, -1.0f, -1.0f};
    const std::vector<float> hi_z{1.0f, 1.0f, 1.0f, 1.0f};
    const auto view{sgl::soa::make_boxes(sgl::soa::make_points(lo_x, lo_y, lo_z), sgl::soa::make_points(hi_x, hi_y, hi_z))};
    const sgl::vec3 origin{0.0f, 0.0f, 0.0f};

    std::vector<float> t(4);
    EXPECT_EQ(sgl::soa::hit_distances(view, sgl::soa::hit_by(origin, sgl::vec3{1.0f, 0.0f, 0.0f}), t), 2u);
    EXPECT_FLOAT_EQ(t[0], 2.0f);
    EXPECT_EQ(t[1], inf);
    EXPECT_FLOAT_EQ(t[2], 0.0f) << "origin inside: entry clamps to t_min";
    EXPECT_EQ(t[3], inf) << "inverted box";

    /* a line (t_min = -inf) reaches the box behind too, with its true entry */
    EXPECT_EQ(sgl::soa::hit_distances(view, sgl::soa::hit_by(origin, sgl::vec3{1.0f, 0.0f, 0.0f}, -inf, inf), t), 3u);
    EXPECT_FLOAT_EQ(t[1], -3.0f);
    EXPECT_FLOAT_EQ(t[2], -1.0f);

    /* segment [origin, origin + 1.5 x]: stops short of box 0 */
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(origin, sgl::vec3{1.0f, 0.0f, 0.0f}, 0.0f, 1.5f)), 1u);

    /* zero direction: every axis parallel, so it is a containment test of the origin */
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(origin, sgl::vec3{0.0f, 0.0f, 0.0f})), 1u);
}

/* ------------------------------------------------------------
 * Bounds
 * ------------------------------------------------------------ */

TEST(Soa, BoundsMatchReference) {
    for (const auto n : sizes) {
        if (n == 0) {
            continue;
        }
        const auto c{random_cloud(n, 5u + static_cast<unsigned>(n))};
        std::array<float, 3> lo{inf, inf, inf};
        std::array<float, 3> hi{-inf, -inf, -inf};
        for (std::size_t a{}; a < 3; ++a) {
            for (const float x : c.axis[a]) {
                lo[a] = std::fmin(lo[a], x);
                hi[a] = std::fmax(hi[a], x);
            }
        }
        const auto b{sgl::soa::bounds(c.view())};
        EXPECT_EQ(b.min.x, lo[0]) << "n=" << n;
        EXPECT_EQ(b.min.y, lo[1]) << "n=" << n;
        EXPECT_EQ(b.min.z, lo[2]) << "n=" << n;
        EXPECT_EQ(b.max.x, hi[0]) << "n=" << n;
        EXPECT_EQ(b.max.y, hi[1]) << "n=" << n;
        EXPECT_EQ(b.max.z, hi[2]) << "n=" << n;
    }
}

TEST(Soa, BoundsOfEmptyIsInvertedAndNaNIsSkipped) {
    const std::vector<float> none;
    const auto empty{sgl::soa::bounds(sgl::soa::make_points(none, none))};
    EXPECT_EQ(empty.min.x, inf);
    EXPECT_EQ(empty.max.y, -inf);
    EXPECT_FALSE(sgl::is_valid(empty));

    std::vector<float> x(40, 1.0f);
    std::vector<float> y(40, 2.0f);
    x[0] = qnan;
    x[17] = -5.0f;
    x[39] = qnan;
    y[38] = 9.0f;
    const auto b{sgl::soa::bounds(sgl::soa::make_points(x, y))};
    EXPECT_EQ(b.min.x, -5.0f);
    EXPECT_EQ(b.max.x, 1.0f);
    EXPECT_EQ(b.min.y, 2.0f);
    EXPECT_EQ(b.max.y, 9.0f);
}
