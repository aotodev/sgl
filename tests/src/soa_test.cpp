/**
 * @file soa_test.cpp
 * @brief Unit tests for the SoA batch queries, checked against scalar references.
 *
 * Every reference test runs on three layouts of the same data: an unpadded view (scalar
 * tail), an owning buffer, and a padded view whose slack holds values chosen to match the
 * query, so any tail lane that leaks into a result shows up.
 */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory_resource>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

import sgl;

namespace {

constexpr float inf{std::numeric_limits<float>::infinity()};
constexpr float qnan{std::numeric_limits<float>::quiet_NaN()};

/* Sizes that exercise every path of the drivers: scalar-only tails, partial and full
 * native batches, exact and off-by-one blocks of 64. */
constexpr std::array<std::size_t, 12> sizes{0, 1, 3, 4, 7, 8, 9, 63, 64, 65, 130, 1001};

using axes3 = std::array<std::vector<float>, 3>;

std::size_t padded_size(const std::size_t n) {
    return (n + sgl::soa::block_size - 1) / sgl::soa::block_size * sgl::soa::block_size;
}

struct cloud {
    axes3 axis;
    std::size_t size() const { return axis[0].size(); }
    sgl::soa::points3f view() const { return sgl::soa::make_points(axis[0], axis[1], axis[2]); }
};

struct box_set {
    axes3 lo;
    axes3 hi;
    std::size_t size() const { return lo[0].size(); }
    sgl::soa::boxes3f view() const { return sgl::soa::make_boxes(sgl::soa::make_points(lo[0], lo[1], lo[2]), sgl::soa::make_points(hi[0], hi[1], hi[2])); }
};

/* Copies of the arrays extended to the padded size with a poison value per axis. */
axes3 poisoned(const axes3& axis, const std::array<float, 3>& poison) {
    axes3 out{axis};
    for (std::size_t a{}; a < 3; ++a) {
        out[a].resize(padded_size(axis[a].size()), poison[a]);
    }
    return out;
}

sgl::soa::points<float, 3, sgl::soa::block_size> padded_view(const axes3& axis, const std::size_t n) {
    return {{axis[0].data(), axis[1].data(), axis[2].data()}, n};
}

/* Runs check(range, layout_name) on the cloud as a view, as a buffer, and as a padded view
 * poisoned with `poison`. */
template <class F> void for_each_layout(const cloud& c, const std::array<float, 3>& poison, F&& check) {
    check(c.view(), "view");

    sgl::soa::point_buffer3f buf;
    for (std::size_t i{}; i < c.size(); ++i) {
        buf.push_back({c.axis[0][i], c.axis[1][i], c.axis[2][i]});
    }
    check(buf, "buffer");

    const auto p{poisoned(c.axis, poison)};
    check(padded_view(p, c.size()), "padded");
}

template <class F> void for_each_layout(const box_set& b, const std::array<float, 3>& poison_lo, const std::array<float, 3>& poison_hi, F&& check) {
    check(b.view(), "view");

    sgl::soa::box_buffer3f buf;
    for (std::size_t i{}; i < b.size(); ++i) {
        buf.push_back({b.lo[0][i], b.lo[1][i], b.lo[2][i]}, {b.hi[0][i], b.hi[1][i], b.hi[2][i]});
    }
    check(buf, "buffer");

    const auto lo{poisoned(b.lo, poison_lo)};
    const auto hi{poisoned(b.hi, poison_hi)};
    check(sgl::soa::make_boxes(padded_view(lo, b.size()), padded_view(hi, b.size())), "padded");
}

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

std::array<float, 3> at(const axes3& axis, const std::size_t i) {
    return {axis[0][i], axis[1][i], axis[2][i]};
}

bool ref_inside(const std::array<float, 3>& p, const sgl::box3d& b) {
    return b.min.x <= p[0] && p[0] <= b.max.x && b.min.y <= p[1] && p[1] <= b.max.y && b.min.z <= p[2] && p[2] <= b.max.z;
}

bool ref_overlaps(const std::array<float, 3>& lo, const std::array<float, 3>& hi, const sgl::box3d& q) {
    return lo[0] <= q.max.x && q.min.x <= hi[0] && lo[1] <= q.max.y && q.min.y <= hi[1] && lo[2] <= q.max.z && q.min.z <= hi[2];
}

/* The slab test with parallel axes handled explicitly, independent of the library's NaN
 * ordering trick. */
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
constexpr std::array<float, 3> query_center{4.5f, 5.5f, 3.5f};
constexpr std::array<float, 3> huge_lo{-1e30f, -1e30f, -1e30f};
constexpr std::array<float, 3> huge_hi{1e30f, 1e30f, 1e30f};

} // namespace

/* ------------------------------------------------------------
 * Points inside a box
 * ------------------------------------------------------------ */

TEST(Soa, InsideMatchesReference) {
    for (const auto n : sizes) {
        const auto c{random_cloud(n, static_cast<unsigned>(n) + 1)};
        const auto pred{sgl::soa::inside(query_box)};

        std::vector<std::size_t> expected;
        for (std::size_t i{}; i < n; ++i) {
            if (ref_inside(at(c.axis, i), query_box)) {
                expected.push_back(i);
            }
        }

        for_each_layout(c, query_center, [&](const auto& pts, const std::string& layout) {
            SCOPED_TRACE(layout + " n=" + std::to_string(n));

            std::vector<std::uint64_t> bits((n + 63) / 64 + 1, ~std::uint64_t{});
            sgl::soa::mask(pts, pred, bits);
            std::vector<std::size_t> from_bits;
            for (std::size_t i{}; i < (n + 63) / 64 * 64; ++i) {
                if ((bits[i / 64] >> (i % 64)) & 1u) {
                    from_bits.push_back(i);
                }
            }
            EXPECT_EQ(from_bits, expected) << "mask, including zero bits past the end";
            EXPECT_EQ(bits.back(), ~std::uint64_t{}) << "wrote past the required words";

            std::vector<std::size_t> matches;
            sgl::soa::for_each_match(pts, pred, [&](const std::size_t i) { matches.push_back(i); });
            EXPECT_EQ(matches, expected);
            EXPECT_EQ(sgl::soa::count(pts, pred), expected.size());
            EXPECT_EQ(sgl::soa::any(pts, pred), !expected.empty());
        });
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

    sgl::soa::point_buffer2f buf;
    buf.push_back(sgl::vec2{0.5f, 0.5f});
    buf.push_back(sgl::vec2{3.0f, 0.5f});
    EXPECT_EQ(sgl::soa::count(buf, sgl::soa::inside(sgl::box2d{{0.0f, 0.0f}, {1.0f, 1.0f}})), 1u);
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

        std::size_t ring{};
        std::size_t either{};
        std::size_t outside{};
        for (std::size_t i{}; i < n; ++i) {
            const auto p{at(c.axis, i)};
            ring += (ref_inside(p, query_box) && !ref_inside(p, hole)) ? 1u : 0u;
            either += (ref_inside(p, query_box) || ref_inside(p, other)) ? 1u : 0u;
            outside += ref_inside(p, query_box) ? 0u : 1u;
        }

        /* Poison far away: it fails inside(), so a leaked lane would pass the negation. */
        for_each_layout(c, huge_hi, [&](const auto& pts, const std::string& layout) {
            SCOPED_TRACE(layout + " n=" + std::to_string(n));
            EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(query_box) && !sgl::soa::inside(hole)), ring);
            EXPECT_EQ(sgl::soa::count(pts, sgl::soa::inside(query_box) || sgl::soa::inside(other)), either);
            EXPECT_EQ(sgl::soa::count(pts, !sgl::soa::inside(query_box)), outside);
        });
    }
}

/* ------------------------------------------------------------
 * Boxes against a box or a point
 * ------------------------------------------------------------ */

TEST(Soa, OverlapsAndContainsMatchReference) {
    const std::array<float, 3> probe{4.5f, 5.0f, 3.5f};
    for (const auto n : sizes) {
        const auto b{random_boxes(n, 9u + static_cast<unsigned>(n))};

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

        /* Poison: a box covering everything, which every predicate here would match. */
        for_each_layout(b, huge_lo, huge_hi, [&](const auto& view, const std::string& layout) {
            SCOPED_TRACE(layout + " n=" + std::to_string(n));
            std::vector<std::size_t> got_over;
            std::vector<std::size_t> got_cont;
            sgl::soa::for_each_match(view, sgl::soa::overlaps(query_box), [&](const std::size_t i) { got_over.push_back(i); });
            sgl::soa::for_each_match(view, sgl::soa::contains(sgl::vec3{probe[0], probe[1], probe[2]}), [&](const std::size_t i) { got_cont.push_back(i); });
            EXPECT_EQ(got_over, over);
            EXPECT_EQ(got_cont, cont);
        });
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
    const std::array<std::array<float, 3>, 5> dirs{{{1.0f, 0.7f, 0.3f}, {-0.4f, 1.0f, -0.2f}, {0.0f, 1.0f, 0.5f}, {0.0f, 0.0f, -1.0f}, {-0.0f, 0.3f, 0.0f}}};
    const std::array<float, 3> o{0.5f, 0.25f, 9.0f};
    constexpr float sentinel{-7.0f};

    for (const auto n : sizes) {
        const auto b{random_boxes(n, 31u + static_cast<unsigned>(n))};
        for (const auto& d : dirs) {
            for (const auto [t_min, t_max] : std::array<std::array<float, 2>, 3>{{{0.0f, inf}, {1.0f, 6.0f}, {-inf, inf}}}) {
                const auto ray{sgl::soa::hit_by(sgl::vec3{o[0], o[1], o[2]}, sgl::vec3{d[0], d[1], d[2]}, t_min, t_max)};

                std::vector<ref_hit> expected(n);
                std::size_t expected_hits{};
                for (std::size_t i{}; i < n; ++i) {
                    expected[i] = ref_ray(at(b.lo, i), at(b.hi, i), o, d, t_min, t_max);
                    expected_hits += expected[i].hit ? 1u : 0u;
                }

                for_each_layout(b, huge_lo, huge_hi, [&](const auto& view, const std::string& layout) {
                    SCOPED_TRACE(layout + " n=" + std::to_string(n));
                    std::vector<float> t(n + sgl::soa::block_size, sentinel);
                    EXPECT_EQ(sgl::soa::hit_distances(view, ray, t), expected_hits);
                    for (std::size_t i{}; i < n; ++i) {
                        /* the library clamps t_min to the finite range */
                        const float want{expected[i].t == -inf ? std::numeric_limits<float>::lowest() : expected[i].t};
                        EXPECT_EQ(t[i], want) << "i=" << i;
                    }
                    for (std::size_t i{n}; i < t.size(); ++i) {
                        EXPECT_EQ(t[i], sentinel) << "wrote past the end, i=" << i;
                    }
                    EXPECT_EQ(sgl::soa::count(view, ray), expected_hits);
                });
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
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(sgl::vec2{-1.0f, -0.0001f}, sgl::vec2{1.0f, 0.0f})), 0u);
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(sgl::vec2{-1.0f, -0.0001f}, sgl::vec2{1.0f, 0.0f}, -inf, inf)), 0u);
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
    EXPECT_EQ(sgl::soa::count(view, sgl::soa::hit_by(origin, sgl::vec3{0.0f, 0.0f, 0.0f}, -inf, inf)), 1u);
}

/* ------------------------------------------------------------
 * Index compaction
 * ------------------------------------------------------------ */

namespace {

/* match_indices into the first `room` entries of a larger buffer, checking the rest is
 * untouched and returning the written indices. */
template <class R, class P> std::vector<std::uint32_t> run_match_indices(const R& range, const P& pred, const std::size_t room) {
    constexpr std::uint32_t sentinel{0xdeadbeef};
    std::vector<std::uint32_t> storage(room + sgl::soa::block_size, sentinel);
    const std::size_t written{sgl::soa::match_indices(range, pred, std::span{storage.data(), room})};
    EXPECT_LE(written, room);
    for (std::size_t i{room}; i < storage.size(); ++i) {
        EXPECT_EQ(storage[i], sentinel) << "wrote past the span, i=" << i;
    }
    storage.resize(written);
    return storage;
}

std::vector<std::uint32_t> as_u32(const std::vector<std::size_t>& v) {
    return {v.begin(), v.end()};
}

} // namespace

TEST(Soa, MatchIndicesMatchesReference) {
    const sgl::box3d everything{{-1.0f, -1.0f, -1.0f}, {11.0f, 11.0f, 11.0f}};
    const sgl::box3d sliver{{0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 0.3f}};
    for (const auto n : sizes) {
        const auto c{random_cloud(n, 211u + static_cast<unsigned>(n))};

        /* dense (every point), medium (~12%) and sparse (~3%): all three compaction paths */
        for (const auto& q : {everything, query_box, sliver}) {
            std::vector<std::size_t> expected;
            for (std::size_t i{}; i < n; ++i) {
                if (ref_inside(at(c.axis, i), q)) {
                    expected.push_back(i);
                }
            }
            for_each_layout(c, query_center, [&](const auto& pts, const std::string& layout) {
                SCOPED_TRACE(layout + " n=" + std::to_string(n));
                EXPECT_EQ(run_match_indices(pts, sgl::soa::inside(q), n), as_u32(expected));
            });
        }
    }
}

TEST(Soa, MatchIndicesShortSpanGetsTheFirstMatches) {
    const std::size_t n{1001};
    const auto c{random_cloud(n, 99u)};
    const sgl::box3d everything{{-1.0f, -1.0f, -1.0f}, {11.0f, 11.0f, 11.0f}};

    for (const std::size_t room : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{65}, std::size_t{500}, n - 1}) {
        SCOPED_TRACE("room=" + std::to_string(room));
        std::vector<std::uint32_t> expected(room);
        for (std::size_t i{}; i < room; ++i) {
            expected[i] = static_cast<std::uint32_t>(i);
        }
        EXPECT_EQ(run_match_indices(c.view(), sgl::soa::inside(everything), room), expected);
    }

    /* medium density: the first `room` of the true matches */
    std::vector<std::uint32_t> all;
    for (std::size_t i{}; i < n; ++i) {
        if (ref_inside(at(c.axis, i), query_box)) {
            all.push_back(static_cast<std::uint32_t>(i));
        }
    }
    ASSERT_GT(all.size(), 20u);
    const std::size_t room{all.size() / 2};
    EXPECT_EQ(run_match_indices(c.view(), sgl::soa::inside(query_box), room),
        std::vector<std::uint32_t>(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(room)));
}

TEST(Soa, MatchIndicesOfRayHits) {
    const auto b{random_boxes(1001, 5u)};
    const std::array<float, 3> o{0.5f, 0.25f, 9.0f};
    const std::array<float, 3> d{1.0f, 0.7f, -0.3f};
    const auto ray{sgl::soa::hit_by(sgl::vec3{o[0], o[1], o[2]}, sgl::vec3{d[0], d[1], d[2]})};

    std::vector<std::uint32_t> expected;
    for (std::size_t i{}; i < b.size(); ++i) {
        if (ref_ray(at(b.lo, i), at(b.hi, i), o, d, 0.0f, inf).hit) {
            expected.push_back(static_cast<std::uint32_t>(i));
        }
    }
    ASSERT_FALSE(expected.empty());
    for_each_layout(b, huge_lo, huge_hi, [&](const auto& view, const std::string& layout) {
        SCOPED_TRACE(layout);
        EXPECT_EQ(run_match_indices(view, ray, b.size()), expected);
    });
}

/* ------------------------------------------------------------
 * Bounds
 * ------------------------------------------------------------ */

TEST(Soa, BoundsMatchReference) {
    for (const auto n : sizes) {
        if (!n) {
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

        /* Poison below every point, so a leaked lane would lower the min. */
        for_each_layout(c, huge_lo, [&](const auto& pts, const std::string& layout) {
            SCOPED_TRACE(layout + " n=" + std::to_string(n));
            const auto b{sgl::soa::bounds(pts)};
            EXPECT_EQ(b.min.x, lo[0]);
            EXPECT_EQ(b.min.y, lo[1]);
            EXPECT_EQ(b.min.z, lo[2]);
            EXPECT_EQ(b.max.x, hi[0]);
            EXPECT_EQ(b.max.y, hi[1]);
            EXPECT_EQ(b.max.z, hi[2]);
        });
    }
}

TEST(Soa, BoundsOfEmptyIsInvertedAndNaNIsSkipped) {
    const std::vector<float> none;
    const auto empty{sgl::soa::bounds(sgl::soa::make_points(none, none))};
    EXPECT_EQ(empty.min.x, inf);
    EXPECT_EQ(empty.max.y, -inf);
    EXPECT_FALSE(sgl::is_valid(empty));
    EXPECT_FALSE(sgl::is_valid(sgl::soa::bounds(sgl::soa::point_buffer3f{})));

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

/* ------------------------------------------------------------
 * Buffers
 * ------------------------------------------------------------ */

namespace {

/* Forwards to new_delete_resource and records what it is asked for. */
class counting_resource final : public std::pmr::memory_resource {
public:
    std::size_t allocations{};
    std::size_t live{};
    std::size_t last_alignment{};

private:
    void* do_allocate(const std::size_t bytes, const std::size_t alignment) override {
        ++allocations;
        ++live;
        last_alignment = alignment;
        return std::pmr::new_delete_resource()->allocate(bytes, alignment);
    }
    void do_deallocate(void* p, const std::size_t bytes, const std::size_t alignment) override {
        --live;
        std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

bool aligned64(const float* p) {
    return !(reinterpret_cast<std::uintptr_t>(p) % 64);
}

} // namespace

TEST(SoaBuffer, CapacityIsPaddedAndArraysAligned) {
    sgl::soa::point_buffer3f pts;
    EXPECT_EQ(pts.capacity(), 0u);
    pts.push_back(sgl::vec3{1.0f, 2.0f, 3.0f});
    EXPECT_EQ(pts.capacity() % sgl::soa::block_size, 0u);
    EXPECT_GE(pts.capacity(), 1u);
    for (std::size_t a{}; a < 3; ++a) {
        EXPECT_TRUE(aligned64(pts.axis(a).data())) << "axis " << a;
    }

    sgl::soa::box_buffer3f bxs(100);
    EXPECT_EQ(bxs.size(), 100u);
    EXPECT_EQ(bxs.capacity() % sgl::soa::block_size, 0u);
    for (std::size_t a{}; a < 3; ++a) {
        EXPECT_TRUE(aligned64(bxs.min_axis(a).data()));
        EXPECT_TRUE(aligned64(bxs.max_axis(a).data()));
    }
}

TEST(SoaBuffer, GrowthKeepsDataAndNewElementsAreZero) {
    sgl::soa::point_buffer3f pts;
    for (std::size_t i{}; i < 70; ++i) {
        const auto f{static_cast<float>(i)};
        pts.push_back({f, f + 0.5f, -f});
    }
    pts.reserve(1000);
    ASSERT_EQ(pts.size(), 70u);
    for (std::size_t i{}; i < 70; ++i) {
        const auto f{static_cast<float>(i)};
        EXPECT_EQ(pts.axis(0)[i], f);
        EXPECT_EQ(pts.axis(1)[i], f + 0.5f);
        EXPECT_EQ(pts.axis(2)[i], -f);
    }

    pts.resize(10);
    pts.resize(20);
    for (std::size_t i{10}; i < 20; ++i) {
        EXPECT_EQ(pts.axis(0)[i], 0.0f) << "regrown element must be zero, not stale";
    }
    pts.clear();
    EXPECT_TRUE(pts.empty());
    EXPECT_GE(pts.capacity(), 1000u);
}

TEST(SoaBuffer, AllocatesFromItsResource) {
    counting_resource res;
    {
        sgl::soa::box_buffer3f bxs(&res);
        EXPECT_EQ(res.allocations, 0u) << "no allocation until needed";
        bxs.reserve(500);
        EXPECT_EQ(res.allocations, 1u) << "one allocation for all six arrays";
        EXPECT_EQ(res.last_alignment, 64u);
        for (std::size_t i{}; i < 500; ++i) {
            bxs.push_back(sgl::box3d{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}});
        }
        EXPECT_EQ(res.allocations, 1u) << "reserved up front, so no growth";
        EXPECT_EQ(bxs.get_memory_resource(), &res);
    }
    EXPECT_EQ(res.live, 0u);
}

TEST(SoaBuffer, CopyAndMoveFollowPmr) {
    counting_resource a;
    counting_resource b;

    sgl::soa::point_buffer3f src(&a);
    src.push_back(sgl::vec3{1.0f, 2.0f, 3.0f});
    src.push_back(sgl::vec3{4.0f, 5.0f, 6.0f});

    const sgl::soa::point_buffer3f copy_default{src};
    EXPECT_EQ(copy_default.get_memory_resource(), std::pmr::get_default_resource());
    EXPECT_EQ(copy_default.axis(2)[1], 6.0f);

    sgl::soa::point_buffer3f copy_b{src, &b};
    EXPECT_EQ(copy_b.get_memory_resource(), &b);
    EXPECT_EQ(copy_b.size(), 2u);

    const std::size_t before{a.allocations};
    sgl::soa::point_buffer3f moved{std::move(src)};
    EXPECT_EQ(a.allocations, before) << "move construction steals";
    EXPECT_EQ(moved.get_memory_resource(), &a);
    EXPECT_EQ(moved.axis(1)[0], 2.0f);

    copy_b = std::move(moved);
    EXPECT_EQ(copy_b.get_memory_resource(), &b) << "assignment keeps the target's resource";
    EXPECT_EQ(copy_b.axis(0)[1], 4.0f);
}

TEST(SoaBuffer, ViewConvertsToUnpaddedView) {
    sgl::soa::point_buffer3f pts;
    pts.push_back(sgl::vec3{4.0f, 4.0f, 4.0f});
    const sgl::soa::points3f plain = pts.view();
    EXPECT_EQ(plain.size, 1u);
    EXPECT_EQ(sgl::soa::count(plain, sgl::soa::inside(query_box)), 1u);
}
