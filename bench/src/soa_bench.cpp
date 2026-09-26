/**
 * @file soa_bench.cpp
 * @brief SoA kernels against the loops a user would otherwise write: a plain scalar loop over
 *        the same SoA arrays (left to the auto-vectoriser) and the one-at-a-time AoS API.
 */
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

import sgl;

namespace {

constexpr float inf{std::numeric_limits<float>::infinity()};

struct points_data {
    std::array<std::vector<float>, 3> axis;
    std::vector<sgl::vec3> aos;
    sgl::soa::point_buffer3f buf;
    sgl::soa::points3f view() const { return sgl::soa::make_points(axis[0], axis[1], axis[2]); }
};

struct boxes_data {
    std::array<std::vector<float>, 3> lo;
    std::array<std::vector<float>, 3> hi;
    std::vector<sgl::box3d> aos;
    sgl::soa::box_buffer3f buf;
    sgl::soa::boxes3f view() const { return sgl::soa::make_boxes(sgl::soa::make_points(lo[0], lo[1], lo[2]), sgl::soa::make_points(hi[0], hi[1], hi[2])); }
};

points_data make_points_data(const std::size_t n) {
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> u{0.0f, 100.0f};
    points_data d;
    for (auto& a : d.axis) {
        a.resize(n);
    }
    d.aos.resize(n);
    for (std::size_t i{}; i < n; ++i) {
        d.aos[i] = {u(rng), u(rng), u(rng)};
        d.axis[0][i] = d.aos[i].x;
        d.axis[1][i] = d.aos[i].y;
        d.axis[2][i] = d.aos[i].z;
    }
    d.buf.reserve(n);
    for (const auto& p : d.aos) {
        d.buf.push_back(p);
    }
    return d;
}

boxes_data make_boxes_data(const std::size_t n) {
    std::mt19937 rng{7};
    std::uniform_real_distribution<float> pos{0.0f, 100.0f};
    std::uniform_real_distribution<float> ext{0.0f, 4.0f};
    boxes_data d;
    for (std::size_t a{}; a < 3; ++a) {
        d.lo[a].resize(n);
        d.hi[a].resize(n);
    }
    d.aos.resize(n);
    for (std::size_t i{}; i < n; ++i) {
        const sgl::vec3 lo{pos(rng), pos(rng), pos(rng)};
        const sgl::vec3 hi{lo.x + ext(rng), lo.y + ext(rng), lo.z + ext(rng)};
        d.aos[i] = {lo, hi};
        d.lo[0][i] = lo.x;
        d.lo[1][i] = lo.y;
        d.lo[2][i] = lo.z;
        d.hi[0][i] = hi.x;
        d.hi[1][i] = hi.y;
        d.hi[2][i] = hi.z;
    }
    d.buf.reserve(n);
    for (const auto& b : d.aos) {
        d.buf.push_back(b);
    }
    return d;
}

/* ~25% of the points, ~25% of the boxes. */
const sgl::box3d point_query{{10.0f, 20.0f, 15.0f}, {73.0f, 83.0f, 78.0f}};
const sgl::box3d box_query{{10.0f, 20.0f, 15.0f}, {71.0f, 81.0f, 76.0f}};
const sgl::box3d hole{{30.0f, 30.0f, 30.0f}, {50.0f, 50.0f, 50.0f}};
const sgl::vec3 ray_origin{-5.0f, 3.0f, 7.0f};
const sgl::vec3 ray_dir{1.0f, 0.9f, 0.8f};

void set_items(benchmark::State& state) {
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * state.range(0));
}

std::size_t size_arg(const benchmark::State& state) {
    return static_cast<std::size_t>(state.range(0));
}

/* ------------------------------------------------------------
 * Points in a box: count
 * ------------------------------------------------------------ */

void points_in_box_count_soa(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.view(), sgl::soa::inside(point_query)));
    }
    set_items(state);
}

void points_in_box_count_buffer(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.buf, sgl::soa::inside(point_query)));
    }
    set_items(state);
}

void points_in_box_count_scalar(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto& [x, y, z]{d.axis};
    const auto& q{point_query};
    for (auto _ : state) {
        std::size_t c{};
        for (std::size_t i{}; i < x.size(); ++i) {
            if (q.min.x <= x[i] && x[i] <= q.max.x && q.min.y <= y[i] && y[i] <= q.max.y && q.min.z <= z[i] && z[i] <= q.max.z) {
                ++c;
            }
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

void points_in_box_count_scalar_branchless(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto& [x, y, z]{d.axis};
    const auto& q{point_query};
    for (auto _ : state) {
        std::size_t c{};
        for (std::size_t i{}; i < x.size(); ++i) {
            c +=
                static_cast<std::size_t>((q.min.x <= x[i]) & (x[i] <= q.max.x) & (q.min.y <= y[i]) & (y[i] <= q.max.y) & (q.min.z <= z[i]) & (z[i] <= q.max.z));
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

void points_in_box_count_aos(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        std::size_t c{};
        for (const auto& p : d.aos) {
            c += sgl::contains_point(point_query, p) ? 1u : 0u;
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Points in a box: bit mask
 * ------------------------------------------------------------ */

void points_in_box_mask_soa(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    std::vector<std::uint64_t> bits((d.axis[0].size() + 63) / 64);
    for (auto _ : state) {
        sgl::soa::mask(d.view(), sgl::soa::inside(point_query), bits);
        benchmark::ClobberMemory();
    }
    set_items(state);
}

void points_in_box_mask_scalar(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto& [x, y, z]{d.axis};
    const auto& q{point_query};
    std::vector<std::uint64_t> bits((x.size() + 63) / 64);
    for (auto _ : state) {
        for (std::size_t w{}; w < bits.size(); ++w) {
            std::uint64_t word{};
            for (std::size_t k{}; k < 64 && w * 64 + k < x.size(); ++k) {
                const std::size_t i{w * 64 + k};
                const bool in{
                    static_cast<bool>((q.min.x <= x[i]) & (x[i] <= q.max.x) & (q.min.y <= y[i]) & (y[i] <= q.max.y) & (q.min.z <= z[i]) & (z[i] <= q.max.z))};
                word |= std::uint64_t{in} << k;
            }
            bits[w] = word;
        }
        benchmark::ClobberMemory();
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Composed predicate: inside a box but outside a hole
 * ------------------------------------------------------------ */

void points_in_box_minus_hole_soa(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.view(), sgl::soa::inside(point_query) && !sgl::soa::inside(hole)));
    }
    set_items(state);
}

void points_in_box_minus_hole_scalar(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto& [x, y, z]{d.axis};
    const auto& q{point_query};
    const auto& h{hole};
    for (auto _ : state) {
        std::size_t c{};
        for (std::size_t i{}; i < x.size(); ++i) {
            const bool in_q{
                static_cast<bool>((q.min.x <= x[i]) & (x[i] <= q.max.x) & (q.min.y <= y[i]) & (y[i] <= q.max.y) & (q.min.z <= z[i]) & (z[i] <= q.max.z))};
            const bool in_h{
                static_cast<bool>((h.min.x <= x[i]) & (x[i] <= h.max.x) & (h.min.y <= y[i]) & (y[i] <= h.max.y) & (h.min.z <= z[i]) & (z[i] <= h.max.z))};
            c += static_cast<std::size_t>(in_q & !in_h);
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Boxes overlapping a box
 * ------------------------------------------------------------ */

void boxes_overlap_count_soa(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.view(), sgl::soa::overlaps(box_query)));
    }
    set_items(state);
}

void boxes_overlap_count_buffer(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.buf, sgl::soa::overlaps(box_query)));
    }
    set_items(state);
}

void boxes_overlap_count_scalar(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    const auto& q{box_query};
    const std::size_t n{d.aos.size()};
    for (auto _ : state) {
        std::size_t c{};
        for (std::size_t i{}; i < n; ++i) {
            c += static_cast<std::size_t>((d.lo[0][i] <= q.max.x) & (q.min.x <= d.hi[0][i]) & (d.lo[1][i] <= q.max.y) & (q.min.y <= d.hi[1][i]) &
                                          (d.lo[2][i] <= q.max.z) & (q.min.z <= d.hi[2][i]));
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

void boxes_overlap_count_aos(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    for (auto _ : state) {
        std::size_t c{};
        for (const auto& b : d.aos) {
            c += sgl::overlaps(b, box_query) ? 1u : 0u;
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Ray against boxes
 * ------------------------------------------------------------ */

void ray_boxes_count_soa(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.view(), sgl::soa::hit_by(ray_origin, ray_dir)));
    }
    set_items(state);
}

void ray_boxes_distances_soa(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    std::vector<float> t(d.aos.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::hit_distances(d.view(), sgl::soa::hit_by(ray_origin, ray_dir), t));
        benchmark::ClobberMemory();
    }
    set_items(state);
}

void ray_boxes_count_buffer(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(d.buf, sgl::soa::hit_by(ray_origin, ray_dir)));
    }
    set_items(state);
}

void ray_boxes_distances_buffer(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    std::vector<float> t(d.aos.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::hit_distances(d.buf, sgl::soa::hit_by(ray_origin, ray_dir), t));
        benchmark::ClobberMemory();
    }
    set_items(state);
}

/* The usual branchless slab loop (min/max swap), with the reciprocal hoisted. */
void ray_boxes_distances_scalar(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    const std::size_t n{d.aos.size()};
    std::vector<float> t(n);
    const std::array<float, 3> o{ray_origin.x, ray_origin.y, ray_origin.z};
    const std::array<float, 3> inv{1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z};
    for (auto _ : state) {
        std::size_t hits{};
        for (std::size_t i{}; i < n; ++i) {
            float t_near{0.0f};
            float t_far{inf};
            for (std::size_t a{}; a < 3; ++a) {
                const float t1{(d.lo[a][i] - o[a]) * inv[a]};
                const float t2{(d.hi[a][i] - o[a]) * inv[a]};
                t_near = std::max(t_near, std::min(t1, t2));
                t_far = std::min(t_far, std::max(t1, t2));
            }
            const bool hit{t_near <= t_far};
            t[i] = hit ? t_near : inf;
            hits += hit ? 1u : 0u;
        }
        benchmark::DoNotOptimize(hits);
        benchmark::ClobberMemory();
    }
    set_items(state);
}

void ray_boxes_distances_aos(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    std::vector<float> t(d.aos.size());
    for (auto _ : state) {
        std::size_t hits{};
        for (std::size_t i{}; i < d.aos.size(); ++i) {
            const auto h{sgl::intersect_ray_box(ray_origin, ray_dir, d.aos[i])};
            t[i] = h.hit ? h.t_entry : inf;
            hits += h.hit ? 1u : 0u;
        }
        benchmark::DoNotOptimize(hits);
        benchmark::ClobberMemory();
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Bounds
 * ------------------------------------------------------------ */

void bounds_soa(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::bounds(d.view()));
    }
    set_items(state);
}

void bounds_buffer(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::bounds(d.buf));
    }
    set_items(state);
}

void bounds_scalar(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        std::array<float, 3> lo{inf, inf, inf};
        std::array<float, 3> hi{-inf, -inf, -inf};
        for (std::size_t a{}; a < 3; ++a) {
            for (const float x : d.axis[a]) {
                lo[a] = x < lo[a] ? x : lo[a];
                hi[a] = x > hi[a] ? x : hi[a];
            }
        }
        benchmark::DoNotOptimize(lo);
        benchmark::DoNotOptimize(hi);
    }
    set_items(state);
}

void bounds_aos(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    for (auto _ : state) {
        sgl::vec3 lo{inf, inf, inf};
        sgl::vec3 hi{-inf, -inf, -inf};
        for (const auto& p : d.aos) {
            lo = sgl::min(p, lo);
            hi = sgl::max(p, hi);
        }
        benchmark::DoNotOptimize(lo);
        benchmark::DoNotOptimize(hi);
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Index compaction, by match density: range(0) points, range(1) percent inside
 * ------------------------------------------------------------ */

sgl::box3d density_query(const benchmark::State& state) {
    const float side{100.0f * std::cbrt(static_cast<float>(state.range(1)) / 100.0f)};
    return {{0.0f, 0.0f, 0.0f}, {side, side, side}};
}

void indices_soa(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto q{density_query(state)};
    std::vector<std::uint32_t> out(d.aos.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::match_indices(d.view(), sgl::soa::inside(q), out));
        benchmark::ClobberMemory();
    }
    set_items(state);
}

void indices_for_each_match(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto q{density_query(state)};
    std::vector<std::uint32_t> out(d.aos.size());
    for (auto _ : state) {
        std::size_t w{};
        sgl::soa::for_each_match(d.view(), sgl::soa::inside(q), [&](const std::size_t i) { out[w++] = static_cast<std::uint32_t>(i); });
        benchmark::DoNotOptimize(w);
        benchmark::ClobberMemory();
    }
    set_items(state);
}

/* The classic branchless compaction: always store, advance by the predicate. */
void indices_scalar_branchless(benchmark::State& state) {
    const auto d{make_points_data(size_arg(state))};
    const auto q{density_query(state)};
    const auto& [x, y, z]{d.axis};
    std::vector<std::uint32_t> out(x.size() + 1);
    for (auto _ : state) {
        std::size_t w{};
        for (std::size_t i{}; i < x.size(); ++i) {
            out[w] = static_cast<std::uint32_t>(i);
            w +=
                static_cast<std::size_t>((q.min.x <= x[i]) & (x[i] <= q.max.x) & (q.min.y <= y[i]) & (y[i] <= q.max.y) & (q.min.z <= z[i]) & (z[i] <= q.max.z));
        }
        benchmark::DoNotOptimize(w);
        benchmark::ClobberMemory();
    }
    set_items(state);
}

void indices_ray_soa(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    std::vector<std::uint32_t> out(d.aos.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::match_indices(d.view(), sgl::soa::hit_by(ray_origin, ray_dir), out));
        benchmark::ClobberMemory();
    }
    set_items(state);
}

/* ------------------------------------------------------------
 * Other element types: the same data and queries as float, converted (int32 scaled by 100)
 * ------------------------------------------------------------ */

template <class T> constexpr double type_scale{std::is_floating_point_v<T> ? 1.0 : 100.0};

template <class T> T to_t(const float x) {
    return static_cast<T>(static_cast<double>(x) * type_scale<T>);
}

template <class T> std::array<std::vector<T>, 3> typed_points(const std::size_t n) {
    const auto d{make_points_data(n)};
    std::array<std::vector<T>, 3> out;
    for (std::size_t a{}; a < 3; ++a) {
        out[a].reserve(n);
        for (const float x : d.axis[a]) {
            out[a].push_back(to_t<T>(x));
        }
    }
    return out;
}

template <class T> sgl::soa::aabb<T, 3> typed_box(const sgl::box3d& b) {
    return {{to_t<T>(b.min.x), to_t<T>(b.min.y), to_t<T>(b.min.z)}, {to_t<T>(b.max.x), to_t<T>(b.max.y), to_t<T>(b.max.z)}};
}

template <class T> void typed_points_in_box_count_soa(benchmark::State& state) {
    const auto a{typed_points<T>(size_arg(state))};
    const auto q{typed_box<T>(point_query)};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::count(sgl::soa::make_points(a[0], a[1], a[2]), sgl::soa::inside(q)));
    }
    set_items(state);
}

template <class T> void typed_points_in_box_count_scalar(benchmark::State& state) {
    const auto a{typed_points<T>(size_arg(state))};
    const auto q{typed_box<T>(point_query)};
    const auto& [x, y, z]{a};
    for (auto _ : state) {
        std::size_t c{};
        for (std::size_t i{}; i < x.size(); ++i) {
            c += static_cast<std::size_t>(
                (q.min[0] <= x[i]) & (x[i] <= q.max[0]) & (q.min[1] <= y[i]) & (y[i] <= q.max[1]) & (q.min[2] <= z[i]) & (z[i] <= q.max[2]));
        }
        benchmark::DoNotOptimize(c);
    }
    set_items(state);
}

template <class T> void typed_indices_soa(benchmark::State& state) {
    const auto a{typed_points<T>(size_arg(state))};
    const auto q{typed_box<T>(point_query)};
    std::vector<std::uint32_t> out(a[0].size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::match_indices(sgl::soa::make_points(a[0], a[1], a[2]), sgl::soa::inside(q), out));
        benchmark::ClobberMemory();
    }
    set_items(state);
}

template <class T> void typed_bounds_soa(benchmark::State& state) {
    const auto a{typed_points<T>(size_arg(state))};
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::bounds(sgl::soa::make_points(a[0], a[1], a[2])));
    }
    set_items(state);
}

template <class T> void typed_bounds_scalar(benchmark::State& state) {
    const auto a{typed_points<T>(size_arg(state))};
    for (auto _ : state) {
        std::array<T, 3> lo{};
        std::array<T, 3> hi{};
        for (std::size_t k{}; k < 3; ++k) {
            lo[k] = std::numeric_limits<T>::max();
            hi[k] = std::numeric_limits<T>::lowest();
            for (const T v : a[k]) {
                lo[k] = v < lo[k] ? v : lo[k];
                hi[k] = v > hi[k] ? v : hi[k];
            }
        }
        benchmark::DoNotOptimize(lo);
        benchmark::DoNotOptimize(hi);
    }
    set_items(state);
}

void double_ray_distances_soa(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    std::array<std::vector<double>, 3> lo;
    std::array<std::vector<double>, 3> hi;
    for (std::size_t a{}; a < 3; ++a) {
        lo[a].assign(d.lo[a].begin(), d.lo[a].end());
        hi[a].assign(d.hi[a].begin(), d.hi[a].end());
    }
    const auto view{sgl::soa::make_boxes(sgl::soa::make_points(lo[0], lo[1], lo[2]), sgl::soa::make_points(hi[0], hi[1], hi[2]))};
    const auto ray{sgl::soa::hit_by(std::array<double, 3>{ray_origin.x, ray_origin.y, ray_origin.z}, std::array<double, 3>{ray_dir.x, ray_dir.y, ray_dir.z})};
    std::vector<double> t(d.aos.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(sgl::soa::hit_distances(view, ray, t));
        benchmark::ClobberMemory();
    }
    set_items(state);
}

void double_ray_distances_scalar(benchmark::State& state) {
    const auto d{make_boxes_data(size_arg(state))};
    std::array<std::vector<double>, 3> lo;
    std::array<std::vector<double>, 3> hi;
    for (std::size_t a{}; a < 3; ++a) {
        lo[a].assign(d.lo[a].begin(), d.lo[a].end());
        hi[a].assign(d.hi[a].begin(), d.hi[a].end());
    }
    const std::size_t n{d.aos.size()};
    std::vector<double> t(n);
    const std::array<double, 3> o{ray_origin.x, ray_origin.y, ray_origin.z};
    const std::array<double, 3> inv{1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z};
    for (auto _ : state) {
        std::size_t hits{};
        for (std::size_t i{}; i < n; ++i) {
            double t_near{0.0};
            double t_far{std::numeric_limits<double>::infinity()};
            for (std::size_t a{}; a < 3; ++a) {
                const double t1{(lo[a][i] - o[a]) * inv[a]};
                const double t2{(hi[a][i] - o[a]) * inv[a]};
                t_near = std::max(t_near, std::min(t1, t2));
                t_far = std::min(t_far, std::max(t1, t2));
            }
            const bool hit{t_near <= t_far};
            t[i] = hit ? t_near : std::numeric_limits<double>::infinity();
            hits += hit ? 1u : 0u;
        }
        benchmark::DoNotOptimize(hits);
        benchmark::ClobberMemory();
    }
    set_items(state);
}

} // namespace

#define SGL_BENCH_TYPED(fn, T) BENCHMARK_TEMPLATE(fn, T)->Arg(1 << 12)->Arg(1 << 18)

SGL_BENCH_TYPED(typed_points_in_box_count_soa, double);
SGL_BENCH_TYPED(typed_points_in_box_count_scalar, double);
SGL_BENCH_TYPED(typed_points_in_box_count_soa, std::int32_t);
SGL_BENCH_TYPED(typed_points_in_box_count_scalar, std::int32_t);
SGL_BENCH_TYPED(typed_indices_soa, double);
SGL_BENCH_TYPED(typed_indices_soa, std::int32_t);
SGL_BENCH_TYPED(typed_bounds_soa, double);
SGL_BENCH_TYPED(typed_bounds_scalar, double);
SGL_BENCH_TYPED(typed_bounds_soa, std::int32_t);
SGL_BENCH_TYPED(typed_bounds_scalar, std::int32_t);
BENCHMARK(double_ray_distances_soa)->Arg(1 << 12)->Arg(1 << 18);
BENCHMARK(double_ray_distances_scalar)->Arg(1 << 12)->Arg(1 << 18);

#define SGL_BENCH_DENSITY(fn) BENCHMARK(fn)->ArgsProduct({{1 << 12}, {1, 5, 12, 25, 50, 90, 100}})

SGL_BENCH_DENSITY(indices_soa);
SGL_BENCH_DENSITY(indices_for_each_match);
SGL_BENCH_DENSITY(indices_scalar_branchless);
BENCHMARK(indices_ray_soa)->Arg(1 << 12)->Arg(1 << 18);

/* 4 Ki fits L1/L2, 256 Ki spills to L3, 4 Mi is memory bound. */
#define SGL_BENCH(fn) BENCHMARK(fn)->Arg(1 << 12)->Arg(1 << 18)->Arg(1 << 22)
/* Short ranges (leaf buckets, grid cells), where the tail is a large share of the work. */
#define SGL_BENCH_SHORT(fn) BENCHMARK(fn)->Name(#fn "_short")->Arg(13)->Arg(100)->Arg(1000)

SGL_BENCH(points_in_box_count_soa);
SGL_BENCH(points_in_box_count_buffer);
SGL_BENCH(points_in_box_count_scalar);
SGL_BENCH(points_in_box_count_scalar_branchless);
SGL_BENCH(points_in_box_count_aos);
SGL_BENCH(points_in_box_mask_soa);
SGL_BENCH(points_in_box_mask_scalar);
SGL_BENCH(points_in_box_minus_hole_soa);
SGL_BENCH(points_in_box_minus_hole_scalar);
SGL_BENCH(boxes_overlap_count_soa);
SGL_BENCH(boxes_overlap_count_buffer);
SGL_BENCH(boxes_overlap_count_scalar);
SGL_BENCH(boxes_overlap_count_aos);
SGL_BENCH(ray_boxes_count_soa);
SGL_BENCH(ray_boxes_count_buffer);
SGL_BENCH(ray_boxes_distances_soa);
SGL_BENCH(ray_boxes_distances_buffer);
SGL_BENCH(ray_boxes_distances_scalar);
SGL_BENCH(ray_boxes_distances_aos);
SGL_BENCH(bounds_soa);
SGL_BENCH(bounds_buffer);
SGL_BENCH(bounds_scalar);
SGL_BENCH(bounds_aos);

SGL_BENCH_SHORT(points_in_box_count_soa);
SGL_BENCH_SHORT(points_in_box_count_buffer);
SGL_BENCH_SHORT(points_in_box_count_scalar_branchless);
SGL_BENCH_SHORT(ray_boxes_distances_soa);
SGL_BENCH_SHORT(ray_boxes_distances_buffer);
SGL_BENCH_SHORT(ray_boxes_distances_scalar);
SGL_BENCH_SHORT(bounds_soa);
SGL_BENCH_SHORT(bounds_buffer);
SGL_BENCH_SHORT(bounds_scalar);
