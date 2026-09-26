/**
 * @file soa.cppm
 * @brief Batch queries over structure-of-arrays points and boxes.
 *
 * A query is a predicate over one kind of range (points or boxes). Predicates compose with
 * `&&`, `||` and `!` into a single pass, and a driver (mask, count, any, for_each_match)
 * runs the composed predicate over the range. See docs/soa.md.
 */
module;

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

export module sgl:soa;

import :types;
import :batch;

/* ============================================================
 * Ranges
 * ============================================================ */

export namespace sgl::soa {

/** @brief @c D coordinate arrays of @c size elements each. Pointers need no alignment. */
template <class T, std::size_t D> struct points {
    using value_type = T;
    static constexpr std::size_t dimension = D;

    std::array<const T*, D> axis;
    std::size_t size;
};

/** @brief Axis-aligned boxes as @c 2D coordinate arrays: the min corners, then the max corners. */
template <class T, std::size_t D> struct boxes {
    using value_type = T;
    static constexpr std::size_t dimension = D;

    std::array<const T*, D> min;
    std::array<const T*, D> max;
    std::size_t size;
};

using points2f = points<float, 2>;
using points3f = points<float, 3>;
using boxes2f = boxes<float, 2>;
using boxes3f = boxes<float, 3>;

/** @brief View one contiguous range per axis as points; every range must have the same size. */
template <std::ranges::contiguous_range R, std::ranges::contiguous_range... Rs>
    requires std::ranges::sized_range<R> && (std::same_as<std::ranges::range_value_t<R>, std::ranges::range_value_t<Rs>> && ...)
constexpr points<std::ranges::range_value_t<R>, 1 + sizeof...(Rs)> make_points(const R& first, const Rs&... rest) noexcept {
    const auto n{static_cast<std::size_t>(std::ranges::size(first))};
    assert(((static_cast<std::size_t>(std::ranges::size(rest)) == n) && ...));
    return {{std::ranges::data(first), std::ranges::data(rest)...}, n};
}

/** @brief View two point ranges of equal size as the min and max corners of boxes. */
template <class T, std::size_t D> constexpr boxes<T, D> make_boxes(const points<T, D>& min, const points<T, D>& max) noexcept {
    assert(min.size == max.size);
    return {min.axis, max.axis, min.size};
}

} // namespace sgl::soa

/* ============================================================
 * Internals: vocabulary mapping, row loads, axis folds
 * ============================================================ */

namespace sgl::detail {

template <class T, std::size_t D> struct soa_vocab;
template <> struct soa_vocab<float, 2> {
    using vec = vec2;
    using box = box2d;
};
template <> struct soa_vocab<float, 3> {
    using vec = vec3;
    using box = box3d;
};

template <class T, std::size_t D> using soa_vec_t = typename soa_vocab<T, D>::vec;
template <class T, std::size_t D> using soa_box_t = typename soa_vocab<T, D>::box;

template <class V> constexpr std::size_t dimension_of = 0;
template <> constexpr std::size_t dimension_of<vec2> = 2;
template <> constexpr std::size_t dimension_of<vec3> = 3;
template <> constexpr std::size_t dimension_of<box2d> = 2;
template <> constexpr std::size_t dimension_of<box3d> = 3;

constexpr std::array<float, 2> coords(const vec2& v) noexcept {
    return {v.x, v.y};
}
constexpr std::array<float, 3> coords(const vec3& v) noexcept {
    return {v.x, v.y, v.z};
}

template <class T, std::size_t D> constexpr soa_vec_t<T, D> to_vec(const std::array<T, D>& c) noexcept {
    if constexpr (D == 2) {
        return {c[0], c[1]};
    } else {
        return {c[0], c[1], c[2]};
    }
}

template <class B, std::size_t D> using point_row = std::array<B, D>;

template <class B, std::size_t D> struct box_row {
    std::array<B, D> min;
    std::array<B, D> max;
};

template <class B, class T, std::size_t D> std::array<B, D> load_axes(const std::array<const T*, D>& axes, const std::size_t i) noexcept {
    return [&]<std::size_t... A>(std::index_sequence<A...>) { return std::array<B, D>{B::load(axes[A] + i)...}; }(std::make_index_sequence<D>{});
}

template <class B, class T, std::size_t D> point_row<B, D> load_row(const soa::points<T, D>& p, const std::size_t i) noexcept {
    return load_axes<B>(p.axis, i);
}

template <class B, class T, std::size_t D> box_row<B, D> load_row(const soa::boxes<T, D>& b, const std::size_t i) noexcept {
    return {load_axes<B>(b.min, i), load_axes<B>(b.max, i)};
}

template <class B, class T, std::size_t D> std::array<B, D> broadcast(const std::array<T, D>& c) noexcept {
    return [&]<std::size_t... A>(std::index_sequence<A...>) { return std::array<B, D>{B::broadcast(c[A])...}; }(std::make_index_sequence<D>{});
}

template <class B, std::size_t D, class T> std::array<B, D> splat(const T s) noexcept {
    return [&]<std::size_t... A>(std::index_sequence<A...>) { return std::array<B, D>{((void)A, B::broadcast(s))...}; }(std::make_index_sequence<D>{});
}

/* AND of f(axis) over every axis, unrolled at compile time. */
template <std::size_t D, class F> auto all_axes(F&& f) noexcept {
    return [&]<std::size_t... A>(std::index_sequence<A...>) { return (f(A) & ...); }(std::make_index_sequence<D>{});
}

/* Slab test with the per-axis near/far planes picked once from the ray's sign, rather than
 * a per-lane min/max swap: that keeps inverted (min > max) boxes a miss. Axes with an
 * infinite reciprocal are tested as a containment instead, since (min - o) * inf is NaN
 * when the origin lies on the slab. */
template <class B, std::size_t D> struct ray_slabs {
    std::array<B, D> origin;
    std::array<B, D> inv_dir;
    std::array<bool, D> parallel;
    std::array<bool, D> negative;
    B t_min;
    B t_max;

    struct entry_result {
        B t;
        typename B::mask hit;
    };

    entry_result entry(const box_row<B, D>& r) const noexcept {
        B t_near{t_min};
        B t_far{t_max};
        for (std::size_t a{}; a < D; ++a) {
            if (parallel[a]) {
                continue;
            }
            const B& lo{negative[a] ? r.max[a] : r.min[a]};
            const B& hi{negative[a] ? r.min[a] : r.max[a]};
            t_near = max((lo - origin[a]) * inv_dir[a], t_near);
            t_far = min((hi - origin[a]) * inv_dir[a], t_far);
        }
        auto hit{t_near <= t_far};
        for (std::size_t a{}; a < D; ++a) {
            if (parallel[a]) {
                hit = hit & (r.min[a] <= origin[a]) & (origin[a] <= r.max[a]);
            }
        }
        return {t_near, hit};
    }

    typename B::mask operator()(const box_row<B, D>& r) const noexcept { return entry(r).hit; }
};

/* Runs the predicate over the range in blocks of 64 and hands each block to on_word as a
 * bit mask (bit k = element 64 * word + k; bits past the end are zero). on_word returns
 * false to stop early. */
template <class R, class P, class F> void scan_words(const R& range, const P& pred, F&& on_word) noexcept {
    using T = typename R::value_type;
    using wide = batch<T, native_isa>;
    using narrow = batch<T, scalar_isa>;
    constexpr std::size_t lanes{wide::width};
    static_assert(64 % lanes == 0);

    const auto wide_pred{pred.template bind<wide>()};
    const auto narrow_pred{pred.template bind<narrow>()};

    const std::size_t n{range.size};
    std::size_t base{};
    for (; base + 64 <= n; base += 64) {
        std::uint64_t word{};
        for (std::size_t k{}; k < 64; k += lanes) {
            word |= std::uint64_t{wide_pred(load_row<wide>(range, base + k)).bits()} << k;
        }
        if (!on_word(base / 64, word)) {
            return;
        }
    }
    if (base == n) {
        return;
    }

    std::uint64_t word{};
    std::size_t k{};
    for (; base + k + lanes <= n; k += lanes) {
        word |= std::uint64_t{wide_pred(load_row<wide>(range, base + k)).bits()} << k;
    }
    for (; base + k < n; ++k) {
        word |= std::uint64_t{narrow_pred(load_row<narrow>(range, base + k)).bits()} << k;
    }
    on_word(base / 64, word);
}

} // namespace sgl::detail

/* ============================================================
 * Predicates
 * ============================================================ */

export namespace sgl::soa {

/** @brief A predicate over the elements of @c P::range_type. */
template <class P>
concept predicate = requires { typename P::range_type; };

template <class P, class R>
concept predicate_for = predicate<P> && std::same_as<typename P::range_type, R>;

/** @brief Points inside a closed box. */
template <class T, std::size_t D> struct inside_box {
    using range_type = points<T, D>;
    std::array<T, D> lo;
    std::array<T, D> hi;

    template <class B> auto bind() const noexcept {
        return [l = detail::broadcast<B>(lo), h = detail::broadcast<B>(hi)](const detail::point_row<B, D>& p) noexcept {
            return detail::all_axes<D>([&](const std::size_t a) { return (l[a] <= p[a]) & (p[a] <= h[a]); });
        };
    }
};

/** @brief Boxes that overlap a closed box (touching counts). Boxes must be valid or the empty box (+inf, -inf). */
template <class T, std::size_t D> struct overlaps_box {
    using range_type = boxes<T, D>;
    std::array<T, D> lo;
    std::array<T, D> hi;

    template <class B> auto bind() const noexcept {
        return [l = detail::broadcast<B>(lo), h = detail::broadcast<B>(hi)](const detail::box_row<B, D>& r) noexcept {
            return detail::all_axes<D>([&](const std::size_t a) { return (r.min[a] <= h[a]) & (l[a] <= r.max[a]); });
        };
    }
};

/** @brief Closed boxes that contain a point. */
template <class T, std::size_t D> struct contains_point {
    using range_type = boxes<T, D>;
    std::array<T, D> point;

    template <class B> auto bind() const noexcept {
        return [p = detail::broadcast<B>(point)](const detail::box_row<B, D>& r) noexcept {
            return detail::all_axes<D>([&](const std::size_t a) { return (r.min[a] <= p[a]) & (p[a] <= r.max[a]); });
        };
    }
};

/** @brief Closed boxes hit by the ray origin + t * direction for some t in [t_min, t_max]. */
template <class T, std::size_t D> struct ray_hit {
    using range_type = boxes<T, D>;
    std::array<T, D> origin;
    std::array<T, D> inv_dir;
    T t_min;
    T t_max;

    template <class B> detail::ray_slabs<B, D> bind() const noexcept {
        detail::ray_slabs<B, D> s{detail::broadcast<B>(origin), detail::broadcast<B>(inv_dir), {}, {}, B::broadcast(t_min), B::broadcast(t_max)};
        for (std::size_t a{}; a < D; ++a) {
            s.parallel[a] = std::isinf(inv_dir[a]);
            s.negative[a] = std::signbit(inv_dir[a]);
        }
        return s;
    }
};

template <predicate P, predicate Q> struct conjunction {
    using range_type = typename P::range_type;
    P p;
    Q q;

    template <class B> auto bind() const noexcept {
        return [pb = p.template bind<B>(), qb = q.template bind<B>()](const auto& row) noexcept { return pb(row) & qb(row); };
    }
};

template <predicate P, predicate Q> struct disjunction {
    using range_type = typename P::range_type;
    P p;
    Q q;

    template <class B> auto bind() const noexcept {
        return [pb = p.template bind<B>(), qb = q.template bind<B>()](const auto& row) noexcept { return pb(row) | qb(row); };
    }
};

template <predicate P> struct negation {
    using range_type = typename P::range_type;
    P p;

    template <class B> auto bind() const noexcept {
        return [pb = p.template bind<B>()](const auto& row) noexcept { return ~pb(row); };
    }
};

template <predicate P, predicate Q>
    requires std::same_as<typename P::range_type, typename Q::range_type>
constexpr conjunction<P, Q> operator&&(const P& p, const Q& q) noexcept {
    return {p, q};
}

template <predicate P, predicate Q>
    requires std::same_as<typename P::range_type, typename Q::range_type>
constexpr disjunction<P, Q> operator||(const P& p, const Q& q) noexcept {
    return {p, q};
}

template <predicate P> constexpr negation<P> operator!(const P& p) noexcept {
    return {p};
}

/** @brief Points inside @p box (closed). */
template <box_type Box> constexpr inside_box<float, detail::dimension_of<Box>> inside(const Box& box) noexcept {
    return {detail::coords(box.min), detail::coords(box.max)};
}

/** @brief Boxes overlapping @p box (closed). */
template <box_type Box> constexpr overlaps_box<float, detail::dimension_of<Box>> overlaps(const Box& box) noexcept {
    return {detail::coords(box.min), detail::coords(box.max)};
}

/** @brief Boxes containing @p point (closed). */
template <spatial_vector Vec> constexpr contains_point<float, detail::dimension_of<Vec>> contains(const Vec& point) noexcept {
    return {detail::coords(point)};
}

/**
 * @brief Boxes hit by a ray.
 *
 * @p direction need not be normalized; t is measured in units of it, so a segment [a, b] is
 * hit_by(a, b - a, 0, 1). A zero direction component is fine. A NaN coordinate in a box
 * makes its result unspecified (but the same on every backend).
 */
template <spatial_vector Vec>
ray_hit<float, detail::dimension_of<Vec>> hit_by(
    const Vec& origin, const Vec& direction, const float t_min = 0.0f, const float t_max = std::numeric_limits<float>::infinity()) noexcept {
    constexpr std::size_t D{detail::dimension_of<Vec>};
    const auto d{detail::coords(direction)};
    std::array<float, D> inv{};
    for (std::size_t a{}; a < D; ++a) {
        inv[a] = 1.0f / d[a];
    }
    return {detail::coords(origin), inv, t_min, t_max};
}

/* ============================================================
 * Drivers
 * ============================================================ */

/**
 * @brief Write one bit per element: bit k of out[w] is element 64 * w + k.
 *
 * @p out needs (size + 63) / 64 words; bits past the end of the range are written as zero.
 */
template <class R, predicate_for<R> P> void mask(const R& range, const P& pred, const std::span<std::uint64_t> out) noexcept {
    assert(out.size() >= (range.size + 63) / 64);
    detail::scan_words(range, pred, [out](const std::size_t w, const std::uint64_t bits) noexcept {
        out[w] = bits;
        return true;
    });
}

/** @brief Number of elements matching @p pred. */
template <class R, predicate_for<R> P> [[nodiscard]] std::size_t count(const R& range, const P& pred) noexcept {
    std::size_t n{};
    detail::scan_words(range, pred, [&n](std::size_t, const std::uint64_t bits) noexcept {
        n += static_cast<std::size_t>(std::popcount(bits));
        return true;
    });
    return n;
}

/** @brief Whether any element matches; stops at the first block of 64 with a match. */
template <class R, predicate_for<R> P> [[nodiscard]] bool any(const R& range, const P& pred) noexcept {
    bool found{};
    detail::scan_words(range, pred, [&found](std::size_t, const std::uint64_t bits) noexcept {
        found = bits != 0;
        return !found;
    });
    return found;
}

/** @brief Call @p fn with the index of every matching element, in increasing order. */
template <class R, predicate_for<R> P, std::invocable<std::size_t> F> void for_each_match(const R& range, const P& pred, F&& fn) {
    detail::scan_words(range, pred, [&fn](const std::size_t w, std::uint64_t bits) {
        for (; bits != 0; bits &= bits - 1) {
            fn(w * 64 + static_cast<std::size_t>(std::countr_zero(bits)));
        }
        return true;
    });
}

/**
 * @brief Ray entry distance per box: max(t_min, entry) on a hit, +inf on a miss.
 *
 * @p t_entry needs @c boxes.size elements. Returns the number of hits.
 */
template <class T, std::size_t D>
std::size_t hit_distances(const boxes<T, D>& bxs, const ray_hit<T, D>& ray, const std::span<std::type_identity_t<T>> t_entry) noexcept {
    using wide = detail::batch<T, detail::native_isa>;
    using narrow = detail::batch<T, detail::scalar_isa>;
    assert(t_entry.size() >= bxs.size);

    const auto wide_ray{ray.template bind<wide>()};
    const auto narrow_ray{ray.template bind<narrow>()};
    const auto wide_miss{wide::broadcast(std::numeric_limits<T>::infinity())};
    const auto narrow_miss{narrow::broadcast(std::numeric_limits<T>::infinity())};

    std::size_t hits{};
    std::size_t i{};
    for (; i + wide::width <= bxs.size; i += wide::width) {
        const auto [t, hit]{wide_ray.entry(detail::load_row<wide>(bxs, i))};
        select(hit, t, wide_miss).store(t_entry.data() + i);
        hits += static_cast<std::size_t>(std::popcount(hit.bits()));
    }
    for (; i < bxs.size; ++i) {
        const auto [t, hit]{narrow_ray.entry(detail::load_row<narrow>(bxs, i))};
        select(hit, t, narrow_miss).store(t_entry.data() + i);
        hits += hit.bits();
    }
    return hits;
}

/**
 * @brief Axis-aligned bounds of the points. NaN coordinates are skipped.
 *
 * An empty range gives the inverted box min = +inf, max = -inf, which is_valid rejects and
 * merge treats as the identity.
 */
template <class T, std::size_t D> [[nodiscard]] detail::soa_box_t<T, D> bounds(const points<T, D>& pts) noexcept {
    using wide = detail::batch<T, detail::native_isa>;
    constexpr std::size_t lanes{wide::width};
    constexpr T inf{std::numeric_limits<T>::infinity()};

    /* Two accumulator sets hide the min/max latency. */
    auto lo0{detail::splat<wide, D>(inf)};
    auto hi0{detail::splat<wide, D>(-inf)};
    auto lo1{lo0};
    auto hi1{hi0};

    std::size_t i{};
    for (; i + 2 * lanes <= pts.size; i += 2 * lanes) {
        for (std::size_t a{}; a < D; ++a) {
            const auto x0{wide::load(pts.axis[a] + i)};
            const auto x1{wide::load(pts.axis[a] + i + lanes)};
            lo0[a] = min(x0, lo0[a]);
            hi0[a] = max(x0, hi0[a]);
            lo1[a] = min(x1, lo1[a]);
            hi1[a] = max(x1, hi1[a]);
        }
    }
    for (; i + lanes <= pts.size; i += lanes) {
        for (std::size_t a{}; a < D; ++a) {
            const auto x{wide::load(pts.axis[a] + i)};
            lo0[a] = min(x, lo0[a]);
            hi0[a] = max(x, hi0[a]);
        }
    }

    std::array<T, D> lo{};
    std::array<T, D> hi{};
    for (std::size_t a{}; a < D; ++a) {
        lo[a] = reduce_min(min(lo1[a], lo0[a]));
        hi[a] = reduce_max(max(hi1[a], hi0[a]));
        for (std::size_t j{i}; j < pts.size; ++j) {
            const T x{pts.axis[a][j]};
            lo[a] = x < lo[a] ? x : lo[a];
            hi[a] = x > hi[a] ? x : hi[a];
        }
    }
    return {detail::to_vec<T, D>(lo), detail::to_vec<T, D>(hi)};
}

} // namespace sgl::soa
