/**
 * @file soa.cppm
 * @brief Batch queries over structure-of-arrays points and boxes.
 *
 * A query is a predicate over one kind of range (points or boxes). Predicates compose with
 * `&&`, `||` and `!` into a single pass, and a driver (mask, count, any, for_each_match)
 * runs the composed predicate over the range. See docs/soa.md.
 */
module;

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

export module sgl:soa;

import :types;
import :batch;

/* ============================================================
 * Views
 * ============================================================ */

export namespace sgl::soa {

/** @brief Block size of the drivers, and the padding of the owning buffers. */
constexpr std::size_t block_size{64};

template <class T, std::size_t D> struct point_kind {};
template <class T, std::size_t D> struct box_kind {};

/** @brief A closed axis-aligned box of any element type, for queries and bounds beyond the float vocabulary types. */
template <class T, std::size_t D> struct aabb {
    std::array<T, D> min;
    std::array<T, D> max;
};

/**
 * @brief @c D coordinate arrays of @c size elements each. Pointers need no alignment.
 *
 * @c Pad > 1 promises that elements up to @c size rounded up to a multiple of @c Pad are
 * readable; their contents do not matter. Kernels use it to drop the scalar tail.
 */
template <class T, std::size_t D, std::size_t Pad = 1> struct points {
    using value_type = T;
    using kind = point_kind<T, D>;
    static constexpr std::size_t dimension{D};
    static constexpr std::size_t padding{Pad};

    std::array<const T*, D> axis;
    std::size_t size;

    constexpr operator points<T, D>() const noexcept
        requires(Pad != 1)
    {
        return {axis, size};
    }
};

/** @brief Axis-aligned boxes as @c 2D coordinate arrays: the min corners, then the max corners. See @c points for @c Pad. */
template <class T, std::size_t D, std::size_t Pad = 1> struct boxes {
    using value_type = T;
    using kind = box_kind<T, D>;
    static constexpr std::size_t dimension{D};
    static constexpr std::size_t padding{Pad};

    std::array<const T*, D> min;
    std::array<const T*, D> max;
    std::size_t size;

    constexpr operator boxes<T, D>() const noexcept
        requires(Pad != 1)
    {
        return {min, max, size};
    }
};

using points2f = points<float, 2>;
using points3f = points<float, 3>;
using boxes2f = boxes<float, 2>;
using boxes3f = boxes<float, 3>;

/** @brief A view or buffer of points. */
template <class R>
concept point_range = std::same_as<typename R::kind, point_kind<typename R::value_type, R::dimension>>;

/** @brief A view or buffer of boxes. */
template <class R>
concept box_range = std::same_as<typename R::kind, box_kind<typename R::value_type, R::dimension>>;

/** @brief View one contiguous range per axis as points; every range must have the same size. */
template <std::ranges::contiguous_range R, std::ranges::contiguous_range... Rs>
    requires std::ranges::sized_range<R> && (std::same_as<std::ranges::range_value_t<R>, std::ranges::range_value_t<Rs>> && ...)
constexpr points<std::ranges::range_value_t<R>, 1 + sizeof...(Rs)> make_points(const R& first, const Rs&... rest) noexcept {
    const auto n{static_cast<std::size_t>(std::ranges::size(first))};
    assert(((static_cast<std::size_t>(std::ranges::size(rest)) == n) && ...));
    return {{std::ranges::data(first), std::ranges::data(rest)...}, n};
}

/** @brief View two point ranges of equal size as the min and max corners of boxes. */
template <class T, std::size_t D, std::size_t Pad> constexpr boxes<T, D, Pad> make_boxes(const points<T, D, Pad>& min, const points<T, D, Pad>& max) noexcept {
    assert(min.size == max.size);
    return {min.axis, max.axis, min.size};
}

} // namespace sgl::soa

/* ============================================================
 * Internals
 * ============================================================ */

namespace sgl::detail {

/* Element type T in D dimensions maps to the core vocabulary box where one exists (float),
 * and to soa::aabb otherwise. */
template <class T, std::size_t D> struct soa_vocab {
    using box = soa::aabb<T, D>;
};
template <> struct soa_vocab<float, 2> {
    using vec = vec2;
    using box = box2d;
};
template <> struct soa_vocab<float, 3> {
    using vec = vec3;
    using box = box3d;
};

template <class T, std::size_t D> using soa_box_t = typename soa_vocab<T, D>::box;

/* Core point types usable as queries: their element type and dimension. */
template <class V> struct vocab_point;
template <> struct vocab_point<vec2> {
    using value_type = float;
    static constexpr std::size_t dimension{2};
};
template <> struct vocab_point<vec3> {
    using value_type = float;
    static constexpr std::size_t dimension{3};
};
template <> struct vocab_point<ivec2> {
    using value_type = std::int32_t;
    static constexpr std::size_t dimension{2};
};
template <> struct vocab_point<ivec3> {
    using value_type = std::int32_t;
    static constexpr std::size_t dimension{3};
};

template <class V>
concept vocab_point_type = requires { vocab_point<std::remove_cvref_t<V>>::dimension; };

template <class V> constexpr std::size_t dimension_of = vocab_point<V>::dimension;
template <> constexpr std::size_t dimension_of<box2d> = 2;
template <> constexpr std::size_t dimension_of<box3d> = 3;

template <class V> using point_value_t = typename vocab_point<V>::value_type;

constexpr std::array<float, 2> coords(const vec2& v) noexcept {
    return {v.x, v.y};
}
constexpr std::array<float, 3> coords(const vec3& v) noexcept {
    return {v.x, v.y, v.z};
}
constexpr std::array<std::int32_t, 2> coords(const ivec2& v) noexcept {
    return {v.x, v.y};
}
constexpr std::array<std::int32_t, 3> coords(const ivec3& v) noexcept {
    return {v.x, v.y, v.z};
}

template <class T, std::size_t D> constexpr soa_box_t<T, D> to_box(const std::array<T, D>& lo, const std::array<T, D>& hi) noexcept {
    if constexpr (std::same_as<soa_box_t<T, D>, soa::aabb<T, D>>) {
        return {lo, hi};
    } else if constexpr (D == 2) {
        return {{lo[0], lo[1]}, {hi[0], hi[1]}};
    } else {
        return {{lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}};
    }
}

/* The identity of min / max: +-inf where the type has it, the extreme values otherwise. */
template <class T> constexpr T empty_min{std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max()};
template <class T> constexpr T empty_max{std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest()};

constexpr std::size_t round_up(const std::size_t n, const std::size_t m) noexcept {
    return (n + m - 1) / m * m;
}

/* Views pass through; buffers hand out their padded view. */
template <class R> constexpr auto as_view(const R& r) noexcept {
    if constexpr (requires { r.view(); }) {
        return r.view();
    } else {
        return r;
    }
}

/* Whether a view's padding lets whole native batches run past its size. */
template <class V, class B> constexpr bool padded_for{!(V::padding % B::width)};

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

/* Slab test. The near and far plane of each axis are picked once from the ray's sign, at bind
 * time, instead of a per-lane min/max swap: no branch in the loop, and inverted (min > max)
 * boxes stay a miss.
 *
 * A parallel axis needs no special case, as long as the new t stays the FIRST operand of
 * max/min: its reciprocal is +-inf, so a plane gives +-inf, or NaN when the origin lies on
 * it, and max/min return their second operand on NaN. The slab degrades to a closed
 * containment test. That relies on t_min / t_max being finite (bind clamps them): a missed
 * parallel slab sets t_near to +inf, which must exceed t_far. */
template <class B, std::size_t D> struct ray_slabs {
    using T = typename B::value_type;

    std::array<const T*, D> near_plane;
    std::array<const T*, D> far_plane;
    std::array<B, D> origin;
    std::array<B, D> inv_dir;
    B t_min;
    B t_max;

    struct entry_result {
        B t;
        typename B::mask hit;
    };

    entry_result entry(const std::size_t i) const noexcept {
        B t_near{t_min};
        B t_far{t_max};
        for (std::size_t a{}; a < D; ++a) {
            t_near = max((B::load(near_plane[a] + i) - origin[a]) * inv_dir[a], t_near);
            t_far = min((B::load(far_plane[a] + i) - origin[a]) * inv_dir[a], t_far);
        }
        return {t_near, t_near <= t_far};
    }

    typename B::mask operator()(const std::size_t i) const noexcept { return entry(i).hit; }
};

/* The bit mask of the last, partial block starting at base (bits past the end are zero). A
 * padded view runs it on whole native batches and masks the excess; an unpadded one
 * finishes on the scalar backend. */
template <class V, class P, class W> std::uint64_t tail_word(const V& view, const P& pred, const W& wide_pred, const std::size_t base) noexcept {
    using T = typename V::value_type;
    using wide = batch<T, native_isa>;
    constexpr std::size_t lanes{wide::width};

    const std::size_t rem{view.size - base};
    std::uint64_t word{};
    if constexpr (padded_for<V, wide>) {
        for (std::size_t k{}; k < rem; k += lanes) {
            word |= std::uint64_t{wide_pred(base + k).bits()} << k;
        }
        word &= (std::uint64_t{1} << rem) - 1;
    } else {
        using narrow = batch<T, scalar_isa>;
        const auto narrow_pred{pred.template bind<narrow>(view)};
        std::size_t k{};
        for (; k + lanes <= rem; k += lanes) {
            word |= std::uint64_t{wide_pred(base + k).bits()} << k;
        }
        for (; k < rem; ++k) {
            word |= std::uint64_t{narrow_pred(base + k).bits()} << k;
        }
    }
    return word;
}

/* Runs the predicate over the view in blocks of 64 and hands each block to on_word as a bit
 * mask (bit k = element 64 * word + k; bits past the end are zero). on_word returns false to
 * stop early. */
template <class V, class P, class F> void scan_words(const V& view, const P& pred, F&& on_word) noexcept {
    using T = typename V::value_type;
    using wide = batch<T, native_isa>;
    constexpr std::size_t lanes{wide::width};
    static_assert(!(soa::block_size % lanes));

    const auto wide_pred{pred.template bind<wide>(view)};
    const std::size_t n{view.size};

    std::size_t base{};
    for (; base + soa::block_size <= n; base += soa::block_size) {
        std::uint64_t word{};
        for (std::size_t k{}; k < soa::block_size; k += lanes) {
            word |= std::uint64_t{wide_pred(base + k).bits()} << k;
        }
        if (!on_word(base / soa::block_size, word)) {
            return;
        }
    }
    if (base < n) {
        on_word(base / soa::block_size, tail_word(view, pred, wide_pred, base));
    }
}

/* N arrays of T in one 64-byte aligned allocation, all with the same capacity, a multiple of
 * soa::block_size. Everything past size is initialized, which is what lets padded views read it. */
template <class T, std::size_t N> class padded_arrays {
    static_assert(std::is_trivially_copyable_v<T>);
    static constexpr std::size_t alignment{64};

public:
    explicit padded_arrays(std::pmr::memory_resource* r) noexcept : resource_{r} {}

    padded_arrays(const padded_arrays& other, std::pmr::memory_resource* r) : resource_{r} { assign(other); }

    padded_arrays(padded_arrays&& other) noexcept
        : resource_{other.resource_}, data_{std::exchange(other.data_, nullptr)}, size_{std::exchange(other.size_, 0)},
          capacity_{std::exchange(other.capacity_, 0)} {}

    padded_arrays& operator=(const padded_arrays& other) {
        if (this != &other) {
            assign(other);
        }
        return *this;
    }

    /* pmr semantics: the resource stays; storage is stolen only when the resources agree. */
    padded_arrays& operator=(padded_arrays&& other) {
        if (this == &other) {
            return *this;
        }
        if (*resource_ == *other.resource_) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
        } else {
            assign(other);
        }
        return *this;
    }

    ~padded_arrays() { release(); }

    T* array(const std::size_t a) noexcept { return data_ + a * stride(capacity_); }
    const T* array(const std::size_t a) const noexcept { return data_ + a * stride(capacity_); }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::pmr::memory_resource* resource() const noexcept { return resource_; }

    void reserve(const std::size_t n) {
        if (n <= capacity_) {
            return;
        }
        const std::size_t cap{round_up(std::max(n, 2 * capacity_), soa::block_size)};
        auto* fresh{static_cast<T*>(resource_->allocate(bytes(cap), alignment))};
        for (std::size_t a{}; a < N; ++a) {
            T* dst{fresh + a * stride(cap)};
            if (size_) {
                std::memcpy(dst, array(a), size_ * sizeof(T));
            }
            std::fill(dst + size_, dst + stride(cap), T{});
        }
        release();
        data_ = fresh;
        capacity_ = cap;
    }

    void resize(const std::size_t n) {
        reserve(n);
        if (n > size_) {
            for (std::size_t a{}; a < N; ++a) {
                std::fill(array(a) + size_, array(a) + n, T{});
            }
        }
        size_ = n;
    }

    void clear() noexcept { size_ = 0; }

    void push_back(const std::array<T, N>& v) {
        if (size_ == capacity_) {
            reserve(size_ + 1);
        }
        for (std::size_t a{}; a < N; ++a) {
            array(a)[size_] = v[a];
        }
        ++size_;
    }

private:
    void assign(const padded_arrays& other) {
        size_ = 0;
        reserve(other.size_);
        for (std::size_t a{}; a < N; ++a) {
            if (other.size_) {
                std::memcpy(array(a), other.array(a), other.size_ * sizeof(T));
            }
        }
        size_ = other.size_;
    }

    /* Zero without storage: array(a) on an empty buffer must not offset its null data pointer. */
    static constexpr std::size_t stride(const std::size_t cap) noexcept { return cap ? cap + soa::block_size : 0; }
    static constexpr std::size_t bytes(const std::size_t cap) noexcept { return N * stride(cap) * sizeof(T); }

    void release() noexcept {
        if (data_) {
            resource_->deallocate(data_, bytes(capacity_), alignment);
            data_ = nullptr;
        }
    }

    std::pmr::memory_resource* resource_;
    T* data_{};
    std::size_t size_{};
    std::size_t capacity_{};
};

} // namespace sgl::detail

/* ============================================================
 * Owning buffers
 * ============================================================ */

export namespace sgl::soa {

/**
 * @brief Owning SoA points on a memory resource: one 64-byte aligned allocation, capacity a
 *        multiple of block_size. Drivers take it directly and use its padded view.
 *
 * Copies follow pmr: copy construction uses the default resource unless one is given, and
 * assignment keeps the target's resource.
 */
template <class T, std::size_t D> class point_buffer {
public:
    using value_type = T;
    using kind = point_kind<T, D>;
    static constexpr std::size_t dimension{D};

    explicit point_buffer(std::pmr::memory_resource* r = std::pmr::get_default_resource()) noexcept : arrays_{r} {}
    explicit point_buffer(const std::size_t n, std::pmr::memory_resource* r = std::pmr::get_default_resource()) : arrays_{r} { arrays_.resize(n); }
    point_buffer(const point_buffer& other) : arrays_{other.arrays_, std::pmr::get_default_resource()} {}
    point_buffer(const point_buffer& other, std::pmr::memory_resource* r) : arrays_{other.arrays_, r} {}
    point_buffer(point_buffer&&) noexcept = default;
    point_buffer& operator=(const point_buffer&) = default;
    point_buffer& operator=(point_buffer&&) = default;
    ~point_buffer() = default;

    std::size_t size() const noexcept { return arrays_.size(); }
    std::size_t capacity() const noexcept { return arrays_.capacity(); }
    bool empty() const noexcept { return !arrays_.size(); }
    std::pmr::memory_resource* get_memory_resource() const noexcept { return arrays_.resource(); }

    void reserve(const std::size_t n) { arrays_.reserve(n); }
    /** @brief New elements are zero. */
    void resize(const std::size_t n) { arrays_.resize(n); }
    void clear() noexcept { arrays_.clear(); }

    std::span<T> axis(const std::size_t a) noexcept { return {arrays_.array(a), arrays_.size()}; }
    std::span<const T> axis(const std::size_t a) const noexcept { return {arrays_.array(a), arrays_.size()}; }

    void push_back(const std::array<T, D>& p) { arrays_.push_back(p); }

    template <detail::vocab_point_type Vec>
        requires(std::same_as<T, detail::point_value_t<Vec>> && detail::dimension_of<Vec> == D)
    void push_back(const Vec& p) {
        arrays_.push_back(detail::coords(p));
    }

    points<T, D, block_size> view() const noexcept {
        return {[&]<std::size_t... A>(std::index_sequence<A...>) { return std::array<const T*, D>{arrays_.array(A)...}; }(std::make_index_sequence<D>{}),
            arrays_.size()};
    }

private:
    detail::padded_arrays<T, D> arrays_;
};

/** @brief Owning SoA boxes; see @c point_buffer. The min corners are arrays [0, D), the max corners [D, 2D). */
template <class T, std::size_t D> class box_buffer {
public:
    using value_type = T;
    using kind = box_kind<T, D>;
    static constexpr std::size_t dimension{D};

    explicit box_buffer(std::pmr::memory_resource* r = std::pmr::get_default_resource()) noexcept : arrays_{r} {}
    explicit box_buffer(const std::size_t n, std::pmr::memory_resource* r = std::pmr::get_default_resource()) : arrays_{r} { arrays_.resize(n); }
    box_buffer(const box_buffer& other) : arrays_{other.arrays_, std::pmr::get_default_resource()} {}
    box_buffer(const box_buffer& other, std::pmr::memory_resource* r) : arrays_{other.arrays_, r} {}
    box_buffer(box_buffer&&) noexcept = default;
    box_buffer& operator=(const box_buffer&) = default;
    box_buffer& operator=(box_buffer&&) = default;
    ~box_buffer() = default;

    std::size_t size() const noexcept { return arrays_.size(); }
    std::size_t capacity() const noexcept { return arrays_.capacity(); }
    bool empty() const noexcept { return !arrays_.size(); }
    std::pmr::memory_resource* get_memory_resource() const noexcept { return arrays_.resource(); }

    void reserve(const std::size_t n) { arrays_.reserve(n); }
    /** @brief New elements are the zero box. */
    void resize(const std::size_t n) { arrays_.resize(n); }
    void clear() noexcept { arrays_.clear(); }

    std::span<T> min_axis(const std::size_t a) noexcept { return {arrays_.array(a), arrays_.size()}; }
    std::span<const T> min_axis(const std::size_t a) const noexcept { return {arrays_.array(a), arrays_.size()}; }
    std::span<T> max_axis(const std::size_t a) noexcept { return {arrays_.array(D + a), arrays_.size()}; }
    std::span<const T> max_axis(const std::size_t a) const noexcept { return {arrays_.array(D + a), arrays_.size()}; }

    void push_back(const std::array<T, D>& lo, const std::array<T, D>& hi) {
        std::array<T, 2 * D> c{};
        std::copy(lo.begin(), lo.end(), c.begin());
        std::copy(hi.begin(), hi.end(), c.begin() + D);
        arrays_.push_back(c);
    }

    template <box_type Box>
        requires(std::same_as<T, float> && detail::dimension_of<Box> == D)
    void push_back(const Box& b) {
        push_back(detail::coords(b.min), detail::coords(b.max));
    }

    void push_back(const aabb<T, D>& b) { push_back(b.min, b.max); }

    boxes<T, D, block_size> view() const noexcept {
        return {[&]<std::size_t... A>(std::index_sequence<A...>) { return std::array<const T*, D>{arrays_.array(A)...}; }(std::make_index_sequence<D>{}),
            [&]<std::size_t... A>(std::index_sequence<A...>) { return std::array<const T*, D>{arrays_.array(D + A)...}; }(std::make_index_sequence<D>{}),
            arrays_.size()};
    }

private:
    detail::padded_arrays<T, 2 * D> arrays_;
};

using point_buffer2f = point_buffer<float, 2>;
using point_buffer3f = point_buffer<float, 3>;
using box_buffer2f = box_buffer<float, 2>;
using box_buffer3f = box_buffer<float, 3>;

/* ============================================================
 * Predicates
 * ============================================================ */

/** @brief A predicate over the elements of one kind of range. */
template <class P>
concept predicate = requires { typename P::kind; };

template <class P, class R>
concept predicate_for = predicate<P> && std::same_as<typename P::kind, typename R::kind>;

/** @brief Points inside a closed box. */
template <class T, std::size_t D> struct inside_box {
    using kind = point_kind<T, D>;
    std::array<T, D> lo;
    std::array<T, D> hi;

    template <class B, class V> auto bind(const V& v) const noexcept {
        return [axis = v.axis, l = detail::broadcast<B>(lo), h = detail::broadcast<B>(hi)](const std::size_t i) noexcept {
            return detail::all_axes<D>([&](const std::size_t a) {
                const auto p{B::load(axis[a] + i)};
                return (l[a] <= p) & (p <= h[a]);
            });
        };
    }
};

/** @brief Boxes that overlap a closed box (touching counts). Boxes must be valid or the empty box (+inf, -inf). */
template <class T, std::size_t D> struct overlaps_box {
    using kind = box_kind<T, D>;
    std::array<T, D> lo;
    std::array<T, D> hi;

    template <class B, class V> auto bind(const V& v) const noexcept {
        return [mn = v.min, mx = v.max, l = detail::broadcast<B>(lo), h = detail::broadcast<B>(hi)](const std::size_t i) noexcept {
            return detail::all_axes<D>([&](const std::size_t a) { return (B::load(mn[a] + i) <= h[a]) & (l[a] <= B::load(mx[a] + i)); });
        };
    }
};

/** @brief Closed boxes that contain a point. */
template <class T, std::size_t D> struct contains_point {
    using kind = box_kind<T, D>;
    std::array<T, D> point;

    template <class B, class V> auto bind(const V& v) const noexcept {
        return [mn = v.min, mx = v.max, p = detail::broadcast<B>(point)](const std::size_t i) noexcept {
            return detail::all_axes<D>([&](const std::size_t a) { return (B::load(mn[a] + i) <= p[a]) & (p[a] <= B::load(mx[a] + i)); });
        };
    }
};

/** @brief Closed boxes hit by the ray origin + t * direction for some t in [t_min, t_max]. */
template <std::floating_point T, std::size_t D> struct ray_hit {
    using kind = box_kind<T, D>;
    std::array<T, D> origin;
    std::array<T, D> inv_dir;
    T t_min;
    T t_max;

    template <class B, class V> detail::ray_slabs<B, D> bind(const V& v) const noexcept {
        constexpr T lowest{std::numeric_limits<T>::lowest()};
        constexpr T highest{std::numeric_limits<T>::max()};
        detail::ray_slabs<B, D> s{{}, {}, detail::broadcast<B>(origin), detail::broadcast<B>(inv_dir), B::broadcast(std::clamp(t_min, lowest, highest)),
            B::broadcast(std::clamp(t_max, lowest, highest))};
        for (std::size_t a{}; a < D; ++a) {
            const bool negative{std::signbit(inv_dir[a])};
            s.near_plane[a] = negative ? v.max[a] : v.min[a];
            s.far_plane[a] = negative ? v.min[a] : v.max[a];
        }
        return s;
    }
};

template <predicate P, predicate Q> struct conjunction {
    using kind = typename P::kind;
    P p;
    Q q;

    template <class B, class V> auto bind(const V& v) const noexcept {
        return [pb = p.template bind<B>(v), qb = q.template bind<B>(v)](const std::size_t i) noexcept { return pb(i) & qb(i); };
    }
};

template <predicate P, predicate Q> struct disjunction {
    using kind = typename P::kind;
    P p;
    Q q;

    template <class B, class V> auto bind(const V& v) const noexcept {
        return [pb = p.template bind<B>(v), qb = q.template bind<B>(v)](const std::size_t i) noexcept { return pb(i) | qb(i); };
    }
};

template <predicate P> struct negation {
    using kind = typename P::kind;
    P p;

    template <class B, class V> auto bind(const V& v) const noexcept {
        return [pb = p.template bind<B>(v)](const std::size_t i) noexcept { return ~pb(i); };
    }
};

template <predicate P, predicate Q>
    requires std::same_as<typename P::kind, typename Q::kind>
constexpr conjunction<P, Q> operator&&(const P& p, const Q& q) noexcept {
    return {p, q};
}

template <predicate P, predicate Q>
    requires std::same_as<typename P::kind, typename Q::kind>
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

template <class T, std::size_t D> constexpr inside_box<T, D> inside(const aabb<T, D>& box) noexcept {
    return {box.min, box.max};
}

/** @brief Boxes overlapping @p box (closed). */
template <box_type Box> constexpr overlaps_box<float, detail::dimension_of<Box>> overlaps(const Box& box) noexcept {
    return {detail::coords(box.min), detail::coords(box.max)};
}

template <class T, std::size_t D> constexpr overlaps_box<T, D> overlaps(const aabb<T, D>& box) noexcept {
    return {box.min, box.max};
}

/** @brief Boxes containing @p point (closed). */
template <detail::vocab_point_type Vec> constexpr contains_point<detail::point_value_t<Vec>, detail::dimension_of<Vec>> contains(const Vec& point) noexcept {
    return {detail::coords(point)};
}

template <class T, std::size_t D> constexpr contains_point<T, D> contains(const std::array<T, D>& point) noexcept {
    return {point};
}

/**
 * @brief Boxes hit by a ray.
 *
 * @p direction need not be normalized; t is measured in units of it, so a segment [a, b] is
 * hit_by(a, b - a, 0, 1). A zero direction component is fine. t_min and t_max are clamped to
 * the finite range. A NaN coordinate in a box makes its result unspecified (but the same on
 * every backend).
 */
template <std::floating_point T, std::size_t D>
ray_hit<T, D> hit_by(
    const std::array<T, D>& origin, const std::array<T, D>& direction, const T t_min = 0, const T t_max = std::numeric_limits<T>::infinity()) noexcept {
    std::array<T, D> inv{};
    for (std::size_t a{}; a < D; ++a) {
        inv[a] = T{1} / direction[a];
    }
    return {origin, inv, t_min, t_max};
}

template <spatial_vector Vec>
ray_hit<float, detail::dimension_of<Vec>> hit_by(
    const Vec& origin, const Vec& direction, const float t_min = 0.0f, const float t_max = std::numeric_limits<float>::infinity()) noexcept {
    return hit_by(detail::coords(origin), detail::coords(direction), t_min, t_max);
}

/* ============================================================
 * Drivers: each takes a view or a buffer
 * ============================================================ */

/**
 * @brief Write one bit per element: bit k of out[w] is element 64 * w + k.
 *
 * @p out needs (size + 63) / 64 words; bits past the end of the range are written as zero.
 */
template <class R, predicate_for<R> P> void mask(const R& range, const P& pred, const std::span<std::uint64_t> out) noexcept {
    const auto view{detail::as_view(range)};
    assert(out.size() >= (view.size + block_size - 1) / block_size);
    detail::scan_words(view, pred, [out](const std::size_t w, const std::uint64_t bits) noexcept {
        out[w] = bits;
        return true;
    });
}

/** @brief Number of elements matching @p pred. */
template <class R, predicate_for<R> P> std::size_t count(const R& range, const P& pred) noexcept {
    std::size_t n{};
    detail::scan_words(detail::as_view(range), pred, [&n](std::size_t, const std::uint64_t bits) noexcept {
        n += static_cast<std::size_t>(std::popcount(bits));
        return true;
    });
    return n;
}

/** @brief Whether any element matches; stops at the first block of 64 with a match. */
template <class R, predicate_for<R> P> bool any(const R& range, const P& pred) noexcept {
    bool found{};
    detail::scan_words(detail::as_view(range), pred, [&found](std::size_t, const std::uint64_t bits) noexcept {
        found = static_cast<bool>(bits);
        return !found;
    });
    return found;
}

/** @brief Call @p fn with the index of every matching element, in increasing order. */
template <class R, predicate_for<R> P, std::invocable<std::size_t> F> void for_each_match(const R& range, const P& pred, F&& fn) {
    detail::scan_words(detail::as_view(range), pred, [&fn](const std::size_t w, std::uint64_t bits) {
        for (; bits; bits &= bits - 1) {
            fn(w * block_size + static_cast<std::size_t>(std::countr_zero(bits)));
        }
        return true;
    });
}

/**
 * @brief Write the indices of the matching elements to @p out, in increasing order, and
 *        return how many were written.
 *
 * @p out needs room for every match; @c size elements always suffice. A shorter span gets
 * the first out.size() matches. Entries past the returned count are unspecified: the
 * native path stores whole vectors. Ranges are limited to 2^32 elements.
 */
template <class R, predicate_for<R> P> std::size_t match_indices(const R& range, const P& pred, const std::span<std::uint32_t> out) noexcept {
    using V = decltype(detail::as_view(range));
    using wide = detail::batch<typename V::value_type, detail::native_isa>;
    using compactor = detail::index_compactor<detail::native_isa>;
    static_assert(!(compactor::width % wide::width), "a compactor store covers whole batches");
    constexpr std::size_t lanes{wide::width};
    constexpr std::size_t chunks{block_size / lanes};
    /* At or below this many matches in a block, walking the set bits beats storing every chunk. */
    constexpr int sparse_block{8};

    const V view{detail::as_view(range)};
    assert(static_cast<std::uint64_t>(view.size) <= std::uint64_t{1} << 32);
    const auto wide_pred{pred.template bind<wide>(view)};
    std::uint32_t* const dst{out.data()};
    const std::size_t room{out.size()};
    const std::size_t size{view.size};
    std::size_t n{};

    /* The dense path feeds each batch's mask straight to the compactor. The 64-bit word is
     * still built: one popcount for the decision and one bit walk for sparse blocks beat
     * doing either per batch. */
    std::size_t base{};
    for (; base + block_size <= size; base += block_size) {
        std::array<std::uint32_t, chunks> m{};
        std::uint64_t word{};
        for (std::size_t c{}; c < chunks; ++c) {
            m[c] = wide_pred(base + c * lanes).bits();
            word |= std::uint64_t{m[c]} << (c * lanes);
        }
        const auto first{static_cast<std::uint32_t>(base)};
        if (n + block_size > room) {
            /* Near the end of a short span: one match at a time, bounds checked. */
            for (; word && n < room; word &= word - 1) {
                dst[n++] = first + static_cast<std::uint32_t>(std::countr_zero(word));
            }
            if (n == room) {
                return n;
            }
        } else if (std::popcount(word) <= sparse_block) {
            for (; word; word &= word - 1) {
                dst[n++] = first + static_cast<std::uint32_t>(std::countr_zero(word));
            }
        } else {
            /* Stores whole chunks, up to a block past the count: covered by the room check above.
             * Batches narrower than the compactor (double) are fed from slices of the word. */
            if constexpr (compactor::width == lanes) {
                for (std::size_t c{}; c < chunks; ++c) {
                    n += compactor::store(dst + n, first + static_cast<std::uint32_t>(c * lanes), m[c]);
                }
            } else {
                constexpr std::uint64_t slice{(std::uint64_t{1} << compactor::width) - 1};
                for (std::size_t k{}; k < block_size; k += compactor::width) {
                    n += compactor::store(dst + n, first + static_cast<std::uint32_t>(k), static_cast<std::uint32_t>((word >> k) & slice));
                }
            }
        }
    }
    if (base < size) {
        const auto first{static_cast<std::uint32_t>(base)};
        for (std::uint64_t bits{detail::tail_word(view, pred, wide_pred, base)}; bits && n < room; bits &= bits - 1) {
            dst[n++] = first + static_cast<std::uint32_t>(std::countr_zero(bits));
        }
    }
    return n;
}

/**
 * @brief Ray entry distance per box: max(t_min, entry) on a hit, +inf on a miss.
 *
 * @p t_entry needs @c size elements; a shorter span is filled as far as it goes. Returns the
 * number of hits.
 */
template <box_range R, class T, std::size_t D>
    requires std::same_as<typename R::kind, box_kind<T, D>>
std::size_t hit_distances(const R& range, const ray_hit<T, D>& ray, const std::span<std::type_identity_t<T>> t_entry) noexcept {
    using V = decltype(detail::as_view(range));
    using wide = detail::batch<T, detail::native_isa>;
    constexpr std::size_t lanes{wide::width};

    const V view{detail::as_view(range)};
    assert(t_entry.size() >= view.size);
    /* The min also keeps GCC's -Warray-bounds from flagging the wide store on short spans. */
    const std::size_t n{std::min(view.size, t_entry.size())};
    T* const out{t_entry.data()};

    const auto wide_ray{ray.template bind<wide>(view)};
    const auto wide_miss{wide::broadcast(std::numeric_limits<T>::infinity())};

    std::size_t hits{};
    std::size_t i{};
    for (; i + lanes <= n; i += lanes) {
        const auto [t, hit]{wide_ray.entry(i)};
        select(hit, t, wide_miss).store(out + i);
        hits += static_cast<std::size_t>(std::popcount(hit.bits()));
    }
    if (i == n) {
        return hits;
    }

    if constexpr (detail::padded_for<V, wide>) {
        const auto [t, entered]{wide_ray.entry(i)};
        const auto hit{entered & wide::first_lanes(n - i)};
        std::array<T, lanes> tail{};
        select(hit, t, wide_miss).store(tail.data());
        std::copy_n(tail.begin(), n - i, out + i);
        hits += static_cast<std::size_t>(std::popcount(hit.bits()));
    } else {
        using narrow = detail::batch<T, detail::scalar_isa>;
        const auto narrow_ray{ray.template bind<narrow>(view)};
        const auto narrow_miss{narrow::broadcast(std::numeric_limits<T>::infinity())};
        for (; i < n; ++i) {
            const auto [t, hit]{narrow_ray.entry(i)};
            select(hit, t, narrow_miss).store(out + i);
            hits += hit.bits();
        }
    }
    return hits;
}

/**
 * @brief Axis-aligned bounds of the points. NaN coordinates are skipped.
 *
 * Returns the core box type for float and aabb<T, D> otherwise. An empty range gives the
 * inverted box min = +inf, max = -inf (the extreme values for integers), which is_valid
 * rejects and merge treats as the identity.
 */
template <point_range R> auto bounds(const R& range) noexcept {
    using T = typename R::value_type;
    constexpr std::size_t D{R::dimension};
    using V = decltype(detail::as_view(range));
    using wide = detail::batch<T, detail::native_isa>;
    constexpr std::size_t lanes{wide::width};

    const V view{detail::as_view(range)};
    const std::size_t n{view.size};

    /* Floating point keeps two accumulator sets to hide the min/max latency. Integer min/max
     * takes a cycle, and a second set would spill on AVX2 (12 accumulators plus loads). */
    auto lo0{detail::splat<wide, D>(detail::empty_min<T>)};
    auto hi0{detail::splat<wide, D>(detail::empty_max<T>)};
    auto lo1{lo0};
    auto hi1{hi0};

    std::size_t i{};
    for (; std::floating_point<T> && i + 2 * lanes <= n; i += 2 * lanes) {
        for (std::size_t a{}; a < D; ++a) {
            const auto x0{wide::load(view.axis[a] + i)};
            const auto x1{wide::load(view.axis[a] + i + lanes)};
            lo0[a] = min(x0, lo0[a]);
            hi0[a] = max(x0, hi0[a]);
            lo1[a] = min(x1, lo1[a]);
            hi1[a] = max(x1, hi1[a]);
        }
    }
    for (; i + lanes <= n; i += lanes) {
        for (std::size_t a{}; a < D; ++a) {
            const auto x{wide::load(view.axis[a] + i)};
            lo0[a] = min(x, lo0[a]);
            hi0[a] = max(x, hi0[a]);
        }
    }
    if constexpr (detail::padded_for<V, wide>) {
        if (i < n) {
            const auto keep{wide::first_lanes(n - i)};
            for (std::size_t a{}; a < D; ++a) {
                const auto x{wide::load(view.axis[a] + i)};
                lo1[a] = min(select(keep, x, wide::broadcast(detail::empty_min<T>)), lo1[a]);
                hi1[a] = max(select(keep, x, wide::broadcast(detail::empty_max<T>)), hi1[a]);
            }
            i = n;
        }
    }

    std::array<T, D> lo{};
    std::array<T, D> hi{};
    for (std::size_t a{}; a < D; ++a) {
        lo[a] = reduce_min(min(lo1[a], lo0[a]));
        hi[a] = reduce_max(max(hi1[a], hi0[a]));
        for (std::size_t j{i}; j < n; ++j) {
            const T x{view.axis[a][j]};
            lo[a] = x < lo[a] ? x : lo[a];
            hi[a] = x > hi[a] ? x : hi[a];
        }
    }
    return detail::to_box<T, D>(lo, hi);
}

} // namespace sgl::soa
