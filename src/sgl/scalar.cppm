/**
 * @file scalar.cppm
 * @brief Scalar and vector floating-point comparisons, directional helpers, and box subdivision.
 */
module;

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

export module sgl:scalar;

import :types;

export namespace sgl {

/* ============================================================
 * Floating-point tolerances
 * ============================================================ */

constexpr float fp32_abs_tol{1e-6f};

/* ~1.52e-5 — allows ~7 bits of accumulated rounding error */
constexpr float fp32_rel_tol{128.0f * std::numeric_limits<float>::epsilon()};

/* ============================================================
 * Scalar comparisons
 * ============================================================ */

/**
 * Returns true when lhs and rhs agree within an absolute or relative tolerance.
 * Bitwise equality is checked first to handle ±inf and ±0 correctly.
 */
inline constexpr bool nearly_equal(const float lhs, const float rhs, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    if (lhs == rhs) {
        return true;
    }

    const auto diff{std::abs(lhs - rhs)};

    /* absolute check handles values near zero */
    if (diff <= abs_tol) {
        return true;
    }

    /* relative check for everything else */
    const auto largest{std::max(std::abs(lhs), std::abs(rhs))};
    return diff <= largest * rel_tol;
}

inline constexpr bool nearly_zero(const float val, const float tolerance = fp32_abs_tol) noexcept {
    return std::abs(val) < tolerance;
}

/* ============================================================
 * Vector comparisons
 * ============================================================ */

inline constexpr bool nearly_equal(const vec2 lhs, const vec2 rhs, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    return nearly_equal(lhs.x, rhs.x, abs_tol, rel_tol) && nearly_equal(lhs.y, rhs.y, abs_tol, rel_tol);
}

inline constexpr bool nearly_equal(const vec3 lhs, const vec3 rhs, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    return nearly_equal(lhs.x, rhs.x, abs_tol, rel_tol) && nearly_equal(lhs.y, rhs.y, abs_tol, rel_tol) && nearly_equal(lhs.z, rhs.z, abs_tol, rel_tol);
}

inline constexpr bool nearly_equal(const vec4 lhs, const vec4 rhs, const float abs_tol = fp32_abs_tol, const float rel_tol = fp32_rel_tol) noexcept {
    return nearly_equal(lhs.x, rhs.x, abs_tol, rel_tol) && nearly_equal(lhs.y, rhs.y, abs_tol, rel_tol) && nearly_equal(lhs.z, rhs.z, abs_tol, rel_tol) &&
           nearly_equal(lhs.w, rhs.w, abs_tol, rel_tol);
}

inline constexpr bool nearly_zero(const vec2 val, const float tolerance = fp32_abs_tol) noexcept {
    return nearly_zero(val.x, tolerance) && nearly_zero(val.y, tolerance);
}

inline constexpr bool nearly_zero(const vec3 val, const float tolerance = fp32_abs_tol) noexcept {
    return nearly_zero(val.x, tolerance) && nearly_zero(val.y, tolerance) && nearly_zero(val.z, tolerance);
}

inline constexpr bool nearly_zero(const vec4 val, const float tolerance = fp32_abs_tol) noexcept {
    return nearly_zero(val.x, tolerance) && nearly_zero(val.y, tolerance) && nearly_zero(val.z, tolerance) && nearly_zero(val.w, tolerance);
}

/* ============================================================
 * Directional helpers
 * ============================================================ */

/* Returns the unit cardinal axis (+x, +y, or +z) closest to `direction`, sign-preserved. */
inline constexpr vec3 snap_to_axis(const vec3& direction) noexcept {
    const auto ax{std::abs(direction.x)};
    const auto ay{std::abs(direction.y)};
    const auto az{std::abs(direction.z)};

    if (ax >= ay && ax >= az) {
        return {std::copysign(1.0f, direction.x), 0.0f, 0.0f};
    }
    if (ay >= az) {
        return {0.0f, std::copysign(1.0f, direction.y), 0.0f};
    }
    return {0.0f, 0.0f, std::copysign(1.0f, direction.z)};
}

/**
 * Build an orthonormal basis for a plane given its origin and unit normal.
 *
 * Uses the Frisvad / Duff et al. method: derives sign from n.z to avoid the
 * singularity at n = (0, 0, -1), then constructs {u, v} analytically without
 * a branch or a cross product.
 */
inline constexpr plane_basis build_plane_basis(const vec3 plane_origin, const vec3 plane_normal) noexcept {
    const auto nx{plane_normal.x};
    const auto ny{plane_normal.y};
    const auto nz{plane_normal.z};

    const auto sign{std::copysign(1.0f, nz)};
    const auto a{-1.0f / (sign + nz)};
    const auto b{nx * ny * a};

    const vec3 u{1.0f + sign * nx * nx * a, sign * b, -sign * nx};
    const vec3 v{b, sign + ny * ny * a, -ny};

    return {plane_origin, u, v, plane_normal};
}

/* ============================================================
 * Box subdivision
 * ============================================================ */

/* Split a 2D box at parametric position `t` along `axis` (0 = x, 1 = y).
 * Returns {lower half, upper half}. */
inline box2d_pair subdivide(const box2d& box, const std::uint32_t axis, const float t) noexcept {
    const auto split{box.min[axis] + t * (box.max[axis] - box.min[axis])};
    auto first{box};
    auto second{box};
    first.max[axis] = split;
    second.min[axis] = split;
    return {first, second};
}

/* Split a 3D box at parametric position `t` along `axis` (0 = x, 1 = y, 2 = z).
 * Returns {lower half, upper half}. */
inline box3d_pair subdivide(const box3d& box, const std::uint32_t axis, const float t) noexcept {
    const auto split{box.min[axis] + t * (box.max[axis] - box.min[axis])};
    auto first{box};
    auto second{box};
    first.max[axis] = split;
    second.min[axis] = split;
    return {first, second};
}

/* ============================================================
 * 2D polygon geometry (Shoelace / Gauss formula)
 * ============================================================ */

/* Signed area; positive for CCW winding, negative for CW. */
inline float polygon_area_signed(std::span<const vec2> pts) noexcept {
    float sum{};

    const auto n{pts.size()};
    for (std::size_t i{}; i < n; ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }

    return sum * 0.5f;
}

/* Unsigned area; works for CW or CCW vertex order. */
inline float polygon_area(std::span<const vec2> pts) noexcept {
    return std::abs(polygon_area_signed(pts));
}

/* Sum of all edge lengths of a closed polygon. */
inline float polygon_perimeter(std::span<const vec2> pts) noexcept {
    float len{0.0f};

    const auto n{pts.size()};
    for (std::size_t i{0}; i < n; ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % n];

        const auto dx{b.x - a.x};
        const auto dy{b.y - a.y};

        len += std::sqrt(dx * dx + dy * dy);
    }
    return len;
}

} // namespace sgl
