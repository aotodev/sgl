/**
 * @file types.cppm
 * @brief Core math types: vectors, matrices, quaternions, bounding boxes, and geometric result types.
 */
module;

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

export module sgl:types;

namespace sgl {

/* ============================================================
 * Integer vectors
 * ============================================================ */

/** @brief 2-component 32-bit integer vector. */
export struct ivec2 {
    std::int32_t x{}; ///< @brief x component.
    std::int32_t y{}; ///< @brief y component.
};

/** @brief 3-component 32-bit integer vector. */
export struct ivec3 {
    std::int32_t x{}; ///< @brief x component.
    std::int32_t y{}; ///< @brief y component.
    std::int32_t z{}; ///< @brief z component.
};

/* ============================================================
 * Float vectors
 * ============================================================ */

/** @brief 2-component single-precision float vector (8-byte aligned). */
export struct alignas(8) vec2 {
    float x{}; ///< @brief x component.
    float y{}; ///< @brief y component.

    constexpr float operator[](std::size_t i) const noexcept {
        assert(i < 2);
        return (&x)[i];
    }

    constexpr float& operator[](std::size_t i) noexcept {
        assert(i < 2);
        return (&x)[i];
    }
};

/**
 * @brief 3-component single-precision float vector (16-byte aligned).
 *
 * Contains an explicit @c pad lane so the struct occupies a full 16-byte
 * SSE register. SIMD stores may write to @c pad; callers must treat it
 * as reserved and always zero.
 */
export struct alignas(16) vec3 {
    float x{}; ///< @brief x component.
    float y{}; ///< @brief y component.
    float z{}; ///< @brief z component.

    [[maybe_unused]] float pad{}; /**< @brief Reserved. Written by 128-bit SIMD stores, must stay zero. */

    constexpr float operator[](std::size_t i) const noexcept {
        assert(i < 3);
        return (&x)[i];
    }

    constexpr float& operator[](std::size_t i) noexcept {
        assert(i < 3);
        return (&x)[i];
    }
};

/** @brief 4-component single-precision float vector (16-byte aligned). */
export struct alignas(16) vec4 {
    float x{}; ///< @brief x component.
    float y{}; ///< @brief y component.
    float z{}; ///< @brief z component.
    float w{}; ///< @brief w component.

    constexpr float operator[](std::size_t i) const noexcept {
        assert(i < 4);
        return (&x)[i];
    }

    constexpr float& operator[](std::size_t i) noexcept {
        assert(i < 4);
        return (&x)[i];
    }
};

/* ============================================================
 * Matrices (column-major)
 * ============================================================ */

/**
 * @brief Column-major 2×2 matrix.
 *
 * Layout: @c cols[0] = first column = {a, c}, @c cols[1] = {b, d}.
 * @code
 *   | a  b |
 *   | c  d |
 * @endcode
 */
export struct alignas(16) mat2 {
    vec2 cols[2]{}; /**< @brief Column vectors: cols[0] = first column, cols[1] = second column. */
};

/**
 * @brief Column-major 3×3 matrix.
 *
 * Each column is a @c vec3 with an implicit pad lane to allow aligned
 * 128-bit SIMD loads. The pad lane of each column is always zero.
 */
export struct alignas(16) mat3 {
    vec3 cols[3]{}; /**< @brief Column vectors. */
};

/**
 * @brief Column-major 4×4 matrix (64-byte aligned for cache-line friendliness).
 *
 * Standard column-major layout: @c cols[j][i] = element at row @c i, column @c j.
 */
export struct alignas(64) mat4 {
    vec4 cols[4]{}; /**< @brief Column vectors: cols[j] is the j-th column. */
};

/* ============================================================
 * Quaternion
 * ============================================================ */

/**
 * @brief Unit quaternion representing a 3D rotation: q = xi + yj + zk + w.
 *
 * The default-constructed quaternion is the identity rotation (w = 1, xyz = 0).
 * All operations in this library assume quaternions are unit-length; use
 * @c normalized(q) to re-normalise after many accumulated multiplications.
 */
export struct alignas(16) quat {
    float x{};     ///< @brief x component.
    float y{};     ///< @brief y component.
    float z{};     ///< @brief z component.
    float w{1.0f}; ///< @brief w component.
};

/* ============================================================
 * Axis-aligned bounds
 * ============================================================ */

/**
 * @brief Axis-aligned bounding box in 2D.
 *
 * Stored as a packed pair of vec2 so the entire box fits in one 128-bit
 * SSE register: memory layout is @c {min.x, min.y, max.x, max.y}.
 * A box is considered valid when @c min <= max element-wise.
 */
export struct alignas(16) box2d {
    vec2 min{}; /**< @brief Minimum corner. */
    vec2 max{}; /**< @brief Maximum corner. */
};

/**
 * @brief Axis-aligned bounding box in 3D.
 *
 * Each corner is a @c vec3 with an implicit pad lane.
 * A box is considered valid when @c min <= max element-wise.
 */
export struct alignas(16) box3d {
    vec3 min{}; /**< @brief Minimum corner. */
    vec3 max{}; /**< @brief Maximum corner. */
};

/**
 * @brief Plane in Hessian normal form: dot(normal, x) = d.
 *
 * The @c normal must be unit-length for signed-distance queries to return
 * meaningful metric distances.
 */
export struct plane {
    vec3 normal{}; /**< @brief Unit outward normal. */
    float d{};     /**< @brief Signed distance from the origin to the plane along the normal. */
};

/* ============================================================
 * Geometric result types
 * ============================================================ */

/**
 * @brief Orthonormal basis on a plane, suitable for projecting between 3D and 2D.
 *
 * The three vectors @c {u, v, normal} form a right-handed orthonormal frame
 * anchored at @c origin. Project a 3D point @c p onto the plane with
 * @c {dot(p - origin, u), dot(p - origin, v)}.
 */
export struct plane_basis {
    vec3 origin{}; /**< @brief A point on the plane. */
    vec3 u{};      /**< @brief First tangent axis (unit length). */
    vec3 v{};      /**< @brief Second tangent axis (unit length, perpendicular to u and normal). */
    vec3 normal{}; /**< @brief Outward normal (unit length). */
};

/** @brief Result of a 2D segment-segment intersection test. */
export struct segment_intersection_2d {
    vec2 point{};      /**< @brief Intersection point (valid only when @c intersects is true). */
    float t{};         /**< @brief Parametric position on the first segment [0, 1]. */
    float u{};         /**< @brief Parametric position on the second segment [0, 1]. */
    bool intersects{}; /**< @brief True when the segments cross. */
};

/**
 * @brief Result of a ray-AABB intersection (slab method).
 *
 * @c t_entry may be negative when the ray origin is inside the box.
 * A hit requires @c t_entry <= t_exit and @c t_exit >= 0.
 */
export struct ray_box_hit {
    float t_entry{}; /**< @brief Parametric distance to the entry face (may be negative if origin is inside). */
    float t_exit{};  /**< @brief Parametric distance to the exit face. */
    bool hit{};      /**< @brief True when the ray intersects the box. */
};

/**
 * @brief Nearest intersection of a query segment against a polygon edge (2D).
 *
 * The segment @c [a, b] is tested against all polygon edges; only the hit
 * with the smallest @c t is returned.
 */
export struct segment_polygon_hit {
    vec2 point{};                /**< @brief Intersection point (valid when @c hit is true). */
    float t{};                   /**< @brief Parametric position on the query segment [0, 1]. */
    std::int32_t edge_index{-1}; /**< @brief Index of the first vertex of the hit edge, or -1 if no hit. */
    bool hit{};                  /**< @brief True when an intersection was found. */
};

/* ============================================================
 * Grid types
 * ============================================================ */

/** @brief Inclusive integer cell range for a 2D spatial grid. */
export struct grid_cell_range_2d {
    ivec2 min{}; /**< @brief First cell (inclusive). */
    ivec2 max{}; /**< @brief Last cell (inclusive). */
};

/** @brief Inclusive integer cell range for a 3D spatial grid. */
export struct grid_cell_range_3d {
    ivec3 min{}; /**< @brief First cell (inclusive). */
    ivec3 max{}; /**< @brief Last cell (inclusive). */
};

export struct segment_2d {
    vec2 start;
    vec2 end;
};

export struct segment_3d {
    vec3 start;
    vec3 end;
};

/* ============================================================
 * Box pair helpers
 * ============================================================ */

/** @brief Pair of 2D AABBs, typically produced by subdivision operations. */
export struct box2d_pair {
    box2d first{};  /**< @brief Lower half after split. */
    box2d second{}; /**< @brief Upper half after split. */
};

/** @brief Pair of 3D AABBs, typically produced by subdivision operations. */
export struct box3d_pair {
    box3d first{};  /**< @brief Lower half after split. */
    box3d second{}; /**< @brief Upper half after split. */
};

/**
 * @brief Represents a rigid transform with uniform scale: T(p) = R(s * p) + t
 *
 * Storage format matching Unreal's FTransform layout conceptually.
 * For actual computation (transforming points), convert to mat4.
 *
 * Memory layout: 3 * 16 bytes = 48 bytes
 *   - position: vec3 (16 bytes, padded)
 *   - rotation: quat (16 bytes)
 *   - scale:    vec3 (16 bytes, padded)
 */
export struct alignas(16) transform {
    vec3 position{};
    quat rotation{0.0f, 0.0f, 0.0f, 1.0f}; /* identity */
    vec3 scale{1.0f, 1.0f, 1.0f};
};

/* ============================================================
 * Concepts
 * ============================================================ */
template <typename T, typename... Ts>
concept one_of_decayed = (std::same_as<std::decay_t<T>, Ts> || ...);

/** @brief Matches @c box2d or @c box3d. */
template <typename Box>
concept box_type = one_of_decayed<Box, box2d, box3d>;

/** @brief Matches @c vec2, @c vec3, or @c vec4. */
template <typename Vec>
concept vector = one_of_decayed<Vec, vec2, vec3, vec4>;

/** @brief Matches @c vec2 or @c vec3 — vectors with a well-defined Euclidean geometry (excludes projective @c vec4). */
template <typename Vec>
concept spatial_vector = one_of_decayed<Vec, vec2, vec3>;

/* ============================================================
 * Useful constants
 * ============================================================ */

/** @brief Identity quaternion — represents no rotation. */
export constexpr quat quat_identity{0.0f, 0.0f, 0.0f, 1.0f};

/** @brief 2×2 identity matrix. */
export constexpr mat2 mat2_identity{.cols = {vec2{1.0f, 0.0f}, vec2{0.0f, 1.0f}}};

/** @brief 3×3 identity matrix. */
export constexpr mat3 mat3_identity{.cols = {vec3{1.0f, 0.0f, 0.0f}, vec3{0.0f, 1.0f, 0.0f}, vec3{0.0f, 0.0f, 1.0f}}};

/** @brief 4×4 identity matrix. */
export constexpr mat4 mat4_identity{
    .cols = {vec4{1.0f, 0.0f, 0.0f, 0.0f}, vec4{0.0f, 1.0f, 0.0f, 0.0f}, vec4{0.0f, 0.0f, 1.0f, 0.0f}, vec4{0.0f, 0.0f, 0.0f, 1.0f}}};

/** @brief Zero vec2. */
export constexpr vec2 vec2_zero{0.0f, 0.0f};
/** @brief Ones vec2. */
export constexpr vec2 vec2_one{1.0f, 1.0f};
/** @brief vec2 unit X axis. */
export constexpr vec2 vec2_right{1.0f, 0.0f};
/** @brief vec2 unit Y axis. */
export constexpr vec2 vec2_up{0.0f, 1.0f};

/** @brief Zero vec3. */
export constexpr vec3 vec3_zero{0.0f, 0.0f, 0.0f};
/** @brief Ones vec3. */
export constexpr vec3 vec3_one{1.0f, 1.0f, 1.0f};
/** @brief Forward direction: +X axis. */
export constexpr vec3 vec3_forward{1.0f, 0.0f, 0.0f};
/** @brief Backward direction: -X axis. */
export constexpr vec3 vec3_backward{-1.0f, 0.0f, 0.0f};
/** @brief Right direction: +Y axis. */
export constexpr vec3 vec3_right{0.0f, 1.0f, 0.0f};
/** @brief Left direction: -Y axis. */
export constexpr vec3 vec3_left{0.0f, -1.0f, 0.0f};
/** @brief Up direction: +Z axis. */
export constexpr vec3 vec3_up{0.0f, 0.0f, 1.0f};
/** @brief Down direction: -Z axis. */
export constexpr vec3 vec3_down{0.0f, 0.0f, -1.0f};

/** @brief Zero vec4. */
export constexpr vec4 vec4_zero{0.0f, 0.0f, 0.0f, 0.0f};
/** @brief Ones vec4. */
export constexpr vec4 vec4_one{1.0f, 1.0f, 1.0f, 1.0f};

} // namespace sgl
