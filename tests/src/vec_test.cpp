/**
 * @file vec_test.cpp
 * @brief Unit tests for vector operations (vec2, vec3, vec4).
 */
#include <cmath>
#include <gtest/gtest.h>

import sgl;

/* --- nearly_equal / nearly_zero --- */

TEST(Vec2, NearlyEqual) {
    EXPECT_TRUE(sgl::nearly_equal(sgl::vec2{1.0f, 2.0f}, sgl::vec2{1.0f, 2.0f}));
    EXPECT_FALSE(sgl::nearly_equal(sgl::vec2{1.0f, 2.0f}, sgl::vec2{1.0f, 3.0f}));
}
TEST(Vec3, NearlyEqual) {
    EXPECT_TRUE(sgl::nearly_equal(sgl::vec3{1, 2, 3}, sgl::vec3{1, 2, 3}));
    EXPECT_FALSE(sgl::nearly_equal(sgl::vec3{1, 2, 3}, sgl::vec3{1, 2, 4}));
}
TEST(Vec4, NearlyEqual) {
    EXPECT_TRUE(sgl::nearly_equal(sgl::vec4{1, 2, 3, 4}, sgl::vec4{1, 2, 3, 4}));
}

TEST(Vec2, NearlyZero) {
    EXPECT_TRUE(sgl::nearly_zero(sgl::vec2{0, 0}));
    EXPECT_FALSE(sgl::nearly_zero(sgl::vec2{1, 0}));
}
TEST(Vec3, NearlyZero) {
    EXPECT_TRUE(sgl::nearly_zero(sgl::vec3{0, 0, 0}));
    EXPECT_FALSE(sgl::nearly_zero(sgl::vec3{0, 1, 0}));
}

TEST(Vec2, AnyNearlyZero) {
    EXPECT_TRUE(sgl::any_nearly_zero(sgl::vec2{0, 5}));
    EXPECT_FALSE(sgl::any_nearly_zero(sgl::vec2{1, 2}));
}
TEST(Vec3, AnyNearlyZero) {
    EXPECT_TRUE(sgl::any_nearly_zero(sgl::vec3{1, 0, 3}));
    EXPECT_FALSE(sgl::any_nearly_zero(sgl::vec3{1, 2, 3}));
}

/* --- Unary negate --- */

TEST(Vec2, Negate) {
    auto r = -sgl::vec2{1, -2};
    EXPECT_FLOAT_EQ(r.x, -1);
    EXPECT_FLOAT_EQ(r.y, 2);
}
TEST(Vec3, Negate) {
    auto r = -sgl::vec3{1, -2, 3};
    EXPECT_FLOAT_EQ(r.x, -1);
    EXPECT_FLOAT_EQ(r.y, 2);
    EXPECT_FLOAT_EQ(r.z, -3);
}

/* --- Vec-vec arithmetic --- */

TEST(Vec3, Add) {
    auto r = sgl::vec3{1, 2, 3} + sgl::vec3{4, 5, 6};
    EXPECT_FLOAT_EQ(r.x, 5);
    EXPECT_FLOAT_EQ(r.y, 7);
    EXPECT_FLOAT_EQ(r.z, 9);
}
TEST(Vec3, Sub) {
    auto r = sgl::vec3{4, 5, 6} - sgl::vec3{1, 2, 3};
    EXPECT_FLOAT_EQ(r.x, 3);
    EXPECT_FLOAT_EQ(r.y, 3);
    EXPECT_FLOAT_EQ(r.z, 3);
}
TEST(Vec3, Mul) {
    auto r = sgl::vec3{2, 3, 4} * sgl::vec3{5, 6, 7};
    EXPECT_FLOAT_EQ(r.x, 10);
    EXPECT_FLOAT_EQ(r.y, 18);
    EXPECT_FLOAT_EQ(r.z, 28);
}
TEST(Vec3, Div) {
    auto r = sgl::vec3{10, 12, 14} / sgl::vec3{2, 3, 7};
    EXPECT_FLOAT_EQ(r.x, 5);
    EXPECT_FLOAT_EQ(r.y, 4);
    EXPECT_FLOAT_EQ(r.z, 2);
}

TEST(Vec2, AddAssign) {
    sgl::vec2 v{1, 2};
    v += sgl::vec2{3, 4};
    EXPECT_FLOAT_EQ(v.x, 4);
    EXPECT_FLOAT_EQ(v.y, 6);
}
TEST(Vec3, SubAssign) {
    sgl::vec3 v{5, 6, 7};
    v -= sgl::vec3{1, 2, 3};
    EXPECT_FLOAT_EQ(v.x, 4);
    EXPECT_FLOAT_EQ(v.y, 4);
    EXPECT_FLOAT_EQ(v.z, 4);
}
TEST(Vec3, MulAssign) {
    sgl::vec3 v{2, 3, 4};
    v *= sgl::vec3{2, 2, 2};
    EXPECT_FLOAT_EQ(v.x, 4);
    EXPECT_FLOAT_EQ(v.y, 6);
    EXPECT_FLOAT_EQ(v.z, 8);
}
TEST(Vec3, DivAssign) {
    sgl::vec3 v{10, 12, 14};
    v /= sgl::vec3{2, 3, 7};
    EXPECT_FLOAT_EQ(v.x, 5);
    EXPECT_FLOAT_EQ(v.y, 4);
    EXPECT_FLOAT_EQ(v.z, 2);
}

/* --- Vec-scalar arithmetic --- */

TEST(Vec3, AddScalar) {
    auto r = sgl::vec3{1, 2, 3} + 10.0f;
    EXPECT_FLOAT_EQ(r.x, 11);
    EXPECT_FLOAT_EQ(r.y, 12);
    EXPECT_FLOAT_EQ(r.z, 13);
}
TEST(Vec3, SubScalar) {
    auto r = sgl::vec3{10, 20, 30} - 5.0f;
    EXPECT_FLOAT_EQ(r.x, 5);
    EXPECT_FLOAT_EQ(r.y, 15);
    EXPECT_FLOAT_EQ(r.z, 25);
}
TEST(Vec3, MulScalar) {
    auto r = sgl::vec3{1, 2, 3} * 3.0f;
    EXPECT_FLOAT_EQ(r.x, 3);
    EXPECT_FLOAT_EQ(r.y, 6);
    EXPECT_FLOAT_EQ(r.z, 9);
}
TEST(Vec3, DivScalar) {
    auto r = sgl::vec3{6, 9, 12} / 3.0f;
    EXPECT_FLOAT_EQ(r.x, 2);
    EXPECT_FLOAT_EQ(r.y, 3);
    EXPECT_FLOAT_EQ(r.z, 4);
}

TEST(Vec3, ScalarAddCommutative) {
    auto r = 10.0f + sgl::vec3{1, 2, 3};
    EXPECT_FLOAT_EQ(r.x, 11);
}
TEST(Vec3, ScalarMulCommutative) {
    auto r = 3.0f * sgl::vec3{1, 2, 3};
    EXPECT_FLOAT_EQ(r.z, 9);
}
TEST(Vec3, ScalarSubNonCommutative) {
    auto r = 10.0f - sgl::vec3{1, 2, 3};
    EXPECT_FLOAT_EQ(r.x, 9);
    EXPECT_FLOAT_EQ(r.z, 7);
}
TEST(Vec3, ScalarDivNonCommutative) {
    auto r = 12.0f / sgl::vec3{3, 4, 6};
    EXPECT_FLOAT_EQ(r.x, 4);
    EXPECT_FLOAT_EQ(r.y, 3);
    EXPECT_FLOAT_EQ(r.z, 2);
}

/* --- Dot / length --- */

TEST(Vec2, Dot) {
    EXPECT_FLOAT_EQ(sgl::dot(sgl::vec2{1, 2}, sgl::vec2{3, 4}), 11.0f);
}
TEST(Vec3, Dot) {
    EXPECT_FLOAT_EQ(sgl::dot(sgl::vec3{1, 2, 3}, sgl::vec3{4, 5, 6}), 32.0f);
}
TEST(Vec4, Dot) {
    EXPECT_FLOAT_EQ(sgl::dot(sgl::vec4{1, 2, 3, 4}, sgl::vec4{5, 6, 7, 8}), 70.0f);
}

TEST(Vec3, LengthSquared) {
    EXPECT_FLOAT_EQ(sgl::length_squared(sgl::vec3{3, 4, 0}), 25.0f);
}
TEST(Vec3, Length) {
    EXPECT_FLOAT_EQ(sgl::length(sgl::vec3{3, 4, 0}), 5.0f);
}
TEST(Vec2, Length) {
    EXPECT_FLOAT_EQ(sgl::length(sgl::vec2{3, 4}), 5.0f);
}

/* --- Normalize --- */

TEST(Vec3, Normalize) {
    sgl::vec3 v{3, 0, 0};
    sgl::normalize(v);
    EXPECT_NEAR(v.x, 1.0f, 1e-6f);
    EXPECT_NEAR(v.y, 0.0f, 1e-6f);
}
TEST(Vec3, Normalized) {
    auto r = sgl::normalized(sgl::vec3{0, 4, 0});
    EXPECT_NEAR(r.y, 1.0f, 1e-6f);
}
TEST(Vec3, NormalizeNR) {
    sgl::vec3 v{0, 0, 5};
    sgl::normalize_nr(v);
    EXPECT_NEAR(v.z, 1.0f, 1e-5f);
}
TEST(Vec3, NormalizedFast) {
    auto r = sgl::normalized_fast(sgl::vec3{3, 0, 0});
    EXPECT_NEAR(r.x, 1.0f, 0.01f);
}

TEST(Vec3, NormalizeSafe_NonZero) {
    sgl::vec3 v{3, 0, 0};
    sgl::normalize_safe(v);
    EXPECT_NEAR(v.x, 1.0f, 1e-5f);
}
TEST(Vec3, NormalizeSafe_Zero) {
    sgl::vec3 v{0, 0, 0};
    sgl::normalize_safe(v);
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}
TEST(Vec3, NormalizedSafe_Zero) {
    auto r = sgl::normalized_safe(sgl::vec3{0, 0, 0});
    EXPECT_FLOAT_EQ(r.x, 0.0f);
}

TEST(Vec2, Normalized) {
    auto r = sgl::normalized(sgl::vec2{0, 3});
    EXPECT_NEAR(r.x, 0.0f, 1e-6f);
    EXPECT_NEAR(r.y, 1.0f, 1e-6f);
}

/* --- Project / Reflect --- */

TEST(Vec3, Project) {
    auto r = sgl::project(sgl::vec3{3, 4, 0}, sgl::vec3{1, 0, 0});
    EXPECT_NEAR(r.x, 3.0f, 1e-6f);
    EXPECT_NEAR(r.y, 0.0f, 1e-6f);
}
TEST(Vec3, Reflect) {
    auto r = sgl::reflect(sgl::vec3{1, -1, 0}, sgl::vec3{0, 1, 0});
    EXPECT_NEAR(r.x, 1.0f, 1e-6f);
    EXPECT_NEAR(r.y, 1.0f, 1e-6f);
}

/* --- Min/Max/Abs/Clamp/Lerp --- */

TEST(Vec3, Min) {
    auto r = sgl::min(sgl::vec3{1, 5, 3}, sgl::vec3{4, 2, 6});
    EXPECT_FLOAT_EQ(r.x, 1);
    EXPECT_FLOAT_EQ(r.y, 2);
    EXPECT_FLOAT_EQ(r.z, 3);
}
TEST(Vec3, Max) {
    auto r = sgl::max(sgl::vec3{1, 5, 3}, sgl::vec3{4, 2, 6});
    EXPECT_FLOAT_EQ(r.x, 4);
    EXPECT_FLOAT_EQ(r.y, 5);
    EXPECT_FLOAT_EQ(r.z, 6);
}
TEST(Vec3, Abs) {
    auto r = sgl::abs(sgl::vec3{-1, 2, -3});
    EXPECT_FLOAT_EQ(r.x, 1);
    EXPECT_FLOAT_EQ(r.y, 2);
    EXPECT_FLOAT_EQ(r.z, 3);
}

TEST(Vec3, ClampVec) {
    auto r = sgl::clamp(sgl::vec3{-1, 5, 2}, sgl::vec3{0, 0, 0}, sgl::vec3{3, 3, 3});
    EXPECT_FLOAT_EQ(r.x, 0);
    EXPECT_FLOAT_EQ(r.y, 3);
    EXPECT_FLOAT_EQ(r.z, 2);
}
TEST(Vec3, ClampScalar) {
    auto r = sgl::clamp(sgl::vec3{-1, 5, 2}, 0.0f, 3.0f);
    EXPECT_FLOAT_EQ(r.x, 0);
    EXPECT_FLOAT_EQ(r.y, 3);
    EXPECT_FLOAT_EQ(r.z, 2);
}
TEST(Vec3, Lerp) {
    auto r = sgl::lerp(sgl::vec3{0, 0, 0}, sgl::vec3{10, 20, 30}, 0.5f);
    EXPECT_FLOAT_EQ(r.x, 5);
    EXPECT_FLOAT_EQ(r.y, 10);
    EXPECT_FLOAT_EQ(r.z, 15);
}

/* --- Distance --- */

TEST(Vec3, DistanceSquared) {
    EXPECT_FLOAT_EQ(sgl::distance_squared(sgl::vec3{0, 0, 0}, sgl::vec3{3, 4, 0}), 25.0f);
}
TEST(Vec3, Distance) {
    EXPECT_FLOAT_EQ(sgl::distance(sgl::vec3{0, 0, 0}, sgl::vec3{3, 4, 0}), 5.0f);
}

/* --- Perpendicular / Parallel --- */

TEST(Vec3, IsPerpendicular) {
    EXPECT_TRUE(sgl::is_perpendicular(sgl::vec3{1, 0, 0}, sgl::vec3{0, 1, 0}));
    EXPECT_FALSE(sgl::is_perpendicular(sgl::vec3{1, 0, 0}, sgl::vec3{1, 1, 0}));
}
TEST(Vec3, IsParallel) {
    EXPECT_TRUE(sgl::is_parallel(sgl::vec3{1, 0, 0}, sgl::vec3{2, 0, 0}));
    EXPECT_FALSE(sgl::is_parallel(sgl::vec3{1, 0, 0}, sgl::vec3{0, 1, 0}));
}
TEST(Vec2, IsParallel) {
    EXPECT_TRUE(sgl::is_parallel(sgl::vec2{1, 2}, sgl::vec2{2, 4}));
    EXPECT_FALSE(sgl::is_parallel(sgl::vec2{1, 0}, sgl::vec2{0, 1}));
}

/* --- Comparison --- */

TEST(Vec3, AllGreaterThan) {
    EXPECT_TRUE(sgl::all_greater_than(sgl::vec3{2, 3, 4}, sgl::vec3{1, 2, 3}));
    EXPECT_FALSE(sgl::all_greater_than(sgl::vec3{2, 2, 4}, sgl::vec3{2, 2, 3}));
}
TEST(Vec3, AllLessThan) {
    EXPECT_TRUE(sgl::all_less_than(sgl::vec3{1, 2, 3}, sgl::vec3{2, 3, 4}));
}
TEST(Vec3, AnyGreaterThan) {
    EXPECT_TRUE(sgl::any_greater_than(sgl::vec3{0, 0, 4}, sgl::vec3{1, 2, 3}));
    EXPECT_FALSE(sgl::any_greater_than(sgl::vec3{0, 0, 0}, sgl::vec3{1, 2, 3}));
}

/* --- Cross product --- */

TEST(Vec2, Cross) {
    EXPECT_FLOAT_EQ(sgl::cross(sgl::vec2{1, 0}, sgl::vec2{0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(sgl::cross(sgl::vec2{0, 1}, sgl::vec2{1, 0}), -1.0f);
}

TEST(Vec3, Cross) {
    auto r = sgl::cross(sgl::vec3{1, 0, 0}, sgl::vec3{0, 1, 0});
    EXPECT_NEAR(r.x, 0.0f, 1e-6f);
    EXPECT_NEAR(r.y, 0.0f, 1e-6f);
    EXPECT_NEAR(r.z, 1.0f, 1e-6f);
}

/* --- Perpendicular --- */

TEST(Vec2, Perpendicular) {
    auto r = sgl::perpendicular(sgl::vec2{1, 0});
    EXPECT_FLOAT_EQ(r.x, 0);
    EXPECT_FLOAT_EQ(r.y, 1);
    EXPECT_FLOAT_EQ(sgl::dot(sgl::vec2{1, 0}, r), 0.0f);
}

TEST(Vec3, Perpendicular) {
    auto r = sgl::perpendicular(sgl::vec3{1, 0, 0});
    EXPECT_NEAR(sgl::dot(sgl::vec3{1, 0, 0}, r), 0.0f, 1e-6f);
}

/* --- Angle between --- */

TEST(Vec3, AngleBetween) {
    auto a = sgl::angle_between(sgl::vec3{1, 0, 0}, sgl::vec3{0, 1, 0});
    EXPECT_NEAR(a, std::acos(0.0f), 1e-5f); /* π/2 */
}

TEST(Vec2, AngleBetween) {
    auto a = sgl::angle_between(sgl::vec2{1, 0}, sgl::vec2{-1, 0});
    EXPECT_NEAR(a, std::acos(-1.0f), 1e-5f); /* π */
}
