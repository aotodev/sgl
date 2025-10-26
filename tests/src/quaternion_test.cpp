/**
 * @file quaternion_test.cpp
 * @brief Unit tests for quaternion operations.
 */
#include <cmath>
#include <gtest/gtest.h>

import sgl;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

/* ============================================================
 * Basic quaternion properties
 * ============================================================ */

TEST(Quat, IdentityIsUnit) {
    EXPECT_NEAR(sgl::norm_sq(sgl::quat_identity), 1.0f, 1e-6f);
    EXPECT_NEAR(sgl::norm(sgl::quat_identity), 1.0f, 1e-6f);
}

TEST(Quat, DotSelf) {
    sgl::quat q{0, 0, 0, 1};
    EXPECT_FLOAT_EQ(sgl::dot(q, q), 1.0f);
}

TEST(Quat, Conjugate) {
    sgl::quat q{1, 2, 3, 4};
    auto c = sgl::conjugate(q);
    EXPECT_FLOAT_EQ(c.x, -1);
    EXPECT_FLOAT_EQ(c.y, -2);
    EXPECT_FLOAT_EQ(c.z, -3);
    EXPECT_FLOAT_EQ(c.w, 4);
}

TEST(Quat, Negate) {
    sgl::quat q{1, 2, 3, 4};
    auto n = sgl::negate(q);
    EXPECT_FLOAT_EQ(n.x, -1);
    EXPECT_FLOAT_EQ(n.w, -4);
}

TEST(Quat, NormalizedIsUnit) {
    sgl::quat q{1, 1, 1, 1};
    auto n = sgl::normalized(q);
    EXPECT_NEAR(sgl::norm_sq(n), 1.0f, 1e-5f);
}

TEST(Quat, InverseUnit) {
    sgl::quat q{0, 0, 0, 1};
    auto inv = sgl::inverse_unit(q);
    EXPECT_FLOAT_EQ(inv.w, 1.0f);
    EXPECT_FLOAT_EQ(inv.x, 0.0f);
}

TEST(Quat, InverseGeneral) {
    sgl::quat q{1, 0, 0, 0}; /* pure imaginary i, not unit length... but norm=1 anyway */
    auto inv = sgl::inverse(q);
    EXPECT_NEAR(inv.x, -1.0f, 1e-5f);
    EXPECT_NEAR(inv.w, 0.0f, 1e-5f);
}

/* ============================================================
 * Multiplication
 * ============================================================ */

TEST(Quat, IdentityMul) {
    sgl::quat q{0.5f, 0.5f, 0.5f, 0.5f};
    auto r = sgl::quat_identity * q;
    EXPECT_NEAR(r.x, q.x, 1e-5f);
    EXPECT_NEAR(r.y, q.y, 1e-5f);
    EXPECT_NEAR(r.z, q.z, 1e-5f);
    EXPECT_NEAR(r.w, q.w, 1e-5f);
}

TEST(Quat, MulInverseIsIdentity) {
    sgl::quat q = sgl::normalized(sgl::quat{1, 2, 3, 4});
    auto r = q * sgl::inverse_unit(q);
    EXPECT_NEAR(r.x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
    EXPECT_NEAR(r.w, 1.0f, 1e-5f);
}

/* ============================================================
 * Rotation
 * ============================================================ */

TEST(Quat, RotateIdentity) {
    sgl::vec3 v{1, 2, 3};
    auto r = sgl::rotate(sgl::quat_identity, v);
    EXPECT_NEAR(r.x, 1.0f, 1e-5f);
    EXPECT_NEAR(r.y, 2.0f, 1e-5f);
    EXPECT_NEAR(r.z, 3.0f, 1e-5f);
}

TEST(Quat, RotateAxisAngle) {
    /* 90° around Z: should map (1,0,0) → (0,1,0) */
    sgl::quat q = sgl::quat_from_axis_angle(sgl::vec3{0, 0, 1}, std::acos(-1.0f) * 0.5f); /* 90° around Z */
    auto r = sgl::rotate(q, sgl::vec3{1, 0, 0});
    EXPECT_NEAR(r.x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.y, 1.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
}

/* ============================================================
 * Conversions
 * ============================================================ */

TEST(Quat, ToMat3Identity) {
    auto m = sgl::to_mat3(sgl::quat_identity);
    EXPECT_NEAR(m.cols[0].x, 1.0f, 1e-5f);
    EXPECT_NEAR(m.cols[1].y, 1.0f, 1e-5f);
    EXPECT_NEAR(m.cols[2].z, 1.0f, 1e-5f);
}

TEST(Quat, ToMat4Identity) {
    auto m = sgl::to_mat4(sgl::quat_identity);
    EXPECT_NEAR(m.cols[0].x, 1.0f, 1e-5f);
    EXPECT_NEAR(m.cols[3].w, 1.0f, 1e-5f);
}

TEST(Quat, ToMat4WithTranslation) {
    auto m = sgl::to_mat4(sgl::quat_identity, sgl::vec3{1, 2, 3});
    EXPECT_NEAR(m.cols[3].x, 1.0f, 1e-5f);
    EXPECT_NEAR(m.cols[3].y, 2.0f, 1e-5f);
    EXPECT_NEAR(m.cols[3].z, 3.0f, 1e-5f);
}

/* ============================================================
 * Comparison utilities
 * ============================================================ */

TEST(Quat, NearlyEqual) {
    sgl::quat a{0, 0, 0, 1};
    sgl::quat b{0, 0, 0, 1};
    sgl::quat c{0, 0, 0, -1}; /* q and -q represent the same rotation */
    EXPECT_TRUE(sgl::nearly_equal(a, b));
    EXPECT_TRUE(sgl::nearly_equal(a, c)); /* same rotation, opposite sign */
}

TEST(Quat, AngleBetween) {
    sgl::quat q1 = sgl::quat_from_axis_angle(sgl::vec3{0, 0, 1}, 0.0f);
    sgl::quat q2 = sgl::quat_from_axis_angle(sgl::vec3{0, 0, 1}, std::acos(-1.0f) * 0.5f); /* 90° */
    auto angle = sgl::angle_between(q1, q2);
    EXPECT_NEAR(angle, std::acos(-1.0f) * 0.5f, 1e-4f);
}

TEST(Quat, IsAxisAligned) {
    EXPECT_TRUE(sgl::is_axis_aligned(sgl::vec3{1, 0, 0}));
    EXPECT_TRUE(sgl::is_axis_aligned(sgl::vec3{0, 0, -1}));
    EXPECT_FALSE(sgl::is_axis_aligned(sgl::vec3{1, 1, 0}));
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
