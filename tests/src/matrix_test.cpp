/**
 * @file matrix_test.cpp
 * @brief Unit tests for matrix operations (mat2, mat3, mat4).
 */
#include <cstddef>

#include <gtest/gtest.h>

import sgl;

/* ============================================================
 * mat2
 * ============================================================ */

TEST(Mat2, AddSubScaleNegate) {
    const sgl::mat2 a{.cols = {sgl::vec2{1, 2}, sgl::vec2{3, 4}}};
    const sgl::mat2 b{.cols = {sgl::vec2{5, 6}, sgl::vec2{7, 8}}};

    auto s = sgl::add(a, b);
    EXPECT_FLOAT_EQ(s.cols[0].x, 6);
    EXPECT_FLOAT_EQ(s.cols[1].y, 12);

    auto d = sgl::sub(b, a);
    EXPECT_FLOAT_EQ(d.cols[0].x, 4);
    EXPECT_FLOAT_EQ(d.cols[1].y, 4);

    auto sc = sgl::scale(a, 2.0f);
    EXPECT_FLOAT_EQ(sc.cols[0].x, 2);
    EXPECT_FLOAT_EQ(sc.cols[1].y, 8);

    auto n = sgl::negate(a);
    EXPECT_FLOAT_EQ(n.cols[0].x, -1);
    EXPECT_FLOAT_EQ(n.cols[1].y, -4);
}

TEST(Mat2, TransformVec2) {
    /* column-major: col0={1,3}, col1={2,4} => mat = | 1 2 |
     *                                               | 3 4 | */
    const sgl::mat2 m{.cols = {sgl::vec2{1, 3}, sgl::vec2{2, 4}}};
    const sgl::vec2 v{1, 1};
    auto r = sgl::transform_vec2(m, v);
    EXPECT_FLOAT_EQ(r.x, 3); /* 1*1 + 2*1 */
    EXPECT_FLOAT_EQ(r.y, 7); /* 3*1 + 4*1 */
}

TEST(Mat2, Mul) {
    /* I * A = A */
    const sgl::mat2 a{.cols = {sgl::vec2{1, 2}, sgl::vec2{3, 4}}};
    auto r = sgl::mat2_identity * a;
    EXPECT_FLOAT_EQ(r.cols[0].x, 1);
    EXPECT_FLOAT_EQ(r.cols[1].y, 4);
}

TEST(Mat2, Transpose) {
    const sgl::mat2 m{.cols = {sgl::vec2{1, 2}, sgl::vec2{3, 4}}};
    auto t = sgl::transpose(m);
    EXPECT_FLOAT_EQ(t.cols[0].x, 1);
    EXPECT_FLOAT_EQ(t.cols[0].y, 3);
    EXPECT_FLOAT_EQ(t.cols[1].x, 2);
    EXPECT_FLOAT_EQ(t.cols[1].y, 4);
}

TEST(Mat2, Determinant) {
    const sgl::mat2 m{.cols = {sgl::vec2{1, 2}, sgl::vec2{3, 4}}};
    EXPECT_FLOAT_EQ(sgl::determinant(m), -2.0f); /* 1*4 - 3*2 = -2 */
}

TEST(Mat2, Inverse) {
    const sgl::mat2 m{.cols = {sgl::vec2{4, 3}, sgl::vec2{3, 2}}};
    auto inv = sgl::inverse(m);
    auto i = m * inv;
    EXPECT_NEAR(i.cols[0].x, 1.0f, 1e-5f);
    EXPECT_NEAR(i.cols[0].y, 0.0f, 1e-5f);
    EXPECT_NEAR(i.cols[1].x, 0.0f, 1e-5f);
    EXPECT_NEAR(i.cols[1].y, 1.0f, 1e-5f);
}

TEST(Mat2, Trace) {
    const sgl::mat2 m{.cols = {sgl::vec2{3, 7}, sgl::vec2{1, 5}}};
    EXPECT_FLOAT_EQ(sgl::trace(m), 8.0f); /* 3 + 5 */
}

/* ============================================================
 * mat4
 * ============================================================ */

TEST(Mat4, Identity) {
    auto m = sgl::mat4_identity;
    EXPECT_FLOAT_EQ(m.cols[0].x, 1);
    EXPECT_FLOAT_EQ(m.cols[1].y, 1);
    EXPECT_FLOAT_EQ(m.cols[2].z, 1);
    EXPECT_FLOAT_EQ(m.cols[3].w, 1);
    EXPECT_FLOAT_EQ(m.cols[0].y, 0);
}

TEST(Mat4, Diagonal) {
    auto m = sgl::mat4_diagonal(3.0f);
    EXPECT_FLOAT_EQ(m.cols[0].x, 3);
    EXPECT_FLOAT_EQ(m.cols[1].y, 3);
    EXPECT_FLOAT_EQ(m.cols[0].y, 0);
}

TEST(Mat4, MulIdentity) {
    /* M * I = M */
    auto r = sgl::mat4_identity * sgl::mat4_identity;
    EXPECT_FLOAT_EQ(r.cols[0].x, 1);
    EXPECT_FLOAT_EQ(r.cols[3].w, 1);
    EXPECT_FLOAT_EQ(r.cols[0].y, 0);
}

TEST(Mat4, Transpose) {
    sgl::mat4 m{};
    m.cols[0] = {1, 0, 0, 0};
    m.cols[1] = {0, 2, 0, 0};
    m.cols[2] = {0, 0, 3, 0};
    m.cols[3] = {4, 5, 6, 1};
    auto t = sgl::transpose(m);
    /* The translation column becomes the last row */
    EXPECT_FLOAT_EQ(t.cols[3].x, 0);
    EXPECT_FLOAT_EQ(t.cols[0].w, 4);
}

TEST(Mat4, InverseIdentity) {
    auto inv = sgl::inverse(sgl::mat4_identity);
    for (std::size_t c{}; c < 4; ++c) {
        for (std::size_t r{}; r < 4; ++r) {
            EXPECT_NEAR(inv.cols[c][r], sgl::mat4_identity.cols[c][r], 1e-5f);
        }
    }
}

TEST(Mat4, InverseMulIsIdentity) {
    sgl::mat4 m = sgl::mat4_identity;
    m.cols[3] = {10, 20, 30, 1}; /* Translation matrix: col3 = {10, 20, 30, 1} */
    auto inv = sgl::inverse(m);
    auto result = m * inv;
    /* Should be close to identity */
    EXPECT_NEAR(result.cols[0].x, 1.0f, 1e-4f);
    EXPECT_NEAR(result.cols[1].y, 1.0f, 1e-4f);
    EXPECT_NEAR(result.cols[2].z, 1.0f, 1e-4f);
    EXPECT_NEAR(result.cols[3].w, 1.0f, 1e-4f);
    EXPECT_NEAR(result.cols[3].x, 0.0f, 1e-4f);
}

TEST(Mat4, TransformPoint) {
    sgl::mat4 m = sgl::mat4_identity;
    m.cols[3] = {5, 10, 15, 1};
    sgl::vec3 p{1, 2, 3};
    auto r = sgl::transform_point(m, p);
    EXPECT_NEAR(r.x, 6.0f, 1e-5f);
    EXPECT_NEAR(r.y, 12.0f, 1e-5f);
    EXPECT_NEAR(r.z, 18.0f, 1e-5f);
}

TEST(Mat4, TransformDir) {
    sgl::mat4 m = sgl::mat4_identity;
    m.cols[3] = {5, 10, 15, 1}; /* translation */
    sgl::vec3 d{1, 0, 0};
    auto r = sgl::transform_dir(m, d); /* translation ignored */
    EXPECT_NEAR(r.x, 1.0f, 1e-5f);
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
}

TEST(Mat4, InverseRigid) {
    /* Build a pure-translation rigid transform */
    sgl::mat4 m = sgl::mat4_identity;
    m.cols[3] = {3, 4, 5, 1};
    auto inv = sgl::inverse_rigid(m);
    auto r = m * inv;
    EXPECT_NEAR(r.cols[0].x, 1.0f, 1e-4f);
    EXPECT_NEAR(r.cols[3].x, 0.0f, 1e-4f);
}

TEST(Mat4, InverseAffineUniform) {
    /* Scale-2 matrix */
    sgl::mat4 m = sgl::mat4_diagonal(2.0f);
    m.cols[3].w = 1.0f; /* translation */
    auto inv = sgl::inverse_affine_uniform(m);
    auto r = m * inv;
    EXPECT_NEAR(r.cols[0].x, 1.0f, 1e-4f);
    EXPECT_NEAR(r.cols[1].y, 1.0f, 1e-4f);
    EXPECT_NEAR(r.cols[2].z, 1.0f, 1e-4f);
}

TEST(Mat4, ToMat3) {
    sgl::mat4 m = sgl::mat4_identity;
    m.cols[0].x = 2;
    auto m3 = sgl::to_mat3(m);
    EXPECT_FLOAT_EQ(m3.cols[0].x, 2);
    EXPECT_FLOAT_EQ(m3.cols[1].y, 1);
}

TEST(Mat4, ToMat4FromMat3) {
    sgl::mat3 m3 = sgl::mat3_identity;
    auto m4 = sgl::to_mat4(m3);
    EXPECT_FLOAT_EQ(m4.cols[3].w, 1);
    EXPECT_FLOAT_EQ(m4.cols[3].x, 0);
}

TEST(Mat4, ToMat4FromMat3WithTranslation) {
    sgl::mat3 m3 = sgl::mat3_identity;
    sgl::vec3 t{1, 2, 3};
    auto m4 = sgl::to_mat4(m3, t);
    EXPECT_FLOAT_EQ(m4.cols[3].x, 1);
    EXPECT_FLOAT_EQ(m4.cols[3].y, 2);
    EXPECT_FLOAT_EQ(m4.cols[3].z, 3);
    EXPECT_FLOAT_EQ(m4.cols[3].w, 1);
}

TEST(Mat4, GetSetTranslation) {
    sgl::mat4 m = sgl::mat4_identity;
    m.cols[3] = {7, 8, 9, 1};

    sgl::vec3 trans = sgl::get_translation(m);
    EXPECT_FLOAT_EQ(trans.x, 7);
    EXPECT_FLOAT_EQ(trans.y, 8);
    EXPECT_FLOAT_EQ(trans.z, 9);

    auto m2 = sgl::set_translation(m, sgl::vec3{1, 2, 3});
    EXPECT_FLOAT_EQ(m2.cols[3].x, 1);
    EXPECT_FLOAT_EQ(m2.cols[3].y, 2);
    EXPECT_FLOAT_EQ(m2.cols[3].z, 3);
    EXPECT_FLOAT_EQ(m2.cols[3].w, 1); /* w preserved */
}

TEST(Mat4, TraceAndDeterminant) {
    EXPECT_FLOAT_EQ(sgl::trace(sgl::mat4_identity), 4.0f);
    EXPECT_NEAR(sgl::determinant(sgl::mat4_identity), 1.0f, 1e-5f);

    auto m2 = sgl::mat4_diagonal(2.0f);
    m2.cols[3].w = 1.0f;
    EXPECT_NEAR(sgl::determinant(m2), 8.0f, 1e-4f); /* 2*2*2*1 */
}

TEST(Mat4, ScaleMatrix) {
    /* Re-set w to 1 for the transform column, scale would have made it 2 */
    auto s = sgl::scale(sgl::mat4_identity, 2.0f);
    EXPECT_FLOAT_EQ(s.cols[0].x, 2);
}
