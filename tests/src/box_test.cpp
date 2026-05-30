/**
 * @file box_test.cpp
 * @brief Unit tests for AABB (box2d / box3d) operations.
 */
#include <gtest/gtest.h>

import sgl;

/* ============================================================
 * box2d
 * ============================================================ */

TEST(Box2d, IsValid) {
    EXPECT_TRUE(sgl::is_valid(sgl::box2d{.min = {0, 0}, .max = {1, 1}}));
    EXPECT_FALSE(sgl::is_valid(sgl::box2d{.min = {1, 0}, .max = {0, 1}}));
}

TEST(Box2d, Center) {
    auto c = sgl::center(sgl::box2d{.min = {0, 0}, .max = {2, 4}});
    EXPECT_FLOAT_EQ(c.x, 1.0f);
    EXPECT_FLOAT_EQ(c.y, 2.0f);
}

TEST(Box2d, HalfExtents) {
    auto h = sgl::half_extents(sgl::box2d{.min = {0, 0}, .max = {4, 6}});
    EXPECT_FLOAT_EQ(h.x, 2.0f);
    EXPECT_FLOAT_EQ(h.y, 3.0f);
}

TEST(Box2d, Extents) {
    auto e = sgl::extents(sgl::box2d{.min = {1, 2}, .max = {4, 7}});
    EXPECT_FLOAT_EQ(e.x, 3.0f);
    EXPECT_FLOAT_EQ(e.y, 5.0f);
}

TEST(Box2d, Translate) {
    auto b = sgl::translate(sgl::box2d{.min = {0, 0}, .max = {1, 1}}, sgl::vec2{2, 3});
    EXPECT_FLOAT_EQ(b.min.x, 2.0f);
    EXPECT_FLOAT_EQ(b.max.y, 4.0f);
}

TEST(Box2d, ScaleUniform) {
    auto b = sgl::scale_uniform(sgl::box2d{.min = {1, 2}, .max = {3, 4}}, 2.0f);
    EXPECT_FLOAT_EQ(b.min.x, 2.0f);
    EXPECT_FLOAT_EQ(b.max.y, 8.0f);
}

TEST(Box2d, Dilate) {
    auto b = sgl::dilate(sgl::box2d{.min = {1, 1}, .max = {3, 3}}, 1.0f);
    EXPECT_FLOAT_EQ(b.min.x, 0.0f);
    EXPECT_FLOAT_EQ(b.max.x, 4.0f);
}

TEST(Box2d, Overlaps) {
    sgl::box2d a{.min = {0, 0}, .max = {2, 2}};
    sgl::box2d b{.min = {1, 1}, .max = {3, 3}};
    sgl::box2d c{.min = {3, 3}, .max = {5, 5}};
    EXPECT_TRUE(sgl::overlaps(a, b));
    EXPECT_FALSE(sgl::overlaps(a, c));
}

TEST(Box2d, Contains_Box2d) {
    sgl::box2d outer{.min = {0, 0}, .max = {10, 10}};
    sgl::box2d inner{.min = {2, 2}, .max = {8, 8}};
    sgl::box2d outside{.min = {9, 9}, .max = {15, 15}};
    EXPECT_TRUE(sgl::contains(outer, inner));
    EXPECT_FALSE(sgl::contains(outer, outside));
}

TEST(Box2d, ContainsPoint) {
    sgl::box2d b{.min = {0, 0}, .max = {10, 10}};
    EXPECT_TRUE(sgl::contains_point(b, sgl::vec2{5, 5}));
    EXPECT_FALSE(sgl::contains_point(b, sgl::vec2{11, 5}));
}

TEST(Box2d, Intersection) {
    sgl::box2d a{.min = {0, 0}, .max = {4, 4}};
    sgl::box2d b{.min = {2, 2}, .max = {6, 6}};
    auto i = sgl::intersection(a, b);
    EXPECT_FLOAT_EQ(i.min.x, 2.0f);
    EXPECT_FLOAT_EQ(i.max.x, 4.0f);
}

TEST(Box2d, Merge) {
    sgl::box2d a{.min = {0, 0}, .max = {2, 2}};
    sgl::box2d b{.min = {1, 1}, .max = {4, 5}};
    auto m = sgl::merge(a, b);
    EXPECT_FLOAT_EQ(m.min.x, 0.0f);
    EXPECT_FLOAT_EQ(m.max.y, 5.0f);
}

TEST(Box2d, Expand) {
    sgl::box2d b{.min = {1, 1}, .max = {3, 3}};
    auto e = sgl::expand(b, sgl::vec2{5, 0});
    EXPECT_FLOAT_EQ(e.max.x, 5.0f);
    EXPECT_FLOAT_EQ(e.min.y, 0.0f);
}

TEST(Box2d, ClosestPoint) {
    sgl::box2d b{.min = {0, 0}, .max = {5, 5}};
    auto p = sgl::closest_point(b, sgl::vec2{-1, 3});
    EXPECT_FLOAT_EQ(p.x, 0.0f);
    EXPECT_FLOAT_EQ(p.y, 3.0f);
}

TEST(Box2d, Distance) {
    sgl::box2d b{.min = {0, 0}, .max = {5, 5}};
    EXPECT_FLOAT_EQ(sgl::distance(b, sgl::vec2{7, 0}), 2.0f); /* point along x-axis at distance 2 from box face */
}

TEST(Box2d, FromCenterHalf) {
    auto b = sgl::box2d_from_center_half(sgl::vec2{5, 5}, sgl::vec2{2, 3});
    EXPECT_FLOAT_EQ(b.min.x, 3.0f);
    EXPECT_FLOAT_EQ(b.max.y, 8.0f);
}

TEST(Box2d, FromPoint) {
    auto b = sgl::box2d_from_point(sgl::vec2{3, 7});
    EXPECT_FLOAT_EQ(b.min.x, 3.0f);
    EXPECT_FLOAT_EQ(b.max.x, 3.0f);
}

TEST(Box2d, Area) {
    EXPECT_FLOAT_EQ(sgl::area(sgl::box2d{.min = {0, 0}, .max = {3, 4}}), 12.0f);
}

TEST(Box2d, Perimeter) {
    EXPECT_FLOAT_EQ(sgl::perimeter(sgl::box2d{.min = {0, 0}, .max = {3, 4}}), 14.0f);
}

/* ============================================================
 * box3d
 * ============================================================ */

TEST(Box3d, IsValid) {
    EXPECT_TRUE(sgl::is_valid(sgl::box3d{.min = {0, 0, 0}, .max = {1, 1, 1}}));
    EXPECT_FALSE(sgl::is_valid(sgl::box3d{.min = {1, 0, 0}, .max = {0, 1, 1}}));
}

TEST(Box3d, ContainsPoint_Box3d) {
    EXPECT_TRUE(sgl::contains(sgl::box3d{.min = {0, 0, 0}, .max = {10, 10, 10}}, sgl::vec3{5, 5, 5}));
}

TEST(Box3d, Volume) {
    EXPECT_FLOAT_EQ(sgl::volume(sgl::box3d{.min = {0, 0, 0}, .max = {2, 3, 4}}), 24.0f);
}

TEST(Box3d, SurfaceArea) {
    /* 2*(2*3 + 2*4 + 3*4) = 2*(6+8+12) = 52 */
    EXPECT_FLOAT_EQ(sgl::surface_area(sgl::box3d{.min = {0, 0, 0}, .max = {2, 3, 4}}), 52.0f);
}

TEST(Box3d, FromCenterHalf) {
    auto b = sgl::box3d_from_center_half(sgl::vec3{0, 0, 0}, sgl::vec3{1, 2, 3});
    EXPECT_FLOAT_EQ(b.min.x, -1.0f);
    EXPECT_FLOAT_EQ(b.max.z, 3.0f);
}

TEST(Box3d, FromPoint) {
    auto b = sgl::box3d_from_point(sgl::vec3{1, 2, 3});
    EXPECT_FLOAT_EQ(b.min.y, 2.0f);
    EXPECT_FLOAT_EQ(b.max.y, 2.0f);
}
