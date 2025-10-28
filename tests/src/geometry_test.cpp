/**
 * @file geometry_test.cpp
 * @brief Unit tests for geometric operations (segments, planes, intersections, grids, transforms).
 */
#include <gtest/gtest.h>

import sgl;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

/* ============================================================
 * Segment helpers
 * ============================================================ */

TEST(Geometry, ClosestPointOnSegment_Vec2) {
    sgl::vec2 p{0, 0}, a{2, 0}, b{4, 0};
    auto r = sgl::closest_point_on_segment(p, a, b);
    EXPECT_NEAR(r.x, 2.0f, 1e-5f); /* clamped to start */
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
}

TEST(Geometry, ClosestPointOnSegment_Vec3) {
    sgl::vec3 p{0, 5, 0}, a{0, 0, 0}, b{10, 0, 0};
    auto r = sgl::closest_point_on_segment(p, a, b);
    EXPECT_NEAR(r.x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
}

TEST(Geometry, DistancePointToSegment) {
    sgl::vec2 p{0, 3}, a{0, 0}, b{4, 0};
    EXPECT_NEAR(sgl::distance_point_to_segment(p, a, b), 3.0f, 1e-5f);
}

TEST(Geometry, DistancePointToSegmentSq) {
    sgl::vec2 p{0, 3}, a{0, 0}, b{4, 0};
    EXPECT_NEAR(sgl::distance_point_to_segment_sq(p, a, b), 9.0f, 1e-4f);
}

/* ============================================================
 * Plane operations
 * ============================================================ */

TEST(Geometry, PlaneFromPointNormal) {
    sgl::plane pl = sgl::plane_from_point_normal(sgl::vec3{0, 0, 5}, sgl::vec3{0, 0, 1});
    EXPECT_NEAR(pl.d, 5.0f, 1e-5f); /* Normal should be (0,0,1) */
}

TEST(Geometry, PlaneFromPoints) {
    sgl::plane pl = sgl::plane_from_points(sgl::vec3{0, 0, 0}, sgl::vec3{1, 0, 0}, sgl::vec3{0, 1, 0});
    EXPECT_NEAR(std::abs(pl.normal.z), 1.0f, 1e-5f);
    EXPECT_NEAR(pl.d, 0.0f, 1e-5f);
}

TEST(Geometry, SignedDistanceToPlane) {
    sgl::plane pl = sgl::plane_from_point_normal(sgl::vec3{0, 0, 0}, sgl::vec3{0, 0, 1});
    EXPECT_NEAR(sgl::signed_distance_to_plane(sgl::vec3{0, 0, 3}, pl), 3.0f, 1e-5f);
    EXPECT_NEAR(sgl::signed_distance_to_plane(sgl::vec3{0, 0, -2}, pl), -2.0f, 1e-5f);
}

TEST(Geometry, DistanceToPlane) {
    sgl::plane pl = sgl::plane_from_point_normal(sgl::vec3{0, 0, 0}, sgl::vec3{0, 0, 1});
    EXPECT_NEAR(sgl::distance_to_plane(sgl::vec3{0, 0, -3}, pl), 3.0f, 1e-5f);
}

TEST(Geometry, ProjectOntoPlane) {
    sgl::plane pl = sgl::plane_from_point_normal(sgl::vec3{0, 0, 0}, sgl::vec3{0, 0, 1});
    auto r = sgl::project_onto_plane(sgl::vec3{3, 4, 5}, pl);
    EXPECT_NEAR(r.x, 3.0f, 1e-5f);
    EXPECT_NEAR(r.y, 4.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
}

/* ============================================================
 * Intersection tests
 * ============================================================ */

TEST(Geometry, IntersectSegments2d_Hit) {
    auto r = sgl::intersect_segments_2d(sgl::vec2{0, 0}, sgl::vec2{2, 2}, sgl::vec2{0, 2}, sgl::vec2{2, 0});
    EXPECT_TRUE(r.intersects);
    EXPECT_NEAR(r.point.x, 1.0f, 1e-5f);
    EXPECT_NEAR(r.point.y, 1.0f, 1e-5f);
}

TEST(Geometry, IntersectSegments2d_Miss) {
    auto r = sgl::intersect_segments_2d(sgl::vec2{0, 0}, sgl::vec2{1, 0}, sgl::vec2{0, 1}, sgl::vec2{1, 1});
    EXPECT_FALSE(r.intersects);
}

TEST(Geometry, IntersectRayBox2d) {
    sgl::box2d box{.min = {0, 0}, .max = {4, 4}};
    auto r = sgl::intersect_ray_box(sgl::vec2{-1, 2}, sgl::vec2{1, 0}, box);
    EXPECT_TRUE(r.hit);
    EXPECT_NEAR(r.t_entry, 1.0f, 1e-4f);
}

TEST(Geometry, IntersectRayBox3d) {
    sgl::box3d box{.min = {0, 0, 0}, .max = {4, 4, 4}};
    auto r = sgl::intersect_ray_box(sgl::vec3{-1, 2, 2}, sgl::vec3{1, 0, 0}, box);
    EXPECT_TRUE(r.hit);
}

TEST(Geometry, SegmentIntersectsBox2d) {
    sgl::box2d box{.min = {0, 0}, .max = {4, 4}};
    EXPECT_TRUE(sgl::segment_intersects_box(sgl::vec2{-1, 2}, sgl::vec2{5, 2}, box));
    EXPECT_FALSE(sgl::segment_intersects_box(sgl::vec2{-2, 6}, sgl::vec2{-1, 6}, box));
}

TEST(Geometry, SegmentIntersectsBox3d) {
    sgl::box3d box{.min = {0, 0, 0}, .max = {4, 4, 4}};
    EXPECT_TRUE(sgl::segment_intersects_box(sgl::vec3{-1, 2, 2}, sgl::vec3{5, 2, 2}, box));
}

TEST(Geometry, PointInPolygon) {
    sgl::vec2 poly[4] = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    EXPECT_TRUE(sgl::point_in_polygon(sgl::vec2{2, 2}, poly, 4));
    EXPECT_FALSE(sgl::point_in_polygon(sgl::vec2{5, 5}, poly, 4));
}

TEST(Geometry, WindingNumber) {
    sgl::vec2 poly[4] = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    EXPECT_NE(sgl::winding_number(sgl::vec2{2, 2}, poly, 4), 0);
    EXPECT_EQ(sgl::winding_number(sgl::vec2{6, 6}, poly, 4), 0);
}

TEST(Geometry, IntersectSegmentPolygon) {
    sgl::vec2 poly[4] = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    auto r = sgl::intersect_segment_polygon(sgl::vec2{-1, 2}, sgl::vec2{5, 2}, poly, 4);
    EXPECT_TRUE(r.hit);
}

/* ============================================================
 * Grid operations
 * ============================================================ */

TEST(Geometry, GridCell2d) {
    auto cell = sgl::grid_cell(sgl::vec2{2.5f, 3.5f}, sgl::vec2{0, 0}, 1.0f);
    EXPECT_EQ(cell.x, 2);
    EXPECT_EQ(cell.y, 3);
}

TEST(Geometry, GridCell3d) {
    auto cell = sgl::grid_cell(sgl::vec3{5.5f, 6.5f, 7.5f}, sgl::vec3{0, 0, 0}, 1.0f);
    EXPECT_EQ(cell.x, 5);
    EXPECT_EQ(cell.y, 6);
    EXPECT_EQ(cell.z, 7);
}

TEST(Geometry, GridCellRange2d) {
    sgl::box2d box{.min = {1, 1}, .max = {3, 3}};
    auto r = sgl::grid_cell_range(box, sgl::vec2{0, 0}, 1.0f);
    EXPECT_EQ(r.min.x, 1);
    EXPECT_EQ(r.max.x, 3);
}

/* ============================================================
 * Transform operations
 * ============================================================ */

TEST(Geometry, TransformToMat4) {
    sgl::transform tf{};
    auto m = sgl::to_mat4(tf);
    EXPECT_NEAR(m.cols[0].x, 1.0f, 1e-5f);
    EXPECT_NEAR(m.cols[3].w, 1.0f, 1e-5f);
}

TEST(Geometry, TransformPoint) {
    sgl::transform tf{};
    tf.position = {1, 2, 3};
    auto r = sgl::transform_point(tf, sgl::vec3{0, 0, 0});
    EXPECT_NEAR(r.x, 1.0f, 1e-5f);
    EXPECT_NEAR(r.y, 2.0f, 1e-5f);
}

TEST(Geometry, TransformDirection) {
    sgl::transform tf{};
    tf.position = {100, 200, 300}; /* translation should NOT affect direction */
    auto r = sgl::transform_direction(tf, sgl::vec3{1, 0, 0});
    EXPECT_NEAR(r.x, 1.0f, 1e-5f);
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
}

TEST(Geometry, InverseUniformTransform) {
    sgl::transform tf{};
    tf.position = {1, 2, 3};
    tf.scale = {2, 2, 2};
    auto inv = sgl::inverse_uniform(tf);
    EXPECT_NEAR(inv.scale.x, 0.5f, 1e-5f);
}

TEST(Geometry, InverseTransform) {
    sgl::transform tf{};
    tf.position = {1, 2, 3};
    tf.scale = {2, 4, 8};
    auto inv = sgl::inverse(tf);
    EXPECT_NEAR(inv.scale.x, 0.5f, 1e-5f);
    EXPECT_NEAR(inv.scale.y, 0.25f, 1e-5f);
}

TEST(Geometry, ComposeTransform) {
    sgl::transform a{};
    a.position = {1, 0, 0};
    a.scale = {1, 1, 1};
    sgl::transform b{};
    b.position = {2, 0, 0};
    b.scale = {1, 1, 1};

    /* a applied after b: position should be a.pos + rotate(a.rot, a.scale * b.pos) */
    auto c = sgl::compose(a, b);
    EXPECT_NEAR(c.position.x, 3.0f, 1e-4f);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
