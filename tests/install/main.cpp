/**
 * @file main.cpp
 * @brief Imports an installed sgl and touches both the AoS and the SoA surface.
 */
#include <cstdio>
#include <vector>

import sgl;

int main() {
    const sgl::vec3 c{sgl::cross(sgl::vec3{1, 0, 0}, sgl::vec3{0, 1, 0})};

    sgl::soa::point_buffer3f pts;
    pts.push_back(sgl::vec3{1, 1, 1});
    pts.push_back(sgl::vec3{5, 5, 5});
    const auto inside{sgl::soa::count(pts, sgl::soa::inside(sgl::box3d{{0, 0, 0}, {2, 2, 2}}))};

    const bool ok{c.z == 1.0f && inside == 1};
    std::puts(ok ? "sgl install ok" : "sgl install broken");
    return ok ? 0 : 1;
}
