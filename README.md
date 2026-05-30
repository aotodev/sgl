# sgl

A C++20 **module-based, SIMD-accelerated computational geometry library**.

`sgl` is a SIMD-first library for fixed-size linear algebra and geometry —
vectors, matrices, quaternions, axis-aligned boxes, planes, transforms, and the
intersection and distance queries built on them. Every operation is written
directly against SSE/AVX2 or NEON intrinsics, and C++20 templates collapse the
per-type and per-dimension boilerplate, so the API surface stays broad while the
implementation stays compact. It covers the everyday graphics and geometry math,
and is at its strongest in the spatial queries that run in tight loops —
containment, closest-point, and intersection over many primitives.

It ships as a single C++20 named module (`sgl`). There are no headers to include
and no macros to manage: consumers just `import sgl;`.

```cpp
import sgl;

int main() {
    const sgl::vec3 a{1, 2, 3};
    const sgl::vec3 b{4, 5, 6};

    const auto c = sgl::cross(a, b);
    const auto  dir = sgl::normalized(a);

    const auto d{sgl::dot(a, b)};

    sgl::quat q = sgl::quat_from_axis_angle(sgl::vec3{0, 0, 1}, 1.57079633f);
    sgl::vec3 r = sgl::rotate(q, sgl::vec3{1, 0, 0});
}
```

## SIMD backends

The intrinsic-level implementation is selected automatically at configure time
based on the target architecture, behind one uniform API:

| Architecture        | Backend                          |
| ------------------- | -------------------------------- |
| x86-64              | SSE / AVX2 + FMA             |
| AArch64 (Apple Silicon, ARMv8.2-A+) | NEON (D-form + Q-form) |

Both backends expose the exact same `sgl::` names, so code written against one
compiles unchanged against the other.

## What it supports

- **Vectors** — `vec2`, `vec3`, `vec4` (and integer `ivec2`/`ivec3`): full
  arithmetic operators, dot/cross, length, several normalization variants
  (precise, fast-reciprocal, Newton-Raphson, and zero-safe), projection,
  reflection, clamp/lerp/min/max/abs, distance, angle, and parallel/perpendicular
  tests.
- **Matrices** — `mat2` and `mat4` (column-major): add/sub/scale/negate, multiply,
  transpose, determinant, trace, a general inverse plus dedicated closed-form
  rigid and uniform-affine inverses; `mat3` is carried as a rotation/normal block
  with `mat3`<->`mat4` conversions.
- **Quaternions** — `quat`: Hamilton product, conjugate/inverse, normalization,
  vector rotation, axis-angle construction, matrix conversions, and angular
  distance.
- **Axis-aligned boxes** — `box2d`, `box3d`: validity, center/extents,
  translate/scale/dilate, overlap/containment, intersection/merge/expand,
  closest-point and distance queries, area/volume/surface-area.
- **Planes & transforms** — `plane`, `plane_basis`, `transform`: plane
  construction and signed-distance/projection queries, 3D<->2D plane-basis
  projection (with batch variants), and full TRS transform composition,
  inversion, and point/direction application.
- **Intersection & spatial queries** — 2D segment intersection, ray– and
  segment–AABB tests, closest distance between two 3D segments (for
  capsule/clearance checks), point-in-polygon (winding number), segment–polygon,
  and integer grid-cell mapping.

## Design notes

- **SIMD-first, one API across backends.** Every operation is built on intrinsics
  rather than scalar fallbacks, and the AVX2 and NEON backends export identical
  names — the SIMD layer is an implementation detail, not part of the surface.
  Templates carry the weight that boilerplate usually does, which is why the
  surface above fits in a compact codebase.
- **Closed-form transform inverses.** Rigid (`R | t`) and uniform-scale affine
  transforms are inverted by exploiting their orthogonality — transpose the
  rotation, remap the translation — rather than running the general
  cofactor/determinant inverse. For those transform classes that is the same
  result for a fraction of the arithmetic; the general `inverse` remains available
  when the matrix is arbitrary.

## A note on the API: pass-by-`const&`

Public functions take their vector/matrix arguments by `const T&` rather than by
value. This is deliberate. The hot internals operate on register types
(`__m128` / `float32x4_t`) passed *by value* — that's where register passing
matters — while the user-facing aggregates (`vec*`, `mat*`, `quat`, ...) are
passed by reference.

The reasoning: under both the System V (x86-64) and AAPCS (AArch64) ABIs, a
16-byte float aggregate passed by value does not arrive in a single vector
register — it is split across two (SysV) or up to four (AAPCS) registers and must
be recombined before any SIMD op, and aggregates larger than 16 bytes (`mat3`,
`mat4`, `box3d`, `transform`, ...) are passed indirectly with a forced copy anyway.
So by-value buys nothing for these types and can cost a little. More importantly,
because the operations are `inline` and the templates instantiate in the caller,
inlining makes the calling convention disappear at the vast majority of call
sites — so the signature choice is a wash on the hot path and `const&` wins
cleanly on the cold one.

## Requirements

- **CMake** ≥ 3.30 (for `CXX_MODULES` install/export support)
- **GCC** ≥ 14, **Clang** ≥ 17, or **MSVC** ≥ 19.40 (VS 2022 17.10) — each is the
  floor for C++20 named modules plus CWG 2518 (`static_assert(false)` in
  uninstantiated template branches)
- A C++20 standard library and toolchain with named-module support

## Building

```sh
cmake -B build -G Ninja
cmake --build build
```

> Use the **Ninja** or **Visual Studio** generator — CMake's C++20 module support
> does not work with the Makefiles generators.

### Toolchains

`sgl` builds with GCC, Clang (libstdc++ or libc++), and MSVC. All three are
verified in C++20 mode; the ISA flags (`-mavx2 -mfma` / `/arch:AVX2`) are applied
automatically per compiler frontend.

```sh
# GCC (libstdc++)
CXX=g++ cmake -B build -G Ninja

# Clang + libc++
CXX=clang++ cmake -B build -G Ninja -DCMAKE_CXX_FLAGS="-stdlib=libc++"

# MSVC (x64 Native Tools prompt, or a cross wrapper) — Ninja generator
cmake -B build -G Ninja
```

Then `cmake --build build` as usual.

### Consuming

The most robust way to consume a C++20 module library is to build it as part of
your tree (`add_subdirectory` or `FetchContent`) and link the `sgl::sgl` target:

```cmake
target_link_libraries(your_target PRIVATE sgl::sgl)
```

Installing and using `find_package` is also supported (see below).

> **Ordering note (GCC):** put textual `#include`s *before* any `import sgl;` in
> a translation unit. Importing first and `#include`-ing standard headers
> afterwards can trip a libstdc++ redefinition error.

### Installing

```sh
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build
cmake --install build
```

This installs the static archive, the arch-selected module source (consumers
recompile the BMI against their own toolchain, BMIs are not portable), and a
package config. Then, from another project:

```cmake
find_package(sgl REQUIRED)
target_link_libraries(your_target PRIVATE sgl::sgl)
```

The exported target carries the x86-64 ISA flags as a usage requirement
(`-mavx2 -mfma`, or `/arch:AVX2` under MSVC), so your build compiles the imported
module interface with the matching instruction set.

### Options

| Option         | Default        | Description                                  |
| -------------- | -------------- | -------------------------------------------- |
| `SGL_TESTS`    | `OFF`          | Build the GoogleTest unit-test suite         |
| `SGL_PIC`      | `OFF`          | Build with position-independent code         |
| `SGL_INSTALL`  | top-level only | Generate install + `find_package()` rules    |

### Running the tests

```sh
cmake -B build -G Ninja -DSGL_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Each case is registered with CTest (`gtest_discover_tests`), so you can add
`-j<N>` to run them in parallel, `-R <regex>` to select a subset, and
`--rerun-failed` to re-run only the previous failures. The binary can also be run
directly for the fastest single-shot run: `./build/tests/unit_tests`.
