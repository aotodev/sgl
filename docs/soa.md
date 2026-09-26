# Batch queries over SoA data

`sgl::soa` runs one query against many primitives stored as structure-of-arrays: one
contiguous array per coordinate. The kernels are written once against an internal SIMD
wrapper and run 8 lanes wide on AVX2, 4 on NEON, and 1 on the scalar fallback.

```cpp
import sgl;

std::pmr::monotonic_buffer_resource arena;
sgl::soa::point_buffer3f pts{&arena};              // owning, padded, 64-byte aligned
pts.reserve(input.size());
for (const sgl::vec3& p : input) { pts.push_back(p); }

const sgl::box3d zone{{0, 0, 0}, {10, 10, 10}};
const sgl::box3d hole{{4, 4, 4}, {6, 6, 6}};

std::size_t kept{sgl::soa::count(pts, sgl::soa::inside(zone) && !sgl::soa::inside(hole))};
sgl::soa::for_each_match(pts, sgl::soa::inside(zone), [](std::size_t i) { /* ... */ });
const sgl::box3d b{sgl::soa::bounds(pts)};

/* Data you already own works too, through a non-owning view. */
std::vector<float> x, y, z;
std::size_t hits{sgl::soa::count(sgl::soa::make_points(x, y, z), sgl::soa::inside(zone))};
```

## Ranges

Every driver takes a view or a buffer.

| Type | Holds |
| --- | --- |
| `points<T, D, Pad = 1>` (`points2f`, `points3f`) | `D` coordinate pointers and a size |
| `boxes<T, D, Pad = 1>` (`boxes2f`, `boxes3f`) | `D` min pointers, `D` max pointers and a size |
| `point_buffer<T, D>` (`point_buffer2f`, `point_buffer3f`) | owning points on a `std::pmr::memory_resource` |
| `box_buffer<T, D>` (`box_buffer2f`, `box_buffer3f`) | owning boxes on a `std::pmr::memory_resource` |

**Views.** `make_points(ranges...)` views one contiguous range per axis; `make_boxes(min,
max)` pairs two point views. Views do not own memory, pointers need no alignment, and any
size works. With `Pad = 1` the remainder after the last full SIMD batch runs on the scalar
backend with the same formulas, so it gives the same answers.

**Padding.** `Pad > 1` promises that elements up to `size` rounded up to a multiple of
`Pad` are readable. Their contents do not matter: kernels run whole SIMD batches over the
tail and discard the extra lanes, so there is no scalar tail at all. That matters most for
short ranges such as leaf buckets or grid cells. A padded view converts to an unpadded one
(`points3f v = buf.view();`; copy-initialization, since braces would try aggregate init).

**Buffers.** A buffer keeps all its arrays in one 64-byte aligned allocation from its
memory resource, with the capacity a multiple of `block_size` (64). Drivers use its padded
view. `reserve` up front to allocate once; `axis(a)` (or `min_axis` / `max_axis`) gives a
span to fill in bulk; `push_back` takes a `vec` / `box` or coordinate arrays; new elements
from `resize` are zero. Copies follow pmr: copy construction uses the default resource
unless one is passed, and assignment keeps the target's resource.

## Predicates

| Predicate | Range | Matches |
| --- | --- | --- |
| `inside(box)` | points | point in the closed box |
| `overlaps(box)` | boxes | closed boxes intersect (touching counts) |
| `contains(point)` | boxes | point in the closed box |
| `hit_by(origin, direction, t_min = 0, t_max = inf)` | boxes | ray reaches the closed box for some t in [t_min, t_max] |

Predicates over the same kind of range compose with `&&`, `||` and `!`. The composition is
built at compile time and evaluated in a single pass with no intermediate buffers.
Composing predicates over different ranges does not compile.

## Drivers

| Function | Result |
| --- | --- |
| `mask(range, pred, out)` | one bit per element: bit k of `out[w]` is element `64 w + k`; needs `(size + 63) / 64` words, bits past the end are zero |
| `count(range, pred)` | number of matches |
| `any(range, pred)` | whether anything matches; stops after the first block of 64 that does |
| `for_each_match(range, pred, fn)` | calls `fn(index)` for every match, in increasing order |
| `hit_distances(boxes, hit_by(...), t)` | per box, the entry distance `max(t_min, entry)` on a hit and `+inf` on a miss; returns the hit count |
| `bounds(points)` | the bounding box; NaN coordinates are skipped, an empty range gives min = +inf, max = -inf |

## Semantics

- **Boxes are closed.** A point on a face is inside; touching boxes overlap.
- **NaN.** A NaN coordinate in a point makes it match nothing. For `overlaps` and `contains`
  a NaN box matches nothing; for `hit_by` its result is unspecified.
- **Invalid boxes** (min > max on some axis): `hit_by` and `contains` never match them.
  `overlaps` needs valid boxes, except the empty box (+inf, -inf), which overlaps nothing.
- **Rays.** `direction` need not be normalized; t is measured in units of it, so the segment
  [a, b] is `hit_by(a, b - a, 0, 1)`. Zero components are fine: a ray lying exactly in a
  face plane hits. The direction must be finite. `t_min` and `t_max` are clamped to the
  finite range, so a line (`-inf, inf`) that only touches a box along parallel axes reports
  an entry of `-FLT_MAX`.
- **Backends agree.** Min/max and comparisons have one definition across AVX2, NEON and
  scalar (see `src/sgl/batch.cppm`), so results are bit-identical on every target. The same
  tests run on all of them.

## Performance

Ryzen 9 7940HS (Zen 4), one core, `-O3 -mavx2 -mfma`, float. Items per second, higher is
better. "Scalar" is the same query as a plain loop over the same SoA arrays, left to the
auto-vectoriser (the better of a branchy and a branchless version); "AoS" is the
one-at-a-time `sgl` API over `vec3` / `box3d` arrays.

4096 elements (in cache):

| Query | SoA (Clang / GCC) | Scalar | AoS |
| --- | --- | --- | --- |
| points in box, count | 7.8 G / 9.2 G | 3.9 G / 6.6 G | 1.9 G / 1.3 G |
| points in box, bit mask | 7.5 G / 9.2 G | 0.8 G / 0.5 G | |
| in box and not in hole | 4.3 G / 4.3 G | 2.3 G / 4.0 G | |
| boxes overlapping box | 5.8 G / 5.8 G | 4.2 G / 5.9 G | 2.0 G / 1.5 G |
| ray vs boxes, count | 4.7 G / 5.4 G | | |
| ray vs boxes, entry distances | 4.0 G / 4.9 G | 1.8 G / 3.2 G | 0.7 G / 0.7 G |
| bounds | 10.6 G / 9.8 G | 0.8 G / 0.8 G | 0.2 G / 2.2 G |

Short ranges, points in box, count (the padded buffer has no scalar tail):

| Elements | View (Clang / GCC) | Buffer (Clang / GCC) | Scalar (Clang / GCC) |
| --- | --- | --- | --- |
| 13 | 1.1 G / 2.0 G | 2.7 G / 5.1 G | 2.4 G / 2.8 G |
| 100 | 4.4 G / 7.5 G | 5.0 G / 9.3 G | 3.5 G / 6.0 G |
| 1000 | 7.8 G / 10.0 G | 7.8 G / 10.0 G | 3.9 G / 6.6 G |

Past the caches every query is bandwidth bound and the variants converge (about 2.5 to 2.9 G
points per second at 4 Mi points). Reproduce with `-DSGL_BENCHMARKS=ON` and
`./build/bench/soa_bench`.
