# Batch queries over SoA data

`sgl::soa` runs one query against many primitives stored as structure-of-arrays: one
contiguous array per coordinate. The kernels are written once against an internal SIMD
wrapper and run 8 lanes wide on AVX2, 4 on NEON, and 1 on the scalar fallback.

```cpp
import sgl;

std::vector<float> x, y, z;                       // one array per axis
const auto pts{sgl::soa::make_points(x, y, z)};  // sgl::soa::points3f, a view

const sgl::box3d zone{{0, 0, 0}, {10, 10, 10}};
const sgl::box3d hole{{4, 4, 4}, {6, 6, 6}};

std::size_t n{sgl::soa::count(pts, sgl::soa::inside(zone) && !sgl::soa::inside(hole))};
sgl::soa::for_each_match(pts, sgl::soa::inside(zone), [](std::size_t i) { /* ... */ });
const sgl::box3d b{sgl::soa::bounds(pts)};
```

## Ranges

| Type | Holds |
| --- | --- |
| `points<T, D>` (`points2f`, `points3f`) | `D` coordinate pointers and a size |
| `boxes<T, D>` (`boxes2f`, `boxes3f`) | `D` min pointers, `D` max pointers and a size |

`make_points(ranges...)` views one contiguous range per axis; `make_boxes(min, max)` pairs
two point views. Views do not own memory. Pointers need no alignment, and any size works:
the remainder after the last full SIMD batch runs on the scalar backend with the same
formulas, so it gives the same answers.

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
  face plane hits. The direction must be finite.
- **Backends agree.** Min/max and comparisons have one definition across AVX2, NEON and
  scalar (see `src/sgl/batch.cppm`), so results are bit-identical on every target. The same
  tests run on all of them.

## Performance

Ryzen 9 7940HS (Zen 4), one core, `-O3 -mavx2 -mfma`, float, 4096 elements (in cache).
Items per second, higher is better. "Scalar" is the same query as a plain loop over the
same SoA arrays, left to the auto-vectoriser; "AoS" is the one-at-a-time `sgl` API over
`vec3` / `box3d` arrays.

| Query | SoA (Clang / GCC) | Scalar, best of branchy and branchless | AoS |
| --- | --- | --- | --- |
| points in box, count | 8.2 G / 9.7 G | 4.0 G / 6.7 G | 2.0 G / 1.3 G |
| points in box, bit mask | 8.1 G / 9.7 G | 0.8 G / 0.5 G | |
| in box and not in hole | 4.5 G / 4.9 G | 2.3 G / 4.0 G | |
| boxes overlapping box | 6.1 G / 6.2 G | 4.4 G / 6.2 G | 2.1 G / 1.5 G |
| ray vs boxes, count | 3.3 G / 5.7 G | | |
| ray vs boxes, entry distances | 2.4 G / 5.1 G | 1.9 G / 3.3 G | 0.7 G / 0.8 G |
| bounds | 11.2 G / 10.5 G | 0.8 G / 0.8 G | 0.2 G / 2.3 G |

Past the caches every query is bandwidth bound and the variants converge (about 2.9 G
points per second at 4 Mi points). Reproduce with `-DSGL_BENCHMARKS=ON` and
`./build/bench/soa_bench`.
