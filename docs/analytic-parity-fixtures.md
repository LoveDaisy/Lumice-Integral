# Analytic parity fixtures (LI → Lumice)

[中文](analytic-parity-fixtures_zh.md)

Lumice Integral (LI) exports, at a fixed rev, a directory of JSON fixtures that
Lumice copies into its repository and replays in CI against
`liblumice_analytic` module A v0 (Lumice `doc/analytic-api.md` §4:
`EvaluatePath`, seed search, `TraceFiber[Batch]`, point lists only). The
direction is one way (owner ruling 2026-09-28): LI is the reference, Lumice
follows. This page is written so that a reader can be implemented on the Lumice
side without reading LI source. The normative semantics behind the fixtures
are `docs/phase1-math-contract.md` §9 (continuation) and §9.5 (discovery), and
the conventions are those of `docs/conventions.md`.

## 1. Producing the fixtures

```bash
uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
```

- One command writes every fixture of the matrix (§6) and a `manifest.json`.
  The full matrix takes about 16 s on an M2 Max.
- **Deterministic.** The same rev re-exports byte for byte. LI's test
  `tests/test_parity_export.py` exports the matrix in two separate
  interpreters and compares every file. The fixtures do not record time,
  host or paths.
- `--verify` reads every fixture back, recomputes it with the checkout, and
  compares within the fixture's own tolerances (§5). It exits non-zero on any
  failure. `--cells 3-5__random ...` exports a subset.
- The output lives under `artifacts/`, which is not versioned in LI. The pinned
  copy is Lumice's, and each fixture records the LI rev it came from.
- Code: `lumice_integral.parity_export` (formats, selection, verifiers) and
  `scripts/export_analytic_parity.py` (the matrix, a thin CLI).

## 2. Files and common fields

A cell is one row of the matrix, a path and a point category. Its files are
named `<path>__<category>__<kind>[__<label>].json`, where the path is written
with dashes (`3-5-6-7`), the category is `random` / `critical` /
`near_boundary`, and the kind is `evaluate_path` / `trace_fiber` /
`seed_search`. `manifest.json` lists every cell with its `files`, its
`skipped` fixtures (with a reason), its `rationale` and the `selection` record
of §6.

The JSON is UTF-8 with sorted keys. Floats are written in the shortest form
that round-trips to the same float64, so parsing with any conforming JSON
reader (`strtod`) gives the exported bits back. `-0.0` is preserved, and
there are no NaN or infinities. Every fixture has these fields:

| Field | Meaning |
|---|---|
| `format` | `"lumice-integral/analytic-parity"` |
| `schema_version` | `1`. Wave 2 (diagnostics and weights) adds optional fields and does not rename or remove existing ones. A breaking change raises it. |
| `fixture_kind` | `evaluate_path`, `trace_fiber` or `seed_search` |
| `symmetry_semantics` | Always `"none"`. The input is one concrete face sequence and no symmetry reduction is involved (Lumice `doc/analytic-api.md` §3.3 rule 2; `docs/conventions.md` #21). |
| `provenance.li_rev` | Full LI commit SHA of the export |
| `provenance.li_tracked_tree_clean` | `false` if tracked files differed from that commit at export time. Such a fixture should not be copied into Lumice. |
| `provenance.conventions_sha256` | SHA-256 of LI `docs/conventions.md`. A change flags that a convention may have moved. |
| `cell` | `name`, `path`, `category`, `rationale`, `selection`, and for `evaluate_path` a `pose_label` |
| `input` | The call's arguments (below) |
| `expected` | The reference output (below) |
| `tolerance` | One entry per compared quantity: `{"value": number, "basis": text}` |

### 2.1 Shared input fields

| Field | Meaning |
|---|---|
| `crystal` | The `LUMICE_ANALYTIC_Crystal` scalars, field for field: `kind` (`"prism"` / `"pyramid"`), `height` (prism height or the pyramid's prism band `prism_h`, in units of the reference circumscribed diameter), `face_distance[6]`, `upper_h`, `lower_h`, `upper_wedge_deg`, `lower_wedge_deg`. Fields not used by the kind are `0.0`. Semantics are Lumice `doc/configuration.md` §prism / §pyramid. LI builds `HexPrism.from_lumice(height, face_distance)` / `Pyramid.from_lumice(height, upper_h, lower_h, face_distance=..., upper_wedge_deg=..., lower_wedge_deg=...)` with hexagon edge `a = 1`. Every output below is scale-free. |
| `faces` | The concrete face sequence in Lumice face numbers (entry, internal reflections, exit) |
| `refractive_index` | The ice index of the call |
| `incident_direction` | World propagation direction of the sunlight, sun → crystal (`s = -ŝ`) |
| `pose` / `seed_pose` | Nine numbers, a row-major 3×3 rotation, body → world (`v_W = R v_B`) |
| `target_direction` | World propagation direction of the outgoing light, crystal → observer (`d`) |
| `continuation` | LI's `ContinuationOptions` (contract §9.2, reference defaults of §10.1), every field. Lumice's v0 options block has fewer fields. A backend uses its own equivalents and documents the mapping. |

## 3. The three fixture kinds

### 3.1 `evaluate_path`

Input: `crystal`, `faces`, `refractive_index`, `incident_direction`, `pose`.

| `expected` field | Meaning |
|---|---|
| `valid` | The path is realisable at this pose: the entry and exit Snell discriminants and every incidence cosine (entry, each internal face reached from inside, exit) are positive (contract §5.2). An internal reflection's TIR discriminant does not gate validity (conventions #18). |
| `outgoing_direction` | World outgoing propagation direction (only when `valid`) |
| `segment_directions` | `(face_count + 1) × 3`, body-frame propagation directions `Rᵀ v`: the incident ray, the ray after each face, the outgoing ray (only when `valid`) |
| `interface_transmittances` | `face_count` values: the unpolarised (s/p averaged) transmittance `T` at entry and exit, the reflectance `R` at each internal face, `1` under TIR (only when `valid`) |
| `fresnel_transmission` | `T_entry · Π R_k · T_exit`, `0.0` when not `valid` |
| `diagnostics` | Not compared: every margin LI evaluated (`validity_margins`), the smallest validity margin and its name, and for an invalid pose the failure message |

For an invalid pose only `valid` and `fresnel_transmission` are compared.

### 3.2 `trace_fiber`

Input: the shared fields plus `target_direction`, `seed_pose` and
`continuation`. The seed lies on the fiber: `target_direction` is computed
from the seed pose, so its residual is at rounding level.

`expected.traces` holds one or two traces. The first is traced with
`initial_tangent_sign = +1`. When it does not close, a second trace from the
same seed with the sign reversed follows (contract §9.5.5). Each trace has
`initial_tangent_sign`, `status`, `reason`, `poses` `(N, 9)`,
`crystal_frame_sun_directions` `u = Rᵀ(−s)` `(N, 3)`, `arclength_increments`
`(N − 1)`, `residual_norms` `(N)`, `tangents` `(N, 3)` (contract §9.3) and
`arclength`. A trace may hold one pose (no accepted step).

**The curve** of a fixture is the closed first trace if it closed. Otherwise
it is the second trace's poses after the seed in reverse order followed by the
first trace's poses, an open polyline through the seed (contract §9.5.5,
`resample.stitch_open_arc`). The sign of the seed tangent is deterministic for
one LAPACK build but not across builds (contract §10.1), so a backend's `+1`
may be LI's `−1`. The recipe (§4) is therefore orientation-free.

### 3.3 `seed_search`

Input: the shared fields plus `target_direction`, `extra_seeds` (empty in
v0), `band_half_width_deg`, `cluster_radius_rad`, `distance_threshold`,
`continuation`, and `sample`:

| `sample` field | Meaning |
|---|---|
| `sampler`, `n` | How LI generated the sample: the antipodal Fibonacci lattice of contract §9.5.2 with `n` points, kept where `w = A T > 0` |
| `band_u`, `band_phi`, `band_deviation` | The band itself in pool order: `u_i` (sun in the crystal frame), `phi_i` (body-frame outgoing propagation direction), `D_i` (radians), increasing `D_i` |

A backend replays contract §9.5.3–§9.5.5 on this band. It builds the candidate
poses `R_i = W F_iᵀ` (§9.5.3, with `ŝ = −incident_direction` and `d =
target_direction`), then clusters, corrects, gates, deduplicates, traces and
classifies. The sampler is compared separately against §9.5.2, not through
these fixtures (§9.5.8).

| `expected` field | Meaning |
|---|---|
| `completeness` | `complete` / `unknown` (procedural, §9.5.6) |
| `pool_count`, `extra_seed_count`, `raw_cluster_count`, `admissible_count` | The funnel counts |
| `events` | The six counters, every one present |
| `components` | In trace order: `kind` (`closed` / `arc`), `seed` (9), `status`, `reason`, `start_reason` (`null` for closed), `arclength`, `curve_poses` `(N, 9)` (the closed trace or the stitched arc), and `seed_index` for an arc |
| `incomplete` | `cause`, `seed`, `status`, `reason` |

## 4. Comparison recipes

- **Vectors and scalars**: the largest absolute componentwise difference is at
  most the tolerance.
- **Rotation distance**: `angle(Aᵀ B) = atan2(|vee(skew)|, (tr(Aᵀ B) − 1)/2)`,
  with `skew = (M − Mᵀ)/2` and `M = Aᵀ B`.
- **Curve distance** between two pose polylines `P` and `Q`, each open or
  closed: densify each polyline along its geodesic chords
  (`P_i exp(t log(P_iᵀ P_{i+1}))`) at a spacing of at most `1e-3` rad, adding
  the chord from the last pose back to the first when the curve is closed.
  Then take the symmetric Hausdorff distance, the larger of the two values
  "the largest distance from a pose of one curve to the densified other".
  Densification removes the chord-sampling part of the distance, so two
  samplings of one curve differ by at most half the spacing. Reference
  implementation: `parity_export.curve_distance`.
- **`trace_fiber`**: the number of traces is equal, the multiset of
  `(status, reason)` pairs is equal, the curve distance is within
  `curve_distance_rad`, the summed `arclength` agrees to
  `arclength_relative`, and every `residual_norms` entry is at most
  `residual_norm_bound`. Step counts and individual poses are not compared.
- **`seed_search`**: `completeness`, the four counts and the six counters are
  equal. The components are compared in order: `kind` and `status` are equal,
  and the non-null ends `{reason, start_reason}` are equal as an unordered
  set. The backend's seed lies within `seed_to_curve_rad` of the fixture
  curve (the pose-to-sample distance of contract §9.5.4 step 5). The curves
  are within `curve_distance_rad`. For closed components the `arclength`
  agrees to `closed_arclength_relative`. The `incomplete` causes are equal in
  order.

## 5. Tolerances and their basis

| Quantity | Tolerance | Basis |
|---|---|---|
| `valid` | exact | Every fixture pose has its nearest gate's margin recorded in `diagnostics`. On the current matrix that margin is at least `1e-8` in magnitude, far above the float64 rounding of a margin (`~1e-15`). |
| Directions, transmittances, Fresnel factor | `1e-12 · max(1, 1/(2√d))`, with `d` the pose's smallest Snell discriminant | The same closed-form chain from bit-identical inputs contributes a few ulp per interface, and `1e-12` leaves about `1e3` ulp. The refracted cosine is `√d`, whose derivative `1/(2√d)` amplifies rounding next to a Snell boundary. The arc-end poses of the matrix have `d ≈ 1.2e-8` (amplification about 4600). |
| Curve distance | `0.012` rad | Contract §10.1: the bidirectional sampled-pose set distance between two step-controller settings on the 3-5 loop stays below `0.012` rad (`0.0093` observed). |
| Arclength (a trace; a closed component) | `2e-3` relative | Contract §10.1: one 3-5 loop measures `0.96432` to `0.96472` across initial steps `0.03` to `0.08`, a spread of `1.8e-3` relative. An arc's length is compared only from the same seed (§9.5.6), which holds for `trace_fiber` but not for `seed_search`. |
| Residual norms | `≤ residual_tolerance + relative_residual_tolerance` (`1e-11`) | Contract §9.5.4 step 3 and §10.1. This is a bound on the backend, not an equality with LI. |
| Seed-search counts, counters, kinds, reasons | exact | Contract §9.5.8: the pool is the exported band, and the step order and gates are normative. |
| Seed of a discovered component | within `distance_threshold` (`0.08` rad) of the fixture curve | The contract's own test for "this pose is on that component" (§9.5.4 step 5). |

These are the tolerances under which LI's own read-back passes (`--verify`).
The first cross-backend run is the first evidence about them on the Lumice
side. A tolerance that proves too tight or too loose is changed in LI, with
the new evidence recorded here, and then the fixtures are re-exported. It is
not widened silently in the Lumice reader.

## 6. The matrix

Path topology × point category (Lumice `doc/raypath-analysis.md` §5.1.6, item
3). The sun is at altitude 15°, azimuth 0 (the ch06 scene), and `n = 1.31`.

| Path | Crystal | Topology |
|---|---|---|
| `3-5` | canonical column: prism, `height = 1`, regular | no internal reflection (22° halo) |
| `3-5-6-7` | same | two internal reflections, partial or total and weighted by Fresnel `R`; arcs ending on exit TIR or path infeasibility |
| `13-15-26-28` | asymmetric pyramid: `prism_h = 0.5`, `upper_h = 0.25`, `lower_h = 0.6`, wedges of Miller `(1,0,1)` / `(2,0,3)` (`miller_wedge_deg`), `face_distance = (1, 1.1, 0.9, 1, 1.2, 0.95)` (LI `tests/test_optics_crystal_native.py::ASYMMETRIC_PYRAMID`) | pyramid faces off the prism family |

Points are chosen on the sphere of `u`, the sun in the crystal frame
(`parity_export.choose_point`). From `u` the pose is the one that puts `u` on
`ŝ` with the outgoing direction in the vertical half-plane above `s`
(`s2_store.event_rotations` with azimuth reference `+z`). The target is that
pose's outgoing direction, so every seed is on its fiber.

- **random**: the first draw of `numpy.random.default_rng(20260928)` normals
  that passes discovery's gates (path valid, entry measure positive).
- **critical**: the first interior extremum `D*` of `D_P`
  (`DPField.interior_critical_points`). The point is on its lit side at
  `D* ± 0.5°`, reached along the geodesic from the extremum and bisected, so
  the fiber is a short loop around a Jacobian-degenerate point.
- **near_boundary**: from the random point, march along a fixed geodesic to
  the edge of `U_P`, then bisect to a pose whose smallest validity margin is
  `1e-3`. The mirror image across the edge is exported as an invalid
  `evaluate_path` (`__outside`).

Per cell LI exports `evaluate_path__point` (and `__outside`), `trace_fiber`
from that pose, `evaluate_path__curve_min_margin` (the traced pose with the
smallest validity margin, typically an arc end within the event tolerance),
and `seed_search` at that target (`N = 1e5`, band `0.2°`, cluster radius
`0.3` rad, `distance_threshold = 0.08`).

Skipped, recorded in `manifest.json`:

- `3-5-6-7__critical`: `D_P` of `3-5-6-7` has no interior extremum on this
  crystal. `interior_critical_points` is empty, and its critical values
  (`DPField.critical_values`: 50.06°, 141.84°, 163.47°) are all boundary ones.

The pyramid cells carry `seed_search` since task `s2-store-pyramid-seeds`
(2026-09-29; until then the reference store refused pyramids). The
`13-15-26-28__near_boundary` target lies outside the store's lit range
(`D = 87.33°`; the kept events span 121.51°–149.23°): its point sits next to
the entry incidence gate of `U_P`, which depends on the face normals only,
where the finite crystal's entry measure is already zero (`corridor_empty`),
so the band is empty and the expected result is complete with no component.
It is kept as a negative case: a seed search must not invent a component
there.

The sample store describes the pyramid by its `PyramidShape`
(`s2_store.crystal_description`: `prism_h`, `upper_h`, `lower_h`,
`upper_c_over_a`, `lower_c_over_a`, `face_distance`, `a`), not by the wedge
angles of `crystal` above: `c_over_a` is what `Pyramid.from_lumice` reduces
the Miller indices or the wedge angle to, and it rebuilds the crystal bit for
bit. The two describe the same crystal (`c_over_a = (√3/2) cot(wedge)`); the
store's form is a cache key, not an exchange format.

The first export (2026-09-28) contained 32 fixtures; since 2026-09-29 there
are 35:

| Cell | `trace_fiber` | `seed_search` |
|---|---|---|
| `3-5__random` (`D = 39.32°`) | closed, 5.670 | 1 closed; pool 172, 13 clusters, 12 folded |
| `3-5__critical` (`D = 22.34°`, min 21.84°) | closed, 1.160 | 1 closed; pool 579 |
| `3-5__near_boundary` (`D = 43.23°`, entry incidence cosine `1e-3`) | arc path_infeasible / path_infeasible, 0.067 + 0.916 | 4 arcs, one of them a one-pose arc (TIR at both ends, §9.5.6) |
| `3-5-6-7__random` (`D = 143.87°`) | arc TIR / TIR, 0.922 + 1.545 | 1 arc |
| `3-5-6-7__near_boundary` (`D = 140.72°`, internal incidence cosine `1e-3`) | one side stops at the seed (path_infeasible), the other ends on TIR after 1.189 | 2 arcs |
| `13-15-26-28__random` (`D = 142.54°`) | closed, 2.735 | 1 closed; pool 29, 5 clusters, 4 folded |
| `13-15-26-28__critical` (`D = 148.74°`, max 149.24°) | closed, 0.808 | 1 closed; pool 71, 2 clusters, 1 folded |
| `13-15-26-28__near_boundary` (`D = 87.33°`) | one side stops at the seed, the other ends on TIR after 0.428 | empty band: pool 0, no component (outside the lit range, above) |

## 7. Update flow

1. LI changes behaviour (a solver, a convention, a default) and commits.
2. LI re-exports at the new rev with `--verify`, which must pass, from a clean
   tree (`li_tracked_tree_clean = true`).
3. Lumice copies the whole directory into its repository, replacing the
   previous one.
4. Lumice parity CI goes red wherever the C++ now differs.
5. Lumice fixes the C++ until it is green. It does not edit the fixtures. A
   disagreement about a tolerance or a semantic goes back to step 1.

A change to this format, meaning fields, file names or recipes, is a change of
this page and of `parity_export.SCHEMA_VERSION` in the same LI commit.

## 8. Not in these fixtures

- Diagnostics and weights (`jacobian_diagnostics`, `terminal_payload`,
  `weight_observables`, ...). These belong to wave 2
  (`explore-fiber-diagnostics-contract`) and will be added as optional fields.
- The sampler itself, `check_band_coverage`, and densification behaviour
  (§9.5.7).
- Any symmetry reduction. `symmetry_semantics` is `none` throughout.
