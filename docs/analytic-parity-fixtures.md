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

- One command writes every fixture of the matrix (§6), the edge cells
  (§6.1) and a `manifest.json`, 82 fixtures in about 1 min on an M2 Max
  (the matrix alone took about 16 s before wave 2).
- **Deterministic.** The same rev re-exports byte for byte. LI's test
  `tests/test_parity_export.py` exports the matrix in two separate
  interpreters and compares every file. The fixtures do not record time,
  host or paths.
- `--verify` reads every fixture back, recomputes it with the checkout, and
  compares within the fixture's own tolerances (§5). It exits non-zero on any
  failure. `--cells 3-5__random 3-5__limits ...` exports a subset (matrix
  cells and edge cells by name).
- The output lives under `artifacts/`, which is not versioned in LI. The pinned
  copy is Lumice's, and each fixture records the LI rev it came from.
- Code: `lumice_integral.parity_export` (formats, selection, verifiers) and
  `scripts/export_analytic_parity.py` (the matrix, a thin CLI).

## 2. Files and common fields

A cell is one row of the matrix, a path and a point category. Its files are
named `<path>__<category>__<kind>[__<label>].json`, where the path is written
with dashes (`3-5-6-7`), the category is `random` / `critical` /
`near_boundary`, and the kind is `evaluate_path` / `trace_fiber` /
`seed_search`. An edge cell (§6.1) is named `<path>__<label>` instead, and
its files `<path>__<label>__<kind>[__<suffix>].json`. `manifest.json` lists
every matrix cell under `cells` and every edge cell under `edge_cells`, each
with its `files`, its `skipped` fixtures (with a reason), its `rationale` and
the `selection` record of §6. An edge cell also has `label`, `point` (how its
seed was chosen) and `serves` (the contract §11 rows it certifies,
`docs/phase1-math-contract.md` §11.1). A matrix-only export has no
`edge_cells` key.

The JSON is UTF-8 with sorted keys. Floats are written in the shortest form
that round-trips to the same float64, so parsing with any conforming JSON
reader (`strtod`) gives the exported bits back. `-0.0` is preserved, and
there are no NaN or infinities. Every fixture has these fields:

| Field | Meaning |
|---|---|
| `format` | `"lumice-integral/analytic-parity"` |
| `schema_version` | `1`. Wave 2 added optional fields (§3.1, §3.2) and renamed or removed none, so every v0 key keeps its bytes (checked against the 2026-09-29 export at `5ea2bde`). A fixture without the wave 2 fields comes from an older export, and its recipes are the v0 ones. A breaking change raises the version. |
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
| `branch_margins` | Wave 2. When `valid`: the signed validity margins by name, `entry_incidence_cosine`, `entry_snell_discriminant`, `internal_<k>_incidence_cosine` for each internal reflection `k = 1, 2, ...`, `exit_incidence_cosine`, `exit_snell_discriminant` (contract §9.3 `branch_diagnostics`). Cosines of the ray with the face normal and Snell discriminants `1 − n_rel²(1 − cos²)`, dimensionless, positive on the smooth branch. `null` when not `valid`. |
| `failed_gate` | Wave 2. When not `valid`: `{name, value}` of the first gate in the order above whose margin is not positive (its value is `≤ 0`). `null` when `valid`. |
| `jacobian_available` | Wave 2. `true` exactly when `valid`: the normal Jacobian is defined only on the smooth branch. |
| `normal_jacobian` | Wave 2. `J_perp = σ₁ σ₂` of the `2 × 3` residual Jacobian `A` at this pose, with the target taken as the pose's own outgoing direction (contract §5.4). `A` differentiates the outgoing direction under right-trivialised rotations `R exp([δ]×)` (δ in radians, the §5.1 metric) and projects on an orthonormal basis of the tangent plane at that direction. Dimensionless (radian of direction per radian of pose), basis-invariant, and it depends on the pose and path only. `null` when not available. It can be tiny (`~4e-17` at an extremum of `D_P`) and large next to a Snell boundary (`~2600` at `d ≈ 2e-8`). |
| `singular_values` | Wave 2. `[σ₁, σ₂]`, `σ₁ ≥ σ₂ ≥ 0`, of the same `A`. `null` when not available. |

For an invalid pose only `valid`, `fresnel_transmission`, `failed_gate` and
the availability (`jacobian_available = false`, `null` margins and Jacobian)
are compared.

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
`arclength`. A trace may hold one pose (no accepted step), or none when the
seed itself is rejected (`rank_loss`, `chart_boundary`: the edge cells of
§6.1).

Wave 2 adds per-pose arrays, aligned with `poses` (contract §9.3
`jacobian_diagnostics` and `branch_diagnostics`): `branch_margin_names` (the
`k` names of `branch_margins` in §3.1, in that order), `branch_margins`
`(N, k)`, `jacobian_available` `(N)`, `normal_jacobian` `(N)` and
`singular_values` `(N, 2)`. Their meaning is the §3.1 field's at that pose.

A fixture whose traces end `budget_exhausted` also carries
`expected.reference_curve`: `poses` `(M, 9)` and `closed`, the curve (below)
of the same seed traced with the default options, and a `budget_extent`
tolerance.

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
- **`evaluate_path`, wave 2 fields**: `jacobian_available` is equal and the
  `branch_margins` names are equal. Margins agree to the `branch_margins`
  tolerance (absolute). `normal_jacobian` and each singular value agree to
  their tolerance relative to `max(1, |value|)`. For an invalid pose,
  `failed_gate.name` is equal and its value agrees like a margin.
- **`trace_fiber`**: the number of traces is equal, the multiset of
  `(status, reason)` pairs is equal, the curve distance is within
  `curve_distance_rad`, the summed `arclength` agrees to
  `arclength_relative`, and every `residual_norms` entry is at most
  `residual_norm_bound`. Step counts and individual poses are not compared.
  Two traces that both accept no pose have distance 0. One with and one
  without poses have infinite distance.
- **`trace_fiber`, per-pose arrays** (`pointwise_consistency`,
  `accepted_pose_regularity`): two backends step differently, so the arrays
  are never compared with LI's sample by sample. Instead, at each of its own
  poses `i` the backend's `normal_jacobian[i]`, `singular_values[i]` and
  `branch_margins[i]` must equal what its own `EvaluatePath` returns at
  `poses[i]`, within the §3.1 tolerances at that pose. `EvaluatePath` itself
  is certified against LI at fixed poses. Together they certify the arrays,
  including their alignment with `poses`. Every accepted pose must also be
  regular: `jacobian_available`, every branch margin `> 0`, and
  `normal_jacobian > 0`.
- **`trace_fiber`, budgets** (`budget_extent` present): the `(status,
  reason)` multiset and the residual bound as above, and then, instead of
  curve and length equality, which two controllers cannot share inside one
  budget, the following. A `step_budget` trace has exactly
  `maximum_accepted_steps + 1` poses (the seed plus one pose per accepted
  step). An `arclength_budget` trace has `maximum_arclength −
  maximum_advance < arclength ≤ maximum_arclength` (the next edge would have
  crossed the budget). An `evaluation_budget` trace has at least one pose
  (what one evaluation unit counts is backend-internal, contract §9.2).
  Every pose of every trace lies within `curve_distance_rad` of the densified
  `reference_curve`. A backend that counts steps differently documents the
  mapping and compares the step budget as an interval. The fixture is not
  changed.
- **Option variants** (`trace_fiber__<variant>` of an edge cell): ordinary
  `trace_fiber` fixtures whose `continuation` differs from the defaults. For a
  `perturbation` variant the exporter checks that LI's traced curve still
  matches its default trace under this recipe and refuses to write it
  otherwise. A backend passing every variant thus reproduces the same
  component under each setting (contract C06).
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
| Branch margins, `failed_gate` value | `1e-12 · max(1, 1/(2√d))`, the directions' tolerance | Margins are cosines and discriminants of the same closed-form chain. |
| `normal_jacobian`, singular values | relative to `max(1, |value|)`: `1e-12 · max(1, 1/(4d))` | `J_perp` is one derivative above the directions, so the directions' Snell amplification `1/(2√d)` enters squared. Measured: LI's AD value agrees with an independent central difference (`h = 1e-6`) to `6e-9` relative, and a `1e-14` rad pose nudge moves it by at most `2.5e-12` relative away from Snell boundaries (`4e-8` at `d ~ 1e-8`, where the tolerance is `2.5e-5`). |
| Per-pose arrays of a trace | the two rows above, evaluated at each pose | The comparison is with the backend's own `EvaluatePath` (§4). |
| Accepted-pose regularity, availability, gate names | exact | Contract §6.1 and §5.4: an accepted pose is on the smooth branch and regular. |
| Budget extent | exact counts and bounds (§4) | Contract §9.4: `budget_exhausted` retains partial geometry. The counting rules are stated in contract §11.1. |

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

### 6.1 Edge cells (wave 2)

The matrix samples ordinary points. The edge cells, written after it, are the
cases where contract §11 invariants are decided: very short loops, loops
running along a boundary, TIR-cut arcs, rank loss, budgets, and a cone
crystal. Each serves the §11 rows listed in its manifest entry, and contract
§11.1 is the row-by-row account. The numbers below are those of the export at
this rev.

A seed comes from one of five `point` methods. **`critical_offset`** takes the
lit side of an interior extremum of `D_P` at the given offset, as the matrix's
`critical` category does. **`extremum`** takes the extremum itself.
**`random`** takes the matrix's random draw. **`antipodal_target`** takes that
pose aimed at `−d`. **`target`** uses a given target, and its seeds are the
components that seed search returns on the `N = 1e5` sample, one
`trace_fiber__component_<k>` each. The pixel targets are those of
`camera.linear_pixel_outgoing_direction` with `canonical_scene.CANONICAL_RENDER`.
The deviation targets lie in the vertical plane through `s`, above it. Unless
listed, a cell exports `trace_fiber` (default options),
`evaluate_path__curve_min_margin` and `__curve_min_jacobian` (the traced poses
with the smallest branch margin and the smallest `J_perp`), and for non-target
points `evaluate_path__point`.

| Cell | Serves | Why this case | Observed (LI, this rev) |
|---|---|---|---|
| `3-5__short_loop` | C05 C06 C14 | 0.01° above the 3-5 minimum deviation: the loop is about four initial steps long. The step-aware closure must close it on its first traversal. A loop shorter than the `2 × initial_step` closure extent is traversed twice by design (contract §6.4: at 0.001° the loop is 0.0523 with `initial_step 0.01` and 0.1046 with the default 0.04). `J_perp` is smallest here. Variants `initial_step_0.03`, `initial_step_0.08` and `controller_thresholds` (`minimum_step 2e-5`, `maximum_step 0.10`, `shrink 0.4`, `growth 1.15`, `maximum_retries 10`). | closed, 0.1652 (55 poses; 0.1653 / 0.1652 / 0.1653 under the variants). `J_perp` min 0.00405. |
| `3-5__strip_short_loop_r100_c126` | C06 C15 | ch06 pixel (100, 126): a loop shorter than π that the retired absolute closure gate traversed twice. | closed, 1.6452; seed search pool 553, 3 clusters, 1 closed. |
| `3-5__caustic_loop_r49_c0` | C06 C15 | ch06 pixel (49, 0): a 0.19 loop at the caustic edge whose extra seeds used to end on a spurious event. | closed, 0.1898; pool 320, 1 cluster. |
| `3-5__boundary_hugging_r700_c150`, `__r780_c150` | C06 C08 C16 | Loops running along the exit TIR boundary (smallest margin 0.0142 / 0.0062). They exhausted the step budget before the rate-based event slowdown (contract §6.3). | closed, 5.4085 / 5.6359; pool 185 / 175, 13 / 15 clusters, every other candidate folded, `complete`. |
| `1-3__two_arcs_60deg` | C06 C08 C17 C18 | Path `1-3` at δ = 60°: two distinct components, each an arc cut by exit TIR at one end and by the entry ray leaving face 1 at the other. Variants `initial_step_0.03` and `initial_step_0.08` on component 0. | `path_infeasible` 0.1397 + `tir_boundary` 0.4749, and `tir_boundary` 0.3820 + `path_infeasible` 0.2325. Seed search: 2 arcs (0.6146, 0.6145), 1 fold. `J_perp` 2581 at the TIR end (`d = 2.4e-8`). |
| `3-5__rank_loss_extremum` | C07 | The seed is the interior minimum of `D_P` (21.84°), where the fiber degenerates to a point. | Both orientations `rank_loss` with no pose. At the seed `σ₂ = 1.1e-16`, `J_perp = 4.2e-17`. |
| `3-5__limits` | C09 C11 | The matrix's 3-5 random seed under options that end the trace early: variants `step_budget` (`maximum_accepted_steps 5`), `arclength_budget` (`maximum_arclength 0.3`), `evaluation_budget` (`maximum_evaluations 15`), `corrector_failure` (`maximum_advance 0.01`, `maximum_retries 0`) and `step_underflow` (`initial_step 0.04`, `minimum_step 0.03`, `maximum_step 0.04`, `maximum_advance 0.01`). Its default `trace_fiber` is the budgets' `reference_curve`. | Per orientation: 6 poses (0.2000); 8 poses (0.2800); 3 poses (0.0800); 1 pose `corrector_failure`; 1 pose `step_underflow`. |
| `3-5__antipodal_target` | C04 | The 3-5 random pose aimed at the antipode `−d` of its own outgoing direction: an algebraic zero of the projected residual. | Both orientations `chart_boundary` with no pose. |
| `13-24-26__boundary_arc_90deg` | C18 C19 | Pyramid path `13-24-26` on the reference pyramid of LI `tests/test_discovery.py` (`prism_h = upper_h = lower_h = 0.5`, both wedges `90° − pyramid_face_angle()`, regular). At 90° it is one arc cut by the path domain at both ends, and the path has no interior critical point. | `path_infeasible` 1.4403 + `path_infeasible` 0.6801. Seed search: 1 arc (2.1205) from 3 clusters. |

Not covered: a deterministic `linear_solve_failure` (contract §11 C09, open)
and the `D3h` prism of C19. A short loop on a pyramid is the matrix's
`13-15-26-28__critical` (0.808).

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

- Controller-internal records (`step_diagnostics`, `closure_diagnostics`,
  `terminal_payload`) and weights (`weight_observables`). By the owner ruling
  of 2026-09-29 a backend is certified on outputs only, and LI evaluates the
  weights itself on the returned poses. Contract §11.1 names, for every §11
  row, what the outputs certify and what they leave to the backend's own
  tests. From the diagnostics the fixtures carry only `J_perp`, the singular
  values and the branch margins (§3.1, §3.2).
- The sampler itself, `check_band_coverage`, and densification behaviour
  (§9.5.7).
- Any symmetry reduction. `symmetry_semantics` is `none` throughout.
