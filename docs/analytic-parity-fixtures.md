# Analytic parity fixtures (LI → Lumice)

[中文](analytic-parity-fixtures_zh.md)

Lumice Integral (LI) exports, at a fixed rev, a directory of JSON fixtures that
Lumice copies into its repository and replays in CI against
`liblumice_analytic` module A v0 (Lumice `doc/analytic-api.md` §4:
`EvaluatePath`, seed search, `TraceFiber[Batch]`, point lists only) and,
since wave 2, against module B, the single-path band sum
([band-sum-contract.md](band-sum-contract.md)). The
direction is one way (owner ruling 2026-09-28): LI is the reference, Lumice
follows. This page is written so that a reader can be implemented on the Lumice
side without reading LI source. The normative semantics behind the fixtures
are `docs/phase1-math-contract.md` §9 (continuation) and §9.5 (discovery) and
`docs/band-sum-contract.md` (the band sum), and the conventions are those of
`docs/conventions.md`.

## 1. Producing the fixtures

```bash
uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
```

- One command writes every fixture of the matrix (§6), the edge cells
  (§6.1), the band-sum cells (§6.2), the module C cells (§6.3) and a
  `manifest.json`, 120 fixtures. On an idle M2 Max (2026-09-29) the export
  and the read-back of the 93 before the `n = 1.307` cell took 64 s together
  (the 94 took 7 min on 2026-09-30 with a second export running beside it);
  from wave 3 on (2026-10-07) all 120 took 5:39 with the read-back, of which
  the 26 module C fixtures took about 2:50 (the three plate-crystal class
  cells at `1e5` poses each dominate).  The final clean-tree repeat on the
  same day took 3:32 with the read-back.
  Before the four family-pinned cells were added, at load average 20–30, the
  export of 89 fixtures took 50 s and the read-back 153 s. The seven band-sum
  fixtures of that set took 38 s, because the statistical rank-0 cell replays a
  `4e6`-pose Haar stream. The 82 fixtures before those took about 1 min with
  the read-back on an idle machine, and the matrix alone took about 16 s
  before wave 2.
- **Deterministic.** The same rev re-exports byte for byte. LI's test
  `tests/test_parity_export.py` exports the matrix in two separate
  interpreters and compares every file. The fixtures do not record time,
  host or paths.
- `--verify` reads every fixture back, recomputes it with the checkout, and
  compares within the fixture's own tolerances (§5). It exits non-zero on any
  failure. `--cells 3-5__random 3-5__limits 3-5__band_sum_plate ...` exports
  a subset (matrix, edge and band-sum cells by name).
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
`edge_cells` key. A band-sum cell (§6.2) is one file,
`<path>__band_sum_<label>.json`, listed under `band_sum_cells` with its
`label`, `rank`, `pose_density` family, `projection` kind and `rationale`; an
export without band-sum cells has no `band_sum_cells` key, so the files and
the manifest of §6 and §6.1 keep their bytes. A module C cell (§6.3) is named
`<path>__mc_<label>`, its files `<path>__mc_<label>__<kind>.json`, listed
under `module_c_cells` with each exported kind's `detects` text; an export
without module C cells has no `module_c_cells` key, so the earlier manifests
keep their bytes. A reader MUST ignore top-level manifest keys it does not
know: later waves add fixture kinds under new keys.

The JSON is UTF-8 with sorted keys. Floats are written in the shortest form
that round-trips to the same float64, so parsing with any conforming JSON
reader (`strtod`) gives the exported bits back. `-0.0` is preserved, and
there are no NaN or infinities. Every fixture has these fields:

| Field | Meaning |
|---|---|
| `format` | `"lumice-integral/analytic-parity"` |
| `schema_version` | `1`. Wave 2 added optional fields (§3.1, §3.2) and renamed or removed none, so every v0 key keeps its bytes (checked against the 2026-09-29 export at `5ea2bde`). A fixture without the wave 2 fields comes from an older export, and its recipes are the v0 ones. A breaking change raises the version. |
| `fixture_kind` | `evaluate_path`, `trace_fiber`, `seed_search`, `band_sum`, or the module C kinds of §6.3 (`dp_field_sample`, `dp_field_topology`, `dp_field_kinks`, `focusing_classify`, `wavelength_critical_table`, `chromatic_diagnose`, `chromatic_class`) |
| `symmetry_semantics` | Always `"none"`. The input is one concrete face sequence and no symmetry reduction is involved (Lumice `doc/analytic-api.md` §3.3 rule 2; `docs/conventions.md` #21). |
| `provenance.li_rev` | Full LI commit SHA of the export |
| `provenance.li_tracked_tree_clean` | `false` if tracked files differed from that commit at export time. Such a fixture should not be copied into Lumice. |
| `provenance.conventions_sha256` | SHA-256 of LI `docs/conventions.md`. A change flags that a convention may have moved. |
| `cell` | `name`, `path`, `category`, `rationale`, `selection`, and for `evaluate_path` a `pose_label`; for `band_sum`: `name`, `path`, `label`, `rationale`, `rank` |
| `input` | The call's arguments (below) |
| `expected` | The reference output (below) |
| `tolerance` | One entry per compared quantity: `{"value": number, "basis": text}` |

### 2.1 Shared input fields

| Field | Meaning |
|---|---|
| `crystal` | The `LUMICE_ANALYTIC_Crystal` scalars, field for field: `kind` (`"prism"` / `"pyramid"`), `height` (prism height or the pyramid's prism band `prism_h`, in units of the reference circumscribed diameter), `face_distance[6]`, `upper_h`, `lower_h`, `upper_wedge_deg`, `lower_wedge_deg`. Fields not used by the kind are `0.0`. Semantics are Lumice `doc/configuration.md` §prism / §pyramid. LI builds `HexPrism.from_lumice(height, face_distance, a = 0.5)` / `Pyramid.from_lumice(height, upper_h, lower_h, face_distance=..., upper_wedge_deg=..., lower_wedge_deg=..., a = 0.5)` — hexagon edge `a = 0.5`, so the circumscribed diameter is 1, the engine's closed-form reference scale (`geo3d_closedform` builds its vertices at `0.5` unit; the parity consumer instantiates the same reference). Angles and structural outputs are scale-free; area quantities (`a_p`, `w`, the tint energies and chromatic weights) are in units of the reference circumscribed diameter squared and are only comparable at this scale. |
| `faces` | The concrete face sequence in Lumice face numbers (entry, internal reflections, exit) |
| `refractive_index` | The ice index of the call |
| `incident_direction` | World propagation direction of the sunlight, sun → crystal (`s = -ŝ`) |
| `pose` / `seed_pose` | Nine numbers, a row-major 3×3 rotation, body → world (`v_W = R v_B`) |
| `target_direction` | World propagation direction of the outgoing light, crystal → observer (`d`) |
| `continuation` | LI's `ContinuationOptions` (contract §9.2, reference defaults of §10.1), every field. Lumice's v0 options block has fewer fields. A backend uses its own equivalents and documents the mapping. |

## 3. The four fixture kinds

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

### 3.4 `band_sum`

Module B (`docs/band-sum-contract.md`): one concrete path, one pose density,
a table of pixels given as directions. Input: `crystal`, `faces`,
`refractive_index`, `incident_direction` (§2.1), and

| `input` field | Meaning |
|---|---|
| `pose_density` | `family` (`random` / `column` / `plate` / `parry` / `lowitz`) and its resolved parameters in degrees (contract §2.2). `normalization_informative` holds LI's `I` (and `Q`); not compared. |
| `sample` | `sampler` and `n`: the antipodal Fibonacci lattice of `n` points (contract §3). |
| `pixels` | `labels` `(P, 2)`, `centre` `(P, 3)`, `corners` `(P, 4, 3)` in cyclic order, `solid_angle` `(P)`: outgoing propagation directions, the sky point is their negative (contract §2.3). `projection` records how LI expanded them (a Lumice linear `render` block, or a single-disk Lambert view) and is informative. |
| `events` | Layer 1 (contract §7.1). Rank 2: `u`, `phi`, `deviation`, `w` of every kept event in some non-singular pixel's band, in LI's order (increasing `D`). Rank 0 under the random density: `w` of every kept event. Absent for a statistical rank-0 cell. |

| `expected` field | Meaning |
|---|---|
| `rank` | `2` (a band sum) or `0` (a point mass, contract §5) |
| `pixels` | In table order: `label`, `status`, `value`; for rank 2 also `delta`, `delta_lo`, `delta_hi`, `K`, `K_rho_pos`, `K_eff` (null when `singular`), `total`, `square` (informative), and `allowance` (below) |
| `pixels[].allowance` | Rank 2, non-singular: `K_rho_pos_subnormal` (band events whose `c_i` is subnormal in LI), the layer-2 allowances `K_layer2`, `K_rho_pos_layer2`, `value_layer2`, `K_eff_layer2`, and the `candidates` they come from (`band_end`, `gate`, `gate_without_finite_D`; contract §7.2) |
| `point_mass` | Rank 0: `m`, `method` (`lattice_mean`, or `haar_stream` with `error_estimate`, `sample_count`, `rng_seed`); a lattice-mean cell also records LI's `haar_check_informative` |

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
- **`band_sum`**, in two layers (contract §7). Layer 1 runs the estimator on
  `input.events`; layer 2 regenerates the sample and runs the whole call. In
  both, every pixel's `status` is equal (a singular pixel has no value). For
  an `ok` pixel of a rank-2 cell: `K` is equal in layer 1 and within
  `allowance.K_layer2` in layer 2; `K_rho_pos` within
  `allowance.K_rho_pos_subnormal` (layer 1) or `allowance.K_rho_pos_layer2`
  (layer 2); `value` and `K_eff` within `value_relative` × |expected|, plus
  `allowance.value_layer2` / `allowance.K_eff_layer2` in layer 2. A rank-0
  cell compares `m` and each pixel's value to `point_mass_relative`; for a
  `haar_stream` cell that tolerance is five of LI's standard errors, relative,
  and a backend with its own estimator error `σ` widens it to
  `5 √(σ_LI² + σ²)`. LI's `--verify` runs layer 1 through both of its forms
  of the sum (the production scatter and the per-pixel gather), checks that
  the fixture's events are its regenerated sample's, and runs layer 2.

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
| Band sum: statuses, layer-1 `K` | exact | Singular iff the pixel contains `s` or `−s`; band membership is `delta_lo ≤ D < delta_hi` on the fixture's own `D` (contract §4.1, §4.2). |
| Band sum: value, `K_eff` (layer 1) | `1e-10` relative | The same sum over bit-identical events: summation order and the rounding of `arccos`, `atan2`, `exp`. LI's scatter and gather agree to `1e-12` (`3e-12` for `K_eff` near one); the spec-only reference implementation (`tests/test_band_sum_spec_reference.py`, scipy quadrature for `I` and the closed form for `Q`) agrees with every exported value to `8e-14`. |
| Band sum: `K_rho_pos` (layer 1) | per pixel `allowance.K_rho_pos_subnormal` | Contract §4.5 requires gradual underflow; a band event whose `c_i` is subnormal in LI is the only one a flush-to-zero backend would count differently (up to 12 per pixel on the Parry cell). |
| Band sum, layer 2 | per pixel `allowance.*_layer2` | Contract §7.2: the lattice points within `edge_epsilon_rad = 1e-9` of a band end, `gate_epsilon = 1e-9` of a validity gate, or with `0 < w ≤ weight_epsilon = 1e-9`, and the most they can move each quantity. On the exported cells every count is 0, so layer 2 is as tight as layer 1. |
| Rank-0 point mass | `1e-10` relative (lattice mean); `5 σ_LI` relative (Haar stream) | Contract §5: the lattice mean is deterministic; LI's Haar stream cannot be reproduced by another backend. |

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
| `3-5__boundary_hugging_r700_c150`, `__r780_c150` | C06 C08 C16 | Loops running along the exit TIR boundary (smallest margin 0.0142 / 0.0062). They exhausted the step budget before the rate-based event slowdown (contract §6.3). | closed, 5.4085 / 5.6359; pool 185 / 175, 13 / 13 clusters, every other candidate folded, `complete`. |
| `1-3__two_arcs_60deg` | C06 C08 C17 C18 | Path `1-3` at δ = 60°: two distinct components, each an arc cut by exit TIR at one end and by the entry ray leaving face 1 at the other. Variants `initial_step_0.03` and `initial_step_0.08` on component 0. | `path_infeasible` 0.1397 + `tir_boundary` 0.4749, and `tir_boundary` 0.3820 + `path_infeasible` 0.2325. Seed search: 2 arcs (0.6146, 0.6145), 1 fold. `J_perp` 2581 at the TIR end (`d = 2.4e-8`). |
| `3-5__rank_loss_extremum` | C07 | The seed is the interior minimum of `D_P` (21.84°), where the fiber degenerates to a point. | Both orientations `rank_loss` with no pose. At the seed `σ₂ = 1.1e-16`, `J_perp = 4.2e-17`. |
| `3-5__limits` | C09 C11 | The matrix's 3-5 random seed under options that end the trace early: variants `step_budget` (`maximum_accepted_steps 5`), `arclength_budget` (`maximum_arclength 0.3`), `evaluation_budget` (`maximum_evaluations 15`), `corrector_failure` (`maximum_advance 0.01`, `maximum_retries 0`) and `step_underflow` (`initial_step 0.04`, `minimum_step 0.03`, `maximum_step 0.04`, `maximum_advance 0.01`). Its default `trace_fiber` is the budgets' `reference_curve`. | Per orientation: 6 poses (0.2000); 8 poses (0.2800); 3 poses (0.0800); 1 pose `corrector_failure`; 1 pose `step_underflow`. |
| `3-5__antipodal_target` | C04 | The 3-5 random pose aimed at the antipode `−d` of its own outgoing direction: an algebraic zero of the projected residual. | Both orientations `chart_boundary` with no pose. |
| `13-24-26__boundary_arc_90deg` | C18 C19 | Pyramid path `13-24-26` on the reference pyramid of LI `tests/test_discovery.py` (`prism_h = upper_h = lower_h = 0.5`, both wedges `90° − pyramid_face_angle()`, regular). At 90° it is one arc cut by the path domain at both ends, and the path has no interior critical point. | `path_infeasible` 1.4403 + `path_infeasible` 0.6801. Seed search: 1 arc (2.1205) from 3 clusters. |

Not covered: a deterministic `linear_solve_failure` (contract §11 C09, open)
and the `D3h` prism of C19. A short loop on a pyramid is the matrix's
`13-15-26-28__critical` (0.808).

### 6.2 Band-sum cells (module B)

The sun is the matrix's (altitude 15°, azimuth 0), `n = 1.31`, except in
`1-3-4-2__band_sum_plate_n1.307` (below). Pixel tables
come from three views: a Lumice linear lens (41 × 41, 60° field, elevation
15°, the sun at pixel (20, 20), about 1.5° per pixel), a Lambert view about
the sun (65 × 65, 30° field radius, about 0.93° per pixel, the sun at the
centre of (32, 32)) and one about the antisun (65 × 65, 60°, about 1.8°); the
rank-0 cells use a small Lambert view about the sun (9 × 9, 5°, the sun at
(4, 4)). The family-pinned cells use the same small view centred on their
spot: the 120° parhelion (azimuth 120°, elevation 15°), the subsun (azimuth
0°, elevation −15°) and the 120° subparhelion (azimuth 120°, elevation −15°).
Each cell's pixels were picked from a scan of its view. Every
pixel is labelled with its table `(row, column)` or `(y, x)`.

| Cell | Path, crystal, `N` | Density, view | Covers | Observed (LI, this rev) |
|---|---|---|---|---|
| `3-5__band_sum_random` | 3-5, canonical column, `2e4` | random, linear | no internal reflection | sun pixel singular; two empty bands (inside the halo); lit `K` 112–384, `K_eff/K` 0.88–1.00 |
| `3-5__band_sum_plate` | same | plate 1°, Lambert (sun) | a narrow zenith family | the single path's parhelion (one side only), value 108 at `K_eff` 21.5; tails to `9e-88`; the mirror side has `K = 258`, `K_rho_pos = 0`; subnormal allowances up to 6 |
| `3-5__band_sum_parry` | same | Parry 1° / 1°, linear | the roll-locked family (reads `e1`, `e2`, `e3`) | upper Parry arc 0.017–1.6 at `K_eff` 2.4–4.7; tails to `8e-238` with `K_eff = 0` (the squares underflow); subnormal allowances up to 12 |
| `3-5-6-7__band_sum_random` | 3-5-6-7, canonical column, `5e4` | random, Lambert (antisun) | two internal reflections (Fresnel `R`) | the antisolar pixel singular (`δ = π`); lit `K` 12–103; a dark pixel beyond the largest deviation |
| `13-15-26-28__band_sum_random` | the matrix's asymmetric pyramid, `5e4` | random, Lambert (antisun) | pyramid faces | lit ring (`D` 121–149°) `K` 44–107; dark on both sides |
| `3-6__band_sum_rank0` | 3-6, canonical column, `2e4` | random, Lambert (sun, small) | a rank-0 point mass, deterministic | `m = 0.118166` (lattice mean; LI's Haar check `0.11845 ± 0.00030`), on pixel (4, 4) only |
| `3-6__band_sum_rank0_plate` | same | plate 1° | a rank-0 point mass, statistical | `m = 0.157 ± 0.011` (Haar stream, `4e6` poses) |
| `3-6-4-8__band_sum_plate` | 3-6-4-8, canonical column, `2e4` | plate 1°, Lambert (120° parhelion, small) | family pinned, reflection group element 6 | spot centre (4, 4) `9.51e-3` at `K = 48`, `K_eff = 14.9` (`δ = 113.548°`); neighbours `1.7e-3`–`7.9e-3`; tails to `5.1e-26`; `K_rho_pos = K/2` (the upside-down half of the band events has `rho = 0`) |
| `3-6-4-8__band_sum_plate_sigma_0.5` | same | plate 0.5°, same view and pixels | the same spot at half the spread | centre `2.14e-2`, horizontal neighbours `4.4e-4`–`5.9e-4`, tails to `2.4e-92` |
| `1-2-1__band_sum_plate` | 1-2-1, thin plate (`height = 0.2`), `2e4` | plate 1°, Lambert (subsun, small) | family pinned, element 11 (basal reflection) | subsun centre `17.9` at `K = 95`, `K_eff = 95.0` (`δ = 30°`); lit along the vertical (`1.4`–`15.6`), `8.5e-16` four pixels across |
| `1-2-3-4-1__band_sum_plate` | 1-2-3-4-1, thin plate, `2e4` | plate 1°, Lambert (120° subparhelion, small) | family pinned, element 5 | centre `0.102` at `K = 123` (`δ = 122.242°`); (4, 6) sits on the path's smallest deviation, 120°, and has the only nonzero layer-2 allowance (`K_layer2 = 3`); (4, 7) is an empty band |
| `1-3-4-2__band_sum_plate_n1.307` | 1-3-4-2, plate with two long faces (`height = 0.3`, `face_distance = (1.5, 1, 1, 1.5, 1, 1)`), `n = 1.307`, **sun on the horizon**, `2e4` | plate 1°, Lambert (small, centred on azimuth 120°, elevation 0) | an index other than 1.31; the entry measure's exit gate at the caller's index (below) | a vertical streak in column 4 (azimuth 120°): (2, 4) `8.56e-4` at `K = 132`, `K_eff = 34.3`, (0, 4)–(4, 4) `9.7e-5`–`8.6e-4`, down to `6.1e-13` at (8, 4); horizontal tails (2, 5) `1.2e-12`, (2, 6) `3.6e-38`; (2, 3) an empty band |

The LI side of the Lambert cells is computed by the same two forms of the sum
as the linear ones: the camera enters LI's estimator only through a pixel's
directions (contract §10).

**Family-pinned cells.** On these paths the `σ → 0` plate family lies in one
level set of `D_P`, so the whole family lands on one sky point, and at
`σ > 0` the spot's width scales with `σ`. The criterion is
`focusing.family_pinned`: rank 2, wedge 0, a fold matrix that commutes with
`R_z` (reflection group elements 3, 4, 5, 6 and 11), and a plate or Lowitz
density. The source is exploration 46.4 (`degenerate-path-family-coverage`).
`3-5__band_sum_plate` is the control: the same family on a path that is not
pinned. Its fold matrix also commutes with `R_z`, and its 60° wedge is what
excludes it. The cells are rows of `band_sum_cells`, not of the path ×
category matrix (§6).

- *Crystal.* 1-2-1 and 1-2-3-4-1 use a thin plate
  (`prism_crystal(0.2)`). On the canonical column the ray leaves through a
  prism face after the basal reflection, so no pose of the `σ = 0` ring (c
  axis vertical, every azimuth) is valid (`w = 0`), and a plate cell there
  would carry only Gaussian tails (`1e-69` and lower). The other element-5
  candidate, 3-1-6-4-8, has no valid pose on the ring on any prism of height
  0.1–1.0, because the ray enters a prism face going down and never reaches
  the top face. 3-6-4-8 stays on the column: it has no valid pose on plates of
  height 0.3 or less.
- *Observed spread (LI self-observation, not an independent check).* On the
  full 9 × 9 view the intensity-weighted standard deviation of the 3-6-4-8
  spot is (1.52, 0.52) px (vertical, horizontal) at `σ = 1°` and
  (0.77, 0.21) px at `σ = 0.5°`. It is unchanged at `N = 5e4`. The 3-5
  parhelion on the 65 × 65 sun view is (0.58, 2.14) px at 1° and
  (0.51, 2.49) px at 0.5°: along its level set it does not narrow. The two
  3-6-4-8 cells carry this contrast on fixed pixels. The `σ` dependence of
  `D` itself (`std/σ ≈ 0.577` for 3-6-4-8, 5.67° fixed for 3-5) is the
  exploration's measurement, with a signed zenith sampler; under the plate
  density these cells use it is `0.818` (a factor `√2`, [phase2.md](phase2.md) §10).
- *Mechanism check outside the band sum.*
  `tests/test_parity_export.py::test_family_pinned_cells_put_their_whole_sigma_zero_family_on_one_deviation`
  evaluates `D` and `w` on the `σ = 0` ring. Each pinned cell has valid poses
  there and a single `D` (spread below `1e-9°`). The control spreads by more
  than 10°. `family_pinned` names exactly the four pinned cells. (The
  `n = 1.307` cell below is family pinned too, element 6, but it is there
  for its index, not for the family.)
- *No fiber-layer cell.* At the 120° parhelion target, 3-6-4-8's fiber is two
  arcs, and on one of them the c axis stays vertical: that arc is the `σ = 0`
  family itself. No contract §11 row certifies this beyond what C17/C18
  already cover with `1-3__two_arcs_60deg`, so it is recorded here and not
  exported.

**The `n = 1.307` cell.** Every other fixture is at `n = 1.31`, where the
package constant and the call's index agree. Path 1-3-4-2 enters through one
basal face, reflects on prism faces 3 and 4 and leaves through the parallel
basal face; with the sun on the horizon the entry is grazing, so the internal
ray lies just inside the critical cone of the call's index,
`cos θ_c = sqrt(1 − 1/n²)`. At `n = 1.307` that cone is wider than the one of
1.31, and the ray falls between the two. The entry measure's exit gate
(contract §3, `geometry.entry_measure`) must use the call's index. Until task
`entry-measure-exit-gate-index` (2026-09-30) LI used the package constant
`N_ICE = 1.31` there, which rejected every grazing event: the spot centre was
`4.8e-10` from 18 non-grazing events instead of `8.56e-4` from 132. Ice Halo's
`src/analytic/entry_measure` already takes the call's index, so its expected
parity on this cell is a plain pass; a backend that gates at a fixed 1.31
fails it by six orders of magnitude on column 4 while (2, 5) and (2, 6), whose
events are not grazing, agree bit for bit either way. The spot's position does
not depend on `n` (parallel entry and exit faces), only its weights do.
`tests/test_parity_export.py::test_non_canonical_index_cell_lights_the_grazing_slab_spot`
pins the lit streak.

### 6.3 Module C cells (wave 3)

The u-S² geometry layer itself — `dp_field` (the `D_P` field, its gates, the
interval partition, the weight kinks), `focusing` (critical-value labels) and
`chromatic` (the colour criterion) — as fixtures for Lumice's schema3 geometry
port (Lumice scrum `schema3-geometry-port`).  The cells carry the anchors the
dissolution probe pinned (scrum `wave3-pull-forward`): C02/C05/C06 and the
splits.  Unlike the module A/B fixtures these are **sun-free**: the field
layer reads the crystal, the face sequence and the index only, so the inputs
carry no `incident_direction`.  Every angle is display delta in radians (no
180° − θ conversion); the cell's index is Lumice's `n(λ)`
(`spectrum.dispersion.refractive_index`, `n(550) = 1.3110129170742788`), and
each fixture records its index and wavelength label.  The kink sweeps carry
400/550/700 nm (the C02/C06 calibers), the wavelength-critical table 450/550/650
nm (task 42.4's caliber).  The chromatic kinds take their pair from
`chromatic.N_RED`/`N_BLUE` (1.307/1.317) and record it with the thresholds.

Per-cell sample points (`dp_field_sample`): a strided subset of about 256
valid points of the 20000-point antipodal Fibonacci lattice plus named
interior anchors (interior critical points, interior slab-axis points,
mid-arc kink points; the boundary extrema sit on `dU_P` itself, where a gate
margin of ~0 makes the exit chain's square root undefined, and belong to the
topology fixture).  The weights follow the twist-invariant convention:
`A_P` is the entry measure at any pose with `R u = ŝ_probe = [0, 0, 1]`,
`T_P` the Fresnel path factor and `w = A_P·T_P` the one kernel
`path_weight.weighted_power` — observable separately before their product.
`D_P`, the gradient norm and the gate margins come straight from the field
layer.

| Kind | Input | Expected (what is compared) |
|---|---|---|
| `dp_field_sample` | crystal, faces, index (+label), `lattice_n`, the sample record, `u` | per point: `d_p`, `gradient_norm`, `valid`, the `margin_names`/`margins` of `U_P`'s gates, `a_p`, `t_p`, `w` |
| `dp_field_topology` | crystal, faces, index (+label), `lattice_n` | `interval_partition` rows `[lo, hi, n_components, n_closed, n_open]`, `critical_values`, interior critical points (position, value, kind, Morse index, gradient norm, Hessian eigenvalues), the boundary loop (pieces: structure + D envelope + endpoint positions, the interior point lists informative; corners with their margin signatures; restricted critical points; plateau value), `degenerate_fold` (axis, axis-point locations, `circle_interior_fraction`, the crease fields), `domain_topology` (counts, `is_disk`, the chart-grid audit) |
| `dp_field_kinks` | crystal, faces, `indices` (label → n), `lattice_n` | per label, per curve: `(step, margin, method, normal, note, failed_seeds, complete)`, arcs (points, values, `closed`, `ends` gate names; points informative), `value_min`/`value_max`/`spread`, `nonfinite_values_dropped` |
| `focusing_classify` | crystal, faces, index (+label), `pose_density` (random), `lattice_n` | `focusing.FocusingClassification.as_json`: mechanism, onsets (value_deg, location, source, profile, jacobian_focusing, gradient_norm, measure_limit, multiplicity), `gradient_norm_range` |
| `wavelength_critical_table` | crystal, faces, `pose_density`, `indices` | `focusing.WavelengthCriticalTable.as_json`: per onset row the per-label values (deg) and `displacement_deg` |
| `chromatic_diagnose` | crystal, faces, `n_red`, `n_blue`, `lattice_n`, the `thresholds` snapshot | `chromatic.ChromaticVerdict`: kind/color/visible/position, the features (kind, source, color, positive_fraction, delta_red/delta_blue/shift/spread/direction_dispersion, contrast, weight, lit_fraction, visible; a `gate_edge` feature carries `null` for `lit_fraction`/`weight`/`direction_dispersion` — convention 3 below), notes, coverage |
| `chromatic_class` | crystal, representative, `family` (plate: sun altitude 9°, 1° zenith, 1e5 poses, seed 3), `n_red`/`n_blue`, `thresholds` | members (the PBD orbit), `lit_members` per index, the verdict with its tint metrics (energies, ratio, TIR fractions, direction dispersion) |

Tolerances: structural fields and labels exact; partition bounds, critical
values and located positions `1e-9` rad (the walk locates corners and extrema
to ~1e-12); Hessian eigenvalues `1e-8` relative; onset values `1e-8` deg
(corner rows of a fixture that emitted the corner tier read
`degrees(EXTREMUM_ATOL)` — convention 1's value half below); gradient
quantities `1e-6` relative; the sampled medians of
`chromatic_diagnose` `2e-3` rad; `a_p`/`t_p`/`w` relative `1e-10`; the
plate-class tint is statistical (`5e-2`; LI's sample is a numpy PCG64
stream, its split-half σ of the ratio is ≤ 3e-3 at 1e5 poses).

Three conventions pin the three spots where a value is not reproducible across
backends at the printed precision.  (1) *Divergent corner gradients*: at an
exit-TIR corner `|grad D_P|` is mathematically unbounded; a numerically
finite evaluation of it is floating-point luck (another backend gets NaN or
a different magnitude).  An onset gradient that is non-finite or at or above
`focusing.DIVERGENT_GRADIENT_NORM` (`1e6`; every legitimate onset/lattice
norm is `≤ ~1.2e2`, the corner evaluations are `≥ 1.8e7`, measured on the
reference-scale cells) is exported as `null` on both sides, and usability
(null vs not) is compared exactly.  The same convention carries the corner's
*value*: the walk's located position on an unbounded gradient is each
platform's rounding luck, and the sqrt fold of `D_P` at the corner carries it
into the value — measured 3.5e-7..7.6e-7 deg across ISAs on Lumice PR #477's
CI (2026-10-09; only the corner rows drifted, the interior/boundary-extremum
rows matched to 1e-14 there; the same family as the 1e-8 cross-ISA red behind
Lumice's ba512cd1).  A fixture whose classification carries such a corner
(`source = "corner"` with a null gradient norm) therefore emits a per-row
corner tier in its tolerance block — `onset_value_deg_corner` for
`focusing_classify`, `values_deg_corner` and `displacement_deg_corner` for
`wavelength_critical_table`, each `degrees(EXTREMUM_ATOL)` ≈ `5.73e-6` deg —
and the corner rows read the tier while every other row keeps the default
`1e-8`/`2e-8` deg.  Key-present means every corner row of the fixture is one
(the emission is fixture-level, keyed on the divergence predicate); a fixture
that someday mixes well-behaved corners (finite gradient norm) with singular
ones must move the tier to per-row emission first.  (2)
*Near-degenerate corner pairs*: two corners whose values differ by more
than the default `1e-8` deg tolerance yet within `EXTREMUM_ATOL` are merged
by `focusing`, and which member survives is each backend's floating-point
order — the merged onset value can shift by the pair's difference, up to the
merge tolerance itself.  A focusing fixture whose path has such a pair
carries `onset_value_deg` widened to `degrees(EXTREMUM_ATOL)` ≈ `5.73e-6`
deg in its tolerance block, with the mechanism in the basis; exactly
degenerate symmetric pairs (diffs at machine epsilon) and every other
fixture keep `1e-8` deg.  (3) *Fields evaluated on a gate's own zero set*:
a `chromatic_diagnose` feature with `kind = "gate_edge"` is walked on the
gate margin's zero level itself, so `lit_fraction` (the rounding sign of the
exit discriminant inside the weight kernel), `weight` (`T_exit` at the
sqrt-residual scale, `~1e-8` against a `~2.5e-2` in-domain median) and
`direction_dispersion` (the median of the finite survivors of a `dD_P/dn`
that diverges on the curve) are each backend's floating-point luck.
Measured 2026-10-08 on the 3-1-5 exit gate: LI's 344 walk points carry
margin residuals within `±5.2e-16` (lit 269/344) while Lumice's own walk
splits its own way (205/344), and the dispersion medians differ by
`6.2e4`.  The same luck holds across platforms of one backend as well:
PR #52's CI (2026-10-08) measured LI's own Linux x86 walk 4/344 away from
the macOS side, the sign not printed by the CI message (the same family
as Lumice scrum-649's 1e-8 cross-ISA red).  These fields are exported as
`null` for every `gate_edge` feature on both sides and availability (null
vs not) is compared exactly;
a fixture that still carries a bare value is rejected with its own
message.  `visible` keeps its bare value — its shift/spread clauses decide
it (the 3-1-5 gate sits 34.6% below `EDGE_MIN_SHIFT_RAD` with spread
`≫ |shift|`), its lit input is not pinned.  An `edge` (weight-kink)
feature keeps every field: the reflected branch is continuous through its
onset and its lit predicate is the structural entry-corridor one (the
3-1-6 kink's `0.845` lit fraction is the corridor-emptiness fraction,
green cross-backend).

**The cells** (15 cells, 26 fixtures; `serves` names Lumice's A-lines,
`schema2-redesign-discussion/conclusions.md` §7: A1 field layer, A2
partition/escape, A3 boundary walk + kinks, A4 focusing/chromatic):

| Cell | Crystal, λ | Kinds | Serves | The anchor it carries |
|---|---|---|---|---|
| `3-5__mc_field` | canonical column, 450/550/650 | sample, topology, classify, table | A1 A2 A4 | the 22° inner edge, the closed form `2 asin(n sin 30°) − 60°` = 21.916127° at n(550); the wavelength table's 0.580962° displacement |
| `3-1-5__mc_field` | canonical column, 550 | sample, topology, classify | A1 A2 A3 A4 | the boundary extremum D_min = 21.916127° (the same closed form); the kind-2 attribution |
| `3-1-5__mc_kinks` | canonical column, 400/550/700 | kinks | A3 | the C02 sweep: 700 nm [129.364, 134.255], 550 [132.458, 136.842], 400 nm [141.004, 143.652]; 131.030° lies in the 700 nm span |
| `3-1-6__mc_field` | canonical column, 550 | sample, topology | A1 A2 | the whole-range partition [0°, 180°] |
| `3-1-6__mc_kinks` | canonical column, 550 | kinks | A3 | the constant kink circle `2 asin √(n²−1)` = 115.945100° at n(550), spread < 1e-12 |
| `3-1-4-5__mc_field` | canonical column, 550 | topology, kinks | A2 A3 | the partition split at exactly 120.000000°; the kink reaching 149.246753° |
| `3-4-1-5__mc_field` | canonical column, 550 | topology | A2 | the complementary-circle twin: the same split, the kink on the second reflection |
| `3-5-6-7__mc_field` | canonical column, 550 | sample, topology, kinks | A1 A2 A3 | the machinery regression: the partition complete from 50.161742°, one domain component (audit `corrected`), kink arcs without NaN |
| `4-8-7-5__mc_field` | **β** `from_lumice(3.0, [2,1,1,2,1,1])`, 550 | sample, topology, classify | A1 A2 A4 | **C05**: the slab partition [0°, 50.161742°, 120°], both intervals (2, 0, 2), the fold on the c axis (22.764% of the crease inside `U_P`), the blade critical value 120°, `classify` = jacobian@blade |
| `4-8-1-7-5__mc_kinks` | β, 400/550/700 | kinks | A3 | **C06**: the basal kink's constant circle 149.246753° at 550, dispersion +1.850629° (400 − 700) |
| `3-1-6__mc_chromatic` | `from_lumice(0.5)`, N_RED/N_BLUE | chromatic | A4 | edge/blue/visible at `2 asin √(n_b²−1)`, shift 3.354°, spread 0 |
| `3-1-5__mc_chromatic` | `from_lumice(0.5)`, N_RED/N_BLUE | chromatic | A4 | the C02 criterion layer: the kink edge blue/visible (σ 4.878° vs Δ 7.325°), the exit gate red/not visible (σ 108.65° vs |Δ| 0.33°) |
| `1-3-5-2__mc_chromatic_class` | rhombic plate `from_lumice(0.5, [1.5,1,1,1.5,1,1])` | chromatic_class | A4 | tint blue, ratio 1.492 (printed digits), 4 of 24 members lit |
| `1-3-4-2__mc_chromatic_class` | same plate | chromatic_class | A4 | the white control, ratio 0.965 |
| `3-5-6-8__mc_chromatic_class` | same plate | chromatic_class | A4 | the impossible-literal class: four other members carry it, white at 1.029 |

A5 (the contour line integral) is **not directly served**: the issue's scope
carries no contour objects, and A5 consumes the field and topology cells
above as its input layer — the limitation is recorded here and in the
manifest rationale.

**Anchor calibers.** The 52 dissolution-probe records were computed at the
truncated index `n = 1.3110129`; the fixtures carry the full `n(λ)`, so
115.945094 → 115.945100 and 50.161740 → 50.161742 (the truncated index
reproduces the recorded digits; both directions are pinned in
`tests/test_parity_export_module_c.py`).  The C02 "red/blue ends" of the
record are `n(700)`/`n(400)` — read from the sweep data, not the chromatic
pair 1.307/1.317 (at 1.307 the span is [129.451, 134.329], which the record's
129.364 excludes) — and the corpus' "blue step at 131°" lies inside the
700 nm span.  Lumice's external anchors (the +1.87° dispersion against the
fixture's +1.850629, the blade's 119.99999999 at grid resolution against the
critical value 120) are the cross-engine calibers this table reconciles.

**Crystals.** The β crystal and the rhombic plate rebuild exactly through the
§2.1 closed-form scalars (`HexPrism.from_lumice`; `face_distance` is the
apothem-ratio field, Lumice's `height` is `h/(2a)`), as the dissolution probe
verified member by member.  One geometric fact belongs here because it bounds
what these fixtures constrain: for prism crystals the side-face normals sit
at fixed azimuths (`i·60°`; `face_distance` translates planes, it does not
rotate them), so `D_P`, `U_P`, the topology, the kink curves and
`focusing.classify` are **position-free** — `fd` and `h` enter the direction
layer through nothing.  The position-sensitive surface is the entry measure:
the β cell's `A_P` (`dp_field_sample`) is positive only on the β crystal (on
the canonical column the corridor of 4-8-7-5 is empty, `A_P ≡ 0` on the whole
lattice), and the rhombic plate's class lit sets differ (4 of 24 members
against the regular prism's 12; `3-5-6-8` white on the plate against not lit
at all on the column).  The export-time probe that recorded this
(`crystal_consumption_probe.py`, task `module-c-fixture-export`) also verified
the position-freedom empirically: β against the regular column agree on the
partition to 2.5e-29°, on the kink envelopes to 4.4e-16 rad, and on the
classification to ulp level.

**The chromatic thresholds.** Each `chromatic_*` fixture embeds
`chromatic`'s criterion constants (`N_RED`, `N_BLUE`, `EDGE_MIN_SHIFT_RAD`,
`EDGE_SPREAD_PER_SHIFT`, `TINT_RATIO_MIN`, `CALIBRATION_WHITE_MAX_DEVIATION`)
as a **recorded snapshot** so the port can replay the criterion itself.  The
authority stays the module (`lumice_integral.chromatic`); a constant changed
there re-exports these fixtures rather than handing the port a second
implementation to keep in step.

**Not covered** (second batch, LI backlog): the new structure objects
(S1/S2-restricted corridors, the A = 0 corridor, S6 junctions); contour
quadrature values; module C cells on pyramids (the module A/B matrix already
carries pyramids; module C's own would follow the same recipe when the port
needs them).

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

Behaviour changes that moved fixture values:

| Date, task | Change | Fixtures that moved |
|---|---|---|
| 2026-09-29, `discovery-cluster-min-index` | Cluster centres are the lowest unassigned pool index (contract §9.5.4 step 1); the reference used to take a Python `set`'s first element, which differs once the set's table shrinks. | Only `3-5__boundary_hugging_r780_c150__seed_search`: `raw_cluster_count` 15 → 13, `admissible_count` 15 → 13, `dedup_merged` 14 → 12; the one closed component is unchanged. The other 92 fixtures differ only in `provenance`. |
| 2026-09-30, `entry-measure-exit-gate-index` | The entry measure's exit gate uses the critical angle of the call's index, `cos_critical(n)`, not of the package constant 1.31. | None: all 93 fixtures are at `n = 1.31` and are byte identical apart from `li_rev`. The new `1-3-4-2__band_sum_plate_n1.307` (§6.2) is the first fixture at another index. |

## 8. Not in these fixtures

- Controller-internal records (`step_diagnostics`, `closure_diagnostics`,
  `terminal_payload`) and weights (`weight_observables`). By the owner ruling
  of 2026-09-29 a backend is certified on outputs only, and LI evaluates the
  weights itself on the returned poses. Contract §11.1 names, for every §11
  row, what the outputs certify and what they leave to the backend's own
  tests. From the diagnostics the fixtures carry only `J_perp`, the singular
  values and the branch margins (§3.1, §3.2).
- The sampler itself, `check_band_coverage`, and densification behaviour
  (§9.5.7). A band-sum backend's sampler is exercised only through layer 2's
  result (§4).
- Any symmetry reduction. `symmetry_semantics` is `none` throughout; a class
  (an L2 row) is the caller's sum of its members (contract §1).
- Band sums over several wavelengths, divergent light, and a deterministic
  rank-0 point mass under a non-random density (contract §1, §5).
- A radiometric comparison of the family-pinned cells with Lumice. The cells
  certify a backend's band sum, not the physics of the degenerate families.
  The Lumice comparison of the 3-6-4-8 plate family (task
  `lumice-radiometric-check-degenerate-plate`, [phase2.md](phase2.md) §10)
  agrees in merged flux within `2e-4` (single seeds within `3e-4`) and in deviation spread within `1e-3` at
  `σ = 0.5–4°`.
