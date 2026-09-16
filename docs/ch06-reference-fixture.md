# Chapter 6 Reference Fixture Specification

> Status: reconstruction specification, 2026-09-16. This document separates
> surviving historical evidence from new canonical choices. It does not claim
> that the Phase I geometry core can already reproduce physical radiance.

## 1. Purpose

Chapter 6 of *Modern Ice Halo Research Notes* contains two artifacts from the
lost direct-integration prototype:

- `state_space_integrate.jpg`, a diagnostic view of a one-dimensional pose
  fiber and several unnamed quantities sampled along it;
- a `251 x 801` local 3-5 tangent-arc radiance strip, preserved as both raw
  float data and a display-rendered JPEG.

The fixtures below serve three different purposes and MUST remain distinct:

1. preserve and explain the historical artifacts;
2. provide a fully specified, implementation-independent single-fiber truth
   case for the current reference solver;
3. define the intended image-level scene without pretending that its missing
   physical stages are implemented.

## 2. Provenance Vocabulary

Every recovered value uses one of these labels:

| Label | Meaning |
|-------|---------|
| `historical-direct` | Present in the chapter text, an author clarification, old filename, old script, image metadata, or raw bytes. |
| `historical-inferred` | Supported by two or more surviving observations but not explicitly serialized by the prototype. |
| `canonical-new` | Chosen after the prototype was lost to make a reproducible replacement fixture. |
| `unknown` | No surviving evidence is sufficient. The implementation MUST NOT invent a value silently. |

The 2026 Lumice recreation is an independent validation artifact. Its settings
are `canonical-new` unless a separate historical source supports them.

## 3. Historical Artifact Fixture

### 3.1 Raw radiance array

The authoritative historical numeric artifact remains in the Writing-Lab
project as `06-monte-carlo-vs-integration/data/data_251x801.bin`.

| Property | Value | Provenance |
|----------|-------|------------|
| SHA-256 | `aeeeaa5488d5b6225e960f605cc743904e5ac81b14e4dd1798666a1c3d408453` | `historical-direct` |
| Storage | headerless little-endian IEEE-754 float32 | `historical-direct` from the old Matlab reader and byte count |
| Logical shape | `(height=801, width=251)` | `historical-direct` |
| Value range | `[0, 2.205009698867798]` | measured from the surviving bytes |
| Zero fraction | `0.05561772883497222` | measured from the surviving bytes |
| Nonzero bounding box | rows `40..800`, columns `0..250` | measured from the surviving bytes |

The byte array has no embedded units, coordinate axes, normalization, scene
parameters, or solver version. Consumers MUST bind it to this specification
rather than treating a bare binary file as a self-describing truth source.

### 3.2 Display artifact

The old Matlab display path reads the binary as `[251, 801]` and transposes it.
It then applies:

```text
white_lim = percentile_99(monte_carlo_reference)
black_lim = 2^-7 * white_lim
y = (x + black_lim) * white_lim / (x + white_lim)
y = log10(y)
y = (y - log10(black_lim)) /
    (log10(white_lim) - log10(black_lim))
y = srgb_inverse_gamma(y)
```

The missing `column1.0_rp35_180+15.mat` supplied the Monte Carlo array used for
`white_lim` and the `img_x`/`img_y` axes. Consequently:

- raw-value regression MUST compare the float array, not the JPEG;
- display regression MAY compare `column1.0_direct.jpg` after explicitly
  declaring reconstructed display limits;
- the fitted historical limits (`white_lim` approximately `0.482`,
  `black_lim` approximately `0.00377`) are diagnostic estimates, not exact
  restored metadata.

Direct raw values and the old direct JPEG have rank correlation `0.9861` in
the native vertical orientation. A vertical flip reduces it to `0.5443`.
Horizontal sign remains ambiguous from this near-symmetric image alone.

### 3.3 Historical scene evidence

| Property | Value | Provenance |
|----------|-------|------------|
| Crystal | hexagonal column, height/radius ratio `1.0` | `historical-direct`: `column1.0` filenames |
| Ray path | side faces `3-5` | `historical-direct`: `rp35` filename and chapter text |
| Solar altitude | `15 deg` | `historical-inferred`: `180+15` filename plus the old all-sky annotation |
| Image size | `251 x 801` | `historical-direct` |
| Monte Carlo levels | `10^5, 10^6, 10^7, 10^8, 10^9` rays | `historical-direct` |
| Refractive index | ice model used by the lost prototype | `unknown` as serialized metadata; `1.31` is the Phase I canonical value |
| Wavelength/spectrum | grayscale/single-channel result | exact old wavelength is `unknown`; `550 nm` is `canonical-new` |
| Pixel angular axes | missing with the `.mat` file | `unknown` |
| Pose-density width | column-oriented distribution | exact old width is `unknown`; `0.5 deg` is `canonical-new` |

The independently recreated Lumice scene uses a point source at altitude
`15 deg`, a `550 nm` monochromatic spectrum, column zenith standard deviation
`0.5 deg`, a linear `6 deg` field of view, view `(azimuth=0 deg,
elevation=-15 deg)`, and resolution `251 x 801`. After a small integer
registration its `10^9`-ray image correlates with the old `10^9`-ray image at
approximately `0.963`. This supports it as the canonical validation scene, not
as proof of exact historical settings.

### 3.4 Historical convergence evidence

Using the old JPEGs in their stored orientation, correlation with the direct
JPEG increases monotonically with Monte Carlo sample count:

| Rays | Correlation with direct JPEG |
|------|------------------------------|
| `10^5` | `0.155` |
| `10^6` | `0.410` |
| `10^7` | `0.808` |
| `10^8` | `0.958` |
| `10^9` | `0.985` |

These numbers are display-space historical evidence. They are not radiometric
error estimates because the original exposure calibration is incomplete.

### 3.5 Historical diagnostic-figure semantics

The author clarified the intended semantics of `state_space_integrate.jpg` on
2026-09-16. These statements are `historical-direct` author testimony even
though the original numeric arrays and plotting code are no longer available.

The upper-left panel visualizes SO(3) through unit quaternions. Because the unit
constraint leaves three degrees of freedom, the historical plot uses the first
three quaternion components. Its blue points are feasible results from a
fixed-resolution parameter-space prescan; a first feasible pose could seed the
subsequent solve. The red curve is the one-dimensional solution fiber produced
by that solve. The exact quaternion component ordering, sign-continuation rule,
scan spacing, and feasibility tolerance remain `unknown`.

The lower-left panel maps the same solution fiber to three Euler-like crystal
coordinates: longitude and latitude of the crystal C axis, plus rotation about
that axis. Its blue points are the adaptive PDE/continuation samples, which
become denser at high curvature. The red curve is an interpolated densification
of those samples for display. Exact angular signs, wrapping intervals, units,
and interpolation method remain `unknown`.

The right panel contains factors from four physical groups: optical throughput
(principally Fresnel), finite-face geometric visibility or mutual occlusion,
the Jacobian, and the SO(3) pose probability density. The thick black curve is
their final product, whose line integral over the one-dimensional solution
gives that pixel's brightness. The color-to-factor mapping and horizontal-axis
parameterization are no longer known.

The incident/outgoing pair used for this diagnostic is also unknown. It may
have been a point on a plate-crystal parhelic-circle path or a column-crystal
lower-tangent-arc path. Therefore this JPEG documents the prototype algorithm
and data flow, but MUST NOT be treated as numeric evidence for the canonical
3-5 single-fiber fixture.

## 4. Canonical Single-Fiber Fixture

This is the current reproducible geometry fixture. It is not asserted to be a
pixel sampled from the historical strip.

| Input | Value | Provenance |
|-------|-------|------------|
| Refractive index | `1.31` | `canonical-new`, Phase I reference |
| Incident direction | `minimum_deviation_incident(1.31)` = `[-0.755628877161269, 0.655, 0]` | analytic construction |
| Seed | `Exp([0.15, 0.08, -0.05])` in right-trivialized SO(3) coordinates | `canonical-new` |
| Target | outgoing direction of the seed = `[-0.945855940121069, 0.322542861173560, -0.036368162500488]` | derived fixture value |
| Path | fixed `3-5` refraction | Phase I optics fixture |
| Scalar dtype | float64 | Phase I reference contract |

Current expected evidence:

| Output | Expected value |
|--------|----------------|
| Terminal state | `closed / closed_loop` |
| Stored poses | `193` |
| Accepted steps | `192` |
| Maximum target residual | at most `5e-16` in the recorded reference environment |
| Fiber length | `3.857976632802349` rad |
| Closure gap | approximately `1.05e-14` rad |
| Normal Jacobian range | approximately `0.02494 .. 0.04458` |

Tests MUST use the tolerances and invariants in `phase1-math-contract.md`, not
freeze incidental accepted-step counts as a correctness requirement.

### 4.1 Canonical pixel fiber with named physical factors

The fixture above binds a self-consistent target (seed maps to itself). The
physical-integrand stage instead binds one real pixel of the section 3.3
canonical scene. All values are `canonical-new`; the implementation lives in
`lumice_integral.canonical_scene` (single authority for constants and problem
assembly) and `lumice_integral.camera` (pixel-to-direction adapter).

| Input | Value | Provenance |
|-------|-------|------------|
| Camera | Lumice `linear` lens, `fov = 6 deg`, resolution `251 x 801`, view `(azimuth 0, elevation -15)` | `canonical-new`, section 3.3 |
| Pixel-to-direction convention | transcribed from Lumice `v4.6.0` `MakeCameraRotation` / `ProjectExitToPixel` / `ComputeLensScale` (read as evidence, never linked); same chain as the writing project's `halo_notes/sim/projection.py` | convention evidence |
| Sun | altitude `15 deg`, azimuth `0`; `s = -(cos 15, 0, sin 15)` | `canonical-new` |
| Pixel | row `150`, column `150` (pixel centre `u = 150.5`, `v = 150.5`); sky elevation `-9.0395 deg`, azimuth `+0.6024 deg`; deviation from the sun `24.047 deg` | `canonical-new` |
| Pixel rationale | on the lit `3-5` band of the historical strip (raw value about `0.23`), about `2.2 deg` below the inner-edge caustic, right of the sun azimuth (`3-5` chirality) | historical raw array plus screen handedness |
| Target `d` | `[-0.9875255807607193, -0.010382806499944452, 0.1571159593179154]` (crystal to observer; the sky direction negated at the adapter boundary) | derived |
| Crystal | hexagonal column `h/a = 1.0` (`HexPrism.from_ratio(1.0)`, `a = 1`) | `historical-direct` ratio |
| Pose density | c-axis zenith Gaussian, mean `90 deg`, std `0.5 deg`, uniform azimuth and spin, relative to Haar probability | `canonical-new` |
| Seed | `Exp([-1.6021189246370302, -0.06647507226372225, 0.614105700979059])` | one recorded prescan, below |

Seed provenance: `scripts/discover_canonical_pixel_seed.py --row 150 --column
150` with `400000` Haar-uniform samples, `2 deg` direction tolerance, `12`
Newton-corrected candidates, RNG seed `20260916`; among the admissible
survivors (`path_3_5_domain` valid, `entry_measure > 0`) the one whose c-axis
zenith is closest to `90 deg` was frozen. This is a fixture-selection scan,
not component discovery; completeness of the component set remains `unknown`.
`lumice_integral.discovery.discover_components` on the same pixel and RNG
seed (`tests/test_discovery.py`) finds one closed component of arclength
`4.758247` from `7` admissible clusters, which is procedural evidence for a
single component, not a completeness certificate.

Current expected evidence (Mac reference environment):

| Output | Expected value |
|--------|----------------|
| Terminal state | `closed / closed_loop` |
| Stored poses | `120` (not a correctness requirement) |
| Fiber length | `4.758228` rad |
| c-axis zenith along the loop | about `87.39 .. 90.09 deg` |
| `rho_pose` | `1.1e-4 .. 91.43` (dimensionless, Haar-relative) |
| `entry_measure` | `0.278 .. 0.559` (`length^2`, `a = 1`) |
| `fresnel_transmission` | `0.9354 .. 0.9416` |
| `path_validity` | `1` at every accepted pose |
| Normal Jacobian range | about `0.0822 .. 0.1497` |
| `visibility`, `source_factor`, `pixel_factor`, `other_radiometric` | `unavailable` |
| Line integral `value` (Haar-converted, `partial`) | `4.728847630` with `error_estimate` about `1.2e-8` (`raw_value` about `373.374843`, `raw_error_estimate` about `9.3e-7` before the `1/(8 pi^2)` factor) |
| Quadrature method | adaptive composite Simpson over chord-parametrised edges, corrector-retracted midpoints, Richardson error estimate; `epsilon = 1e-6`, `relative_tolerance = 1e-8`, `maximum_refinement_depth = 24` |
| Quadrature work | about `235` refinements, `1417` adaptive nodes (`120` accepted plus retracted midpoints), maximum depth reached `16`, no depth exhaustion, no retraction failure |
| Convergence order | median per-edge Richardson order about `3.99`; global uniform-bisection order about `1.99` because `entry_measure` has slope jumps inside edges `3, 17, 32, 47, 77, 92, 106` (footprint-clipping vertex events), see below |

Quadrature evidence (`tests/test_quadrature.py`): on the analytic circle a
constant weight reproduces `2 pi / (1 + epsilon)` to `1e-13` and the Haar
identity `1/(4 pi)` within `epsilon`; the weight `1 + cos(theta)/2` matches
`2 pi / (1 + epsilon)` inside the reported error with an empirical order of
`4.00`; reversing the seed orientation (`initial_tangent_sign = -1`) keeps
the arclength and the integral. On the canonical pixel fiber the integrals
for `initial_step` `0.03 / 0.04 / 0.08`, `relative_tolerance` `1e-6 / 1e-8 /
1e-9`, and both seed orientations agree within the sum of their error
estimates (observed differences `3e-11 .. 5e-7`), without comparing sample
counts. The integrand is only piecewise smooth: `entry_measure` (a clipped
polygon area) changes slope inside seven edges, so uniform bisection shows
order about `2` there while the smooth edges show Simpson's `4`; the adaptive
pass localises those kinks (depth `16` at `1e-8`, `20` at `1e-9`), which is
why the default depth is `24`. The value is `partial`: one component from one
seed, completeness `unknown`; `visibility` and the radiometric factors are
not in the product.

## 5. Figure Capability Matrix

| Figure or chapter need | Data owner | Current status | Missing capability |
|------------------------|------------|----------------|--------------------|
| ch06 crystal-orientation schematic | Writing-Lab drawing code | Supported | None in Lumice Integral; not a numerical-solver responsibility. |
| ch06 ray-splitting schematic | Writing-Lab drawing code | Supported | None in Lumice Integral. |
| ch06 all-sky Monte Carlo example | Lumice through Writing-Lab validation glue | Supported | Not a Lumice Integral product output. |
| ch06 pose-fiber geometry | Lumice Integral | Supported for one supplied regular seed/component; seeds for one pixel can come from `lumice_integral.discovery` | Add continuous-sign unit-quaternion and C-axis longitude/latitude/spin adapters; the prescan-cloud figure still needs recorded spacing/feasibility output from the discovery scan. |
| ch06 solver/Jacobian diagnostics | Lumice Integral data; Writing-Lab presentation | Supported as versioned figure data | A production plotting consumer still belongs in Writing-Lab; an independent prototype consumer has been verified. |
| ch06 named physical-factor curves | Lumice Integral | Supported for `rho_pose`, `entry_measure`, `fresnel_transmission`, `path_validity` on the canonical pixel fiber (section 4.1, figure-data v2 `weight_<name>` arrays); the pointwise final `integrand` curve (product over `J_perp + epsilon`) is exported alongside | `visibility` (finite-face obstruction) and the radiometric factors remain unavailable. |
| ch06 one-pixel integrand/integral | Lumice Integral | Supported as a `partial` value: converged adaptive line quadrature over the closed canonical fiber with error estimate and order evidence (sections 4.1 and 6, `result.quadrature`); single-pixel component discovery is available as a separate primitive (`lumice_integral.discovery`, procedural `completeness` only) | Integration of discovered components into the quadrature product; a completeness certificate; pixel averaging (point value only); the missing factors above. |
| ch06 `251 x 801` direct strip | Lumice Integral | Supported as a `partial` physical rerender: `scripts/render_ch06_strip.py` (`lumice_integral.strip_pixel` / `strip_driver` / `strip_io`) renders any window of the canonical `251 x 801` grid with column-wise hot-start continuation, cold-prescan fallback and spot checks, and writes float64/float32 raw in the historical layout plus a per-pixel status layer and `provenance.json`; pixel model: pixel-centre point value with `epsilon` regularisation (section 7, stage 4) | A completeness certificate (the status layer is procedural); the missing factors of the one-pixel row; sub-pixel averaging is implemented but off by default (6-10x cost; `O(10-40 %)` effect in the centre-column caustic band, section 7 stage 4); finite-sun averaging; rows `600-800` (the lower quarter) come back `unknown` with value `0` because every discovery candidate there stays `incomplete` (section 7 stage 4); a full-resolution run is `1-1.5 days` on a 30-core machine. |
| ch10 halo-map/Jacobian/fold figures | Lumice Integral numerical data; Writing-Lab presentation | Partially supported | Target sweeps and singular/fold localization beyond one regular fiber. |
| ch11 orientation-family comparison | Lumice Integral and/or independent Lumice validation | Not supported by the current ordinary-density slice | Pose-density models, physical weights, image driver; exactly constrained families require a separate measure/domain contract. |

## 6. Figure-Data Product

Solver objects are useful in Python but are not a stable boundary for the
writing project. `export_fiber_figure_data` writes versioned JSON metadata plus
an NPZ array payload. The canonical fixtures can be exported with:

```bash
uv run python scripts/export_path_3_5_figure_data.py <output-directory>      # section 4, geometry only
uv run python scripts/export_canonical_pixel_figure_data.py <output-directory>  # section 4.1, with weights
```

The current `lumice-integral.figure-data/v2` payload contains:

```text
schema: lumice-integral.figure-data/v2
metadata:
  path, incident, target, material, wavelength
  pose convention, metric/measure, solver options
  component scope/completeness, terminal state
  conventions: haar_to_dvol_g_factor (1/(8 pi^2)), coarea_denominator (J_perp)
arrays:
  poses, cumulative_arclength, tangents, residual_norm
  singular_values, normal_jacobian, condition
  named branch margins
  weight_<name> for every available factor (one float64 sample per pose)
  integrand: rho_pose * entry_measure * fresnel_transmission * path_validity
             / (normal_jacobian + epsilon) per pose (only with quadrature)
weights (result.weight_observables):
  each requested factor: status, unit, normalization, array name or null
quadrature (result.quadrature, null when the export ran without it):
  status, method, fiber_status, coverage, component_completeness
  density_factor_name, factor_names, epsilon, relative_tolerance,
  maximum_refinement_depth
  refinements, maximum_depth_reached, node_count
  value, error_estimate (Haar-converted), raw_value, raw_error_estimate,
  haar_to_dvol_g_factor
  convergence_order_estimate, convergence_order_note,
  raw_convergence_order_levels, convergence_order_node_count,
  median_edge_convergence_order, low_order_edges
  refinement_failures, depth_exhausted_edges, integrand_array
```

Schema history:

- `v1`: geometry, diagnostics, and branch margins; `result.weight_observables`
  mapped each factor name to a bare status string and no weight arrays existed.
- `v2` (physical integrand stage): `result.weight_observables` values become
  objects `{status, unit, normalization, array}`; every `available` factor adds
  a `weight_<name>` array whose metadata entry carries the factor unit;
  unavailable factors keep `array: null` and no stand-in values. The version
  string changes because the field type changed; readers of `v1` status
  strings must read `status` instead. The only known `v1` consumer was the
  out-of-repository prototype below, which read geometry/diagnostic arrays
  only (checked: no checked-in code reads `weight_observables` values).
- `v2`, line-quadrature stage (`task-single-fiber-line-quadrature`): the
  optional `result.quadrature` object and the `integrand` sample array are
  populated when the exporter is given a `QuadratureResult`; both are pure
  additions (`quadrature` was a documented placeholder before, no existing
  field changed type), so the version string stays `v2`. `normal_jacobian`
  remains the unregularised `J_perp`; `epsilon` lives only in the integrand
  and in `result.quadrature.epsilon`.

JSON metadata carries the semantic names, units, conventions, shapes,
and SHA-256 of the NPZ payload. Empty failed fibers are represented without
invented samples; non-finite unavailable closure values become JSON `null`.
The format does not serialize arbitrary Python objects or require Lumice at
read time. The canonical fixture produces byte-identical JSON and NPZ files on
repeated exports in the recorded reference environment.

An independent prototype consumer has loaded only these two files and produced
an orientation-body-axis projection, local-map/domain-margin curves, and
solver diagnostics without importing Lumice Integral. That experiment proves
the boundary is sufficient for current geometry figures; it is not a checked-in
chapter plotting implementation and does not recover the historical
color-to-factor mapping.

## 7. Acceptance Stages

1. **Historical loader**: validate the raw hash/shape/dtype and reproduce the
   recorded raw statistics without assigning invented coordinates.
2. **Single-fiber figure data**: export the canonical geometry, diagnostics,
   provenance, and explicit unavailable-factor states.
3. **Physical one-pixel result**: expose every named factor, the coarea
   denominator, quadrature refinements, and a convergence estimate. Status:
   the four factors of section 4.1 and `J_perp` are exposed pointwise with
   units and normalization; the adaptive line quadrature of section 4.1
   reports method, refinements, node count, value, error estimate, `epsilon`
   and order evidence for the canonical pixel fiber (`tests/test_quadrature.py`,
   `result.quadrature` in the figure data). The value stays `partial` (single
   component, missing factors); strip-level coverage is stage 4.
4. **Historical image scene**: render the canonical `251 x 801` strip and
   compare raw profiles with the historical binary plus an independently
   converged Lumice result after coordinate/radiometric alignment. Status
   (`task-strip-image-driver`, 2026-09-16): the strip driver exists and its
   output format is self-describing (`lumice_integral.strip_io`,
   `strip_float64.bin` / `strip_float32.bin` in the `(801, 251)` historical
   layout, `status_uint8.bin` with a documented bit mask, `provenance.json`
   with per-parameter provenance tags, all numerical options and payload
   SHA-256). Every pixel value is the linear sum over the integrated closed
   `3-5` components of the section 4.1 partial integrand; pixels whose
   discovery left an unclassified candidate or whose production trace did not
   close carry the `unknown_completeness` status bit and hold the partial sum
   (never `NaN`), so they are distinguishable from complete zeros.
   Pixel-model probe at the `22 deg` inner-edge caustic (columns 150 and 125,
   the model's edge row is 57 in both; `3 x 3` sub-pixel grid;
   `scratchpad/.../artifacts/pixel_model_probe_results.csv`): at column 150
   (25 columns off centre) the point value differs from the sub-pixel mean by
   `+5 %` on the edge row, `+4 %` two rows below, and `< 0.5 %` from the
   fourth row on; at column 125 (the image centre, where the edge is two
   orders of magnitude brighter) the difference is `+40 %` on the edge row
   (whose point pixel is itself `unknown`), `-33 %` two rows below, and still
   `5-7 %` with changing sign at rows 60, 61 and 63, i.e. the probe did not
   reach a row where the pixel model stops mattering. One row above the edge
   the point value is `0` while sub-pixels already straddle the caustic. The
   sub-pixel model costs `6-10x` per pixel. The image default is therefore
   the point model, and the caustic band near the centre column is a known
   `O(10-40 %)` pixel-model effect, not averaged at full cost; the driver's
   `--pixel-model subpixel --subpixel-rows a:b` row-band option exists for a
   later targeted rerender once more columns have been probed.
   Rendered coverage (2026-09-17): a full-height preview of every ninth
   column (28 columns, `22428` pixels, point model, `home-wsl`, 28 workers,
   `4.1 h` wall clock, `3.6-4.1 h` per column) is the delivered product
   (`scratchpad/.../artifacts/home-wsl-preview-step9/`); the full `251 x 801`
   render resumes from those column checkpoints on the same machine (`30`
   workers, projected `1-1.5 days`) and its numbers are appended to the task
   SUMMARY when it finishes. Per lit pixel the mean cost is `5.4 s`, of which
   line quadrature is `4.4 s`; hot start succeeds on `99.4 %` of lit pixels.
   Findings on that coverage:
   - Above the inner edge the strip is dark; rows `57-600` are lit and
     complete in every rendered column. The edge is at row `57` in the centre
     columns and rises to row `49` at the outer columns (the arc curvature),
     while the historical raw first lights at row `47` at columns 126 and 153
     and at row `44` at column 198: a consistent `10`-row offset, plausibly
     from the `historical-inferred` sun elevation of section 3.3; the offset
     is reported, not corrected.
   - Rows `600-700` are `43 %` `unknown_completeness` and rows `700-800` are
     `100 %` `unknown`: cold discovery there returns `9-14` candidates per
     pixel that all stay `incomplete` after a full production-budget retrace
     (`event_incomplete_retry` equals the candidate count, no
     `production_not_closed`), so these pixels hold the partial sum `0` with
     the status bit set while the historical raw is lit (`0.0085` mean). This
     is the incomplete-candidate regime of `(700, 150)` / `(780, 150)` from
     the discovery task, now known to span the whole lower quarter of the
     strip; on that preview it cost `71-73 s` per pixel (discovery
     semantics, out of the driver task's scope). Only `15` of the `4045`
     unknown pixels are lit.
     Early exit (task-discovery-stall-early-exit, 2026-09-17):
     `explore-continuation-degenerate-stall-diagnosis` traced the cost to
     `_adapt_accepted_step`, whose `not clear_of_event` term shrinks the
     step whenever a domain margin (here `exit_snell_discriminant`) sits
     below `event_slowdown_margin` without tending to zero, so the step
     clamps to `minimum_step` and never grows again until the budget is
     spent. The production retrace of such a candidate repeats its
     discovery steps exactly (same seed, same numerics, larger budget) and
     crawls the same floor, so `strip_pixel` now reads the discovery trace
     it already paid for: a `step_budget` candidate whose last
     `stall_floor_window = 100` accepted steps all proposed `minimum_step`
     (`discovery.is_floor_locked`) is kept `incomplete` without a retrace
     and counted as `incomplete_stall_skip`; the pixel classification is
     unchanged (`unknown`, value `0`). The window is a procedural
     calibration, not a proof: on `82` candidates of `10` pixels
     (`700/150`, `780/150`, `720/27`, `790/216`, `658/72` in the band;
     `49/0`, `50/9`, `54/45`, `57/99`, `54/54` at the inner edge) every
     candidate that the production budget does close (`15`, after
     `1241-1298` steps) never proposed `minimum_step` even once, while
     `59` of the `60` band candidates end in a trailing floor run of
     `141-238` steps; the window sits inside that gap and gives `0`
     false positives on this sample. Not covered: a band candidate that
     leaves the floor before the budget ends (`1` of `60`), candidates that
     stall without touching the floor (`54/54`), and the hot-start path
     (`_hot_start_all` has no cheap discovery trace to read). Same-machine
     cost of `render_pixel(700, 150)` (M2 Max): `29.6 s` with the skip
     disabled (`stall_floor_window = 251`) against `3.0 s` with it, i.e.
     the cold prescan plus twelve 250-step traces. Evidence:
     `scratchpad/scrum-ch06-direct-integration/task-discovery-stall-early-exit/`
     (`scripts/calibrate_stall_window.py`, `artifacts/stall_calibration.json`).
   - Morphology against the historical raw (native orientation, Spearman
     rank correlation): `0.990` on the `16876` pixels lit in both, `0.970` on
     the `18383` complete pixels, `0.657` on all rendered pixels (the
     difference is the unknown lower band plus the edge offset; lit-fraction
     agreement `0.81`). Against the Lumice remake grey PNG: `0.682` on all
     rendered pixels. Row profiles across the 28 columns give Spearman
     `0.99-1.00` versus historical at rows 150/300/500; the brightest column
     and the right-of-centre asymmetry (ours `1.75` versus `1.39` mean at row
     150, historical `0.081` versus `0.068`) sit on the same side in both, so
     the `3-5` = right-parhelion chirality is consistent with the historical
     raw. Left-right flip changes the whole-mask correlation by only `2e-5`,
     so correlation alone does not pin chirality; the side agreement above does.
   - Mac versus `home-wsl` cross-check on the 20 pixels both rendered
     (column 153, rows 140-159): maximum relative difference `3.1e-7` (below
     the `1e-6` quadrature tolerance), component counts identical, status
     bits identical except the window-relative `cold_discovery` spot-check phase.
   Comparison with the historical raw and the Lumice remake is morphology only
   (rank correlation, profiles, side agreement); radiometric normalisation is
   not aligned (the strip is the partial integrand, the historical raw has
   unknown units, the Lumice PNG is tone-mapped 8-bit). Scripts and JSON:
   `scratchpad/scrum-ch06-direct-integration/task-strip-image-driver/`
   (`scripts/compare_with_historical.py`, `scripts/cross_check.py`,
   `artifacts/compare-home-wsl-preview-step9/comparison.json`,
   `artifacts/cross_check_preview.json`).
5. **Writing-Lab figures**: consume only the versioned data product; composition,
   typography, annotations, and chapter-specific styling remain there.

## 8. Reproduction Guidance and Remaining Unknowns

The historical panel structure is now understood, but exact reproduction is
still limited by missing data and conventions:

1. The quaternion view needs an explicit component order and continuous sign
   choice because `q` and `-q` represent the same rotation.
2. The C-axis longitude/latitude/spin view needs explicit sign, wrapping, pole,
   and unit conventions. It may be derived from current rotation matrices.
3. The upper-left blue prescan cloud is not continuation output; reproducing it
   requires a seed-discovery scan with recorded spacing and feasibility tests.
4. The right panel should use explicit modern labels for the four named factor
   groups and final product. Historical colors and horizontal parameterization
   MUST NOT be guessed.
5. The historical incident/outgoing pair is unknown, so a modern reconstruction
   MUST identify itself as the canonical 3-5 fixture rather than the old pixel.

A new diagnostic figure SHOULD label cumulative SO(3) arclength, every physical
factor, the final integrand, and solver/interpolation samples explicitly. This
preserves the historical explanatory structure without manufacturing lost
metadata.
