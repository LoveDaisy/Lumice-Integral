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
| `run-option` | Not a fixture value: a run of the strip renderer chose it (for example `--pose-density-family`); `provenance.json` records it in place of the `canonical-new` value it replaces. |

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
| Crystal | hexagonal column, height/edge ratio `h/a = 2.0` | `canonical-new`: the `column1.0` filenames carry a convention-dependent number, not the crystal itself; see the height note below |
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

Crystal height note (2026-09-20). The `column1.0` filenames and the Lumice
remake's `"height": 1.0` are the same number under the Lumice convention,
`h / (base circumscribed diameter)`: in Lumice `src/core/geo3d_closedform.cpp`
(read as evidence, never linked) the six side faces sit at inradius
`sqrt(3)/4 * dist` and the bases at `z = +-h/2`, so `dist` is the circumscribed
diameter and the hexagon edge is `a = dist / 2`. Lumice Integral's
`HexPrism.from_ratio` takes `h / a` with `a` the hexagon edge, so the same
crystal is `h / a = 2 * 1.0 = 2.0` (`canonical_scene.LUMICE_HEIGHT_OVER_DIAMETER`
and `CANONICAL_HEIGHT_RATIO`). Until 2026-09-20 the canonical scene used
`h / a = 1.0`, half the Lumice height; a single-column probe (column `126`,
rows `100-650`, `scrum-strip-pipeline-v2/task-strip-rerender-and-compare`,
`defect2_findings.md` section 8) showed that the `h / a = 1` crystal loses the
historical plateau at rows `175-400` (ratio to row `150` falling `0.97 -> 0.55`)
while `h / a = 2` reproduces it (`0.98-1.06`). The ratio is therefore
`canonical-new` evidence tied to the Lumice convention, not a historically
recorded crystal dimension.

Lumice float oracle (2026-09-20, task `lumice-raw-profile-oracle`). The
Lumice remake above is only known as a tone-mapped 8-bit PNG, so it can
arbitrate morphology but not radiometry. `Lumice render --format npy`
(Ice Halo `747b2ec2`, run as an external oracle, never linked) exports the
unexposed linear XYZ accumulator; the Y channel is proportional to the
energy that landed in each pixel (`raw[p] = sum of w_ray * CMF(550 nm)`,
no exposure, no per-sr normalisation, no gamma). Two seeded single-worker
runs of the `band1e9` scene (`ray_num 5e8`, seeds `7` / `11`, `460 s` each
on the M2 Max, `emitted_energy = 5e8 = sim_ray_num`) summed to `1e9` rays
are the `canonical-new` radiometric reference of the scene; the run-to-run
difference is its noise floor. The `PBD` raypath filter admits `12`
equivalent `3-5` images per emitted ray against `6` for `P` (same seed,
`1e7` rays: total Y ratio `2.003`, `1.995-2.03` per `100`-row band), a
bookkeeping on `emitted_energy` that max-normalisation removes, so the
export is compared to the single-path Lumice Integral strip without a
per-pixel factor. The filter is an exact match on the reduced raypath
(`src/core/filter_spec.cpp`, `RaypathOrbit::Contains`: equal length and
`memcmp`), so a `[3,5]` filter never admits `3-1-2-5` at any `max_hits`.
Against this oracle the historical raw is the outlier of the three (section
7, stage 4): both simulators agree at the noise floor, the historical edge
sits `0.23 deg` inside the point-sun minimum deviation, and its tail and
lateral shape belong to none of the one-parameter families probed (solar
altitude, wavelength, pixel scale, zenith width). The `historical-inferred`
solar altitude and the `unknown` axes / refractive index / width above are
therefore known to differ from the canonical scene in at least one respect
that is not recoverable from the surviving bytes.

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
| Stored poses | `49` (not a correctness requirement) |
| Accepted steps | `48` |
| Maximum target residual | at most `5e-16` in the recorded reference environment |
| Fiber length | `0.9643243178593905` rad (one traversal; the `3.857976632802349` recorded until 2026-09-17 was four traversals of this loop under the absolute closure gate retired by `task-continuation-gates-and-fixtures`) |
| Closure gap | approximately `3.6e-15` rad |
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
| Crystal | hexagonal column `h/a = 2.0` (`HexPrism.from_ratio(2.0)`, `a = 1`; equals Lumice `height 1.0` = h / diameter) | `canonical-new`, section 3.3 height note |
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
`2.379121` from `7` admissible clusters, which is procedural evidence for a
single component, not a completeness certificate. (The `4.758247` recorded
until 2026-09-17 was this loop traversed twice under the absolute closure
gate, see `task-continuation-gates-and-fixtures`; every length and integral
below halved accordingly.)

Current expected evidence (Mac reference environment):

| Output | Expected value |
|--------|----------------|
| Terminal state | `closed / closed_loop` |
| Stored poses | `61` (not a correctness requirement) |
| Fiber length | `2.379121` rad |
| c-axis zenith along the loop | about `87.39 .. 90.09 deg` |
| `rho_pose` | `1.1e-4 .. 91.43` (dimensionless, Haar-relative) |
| `entry_measure` | `0.763 .. 1.111` (`length^2`, `a = 1`, `h = 2`; `0.278 .. 0.559` on the `h/a = 1` crystal used until 2026-09-20) |
| `fresnel_transmission` | `0.9354 .. 0.9416` |
| `path_validity` | `1` at every accepted pose |
| Normal Jacobian range | about `0.0822 .. 0.1497` |
| `visibility`, `source_factor`, `pixel_factor`, `other_radiometric` | `unavailable` |
| Line integral `value` (Haar-converted, `partial`) | `6.581373260` with `error_estimate` about `1.4e-4` (`raw_value` about `519.6444`, before the `1/(8 pi^2)` factor) on the `h/a = 2` crystal (2026-09-20, task `defect2-crystal-height-convention`; the pipeline's discovered-seed value is `6.581365570` since the seeds come from the S^2 store, 2026-09-25, `6.581419934` from the retired prescan seed: the same 61-pose loop resampled from another start, `8e-6` apart, `tests/test_strip_pixel.py`). On the `h/a = 1` crystal used until then the same fiber gave `2.364400114` (`raw_value` about `186.6855`), `1.0e-5` below the retired adaptive integrator's `2.364423815 +- 6.0e-9` (rtol `1e-8`); that value stays the frozen alignment reference and its test binds the `h/a = 1` crystal explicitly (`tests/test_resample_quadrature.py::ADAPTIVE_REFERENCE`, `REFERENCE_CRYSTAL`), because the retired integrator cannot re-record on the new crystal. The crystal only enters through `entry_measure`; poses, length, `J_perp` and the node count are unchanged |
| Quadrature method | resampled fixed grid (`task-resample-and-integrate`): C1 cubic Hermite quaternion spline through the accepted poses with the trace's exact tangents, uniform grid of the cumulative-chord parameter, every node retracted onto the fiber by `2` batched bordered Newton iterations, exact `ds/dt` from the implicit function theorem at the retracted node, composite Simpson, error estimate `\|I_N - I_(N+1)/2\|`, node count doubled (`N -> 2N - 1`) until the estimate meets `relative_tolerance`; `epsilon = 1e-6`, `relative_tolerance = 1e-4`, `initial_node_count = 129`, `maximum_node_count = 1025` |
| Quadrature work | `257` grid nodes after one doubling (`129 -> 257`), predictor residual before retraction at most `1.2e-6`, after retraction at most `3.6e-16`, no non-finite node; about `19 ms` per fiber (Mac reference environment, warm), against `2.4 s` for the retired adaptive integrator at rtol `1e-8` (`713` nodes) and `0.92 s` at its production rtol `1e-6` |
| Convergence | the uniform grid converges at order about `2` because `entry_measure` has slope jumps (footprint-clipping vertex events) that fall between grid nodes: deviation from the adaptive reference `7.0e-5 / 3.4e-5 / 1.0e-5 / 3.4e-6` at `65 / 129 / 257 / 513` nodes; the `\|I_N - I_(N+1)/2\|` estimate bounded the actual deviation on every fixture checked |
| Retired adaptive integrator (historical, `2026-09-17`) | adaptive composite Simpson over chord-parametrised edges with one host-side Newton retraction per refinement node: `235` refinements, `1417` nodes, depth `16` at rtol `1e-8`, median per-edge Richardson order `3.99`, global order `1.99` (kinks inside edges `3, 17, 32, 47, 77, 92, 106`); removed because it cost `4-7 s` per lit strip pixel (85 % of the per-pixel budget) |

Quadrature evidence (`tests/test_resample_quadrature.py`): on the analytic
circle a constant weight reproduces `2 pi / (1 + epsilon)` to `1e-9`
(the spline parameter's C1 knots leave a round-off-level kink in `ds/dt`)
and the weight `1 + cos(theta)/2` matches `2 pi / (1 + epsilon)` to `1e-8`
on every grid from `17` nodes; the node count doubles `5 -> 9 -> 17 -> 33`
on `1 / (1.2 + cos theta)` until the estimate meets `1e-7`, and hitting
`maximum_node_count` is reported as `node_count_exhausted`, never as
converged. An event-terminated open arc of the circle integrates to its
extent to `1e-9` and its `endpoint_truncation_estimate` (terminal integrand
times the linear-rate arclength to the event, `continuation.arclength_to_event`)
matches the analytic remainder; a budget-truncated arc gets no estimate
(reported as unbounded). On the canonical pixel fiber the integrals for
`initial_step` `0.03 / 0.04 / 0.08`, `relative_tolerance` `1e-3 / 1e-4 /
1e-5` and both seed orientations agree within the sum of their error
estimates, without comparing sample counts. External alignment: with the
default options the canonical pixel and strip rows `100 / 300 / 500`
(column `126`) stay within `3.6e-5` of the retired adaptive integrator's
rtol `1e-8` values (`129 / 513 / 513` nodes on the strip rows, `13-26 ms`
per fiber). The integrand is only piecewise smooth: `entry_measure` (a
clipped polygon area) changes slope at footprint-clipping vertex events, so
the uniform grid converges at order about `2` and `relative_tolerance =
1e-4` is what the `1e-4` alignment requires (`1e-3` stops at `129` nodes
and misses it on the longer loops). The value is `partial`: one component
from one seed, completeness `unknown`; `visibility` and the radiometric
factors are not in the product.

## 5. Figure Capability Matrix

| Figure or chapter need | Data owner | Current status | Missing capability |
|------------------------|------------|----------------|--------------------|
| ch06 crystal-orientation schematic | Writing-Lab drawing code | Supported | None in Lumice Integral; not a numerical-solver responsibility. |
| ch06 ray-splitting schematic | Writing-Lab drawing code | Supported | None in Lumice Integral. |
| ch06 all-sky Monte Carlo example | Lumice through Writing-Lab validation glue | Supported | Not a Lumice Integral product output. |
| ch06 pose-fiber geometry | Lumice Integral | Supported for one supplied regular seed/component; seeds for one pixel can come from `lumice_integral.discovery`; continuous-sign unit-quaternion adapters exist (`so3.quaternion_from_rotation` / `continuous_quaternion_signs`, used by `resample.fiber_spline`) | Add C-axis longitude/latitude/spin adapters; the prescan-cloud figure still needs recorded spacing/feasibility output from the discovery scan. |
| ch06 solver/Jacobian diagnostics | Lumice Integral data; Writing-Lab presentation | Supported as versioned figure data | A production plotting consumer still belongs in Writing-Lab; an independent prototype consumer has been verified. |
| ch06 named physical-factor curves | Lumice Integral | Supported for `rho_pose`, `entry_measure`, `fresnel_transmission`, `path_validity` on the canonical pixel fiber (section 4.1, figure-data `weight_<name>` arrays); the pointwise final `integrand` curve (product over `J_perp + epsilon`) is exported alongside | `visibility` (finite-face obstruction) is not a separate curve because it is not a separate factor: for the convex prism it is contained in `entry_measure`, whose corridor intersection keeps only entry points whose internal segment reaches every next face's finite polygon (`geometry/entry_measure.py` algorithm step 4, `geometry/feasibility.py` `corridor_intersection`); for a convex body that is the same as the next face being the first one hit, and no incoming or outgoing ray can be obstructed; absolute agreement with Lumice ray tracing to `0.3 %` is the evidence (section 7, stage 4, absolute scale). Uncovered: non-convex crystals and shadowing between crystals, both outside the current scene. `source_factor` / `pixel_factor` are not evaluated in code; their conversion to Lumice's `raw / emitted_energy` is derived and checked (section 7, stage 4). |
| ch06 one-pixel integrand/integral | Lumice Integral | Supported as a `partial` value: resampled fixed-grid line quadrature over the closed canonical fiber with error estimate and grid/retraction evidence (sections 4.1 and 6, `result.quadrature`); single-pixel component discovery is available as a separate primitive (`lumice_integral.discovery`, procedural `completeness` only) | A completeness certificate; pixel averaging (point value only); the radiometric factors above (derived conversion, not evaluated in code). `visibility` is covered for the convex crystal as in the row above. |
| ch06 single-pixel pipeline (`strip_pixel.render_pixel`) | Lumice Integral | Supported (task-pixel-pipeline-v2, section 7 stage 4): one discovery pass per pixel over the scene prescan table plus the warm seeds of any neighbouring pixels, SO(3)-distance dedup before tracing, one production trace per distinct candidate, closed loops *and* open arcs (forward + backward trace stitched, `resample.OpenArc`) integrated by the resampled quadrature with per-end truncation estimates, linear component sum; `0.07 s` per lit pixel and `0.13 s` per lower-band pixel on the M2 Max | A completeness certificate (`completeness` is procedural); no real open arc exists in the current picture, so the arc path is validated on the analytic two-sided fixture only; the missing factors of the one-pixel row. |
| ch06 `251 x 801` direct strip | Lumice Integral | Supported as a `partial` physical rerender: `scripts/render_ch06_strip.py` (`lumice_integral.strip_pixel` / `strip_driver` / `strip_io`, format `lumice-integral.strip/v2`) renders any window of the canonical `251 x 801` grid column-wise, each pixel warmed by the one above, and writes float64/float32 raw in the historical layout plus a per-pixel status layer (`has_arc`, `quadrature_unavailable`, ...) and `provenance.json`; pixel model: pixel-centre point value with `epsilon` regularisation (section 7, stage 4); `--workers` is capped at `4` on macOS | A completeness certificate (the status layer is procedural); the missing factors of the one-pixel row; sub-pixel averaging is implemented but off by default (6-10x cost; `O(10-40 %)` effect in the centre-column caustic band, section 7 stage 4); finite-sun averaging. Known limitation on the delivered full image (section 7, stage 4, full rerender): the vertical decay along the centre column was steeper than the historical raw by `3-4x` between rows `150` and `600` (defect 2); with the `h/a = 2` crystal (2026-09-20) the centre column reproduces the historical plateau on rows `150-400` (`0.89-1.06` relative to row `150`), while the height-independent tail below row `400` (`x4` dark by row `600`) and the horizontal narrowness off the centre column (`0.5-0.7x` at `+-20` columns on rows `300-400`) remain against the historical raw, see `scratchpad/scrum-strip-pipeline-v2/task-strip-rerender-and-compare/artifacts/defect2_findings.md` sections 8-9; the `1e9`-ray Lumice float export (section 3.3, task `lumice-raw-profile-oracle`, 2026-09-20) reproduces both, agreeing with the strip at the Monte Carlo noise floor (column log-RMS `0.03` against a `0.04-0.06` run-to-run floor), so they are differences of the historical raw, not of this renderer. |
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

The current `lumice-integral.figure-data/v3` payload contains:

```text
schema: lumice-integral.figure-data/v3
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
  initial_node_count, maximum_node_count, retraction_iterations
  node_count, refinement_rounds, node_count_exhausted,
  node_count_history ([[N, raw I_N], ...] including the coarsest half grid)
  value, error_estimate (Haar-converted), raw_value, raw_error_estimate,
  haar_to_dvol_g_factor
  residual_before_max, residual_before_median (spline predictor off the fiber),
  residual_after_max, residual_after_median (after the batched retraction),
  non_finite_node_count
  endpoint_truncation_estimate (null on closed loops and non-event arcs),
  endpoint_truncation_note, integrand_array
  (the in-memory result's factor_seconds wall clock is not exported: the
   canonical export stays byte-identical across runs)
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
- `v3` (`task-resample-and-integrate`, 2026-09-17): `result.quadrature`
  describes the resampled fixed-grid quadrature that replaced the adaptive
  integrator. The adaptive method's fields are removed (`maximum_refinement_depth`,
  `refinements`, `maximum_depth_reached`, `convergence_order_estimate`,
  `convergence_order_note`, `raw_convergence_order_levels`,
  `convergence_order_node_count`, `median_edge_convergence_order`,
  `low_order_edges`, `refinement_failures`, `depth_exhausted_edges`) and the
  grid/retraction evidence fields listed above are added; `status`, `method`,
  `value`, `error_estimate`, `raw_*`, `epsilon`, `relative_tolerance`,
  `node_count`, `factor_names` and `integrand_array` keep their names and
  types. The version string changes because fields a `v2` reader may have
  read no longer exist (not an additive change); every array and every other
  metadata field is unchanged. The `integrand` array is still the pointwise
  value at the accepted poses, not at the quadrature grid nodes.

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
   units and normalization; the resampled fixed-grid line quadrature of
   section 4.1 reports method, grid node count and doubling history,
   retraction residuals, value, error estimate and `epsilon` for the
   canonical pixel fiber (`tests/test_resample_quadrature.py`,
   `result.quadrature` in the figure data), aligned to `1e-4` with the
   retired adaptive integrator whose values are frozen there. The value stays `partial` (single
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
   (`scratchpad/.../artifacts/home-wsl-preview-step9/`); the v1 full
   `251 x 801` render was never completed (projected `1-1.5 days`), the
   full image was rendered on the v2 pipeline instead (see the full-rerender
   entry below). Per lit pixel the mean cost was `5.4 s`, of which
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
   - Scene-level prescan table (task-scene-prescan-table, 2026-09-17): cold
     discovery no longer throws `400k` Haar poses per pixel; `strip_driver`
     builds one `prescan.PrescanTable` per scene (`DEFAULT_SAMPLE_COUNT =
     4_000_000` poses, seed `20260916`, the `16 %` that pass the four
     `optics.path_3_5_domain_batch` gates kept with their outgoing
     directions in a kd-tree) before the workers start, and each pixel
     queries `table.candidates(d, 2 deg)`. Density evidence
     (`scripts/prescan_density_survey.py`, M2 Max, `32` pixels: rows
     `40-800` on column 150, the `225/226` pair, the caustic band
     `700-800` on columns 0/50/200/250, the inner-edge slow closers
     `(49,0)`/`(50,9)`; one `16M` table and its exact prefixes `500k` ..
     `8M`, so every rung is a prefix of the same sampling stream):

     | N | valid | pixels whose components changed vs N/2 | clusters (sum) | components (sum) | incomplete (sum) |
     |---|---|---|---|---|---|
     | 500k | 80550 | - | 295 | 32 | 3 |
     | 1M | 161124 | 1 `(50,9)` | 305 | 33 | 1 |
     | 2M | 321004 | 1 `(50,9)` | 309 | 32 | 1 |
     | 4M | 642416 | 0 | 311 | 32 | 2 |
     | 8M | 1283274 | 1 `(50,9)` | 313 | 31 | 2 |
     | 16M | 2565241 | 0 | 313 | 31 | 0 |

     The criterion is the discovered result (component count and arclength
     multiset within `1e-3`), not the raw cluster count: the geodesic
     clustering keeps splitting a denser pool into one more cluster on
     lower-band pixels (`(300,150)` `9 -> 11`, `(400,250)` `10 -> 12 -> 11`)
     all the way to `16M` without finding anything new, so cluster counts
     do not converge and cannot pin `N`. On `31` of the `32` pixels the
     result is identical from `500k` to `16M`; `(50,9)` is the one
     exception and it is a dedup-tolerance effect, not a density effect:
     it has `5` clusters at every rung up to `8M` (`4` at `16M`), but the
     single `0.165 rad` loop there is traced from different entry points
     with arclengths
     `0.16520 / 0.16538 / 0.16560`, i.e. `1.4e-3` apart, just outside
     `dedup_components`' `arclength_rtol = 1e-3`, so it is reported as
     `1-3` components depending on which entries the pool contains (the
     independent cold check below reproduces the pair `0.165424 / 0.165602`
     with a different seed). `4M` is therefore the first rung whose halving
     changes no pixel, and the default; the tolerance question on very
     short loops is left to the discovery contract, not to `N`. The rows
     `700-800` pixels that stayed `incomplete` on the preview now close on
     this branch at every `N` (the continuation-gate change of `2352724`,
     not the table); the only `incomplete` candidates in the survey are
     `1-2` event-terminated seeds (`tir_boundary` / `path_infeasible` at
     their first step) at `(49,0)` and `(50,9)`.
     Independent cold check (`scripts/prescan_cold_check.py`, throw-away
     `16M` table, seed `20260917`): `(150,150)` `1` component `2.379108`
     (survey `2.379109-2.379116` over the ladder), `(700,150)` `1` component
     `5.408495` (survey `5.408495` at every `N`), `(50,9)` the dedup pair
     above. Cost (`benchmarks/benchmark_prescan_table.py`, M2 Max, CPU
     JAX): `4M` builds in `0.62 s`, `candidates` costs `0.14 ms` per pixel
     (`2252` candidates mean, `4443` max on column 150; target `<= 5 ms`),
     the pickled table is `87 MB` and a worker unpickles it in `0.19 s`;
     `16M` is `2.9 s` / `0.71 ms` / `349 MB` / `1.06 s`. Cold discovery per
     pixel is `1.65 s` mean at `4M` on the survey pixels (`1.83 s` at
     `500k`: the pool query is not the cost, the traces are). Evidence:
     `scratchpad/scrum-strip-pipeline-v2/task-scene-prescan-table/artifacts/`
     (`prescan-density/density_survey.{csv,md}`,
     `benchmark_prescan_table_mac.json`, `prescan_cold_check_mac.log`);
     `home-wsl` numbers are not recorded yet. `provenance.json`'s
     `options.discovery` block changed shape with this table: the flat
     `rng_seed`/`prescan_samples` fields were replaced by a nested
     `prescan` object (`sample_count`/`rng_seed`/`cache_path`); no consumer
     in this repository reads the old flat fields.
   - Seed store replaces the prescan table (task `phase1-seeds-from-store`,
     2026-09-25; the prescan density survey above, its cold check and its
     benchmark are history). The table and the $S^2$ event store are one
     presampling (`docs/phase2.md` section 6): discovery now takes the band
     `|D_i - delta| <= 0.2 deg` of the `3-5` store (`s2_store.StoreSeeds`,
     `DEFAULT_SEED_STORE_N = 1e6` Fibonacci points, `160216` kept events,
     `9.8 MB`, `~1 s` in memory), each event posed exactly in the pixel's
     azimuth. Probe against the `4M` table on the same `32` pixels (every
     store `N` in `1e5 .. 1e8` times half-widths `2 / 0.2 / 0.02 deg`, `12`
     configurations): component count and kinds identical on every pixel and
     configuration (`31` components, `0` incomplete); arclengths within
     `1.8e-4` on the loops longer than `0.5 rad` and within `1.5e-3` on the
     `0.19 / 0.17 rad` caustic loops `(49,0)`/`(50,9)` (the start-point
     dependence of a short loop's polyline length recorded above, not a
     density effect); the smallest configuration tried, `N = 1e5` at
     `0.02 deg` (median pool `36`), already finds every component. The dark
     row `40` lies inside the minimum deviation (`21.84 deg`): an empty band
     where the table held `2072` domain-valid samples, `0` components either
     way. Production `N = 1e6` / `0.2 deg`: median pool `3172` (table
     `2451`), `0.08 s` per pixel of discovery (table `0.10 s`). Regression
     against `artifacts/strip-full` (column `126` through the production
     column chain plus the `32` pixels cold, `833` pixels, `775` lit):
     component count, kinds and completeness identical everywhere; values
     median `1.9e-5`, p99 `7.1e-5`, max `1.02e-4` relative; `773` of `775`
     inside `err_new + err_old`, the two others `(63,126)` (`4.3x`) and
     `(58,126)` (`1.26x`) are short caustic loops whose value, integrated
     from `16` start points on the same loop, spreads over `2.8e-3` /
     `4.1e-3` with `strip-full` inside that spread: the resampled
     quadrature depends on where the trace starts and its error estimate
     is not conservative against that (it varies by `1-2` orders along the
     loop; `(150,150)` spreads `1.6e-4` relative), a Phase I property the
     seed source only exposes. The completeness cross-check
     `discovery.check_band_coverage` revisits every band event: on the `32`
     pixels at `N = 1e6` / `0.2 deg`, `0` suspects (`1165` events corrected
     first, the rest within `0.08` of a traced curve as posed; on `(49,0)`
     `541` of `3225`), and the least-covered component carries `1665` band
     events (miss bound `exp(-1665)`); removing the traced loop of a pixel
     turns every event into a suspect. `scripts/store_seed_density_survey.py`
     reproduces the store side of the probe and the cross-check (all `12`
     configurations agree with `N = 1e8` / `0.2 deg` on every pixel at
     arclength rtol `2e-3`). Evidence:
     `scratchpad/scrum-phase2-contour-quadrature/task-phase1-seeds-from-store/probe/`
     (`probe.{csv,md,log}`, `regress.{json,log}`, `start_point_spread.log`,
     `store-survey/store_seed_density_survey.{csv,md}`).
   - Single-pixel pipeline v2 (task-pixel-pipeline-v2, 2026-09-17): the
     hot-start chain, the small discovery budget with its production
     retrace, the arclength-fingerprint dedup, the arclength-jump gate, the
     floor-lock early exit and the periodic cold check above are all retired
     (their evidence stays here as history).  `render_pixel` now runs one
     `discovery.discover_components` pass per pixel: the prescan pool plus
     the integrated components of the pixel above as warm Gauss-Newton
     starts, greedy geodesic clustering, and *before* any trace a fold of
     every corrected candidate that lies within `distance_threshold = 0.08`
     (the continuation's `closure_distance`) of an already traced curve, so
     a loop reached by `7-13` candidates is traced once (`(700,150)`: `12`
     admissible, `1` trace, `11` `dedup_merged`); each distinct candidate is
     traced once with the production options, a closed trace is a `closed`
     component and a trace ended by a named event is traced backward from
     the same seed and stitched into an `arc` component
     (`docs/phase1-math-contract.md` sections 7-8).  One continuation
     change came out of it: a Newton iterate outside the corrector trust
     region that lands in an invalid domain is a rejected trial, not an
     event (`(50,9)`: the `0.17` loop's first `0.04` predictor sent the
     corrector `1.18 rad` into a TIR region and was reported as
     `tir_boundary`; the `1-2` first-step event candidates of `(49,0)` /
     `(50,9)` in the density survey above were this).  Open-arc census
     (Step 0, current picture, every 10th row and column, `2106` pixels,
     `4` workers, `671 s`): `1954` lit, all closed, no multi-pose
     event-terminated candidate, so the arc path is validated on the
     analytic two-sided circle only (`tests/test_discovery.py`,
     `tests/test_resample_quadrature.py`).  Baselines (on the `h/a = 1`
     crystal of that date): canonical `2.364412980` unchanged, `(700,150)`
     `5.408495`, `(780,150)` `5.635867`, `(60,126)` `0.466397` / `19.038`,
     `(50,9)` one loop `0.165603` (the dedup pair of the survey is folded).
     Since 2026-09-20 (`h/a = 2`, task `defect2-crystal-height-convention`)
     the lengths and counts are unchanged and the values are canonical
     `6.581419934`, `(60,126)` `39.366`.  Cost on the M2 Max (warm process, medians,
     `probe_step7_timing.py`: the same pixel rerun with every kernel shape
     already cached, i.e. a hot-cache lower bound, not the strip's cost;
     see the per-pixel cost item below for the column ruler):
     canonical `0.067 s` (trace `0.040`, quadrature `0.021`), lit warm
     `0.075 s`, lower band `0.12-0.13 s`, dark `0.005 s`, against `5.8 s`
     for the canonical pixel before; the trace is now ~60 % of a lit pixel.
     Mac smoke `rows 140:160 x columns 145:155`, `4` workers: `200` pixels
     in `21.6 s` wall, `0.355 s` per pixel including JIT warm-up, all
     `complete`.  Format `lumice-integral.strip/v2`: status bits `rendered`
     / `unknown_completeness` / `has_component` / `has_arc` /
     `quadrature_unavailable` / `node_count_exhausted`; `pixels.csv` gains
     `component_kinds`, `component_end_reasons` and both truncation
     columns; checkpoints carry the format tag and v1 checkpoints are
     recomputed.  Evidence:
     `scratchpad/scrum-strip-pipeline-v2/task-pixel-pipeline-v2/`
     (`probe_step0.py`, `probe_step0_scan.json`, `probe_step7_timing.py`).
   - Full rerender and log-domain comparison (task-strip-rerender-and-compare,
     2026-09-17): the whole `251 x 801` grid on the v2 pipeline, `home-wsl`
     (`32` cores, `30` workers, `JAX_PLATFORMS=cpu`, prescan table `4M`
     samples built in `1.1 s`), `6699 s` wall clock (`1.86 h`; the task's
     `30 min` target was missed by `3.7x`, the `2 h` hard stop was not
     reached), `618-894 s` per column (median `750 s`), i.e. `0.8-1.1 s`
     per pixel per worker against `0.07-0.13 s` measured single-process on
     the M2 Max: the per-pixel cost under `30` workers was about `10x` the
     warm single-process figure (load average `48` on `32` cores, `56 GB`
     resident); diagnosed and reduced by the per-pixel cost item below.
     Result: `201051` pixels rendered, all `complete`, `0`
     `unknown_completeness`, `0` `has_arc`, `187406` lit, `845`
     `node_count_exhausted`; the lower quarter that was `43-100 %` unknown
     on the v1 preview is fully resolved (defect 3 closed on the full
     image). Comparison (`scripts/compare_strip_v2.py`, every metric
     sensitive to multiplicative bias, every panel on a log scale; the
     historical raw and the strip have unrelated units so ratios are
     reported relative to the whole-image median):
     - Row-band ratio `median(ours / historical)`, centre columns
       `101-151`, per `50` rows, relative to the whole-image median: `3.29`
       (rows `50-100`), `3.04`, `2.77`, `2.51`, `2.20` (`250-300`), `1.81`,
       `1.44`, `1.14` (`400-450`), `0.93`, `0.81`, `0.75`, `0.71`
       (`600-650`), `0.63`, `0.52`, `0.39` (`750-800`): a monotone `8.4x`
       slide from the inner edge to the bottom, no step (the `x2` plateau
       of defect 1 on the v1 preview is gone: the v1 baseline on the same
       columns reads `6.70 / 3.85 / 3.54 / 2.83` for rows `50-250` against
       `3.29 / 3.04 / 2.77 / 2.51` here). Along column `126` the
       max-normalised ratio ours/historical falls from `1.0` at rows
       `100-150` to `0.75` (`300-350`), `0.44` (`400-450`), `0.31`
       (`450-500`), `0.23` (`600-650`), `0.13` (`750-800`): the historical
       raw holds a plateau at rows `250-450` (a slight rise `0.174 -> 0.180`
       at `350-400`) that ours does not have; this is defect 2 quantified.
     - Log-domain profile RMS (`log10` of max-normalised profiles, lit band
       = both above `1e-3`): column `126` versus historical `0.443`
       (`697` points; `0.486` over all `744` positive points), row `150`
       `0.487`, row `300` `0.688`, row `450` `0.124`. Against the Lumice
       grey PNG the same numbers are `0.95 / 0.40 / 0.57 / 0.64`, and
       Lumice-versus-historical is `0.56 / 0.11 / 0.18 / 0.53`, so on rows
       `150-300` the historical raw and the Lumice remake agree with each
       other better than either agrees with ours (ours is narrower: the
       width above `1e-3` of the max-normalised row is `119` columns
       against `143` at row `150` and `121` against `157` at row `300`,
       while at row `450` they agree, `127` against `129`).
     - Inner-edge row (first row of the max-normalised column profile above
       `1e-3 / 1e-2 / 1e-1`): historical `47 / 47 / 48`, ours `57 / 57 /
       58`, Lumice `60 / 60 / 60` at columns `100`, `126` and `150`: the
       `10`-row offset of the v1 preview is confirmed on the full image and
       the Lumice remake sits `3` rows further in, so the offset is on the
       historical side (the `historical-inferred` sun elevation of section
       3.3 remains the candidate), reported, not corrected.
     - Spearman (auxiliary only): `0.985` on the `187406` pixels lit in
       both, `0.970` on all `201051`, `0.947` against the Lumice PNG.
     Defect 2 localisation (`scripts/probe_defect2_factors.py`, column
     `126`, rows `150 / 300 / 450 / 600`, four factors and `J_perp`
     exported along each fibre): `rho_pose` agrees pointwise with an
     independent quadrature of the Lumice `gauss` zenith definition (ratio
     `1.000000`; the `0.5 deg` width is the same definition, unit and angle
     convention as the Lumice remake's `crystal[0].axis.zenith`), and
     `entry_measure` agrees with an independent ray-cast Monte Carlo
     (median ratio `0.996-1.007`), so neither is a bookkeeping error; the
     `150 -> 450` decay is carried by `entry_measure` (`/3.7`) and `J_perp`
     (`x4.0` in the denominator), and no zenith width (`0.25-2 deg`) moves
     the rows `300-450` by more than `1.5x`. Not converged to a single
     cause; the remaining suspects are an independent check of `J_perp` on
     rows `300-600` and a non-tone-mapped Lumice profile that separates
     "what the historical raw contains" from the single-path physics
     (`defect2_findings.md` sections 5-6, both on the backlog). Evidence:
     `scratchpad/scrum-strip-pipeline-v2/task-strip-rerender-and-compare/artifacts/`
     (`compare_metrics.json`, `strip_images_v2.png`, `strip_profiles_v2.png`,
     `defect2_probe.json`, `defect2_findings.md`); the render itself is
     `artifacts/strip-full-h1/` since 2026-09-20 (git-ignored,
     `provenance.json` format `lumice-integral.strip/v2`,
     `columns_resumed = 0`; `artifacts/strip-full/` now holds the
     `h/a = 2` render, see the crystal height item below).
   - Morphology of the v1 every-ninth-column preview against the historical
     raw (native orientation, Spearman rank correlation; superseded by the
     full-rerender comparison above): `0.990` on the `16876` pixels lit in both, `0.970` on
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
   - Per-pixel cost (task-pixel-cost-shape-stable-kernels, 2026-09-20).
     Ruler: `benchmarks/benchmark_column_steady_state.py` renders one
     column single-process through `strip_driver.render_column` (warm
     seeds from the pixel above, production options), times every pixel
     and counts XLA compilations (`jax.log_compiles`); steady state =
     median after the first `10` pixels. Before: column `126` rows
     `100-160` on the M2 Max, `1065` compilations for `60` pixels, steady
     mean `0.273 s` per pixel (median `0.137`), pixels alternating between
     `0.13 s` / `3` compilations and `0.55 s` / `35` compilations, i.e. the
     hot single-pixel `0.067 s` above is a lower bound the strip did not
     reach. Cause: `jax.jit(vmap(rotation_distance))` in `discovery`
     recompiled for every distinct candidate-pool size and curve length,
     and the eager `jax.vmap` quaternion batches in `resample.py`
     recompiled ~`30` primitives for every new accepted-pose count. Fix:
     `so3.rotation_distances` (numpy batch of the same formula, ulp-locked
     to `rotation_distance` by `tests/test_so3.py`) for pool clustering and
     curve dedup; the quaternion batches compiled once per power-of-two
     bucket (padding is elementwise-isolated, bit-identical to the unpadded
     kernel, `tests/test_resample.py`). After: `101` compilations for the
     same `60` pixels (`5` past the first pixel), steady mean `0.066 s`
     (median `0.064`); the full column `0-801`: `85.7 s`, steady median
     `0.109 s` per pixel, `143` compilations with only `6` pixels compiling
     (the first, the first lit one, and new resample grid sizes); on
     `home-wsl` single-process `138 s`, median `0.181 s` per pixel. Column
     `126` against `artifacts/strip-full`: values within `7.3e-14`
     relative, status bits and component counts identical. Worker probe on
     `home-wsl` (Ryzen 9 9950X, `16` cores / `32` threads, `94 GB`;
     `JAX_PLATFORMS=cpu`, `--xla_cpu_multi_thread_eigen=false`,
     `OMP_NUM_THREADS=1`, no persistent compile cache, one output directory
     per worker count): per-column seconds (min/median/max) `138` at `1`
     worker, `168/169/172` at `4`, `204-208` at `8`, `247/256/263` at `16`,
     `213/228/249` at `30`, `217/239/250` at `32`; total throughput
     `5.8 / 19 / 31 / 48 / 100 / 95` px/s; load average `1.06-1.5x` the
     worker count in steady state (`1.7x` for the first minute while every
     worker compiles its first pixel), `530-560 MB` resident per worker,
     flat. A worker is one busy main thread plus three JAX execution
     helpers (`1.06` cores in total); `llvm-worker` compile threads are
     idle after the first pixel, so compile-thread oversubscription is gone
     and neither `jax_compilation_cache_dir` nor a compile-thread cap is
     needed. The per-worker rate falls from `5.8` to `3.1-3.5` px/s with
     SMT sharing beyond `16` workers; throughput is flat between `30` and
     `32`, so `30` workers (the logical CPU count minus two) is the
     recommendation. Full `251 x 801` rerender (`30` workers, 14:15-14:49
     UTC+8): `2085 s` wall clock (`35 min`, `3.2x` faster than `6699 s`),
     `204-257 s` per column (median `230`), `201051` pixels, `187406` lit,
     `0` `unknown_completeness`, `0` `has_arc`; against
     `artifacts/strip-full` every status byte and component count is
     identical, the `float32` layer is byte-identical, `strip_float64.bin`
     differs by at most `8.1e-13` relative (`24476` pixels bit-identical;
     the rest at the last-bit level, consistent with XLA fusion in the now-jitted quaternion batches).
     The `<= 15 min` target is missed by `2.3x`: the strip now runs at the
     hot-cache per-pixel floor, so the remaining wall clock is trace and
     quadrature work times `251 x 801 / (16 physical cores x SMT)`.
     Evidence: `scratchpad/task-pixel-cost-shape-stable-kernels/evidence/`
     (`mac_col126_*.json`, `wsl_col126_rows0-801.json`, `probes/*/`,
     `full-w30/`); the render is `artifacts/strip-full-v3-h1/` (renamed
     from `strip-full-v3` on 2026-09-20, `h/a = 1`).
   - Mac versus `home-wsl` cross-check on the 20 pixels both rendered
     (column 153, rows 140-159): maximum relative difference `3.1e-7` (below
     the `1e-6` quadrature tolerance), component counts identical, status
     bits identical except the window-relative `cold_discovery` spot-check phase.
   - Crystal height convention and the independent `J_perp` check
     (task-defect2-crystal-height-convention, 2026-09-20). Defect 2, first
     half: the canonical crystal was `h/a = 1` while the Lumice remake's
     `height 1.0` is `h / diameter = 2` in that ratio (section 3.3 height
     note); the canonical scene now uses `h/a = 2`. The crystal enters the
     integrand only through `entry_measure`, so every fiber, length, pose
     count, `J_perp` and discovery count baseline is unchanged
     (`tests/test_discovery.py`, `tests/test_reference_core_conformance.py`
     pass untouched) and only the pixel values move (canonical
     `2.364412980 -> 6.581419934`, `(60,126)` `19.038 -> 39.366`); the
     baselines recorded on the old crystal by tools that cannot re-record
     (the retired adaptive integrator, the independent `scipy.integrate.quad`
     zenith-width probe) keep that crystal explicitly in their tests. The
     `h/a = 1` full render is kept as `artifacts/strip-full-h1/` (and the
     bit-identical `strip-full-v3-h1/`). `J_perp` independent check
     (`scratchpad/task-defect2-crystal-height-convention/artifacts/probe_jperp_fd.py`,
     column `126`, rows `150-650` step `50`, `1140` accepted poses of the
     `11` production traces): the `(2, 3)` target-chart residual Jacobian
     that `continuation._local_residual_jacobian_kernel` differentiates
     with `jax.jacfwd` (`delta -> chart_basis.T @ (direction(R exp(delta))
     - chart_direction)`, right-trivialized coordinates) was rebuilt by
     central differences of the same map and its two singular values
     multiplied. Step sweep `1e-4 / 1e-5 / 1e-6 / 1e-7`: the worst relative
     difference per row scales as `h^2` from `1e-4` to `1e-5`
     (`7.5e-8 .. 5.5e-6` to `7.6e-10 .. 5.5e-8`), plateaus at `1e-6`
     (`2.8e-10 .. 7.5e-10`) and rises again at `1e-7` (round-off,
     `1.9e-9 .. 5.9e-9`); at the plateau step the worst relative
     difference over all `1140` poses is `7.5e-10` against the `1e-4`
     acceptance (`jperp_fd.json`). `J_perp` along column `126` grows
     from `0.082-0.149` at row `150` to `0.477-2.09` at row `650`; the AD
     value is confirmed, so the `x4` denominator growth of the defect 2
     localisation is real geometry, not a differentiation error (this
     check is crystal-independent). Full rerender on the `h/a = 2`
     crystal (`home-wsl`, `30` workers, `2030 s` wall clock, `201051`
     pixels all `complete`, `0` `has_arc`, `187406` lit, `15`
     `node_count_exhausted` against `845` before): along column `126`
     the ratio ours/historical relative to row `150` (the section 8
     probe's definition, `column_plateau.py`) is `0.886-1.058` on rows
     `150-400`, `248` of `251` rows inside `0.9-1.1` and the three
     outside are rows `398-400` (`0.896 / 0.891 / 0.886`) where the tail
     decay begins (`0.777` by row `420`); with `h/a = 1` the same rows
     read `0.549-1.0` (`39 %` inside). Per-50-row max-normalised ratios
     (`compare_strip_v2.py::column_decay_ratio`): `1.03 / 1.02 / 1.06 /
     1.10 / 1.04` on rows `150-400` (`0.93 / 0.88 / 0.83 / 0.75 / 0.61`
     before), then `0.78 / 0.49 / 0.33 / 0.28 / 0.26 / 0.22` on rows
     `400-700` (`0.44 / 0.31 / 0.26 / 0.24 / 0.23 / 0.20` before): the
     plateau is reproduced, the height-independent tail is not moved
     (second half of defect 2, `lumice-raw-profile-oracle`). The plateau
     is a centre-column result: `+-5` columns `93-96 %` of rows `150-400`
     inside `0.9-1.1`, `+-10` columns `68-82 %`, `+-20` columns `37-39 %`
     with rows `350-400` at `0.52-0.57` (the old centre-column
     signature); the horizontal profile at row `150` now matches the
     historical raw to `+-7 %` over `+-20` columns (`0.69-0.87` before)
     while row `300` is still `0.6-0.8x` at `+-20` columns (`0.37-0.61`
     before), and the lit width above `1e-3` is `121 / 129 / 139` columns
     at rows `150 / 300 / 450` against the historical `143 / 157 / 129`
     (`119 / 121 / 127` before). Log-RMS along column `126` versus
     historical `0.443 -> 0.389`, row `150` `0.487 -> 0.447`, row `300`
     `0.688 -> 0.542`, row `450` `0.124 -> 0.224` (the turnover row);
     whole-image Spearman versus historical `0.985 -> 0.949` (lit in
     both), versus the Lumice PNG `0.947 -> 0.967`; inner-edge rows
     unchanged (`47 / 57 / 60`). The row-band ratio relative to the
     whole-image median now reads `1.79` (rows `50-100`) `.. 1.28` (rows
     `350-400`) `.. 0.21` (rows `750-800`), a `8.7x` slide against `8.4x`
     before, because the plateau rows rose while the tail did not.
     Evidence: `scratchpad/task-defect2-crystal-height-convention/artifacts/`
     (`column_plateau.{py,json,log}`, `compare_metrics.json`,
     `compare_strip_v2.log`, `strip_images_v2.png`,
     `strip_profiles_v2.png`, `jperp_fd.json`, `figure-data/`); the
     render is `artifacts/strip-full/` (git-ignored, `columns_resumed =
     0`, fresh directory, so `--resume` could not have mixed `h/a = 1`
     checkpoints in: the driver's options fingerprint does not cover the
     crystal).
   - Lumice float oracle and the second half of defect 2
     (task-lumice-raw-profile-oracle, 2026-09-20). The `h/a = 2` strip was
     compared with the historical raw and the `1e9`-ray Lumice float export
     of section 3.3 (`scripts/compare_strip_v2.py --lumice-float
     --lumice-float-run2`, every profile max-normalised, log10, RMS on the
     lit band above `1e-3`). Vertical profiles, columns `106 / 126 / 146`:
     ours versus Lumice float `0.034 / 0.029 / 0.035`, Lumice float versus
     historical `0.405 / 0.391 / 0.397`, ours versus historical `0.407 /
     0.389 / 0.404`; the two independent `5e8` runs differ from each other
     by `0.060 / 0.043 / 0.062` on the same measure (the merged profile's
     own noise is about half of that), so Lumice Integral and Lumice agree
     at the Monte Carlo noise floor and the historical raw is the outlier.
     Horizontal profiles, rows `150 / 300 / 450 / 600`: ours versus Lumice
     float `0.085 / 0.071 / 0.074 / 0.086` against a run-to-run `0.055 /
     0.090 / 0.142 / 0.096`; Lumice float versus historical `0.486 / 0.569
     / 0.207 / 0.612`. Per-50-row max-normalised ratio along column `126`,
     Lumice float / historical: `0.99` (rows `350-400`), `0.75`, `0.48`,
     `0.33`, `0.29`, `0.26` (rows `600-650`) .. `0.14` (rows `750-800`),
     band for band the ours / historical signature of the previous bullet,
     while ours / Lumice float stays within `0.93-1.04` on rows `100-750`
     (`0.90-1.04` over the column) and the whole-image centre-column band
     ratio ours / Lumice float relative to its median stays within
     `0.94-1.02` on every band. The horizontal narrowness is on the Lumice
     side too: lit width above `1e-2` at rows `150 / 300 / 450 / 600` is
     `125 / 135 / 101 / 233` columns for the historical raw, `107 / 107 /
     113 / 251` for ours and `105 / 105 / 113 / 251` for the Lumice float
     (the historical is wider near the edge and *narrower* at rows `450 /
     600`, where both simulators are nearly flat across the strip). Inner
     edge, three columns and three thresholds: `47 / 57 / 60` (historical /
     ours / Lumice float) everywhere; through the canonical camera the rows
     are `21.61 / 21.85 / 21.92 deg` from the sun against the point-sun
     minimum deviation `21.84 deg` (`n = 1.31`), so ours starts half a
     pixel from the caustic, the Lumice float three rows later (threshold
     on a Monte Carlo caustic peak) and the historical `0.23 deg` inside
     it, where no canonical `3-5` ray exists. A `+0.3 deg` solar altitude
     puts the Lumice edge at row `47` (`5e7`-ray probe), and so does `700
     nm` (`n` about `1.307`); a `5.79 deg` field of view would as well;
     none of the three moves the tail (`0.87 / 0.73 / 0.49 / 0.35 / 0.28`
     and `0.85 / 0.63 / 0.40 / 0.28 / 0.22` on rows `350-600` against the
     `0.97 / 0.75 / 0.51 / 0.32 / 0.29` baseline), and a `13`-row shift
     changes the tail ratio by `x0.91` where the deficit is `x4`, so the
     edge offset and the tail are not one root cause. A wider zenith
     distribution lifts the tail (`std 5 deg`: rows `600-700` back to
     `1.0`) but widens rows `150-300` to the full `251` columns at `2 deg`
     already (historical `125-135`), so the historical shape is not in
     that family either. The `3-1-2-5` candidate of task
     `path-class-rendering-unit` is empty on the `h/a = 2` crystal: a
     `[3,1,2,5]` Lumice filter gives `0` pixels at `height 1.0` for
     `max_hits 4 / 5 / 8`, `15` pixels at `0.75` and `12495` pixels (rows
     `263-796`) at `0.5`, the `h/a = 1` crystal its fixtures were
     recorded on. Verdict: the tail deficit and the lateral narrowness are
     differences of the historical raw against two independent solvers,
     not a Lumice Integral defect; no fix task is opened. Evidence:
     `scratchpad/task-lumice-raw-profile-oracle/artifacts/`
     (`compare/compare_metrics.json`, `compare/strip_profiles_lumice_float.png`,
     `edge-offset-analysis.md`, `lumice-raw/run1|run2/` with `config.json`,
     `img_01.{npy,json}`, `run.log`, and the `probe-*/` runs).
   - Absolute radiometric scale, current Lumice (task
     `lumice-area-weighting-recheck`, 2026-09-24; Ice Halo `2056f699`,
     `Lumice 4.6.1-dev`). Ice Halo #597 made Lumice weigh each ray by the
     crystal's projected area. At entry (`InitRay_p_fid`,
     `src/core/simulator.cpp:213`) the ray's weight is multiplied by
     `lm_pcg::entry_weight = min(1, A_tot / (S/2))`
     (`src/core/shared/pcg_shared.h:627`), where `S` is the crystal's total
     surface area; `S/2` bounds the projected area of every convex body, so
     the `min` never binds. The entry sub-triangle is then still drawn with
     probability `A_face / A_tot`, and the pose still from `rho` alone. No
     ray is discarded: `6fc48bb4` replaced the per-ray accept/reject of
     597.1-597.3 by this expected weight on every backend. An emitted ray
     therefore lands on `3-5` with expected weight `A_P / (S/2)`: `A_tot`
     cancels, and `A_P` is exactly the `entry_measure` inside `V`.
     `emitted_energy` keeps its meaning: `emitted_weight x
     emitted_ray_equivalent` (`simulator.cpp:1737, 1760, 2032`; one
     scattering entry under `proportional` allocation gives `ray_num`), and
     the weight a ray loses at entry still counts toward it
     (`doc/configuration.md`, the note after the `ray_allocation` table); the
     runs below record `emitted_energy = sim_ray_num = 5e8`. Hence
     `raw[p] / emitted_energy = K_p V(w_p)` with
     `K_p = N_sym * ybar(550) * Omega_p / (S/2)`, `S = 6ah + 3 sqrt(3) a^2 =
     17.196 a^2` at `h/a = 2`. `K_p` depends on the pixel through
     `Omega_p` alone (`3.0 %` over the bright band). The ratio
     `A_tot / (S/2)` is scale free, so Lumice's equal-surface-area crystal
     size convention (`docs/conventions.md` row 17) only matters between
     crystals of different shapes. Check (`scripts/probe_absolute_scale.py`,
     nothing fitted; the task 8 `config.json` unchanged under the new binary,
     seeds `7 / 11` at `5e8` rays each, merged): on the bright band (rows
     `60` to `440-500`, caustic row `60` excluded, `38-44` pixels per column) the
     measured over predicted `raw / E` is `0.998 +- 0.004 / 0.998 +- 0.003 /
     0.998 +- 0.005` (median +- standard error) on columns `106 / 126 / 146`
     at Lumice's `n(550) = 1.3110129`, and `1.013 / 1.027 / 1.009` at the
     canonical `n = 1.31` (the index mismatch of the next bullet). The area
     each pixel implies, `N_sym ybar Omega_p V / (raw / E)`, has median
     `8.62 a^2` on all three columns (`S/2 = 8.598 a^2`) and a per-pixel
     relative spread of `1.6-3.0 %` against a merged Monte Carlo noise of
     `2.0-2.5 %`: the pixel-dependent `A_eff` of the old Lumice (`3.8-4.5
     a^2`, `~15 %`) is gone, and `K_p` is the pixel-independent constant
     `N_sym ybar / (S/2)` times `Omega_p`. The caustic row `60` measures
     `0.83 / 0.62 / 0.94` of its point prediction (point versus pixel area,
     as before); the `3 x 3` sub-pixel means at rows `150 / 300 / 450` stay
     within `0.02 %` of the point values. The shape is unchanged: ours
     against the new export has column log-RMS `0.036 / 0.048 / 0.037`
     against a seed-to-seed floor of `0.060 / 0.045 / 0.062`, and the
     `compare_strip_v2.py --absolute-scale-probe` residual on the delivered
     strip is `+1.3 / +2.7 / +0.7 %` (canonical `n`).
     Pose families (`scripts/compare_lumice_family.py`: the band-sum
     renderer's `[3,5]` PBD class at `N = 1e8`, whose value already sums the
     `12` members, against the Lumice `PBD` filter with the same camera; so
     `K_p = ybar Omega_p / (S/2)`). Plate (zenith `gauss(0, 1 deg)`, camera
     `321 x 161`, linear `32 deg`, view elevation `15 deg`, both parhelia):
     at matched `n` the total flux ratio is `0.9999` (seeds `0.9999 /
     0.9999`), the bright pixels (`168`, above `10 %` of the maximum) have
     median ratio `0.995` and relative spread `0.94 %` against `0.78 %`
     expected (Lumice seeds `0.11 %`, band sum `1 / sqrt(K_eff)`), and the
     max-normalised row and column through the peak differ by RMS `0.003 /
     0.004`. At the canonical `n = 1.31` the total is `0.9965` but the
     bright spread is `18 %`, all of it on the parhelion's inner edge: the
     `1e-3` index step moves that caustic by about half a pixel, which a
     `30 x 32` window re-rendered at `n = 1.3110129` confirmed before the
     full image (window `0.9868 -> 0.9999`). Parry (zenith `gauss(90, 1 deg)`,
     roll `gauss(0, 1 deg)`, camera `401 x 401`, linear `100 deg`, view
     elevation `15 deg`; the class images above and below the sun): at
     matched `n` the total flux ratio is `1.0001` (seeds `1.0002 /
     1.0001`) and the two halves of the image, which hold different halo
     features and so different poses, give `1.0002 / 1.0001` (top / bottom,
     `27 / 73 %` of the flux). The `40` bright pixels form one sharp spot:
     median ratio `0.994`, spread `2.1 %` against `0.6 %` expected, falling
     to `1.4 / 1.2 / 0.4 %` when both images are binned `2 x 2 / 3 x 3 / 4 x
     4`, i.e. the band average against the pixel-area average, not a pose
     factor. At the canonical `n` the total is `0.998` and the halves
     `0.990 / 1.001`.
     No `A_eff` has to be folded into a family comparison any more; the
     backlog item that asked for it is void. Evidence:
     `scratchpad/task-lumice-area-weighting-recheck/artifacts/`
     (`lumice-new/run1|run2/`, `absolute-scale/`, `absolute-scale-n-lumice/`,
     `compare/compare_metrics.json`, `lumice-plate|parry/run1|run2/`,
     `li-plate*/`, `li-parry*/`, `family/*.json`; the matched-`n` band sums
     come from `render_band_sum_n_lumice.py` there, a copy of
     `scripts/render_band_sum.py` with only the index changed, whose
     `provenance.json` still records `1.31`).
   - Absolute radiometric scale, Lumice before `6fc48bb4` (history; task
     `phase1-closeout-absolute-scale`, 2026-09-23). This bullet describes the
     Lumice of that date; its conversion does not apply to the current
     binary (previous bullet). The strip value is `V(w) = (1 / 8 pi^2) int rho A_P T /
     (J_perp + epsilon) dH^1`, the Haar expectation of `A_P T delta(w -
     Phi(R))`: power per steradian sent along `3-5` by one crystal of random
     pose per unit incident irradiance, in crystal `length^2 / sr` (hexagon
     edge `a = 1`, `h = 2`). Lumice (Ice Halo `src/core/simulator.cpp`, read
     as evidence, never linked) draws every ray's pose from `rho` alone
     (`InitRay_rot`), gives it emission weight `1` with no pose factor
     (`InitRay_d_w_previdx`), and only then picks the entry point with
     probability proportional to the projected area of each front-facing
     sub-triangle (`InitRay_p_fid`). An emitted ray therefore enters `3-5`
     with probability `A_P / A_tot`, where `A_tot` is the projected
     silhouette, and the pose is *not* weighted by `A_tot`. Fresnel is the
     same per-interface s/p average on both sides
     (`src/core/shared/optics_shared.h` `GetReflectRatio`); the Y channel
     carries `ybar(550) = 0.99495` (`kCmfY`); a linear-lens pixel subtends
     `Omega_p = cos^3(theta_p) * axis_solid_angle` (`1.7438e-7 sr` on axis,
     the sidecar value, equal to `1 / scale^2` of `camera.linear_scale`).
     Under the canonical density (uniform azimuth and roll, zenith symmetric
     about `90 deg`) all `12` `PBD` images of `[3,5]` have the same sky image
     in expectation. Hence
     `raw[p] / emitted_energy = K_p V(w_p)` with
     `K_p = 12 * ybar(550) * Omega_p / A_eff(w_p)`, where `A_eff = V / V~` is
     the fiber-weighted harmonic mean of `A_tot`, and `V~` is the same
     integral with `rho / A_tot` in place of `rho`. `K_p` is not one
     constant. On the bright band `A_eff` runs over `3.8-4.5 a^2`, and `K_p`
     over `4.58-5.16e-7` (column medians `4.62 / 4.92 / 4.62e-7` on columns
     `106 / 126 / 146`). This `~15 %` variation is a modelling difference
     between the two simulators: Lumice Integral weights poses by their
     physical cross-section, Lumice gives every pose the same energy. It sits
     inside the `0.03` log-RMS of the max-normalised comparison above.
     Check (`scripts/probe_absolute_scale.py`, nothing fitted): one discovery
     and two quadratures (`rho` and `rho / A_tot`) per probed pixel, every
     tenth row of columns `106 / 126 / 146`, against the merged `1e9`-ray
     export. On the bright band (Lumice above `10 %` of the column maximum,
     rows `60-500`, `39-45` pixels per column) the measured over predicted
     `raw / E` is `1.013 +- 0.011 / 1.026 +- 0.008 / 1.009 +- 0.011` (median
     +- standard error) at the canonical `n = 1.31`. At Lumice's own
     `n(550) = 1.3110129` (`IceRefractiveIndex` Sellmeier,
     `src/core/optics.cpp`) it is `0.999 +- 0.005 / 0.997 +- 0.010 / 0.998 +-
     0.005`. On column `126`, `V` rises by `0.17-0.37 %` per `1e-4` of `n`
     (rows `400` to `150`), so the canonical-`n` residual is the index
     mismatch. Two terms can explain what remains. Monte Carlo noise: the
     merged run holds `1.3-1.6 %` per bright pixel, and each column median has
     a standard error of `0.5-1 %`. Point versus pixel area: the `3 x 3`
     sub-pixel mean of `V~` at rows `150 / 300 / 450` of each column differs
     from the point value by at most `0.02 %` off the caustic, while the
     caustic rows keep the `O(10-40 %)` of the pixel-model probe above (row
     `60` of column `106` measures `0.58` of its point prediction). The
     sub-pixel check was measured here because the earlier probe covered only
     the caustic. No integer or `pi` factor remains. On the delivered strip
     (`scripts/compare_strip_v2.py --lumice-float ... --absolute-scale-probe`,
     `absolute_scale` block) the residual is `+1.3 / +3.0 / +0.9 %` (standard
     error `~1 %`, canonical `n`), with the strip values within `1e-4` of the
     probe's re-integration. Evidence:
     `scratchpad/task-phase1-closeout-absolute-scale/artifacts/`
     (`absolute-scale/`, `absolute-scale-n-lumice/`,
     `compare/compare_metrics.json`).
   Radiometric normalisation is not aligned in the historical and PNG
   comparisons (the strip is the partial integrand, the historical raw has
   unknown units, the Lumice PNG is tone-mapped 8-bit): the v1 preview
   comparison was morphology only (rank correlation, profiles, side
   agreement), the v2 full-image comparison reports shape-sensitive ratios
   relative to a whole-image median and log-domain profile differences;
   the Lumice float comparison is radiometric up to one scale factor, and
   that factor is derived and checked in absolute terms by the
   absolute-scale bullets. Scripts and JSON
   of the v1 preview:
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
