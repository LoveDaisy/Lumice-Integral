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

## 5. Figure Capability Matrix

| Figure or chapter need | Data owner | Current status | Missing capability |
|------------------------|------------|----------------|--------------------|
| ch06 crystal-orientation schematic | Writing-Lab drawing code | Supported | None in Lumice Integral; not a numerical-solver responsibility. |
| ch06 ray-splitting schematic | Writing-Lab drawing code | Supported | None in Lumice Integral. |
| ch06 all-sky Monte Carlo example | Lumice through Writing-Lab validation glue | Supported | Not a Lumice Integral product output. |
| ch06 pose-fiber geometry | Lumice Integral | Supported for one supplied regular seed/component | Add continuous-sign unit-quaternion and C-axis longitude/latitude/spin adapters; prescan points require seed-discovery output. |
| ch06 solver/Jacobian diagnostics | Lumice Integral data; Writing-Lab presentation | Supported as versioned figure data | A production plotting consumer still belongs in Writing-Lab; an independent prototype consumer has been verified. |
| ch06 named physical-factor curves | Lumice Integral | Supported for `rho_pose`, `entry_measure`, `fresnel_transmission`, `path_validity` on the canonical pixel fiber (section 4.1, figure-data v2 `weight_<name>` arrays) | `visibility` (finite-face obstruction) and the radiometric factors remain unavailable. |
| ch06 one-pixel integrand/integral | Lumice Integral | Named factor values exposed pointwise; quadrature not supported | Converged line quadrature over the closed fiber (`task-single-fiber-line-quadrature`). |
| ch06 `251 x 801` direct strip | Lumice Integral | Historical bytes can be loaded; physical rerender is not supported | Seed/component discovery, neighboring-pixel continuation, camera/pixel model, image driver, and the one-pixel stages above. |
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
weights (result.weight_observables):
  each requested factor: status, unit, normalization, array name or null
quadrature:
  status, method, refinements, value, error estimate   (still open)
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

JSON metadata carries the semantic names, units, conventions, shapes,
and SHA-256 of the NPZ payload. Quadrature data is still not emitted. Empty failed fibers are represented without
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
   units and normalization; quadrature refinements and convergence remain
   open (`task-single-fiber-line-quadrature`).
4. **Historical image scene**: render the canonical `251 x 801` strip and
   compare raw profiles with the historical binary plus an independently
   converged Lumice result after coordinate/radiometric alignment.
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
