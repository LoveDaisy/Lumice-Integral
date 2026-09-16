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
| `historical-direct` | Present in the chapter text, old filename, old script, image metadata, or raw bytes. |
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

## 5. Figure Capability Matrix

| Figure or chapter need | Data owner | Current status | Missing capability |
|------------------------|------------|----------------|--------------------|
| ch06 crystal-orientation schematic | Writing-Lab drawing code | Supported | None in Lumice Integral; not a numerical-solver responsibility. |
| ch06 ray-splitting schematic | Writing-Lab drawing code | Supported | None in Lumice Integral. |
| ch06 all-sky Monte Carlo example | Lumice through Writing-Lab validation glue | Supported | Not a Lumice Integral product output. |
| ch06 pose-fiber geometry | Lumice Integral | Supported for one supplied regular seed/component | Historical pose-coordinate projection still needs author confirmation. |
| ch06 solver/Jacobian diagnostics | Lumice Integral | Supported | A stable figure-data export and plotting consumer. |
| ch06 named physical-factor curves | Lumice Integral | Not supported | Pose density, entry measure/visibility, Fresnel throughput, and evaluated factor samples. |
| ch06 one-pixel integrand/integral | Lumice Integral | Not supported | Named factor evaluation plus converged line quadrature. |
| ch06 `251 x 801` direct strip | Lumice Integral | Historical bytes can be loaded; physical rerender is not supported | Seed/component discovery, neighboring-pixel continuation, camera/pixel model, image driver, and the one-pixel stages above. |
| ch10 halo-map/Jacobian/fold figures | Lumice Integral numerical data; Writing-Lab presentation | Partially supported | Target sweeps and singular/fold localization beyond one regular fiber. |
| ch11 orientation-family comparison | Lumice Integral and/or independent Lumice validation | Not supported by the current ordinary-density slice | Pose-density models, physical weights, image driver; exactly constrained families require a separate measure/domain contract. |

## 6. Proposed Figure-Data Product

Solver objects are useful in Python but are not a stable boundary for the
writing project. A future exporter should write a versioned metadata document
plus array payload, for example:

```text
schema: lumice-integral.figure-data/v1
metadata:
  path, incident, target, material, wavelength
  pose convention, metric/measure, solver options
  component scope/completeness, terminal state
arrays:
  poses, cumulative_arclength, tangents, residual_norm
  singular_values, normal_jacobian, condition
  named branch margins
weights:
  each requested factor: status, unit, normalization, values-if-available
quadrature:
  status, method, refinements, value, error estimate
```

NPZ is sufficient for a Python-first diagnostic artifact; JSON metadata MUST
carry the semantic names, units, conventions, shapes, and SHA-256 of the array
payload. The format must not serialize arbitrary Python objects or require
Lumice at read time.

## 7. Acceptance Stages

1. **Historical loader**: validate the raw hash/shape/dtype and reproduce the
   recorded raw statistics without assigning invented coordinates.
2. **Single-fiber figure data**: export the canonical geometry, diagnostics,
   provenance, and explicit unavailable-factor states.
3. **Physical one-pixel result**: expose every named factor, the coarea
   denominator, quadrature refinements, and a convergence estimate.
4. **Historical image scene**: render the canonical `251 x 801` strip and
   compare raw profiles with the historical binary plus an independently
   converged Lumice result after coordinate/radiometric alignment.
5. **Writing-Lab figures**: consume only the versioned data product; composition,
   typography, annotations, and chapter-specific styling remain there.

## 8. Open Author Semantics

The historical `state_space_integrate.jpg` cannot be faithfully relabeled
without two author clarifications:

1. the pose coordinates/projections in its upper-left and lower-left panels,
   and the meanings of blue versus red samples;
2. the right-panel parameter axis and the identities of its five curves.

Until confirmed, a new diagnostic figure SHOULD use explicit modern labels
(`cumulative SO(3) arclength`, `normal_jacobian`, named domain margins, and
solver residual) rather than visually imitating the unlabeled historical plot.
