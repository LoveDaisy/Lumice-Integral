# Fixed ray-path diagnostic reference

`lumice_integral.raypath_diagnostic_reference` is a narrow, reproducible
reference for two cases. It assembles existing LI authorities; it is not a
second optical solver or a general feature enumerator. The exporter runs
without Lumice source, libraries, binaries, or output files.

Generate the recorded format with:

```bash
uv run python scripts/export_raypath_diagnostic_reference.py \
  --output-dir artifacts/raypath-diagnostic-reference
```

The directory contains `reference.json` (schema and scalar evidence),
`arrays.npz` (poses and sampled factors), and `provenance.json` (LI revision,
source revisions, dirty state, parameters, command, platform, and hashes).
The schema is `lumice-integral.raypath-diagnostic-reference/v1`. Downstream
consumers should cite the LI revision in `provenance.json`; the format is an
export fixture, not a runtime Lumice API.

## Randomly oriented regular prism

The case `random_regular` uses `HexPrism(a=1,h=1)`, normalised Haar measure,
and refractive-index endpoints `red=1.307`, `blue=1.317`.

- `3-5`: the inner edge is evaluated independently by
  `2 asin(n sin 30°) - 60°` and by
  `focusing.wavelength_critical_table`. They agree within the recorded
  float64 residual. The field profile is `finite_jump`, so this evidence is
  an ordinary dispersive minimum-deviation edge, not a divergent Jacobian
  caustic.
- `3-1-5`, solar side: the same ordinary red-to-blue dispersion edge is
  retained. The coincident DPField boundary critical record is separately
  marked `candidate`; its `degenerate` label alone does not establish a
  visible caustic.
- `3-1-5`, antisolar side: the internal-reflection TIR kink is a confirmed
  blue band. The moving exit gate is red but assessed as not visible because
  its deviation spread is far larger than its red/blue shift. Thus “3-1-5
  has no red edge” is not an all-sky statement, and “3-1-5 has only blue” is
  also too broad.
- The `R` counterfactual uses the same kink poses, domains, entry measure,
  and entry/exit Fresnel terms. It removes only the internal reflectance.
  The recorded blue/red `A*T` ratio changes from about 2.07 to 0.75; this
  isolates the blue band's internal-reflection mechanism without turning
  internal TIR into a validity gate.

## Ideal horizontal plates at the two 120-degree azimuths

The case `plate_rhombic_9` fixes
`HexPrism(a=1,h=2,face_distance=[1.5,1,1,1.5,1,1])`, a sun at altitude 9°
and azimuth 180°, and poses `Rz(theta)` under `dtheta/(2*pi)`. The two sky
targets have relative solar azimuths +120° and -120°. Their true spherical
separation from the sun is about 117.599764°, which is recorded separately
from the azimuth label.

The white class is represented by `1-3-4-2`; the blue class by `1-3-5-2`.
Every member of each 24-member L1/PBD orbit is checked independently. For
this fixed instance, entry and exit refractions are parallel, the wedge is
zero, and the fold matrix commutes with `Rz`. On a physical branch,

```text
out(Rz(theta), n) = Rz(theta) M Rz(theta)^T incident = M incident.
```

`M incident` assigns each member once to `plus`, `minus`, or `other` before
integration. A direction residual is verification evidence, never a
theta-dependent mask. The L1 label orbit is not claimed to be an L2 physical
equivalence class.

Finite-crystal support is then found independently from production `A*T`.
The exporter uses 4096/8192 periodic midpoint grids, refines only transitions
already found on the 8192 grid to a bracket width at most `1e-8` rad, and
records both bracket sides and their weights. Consequently it is
`resolution_limited`: a support component narrower than one fine-grid step
has not been excluded.

Per member and refractive index,

```text
E_m,n = (1 / 2*pi) integral A_m,n(theta) T_m,n(theta) dtheta
```

is the midpoint mean. Members are summed after their fixed target assignment.
The two targets are kept separate; `class_total` is only a partition check.
At the recorded resolution each target has blue/red ratio about 1.010 for
the white class and 1.525 for the blue class. `diagnose_class` at 4096 random
spins is stored only as a qualitative class-level cross-check, not as the
per-target energy authority.

The NPZ contains five effective poses per class, their body-frame sun
directions `u`, SO(3) matrices, outgoing directions, `A`, and `T`. Along
these poses `R u = s_hat`, the c axis stays vertical, and red/blue directions
hit the assigned target within the recorded residual. The negative example
left-multiplies a pose by 10° about the world sun axis: it preserves `u` but
breaks the horizontal-family constraint and moves the output off the target.
This excludes “same projection is enough”; it does not assert a general SO(3)
relation or rank-0 collapse.

## Coverage and non-claims

The reference does not evaluate finite-solar-disc convolution, relative
prominence against all paths, the oriented sky curve of a general kink,
other pose families, open/multiple components, cone-crystal empty results,
all-sky feature enumeration, or rank-0 cases. Numerical values carry their
grid, lattice, bisection width, coarse/fine difference, or residual source;
they are convergence evidence, not exact results.
