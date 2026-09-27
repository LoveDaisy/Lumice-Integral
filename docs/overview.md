# Lumice Integral: Overview

Chinese version: [overview_zh.md](overview_zh.md).

Lumice Integral computes ice-halo radiance by deterministic numerical
integration. It is not a general quadrature library and it is not another
Monte Carlo ray tracer. Its central problem is:

> Given an incident-light direction, an outgoing image direction and a ray
> path, find every crystal pose that realises that direction mapping, and
> integrate the physical contribution over that pose set.

It began as the reconstruction of the author's direct-integration prototype
for chapter 6 of the writing series *Modern Ice Halo Research Notes*; the
prototype's solver source was lost, and only rendered data and diagnostic
figures remained. It now also provides the numerical tools for the series'
later chapters (halo maps, Jacobians, caustics, path classes, pose
families).

## 1. Two ways to render a halo

Both approaches compute the same quantity: the light a population of
randomly oriented crystals sends into each sky direction,

$$
I(\mathbf d) = \sum_P \int_{\mathrm{SO}(3)} \rho(R)\,A_P(R)\,T_P(R)\;
\delta_{\mathrm{Dirac}}\big(\mathbf d,\ F_P(R)\big)\,d\mu_{\mathrm{Haar}}(R),
$$

summed over ray paths $P$; $R$ is the crystal pose, $\rho$ the pose density,
$A_P$ the entry measure (how much of the crystal's cross-section feeds the
path), $T_P$ the optical throughput, $F_P$ the halo map from pose to
outgoing direction.

**Monte Carlo forward rendering** ([Lumice](https://github.com/LoveDaisy/ice_halo_sim)):
sample poses and rays, trace each ray through the crystal, and accumulate
where it leaves. The delta function is handled by binning into pixels. It is
general (any crystal, any path, multiple scattering, any source) and simple
to trust, but noisy: precision grows as the square root of the ray count,
and dark or narrow features need very many rays. It says little about
*why* a halo looks the way it does.

**Direct integration** (this project): fix the outgoing direction and solve
for the poses. For one path the delta function cuts $\mathrm{SO}(3)$ down to
the pose set $F_P^{-1}(\mathbf d)$, generically a set of closed curves, and
the pixel value is a line integral along them weighted by the inverse normal
Jacobian (coarea formula). It is deterministic (numerical error, reported,
instead of sampling noise) and it exposes the structure: fibres, Jacobians,
fold caustics, the role of each path and each pose family. Its cost is that
the solver must find every component of every fibre, and that each path is
handled explicitly.

The two are independent implementations of one physics, so agreement
between them is evidence: Lumice is this project's external validation
oracle, never a dependency (section 5.1).

## 2. Phase I and Phase II

**Phase I** ([phase1.md](phase1.md), closed 2026-09-23) follows the
prototype: for each pixel it traces the pose fibre on $\mathrm{SO}(3)$ by
predictor-corrector continuation with automatic differentiation, discovers
every component from a precomputed table of poses, and integrates the named
factors along it. It reproduced the chapter-6 tangent-arc strip
(251 × 801) and agrees with Lumice at the Monte Carlo noise floor in shape
and to `0.997-0.999` in absolute scale. It is the pointwise reference.

**Phase II** ([phase2.md](phase2.md)) uses a fact Phase I cannot see: every
geometric and optical weight depends on the pose only through
$\mathbf u = R^{-1}\hat{\mathbf s}$, the source direction in the crystal
frame. The integral moves to the sphere $S^2$, where the pixel value is a
line integral along a level set of one scalar field, the deviation
$D_P(\mathbf u)$. Two quadratures of it:

- the **band sum** (in production since 2026-09-23): precompute $N$ events
  on $S^2$ once per crystal, path and wavelength, independent of the light
  source, and sum those in each pixel's deviation band. The canonical strip
  takes `31 s` on four Mac workers (scatter form, 2026-09-24; `169 s` before)
  against Phase I's `34.7 min` on 30;
- the **contour method** (milestone M2, done 2026-09-25): trace the level
  sets and integrate along them; with the critical points of $D_P$ it
  certifies that every component was found, which Phase I cannot.

Three quadratures of one integral with unshared failure modes (Phase I
fibres, band sum, contours) plus Lumice's Monte Carlo form the project's
validation web.

## 3. Plan

Milestones and their chapters of the writing series (status and the
near-term queue: [roadmap.md](roadmap.md) §0; decisions: roadmap §9):

| milestone | content | chapters | status |
|---|---|---|---|
| Phase I | $\mathrm{SO}(3)$ continuation renderer, chapter-6 strip, path classes, five pose families, absolute scale | 6, 7-9, 11 | closed 2026-09-23 |
| M1 | $S^2$ event store and band-sum renderer in production; $D_{6h}$ transport; conventions and symmetry authority | 6, 11 | done 2026-09-24 |
| M2 | contour quadrature, critical points and completeness certificate, cross-validation, chapter-10 verdicts | 10 | done 2026-09-25 |
| — | internal reflections as Fresnel splits (the TIR-only gate dropped every partial-reflection branch), store schema 4 | 8, 10, 11 | done 2026-09-25 |
| next | chapter-11 table (path classes × pose families) with the band sum organised by deviation | 11 | queued |
| later | divergent light (street lamps), finite solar disk, multiple wavelengths | — | backlog |

Explicit non-goals:

- A general-purpose numerical integration or differential-geometry library.
- Replacing Lumice's Monte Carlo renderer.
- Multi-scattering scenes.
- Exact lower-dimensional orientation measures.
- A production GUI.
- GPU optimization before the CPU reference result is trustworthy.
- Any production dependency on Lumice or extraction of a Lumice engine API for
  this project.

## 4. Architecture layers

The project will likely need the following conceptual layers, without implying
that each must become a package immediately:

1. **Independent differentiable optical evaluator**: project-owned geometry and
   optics mapping pose, path, and wavelength to outgoing direction, validity,
   named physical weights, and the smooth computation graph required by AD.
   Finite-crystal geometry (convex polyhedra, path unfolding, corridor
   projection intersection, ray-path enumeration, and the single-pose effective
   entry cross-section `entry_measure`) is owned by `lumice_integral.geometry`.
   That subpackage is pure numpy and sits outside the differentiable graph; it
   supplies the geometric factor $A_P(R)$ that the named physical weights
   multiply, and is the designated authoritative implementation; the writing
   project still carries its own copy until its task
   `geometry-depend-on-lumice-integral` (W1) switches it over (roadmap §3.6).
2. **Symmetry and combinatorial classification**: the crystal point groups and
   what they classify are owned by `lumice_integral.symmetry` (task
   `symmetry-authority`, 2026-09-24): `D6` as face permutations, the fold
   group `G` with its published numbers #1–#12, conjugacy and eigenvalue
   classes, the `D6h` element table `signature.D6H` (the only one in the
   repository; `path_class.hexprism_symmetry_matrices` returns it), canonical
   signatures and `Phi` classes (34 on the prism), the chapter-3 ground-truth
   list and the column attitude. Pure numpy, depending on `geometry` and never
   the reverse (static test). One cross-subpackage edge is deliberate:
   `geometry.unfold._wedge_angle_deg_from_normals` exists so that
   `geometry.wedge_angle_deg` and `symmetry.signature.wedge_angle` share one
   wedge-angle kernel; it is not a geometry-only helper and must not be
   removed as unused by geometry. The writing project switches to it in its
   task 33 (W3, roadmap §3.6).
3. **Differential evaluator**: derivatives with respect to local SO(3)
   coordinates, preferably from the same equations as the value evaluator.
4. **Fiber solver**: seed search, predictor-corrector continuation, component
   discovery, closure, and diagnostics.
5. **Integrator**: parameterization-aware line quadrature and error estimates.
6. **Image driver**: source, spectrum, pose distribution, pixel model, caching,
   and output assembly.

The boundaries should emerge from the first working slice. They are not a
request to build six frameworks before tracing one loop.

### 4.1 Crystal construction (closed form) and the crystal's own group

Both phases take the crystal from `lumice_integral.geometry`, so its
construction is described here rather than in a phase document (task
`crystal-closed-form`, scrum `crystal-native-geometry`, 2026-09-27;
conventions #19).

- **Semantics are Lumice's** (`doc/configuration.md`): a prism has a height
  and six `face_distance` ratios of the regular apothem `a·√3/2`, negative
  values allowed; `HexPrism(a, h, face_distance)`, with
  `HexPrism.from_lumice(height, face_distance, a)` for Lumice's
  `height = h / (2a)`. `Pyramid` covers the symmetric subset (regular cross
  section, `upper_h = lower_h` truncated cones); asymmetric and one-sided
  cones are a later stage and the construction does not preclude them.
- **Closed form, no topology discovery** (the route Lumice abandoned in
  PR #214, `doc/crystal-geometry-representation.md` §1, §4): the six side
  planes are a fixed star, so for each side face the other five half-planes
  cut its line to one analytic interval; the face is *present* iff the
  interval has positive length (scale-relative `1e-9`). Corners are the
  intersections of consecutive present faces, written as the regular
  hexagon's corner plus the exact linear correction, so the regular prism
  is the historical one bit for bit. Faces are the present subset of the
  constant numbers 1/2, 3–8 (13–18, 23–28 on the pyramid); an absent face
  number is a `KeyError`, not a silently wrong normal.
- **Rejection is fail-fast**: fewer than three present side faces means the
  cross-section has no area and the constructor raises `ValueError` where
  Lumice drops the crystal. (Positive width of each opposite pair is not
  enough: `[1, 1, -0.5, -0.9, -0.9, 1]` has three positive-width slabs that
  do not meet.)
- **`G_true`** (`symmetry.crystal_group.true_symmetry_group`): every normal
  lies in the six-direction star or on `±c`, so the crystal's own symmetry
  group is the subset of `signature.D6H` that maps the present faces'
  (normal, offset from the vertex centroid) onto themselves, verified to be
  a group. Regular prism and symmetric pyramid 24, `[1, d, 1, d, 1, d]` 12,
  `[1.9, 1, 1, 1.9, 1, 1]` and `[2, 1, 1, 2, 1, 1]` 8, a generic shape 2
  (`{E, σh}`).
- **Normals come from the crystal** (task `optics-reads-crystal`,
  2026-09-27): every single-path function of `optics` (`path_direction`,
  `path_domain[_batch]`, `fresnel_transmission_path[_batch]`, `path_problem`
  and the `path_3_5*` wrappers) takes `crystal` (default: the regular
  `HexPrism()`) and reads the normal of face `f` as
  `crystal.normal(crystal.face(f))` through `optics.face_normals`, the one
  lookup. A face 1–8 whose normal lies on its closed-form star direction
  (`±c`, azimuth `i·60°`) to `1e-12` reads the exact direction (Newell's
  formula lands ~1e-16 off it), so the regular prism evaluates bit for bit
  as on the historical constant table the ch06 fixtures, the strip pipeline
  and the $S^2$ store are pinned to; a face off the star keeps its own
  normal. `HEXPRISM_BODY_NORMALS` is that lookup on the regular prism, kept
  for callers that index a table. The `D_P` kernels (`dp_field`, `contour`,
  `contour_quadrature`, `ch10_verdicts`) take the normals as a traced array
  (`DPField.normals`), so one compilation per face sequence serves every
  crystal. Every consumer that holds a crystal passes it down (`DPField`,
  `focusing.classify`, `weights`, `s2_store.evaluate_fields`, discovery,
  the strip scene); a face the crystal does not have is a `ValueError`, and
  the pyramid's faces 13–28 run through `focusing.classify`. The boundary
  walk's margin identity (`identical_margins`) is confirmed on the crystal's
  normals rather than assumed from face numbers.
- **The reduced cluster fails fast below `D6h`**: `path_class`'s
  `pbd_orbit_hexprism`, `phi_key` and `path_class_symmetry` (and `s2_store`,
  `strip_pixel`, `band_sum`, which reach the crystal's symmetry only through
  them) raise `ValueError` unless `|G_true| = 24`; explicit
  `symmetry_elements` must lie in `G_true`. The `D6h` orbit of a path is not
  the crystal's orbit on a lower-symmetry shape (`3-5` has 12 `D6h` images;
  `[2, 1, 1, 2, 1, 1]` has 8 symmetries), so these would silently merge
  paths that are not images of each other. Generalising orbits, stores and
  transports to `G_true` is the next stage.
- **Checked against Lumice at the validation boundary**:
  `scripts/verify_crystal_closed_form.py` calls Lumice's
  `LUMICE_GetCrystalMesh` through `ctypes` and compares present faces,
  normals and corners on the critical shapes (far faces exactly touching a
  corner) and 3000 random `face_distance` sets (all agree, rejections
  included), next to an independent check (half-space containment,
  `V − E + F = 2`).

## 5. Relationship to other projects

### 5.1 Lumice

Lumice is the forward Monte Carlo simulator and the primary independent
validation oracle. Lumice Integral is a sibling product line with a different
numerical method.

Lumice Integral intentionally does **not** consume Lumice's source code, C API,
libraries, or executable as part of its production computation. The solver must
own the complete differentiable path from pose through geometry and optics to
direction, physical weights, and derivatives. This is a permanent design
boundary, not temporary duplication awaiting a future shared engine.

The reason is structural. Lumice is optimized for forward stochastic sampling
and image accumulation. Lumice Integral needs a fixed-path, piecewise-smooth
computation graph suitable for automatic differentiation and continuation. Face
changes, obstruction, refraction-domain limits, and entry/exit TIR boundaries must be
represented as explicit events around smooth branches; an opaque Lumice call
would sever that graph, while finite differences through it would not provide a
trustworthy foundation near those boundaries or halo-map singularities.

Lumice may be used only across an explicit validation boundary:

- run independently to produce converged Monte Carlo images or profiles;
- export ray-path analysis data for comparison;
- check shared physical and coordinate conventions such as face numbering,
  direction signs, refractive indices, Fresnel factors, and pose distributions.

Validation tools may invoke Lumice and read its files when explicitly requested,
but Lumice must not be required to build or run the Lumice Integral solver. The
two independent implementations strengthen cross-validation: agreement is more
meaningful when it cannot arise from a shared geometry or optics bug.

### 5.2 Modern Ice Halo Research Notes

The writing project motivates the solver and consumes its results, but does not
own its implementation. Lumice Integral should eventually regenerate the
direct-integration strip and state-space diagnostic from chapter 6, and provide
numerical evidence for the halo-map, Jacobian, fiber, and caustic discussion in
later chapters.

Two of the writing project's code layers now live here and are its
authorities: the crystal geometry (`lumice_integral.geometry`, 2026-09-16)
and the symmetry and classification layer (`lumice_integral.symmetry`,
2026-09-24, migrated from `halo_notes.math` with public names kept 1:1). The
chapter-8 and chapter-9 signature tables recomputed with this package
reproduce the published CSV files byte for byte
(`tests/test_symmetry_signature_table_regression.py`, reading the writing
repository's data files only).

### 5.3 The Ice Halo raypath-analysis panel

Owner requirement (recorded authoritatively in Ice Halo's
[`doc/raypath-analysis.md`](https://github.com/LoveDaisy/ice_halo_sim) §5.1,
rewritten the same day by that repository's chore
`raypath-analysis-lumice-integral-plan`, 2026-09-27): after a user selects one
ray path in the panel, three functions follow:

1. **path detail**: the one-dimensional pose fiber, the crystal pose and
   in-crystal ray trajectory (visualization / animation), and the per-segment
   energy allocation (a beam tube is one visual form of it);
2. **whole-sky brightness map for that single path**, with re-selection on it
   returning to function 1 (function 2's product form is on owner hold);
3. **preset points**: the brightest point, critical points/lines ($D_P$
   extrema/saddles, $U_P$ boundaries such as a tangent-arc edge), and
   wavelength-dependent critical points (e.g. where blue is already past TIR
   but red still transmits).

Multi-scattering chains need to be supported.

The three functions are three readings of one $S^2$ object (this is what
belongs in LI's own docs, not a repetition of the panel's product design):
function 2 is the band sum; function 1 is the pixel's level set
$\{D_P(\mathbf u) = \delta\}$ (the events the band sum lands in that pixel's
band are the fiber's discretisation; `band_sum.band_poses` already extracts
those poses); function 3 is $D_P$'s critical structure plus the $U_P$
boundary (`dp_field`, `focusing.classify`), with wavelength criticality being
the $U_P$ boundary moving with $n(\lambda)$. All three share one field
precomputed by crystal × path × wavelength, independent of the light source.
"Why the halo is here" as a product form is function 3's preset points plus
its mechanism label — this is LI's value add that Lumice's Monte Carlo form
cannot give.

Multi-scattering scenes stay outside LI's scope (section 3's non-goal is
unchanged), but a two-bounce chain can be composed at the consumption layer
from single-bounce results through an intermediate direction $\mathbf m$: the
$S^2$ store is independent of the light source, so the second bounce reuses
it with $\mathbf m$ as its source. Whether LI itself owns this composition
layer is undecided; this paragraph only records the requirement's source, it
is not a scope change.

**Compute landing point (owner ruling 2026-09-27, closing explore
`panel-inverse-probe` and `ad-port-probe`; local records
`scratchpad/explore-panel-inverse-probe/SUMMARY.md`,
`scratchpad/explore-ad-port-probe/SUMMARY.md`):** this is not a one-shot
"single implementation vs. dual implementation" architecture choice; it is a
per-module decision timed by module maturity.

Measurements: at panel precision, compute is ample (a $161\times81$ window,
$N=10^6$, full chain $3.4\,\mathrm{s}$; a single pixel's fiber pick
$1$–$2\,\mathrm{ms}$; for scale, the Python renderer runs
$\approx 74\,\mu\mathrm{s}/\mathrm{px}$, so a $512^2$ whole-sky image is
$\approx 20\,\mathrm{s}$). Functions 1/2's core (`band_sum`, `s2_store`,
`weights`, `geometry`) has zero JAX dependency. Function 3's AD reduces
exactly to a forward hyper-dual template in C++ (`Jet2<3>`, $\approx 150$
dependency-free lines): three paths, including one with an internal
reflection and one on the pyramid family (`13-15-26-28`), match JAX's value,
gradient and Hessian to a relative error $\le 10^{-11}$; a single point is
$193\,\mathrm{ns}$ in C++ versus $104\,\mu\mathrm{s}$ in JAX. Function 3's C++
port is estimated at $2800$–$3600$ lines, dominated by the boundary /
level-set walk state machine, not by the AD itself.

The ruling:

1. **Port nothing now.** Everything stays in LI's JAX implementation, free to
   change under research and writing.
2. **Port trigger** = both hold at once: the panel function is actually
   scheduled, and the module has gone without a semantic change for a while
   (reference point: after stage 2, the asymmetric cone, multiple wavelengths
   and multi-scattering composition have all landed).
3. **After porting: JAX is authoritative, C++ is derived**, locked by a
   parity fixture (path topology: no internal reflection / with internal
   reflection / outside the pyramid family, crossed with point class: random
   / critical / near-boundary; template = explore `ad-port-probe`'s
   `compare_*.py`), run in CI; changes flow one way — JAX first, parity red,
   then C++. This is not two implementations of the same semantics silently
   diverging: the divergence is caught automatically, with no "which side is
   right" question.
4. Functions 1/2 are treated the same as function 3 (`band_sum` / `s2_store`
   are equally young); the C++ side uses the same forward hyper-dual template
   for scalar derivatives, not an AD framework.
5. Whether a module that has been stable for a long time should also move LI
   itself onto the C++ path (retiring the JAX version) is left for a later
   decision.
6. This supersedes the earlier "main session leans toward (c), extract a
   portable C++ core" wording: (c)'s shape is kept as the eventual post-port
   form, but its timing is governed by the trigger in point 2, not decided up
   front.

**Constraint on LI in the meantime:** the panel-relevant modules (`optics`,
`weights`, `geometry`, `s2_store`, `band_sum`, `dp_field`, `contour`,
`focusing`) keep evolving as usual and are not frozen for the panel's sake.
Once a module is ported, its changes must land in JAX first and be
synchronized against the parity fixture before touching C++.

To be verified later (not in this chore, flagged only): whether the panel's
symmetry-reduced row semantics (Lumice `doc/raypath-symmetry.md`'s P/B/D) and
LI's `G_true` orbit are the same convention.

## 6. Validation strategy

Validation must combine several independent kinds of evidence:

- analytically tractable maps and symmetry cases;
- local residual and tangent checks along every traced fiber;
- AD derivatives against finite differences at well-conditioned poses;
- quadrature refinement and reparameterization invariance;
- the surviving historical direct-integration output, for morphology and
  provenance only (author's ruling 2026-09-20: the historical raw is not a
  correctness reference; Lumice agreement is the criterion);
- agreement with Lumice after Monte Carlo uncertainty, source model, spectrum,
  pose density, projection, and radiometric normalization are aligned;
- explicit convention fixtures at the validation boundary, without a shared
  geometry or optics implementation;
- Phase II cross-checks: the band sum against Phase I (done, M1), and the
  contour method against both (M2).

A smooth-looking image is not sufficient evidence: missing fiber components can
produce plausible but systematically wrong radiance.

## 7. Documents

| document | role |
|---|---|
| `overview.md` (this) | what the project is, how it differs from Monte Carlo, the phases, the plan |
| [phase1.md](phase1.md) | Phase I design, pipeline, key turns and their reasons; measured record |
| [phase2.md](phase2.md) | Phase II design: the $S^2$ integral, both quadratures, the event store, costs, the band sum by deviation, divergent light; measured record |
| [roadmap.md](roadmap.md) | status, near-term queue, writing-project coupling, decisions log |
| [phase1-math-contract.md](phase1-math-contract.md) | normative Phase I contract (coordinates, measures, events, interfaces, conformance) |
| [conventions.md](conventions.md) | every coordinate, sign and symbol convention with its authority and check |
| [ch06-reference-fixture.md](ch06-reference-fixture.md) | chapter-6 fixture, provenance, acceptance stages, Lumice comparisons |
| [ch11-pose-density-families.md](ch11-pose-density-families.md) | the five pose-density families |
| [decisions/](decisions/) | architecture decision records |

