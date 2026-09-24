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
  takes `169 s` on four Mac workers against Phase I's `34.7 min` on 30;
- the **contour method** (milestone M2, in preparation): trace the level
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
| M2 | contour quadrature, critical points and completeness certificate, cross-validation, chapter-10 verdicts | 10 | bootstrapped |
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
changes, obstruction, refraction-domain limits, and TIR boundaries must be
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

