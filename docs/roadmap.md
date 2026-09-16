# Lumice Integral Roadmap

> Status: initial design, 2026-09-15. This document records the current shared
> understanding; implementation choices remain open until they are tested.

## 1. Project Positioning

Lumice Integral computes ice-halo radiance by deterministic numerical
integration. It is not a general quadrature library and it is not another
Monte Carlo ray tracer. Its central problem is:

> Given an incident-light direction, an outgoing image direction, and a fixed
> ray path, find every crystal pose that realizes that direction mapping and
> integrate the physical contribution along that pose set.

The original prototype demonstrated this construction for one halo example but
its solver source no longer exists. Only rendered data and diagnostic figures
remain. The first phase therefore reconstructs the method from its mathematical
description and observable results rather than porting old code.

"Deterministic" does not mean analytically exact. Root finding, continuation,
automatic differentiation, and quadrature are numerical and must expose their
errors and convergence behavior.

## 2. Mathematical Model

Fix a crystal, ray path `P`, wavelength, and world-space incident direction
`s`. A crystal pose is a rotation `R` in SO(3). Ray propagation defines a halo
map

$$
F_P : SO(3) \longrightarrow S^2,
\qquad R \longmapsto \text{outgoing direction}.
$$

For an image direction $\mathbf d$, the contributing pose set is the inverse
image

$$
X_{\mathbf d}=F_P^{-1}(\mathbf d).
$$

At a regular value of $F_P$, the dimension theorem gives

$$
\dim X_{\mathbf d}=\dim SO(3)-\dim S^2=3-2=1.
$$

Thus the integration domain is generally a collection of one-dimensional
curves, often closed loops, embedded in SO(3). With a smooth pose density, the
pointwise contribution of a ray path has the coarea form

$$
I_P(\mathbf d)=
\int_{X_{\mathbf d}}
\frac{
  \rho(R)\,A_P(R)\,T_P(R)
}{J_{F_P}(R)}
\,d\mathcal H^1(R).
$$

The factors have separate physical and diagnostic meanings:

- $\rho(R)$: pose probability density with respect to Haar measure on SO(3).
- $A_P(R)$: projected area and the measure of geometrically realizable entry
  points, including obstruction and finite-crystal effects.
- $T_P(R)$: optical throughput along the selected path, including Fresnel and
  total-internal-reflection behavior.
- $J_{F_P}(R)$: the normal Jacobian of the halo map.
- $d\mathcal H^1$: arc-length measure along the pose fiber.

The precise normalization, radiometric units, source spectrum, polarization,
and finite-pixel response must be made explicit before results are compared
quantitatively.

## 3. Phase I: SO(3) Continuation

Phase I deliberately follows the original prototype. It treats $F_P$ as a
black-box differentiable map on SO(3) and traces the implicit fiber directly.
The newer reduction to an $S^2$ contour is not part of the Phase I algorithm.
The [Phase I mathematical and numerical contract](phase1-math-contract.md) is
the single source for its coordinate conventions, interface semantics,
measures, event taxonomy, and conformance invariants.

### 3.1 Core algorithm

For one target direction $\mathbf d$:

1. Numerically find a feasible seed $R_0$ satisfying
   $F_P(R_0)=\mathbf d$.
2. Use automatic differentiation to evaluate the differential in local Lie
   algebra coordinates. The target-direction residual has two independent
   components, so the local Jacobian is $2\times3$.
3. Extract the one-dimensional null space of the Jacobian as the tangent to
   $X_{\mathbf d}$, with a consistent orientation from one step to the next.
4. Predict the next pose with an SO(3) exponential-map update.
5. Correct the prediction back onto the constraint manifold. A
   predictor-corrector or pseudo-arclength formulation should control drift and
   remain usable near poorly conditioned regions.
6. Continue until the component closes or reaches a genuine boundary of the
   valid optical domain.
7. Evaluate every physical factor separately along the curve, then perform
   adaptive line quadrature on their product.

Rotations may be stored as unit quaternions, but numerical updates and
derivatives should use the three-dimensional tangent space of SO(3). Closure
logic must respect the double cover $q\sim-q$ and must not rely on quaternion
distance alone.

### 3.2 Reconstruction milestone

The first vertical slice is intentionally narrow:

- hexagonal prism;
- fixed 3-5 ray path;
- single wavelength;
- far-field point source;
- smooth pose density on SO(3);
- one known image direction whose inverse image is a closed loop.

The milestone is complete when the program can:

- find one seed without a hand-entered pose;
- trace the complete known loop with bounded constraint residual;
- detect closure without closing early;
- plot the loop in diagnostic SO(3) coordinates;
- output the Jacobian, pose-density, geometric, and Fresnel factors separately;
- integrate their product with a numerical convergence report.

The surviving chapter-6 artifacts, recovered parameters, canonical replacement
fixture, and figure-level capability gaps are tracked in the
[chapter 6 reference fixture specification](ch06-reference-fixture.md).

This milestone reconstructs the original prototype and produces the information
needed to remake the historical `state_space_integrate` diagnostic figure.

### 3.3 From one fiber to a renderer

Reconstructing one known loop is not yet a general solver. Later Phase I work
must address:

- discovering every connected component rather than tracing only the component
  containing the first seed;
- distinguishing true closure from a near self-approach;
- continuation across coordinate-chart and quaternion-sign boundaries;
- fibers that meet refraction, TIR, or geometric-feasibility boundaries;
- rank loss, bifurcation, and topology changes near caustics;
- reuse of solutions between neighboring pixels without silently losing or
  merging branches;
- pointwise singular density versus radiance integrated over a finite solar disk
  and finite pixel solid angle.

The first image-level target is the historical local tangent-arc strip at
`251 x 801` resolution. It should be compared with both the surviving direct
integration data and an independently converged Lumice Monte Carlo render.
Status: the strip driver exists (`lumice_integral.strip_pixel` /
`strip_driver` / `strip_io`, `scripts/render_ch06_strip.py`): column-wise
neighbouring-pixel hot start gated by arclength-jump detection with cold
prescan fallback and periodic cold spot checks, a per-pixel procedural
completeness status layer, and the point pixel model chosen by a caustic
probe (the probe found `O(10-40 %)` point-vs-sub-pixel differences within
about seven rows of the `22 deg` inner edge at the centre column, so the
caustic band is a documented pixel-model limitation of the default render).
A full-height every-ninth-column preview is rendered and compared: Spearman
`0.99` against the historical raw on pixels lit in both, a consistent
`10`-row inner-edge offset, and the lower quarter of the strip (rows
`600-800`) `unknown` because no discovery candidate closes there; the full
`251 x 801` render is in progress. The evidence is recorded in the
[chapter 6 reference fixture specification](ch06-reference-fixture.md)
sections 5 and 7. Component completeness remains procedural, not certified;
finite solar disk and finite pixel solid angle are still open (the sub-pixel
model is implemented but costs 6-10x and is off by default).

### 3.4 Orientation distributions

Phase I assumes that $\rho$ is an ordinary, possibly narrow, density on all of
SO(3). Plate-like and column-like preferences therefore change the integrand but
do not change the one-dimensional fiber.

Exactly constrained ideal orientation families are singular measures supported
on lower-dimensional subsets of SO(3). Supporting them requires a generalized
domain and a different dimension count; this is explicitly outside the first
reconstruction milestone.

## 4. Phase II: Fiber Reduction

The modern halo-theory framework supplies a second formulation. Let

$$
\mathbf u=R^{-1}\mathbf s,
\qquad
D_P(\mathbf u)=\angle(\Phi_P(\mathbf u),\mathbf u).
$$

For a target direction at angular distance $\delta$ from the source, the SO(3)
fiber can be related to a scalar level set on the sphere:

$$
D_P(\mathbf u)=\delta.
$$

The remaining rotation about the source direction reconstructs the pose that
places the outgoing ray at the target azimuth. This suggests tracing contours
on $S^2$ instead of solving two constraints directly on SO(3).

Phase II will implement this as an independent algorithm, initially for the
same Phase I fixtures. Its first purpose is cross-validation:

- compare connected components found by both formulations;
- compare reconstructed poses and curve orientation;
- compare Jacobian factors after coordinate decomposition;
- detect components or singular behavior missed by either method.

Only after numerical equivalence is established should the project decide
whether the reduced formulation becomes an optimization backend, a theory and
diagnostics tool, or the primary renderer.

## 5. Proposed Responsibility Boundaries

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
   multiply, and is the authoritative implementation that the writing project
   now calls instead of maintaining its own copy.
2. **Differential evaluator**: derivatives with respect to local SO(3)
   coordinates, preferably from the same equations as the value evaluator.
3. **Fiber solver**: seed search, predictor-corrector continuation, component
   discovery, closure, and diagnostics.
4. **Integrator**: parameterization-aware line quadrature and error estimates.
5. **Image driver**: source, spectrum, pose distribution, pixel model, caching,
   and output assembly.

The boundaries should emerge from the first working slice. They are not a
request to build five frameworks before tracing one loop.

## 6. Relationship to Other Projects

### Lumice

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

### Modern Ice Halo Research Notes

The writing project motivates the solver and consumes its results, but does not
own its implementation. Lumice Integral should eventually regenerate the
direct-integration strip and state-space diagnostic from chapter 6, and provide
numerical evidence for the halo-map, Jacobian, fiber, and caustic discussion in
later chapters.

## 7. Validation Strategy

Validation must combine several independent kinds of evidence:

- analytically tractable maps and symmetry cases;
- local residual and tangent checks along every traced fiber;
- AD derivatives against finite differences at well-conditioned poses;
- quadrature refinement and reparameterization invariance;
- reproduction of the surviving historical direct-integration output;
- agreement with Lumice after Monte Carlo uncertainty, source model, spectrum,
  pose density, projection, and radiometric normalization are aligned;
- explicit convention fixtures at the validation boundary, without a shared
  geometry or optics implementation;
- Phase II cross-checks once the reduced formulation exists.

A smooth-looking image is not sufficient evidence: missing fiber components can
produce plausible but systematically wrong radiance.

## 8. Explicit Non-goals for the First Phase

- A general-purpose numerical integration or differential-geometry library.
- Replacing Lumice's Monte Carlo renderer.
- Multi-scattering scenes.
- Exact lower-dimensional orientation measures.
- A production GUI.
- GPU optimization before the CPU reference result is trustworthy.
- Any production dependency on Lumice or extraction of a Lumice engine API for
  this project.

## 9. Decisions to Date

- **2026-09-15**: create Lumice Integral as a repository separate from both
  Lumice and the writing project.
- **2026-09-15**: Phase I follows the original SO(3) continuation formulation so
  its behavior can be judged against the author's prototype experience.
- **2026-09-15**: the $S^2$ fiber-reduction formulation is deferred to Phase II
  and begins as an independent cross-check.
- **2026-09-16**: Phase I uses Python 3.12 and JAX as the durable float64
  reference implementation. GPU acceleration targets large batches rather than
  scalar continuation steps; C++/CUDA remains contingent on representative
  profiling. See [ADR 0001](decisions/0001-phase-i-python-jax.md).
- **2026-09-15**: Lumice Integral owns an independent differentiable geometry
  and optics implementation. Lumice remains an external Monte Carlo validation
  oracle and analysis-data source, never a production dependency or shared
  computational kernel.
- **2026-09-16**: the crystal geometry authority moves from the writing
  project's `halo_notes.geometry` into `lumice_integral.geometry` (migrated
  near-verbatim with public names kept 1:1). The writing project will call this
  package from now on, so the cross-repository relationship does not fork the
  geometry implementation. The subpackage stays pure numpy; splitting it into a
  separately installable unit (to avoid the parent package's JAX import) is
  recorded as a follow-up, not done here.
