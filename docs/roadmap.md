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

The milestone is complete when the program can (each item done; evidence in
`docs/phase1-math-contract.md` section 11 and the named tests):

- find one seed without a hand-entered pose — done: `lumice_integral.discovery`
  seeds every pixel from the scene prescan table
  (`test_canonical_pixel_has_one_closed_component`,
  `test_dark_pixel_has_no_admissible_candidate_and_is_procedurally_complete`);
  the frozen canonical seed is only a fixture constant;
- trace the complete known loop with bounded constraint residual — done:
  C05 (`test_synthetic_3_5_trace_matches_independent_direct_ray_oracle`,
  `test_synthetic_3_5_safe_step_sweep_converges_without_fixed_step_count`);
- detect closure without closing early — done: C06/C12
  (`test_analytic_conjugation_loops_shorter_than_pi_close_on_the_first_traversal`,
  `test_strip_short_loops_close_at_their_single_traversal_length`,
  `test_incompatible_tangent_cannot_pass_final_closure_correction`);
- plot the loop in diagnostic SO(3) coordinates — done as versioned figure
  data (`lumice-integral.figure-data/v3`,
  `test_figure_data_round_trip_preserves_geometry_and_unavailable_weights`),
  consumed by an independent prototype that drew the orientation-body-axis
  projection without importing this package (fixture specification section
  6); the chapter plotting code belongs to Writing-Lab;
- output the Jacobian, pose-density, geometric, and Fresnel factors separately —
  done: C14 (`test_canonical_pixel_fiber_exposes_four_available_factors_pointwise`,
  `J_perp` and `1 / (8 pi^2)` kept apart in `conventions`);
- integrate their product with a numerical convergence report — done: C14
  (`test_default_options_align_with_the_adaptive_reference_within_1e_4`,
  `test_node_count_doubles_until_the_estimate_meets_the_tolerance`,
  `test_canonical_pixel_integral_is_invariant_under_tolerance`: resampled
  fixed-grid quadrature with error estimate and grid evidence).

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
`strip_driver` / `strip_io`, `scripts/render_ch06_strip.py`) on the v2
per-pixel pipeline: one scene-level prescan table queried per pixel, the
integrated components of the pixel above as warm Gauss-Newton starts, one
production trace per distinct candidate, open arcs stitched from the two
half-traces of an event-terminated seed, resampled fixed-grid quadrature; a
per-pixel procedural completeness status layer; and the point pixel model
chosen by a caustic probe (the probe found `O(10-40 %)` point-vs-sub-pixel
differences within about seven rows of the `22 deg` inner edge at the centre
column, so the caustic band is a documented pixel-model limitation of the
default render). The full `251 x 801` image is rendered (`home-wsl`, `30`
workers, `1.86 h` wall clock then; `35 min` since the per-pixel
recompilations were removed, section 3.5 item 1): every pixel `complete`, no `unknown` band,
no open arc found on this scene. Compared with the historical raw on
multiplicative-bias-sensitive metrics: the row-band ratio slides
monotonically by `8.4x` from the inner edge to the bottom with no step (the
`x2` closure-gate step of the v1 preview is gone), the `10`-row inner-edge
offset is confirmed and lies on the historical side (the Lumice remake is
`3` rows further in than ours), and the vertical decay along the centre
column was `3-4x` steeper than the historical raw between rows `150` and
`600` (defect 2). Its first half is closed (2026-09-20, section 3.5 item
2a): the canonical crystal was half the height of the Lumice remake's, and
with `h / edge = 2` the centre column reproduces the historical plateau
(ours/historical relative to row `150` within `0.89-1.06` on rows
`150-400`, `0.55-1.0` before); the tail below row `400` is unchanged
(`x4` dark by row `600`, height-independent, `J_perp` confirmed by finite
differences to `1e-9`) and stays open together with the horizontal
narrowness off the centre column (rows `300-400` are still `0.5-0.7x` at
`+-20` columns; the non-tone-mapped Lumice profile is the next probe). The
evidence is recorded in the
[chapter 6 reference fixture specification](ch06-reference-fixture.md)
sections 5 and 7. Component completeness remains procedural, not certified;
finite solar disk and finite pixel solid angle are still open (the sub-pixel
model is implemented but costs 6-10x and is off by default).
Update (2026-09-20 / 2026-09-23): the tail and the off-centre narrowness were
closed on the historical side by the non-tone-mapped Lumice float export
(section 3.5 item 2b). **The renderer is delivered, and it agrees with an
independently converged Lumice Monte Carlo result.** In shape it agrees at
the Monte Carlo noise floor (lit-band log-RMS `0.03` against a `0.04-0.06`
run-to-run floor, task `lumice-raw-profile-oracle`). In absolute radiometry
it agrees with nothing fitted: `raw / emitted_energy = K_p V` with
`K_p = 12 ybar(550) Omega_p / A_eff`, which gives measured over predicted
`0.997-0.999` on the bright band of columns `106 / 126 / 146` at matched
refractive index (task `phase1-closeout-absolute-scale`, fixture
specification section 7, stage 4). This closes the first image-level target.
Outside the point-source point-pixel model the remaining open items are
unchanged: a completeness certificate, finite solar disk, and pixel
averaging in the caustic band.

### 3.4 Orientation distributions

Phase I assumes that $\rho$ is an ordinary, possibly narrow, density on all of
SO(3). Plate-like and column-like preferences therefore change the integrand but
do not change the one-dimensional fiber.

Exactly constrained ideal orientation families are singular measures supported
on lower-dimensional subsets of SO(3). Supporting them requires a generalized
domain and a different dimension count; this is explicitly outside the first
reconstruction milestone.

### 3.5 Near-term plan (2026-09-18, after scrum `strip-pipeline-v2`)

The writing series' chapters 6-11 fix what the solver must deliver next; the
order below follows their dependencies, not the solver's own curiosity.

1. **Per-pixel cost** (task `pixel-cost-shape-stable-kernels`, done
   2026-09-20): the strip's real single-process cost was `0.29 s` per pixel,
   not the `0.067 s` of the warm single-pixel benchmark, because candidate-pool
   and curve sizes change from pixel to pixel and every change recompiled the
   XLA kernels behind `discovery` and the resampling (`1065-1112` compilations
   for `60` pixels). The ruler is now `benchmarks/benchmark_column_steady_state.py`
   (one column single-process, steady-state s/px plus compile count; the
   single-pixel `probe_step7_timing.py` figure is a hot-cache lower bound).
   Pool clustering and curve dedup use the numpy batch `so3.rotation_distances`
   (same formula as `rotation_distance`, ulp-locked by test) and the knot
   quaternion batches in `resample.py` are compiled once per power-of-two
   bucket: `101` compilations for the same `60` pixels (`143` for a full
   column, `6` pixels compile at all), steady state `0.066-0.11 s` per pixel
   on the M2 Max (`0.17 s` on `home-wsl`). Worker probe on `home-wsl`
   (Ryzen 9950X, 16C/32T): total throughput `5.8 / 19 / 31 / 48 / 100 / 95`
   px/s at `1 / 4 / 8 / 16 / 30 / 32` workers, per-worker rate falling
   from `5.8` to `3.5` px/s with SMT sharing, not compile threads (after the
   fix a worker is `1.06` cores, `llvm-worker` threads idle past the first
   pixel; no persistent compile cache or thread cap needed); the knee is the
   logical CPU count, `30` workers recommended. Full `251 x 801` rerender:
   `2085 s` (`35 min`) wall clock, `3.2x` faster than `1.86 h`, every
   pixel's status bits and component count identical to
   `artifacts/strip-full`, values within `8.1e-13` relative. The `<= 15 min`
   target is missed by `2.3x`: the per-pixel cost is at the hot-cache floor,
   so the remaining levers are algorithmic (trace and quadrature work per
   pixel, out of this task's scope) or more physical cores.
   `continuation.py:_rotation_distance_kernel` (fixed `(3,3),(3,3)` shape,
   one compilation per process) is outside this root cause and stays as is.
2. **Defect 2** (rows `300-650` decay; section 3.3) in two halves. (a) The
   canonical crystal is half as tall as the Lumice remake's: Lumice's
   `height: 1.0` is `h / diameter` (side planes at inradius `√3/4`, basal at
   `±h/2`), i.e. `h = 2 x edge`, while `CANONICAL_HEIGHT_RATIO = 1.0` is
   `h / edge`; the `column1.0` filename was read as `historical-direct` but
   is convention-dependent. Rendering column `126` with `h / edge = 2`
   reproduces the historical plateau of rows `175-400` within `+-6 %`
   (`0.55-0.84` before), taller crystals overshoot. Done 2026-09-20
   (task `defect2-crystal-height-convention`): `CANONICAL_HEIGHT_RATIO =
   2.0`, baselines re-pinned (the fiber, `J_perp` and every discovery count
   are crystal-independent; only `entry_measure` and the pixel values moved,
   canonical `2.364 -> 6.581`), full `251 x 801` rerender on `home-wsl`
   (`30` workers, `2030 s`): column `126` ours/historical relative to row
   `150` is `0.89-1.06` on rows `150-400` (`248` of `251` rows inside
   `0.9-1.1`, the three below are rows `398-400` at `0.886-0.896` where the
   tail decay starts; `0.55-1.0` with `h / edge = 1`), per-50-row
   max-normalised ratios `1.03 / 1.02 / 1.06 / 1.10 / 1.04` (rows
   `150-400`; `0.93 / 0.88 / 0.83 / 0.75 / 0.61` before), the tail
   `400-650` `0.78 / 0.49 / 0.33 / 0.28 / 0.26` is essentially the old
   `0.44 / 0.31 / 0.26 / 0.24 / 0.23` from row `500` on. The plateau is a
   centre-column result: at `+-5` columns `93-96 %` of rows `150-400` are
   inside `0.9-1.1`, at `+-10` columns `68-82 %`, at `+-20` columns
   `37-39 %` (rows `350-400` at `0.52-0.57`, the old centre-column
   signature), i.e. the horizontal narrowness of rows `150-400` is only
   partly closed (row `150` now matches to `+-7 %` over `+-20` columns,
   row `300` is still `0.6-0.8x` there; lit width above `1e-3` at row
   `300` `129` columns against the historical `157`, `121` before).
   Whole-image Spearman against the historical raw fell from `0.985` to
   `0.949` while the Lumice-remake figure rose from `0.947` to `0.967`;
   log-RMS along column `126` `0.443 -> 0.389`, row `150` `0.487 -> 0.447`,
   row `300` `0.688 -> 0.542`, row `450` `0.124 -> 0.224` (the turnover
   row). Not tuned further here: the residual is the tail plus the
   off-centre narrowness, both for (b). (b) The tail, rows
   `475-650`, is independent of crystal height (identical values from
   `h / edge = 1` to `20`) and stays `4x` below the historical raw. Closed
   2026-09-20 (task `lumice-raw-profile-oracle`): the non-tone-mapped
   `1e9`-ray Lumice float export agrees with this renderer at the Monte
   Carlo noise floor on columns `106 / 126 / 146` and rows `150-600`
   (lit-band log-RMS `0.03` against a `0.04-0.06` run-to-run floor), and
   the historical raw is the outlier of the three (`0.39-0.41`): the tail,
   the off-centre narrowness and the inner-edge offset are all properties
   of the historical raw. **Ruling (author, 2026-09-20): the historical
   raw is no longer a correctness reference for this renderer, and its
   remaining discrepancies are not traced further; agreement with the
   Lumice Monte Carlo result is the criterion.** The historical raw stays
   in the fixture only as a morphology / provenance record (section 3.3).
   The absolute radiometric scale between the two, unchecked here (every
   profile above is max-normalised), was checked 2026-09-23 (task
   `phase1-closeout-absolute-scale`): `PBD x12` is confirmed as one scalar,
   and Lumice's pose sampling without silhouette weighting adds a
   pixel-dependent `A_eff` (fixture specification section 7, stage 4). Path-class
   accounting is *not* a suspect here: the `PBD` orbit of `3-5` adds only
   `3-7`, whose image is identical under the zenith-symmetric density (the
   prism's `C2'` rotation maps one to the other), a uniform `x2`; all three
   images are left-right symmetric to `1e-3`.
3. **Path classes as the rendering unit** (writing chapters 7-9): a halo is a
   conjugacy class of paths, not one representative path (a single
   representative can carry `0.4 %` of its class), and comparisons with
   Lumice under `PBD` or with the notes' 34-class table are class-level. The
   driver should take a signature class, enumerate its `PBD` orbit with
   `lumice_integral.geometry`, trace each member and sum. Rank-0 classes
   (`W = 0`: a point mass in the sun direction, estimated from the prescan
   table's Haar samples, never traced) come with it. Independent of defect 2.
   *Done 2026-09-20* (task `path-class-rendering-unit`, PR #10):
   `optics` / `prescan` / `discovery` / `weights` take any face sequence, and
   `lumice_integral.path_class` expands a signature class into its `PBD` orbit
   with `lumice_integral.geometry`, traces every member and sums, with
   per-member contributions in provenance. Under the canonical column
   density the class `[3,5]` has `12` members (`6` rotations each of `3-5`
   and `3-7`, pointwise identical), and the class value is `12x` the single
   `3-5` (`11.99982` measured). A tilted Parry density (`roll_mean_deg = 20`)
   gives the strict counterexample, `3-5` non-zero and `3-7` exactly zero.
   Rank-0 classes (`1-2`, `3-6`, ...) never enter the fiber pipeline and
   become a Haar-mean point mass in the sun direction (`0` in the strip,
   which does not contain the sun). `geometry.unfold.halo_map_rank` and
   `wedge_angle_deg` are path-level properties. `3-1-2-5` (the `M = I` `60 deg`
   wedge with one internal reflection) traces an arc at row `651` column `13`
   whose value is `3.64x` the `3-5` value of the same pixel, and every
   accepted pose maps to the same target through the `3-5` direction map
   (residual `< 1e-8`). A one-member class reproduces the single-path `3-5`
   baseline bit for bit. Not done: a full-image class-level rerender and the
   notes' full `Phi`-class machinery (34-class table, `D6h` reflection
   group).
4. **Phase II as the chapter-10 tool**: `D_P(u)`, its Jacobian and rank, the
   fold caustic and the `I ~ 1 / sqrt(D - D_min)` radial profile are the
   objects chapter 10 needs; section 4's cross-checks against the SO(3) fiber
   are the acceptance. The renderer-backend question stays deferred.
5. **Pose-density families** (chapter 11): plate / column / Parry / Lowitz /
   random with a width parameter, replacing the single zenith-Gaussian column
   model; the fiber is unchanged, only the integrand. This also turns the
   `0.5 deg` guess of section 3.3 into a swept parameter. *Done 2026-09-20*
   (`docs/ch11-pose-density-families.md`): `pose_density.build_pose_density`
   covers the five families with three classes (Haar-uniform; zenith
   Gaussian; zenith Gaussian times a roll-locked Gaussian, roll from the
   Lumice chain `R = Rz(az - pi) Ry(-zenith) Rz(roll)`), every family
   normalised against Haar by independent quadrature, provenance recorded
   per family, and the `0.5 deg` width sweep reproduces the defect-2 table.
   Finding: on the labelled path `3-5` the ch06 strip is a column/random
   strip — plate, Parry and Lowitz put their `3-5` light elsewhere on the
   sky (parhelion, upper Parry arc, Lowitz arcs) and are exactly zero on
   column `126`; their strip contributions come from other labelled paths of
   the class, i.e. item 3. The family parameter reached the production CLI
   on 2026-09-23 (`scripts/render_ch06_strip.py --pose-density-family` with
   the width and mean flags, recorded in `provenance.json`; task
   `phase1-closeout-absolute-scale`). Still open: wrapped (vs single-period)
   roll Gaussian, Lowitz `zigzag`.

Deferred unchanged: pixel-space adaptive sampling, GPU kernels, finite solar
disk (chapter 10's singularity is the point-source one).

**Phase I closed (2026-09-23).** Every section 3.2 milestone is done, and the
section 3.3 renderer is delivered and agrees with Lumice in shape and in
absolute scale. Items 1, 2, 3 and 5 above are done; item 4 moves to the
Phase II scrum. The open items that remain (completeness certificate, finite
solar disk, caustic-band pixel averaging, the explicit event localisation of
contract section 12) are recorded, not blocking. Closeout record:
`scratchpad/task-phase1-closeout-absolute-scale/SUMMARY.md`.

### 3.6 Writing-project coupling

- W1 `geometry-depend-on-lumice-integral` (writing repo task 16): the notes
  depend on this package (path dependency, Python `3.12`), delete their
  `halo_notes.geometry` copy, and pass `pbd_orbit` as the `symmetry_orbit`
  callback. Prerequisite here: `lumice_integral/__init__.py` must not import
  JAX eagerly, so that `import lumice_integral.geometry` stays numpy-only.
- W2: the notes consume figure-data v3 for the chapter-6 state-space figure;
  the chapter-6 strip remake waits for defect 2 and the rerender.

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

The level-set integral has two discretisations: tracing the contours
(section 4.1) and summing precomputed $S^2$ events per deviation band
(section 4.2). They are two quadratures of one integral, not alternatives.

### 4.1 Design findings (2026-09-20 discussion, before the Phase II scrum)

These are derived statements, not yet numerically verified unless a test is
named; the Phase II scrum turns each into a fixture or a design constraint.

**(a) Every geometric and optical weight is a function on $S^2$, not on
SO(3).** `entry_measure` reduces the pose to `s_body = R.T @ incident`
$= \mathbf u$ on its first line and uses nothing else; the TIR gates and the
Fresnel factors depend on the incidence angles, hence on $\mathbf u$ only.
Rotating the crystal about the sun direction (the fiber coordinate $\psi$)
changes none of them. So $A_P$, $T_P$, the validity gates and the feasible
domain $V_P$ are fields on the base $S^2$ of the fibration
$R \mapsto \mathbf u = R^{-1}\mathbf s$; only $\rho$ sees the full pose.
Phase I evaluates them point by point along an SO(3) curve because it cannot
see this structure; Phase II lives on exactly that $S^2$.

**(b) The domain is a precomputed map, and open arcs are a support
question.** For one (crystal, path) the pair $(V_P, A_P)$ is computed once
with the existing `entry_measure` (any $R$ taking $\mathbf u$ to
$\mathbf s$ will do), independent of pixel and of pose density. The pixel
value is a line integral along the level set of the deviation field
(coarea on $S^2$, Haar $= dA(\mathbf u)/4\pi \cdot d\psi/2\pi$):

$$
I(\delta,\alpha)\,\sin\delta \;\propto\;
\int_{D_P=\delta}\frac{\rho\big(R(\mathbf u,\psi(\mathbf u,\alpha))\big)\,
A_P(\mathbf u)\,T_P(\mathbf u)}{|\nabla_{S^2}D_P(\mathbf u)|}\,d\ell .
$$

The integrand vanishes continuously on every boundary: corridor boundaries
(two polygons separating, $A_P \to 0$ continuously), the exit-face TIR
boundary of the formula domain $U_P$ (Fresnel transmittance $\to 0$ at the
critical angle), no critical angle on entry, and partial reflection on the
internal steps is a continuous weight, not a boundary. A contour cut by
$\partial V_P$ therefore needs no event handling: trace it on $U_P$ and let
$A_P T_P$ remove the infeasible part. Phase I's rule "keep tracing, weight
to zero" (contract section 6.3) is the same fact placed inside the tracer.
The only non-smoothness left is the kinks of $A_P$ (a vertex crossing an
edge), which lower the quadrature order locally, as in Phase I.

**(c) Topology and completeness.** Level sets of a scalar field are governed
by its critical points: $\nabla D_P = 0$ (finitely many, found by AD Newton
from grid seeds) plus the critical points of $D_P|_{\partial U_P}$ split the
$\delta$ axis into intervals on which the level-set topology is constant.
Marching once per interval and Newton-refining gives *every* component, so
the completeness certificate that Phase I cannot issue (contract C11,
`completeness` is procedural) becomes a checkable statement. This is the
larger gain of Phase II; the speed is the smaller one.

**(d) Layered invariance (what a halo shares and what varies).** The fiber
of a pixel, $\{R : R\,\Phi_P(R^{-1}\mathbf s) = x\}$, depends on the path
only through $\Phi_P$. Hence:

| layer | object | shared by |
|---|---|---|
| $\Phi$ | the field $D_P$, its contours, $1/\lvert\nabla D_P\rvert$, the correspondence $\psi(\mathbf u,\alpha)$, the critical points | the whole $\Phi$-class, across PBD classes |
| member | the window field $w_m = A_m T_m$ on $S^2$, additive: $w_\Phi = \sum_m w_m$ | one per face sequence |
| symmetry | a proper crystal symmetry $g$ moves the map: $D_{gPg^{-1}}(\mathbf u) = D_P(g^{-1}\mathbf u)$, $\mathrm{fiber}(gPg^{-1}) = \mathrm{fiber}(P)\,g^{-1}$ exactly, windows transported alike | only $\rho(Rg^{-1})$ can tell the members apart |
| $\rho$ | the pose density on the fiber | the *columns* of the writing series' table |

So a $\Phi$-class is one contour family plus one effective window field, and
the writing series' core table (rows = path classes, columns = pose
families) has the skeleton row $= (D_P, w_\Phi)$, column $= \rho$, cell
$=$ the line integral. Task 9's `12x` for the column density is the symmetry
row with a $\rho$ invariant under the $C_6$ rotations and $C_2'$; the tilted
Parry density breaks it through $\rho$ alone. Improper elements (B/D
mirrors) do not transport inside SO(3) and are traced separately. The same
statements hold in Phase I: members with the same $\Phi$ share one SO(3)
curve (task 9 measured "same direction map, different weight" for `3-1-2-5`
on row 651), so class rendering should trace once per $\Phi$-group and sum
the windows — task `path-class-shared-fiber` (tasks.md 12).

**(e) Fixtures this suggests.**
- *Liljequist* (writing chapter 8): `1-3-2` and `3-5-6-7-3` have the same
  $\Phi$ (the mirror in the plane of faces 3/6; three reflections in planes
  at $\pm 60°$ compose to one) and different windows — the `142°` sharp edge
  is the $D_P$ fold and is shape-independent, the narrow peak is the
  `3-5-6-7-3` window and moves with the cross-section. Two pictures on one
  sphere.
- *Parhelic circle*: $D_P(\mathbf u) = \angle(M\mathbf u, \mathbf u)$ has
  $\nabla D_P = 0$ only at $\pm\mathbf n_M$, so the ring has **no fold**;
  its brightness along the ring is entirely the window layer. For plates the
  ring azimuth is linear in the crystal azimuth, so the profile is a sum of
  shifted copies of one window function (three mirror planes of the prism).
  A clean test of the window-sum and symmetry-transport layers without the
  Jacobian in the way.
- *22° halo*: the fold; see the caution below.

**(f) Design constraints carried over from the Phase I cost profile**
(`scratchpad/task-pixel-cost-shape-stable-kernels/evidence/owner_cprofile_col126_rows300-340.prof`):
`74 %` of a lit pixel is the continuation loop, `~95` corrector trials at
`~0.9 ms` each of which the arithmetic is $3 \times 3$; the cost is
per-step Python orchestration and small-kernel dispatch, and 30 workers on
`home-wsl` sit at the physical-core wall (`100` px/s, `35 min` per image).
Phase II must be batched, branch-free and `vmap`-able from the first
design: field evaluation on an $S^2$ grid, contour extraction and
quadrature as array programs, so that a full image is a field computation
and a GPU becomes usable. A per-pixel Python loop would reproduce the same
wall in a new place.

**(g) Open points to settle numerically, not by argument.**
- For a random orientation the minimum of $D_P$ on $S^2$ is isolated and
  non-degenerate, and $\int d\ell/|\nabla D|$ near a two-dimensional minimum
  is finite: the inner edge would be a finite jump, and the
  $I \sim 1/\sqrt{D - D_{\min}}$ fold profile would belong to pose families
  that confine $\mathbf u$ to a curve (columns, tangent arcs). Chapter 10's
  statement should be an acceptance test of the 22° cross-check, not a
  premise.
- Rank-deficient maps: $W = 0$ classes are point masses (task 9); the
  degenerate images of parallel-face classes ($M \ne I$, $W = I$) come from
  $\rho$ confining $\mathbf u$, not from $\Phi$, and need their own
  accounting.
- Non-uniform $\rho$: $\psi(\mathbf u,\alpha)$ is single-valued, so $\rho$
  is evaluated pointwise; only the "convolution on the sky" reading of
  chapter 11 needs uniform $\rho$.
- The Jacobian alignment $1/|\nabla_{S^2} D_P|$ against Phase I's
  $J_\perp$ under the fibration's coordinate change is the cross-validation
  contact point (section 4, third bullet).

### 4.2 Band-sum quadrature (precomputed $S^2$ events)

Source: the Ice Halo Simulation repository,
`doc/research/inverse-rendering.md` (Chinese: `inverse-rendering_zh.md`),
"Inverse Rendering via Precomputed Standard Events", after Gislén et al.
(2004). That note is a design sketch; this section is the authoritative
statement of the idea for this project, with the corrections below. The probe
is `scripts/probe_band_sum.py` (task `band-sum-quadrature-probe`).

**Correspondence.** The note's objects are the section 4.1 fields:

| note | this roadmap |
|---|---|
| standard event $(\hat a_0, \hat b_0)$ | $(\mathbf u, \Phi_P(\mathbf u))$, $\mathbf u = R^{-1}\mathbf s$ |
| scattering angle $\omega$ | deviation field $D_P(\mathbf u)$ |
| event weight $w$ | window field $A_P(\mathbf u)\,T_P(\mathbf u)$, section 4.1(a) |
| rotation $U$ of eq. 19 | the pose $R(\mathbf u, \psi(\mathbf u,\alpha))$ |
| $Q(U)$ | $\rho(R)$ relative to Haar probability |

**Coarea equivalence and the normalisation.** Split Haar probability as
$d\mu_{\mathrm{Haar}} = dA(\mathbf u)/4\pi \cdot d\psi/2\pi$ ($\psi$ the
twist about $\mathbf s$). The twist moves the outgoing direction rigidly
about $\mathbf s$, so at fixed $\mathbf u$ the outgoing azimuth is
$\alpha = \alpha_0(\mathbf u) + \psi$ and $d\psi = d\alpha$, while the
deviation stays $D_P(\mathbf u)$. Pushing $\rho A_P T_P\, d\mu_{\mathrm{Haar}}$
forward to the sky and writing $dA(\mathbf d) = \sin\delta\, d\delta\, d\alpha$
gives the Phase I pixel value (contract section 7, the same $1/(8\pi^2)$) as

$$
I(\delta,\alpha)\,\sin\delta
= \frac{1}{8\pi^2}\int_{S^2}\delta_{\mathrm{Dirac}}\big(D_P(\mathbf u)-\delta\big)\,
  \rho\,A_P T_P\,dA(\mathbf u)
= \frac{1}{8\pi^2}\int_{D_P=\delta}
  \frac{\rho\,A_P T_P}{|\nabla_{S^2}D_P|}\,d\ell ,
$$

which is section 4.1(b) with the proportionality made explicit. There is no
separate $d\psi/2\pi$ factor at a fixed pixel: $\psi$ is consumed by
$d\psi = d\alpha$ and $\rho$ is evaluated at the single-valued
$\psi(\mathbf u,\alpha)$. Integrating over a band
$\delta' \in [\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}]$ at the pixel's
$\alpha$ turns the line integral into an area integral on $S^2$; with $N$
equal-area points ($4\pi/N$ each) the band sum is

$$
\hat I(\delta,\alpha)
= \frac{1}{2\pi N\,\Delta\delta\,\sin\delta}
  \sum_{i:\,D_P(\mathbf u_i)\in[\delta_{\mathrm{lo}},\delta_{\mathrm{hi}}]}
  A_P T_P(\mathbf u_i)\;\rho\big(R_i\big),
\qquad \Delta\delta = \delta_{\mathrm{hi}}-\delta_{\mathrm{lo}},
$$

an estimate of the band average of $I \sin\delta'$ divided by $\sin\delta$.
$R_i$ is the unique pose with $R_i\mathbf u_i = \mathbf s$ and
$R_i\Phi_P(\mathbf u_i)$ at deviation $D_P(\mathbf u_i)$ *and* the pixel's
azimuth: $R_i = [\mathbf s, \mathbf e, \mathbf s\times\mathbf e]\,
[\mathbf u_i, \mathbf f_i, \mathbf u_i\times\mathbf f_i]^{\mathsf T}$ with
$\mathbf e$ the unit azimuth direction of the pixel about $\mathbf s$ and
$\mathbf f_i$ the unit component of $\Phi_P(\mathbf u_i)$ normal to
$\mathbf u_i$. This is eq. 19 evaluated at $\omega = D_P(\mathbf u_i)$ (the
probe checks them equal to `7e-15`), without its $1/\sin^2\omega$. Feeding
eq. 19 the pixel's own $\delta$ instead, as the note's step 3 reads, gives a
non-orthogonal matrix whenever $D_P(\mathbf u_i) \ne \delta$
($\lVert U^{\mathsf T}U - I\rVert_F$ median `5.5e-4`, max `1.4e-3`, on
the column-126 pixels, the size of the band width `4.1e-4`) and $\rho$ of a non-rotation; the
event's own deviation is the one the coarea identity asks for. The constant
$1/(2\pi N)$ matches task 11's convention (explicit Haar $1/(8\pi^2)$, no
hidden reference area) and is used as derived, never fitted.

**Division of labour.** Contour tracing (section 4.1) owns accuracy and the
completeness certificate of 4.1(c); the band sum has neither a certificate
nor a convergence order beyond sampling, but it is one sort, one range query
and one batched $\rho$ evaluation per pixel: branch-free and `vmap`-able as
4.1(f) demands, with the event store shared by every pixel, pose density and
sun elevation (the store is built from $\mathbf u$ in the body frame; a new
sun direction only changes the rotation per pixel, section 4.1(b)). It suits
fast rendering, parameter sweeps and independent cross-checks of the contour
renderer.

**Corrections to the note.**
1. *Not noise-free.* The band sum is deterministic but carries a
   discretisation error: about $1/\sqrt{K_{\mathrm{eff}}}$ for scattered
   points, aliasing for a regular grid (a Fibonacci lattice cut by a thin
   band behaves like scattered points, measured below). "No match means the
   true value is dark" fails under a narrow $\rho$: a pixel whose band
   holds events with $\rho \approx 0$ reads zero whatever the truth.
2. *Narrow $\rho$ is the structural weak point.* The band's event set is
   fixed by $\delta$ alone; $\rho$ selects the part of it that contributes,
   so the effective count $K_{\mathrm{eff}}$ collapses when $\rho$ is
   non-zero on a short stretch of the contour. Oriented-crystal halos
   (parhelia, tangent arcs) are the cases of most interest. How far it
   collapses depends on the pixel, not only on the width of $\rho$: see the
   measurements.
3. *Normalisation must be explicit*: $I\sin\delta$, Haar
   $dA(\mathbf u)/4\pi\cdot d\psi/2\pi$, the constant above; the note's
   "divide by the number of incident directions" is the $1/N$ of it.
4. *"Caustics regulate their own brightness"* is subject to 4.1(g): for a
   random orientation the 22° inner edge may be a finite jump, not a
   $1/\sqrt{\,}$ profile. An acceptance test, not a premise.
5. *Organise the store by $\Phi$-class* (4.1(d)), not one array sorted by
   $\omega$ with `raypath_id` mixed in: members of a $\Phi$-class share
   $D_P$ and add windows, and symmetry transport acts per class.

**Divergent light (deferred).** The note's ray-marching extension (a nearby
source; $\omega$ varies along each view ray, one range query per step with
inverse-square weights) reuses the same event store and fits this project,
but comes after Phase II. Multiple scattering stays a non-goal (section 8).

**Probe results and verdict (2026-09-23).** Path 3-5, canonical crystal
($h/a = 2$), $n = 1.31$, sun altitude 15°, the ch06 strip camera; Fibonacci
lattices of $N = 10^6, 10^7, 5\times10^7, 10^8$ points; the band of a pixel is
the min/max deviation of its four corners (about `4.1e-4` rad, one pixel).
Lit band = reference above `1e-2` of the column maximum. Errors are relative
to the Phase I strip (`artifacts/strip-full`, point pixel model) for the
column density and to Phase I `render_pixel` recomputed with the random
density (62 rows of column 126, all `complete`).

| scene | $N$ | median $K_{\mathrm{eff}}$ (lit) | $K_{\mathrm{eff}}/K$ | RMS rel. error | median \|rel.\| | lit-band sum ratio |
|---|---|---|---|---|---|---|
| column, col. 126 (801 px) | $10^6$ | 151 | 0.65 | 6.8e-2 | 3.7e-2 | 0.9986 |
| | $10^7$ | 1518 | 0.65 | 1.6e-2 | 7.1e-3 | 0.9986 |
| | $5\times10^7$ | 7583 | 0.65 | 6.5e-3 | 2.3e-3 | 0.9986 |
| | $10^8$ | 15153 | 0.65 | 5.0e-3 | 1.2e-3 | 0.9985 |
| random, col. 126 (62 px) | $10^6$ | 169 | 0.90 | 6.3e-2 | 3.5e-2 | 1.0095 |
| | $10^7$ | 1731 | 0.90 | 1.6e-2 | 7.7e-3 | 1.0017 |
| | $5\times10^7$ | 8549 | 0.90 | 5.8e-3 | 2.5e-3 | 0.9996 |
| | $10^8$ | 16993 | 0.90 | 3.9e-3 | 1.5e-3 | 0.9998 |
| column, cols. 26/76/176/226 (404 px) | $10^7$ | 295 | 0.17 | 5.9e-2 | 2.6e-2 | 0.9960 |
| | $10^8$ | 3001 | 0.17 | 1.5e-2 | 5.6e-3 | 1.0001 |

- *Absolute scale.* The derived $1/(2\pi N\,\Delta\delta\sin\delta)$ is
  right without fitting: median ratio `1.0000` (column) and `0.9992`
  (random) at $10^8$. The column's lit-band sum ratio `0.9985` is one
  pixel: row 57, the inner-edge caustic, reads `-9.9 %` at every $N$
  because its band starts `0.0023°` below $\min D_P = 21.8393°$, so a tenth
  of the band is dark. That is the band sum's pixel model (a band average)
  against the reference's point pixel, not sampling error; without rows
  47-67 the sum ratio is `1.000003` and the RMS error `2.5e-3` at $10^8$.
- *Convergence.* $K_{\mathrm{eff}}$ grows as $N^{1.00}$ in all scenes and
  the error as $N^{-0.57}$ to $N^{-0.61}$: the Fibonacci lattice cut by a
  thin band behaves like scattered points ($1/\sqrt{K_{\mathrm{eff}}}$),
  with no aliasing seen.
- *$N$ for a `1e-2` lit-band RMS*: $2.6\times10^7$ for column 126 and
  $2.1\times10^7$ for random (power-law fits over the four tiers; both
  measured below `1e-2` at $5\times10^7$), about $2\times10^8$ for the four
  other columns (median error already `7.5e-3` at $5\times10^7$).
- *Narrow $\rho$ is pixel-dependent, and column 126 is its easy case.*
  Refraction by the 3-5 prism wedge preserves the ray component along the
  prism edge, so every pose of a pixel has $\mathbf c \perp
  (\mathbf b - \mathbf s)$; on the sun's vertical (column 126) this puts the
  c axis within about ±1.2° of horizontal along the whole contour, and the
  column density keeps 65 % of the band ($K_{\mathrm{eff}}/K$; random:
  90 %, the rest being the spread of $A_P T_P$). Off the vertical it keeps
  17 % (median; worst pixel $K_{\mathrm{eff}} = 4.5\times10^{-6}N$). The
  owner's prior $K_{\mathrm{eff}} \approx 2\times10^{-6}N$ is 75× low for
  column 126 and 15× low for the other columns' median, and close to their
  worst pixel. Densities narrow in two directions (roll-locked Parry,
  Lowitz) were not measured and can collapse further.
- *Cost.* Precompute (production batch evaluators, 250k chunks, one core):
  `1.1 s` at $10^6$, `8.5 s` at $10^7$, `40 s` at $5\times10^7$, `79 s`
  at $10^8$, peak RSS `3.2 GB`, 16.0 % of the points kept
  ($A_P T_P > 0$). Rendering at $10^8$: `3.3 ms` per pixel for the column
  density (`2.7 s` for 801 pixels), against `0.11-0.17 s` per pixel for
  Phase I single-process. The whole probe took under four minutes of
  compute; the 2 h budget and the stop-loss rule (`N > 1e8` for `1e-2`)
  were not reached.
- *Self-checks.* Validity, $A_P$, $T_P$, $\Phi_P$ and $D_P$ unchanged under
  three twists about $\mathbf s$ to `1.5e-14` (section 4.1(a)); all three
  `entry_measure` failure reasons occur on the sphere; the Fibonacci mean
  of $A_P T_P$ matches `1e6` independent Haar rotations ($z = 1.35$); the
  frame construction equals eq. 19 at $\omega = D_P$ to `7e-15`.

Verdict: the band sum qualifies as a **rendering backend** for both pose
densities measured, at $N \approx 10^8$ (`80 s` of precompute, reused
across pixels, densities and sun elevations), not only as a cross-check.
It does not replace the contour method: it has no completeness certificate,
its pixel is a band average (the caustic edge differs from a point pixel by
the dark fraction of the band), its accuracy is pixel-dependent through
$K_{\mathrm{eff}}$, and doubly-locked densities are untested. The contour
method stays the accuracy and completeness authority; the band sum becomes
its fast renderer and independent cross-check. Figures and tables:
`scratchpad/task-band-sum-quadrature-probe/artifacts/` (local).

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
   multiply, and is the designated authoritative implementation; the writing
   project still carries its own copy until its task
   `geometry-depend-on-lumice-integral` (W1) switches it over (section 3.6).
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
- **2026-09-18**: the writing project depends on this package directly (no
  separate geometry package): chapter 10 needs the halo map and Jacobian from
  here anyway, so the dependency is unavoidable and a third package would only
  add a repository. The parent package drops its eager JAX import instead.
- **2026-09-18**: near-term order per section 3.5 — per-pixel cost, defect 2
  (crystal height convention, then the height-independent tail), path classes
  as the rendering unit, Phase II as the chapter-10 tool, pose-density
  families. Path-class accounting is decoupled from defect 2.
- **2026-09-20**: canonical crystal `h / edge = 2` (Lumice `height 1.0` is
  `h / diameter`; section 3.5 item 2a); the `column1.0` filename is
  convention-dependent, so the ratio is `canonical-new`, not historical
  evidence. Baselines recorded on the `h / edge = 1` crystal by tools that
  cannot re-record (the retired adaptive integrator, the independent zenith
  width probe) keep that crystal explicitly in their tests; the `h / edge =
  1` full render is kept as `artifacts/strip-full-h1`.
- **2026-09-20**: pose-density families (section 3.5 item 5) implemented;
  the roll-locked families need the crystal's spin about its c axis, taken
  from the Lumice ZYZ chain with `roll = 0` = face-3 normal in the vertical
  plane through the c axis (this renderer's own reference, not Lumice's mesh
  numbering). Provenance grows a trailing `family` key; the column block is
  otherwise unchanged.
- **2026-09-20**: defect 2 closed by the Lumice float oracle (section 3.5
  item 2b): this renderer and Lumice agree at the noise floor, the
  historical raw is the outlier. Author's ruling: the historical raw is no
  longer a correctness reference and is not traced further.
- **2026-09-23**: the Ice Halo "precomputed standard events" note enters as
  the second Phase II discretisation (section 4.2), the band sum over $S^2$
  events, a coarea dual of the contour integral with the explicit constant
  $1/(2\pi N\,\Delta\delta\sin\delta)$. Probe verdict: rendering backend
  for the column and random densities at $N \approx 10^8$ (lit-band error
  `5e-3` / `4e-3` on column 126); the contour method keeps accuracy and the
  completeness certificate. The store is organised by $\Phi$-class, and
  divergent light is deferred.
