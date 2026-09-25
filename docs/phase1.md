# Phase I: Tracing Pose Fibres on $\mathrm{SO}(3)$

Chinese version: [phase1_zh.md](phase1_zh.md) (the measured record of the
appendix is kept in English only).

Phase I is the reconstruction of the author's original direct-integration
prototype: for every pixel, find the one-dimensional set of crystal poses
that send the incident ray to that pixel, and integrate the physical weight
along it. It was closed on 2026-09-23. It is the project's pointwise
reference; the production renderer is now Phase II's band sum
([phase2.md](phase2.md)). The normative definitions (coordinates, measures,
events, interfaces, conformance C01-C14) are in
[phase1-math-contract.md](phase1-math-contract.md); the chapter-6 fixture
and its acceptance stages are in
[ch06-reference-fixture.md](ch06-reference-fixture.md). This document is the
design and its history; the measured record is the appendix.

## 1. The formulation

Fix a crystal, a ray path $P$, a wavelength and the incident propagation
direction $\mathbf s$ (from the sun toward the crystal; the direction toward
the sun is $\hat{\mathbf s} = -\mathbf s$, [conventions.md](conventions.md)).
A crystal pose is a rotation $R \in \mathrm{SO}(3)$, and ray propagation
defines a halo map

$$
F_P : \mathrm{SO}(3) \longrightarrow S^2,
\qquad R \longmapsto \text{outgoing direction}.
$$

For an image direction $\mathbf d$ the contributing poses are the inverse
image $X_{\mathbf d} = F_P^{-1}(\mathbf d)$. At a regular value the dimension
theorem gives $\dim X_{\mathbf d} = 3 - 2 = 1$: the integration domain is a
set of curves in $\mathrm{SO}(3)$, often closed loops. With a smooth pose
density the pointwise contribution of a path has the coarea form

$$
I_P(\mathbf d)=
\int_{X_{\mathbf d}}
\frac{\rho(R)\,A_P(R)\,T_P(R)}{J_{F_P}(R)}\,d\mathcal H^1(R),
$$

with $\rho$ the pose density relative to Haar, $A_P$ the entry measure
(projected area of the geometrically realisable entry points, finite
crystal and obstruction included), $T_P$ the optical throughput (Fresnel
and total internal reflection), $J_{F_P}$ the normal Jacobian of the halo
map ($J_\perp$ in code) and $d\mathcal H^1$ arc length along the fibre.
Each factor is exposed separately; the Haar constant $1/(8\pi^2)$ and
$J_\perp$ are kept apart (contract section 7).

"Deterministic" does not mean exact: root finding, continuation,
automatic differentiation and quadrature are numerical and report their
errors.

## 2. Tracing one fibre

For one target direction $\mathbf d$:

1. Find a feasible seed $R_0$ with $F_P(R_0) = \mathbf d$.
2. Evaluate the differential by automatic differentiation in local Lie
   algebra coordinates; the target residual has two components, so the
   local Jacobian is $2\times 3$.
3. Its one-dimensional null space is the tangent of $X_{\mathbf d}$, with a
   consistent orientation from step to step.
4. Predict with an $\mathrm{SO}(3)$ exponential-map step, correct back onto
   the constraint (predictor-corrector with a trust region).
5. Continue until the component closes or reaches a genuine boundary of the
   valid optical domain.
6. Evaluate every factor along the curve and integrate their product.

Rotations are stored as matrices; quaternions are only an interpolation
chart, and closure respects the double cover $q \sim -q$. The stack is
Python 3.12 + JAX in float64 ([ADR 0001](decisions/0001-phase-i-python-jax.md)):
the probe that chose it showed the continuation is irregular control flow,
a poor scalar GPU workload, so GPUs are for large batches only.

The first vertical slice was deliberately narrow: hexagonal prism, path
`3-5`, one wavelength, a far point source, a smooth pose density, one
known pixel whose fibre is a closed loop. Its milestones (a seed without a
hand-entered pose, the complete loop with bounded residual, closure without
closing early, the loop plotted in diagnostic coordinates, the four factors
separately, their integral with a convergence report) are all done; the
evidence is in the appendix.

## 3. From one fibre to a renderer

A renderer has to find *every* connected component of every pixel,
distinguish closure from a near self-approach, cross chart and quaternion
sign boundaries, stop or continue correctly at refraction, TIR and
feasibility boundaries, survive rank loss near caustics, reuse neighbouring
pixels without silently losing or merging branches, and say what a pixel
means (a point value or a solid-angle average). The production pipeline
(`strip_pixel` / `strip_driver` / `strip_io`, CLI
`scripts/render_ch06_strip.py`) does, per pixel:

- **Candidates** from one scene-level seed store (`s2_store.StoreSeeds`: the
  $S^2$ event store of the path, $10^6$ points, whose deviation band around
  the pixel is posed exactly in the pixel's azimuth; until 2026-09-25 a
  $4\times10^6$-pose Haar prescan table indexed by outgoing direction, see
  section 5) instead of a per-pixel prescan;
- **Newton** onto the fibre, with the components of the pixel above as warm
  starts (warm starts are never a completeness source);
- **one production trace** per distinct candidate; a trace that ends at an
  event is traced again backward from the seed and the two halves are
  stitched into an **open arc** (a single path's solution set is often an
  arc; the six-path class sum is the full halo);
- **dedup** by $\mathrm{SO}(3)$ distance of a new seed to the curves already
  traced;
- **resampled fixed-grid quadrature**: spline resampling of the curve,
  batched back-projection and batched factors, Simpson with $N$ vs $N/2$
  compared panel by panel as the error estimate (section 4, "The
  quadrature's own errors");
- a per-pixel **procedural completeness** status layer, and the **point
  pixel model** (the sub-pixel model exists, costs 6-10×, and matters only
  in the caustic band near the 22° inner edge).

## 4. Key turns and why

**The prototype's source was lost.** Only rendered data and diagnostic
figures survived, so Phase I rebuilt the method from its mathematics and
observable results, with a written contract first (scrum
`phase1-reference-core`) and the chapter-6 artifacts classified by
provenance (explore `ch06-reference-fixture`).

**v1 pipeline (scrum `ch06-direct-integration`) and its three defects.**
Continuation was compiled into JAX kernels (55-61× per fibre), discovery,
factors, line quadrature and a strip driver were built. The owner's
accounting of the first 58 columns: about `18 s` per pixel, 18 % unknown
pixels eating 75 % of the time, lit pixels spending most of their time
re-projecting quadrature nodes, every loop walked three times. Visual review
found three defects:
(1) the closure gate's absolute minimum arc length of $\pi$ let loops
shorter than $\pi$ near the caustic close only on the second traversal,
doubling rows 58-225 — the threshold came from the canonical fixture's
3.86-long loop and failed exactly in the brightest region, and a Spearman
0.99 against the historical image did not notice a uniform ×2;
(2) the tail decayed 3-4× faster than the historical raw;
(3) step control locked at its floor next to a TIR boundary, leaving rows
655-800 unknown — an earlier explore had filed it as "performance, not
correctness", while its end effect was a quarter of the image without a
value.
Lessons kept: rank correlation is not a radiance check (look at log-scale
profiles and ratios), and a threshold taken from one fixture is a proxy that
must be checked where the criterion actually operates.

**v2 pipeline (scrum `strip-pipeline-v2`): the author's prototype order.**
Prescan once per scene, look candidates up, one aggressive trace, closure by
cross-section crossing (relative lower bound instead of $\pi$), event
slow-down by an approach-rate step limit instead of a sign test, open arcs
as first-class components, resample then integrate. Defects 1 and 3 closed
(201 051 pixels all complete, no step-like jump); the full image took
`1.86 h` on 30 workers.

**Per-pixel cost: the ruler was wrong.** The `0.067 s` per pixel of the v2
benchmark was a warm single-pixel figure; a steady-state column measured
`0.29 s`, because candidate-pool and curve sizes change per pixel and each
change recompiled XLA kernels (about 1100 compilations for 60 pixels).
Numpy batch distances and power-of-two bucketed jit fixed it; the full image
dropped to `34.7 min`, the `15 min` target missed by 2.3× at the hot-cache
floor and the physical-core count. The column benchmark
(`benchmarks/benchmark_column_steady_state.py`) is the ruler since.

**Defect 2: a convention, then an oracle.** Half of it was the crystal:
Lumice's `height 1.0` is height over *diameter*, the canonical crystal used
height over edge, i.e. half as tall; `h/a = 2` reproduced the historical
plateau. The height-independent tail was settled by Lumice's
non-tone-mapped float export: this renderer and Lumice agree at the Monte
Carlo noise floor, and the historical raw is the outlier (its tail, its
off-centre narrowness and its inner-edge offset). An 8-bit tone-mapped Lumice
image had earlier looked like agreement with the historical raw. The
author's ruling: the historical raw is no longer a correctness reference.

**Path classes and pose families.** A halo is a conjugacy class of paths,
not one representative (a representative can carry `0.4 %` of its class):
the driver expands a signature class into its PBD orbit and sums; rank-0
classes ($W = 0$) are Haar-mean point masses in the sun direction, never
traced. Members with the same $\Phi$ share one fibre (`3-1-2-5` follows the
`3-5` direction map). Five pose-density families (random, plate, column,
Parry, Lowitz) replace the single zenith-Gaussian model; only the integrand
changes.

**Absolute scale.** With nothing fitted, this renderer and Lumice agree to
`0.997-0.999` on the bright band at matched refractive index. Deriving the
factor first exposed that Lumice gave every pose equal energy instead of
weighting by the crystal's projected area, which made it a per-pixel $K_p$;
the author ruled it a Lumice bug and fixed it there (Ice Halo #597). Against
the fixed Lumice the factor is $K_p = N_{\mathrm{sym}}\,\bar y(550)\,
\Omega_p/(S/2)$, $S$ the crystal's surface area: one constant up to the
pixel's solid angle, re-checked on the column strip (`0.998`) and on the
plate and Parry families (total flux `0.9999` / `1.0001`) (task `lumice-area-weighting-recheck`,
fixture specification section 7, stage 4).

**The quadrature's own errors (2026-09-25, task
`phase1-quadrature-start-and-speed`).** Phase II's contour quadrature, an
independent chain, exposed two defects of the resampled quadrature that its
self-consistency check could not see. (1) The arclength speed differentiated
the phase condition $\boldsymbol\nu\cdot\boldsymbol\delta = 0$ as
$\boldsymbol\nu\cdot\boldsymbol\delta' = 0$, dropping
$\boldsymbol\nu'\cdot\boldsymbol\delta$; $\boldsymbol\delta$ is set by the
fixed predictor spline, so no grid removed the bias (`5.6e-6` on the
canonical pixel). $\boldsymbol\nu'$ now comes analytically from the spline's
second derivative, and Phase I meets Phase II to `3e-9` where it stopped at
`5.6e-6`. (2) The value moved with the trace's start point by up to `1.6e-4`
while the $N$ vs $N/2$ estimate often claimed less. That was not the speed:
the same trace with only its grid origin moved reproduced it. The integrand
has a few `entry_measure` kinks, each kink's Simpson error changes size and
sign with the grid phase, and the global $|I_N - I_{N/2}|$ let them cancel.
The estimate is now summed panel by panel, $\sum|S_h - S_{2h}|$, which bounded
the error at every grid phase checked, at 2-5× the old size; the grids get
finer (median 257 → 513 nodes on column 126) at no measurable column cost.

## 5. Where Phase I stands

Phase I closed on 2026-09-23 with every milestone done and the strip
renderer agreeing with Lumice in shape and absolute scale. Its cost profile
(owner cProfile of lit pixels: `74 %` in the continuation loop, about 95
corrector trials of about `0.9 ms`, each doing 3×3 arithmetic; the cost is
per-step Python orchestration and small-kernel dispatch) is why Phase II is
required to be batched and branch-free, and why Phase I's continuation was
not rewritten.

Open, recorded, not blocking: a completeness certificate (Phase II's
critical points supply one, [phase2.md](phase2.md) section 3.1), finite
solar disk, pixel averaging in the caustic band, the explicit event
localisation of contract section 12.

**Seeds from the $S^2$ event store (2026-09-25, task
`phase1-seeds-from-store`).** The Haar prescan table and the event store were
the same presampling ([phase2.md](phase2.md) section 6): a Haar pose is a
pair $(\mathbf u, \psi)$, the prescan kept the $\psi$ that happened to land
near a pixel, the store quotients $\psi$ out and poses every event of the
pixel's deviation band exactly in its azimuth. Discovery now draws its
candidates from the store (`s2_store.StoreSeeds`, `N = 1e6`, band half-width
`0.2 deg`) and `prescan.PrescanTable` is gone. A 32-pixel probe found the
prescan's components with every store configuration from `N = 1e5` /
`0.02 deg` up (appendix, "Seeds from the store"). The same events give a
statistical completeness cross-check (`discovery.check_band_coverage`: every
band event is revisited; an admissible fiber pose far from every traced curve
is a suspect, and `exp(-k_min)` bounds the chance of missing a component
whose band carries as many events as the least-covered one found). A class
seeds all members from one store through the `D6h` transports of
`path_class.store_plan`, the band-sum renderer's plan, and a store does not
depend on the sun.

Orientation distributions: Phase I assumes $\rho$ is an ordinary, possibly
narrow, density on all of $\mathrm{SO}(3)$. Exactly constrained orientation
families are singular measures on lower-dimensional sets and need a
different dimension count; they are outside Phase I.

## Appendix: measured record

Moved verbatim from `docs/roadmap.md` §3.2, §3.3 and §3.5 on 2026-09-24.
Section references inside refer to the roadmap numbering of that time.

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

### 3.3 From one fiber to a renderer (status record)

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
specification section 7, stage 4). Against Lumice after Ice Halo #597
(2026-09-24, task `lumice-area-weighting-recheck`) `A_eff` is replaced by
the constant `S/2` and the same bright band gives `0.998`. This closes the
first image-level target.
Outside the point-source point-pixel model the remaining open items are
unchanged: a completeness certificate, finite solar disk, and pixel
averaging in the caustic band.

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
   pixel-dependent `A_eff` (fixture specification section 7, stage 4; since
   Ice Halo #597 the constant `S/2`, re-checked 2026-09-24). Path-class
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

### Seeds from the store (2026-09-25, task `phase1-seeds-from-store`)

Discovery's candidates come from the $S^2$ event store
(`s2_store.StoreSeeds`, `N = 1e6`, band half-width `0.2 deg`); the Haar
prescan table (`4M` poses, `2 deg` cone) is deleted. Probe on the `32`
survey pixels, `12` store configurations (`N = 1e5 .. 1e8`, `2 / 0.2 /
0.02 deg`): every component the table found, with the same kind, on every
pixel and configuration. Regression against `artifacts/strip-full` (`833`
pixels, `775` lit): component counts, kinds and completeness identical;
values within `1.02e-4` relative (median `1.9e-5`), `773` of `775` inside the
two error estimates, the two others explained by the quadrature's
start-point dependence (same loop, `16` starts, `strip-full` inside the
spread). Details and evidence: `docs/ch06-reference-fixture.md` section 7,
stage 4.


### Quadrature start point and speed (2026-09-25, task `phase1-quadrature-start-and-speed`)

Ruler: the Phase II contour quadrature (`rtol = 1e-11`, estimates below
`1.2e-10` relative); Phase I at $\varepsilon = 10^{-12}$ to match its
$\varepsilon \to 0$ value. Evidence under the task's scratchpad (`probe/`,
`data/`).

- **Speed term.** Analytic $\boldsymbol\nu'$ against a central difference:
  relative difference `9e-6` at step `1e-3`, falling as $h^2$ to a `~5e-10`
  floor at `3e-6`. Production Phase I at `rtol = 1e-9` against Phase II on
  seven pixels (canonical, rows 150/300/450/600 and the caustic rows 58/63
  of column 126): `|rel| <= 2.6e-9` (canonical was `-5.6e-6`). Against the
  retired adaptive integrator's frozen `h/a = 1` references (rtol `1e-8`):
  `<= 5e-9` on all four, was up to `5.0e-6`.
- **Start point.** 16 re-traces along the loop, default options: the spread
  did not change with the speed term (canonical `1.57e-4` relative, (63,126)
  `8.4e-5`, (58,126) `7.6e-5`), and the same trace with only its grid origin
  moved to 16 knots reproduced it (`5.8e-5` / `1.6e-5` / `8.1e-5`). Over 256
  grid phases per node count (`N = 129 .. 1025`) on the three loops, the
  global $|I_N - I_{N/2}|$ was optimistic on 2-37 % of phases (worst 58×),
  the panel-wise $\sum|S_h - S_{2h}|$ on none (worst ratio `0.7`).
- **Column 126** (rows 48-70 and every 8th row, 106 lit pixels, all
  complete), before → after:
  default options `|rel|` median `1.26e-5` → `6.3e-6`, max `6.0e-5` →
  `5.9e-5`, estimate optimistic `4` → `0` pixels (worst 5.5× → 0.73×), node
  count median `257` → `513` (max `1025` both, never exhausted);
  `rtol = 1e-7`: median `1.8e-7` → `5.7e-9`, max `2.0e-5` → `5.8e-8`,
  optimistic `69` → `0`.
- **Cost.** `benchmarks/benchmark_column_steady_state.py --column 126 --rows
  100:160` on an M2 Max: steady median `68.6 ms` → `65.9 ms` per pixel, 5
  steady compilations both (the quadrature is a small share of a pixel).
- **Re-pinned.** `tests/test_strip_pixel.py` canonical value `6.581365570` →
  `6.581510519` (the same loop driven to `rtol = 1e-9` at $\varepsilon =
  10^{-6}$: `6.581453685`), its grid `257` → `513` nodes (also in
  `tests/test_figure_data.py`); `tests/test_path_class_ac2_symmetry.py` the
  `h/a = 1` class member `2.364363781` → `2.364440640` (against the retired
  adaptive `2.364423815`: `-2.5e-5` → `+7.1e-6`);
  `tests/test_contour_quadrature.py` Phase I against Phase II from "between
  `1e-6` and `1e-5`" to `< 1e-8` (measured `4.4e-11`).
  `artifacts/strip-full` was not re-rendered: its values move by at most the
  old estimate (`~1e-4` relative).
