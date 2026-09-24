# Phase II: The Integral on $S^2$

Chinese version: [phase2_zh.md](phase2_zh.md) (the measured record of the
appendix is kept in English only).

Phase I solves for the pose fibre of every pixel on $\mathrm{SO}(3)$
([phase1.md](phase1.md)). Phase II uses the structure of the problem to move
the integral to the sphere of source directions in the crystal frame, where
it becomes a level-set integral of one scalar field. Two quadratures of that
integral exist: tracing the level sets (contour method) and summing
precomputed events per deviation band (band sum). The band sum is in
production (milestone M1, 2026-09-23); the contour method is milestone M2
(scrum `phase2-contour-quadrature`). This document is the design; decisions
are dated in [roadmap.md](roadmap.md) §9, and the measured record is the
appendix.

Notation follows [conventions.md](conventions.md): $\hat{\mathbf s}$ points
toward the source, the incident propagation direction is
$\mathbf s = -\hat{\mathbf s}$, $\Phi_P$ maps an incident propagation
direction to the outgoing one in the crystal frame, $\psi$ is the twist
about $\hat{\mathbf s}$, $\rho$ the pose density relative to Haar. Statements
are marked **measured** (with evidence), **derived** (follows from the
model, not yet checked numerically) or **design** (a proposed organisation,
cost not measured) where it matters.

## 1. From $\mathrm{SO}(3)$ to $S^2$

Let

$$
\mathbf u=R^{-1}\hat{\mathbf s},
\qquad
D_P(\mathbf u)=\angle\big(-\Phi_P(-\mathbf u),\mathbf u\big)
$$

(the writing series' framework theorem 8): $\mathbf u$ is the source in the
crystal frame, $-R\,\Phi_P(-\mathbf u)$ is the light point on the sky, and
$D_P$ its angular distance from the source. $\mathrm{SO}(3)$ fibres over
$S^2$ by $R \mapsto \mathbf u$; the remaining coordinate is the twist
$\psi$ about $\hat{\mathbf s}$, and Haar probability splits as
$d\mu_{\mathrm{Haar}} = dA(\mathbf u)/4\pi \cdot d\psi/2\pi$.

**Every geometric and optical weight is a function on $S^2$.**
`entry_measure` reduces the pose to `s_body = R.T @ incident` $= -\mathbf u$
on its first line and uses nothing else; the TIR gates and the Fresnel
factors depend on the incidence angles, hence on $\mathbf u$ only. Rotating
the crystal about the source direction changes none of them: the twist
moves the outgoing direction rigidly about $\hat{\mathbf s}$. So the
entry measure $A_P$, the transmission $T_P$, the window $w_P = A_P T_P$, the
validity gates, the feasible domain $V_P$ and the deviation $D_P$ are fields
on the base $S^2$; only $\rho$ sees the full pose. Phase I evaluates them
point by point along an $\mathrm{SO}(3)$ curve because it cannot see this
structure.

**Measured**: validity, $A_P$, $T_P$, $\Phi_P$ and $D_P$ are unchanged under
three twists about $\hat{\mathbf s}$ to `1.5e-14` (task
`band-sum-quadrature-probe`).

### 1.1 What a precomputation depends on

An event $(\mathbf u_i, \Phi_i, D_i, w_i)$ is therefore a pure "incident
direction to outgoing direction" record in the crystal frame. A table of
such events depends on **the crystal shape, the path (its $\Phi$ group,
section 3.2), the refractive index (the wavelength) and the sampling ($N$,
lattice or sampler, deviation window, dtype), and on nothing about the light
source**.

- **Measured** (owner probe, 2026-09-24,
  `scratchpad/task-s2-store-schema-3/owner_probe_sun_indep.py`): $h/a = 2$,
  $n = 1.31$, $N = 2\times10^5$, sun at (15°, 0°), (60°, 37°), (−30°, 200°);
  paths `[3,5]` and `[1,3,2]`: the same event count, $\mathbf u$ equal bit
  for bit, $\Phi$, $D$, $w$ within `1.6e-11`.
- Consequences: a new sun elevation or azimuth, a new pose density or a new
  camera reuse the store and only re-render; a new wavelength needs its own
  store; a new crystal shape or path needs its own store.
- In production (schema 3, task `s2-store-schema-3`, 2026-09-24):
  `S2StoreSpec` and the cache key carry no sun direction; the build aligns
  $R\mathbf u = \hat{\mathbf s}_0$ to one fixed reference direction,
  numerically the canonical sun so that the schema 2 canonical build is
  reproduced bit for bit, and one store serves every sun elevation.
  `tests/test_s2_store.py::test_events_are_independent_of_the_reference_direction`
  turns the probe into a regression (three directions × `[3,5]`, `[1,3,2]`,
  `[1,3,5,2]`, the same kept points, $\Phi$, $D$, $w$ within `1e-10`), and
  `tests/test_band_sum.py::test_one_store_serves_every_sun_altitude` pins
  that a second altitude does not rebuild. Schema 2 (sun direction in the
  key, one `events.npz`) was reproduced bit for bit and is refused on load.

The source enters a rendering only through the pose that places an event
on a pixel and through $\rho$ of that pose.

## 2. One integral, two quadratures

For a pixel at deviation $\delta$ and azimuth $\alpha$ about the source,
push $\rho A_P T_P\,d\mu_{\mathrm{Haar}}$ forward to the sky. At fixed
$\mathbf u$ the outgoing azimuth is $\alpha = \alpha_0(\mathbf u) + \psi$, so
$d\psi = d\alpha$, while the deviation stays $D_P(\mathbf u)$. With
$dA(\mathbf d) = \sin\delta\,d\delta\,d\alpha$ on the sky this gives the
Phase I pixel value (same $1/(8\pi^2)$ as contract section 7):

$$
I(\delta,\alpha)\,\sin\delta
= \frac{1}{8\pi^2}\int_{S^2}\delta_{\mathrm{Dirac}}\big(D_P(\mathbf u)-\delta\big)\,
  \rho\,A_P T_P\,dA(\mathbf u)
= \frac{1}{8\pi^2}\int_{D_P=\delta}
  \frac{\rho\,A_P T_P}{|\nabla_{S^2}D_P|}\,d\ell .
$$

There is no separate $d\psi/2\pi$ at a fixed pixel: $\psi$ is consumed by
$d\psi = d\alpha$, and $\rho$ is evaluated at the single-valued pose
$R(\mathbf u, \psi(\mathbf u,\alpha))$.

The right-hand side can be discretised two ways, and they are two
quadratures of one integral, not alternatives:

- **Contour method** (section 4): trace the level set $\{D_P = \delta\}$
  and integrate along it. Pointwise, deterministic, high order; with the
  critical points of $D_P$ it certifies that every component was found.
- **Band sum** (section 5): integrate over a band
  $[\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}]$ instead, which turns the
  line integral into an area integral on $S^2$, and evaluate it on $N$
  precomputed equal-area points. One sort, one range query and one batched
  $\rho$ evaluation per pixel.

**Boundaries need no event handling.** The integrand vanishes continuously
on every boundary: corridor boundaries (two polygons separating,
$A_P \to 0$ continuously), the exit-face TIR boundary of the formula domain
$U_P$ (Fresnel transmittance $\to 0$ at the critical angle); there is no
critical angle on entry, and partial reflection on internal steps is a
continuous weight. A contour cut by $\partial V_P$ is traced on $U_P$ and
$A_P T_P$ removes the infeasible part; Phase I's rule "keep tracing, weight
to zero" (contract section 6.3) is the same fact placed inside the tracer.
The remaining non-smoothness is the kinks of $A_P$ (a vertex crossing an
edge), which lower the quadrature order locally, as in Phase I.

## 3. Structure the sphere exposes

### 3.1 Topology and completeness

Level sets of a scalar field are governed by its critical points:
$\nabla D_P = 0$ (finitely many, found by AD Newton from grid seeds) plus the
critical points of $D_P|_{\partial U_P}$ split the $\delta$ axis into
intervals on which the level-set topology is constant. Marching once per
interval and Newton-refining gives *every* component, so the completeness
certificate that Phase I cannot issue (contract C11: `completeness` is
procedural) becomes a checkable statement. This is the larger gain of
Phase II; speed is the smaller one. (**Design**; M2 sub-tasks
`dp-field-topology`, `dp-field-layer`, `s2-contour-extraction`.)

**Implemented: the field layer** (`lumice_integral.dp_field`, task
`dp-field-layer`; public surface `DPField` and `TopologyEscape` only, so the
rank-0 refusal in `DPField.build` cannot be bypassed). For one face
sequence, independent of the sun:

- *Evaluation.* $D_P(\mathbf u) = ngle(\Phi_P(-\mathbf u), -\mathbf u)$ at
  the identity pose, batched (`jax.vmap`) value, tangent gradient and
  Riemannian Hessian $P(H - (\mathbf u\cdot\mathbf g)I)P$ — the curvature
  term is what makes it independent of how $D_P$ is extended off the sphere
  (without it the 3-5 minimum reads as a maximum or a saddle, depending on
  the extension). Taken in `atan2` form; equal to the event store's
  `evaluate_fields` to `1e-12`.
- *Fold pre-screen.* $|\mathbf n_a\cdot M^{\mathsf T}\mathbf n_b| = 1$ is a
  slab: $D_P(\mathbf u) = ngle(M\mathbf u, \mathbf u)$, evaluated in that
  closed form, whose critical set is $\pm\mathbf n_M$ and the crease
  $\mathbf u\cdot\mathbf n_M = 0$; otherwise interior critical points come
  from damped tangent-space Newton, batched over the $U_P$ points of a
  Fibonacci lattice, classified by the Hessian.
- *Boundary.* $\partial U_P$ is walked: along the zero set of the active
  margin with $U_P$ on the left, a corner wherever another margin turns
  negative (bisection, then a two-margin Newton, residual `<= 2.3e-16`),
  continuing along the one margin through the corner that keeps the
  boundary, until the walk is back at its first corner — closing the walk
  is the completeness statement for the loop. An incidence cosine is walked
  as a great circle when its normal $\mathbf m = R_{k-1}^{\mathsf T}\mathbf n_k$
  satisfies $\mathbf m\cdot\mathbf n_a = 0$ (then the margin is linear in
  $\mathbf u$; checked per path at run time), otherwise marched like every
  discriminant. Margins equal to an earlier one (three side-face
  reflections stepping by ±60°) are dropped first; a margin vanishing along
  a whole piece (a square, such as `exit_snell_discriminant` $=$
  `entry_incidence_cosine`$^2$ on `3-5-6-7-3`) is recorded as coincident.
  A corner lists every margin vanishing there; the two it is bounded by;
  the others as tangent or transversal to them.
- *Partition.* Critical values are the interior ones, the extrema of $D_P$
  along the loop, and the corner values. On each interval the number of
  open arcs is half the number of crossings of $\delta$ along the loop
  (exact for any topology); closed loops need an interior extremum, and with
  at most one (a non-degenerate minimum or a slab cone point) there is one
  closed loop from its value to the loop extremum where the sublevel set
  first reaches $\partial U_P$ (checked on a small ring). Anything else —
  $U_P$ or its complement not connected on the lattice, several interior
  critical points, a saddle, a crease inside $U_P$, a boundary-born sublevel
  component — raises `TopologyEscape` instead of guessing.
- *Check.* `scripts/verify_dp_field_intervals.py` recomputes every
  interval's counts on a dense grid through `evaluate_fields`: closed loops
  as sub/superlevel regions touching no boundary, arcs as crossings along
  the traced grid boundary moved onto $\partial U_P$ by bisection (node
  values alone fail: $D_P$ falls like a square root off an exit-TIR curve).
  All intervals of the five fixtures agree; the measured values are in the
  appendix.

### 3.2 Layered invariance: what a halo shares and what varies

The fibre of a pixel, $\{R : R\,\Phi_P(-R^{-1}\hat{\mathbf s}) = \mathbf d\}$,
depends on the path only through $\Phi_P$. Hence:

| layer | object | shared by |
|---|---|---|
| $\Phi$ | the field $D_P$, its contours, $1/\lvert\nabla D_P\rvert$, the correspondence $\psi(\mathbf u,\alpha)$, the critical points | the whole $\Phi$-class, across PBD classes |
| member | the window field $w_m = A_m T_m$ on $S^2$, additive: $w_\Phi = \sum_m w_m$ | one per face sequence |
| symmetry | a crystal symmetry $g$ moves the map: $D_{gPg^{-1}}(\mathbf u) = D_P(g^{-1}\mathbf u)$, windows transported alike; on $S^2$ all of $D_{6h}$ acts, mirrors included (section 3.3) | only $\rho$ can tell the members apart |
| $\rho$ | the pose density on the fibre | the *columns* of the writing series' table |

A $\Phi$-class is one contour family plus one effective window field, and
the writing series' core table (rows = path classes, columns = pose
families) has the skeleton row $= (D_P, w_\Phi)$, column $= \rho$, cell $=$
the line integral. In code the $\Phi$ layer is `path_class.phi_key` (equal
keys, equal $\Phi_P$ exactly, no symmetry quotient); the writing series'
coarser classes live in `lumice_integral.symmetry.signature`: the $D_{6h}$
orbit of a key is one canonical signature class, and a `phi_class` is a
union of such orbits (framework theorem 5′: 14 signature classes, 6 $\Phi$
classes): the orbit of a key $(M, a, \tilde a)$ is the class
$(M, \mathbf n_a, M^{-1}\mathbf n_b)$ modulo $D_{6h}$ conjugation, exactly one
for the 60° and 90° wedges, the parallel (0°) orbits merged by the conjugacy
class of $M$
(`test_path_class_phi_key.py::test_d6h_orbit_of_the_key_is_one_signature_class_and_refines_phi_class`).

The same statements hold on $\mathrm{SO}(3)$: for a proper $g$,
$\mathrm{fiber}(gPg^{-1}) = \mathrm{fiber}(P)\,g^{-1}$ exactly, and members with
the same $\Phi$ share one curve (task `path-class-rendering-unit` measured
"same direction map, different weight" for `3-1-2-5` on row 651). Task 9's
`12x` for the column density is the symmetry row with a $\rho$ invariant
under the $C_6$ rotations and $C_2'$; the tilted Parry density breaks it
through $\rho$ alone.

### 3.3 Symmetry is precomputation, not new samples

The representative's field on all of $S^2$ is the same information as every
member's field on a fundamental domain $F = S^2/G$: the representative's
field on the block $hF$ is the field of the member $h^{-1}Ph$ on $F$.
"Transport one event to $|G|$ images" and "compute $1/|G|$ of the sphere"
are the same thing. Symmetry saves repeated evaluation and makes no new
samples; the precision of a band sum is set by the number of *distinct*
precomputed events in the band with non-zero weight
($K_{\mathrm{eff}}$ counts events, not transported images).

Mirrors transport on $S^2$ exactly like rotations:
$w_{gPg^{-1}}(g\mathbf u) = w_P(\mathbf u)$,
$\Phi_{gPg^{-1}}(g\mathbf u) = g\,\Phi_P(\mathbf u)$, the valid domain is the
same (**measured** for all 24 elements to `1e-12`). The pose of a
transported event is rebuilt from $(g\mathbf u, g\Phi, D)$ by two
orthonormal frames, a rotation whatever $\det g$; on the representative's
pose $R$ this is $L_g R g^{\mathsf T}$ with
$L_g = I - (1-\det g)\,\mathbf m\mathbf m^{\mathsf T}$, $\mathbf m$ the normal
of the plane of $\hat{\mathbf s}$ and the pixel. "A mirror needs its own
store" was a Phase I restriction ($R g^{-1}$ must be a rotation in
$\mathrm{SO}(3)$), not an $S^2$ one; one store serves a whole class.

## 4. Quadrature A: tracing the contours (M2, design)

The M2 scrum builds, in order: the topology of $D_P$ on $U_P$ (interior
critical points by AD Newton from lattice seeds, Hessian classification,
restricted critical points and corners on $\partial U_P$, interval
partition with component counts, checked against dense-lattice marching);
the field layer as a batched `vmap`-able module; contour extraction at a
given $\delta$ (seeds from the event store's band and from lattice marching,
Newton onto the level set, closed loops and arcs cut by $\partial U_P$,
component count checked against the partition: the certificate); the line
quadrature with its constant, cross-validated against Phase I and the band
sum; and the chapter-10 verdicts (section 10).

Design constraints carried over from the Phase I cost profile
([phase1.md](phase1.md) section 5; evidence
`scratchpad/task-pixel-cost-shape-stable-kernels/evidence/owner_cprofile_col126_rows300-340.prof`): `74 %` of a lit Phase I pixel is the
continuation loop, dispatch-bound per step, and 30 workers on `home-wsl` sit
at the physical-core wall. Phase II must be batched, branch-free and
`vmap`-able from the first design (field evaluation on an $S^2$ grid,
contour extraction and quadrature as array programs), so that a full image
is a field computation and a GPU becomes usable. A per-pixel Python loop
would reproduce the same wall in a new place.

Fixtures the structure suggests:

- *Liljequist* (writing chapter 8): `1-3-2` and `3-5-6-7-3` have the same
  $\Phi$ (the mirror in the plane of faces 3/6; three reflections in planes
  at ±60° compose to one) and different windows. The 142° sharp edge
  ($= 120° +$ the 21.84° minimum deviation) is a $D_P$ critical value and is
  shape-independent; the narrow peak is the `3-5-6-7-3` window and moves
  with the cross-section. Two pictures on one sphere. *Measured (task
  `dp-field-layer`):* both are slabs with the same $M$, so the fields agree
  pointwise, but $U_P$ differ and so do the critical values on them:
  $\{0°, 115.607°\}$ for `1-3-2`, $\{0°, 153.070°, 180°\}$ for
  `3-5-6-7-3` (normal incidence on face 3 is inside, the backscatter cone
  point). No 142° value on either with this repository's face numbering;
  task `verify-liljequist-face-numbering` holds the manual check.
- *Parhelic circle*: $D_P(\mathbf u) = \angle(M\mathbf u, \mathbf u)$ has
  $\nabla D_P = 0$ only at $\pm\mathbf n_M$ (on `3-1-6` both lie on the entry
  great circle but outside the closure of $U_P$, an internal TIR fails
  there: no interior critical point at all), so the ring has **no fold**;
  its brightness along the ring is entirely the window layer. For plates
  the ring azimuth is linear in the crystal azimuth, so the profile is a sum
  of shifted copies of one window (three mirror planes of the prism): a test
  of the window-sum and transport layers without the Jacobian in the way.
- *22° halo*: the fold; see section 10.

**Why the completeness certificate rarely needs a saddle branch.** A
systematic search over hexagonal-prism paths (entry face in `{1, 3}`,
up to four internal reflections, symmetry-deduplicated, filtered by the
fold discriminant below) found no interior saddle among the 87 non-empty
candidates it reached: every one has $U_P$ a topological disk (itself and
its complement on $S^2$ each connected, checked at two lattice densities),
so Poincaré–Hopf only forces an interior-critical-point index sum of $1$
(one non-degenerate minimum already satisfies it) — a saddle is only
topologically required once $U_P$ stops being simply connected. This does
not prove no hexagonal-prism path ever has a saddle (longer paths were
only spot-checked, and the yield of non-empty candidates falls by an order
of magnitude per added reflection), but it explains why the paths this
project actually renders are unlikely to need the interval-splitting
machinery for a saddle branch, and gives `task-dp-field-layer` a
disk-domain default with an explicit, checkable escape hatch (test $U_P$'s
own and complementary connectivity before assuming Morse-Bott simplicity).
Measured record: appendix, "$D_P$ field topology probes".

**Implemented: contour extraction** (`lumice_integral.contour`, task
`s2-contour-extraction`). `extract_level_sets(field, deltas, store)` returns,
for every $\delta$, every component of $\{D_P = \delta\}\cap U_P$ as a node
sequence (closed loop, or open arc with both ends on $\partial U_P$ and the
smallest margin reported there), certified:

- *Seeds, three sources.* The field layer's critical data first: the
  crossings of $\delta$ by $D_P$ along the boundary loop (bisection along
  the piece, pulled `1e-14` inside; the ends of every open arc) and the first
  crossing along a geodesic ray out of the interior extremum (a point of the
  closed loop around it). They reach what no sampling resolves at
  $\pm10^{-6}$ rad from a critical value: the arc cut off below a loop
  maximum is `~1e-6` rad deep, `~1e-12` in margin on an exit-TIR piece where
  $D_P\sim\sqrt{\text{margin}}$; the loop above the 3-5 minimum is `~2e-3` rad
  across, the one below the `3-5-6-7-3` cone point $D = \pi$ has radius
  `5e-7`. The event store's band ($|D-\delta| < h$, $h$ two mean point
  spacings) and linear crossings on the edges of an orthographic grid of the
  entry hemisphere are the independent check: any of their seeds farther
  from the extracted components than its own resolution is refined and
  walked, and a component it finds is an extra one.
- *Walking.* Lockstep over every curve of every $\delta$ (one `jax.vmap`-ed
  step scanned 64 at a time, finished curves compacted out, batches padded
  to powers of two): geodesic predictor along $\mathbf u\times\nabla D_P$,
  Newton corrector along $\nabla D_P$; a step is accepted inside $U_P$, on
  the level set, and with the tangent turned by at most 5°, otherwise
  halved. Leaving $U_P$ halves down to `1e-13` rad, which puts an arc end on
  $\partial U_P$. A walk closes when its start is within one current step
  ahead — relative, never an absolute distance (Phase I defect 1). Next to an
  exit-TIR piece the curve runs parallel to $\partial U_P$ at a depth of
  `~1e-12` and the geodesic predictor falls out by the boundary's curvature;
  there the predictor keeps the first-order value of the smallest margin.
- *Certificate.* Per $\delta$ the (closed, open) counts are compared with the
  interval of `DPField.interval_partition()` containing it; a mismatch
  raises `ContourCertificateError`, a `TopologyEscape` of the partition (a
  saddle among others) is propagated and nothing is extracted, a $\delta$ at
  a critical value is refused.
- *Sharing.* The result depends on $\delta$ only: repeated values are
  extracted once, and a class member takes `LevelSet.transported(g)`
  ($\mathbf u\to g\mathbf u$, node order reversed for improper $g$ so that
  nodes keep running along $\mathbf u\times\nabla D_P$).
- *Accuracy.* $|D_P - \delta| \le 10^{-12}$ wherever $|\nabla D_P| \le 10^3$;
  next to an exit-TIR curve ($|\nabla D_P|$ up to `~1e7`) the rounding of
  $\mathbf u$ alone moves $D_P$ by $\varepsilon|\nabla D_P|$ and nodes are held
  to $64\,\varepsilon|\nabla D_P|$ (at most `~8e-9` measured). An arc meets
  such a curve tangentially, so its end is located to `~sqrt(1e-13)` along
  the curve.

The saddle branch is covered only as an escape: no fixture has an interior
saddle (above), and extraction refuses to run where the partition escapes.
Measured record: appendix, "Contour extraction".

## 5. Quadrature B: the band sum

Source: the Ice Halo Simulation repository,
`doc/research/inverse-rendering.md` (Chinese: `inverse-rendering_zh.md`;
"Inverse Rendering via Precomputed Standard Events", after Gislén et al.
2004). That note is a design sketch;
this section is the authoritative statement of the idea for this project.
The note's objects are the fields above:

| note | here |
|---|---|
| standard event $(\hat a_0, \hat b_0)$ | the body-frame propagation pair $(-\mathbf u, \Phi_P(-\mathbf u))$ |
| scattering angle $\omega$ | deviation field $D_P(\mathbf u)$ |
| event weight $w$ | window field $A_P T_P(\mathbf u)$ |
| rotation $U$ of eq. 19 | the pose $R(\mathbf u, \psi(\mathbf u,\alpha))$ |
| $Q(U)$ | $\rho(R)$ relative to Haar probability |

**Estimator.** With $N$ equal-area points ($4\pi/N$ each) and the band of a
pixel $[\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}]$ (min/max deviation of
its four corners),

$$
\hat I(\delta,\alpha)
= \frac{1}{2\pi N\,\Delta\delta\,\sin\delta}
  \sum_{i:\,D_P(\mathbf u_i)\in[\delta_{\mathrm{lo}},\delta_{\mathrm{hi}}]}
  A_P T_P(\mathbf u_i)\;\rho\big(R_i\big),
\qquad \Delta\delta = \delta_{\mathrm{hi}}-\delta_{\mathrm{lo}},
$$

an estimate of the band average of $I\sin\delta'$ divided by $\sin\delta$.
The constant is derived, never fitted (**measured**: median ratio `1.0000`
against Phase I at $N = 10^8$).

**The pose uses the event's own deviation.** $R_i$ is the unique pose with
$R_i\mathbf u_i = \hat{\mathbf s}$ and $R_i\Phi_P(-\mathbf u_i)$ at deviation
$D_P(\mathbf u_i)$ *and* the pixel's azimuth:
$R_i = [\hat{\mathbf s}, \mathbf e, \hat{\mathbf s}\times\mathbf e]\,
[\mathbf u_i, \mathbf f_i, \mathbf u_i\times\mathbf f_i]^{\mathsf T}$, with
$\mathbf e$ the pixel's unit azimuth direction about $\hat{\mathbf s}$ and
$\mathbf f_i$ the unit component of $\Phi_P(-\mathbf u_i)$ normal to
$\mathbf u_i$ (`s2_store.event_rotations`). This is the note's eq. 19 at
$\omega = D_P(\mathbf u_i)$. Feeding eq. 19 the pixel's own $\delta$, as the
note's step 3 reads, gives a non-orthogonal matrix whenever
$D_P(\mathbf u_i) \ne \delta$ (median $\lVert U^{\mathsf T}U - I\rVert_F$
`5.5e-4`, max `1.4e-3`, the size of the band) and $\rho$ of a non-rotation.
The frame construction equals eq. 19 at $\omega = D_P$ to `7e-15` without its
$1/\sin^2\omega$. The probe that established the band sum is
`scripts/probe_band_sum.py`.

**Corrections to the note.**

1. *Not noise-free.* The band sum is deterministic but carries a
   discretisation error: about $1/\sqrt{K_{\mathrm{eff}}}$ for scattered
   points, aliasing for a regular grid. A Fibonacci lattice cut by a thin
   band usually behaves like scattered points and often beats
   $1/\sqrt{K_{\mathrm{eff}}}$ by 7-9×, but aliases (Parry at $10^7$), so
   stores are sized by $K_{\mathrm{eff}} \ge 10^4$ at the worst pixel of
   interest, not by the lattice's typical gain.
2. *Narrow $\rho$ lowers $K_{\mathrm{eff}}/K$, pixel by pixel.* The band's
   event set is fixed by $\delta$ alone; $\rho$ selects the part that
   contributes. **Measured**: the band keeps 65 % (column density on the sun
   vertical), 17 % (off it), 3 % (plate), 0.45 % (Parry), 0.58 % (Lowitz) of
   its events, but $K_{\mathrm{eff}}$ still grows as $N^{1.00}$ and never
   collapses on a lit pixel; all five chapter-11 families are served at
   $N = 10^8$ by one uniform store, no $\rho$-aware store needed.
3. *Normalisation is explicit*: $I\sin\delta$, Haar
   $dA/4\pi\cdot d\psi/2\pi$, the constant above.
4. *"Caustics regulate their own brightness"* is an acceptance test, not a
   premise (section 10).
5. *Organise the store by $\Phi$ group* (section 3.2), not one array sorted
   by $\omega$ with a path id mixed in.

**Pixel model.** The band sum's pixel is a band average in $\delta$ and a
point in $\alpha$; Phase I and the contour method give point values. On
steep edges the two differ by the pixel model, not by sampling error (the
worst M1 pixels at the 22° inner-edge caustic have part of their band below
$\min D_P$). Cross-validation compares them in one convention.

**Division of labour.** The contour method owns accuracy and the
completeness certificate; the band sum has neither a certificate nor a
convergence order beyond sampling, but is branch-free, `vmap`-able and
shares one store across every pixel, pose density and source direction. It
is the fast renderer, parameter-sweep tool and independent cross-check.
Production modules: `lumice_integral.s2_store` (build, cache, provenance,
band slices, $D_{6h}$ transport) and `lumice_integral.band_sum` (estimator,
path and class rendering; CLI `scripts/render_band_sum.py`). Canonical
strip at $N = 10^8$: `30.8 s` on four Mac workers (`169 s` before the
scatter form of section 8) against Phase I's `34.7 min` on 30 `home-wsl`
workers (appendix).

## 6. One precomputation, three consumers

Phase I also precomputes. `prescan.PrescanTable` draws $4\times10^6$ Haar
poses on $\mathrm{SO}(3)$ for a fixed sun, keeps the domain-valid ones with
their outgoing directions, and indexes those with a k-d tree; a pixel asks
for the samples in a cap around its own direction and uses them as Newton
seeds.

The two are the same sampling. A Haar sample $R$ is a pair
$(\mathbf u, \psi)$: its deviation is $D_P(\mathbf u)$ and its azimuth is
fixed by $\psi$, which has a closed form given $\mathbf u$ and the target
azimuth. The prescan table samples $\psi$ at random and keeps what happens
to land near the pixel; the store quotients $\psi$ out and constructs, for
every event in the pixel's band, the pose that lands **exactly** on the
pixel's azimuth (`band_sum.band_poses`). The store is the strictly stronger
object:

| | Phase I prescan table | $S^2$ event store |
|---|---|---|
| samples | Haar poses on $\mathrm{SO}(3)$, random | points $\mathbf u$ on $S^2$, Fibonacci lattice |
| per pixel | samples in a cap around the pixel direction | events in a $\delta$ band, each at the pixel's exact azimuth |
| residual of a candidate | two-dimensional (deviation and azimuth) | one-dimensional (deviation) |
| index | k-d tree on outgoing directions | one sort by $D$, `searchsorted` |
| depends on | sun, path, index; not the crystal (infinite prism; `entry_measure` gated later) | crystal, path, index; not the source |
| keeps | domain-valid samples | $w > 0$ events |

A band event serves every azimuth on its ring, a cap sample only its own
neighbourhood; at equal cost the store gives more usable candidates per
pixel and seeds closer to the fibre. It has three consumers:

1. **The band sum** (section 5): the events are quadrature nodes.
2. **Contour tracing** (section 4): a band event lies within half a band
   width of $\{D_P = \delta\}$; it seeds Newton onto the contour.
3. **Phase I seeds** (M2 sub-task `phase1-seeds-from-store`, **design**):
   band poses replace the prescan candidates, and `PrescanTable` goes if a
   32-pixel probe loses no component. The same events give Phase I a
   completeness cross-check: every band event should lie near one of the
   traced fibres, and an event far from all of them marks a missed
   component. The check is statistical, but its miss probability is bounded
   by $N$ times the component's measure in the band, which the density
   survey behind `DEFAULT_SAMPLE_COUNT` does not give. Because the store
   does not depend on the source, Phase I also stops rebuilding per sun.

## 7. Ring invariance and the cost of each route

Let $Q$ be a rotation about $\hat{\mathbf s}$. If $R$ sends the ray to
$\mathbf d$, then $QR$ sends it to $Q\mathbf d$ with the same
$\mathbf u = (QR)^{-1}\hat{\mathbf s} = R^{-1}\hat{\mathbf s}$. So the fibre
of $Q\mathbf d$ is $Q$ times the fibre of $\mathbf d$, and all pixels at one
deviation (one ring about the source) share one level set on $S^2$, one
window along it and one $1/\lvert\nabla D_P\rvert$. The azimuth enters only
through $\rho$. Phase I retraces the same curve at every azimuth because it
works pixel by pixel on $\mathrm{SO}(3)$.

| route | once, resolution-independent | per $\delta$ ring | per pixel | accuracy |
|---|---|---|---|---|
| Phase I | prescan table | — | discovery, trace, integrate: `0.1-0.3 s` (measured) | pointwise, adaptive error estimate |
| band sum | store of $N$ events (`80 s` at $10^8$, measured) | — | $K$ band events: $\rho$ of a matrix product, `0.31 ms` CPU (scatter, section 8; gather `3.3 ms`, measured) | band average in $\delta$; $\sim 1/\sqrt{K_{\mathrm{eff}}}$ |
| contour (design) | $D_P$ field and critical points | extract and refine the level set | $\rho$ along stored nodes | pointwise, deterministic, high order |

Scaling with resolution:

- Phase I: pixels × a large constant.
- Band sum: pixels × $K_{\mathrm{target}}$, a small constant. The band
  follows the pixel, so keeping $K_{\mathrm{eff}}$ when the linear
  resolution doubles needs twice the $N$: the store grows linearly with
  resolution, on disk (the renderer maps it, section 8).
- Contour: the expensive part ∝ rings ∝ linear resolution, as a batched
  field computation; only the final line integral ∝ pixels; no sampling
  noise and no store that grows with resolution (the store only seeds).
  The per-pixel constant is to be measured by M2.
- A $\rho$ invariant about $\hat{\mathbf s}$ (the random density) makes the
  value a function of $\delta$ alone: one number per ring, for both Phase II
  routes. The four other chapter-11 families refer to the c axis and are
  not invariant about $\hat{\mathbf s}$ unless the source is at the zenith.

## 8. Organising the band sum by deviation

Until task `band-sum-scatter-renderer` the production renderer was a
*gather*: per pixel, find the band, `searchsorted` the events, rebuild their
poses at the pixel azimuth, evaluate $\rho$, sum. Jobs were whole columns and
a column spans the whole $\delta$ range, so every spawned worker loaded every
store of the plan in full, and the schema 2 store build kept every $w > 0$
event in memory before one `argsort`. The memory ceiling of section 7 was
that organisation, not the band sum.

Turning the loops inside out (the author's proposal, 2026-09-24, likened to
swapping the loop indices of a matrix product) keeps the sum and changes the
order of work: walk the events in deviation order and let each chunk of
events scatter into every pixel whose band it meets. The gather
(`band_sum.class_band_sum_pixel`) is kept as the test oracle, not as a
second rendering path.

**Storage (task `s2-store-schema-3`).** One `.npy` per array (a `.npz`
cannot be mapped partially), with its SHA-256 and size in the provenance;
`S2EventStore.load(..., mmap_mode="r")` maps them read-only and checks
sizes only, `S2EventStore.verify` hashes on demand, the default load still
hashes everything. The build buckets the kept events by $D$ on disk (1024
equal-width buckets, a performance knob outside the cache key) and sorts
per bucket, which is the one global stable sort bit for bit;
`build_or_load` writes the arrays straight into the cache, so the store is
never in memory during the build (at $N = 10^8$ the build peak halves,
appendix).

**The pose splits into a pixel and an event factor.** The pose of section
5 is $R_i = W F_i^{\mathsf T}$ with the pixel frame
$W = [\hat{\mathbf s}, \mathbf e, \hat{\mathbf s}\times\mathbf e]$ and the
event frame $F_i = [\mathbf u_i, \mathbf f_i, \mathbf u_i\times\mathbf f_i]$
(`s2_store.pixel_world_frame`, `s2_store.event_frames`); $F_i$ depends on
neither the pixel nor the sun. The five pose densities see the pose only
through the zenith components of the body axes, the third row of $R_i$
(`ZenithGaussianPoseDensity` the c axis, `ZenithRollGaussianPoseDensity`
the c axis and the roll $\operatorname{atan2}(-e_2, e_1)$; the random
density none). So

$$
\big(R_i\big)_{3j} = \hat{\mathbf z}\cdot R_i\mathbf e_j
= \big(W^{\mathsf T}\hat{\mathbf z}\big)\cdot\big(F_i^{\mathsf T}\big)_{\cdot j}
= \mathbf a_{\mathrm{pixel}}\cdot\mathbf b_{ij},
$$

a dot product of a pixel vector ($W[2, :]$) and an event vector
($F_i[j, :]$), with no reference azimuth and no rotation $Q_\alpha$ needed.
For $M$ pixels and $K$ events this is one $(M\times 3)(3\times K)$ product
per body axis the density reads. `pose_density` exposes it as
`axis_zeniths` (the components a family reads: `()`, `("e3",)` or
`("e1", "e2", "e3")`) and `evaluate_axis_zeniths(e1=, e2=, e3=)` on arrays
of any shape; `evaluate_batch` stays the defining oracle (the two agree to
`1e-13` on every family, test). The $D_{6h}$ transport
$L_g R g^{\mathsf T}$ of section 3.3 is $W (g F_i J)^{\mathsf T}$ with
$J = \operatorname{diag}(1, 1, \det g)$: a fixed linear map of the event
side only (`s2_store.transported_frames`), the pixel vector unchanged.

**The renderer** (`band_sum.render_band_sum_window`, `scatter_store`).

1. The workers compute every pixel's band (`pixel_band`, columns as jobs;
   the same arithmetic, so $\delta$, the band and its width are bit for bit
   the gather's).
2. The parent orders the pixels by band centre and cuts them into one
   segment per worker of about equal work (events × transports, from the
   mapped `D` arrays). A segment's events are one contiguous stretch of
   each store.
3. Each worker maps every store of the plan read-only in turn
   (`mmap_mode="r"`, one store group at a time, released before the next;
   the content was hashed once by the parent), and walks the events of its
   stretch in chunks of 1024. Per chunk the event frames and their
   transports are built once; the pixels whose band meets the chunk take,
   in blocks of 32, the matrix products above, $\rho$ elementwise, the band
   mask and a sum over the events. The band of a pixel is the index range
   `searchsorted(D, [lo, hi])`, the gather's left-closed right-open
   comparison, so $K$ is identical.
4. Per event the contributions of its transports are summed before
   squaring ($w\rho$ per transport, then summed, the gather's order: a
   product next to underflow rounds the same way), so $K_{\rho>0}$ and
   $K_{\mathrm{eff}}$ keep the per-event semantics of section 3.3.

The value differs from the gather's only in summation order and in how the
three products of $\mathbf a\cdot\mathbf b$ are rounded. A narrow density
amplifies the latter ($d\log\rho = (\theta - \bar\theta)/\sigma^2\,d\theta$;
Lowitz's roll of a nearly vertical c axis has $d\psi \sim \epsilon /
\sin\theta$): a value keeps `1e-12`, a $K_{\mathrm{eff}}$ near one (one or
two events) carries that error undiluted, `3e-12` at worst in the unit
tests. Nothing here writes "a pixel is one $\delta$ band" into the walk:
a pixel is an index range of a sorted store, so the divergent-light form of
section 9 ($D \ge \theta$) is another range.

**Measured** (appendix). The canonical strip at $N = 10^8$: `30.8 s` on four
Mac workers (gather `168.9 s`), `73.7 s` single-process; worker RSS
`441 MB` (the mapped stretch of its segment, shared page cache) against
`1175 MB` per worker; every pixel's $K$ and $K_{\rho>0}$ equal, values
within `7.9e-15`. With many store groups the peak no longer adds up: twelve
$\Phi$ groups, one store each, peak at `119 MB` physical footprint against
`111 MB` for one group, while loading them the gather's way takes
`1285 MB`. A sparse pixel set (a one-pixel-wide profile, bands that do not
overlap) is the gather's best case and runs at the same speed; the scatter
wins where pixels share events. A GPU back end is out of scope; on the CPU
the products (inner dimension three) are cheap next to the elementwise
$\rho$ (`0.14 s` against `3.4 s` for `arccos` alone on the same
$256\times4096$ grid, 200 repeats), so a further factor would come from
$\rho$, not from the products.

## 9. Divergent light

A nearby point source (a street lamp at $L$) breaks one assumption, and it
is not in the crystal kernel. Under parallel light every crystal on a view
ray sees the same (incident, outgoing) pair, so a pixel is one pose
integral. Under a point source the incident direction changes along the
ray. The fields and the store (section 1.1) are reused unchanged; only the
image formation changes.

**Geometry (derived; the closed forms checked numerically on 2000 random
$(\theta, t)$, finite-difference agreement `6e-6`).** Let the observer be at
$O$, $d = |OL|$, the view ray $\mathbf x(t) = O + t\mathbf v$ at angle
$\theta$ from the lamp, and $\beta(t)$ the angle at $L$ in the triangle
$OL\mathbf x$. The deviation of the ray scattered at $\mathbf x$ toward $O$
is the exterior angle $\delta(t) = \theta + \beta(t)$, growing monotonically
from $\theta$ (crystals at the observer) to $\pi$ (far behind,
back-scattering). The law of sines gives

$$
t(\delta) = \frac{d\,\sin(\delta-\theta)}{\sin\delta},\qquad
r(\delta) = |\mathbf x - L| = \frac{d\,\sin\theta}{\sin\delta},\qquad
\frac{dt}{d\delta} = \frac{d\,\sin\theta}{\sin^2\delta}.
$$

For an isotropic source of intensity $J$, single scattering and no
extinction, the pixel radiance is
$\int n\,J/r^2\,I_{\hat{\mathbf s}(t)}(\delta(t),\alpha)\,dt$, with $n$ the
crystal number density and $I_{\hat{\mathbf s}}$ the parallel-light value
for a source in the local direction toward the lamp. In the variable
$\delta$ the factor $(dt/d\delta)/r^2 = 1/(d\sin\theta)$ is **constant along
the ray**:

$$
L(\theta,\alpha) = \frac{J}{d\,\sin\theta}\int_{\theta}^{\pi}
  n\big(\mathbf x(\delta)\big)\, I_{\hat{\mathbf s}(\delta)}(\delta,\alpha)\,d\delta ,
$$

where $d\sin\theta$ is the distance from the lamp to the view line. The
scattering plane (through $O$, $L$ and the ray) is the same for every $t$,
so the azimuth frame is fixed along the ray; only the source direction
turns within that plane, and it matters only through $\rho$.

**Band sum (derived).** With section 2's identity the $\delta$ integral is a
sum over **all** events with $D_i \ge \theta$, each at the unique point
$t_i = t(D_i)$ of the ray, with no band width and no ray marching:

$$
\hat L(\theta,\alpha) = \frac{J}{d\,\sin\theta}\cdot\frac{1}{2\pi N}
  \sum_{i:\,D_i\ge\theta} \frac{n(\mathbf x(D_i))\, w_i\,\rho(R_i)}{\sin D_i},
$$

$R_i$ built as in section 5 with $\hat{\mathbf s}$ replaced by the direction
from $\mathbf x(D_i)$ to the lamp. The locus of fixed $D$ in space is
Minnaert's spindle ("cigar") about $OL$; the note's contrast between ray
marching and Gislén's cigar method disappears in the event form.
$\sin D_i \to 0$ near back-scattering is $t \to \infty$, cut off by a cloud
of finite extent.

- Random density and a uniform cloud:
  $L \propto \frac{1}{\sin\theta}\int_\theta^\pi I(\delta)\,d\delta$, a
  cumulative integral of the sun halo's radial profile; with events sorted
  by $D$, a suffix sum, one lookup per pixel.
- Per pixel the event set is every $D_i \ge \theta$ rather than one band;
  the cost rises by about the deviation range over the band width (the note
  estimates `10-100×`), which makes section 8 more valuable.
- The contour route integrates every level set $\delta \ge \theta$: the
  coarea step runs backward, the pixel is an area integral over
  $\{D_P \ge \theta\}$ on $S^2$, and $1/\lvert\nabla D_P\rvert$ and the fold
  singularity disappear; edges come from the limit $\theta$.
- Phase I takes any (incident, outgoing) pair but needs a fibre per node of
  a $t$ quadrature: a pointwise reference, not a renderer.

**What is hard is not the algorithm**: a scene model this project does not
have (lamp, observer, cloud $n(\mathbf x)$ and extent, extinction, their
conventions); **no oracle** (Lumice's `light_source.type` is `"sun"` only;
the checks available are the far-lamp limit, an in-project brute-force
Monte Carlo that shares the kernel, and Gislén's papers qualitatively);
finite source size (the sun's 0.5° too) is a convolution on the sky, a
separate axis. Trigger: Lumice gains a point source, or the writing series
needs street-lamp halos (backlog).

## 10. Open questions and where they are settled

- **22° inner edge** (writing chapter 10). For a random orientation the
  minimum of $D_P$ on $S^2$ is isolated and non-degenerate, and
  $\int d\ell/\lvert\nabla D\rvert$ near a two-dimensional minimum is finite:
  the inner edge would be a finite jump, and the
  $I \sim 1/\sqrt{D - D_{\min}}$ profile would belong to families that
  confine $\mathbf u$ to a curve (columns, tangent arcs). Chapter 10's
  statement should be an acceptance test, not a premise. M2 sub-task
  `ch10-numerical-verdicts`.
- **Rank-deficient maps.** $W = 0$ classes are point masses in the source
  direction (task `path-class-rendering-unit`); the degenerate images of
  parallel-face classes ($M \ne I$, $W = I$) come from $\rho$ confining
  $\mathbf u$, not from $\Phi$, and need their own accounting ("dimension
  collapse" vs "Jacobian focusing", as an explicit solver output).
  `ch10-numerical-verdicts`.
- **Non-uniform $\rho$.** $\psi(\mathbf u,\alpha)$ is single-valued, so $\rho$
  is evaluated pointwise; only the "convolution on the sky" reading of
  chapter 11 needs a uniform $\rho$.
- **Jacobian alignment.** $1/\lvert\nabla_{S^2} D_P\rvert$ against Phase I's
  $J_\perp$ under the fibration's change of coordinates is the
  cross-validation contact point. `s2-contour-quadrature`.
- **Absolute scale against Lumice** after its projected-area fix (Ice Halo
  #597): done 2026-09-24 (task `lumice-area-weighting-recheck`): band-sum
  class renders of the plate and Parry families against Lumice float
  exports with $K_p = \bar y(550)\,\Omega_p/(S/2)$, nothing fitted
  (`docs/ch06-reference-fixture.md` section 7, stage 4).

| piece | status | where |
|---|---|---|
| band sum, event store, $D_{6h}$ transport, $K_{\mathrm{eff}}$ | measured, in production | appendix; tasks 13-19 |
| store independent of the source; `.npy` + mmap; bucketed build | measured, in production | appendix; task 21 `s2-store-schema-3` |
| band sum by deviation (segments, class accumulation, GEMM) | design | task 22 `band-sum-scatter-renderer` |
| contour quadrature, critical points, certificate | design | scrum 24 `phase2-contour-quadrature` |
| Phase I seeds and cross-check from the store | design | scrum 24 sub-task 5 |
| chapter-10 verdicts | open | scrum 24 sub-task 6 |
| divergent light | derived | backlog |

## Appendix: measured record

Moved verbatim from `docs/roadmap.md` §4.2 on 2026-09-24 (the dated
entries of tasks `band-sum-quadrature-probe`, `narrow-density-band-sum-probe`,
`s2-event-store`, `band-sum-renderer`, `band-sum-full-symmetry`). Section
references inside refer to the roadmap numbering of that time: "section
4.1(x)" is sections 1-4 above, "section 4.2" is section 5.

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
  (\mathbf b + \hat{\mathbf s})$; on the sun's vertical (column 126) this puts the
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
  three twists about $\hat{\mathbf s}$ to `1.5e-14` (section 4.1(a)); all three
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

**Narrow densities: plate, Parry, Lowitz (2026-09-23).** The scenes above
never saw a narrow family: on the labelled path 3-5 plate, Parry and Lowitz
are exactly zero (`docs/ch11-pose-density-families.md` section 5.1). The
follow-up probe (`scripts/probe_band_sum_narrow.py`, task
`narrow-density-band-sum-probe`) renders the ray-path class `[3,5]`: one
Fibonacci event store per PBD member (12 stores, rank 2), and the class
value is the band sum of the pooled member contributions (every member
shares the pixel's band and constant). Same crystal, index and sun as
above; Lumice preset widths: plate zenith std 1°, Parry zenith 1° and roll
1°, Lowitz zenith 40° and roll 1°. A wide band-sum render at $N = 10^7$
(0.6°/px, then 0.1°/px zooms) placed one profile per family at the ch06
pixel scale (0.024°): plate, a horizontal line through the right parhelion
(121 px: dark, inner edge, peak, 2.4° of tail); Parry, a vertical line at
azimuth 15° across the upper suncave Parry arc (181 px); Lowitz, a
horizontal line at elevation 10.8° across the sharp inner edge, the
Lowitz-arc peaks and a second arc crossing (151 px). Reference: Phase I
`render_class_pixel` on every pixel, all `complete`; a refined quadrature
(`rtol 1e-6`, 513-4097 nodes) on six lit pixels per family moved values by
at most `6e-5`. Where the reference changes by more than 10 % between
neighbouring pixels (a criterion on the reference alone: 10 / 18 / 21
pixels), the reference is the Phase I *band average* (eight midpoint
targets across the pixel's deviation band at its azimuth), because that is
the band sum's pixel. There the band-average-to-point ratio reproduces the
band sum's constant offset from the point value pixel by pixel (plate
inner edge `1.0267` vs `+2.76 %`, Lowitz inner edge `1.1344` vs
`+13.5 %`, `0.958` vs `-4.2 %`): the pixel model, not sampling. Lit =
reference above `1e-2` of the profile maximum.

| family | $N$ | median $K_{\mathrm{eff}}$ (lit) | $K_{\mathrm{eff}}/K$ | RMS rel. error | median \|rel.\| | p95 \|rel.\| | max \|rel.\| | RMS vs point ref. | median ratio |
|---|---|---|---|---|---|---|---|---|---|
| plate (113 lit px) | $10^6$ | 115 | 0.029 | 2.0e-2 | 1.1e-2 | 4.5e-2 | 7.5e-2 | 2.0e-2 | 0.9974 |
| | $10^7$ | 1170 | 0.030 | 3.2e-3 | 2.1e-3 | 6.4e-3 | 8.6e-3 | 4.7e-3 | 0.9997 |
| | $5\times10^7$ | 5842 | 0.030 | 1.3e-3 | 8.5e-4 | 2.7e-3 | 4.2e-3 | 3.7e-3 | 1.0001 |
| | $10^8$ | 11665 | 0.030 | 8.2e-4 | 6.0e-4 | 1.5e-3 | 2.2e-3 | 3.6e-3 | 1.0001 |
| Parry (135 lit px) | $10^6$ | 13 | 0.0045 | 8.7e-2 | 5.7e-2 | 1.7e-1 | 2.2e-1 | 8.7e-2 | 0.9928 |
| | $10^7$ | 127 | 0.0044 | 8.4e-2 | 3.4e-2 | 1.9e-1 | 2.6e-1 | 8.4e-2 | 0.9938 |
| | $5\times10^7$ | 628 | 0.0045 | 4.5e-3 | 2.7e-3 | 8.8e-3 | 1.6e-2 | 4.5e-3 | 0.9998 |
| | $10^8$ | 1255 | 0.0045 | 4.4e-3 | 1.9e-3 | 8.0e-3 | 2.9e-2 | 4.4e-3 | 1.0003 |
| Lowitz (137 lit px) | $10^6$ | 27 | 0.0059 | 7.4e-2 | 4.9e-2 | 1.3e-1 | 2.3e-1 | 7.5e-2 | 0.9972 |
| | $10^7$ | 259 | 0.0058 | 9.4e-3 | 4.6e-3 | 2.1e-2 | 3.3e-2 | 1.6e-2 | 0.9990 |
| | $5\times10^7$ | 1294 | 0.0058 | 3.4e-3 | 1.9e-3 | 6.5e-3 | 1.3e-2 | 1.4e-2 | 1.0003 |
| | $10^8$ | 2599 | 0.0058 | 3.7e-3 | 1.5e-3 | 9.2e-3 | 1.4e-2 | 1.4e-2 | 0.9996 |

- *$\rho$ collapses $K_{\mathrm{eff}}/K$, not the estimate.* The band keeps
  3.0 % (plate), 0.45 % (Parry) and 0.58 % (Lowitz) of its events, against
  65 % / 17 % for the column density above. $K_{\mathrm{eff}}$ still grows
  as $N^{1.00}$: $K_{\mathrm{eff}}/N = 1.2\times10^{-4}$ (plate),
  $1.25\times10^{-5}$ (Parry), $2.6\times10^{-5}$ (Lowitz) at the median lit
  pixel, worst pixel $7.2\times10^{-5}$ / $1.2\times10^{-5}$ /
  $1.6\times10^{-5}$ — 6-60× above the owner's prior $2\times10^{-6}N$
  and never collapsing to zero on a lit pixel. The derived constant needs
  no fit: median ratio `0.9996-1.0003` at $10^8$, mean relative error at most `5e-4` in magnitude.
- *The lattice beats $1/\sqrt{K_{\mathrm{eff}}}$, but not reliably.* An
  i.i.d. uniform store (same estimator, $10^7$ and $5\times10^7$) measures
  median $|\mathrm{rel}|\sqrt{K_{\mathrm{eff}}}$ = `0.53-0.79` (random
  sampling: about `0.67`), so $K_{\mathrm{eff}}$ is the right Monte Carlo
  ruler. The Fibonacci store usually reaches `0.07-0.10` (7-9× better)
  but aliases: Parry at $10^7$ shows a sawtooth in $K_{\mathrm{eff}}$ and in
  the error along the profile, `0.40`, no better than random (RMS `8.4e-2`
  vs `8.9e-2`). The error is therefore not monotone in $N$ (Parry: flat
  $10^6 \to 10^7$, then 19× down by $5\times10^7$; Parry and Lowitz flat
  again $5\times10^7 \to 10^8$). Per-pixel errors of different tiers are
  uncorrelated (coefficients -0.05 to 0.08) and average to zero: lattice
  noise, not a floor. Correction 1 above ("no aliasing seen") holds for the
  column scenes only.
- *$N$ for a `1e-2` lit RMS.* Power laws over the four tiers (slopes
  -0.69 / -0.74 / -0.68): $2.5\times10^6$ (plate), $3.7\times10^7$
  (Parry), $1.4\times10^7$ (Lowitz), each confirmed by a measured tier.
  Without the lattice gain (error $1/\sqrt{K_{\mathrm{eff}}}$), $N =
  10^4/(K_{\mathrm{eff}}/N)$: $0.9\times10^8$ / $8.0\times10^8$ /
  $3.8\times10^8$ at the median pixel, $1.4\times10^8$ / $8.3\times10^8$ /
  $6.4\times10^8$ at the worst — all below $10^9$, so no family collapses
  and the rho-aware store (importance sampling of $\mathbf u$ by the
  family's marginal) was not tried.
- *Cost.* Windowed precompute (only events with $D$ in the profiles'
  union band 21.6°-32.9°, bit-identical for these pixels): 87-92 s per
  member at $10^8$, 4.5 min for 12 members on 4 processes (2.3-2.6 GB each;
  together above the 8 GB budget of the task, recorded). The wide render
  (32761 px × 12 stores at $10^7$, three densities at once) took 9 min;
  a profile at $10^8$ 5-7 s including loading 12 stores. Phase I class
  pixels: 1.9-2.2 s each. The probe's compute totalled about 45 min.
- *Scope.* One profile per family, the sun at 15°, the class `[3,5]`
  only; the learnings of the column scenes (column 126 was an easy case)
  say a single line can be optimistic, so the verdict below is per family
  at these settings, with the worst pixel of each profile reported.

Verdict: the band sum qualifies as a **rendering backend for plate,
Parry and Lowitz** on the class `[3,5]`, at $N = 10^8$ Fibonacci points per
member store (lit RMS `8e-4` / `4e-3` / `4e-3`, confirmed; $\le 10^9$ even
without the lattice gain), with the same caveats as above: no completeness
certificate, a band-average pixel (steep edges differ from a point pixel by
up to `13 %` on the Lowitz inner edge, reproduced by Phase I band averages),
and lattice aliasing that makes the error non-monotone in $N$ — size the
store by $K_{\mathrm{eff}}$ ($10^4$ at the worst pixel of interest), not
by the lattice's typical gain. Narrow $\rho$ is not the structural failure
correction 2 feared at these widths: it removes 97-99.5 % of the band,
but the band holds enough events. A rho-aware store is not needed for
these families. Figures and tables:
`scratchpad/task-narrow-density-band-sum-probe/artifacts/` (local).

**The store is in `src/` (2026-09-23, task `s2-event-store`).**
`lumice_integral.s2_store` builds, caches (parameter-hashed directory with a
provenance JSON; a mismatch or a modified file is refused, never silently
rebuilt or reused) and slices the event store; it rebuilds the task 13
stores ($N = 10^6, 10^7$) bit for bit, and both probe scripts now call it.
`path_class.phi_key` groups face sequences by their $\Phi$ (`3-5` and
`3-1-2-5` share one; a group store sums $w_\Phi = \sum_m w_m$ on the same
$\mathbf u$), and `path_class.path_class_symmetry` gives each class member
a $D_{6h}$ element $g$, proper or improper, that serves it from the
representative's store (section 4.1(d); mirrors since task
`band-sum-full-symmetry`, below). Evidence: transported events equal each `[3,5]` member's own store on the
$g$-rotated lattice to `5.2e-13` ($N = 10^6$); on the three task 14
profiles at $N = 10^7$ one store with pose factors equals twelve member
stores to `6.9e-14` relative. The Fibonacci lattice is not closed under
$g$, so against task 14's twelve independent lattices the class sums agree
only at the discretisation level (sum ratio `1.0006` / `1.0015` / `1.0011`,
per pixel `0.89-1.06`). float32 halves the memory but moves the worst
column-126 pixel by `5.5e-3`; float64 stays the default. The probe
scripts' flat `events_N<n>.npz` layout and the library's cache directories
coexist on purpose (the historical artifacts stay readable); the store
format itself has one implementation.

**The production renderer (2026-09-23, task `band-sum-renderer`).**
`lumice_integral.band_sum` holds the estimator (migrated verbatim from the
task 13 probe, which now imports it: the task 13 column-126 metrics are
reproduced bit for bit at $N = 10^6, 10^7, 10^8$) and renders a path or a
PBD class; `scripts/render_band_sum.py` is its CLI (the `strip_io` layout,
so `read_strip` and `compare_strip_v2.py` read it, plus per-pixel $K$,
$K_{\rho>0}$ and Kish $K_{\mathrm{eff}}$, counting distinct events since
task `band-sum-full-symmetry`). A class is grouped by $\Phi$ first, and one
store serves every $\Phi$ group (`band_sum.store_plan`; the class is one
$D_{6h}$ orbit, mirrors included, below). A rank-0 class is the task 9 point
mass on the sun pixel. The stores are built once in the parent, spawned
workers load them and render whole columns. Evidence (canonical scene):

- *Full image vs `artifacts/strip-full`* ($N = 10^8$, 120 947 lit
  pixels): median $|\mathrm{rel}|$ `3.6e-3`, p95 `3.3e-2`, RMS `1.5e-2`,
  log RMS `1.7e-2`, median ratio `0.99997`, lit sum ratio `0.9996`
  ($N = 10^7$: median `1.9e-2`). Per-column medians range `1.2e-3` (column
  126, as in task 13) to `1.2e-2`, quartiles `2.7e-3` / `4.6e-3` /
  `8.7e-3`: columns away from the sun vertical have a smaller
  $K_{\mathrm{eff}}$. Attribution: every lit pixel either has
  $|\mathrm{rel}|\sqrt{K_{\mathrm{eff}}} \le 4$ or a reference that
  changes by more than 10 % to a neighbour; the 27 pixels beyond the noise
  are all of the latter kind and keep their error from $10^7$ to $10^8$
  (the band-average pixel model); the worst, row 56 at `-88 %`, has half
  its band below $\min D_P$ (the inner-edge caustic above). The `+15 %`
  outliers in the bottom corners are the right tail of a skewed sampling
  distribution: over the 14 585 lit pixels with $K_{\mathrm{eff}} < 1000$
  the mean error is `-1.5e-5` and the sum ratio `0.99998`, while the median
  is `-3.9e-3` (skewness `0.8`).
- *Task 14's profiles* (class `[3,5]`, plate / Parry / Lowitz, 453
  pixels): on task 14's own twelve member stores (`--no-symmetry-transport`,
  the same points) the renderer equals the probe's frozen estimates bit for
  bit at $N = 10^6, 10^7, 5\times10^7$. With one store and pose factors
  (different points) at $N = 10^8$ the peak-pixel sum ratios are `0.9998`
  / `1.0002` / `1.0003`, per pixel `0.979-1.019`.
- *Cost* (M2 Max, 4 workers, `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1`): the
  251 × 801 image at $N = 10^8$ in `2.8 min` with the store cached (3.3 ms
  CPU per pixel, about 1.2 GB per process); building that store `80 s`
  once. At $N = 10^7$: `35 s`. Phase I: `34.7 min` on 30 workers.

Reports and the three example images (canonical strip; class `[3,5]` with
plate and with Parry densities):
`scratchpad/task-band-sum-renderer/artifacts/` (local), from
`scripts/regress_band_sum.py`.

**Full $D_{6h}$ and the precomputation view (2026-09-23, task
`band-sum-full-symmetry`).** The representative's store on all of $S^2$
is the same information as every member's store on a fundamental domain
$F = S^2/G$: the representative's field on the block $hF$ is the field of
the member $h^{-1}Ph$ on $F$. "Transport one event to $|G|$ images" and
"compute $1/|G|$ of the sphere" are the same thing, so symmetry saves
repeated evaluation and makes no new samples; the precision is set by the
number of distinct precomputed events that fall in the band with non-zero
weight.

- *Mirrors transport on $S^2$.* $w_{gPg^{-1}}(g\mathbf u) = w_P(\mathbf u)$,
  $\Phi_{gPg^{-1}}(g\mathbf u) = g\,\Phi_P(\mathbf u)$ and the valid domain
  is the same, for all 24 elements (tested to `1e-12` on `3-5`, `1-3`,
  `3-1-2-5` at $h/a = 2$). The pose of a transported event is rebuilt from
  $(g\mathbf u, g\Phi, D)$ by the two frames of `event_rotations`, a
  rotation whatever $\det g$; "a mirror needs its own store" was the SO(3)
  restriction of Phase I ($R g^{-1}$ must be a rotation), not an $S^2$ one.
  On the representative's pose $R$ the rebuild is
  $L_g R g^{\mathsf T}$, $L_g = I - (1 - \det g)\,\mathbf m\mathbf m^{\mathsf T}$
  with $\mathbf m$ the normal of the plane of $\hat{\mathbf s}$ and the pixel
  centre: the old pose factor for a proper $g$ (bit for bit), times the
  reflection in that plane for a mirror (`s2_store.transported_rotations`,
  equal to the literal rebuild to `1e-13` on all 24 elements; calling
  `event_rotations` per transport would cost `1.9x` on a class pixel).
  `store_plan` is therefore one store per class: all 2368 rank-2 classes of
  up to five faces at $h/a = 2$ and $0.3$; `[1,3,5,2]` (four $\Phi$ groups,
  two reached only by mirrors) went from two stores to one.
- *$K_{\mathrm{eff}}$ counts distinct events.* Per event the transports are
  summed first, $c_i = w_i \sum_g \rho(R_i^{(g)})$, and $K$,
  $K_{\rho>0}$, $K_{\mathrm{eff}}$ are of $\{c_i\}$ (provenance
  `options.k_eff_semantics = "per_event"`; a render without the field
  pooled every (event, transport) pair, `per_transport_sample`). Values
  are unchanged bit for bit: the canonical $N = 10^8$ image including its
  `pixels.csv`, the class `[3,5]` plate and Parry example images at $10^7$,
  and task 14's profiles at $10^8$. The pooling overstated
  $K_{\mathrm{eff}}$ by the number of images with the same $\rho$: on the
  example images exactly `6x` (plate) and `2x` (Parry) from the 5th to the
  95th percentile; on task 14's peak pixels at $10^8$ the per-event median
  is `1956` / `624` / `2961` (plate / Parry / Lowitz), the pooled one
  `6.0x` / `2.0x` / `1.0x` that -- the author's count of `6/12`, `2/12`,
  `1/12` images with $\rho(Rg^{-1}) \equiv \rho(R)$. For plate the
  per-event size equals one member's: the six $\rho$-equal members only
  scale $c_i$. Task 14's twelve independent member stores do hold `6x` as
  many independent samples for plate (twelve lattices, twelve times the
  computation), which is why task `band-sum-renderer`'s single-store plate
  comparison was noisier than the twelve-store one: unequal computation,
  not a transport defect.
- *The ruler.* With two i.i.d. stores (`RandomSphereSampler`, $N = 10^7$,
  seeds 1 and 2) Kish is the Monte Carlo noise predictor, and
  $z = (a - b)/\sqrt{a^2/K_a + b^2/K_b}$ on task 14's profiles has RMS
  `1.20` / `0.85` / `0.93` with the per-event $K_{\mathrm{eff}}$ (plate on
  seeds 3/4 and 5/6: `1.01`, `0.97`), against `2.94` / `1.21` / `0.93`
  pooled. On the Fibonacci lattice the class-stage $z$ against task 14
  (combined noise $\sqrt{1/K_{\mathrm{eff}} + 1/K_{\mathrm{eff},14}}$) is
  `0.24` / `0.05` / `0.15`: the lattice beats $1/\sqrt{K_{\mathrm{eff}}}$
  (task 14's `7-9x`), so the `NOISE_Z = 4` attribution with the per-event
  $K_{\mathrm{eff}}$ is conservative; the threshold is unchanged.
- *Cost.* Unchanged within noise (canonical $10^8$ image `171.5 s` vs
  `168.9 s`; the two example images `152 s` / `192 s` vs `162 s` /
  `192 s`). $\rho$ is `13 %` of a class `[3,5]` pixel (the pose products
  are most of it), so merging $\rho$-equal images into a multiplicity is
  not worth doing.

Reports: `scratchpad/task-band-sum-full-symmetry/artifacts/` (local), from
`scripts/regress_band_sum.py --stage class` / `--stage k-eff`;
`tests/test_band_sum.py::test_per_event_k_eff_is_the_iid_noise_of_a_class_band_sum`
(slow) pins the ruler.

**Schema 3: source-independent store, per-array `.npy`, bucketed build
(2026-09-24, task `s2-store-schema-3`).** One M2 Max, `JAX_PLATFORMS=cpu
OMP_NUM_THREADS=1`, canonical `[3,5]` with self-checks; the baseline is the
schema 2 code of the same commit parent rebuilt on the same machine (the
older `79 s` / `3.2 GB` above is another code version). Peak RSS is the
build's own `max_rss_mb_process` (`ru_maxrss` at the end of the build, same
ruler in both), wall clock is scan plus sort (schema 2) or scan plus bucket
merge (schema 3), self-checks excluded.

| $N$ | schema 2 peak RSS / wall | schema 3 peak RSS / wall |
|---|---|---|
| $10^7$ | `1305 MB` / `8.2 s` | `1068 MB` / `10.1 s` |
| $5\times10^7$ | `2339 MB` / `41.9 s` | `1380 MB` / `50.1 s` |
| $10^8$ | `2997 MB` / `94.2 s` | `1529 MB` / `90.2 s` |

- *Bit for bit.* The $N = 10^8$ schema 3 arrays `D`, `u`, `phi`, `w` have
  the SHA-256 of the schema 2 build (1024 buckets; the largest holds
  `263458` events, `1.6 %` of the `16022326` kept). The canonical
  $251\times801$ image re-rendered from a schema 3 store
  (`scripts/render_band_sum.py --store-n 100000000 --workers 4`) equals
  `artifacts/band-sum-full` byte for byte in `strip_float64.bin`,
  `strip_float32.bin`, `status_uint8.bin`, `component_count_uint8.bin` and
  every column of `pixels.csv` (`324 s` wall clock including the `91 s`
  store build).
- *What the peak still is.* The remaining growth with $N$ is not the
  store: `evaluate_fields` alone, keeping nothing, climbs from `362 MB` to
  `1.3-1.4 GB` over the first 150 chunks of $N = 5\times10^7$ and then
  flattens, the same curve as the schema 3 build. The store itself adds
  one bucket (about `19 MB` at $10^8$) instead of all kept events (about
  `1 GB` plus a sorted copy).
- *Wall clock.* Even at $10^8$, `20 %` slower at $5\times10^7$ (system time
  `26 s` vs `15 s`: bucket files appended about 64k times and the data
  written twice); single runs on one machine.

**Scatter renderer (task `band-sum-scatter-renderer`, 2026-09-24).** Section 8's
organisation in production. Mac (M2 Max, 12 cores), `JAX_PLATFORMS=cpu
OMP_NUM_THREADS=1`; the machine was shared with other jobs (load average
5-7 during the timed runs below, 20-45 during earlier ones, which took 2-3×
longer in wall clock at the same CPU time).

| canonical strip, $N = 10^8$, 201051 px | gather (task 16, 2026-09-23) | scatter, 4 workers | scatter, 1 worker |
|---|---|---|---|
| wall clock | `168.9 s` | `30.8 s` | `73.7 s` |
| pixel bands (geometry, part of the wall clock) | inside the pixel loop | `3.4 s` | `9.9 s` |
| segment CPU (sum over workers) | `661 s` | `97 s` | `63 s` |
| max RSS parent / worker | `1187 / 1175 MB` | `416 / 441 MB` | `692 MB` |

- *Agreement with `artifacts/band-sum-full`* (`regress_band_sum.py --stage
  scatter`): every pixel's band, $K$, $K_{\rho>0}$ and lit status equal;
  value within `7.9e-15`, $K_{\mathrm{eff}}$ within `3.5e-15`, no pixel above
  `1e-12`; rows 48-56 (the inner-edge caustic, 794 lit pixels) within
  `7.9e-15`, the same as the rest. Both worker counts give these numbers.
- *Other scenes against the gather*, $N = 10^7$ Fibonacci stores: class
  `[1,3,5]` (24 members, 12 transports improper, random density, a
  $61\times61$ 150° camera): value `1.0e-15`, $K_{\mathrm{eff}}$ `2.1e-15`,
  scatter `0.93 s` against gather `31.7 s`; task 14's plate / Parry / Lowitz
  profiles on class `[3,5]` (one store, 12 transports): value within
  `3.4e-14` / `5.4e-15` / `3.7e-15`, $K$ and $K_{\rho>0}$ equal, `0.28 / 0.71 /
  0.90 s` against the gather's `0.46 / 0.70 / 0.79 s`. The unit tests cover
  all five families on `[3,5]` (proper), `[1,3,5,2]` (improper) and `[3,5]`
  as twelve stores at $N = 10^5$, and two sun altitudes on one store.
- *Why a sparse profile is no faster.* Its bands do not overlap, so an event
  chunk serves one to three pixels and the per-chunk event work (frames and
  their 11 transports) is not shared; profiling found it dominant
  (`1.7 s` of `2.4 s` with an `einsum`; `tensordot` made it 7.5× cheaper).
  Widening chunks where few pixels are live was tried and dropped: the
  masked-out part of each block grew faster than the overhead fell.
- *Tile size.* Chunks of 4096 events × blocks of 256 pixels (8 MB grids)
  doubled the system time against 1024 × 32 (256 kB) at equal user time:
  the large grids were mapped and zero-filled afresh by the allocator.
- *Many store groups* (`scratchpad/task-band-sum-scatter-renderer/probe_group_memory.py`):
  class `[3,5]` with `transport=False`, twelve $\Phi$ groups, one $N = 10^7$
  store each (`1174 MB` of arrays in total), every tenth column of the
  canonical camera (20826 px), one process each. Lifetime maximum physical
  footprint (macOS; clean pages of a mapping are not in it) / max RSS:
  scatter, one group `111 / 306 MB`; scatter, twelve groups `119 / 313 MB`;
  the gather's load of all twelve `1285 / 1359 MB`. The peak is that of one
  group (in fact of the chunk temporaries), not the sum.

**$D_P$ field topology probes (2026-09-24, scrum `phase2-contour-quadrature`,
explores `dp-field-topology`, `dp-field-boundary-corners`,
`dp-field-boundary-deep-internal-faces`, `dp-field-bigon-other-corner-pair`,
`dp-field-exit-tir-marching-generalize`, `dp-field-saddle-search`).**
Scratchpad probes (`src/` untouched) ahead of `task-dp-field-layer`, on the
`R = I` convention with $D_P(\mathbf u) = \arccos(\Phi_P(-\mathbf u)\cdot(-\mathbf u))$.

- *Fold judgement.* $\mathbf n_a\cdot\tilde{\mathbf n}_b = \pm 1$ (entry
  normal vs. the unfolded exit normal) $\iff$ $\Phi_P$ collapses globally to
  a fixed orthogonal map ($D_P(\mathbf u) = \arccos(\mathbf u^{\mathsf T}
  M\mathbf u)$, tangential gradient norm exactly `2`), a zero-cost
  pre-branch that skips lattice-seed Newton search entirely for
  no-interior-fold paths (proved via $\|PM\mathbf u\|^2 = 1-f^2$; `3-5` and
  the `(1,3)` 90° wedge fold, `3-1-6`/parhelic, `1-3-2`, `3-5-6-7-3` do not).
- *$\partial U_P$ has more than the entry/exit pair.* Beyond the entry
  glancing great circle ($\mathbf u\cdot\mathbf n_{\text{entry}} = 0$,
  closed form) and the exit TIR curve, every internal-reflection face
  contributes its own glancing (`internal_k_incidence_cosine = 0`) and
  TIR-broken (`internal_k_tir_discriminant = 0`) candidates. On `3-5-6-7-3`
  three consecutive internal faces whose prism azimuths form an arithmetic
  progression (step 60°) give an exact identity between the first and last
  incidence cosine (max difference `3.3e-16` to `1.2e-15` over $2\times10^4$
  random directions, both on and off $U_P$) — `internal_3_*` duplicates
  `internal_1_*` and must be deduplicated before enumeration, not counted
  as a fourth boundary type. A second, unrelated identity was found on the
  same path: `exit_snell_discriminant ≡ entry_incidence_cosine` as zero
  sets (max `6.66e-16` over the whole entry great circle); the mechanism is
  open (deferred, non-blocking — flagged low priority against the queue).
- *Corners can have algebraic multiplicity `3` with topological multiplicity
  `2`.* Both corner pairs on `3-5-6-7-3`'s entry great circle are exact
  triple points (three margins zero to `≤3.3e-16` by 2D Newton), but the
  third curve is *transversal* at one pair (cuts into the feasible wedge,
  forming a bigon with the known long arc) and *tangent* at the other
  (gradients parallel, `internal_2_incidence_cosine` stays one-signed
  inside the wedge) — algebraic and topological corner multiplicity must be
  distinguished by a local transversality check, not inferred from the
  count of margins vanishing.
- *Marching.* A predictor-corrector (tangent step + Newton correction) on
  `*_tir_discriminant = 0` reaches machine-precision residuals
  (`4e-14`-ish) regardless of target path; a two-stage step law (fixed
  `0.5°` beyond `0.5°` of the target, geometric slowdown
  `step = distance × 0.5` inside it) brings the endpoint error from `4e-3`
  rad down to `<1e-10` rad on four structurally different fixtures (`3-5`,
  `1-3`, `1-3-2`, `3-1-6`) and three margin kinds, unchanged code, provided
  the target corner itself is solved to matching precision (an 8-decimal
  literal from an earlier probe was, at `≈6e-5` rad, imprecise enough to
  look like marching divergence). `*_incidence_cosine`-type boundaries
  are themselves closed-form great circles under `R = I` (interior faces
  included, not only entry) and need no marching at all.
- *No interior saddle found.* A symmetry-deduplicated, fold-judgement- and
  lattice-domain-filtered search over hexagonal-prism paths (entry face in
  `{1, 3}`, up to 4 internal reflections: 7692 distinct classes, 86
  non-empty after two-stage filtering, all 86 exhaustively Newton-checked;
  plus an 800-sample spot check at 5 internal reflections, 1 non-empty)
  found zero interior saddles or multi-critical-point candidates; every
  non-empty $U_P$ tested is a topological disk (see section 4's structural
  argument). Longer paths were not exhaustively covered (non-empty yield
  falls by about one order of magnitude per added reflection); the
  stronger claim "no hexagonal-prism path ever has an interior saddle" is
  open.

Scratchpad: `scratchpad/scrum-phase2-contour-quadrature/explore-dp-field-*/`
(hypothesis.md / experiments.md / insights.md / SUMMARY.md per explore;
local, not part of the source tree).

**$D_P$ field layer (2026-09-24, task `dp-field-layer`,
`lumice_integral.dp_field`).** Refractive index `1.31`, canonical prism,
Fibonacci lattice `N = 20000` for seeds, the disk check and the walk's
start; M2 Max, CPU.

| path | fold dot | interior critical points | corners | critical values (deg) | intervals `(n, closed, open)` |
|---|---|---|---|---|---|
| `3-5` | `-0.5` | minimum `21.839300` (`= 2 asin(n sin 30°) - 60°` to `1e-12`), Hessian `[0.338, 0.965]` | 2 (entry ∩ exit TIR) | 21.839300, 42.990858, 43.465157, 50.062619 | `(1,1,0)`, `(4,0,4)`, `(2,0,2)` |
| `1-3` | `0` | minimum `45.733421` (`= 2 asin(n sin 45°) - 90°`) | 2 (entry ∩ exit TIR) | 45.733421, 57.803628, 73.506892 | `(1,1,0)`, `(2,0,2)` |
| `3-1-6` | `-1` (slab) | none (`±n_M` outside the closure) | 4 | 0, 115.607259 | `(1,0,1)` |
| `1-3-2` | `-1` (slab) | none (`±n_M` outside the closure) | 4 | 0, 115.607259 | `(1,0,1)` |
| `3-5-6-7-3` | `-1` (slab) | `+n_M` (face-3 normal), cone maximum `180` | 4 | 0, 153.069685, 180 | `(2,0,2)`, `(1,1,0)` |

- *Checks.* `D` against `evaluate_fields` on 256 random points of each
  $U_P$: `<= 1e-12` (points within `1e-3` rad of `0` / `π` excluded, where
  `arccos` loses `sqrt(eps)`); slab closed form against the optics chain:
  `<= 1e-12`; gradient against central differences, step scan: plateau
  `<= 1e-9`; the same critical point from lattices of `2000` and `20000`;
  corners to residual `<= 2.3e-16`, and on `3-5` equal to an independent
  bisection on the entry circle with the scalar gates to `1e-10` rad;
  every edge point of the lattice domain within `1.5` lattice spacings of
  the walked loop; `D6h` transport `3-5 → 3-7` and
  `3-5-6-7-3 → 4-8-7-6-4`, proper and improper `g`: corners and interior
  points to `1e-9` rad, values to `5e-8`, identical partitions; all
  intervals against the independent grid (`1201²`, ~45 s for the five).
- *Cost.* Batched on `10^6` points after compilation: `D` `76.6 M/s`,
  tangent gradient `24.1 M/s`, Riemannian Hessian `7.5 M/s`. A whole
  `DPField` (critical points, walk, disk check, partition) `0.4-1.9 s` per
  path.
- *Corrections to the probe record above.* (i) At all four corners of
  `3-5-6-7-3` the third curve is *tangent* to the TIR edge it meets
  (gradient angle `0.0°` at the pair on `internal_2_tir` as at the pair on
  `internal_1_tir`); the bigon between `internal_1_incidence_cosine` and
  `internal_2_tir_discriminant` is a lens between two curves tangent at both
  ends, outside $U_P$; every corner of the five fixtures bounds $U_P$ with
  two edges, and a fourth margin (`exit_snell_discriminant`) vanishes there
  as the square of the entry margin. (ii) An internal incidence cosine is a
  great circle only when $\mathbf m\cdot\mathbf n_a = 0$ (`1-3-2`, `3-1-6`,
  and the exit margin of `1-3`); on `3-5-6-7-3`, $\mathbf n_5\cdot\mathbf n_3
  = -1/2$ and it is marched. (iii) On `3-1-6`, $\pm\mathbf n_M$ lie on the
  entry great circle but `internal_1_tir_discriminant = -0.284` there:
  outside the closure, not on $\partial U_P$. (iv) `3-5-6-7-3` has no fold
  but does have an interior critical point, the non-smooth maximum
  $D = \pi$ at normal incidence on face 3 (the slab set, not found by
  Newton; Newton on the lattice finds nothing on any slab, where
  $|\nabla D_P| = 2$). (v) The 3-5 domain's largest deviation is
  `50.062619°` (the corners), not the `49.16°` of the `N = 200000` lattice.
- *Accuracy limit.* $D_P$ is Hölder-½ across an exit-TIR curve, so values
  on such a boundary piece carry `~1e-8` rad (the square root of the
  `1e-16` margin residual) and the position of a loop extremum on it only
  `~1e-4` rad; corners and interior points are Newton-exact.

**Contour extraction (2026-09-24, task `s2-contour-extraction`,
`lumice_integral.contour`).** Canonical crystal, $n = 1.31$, stores of
`N = 2e5` built in memory, default grid `401²`.

- *Certificate.* On every interval midpoint of the five fixtures and at
  every critical value $\pm10^{-6}$ and $\pm10^{-3}$ rad (including both
  ends of $[\min D_P, \max D_P]$) the extracted (closed, open) counts equal
  the partition's, and each offset pair straddles a count change:

  | path | $\delta$ values | components (closed) | nodes | max $\lvert D-\delta\rvert$, $\lvert\nabla D\rvert\le10^3$ | max overall | nodes above `1e-12` | wall (incl. compile) |
  |---|---|---|---|---|---|---|---|
  | `3-5` | 19 | 35 (5) | 9189 | `3.1e-13` | `7.8e-9` | 14 % | 8.1 s |
  | `1-3` | 14 | 15 (5) | 3416 | `5.0e-13` | `7.9e-9` | 11 % | 5.0 s |
  | `3-1-6` | 9 | 5 (0) | 3103 | `4.4e-16` | `4.4e-16` | 0 | 5.1 s |
  | `1-3-2` | 9 | 5 (0) | 3099 | `4.4e-16` | `4.4e-16` | 0 | 6.1 s |
  | `3-5-6-7-3` | 14 | 15 (5) | 3322 | `4.4e-16` | `4.4e-16` | 0 | 10.1 s |

  The nodes above `1e-12` are all within reach of an exit-TIR curve
  ($|\nabla D|$ up to `2.4e7`), most in the end clusters of arcs (the last
  steps halve towards $\partial U_P$). The loop `1e-6` above the 3-5 minimum
  winds once (tangent-plane angle $2\pi$), radius between `1e-4` and
  `1e-2` rad, at least 72 nodes; the `3-5-6-7-3` loop at $\pi - 10^{-6}$ has
  radius `5e-7` about $\mathbf n_M$ to `1e-6` relative.
- *Independence.* Without the ray seed the 3-5 loop is found by the store
  and grid seeds alone; with those also disabled (`grid=3`, zero band) the
  certificate raises (predicted `(1, 0)`, extracted `(0, 0)`); against a
  partition predicting nothing it raises on the extra loop.
- *Transport.* `3-5 → 3-7` and `3-5-6-7-3 → 4-8-7-6-4`, proper and improper
  $g$, interval midpoints: equal counts, every node of either set on the
  other's polylines (`5 %` of the segment length), except within
  `3.2e-6` rad of an arc end on an exit-TIR piece.
- *Cost* (`benchmarks/benchmark_contour_extraction.py`, path 3-5, store
  `N = 1e6`, M2 Max, one process): 161 equally spaced $\delta$ (the band-sum
  strip's row count): first call `6.3 s` (XLA compilation included),
  steady `1.0 s` (`6.3 ms` per $\delta$), 207 components, `115788` nodes;
  801 $\delta$ (ch06's row count): first call `11.1 s`, steady `6.0 s`
  (`7.5 ms` per $\delta$), 1027 components, `572885` nodes, peak RSS
  `2.5 GB` (the in-memory store included; `1.1 GB` at 161). For scale: the
  ch06 strip took 35 min on 30 workers through Phase I, the band sum 2.8 min
  on 4 workers at `N = 1e8`. Padding every batched call to a power of two
  took the first 161-$\delta$ call from `38 s` to `6.3 s`; a new $\delta$ set of
  another size recompiles only the buckets it has not met.
