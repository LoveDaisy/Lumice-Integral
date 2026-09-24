# One Event Store on $S^2$: Precomputation, Quadratures and Rendering

Design note, 2026-09-24 (author and owner discussion after milestone M1, the
band-sum renderer in production). It gives the overall picture that the
dated entries of `docs/roadmap.md` §4.1–§4.2 build up piece by piece: what
the precomputed event store depends on, which computations consume it, how
their costs scale, how the band sum can be reorganised around the
deviation, and how the same store extends to divergent light.

Scope and authority:

- The roadmap stays the authority for decisions (§9) and measured results.
  This note does not repeat derivations the roadmap already has (the
  coarea identity and the band-sum constant are roadmap §4.2); it cites
  them.
- Notation follows `docs/conventions.md`: $\hat{\mathbf s}$ points toward
  the source, $\mathbf u = R^{-1}\hat{\mathbf s}$ is the source in the crystal
  frame, $D_P(\mathbf u) = \angle(-\Phi_P(-\mathbf u), \mathbf u)$ is the
  deviation field, $w_P = A_P T_P$ the window field, $\psi$ the twist about
  $\hat{\mathbf s}$, $\rho$ the pose density relative to Haar.
- Each statement is marked **measured** (with its evidence), **derived**
  (follows from the stated model, not yet checked numerically) or
  **design** (a proposed organisation, cost not measured). Tasks that turn
  a derived or design statement into a measured one are named.

## 1. What the store depends on

$\mathrm{SO}(3)$ fibres over $S^2$: a pose $R$ is the base point
$\mathbf u = R^{-1}\hat{\mathbf s}$ plus the twist $\psi$ about
$\hat{\mathbf s}$, and Haar probability splits as
$dA(\mathbf u)/4\pi \cdot d\psi/2\pi$ (roadmap §4.2). Everything a crystal
does to one ray depends on $\mathbf u$ only (roadmap §4.1(a)): the entry
measure $A_P$, the Fresnel product $T_P$, the validity gates, the outgoing
direction $\Phi_P(-\mathbf u)$ in the body frame and hence $D_P$. The twist
rotates the outgoing direction rigidly about $\hat{\mathbf s}$ and changes
none of them. Only $\rho$ sees the whole pose.

So an event $(\mathbf u_i, \Phi_i, D_i, w_i)$ is a pure
"incident direction to outgoing direction" record in the crystal frame, and
**the store depends on the crystal shape, the path (its $\Phi$ group), the
refractive index (the wavelength) and the sampling ($N$, lattice or
sampler, deviation window, dtype), and on nothing about the light source**.

- **Measured** (owner probe, 2026-09-24,
  `scratchpad/task-s2-store-schema-3/owner_probe_sun_indep.py`): $h/a = 2$,
  $n = 1.31$, $N = 2\times10^5$, sun at (15°, 0°), (60°, 37°), (−30°, 200°);
  paths `[3,5]` and `[1,3,2]`: the same event count, $\mathbf u$ equal bit
  for bit, $\Phi$, $D$, $w$ within `1.6e-11`.
- Consequences: a new sun elevation or azimuth, a new pose density, a new
  camera reuse the store and only re-render; a new wavelength needs its own
  store (a multi-wavelength render is one store per wavelength); a new
  crystal shape or path needs its own store.
- Gap: `S2StoreSpec` (schema 2) still records `sun_direction` and puts it in
  the cache key, so every sun elevation rebuilds a store (`80 s` and
  `978 MB` at $N = 10^8$). Roadmap §4.2 "Division of labour" already states
  the sharing; the code does not honour it. Task `s2-store-schema-3`
  removes the field (schema 3).

The sun enters the rendering only through the pose that places an event at
a pixel, $R_i\mathbf u_i = \hat{\mathbf s}$ with the outgoing direction at
the pixel's azimuth (`s2_store.event_rotations`), and through $\rho(R_i)$.

## 2. One precomputation, three consumers

Phase I also precomputes. `prescan.PrescanTable` draws $4\times10^6$ Haar
poses on $\mathrm{SO}(3)$ for a fixed sun, keeps the domain-valid ones with
their outgoing directions, and indexes those with a k-d tree; a pixel asks
for the samples whose outgoing direction lies in a cap of radius
`angle_tolerance` around its own direction, and uses them as Newton seeds
for fibre tracing.

The two are the same sampling. A Haar sample $R$ is a pair
$(\mathbf u, \psi)$; its outgoing deviation is $D_P(\mathbf u)$ and its
azimuth is fixed by $\psi$, which has a closed form given $\mathbf u$ and
the target azimuth. The prescan table samples $\psi$ at random and keeps
the samples that happen to land near the pixel; the store quotients $\psi$
out and constructs, for every event in the pixel's $\delta$ band, the pose
that lands **exactly** on the pixel's azimuth (`band_sum.band_poses`). The
store is the strictly stronger object:

| | Phase I prescan table | $S^2$ event store |
|---|---|---|
| samples | Haar poses on $\mathrm{SO}(3)$, random | points $\mathbf u$ on $S^2$, Fibonacci lattice |
| per pixel | the samples in a cap around the pixel direction | the events in a $\delta$ band, each at the pixel's exact azimuth |
| residual of a candidate | two-dimensional (deviation and azimuth) | one-dimensional (deviation only) |
| index | k-d tree on outgoing directions | one sort by $D$, `searchsorted` |
| depends on | sun direction, path, index; not the crystal (infinite prism; `entry_measure` gated later) | crystal, path, index; not the sun |
| keeps | domain-valid samples | $w > 0$ events |

At equal cost the store gives more usable candidates per pixel (a band
event serves every azimuth on its ring, a cap sample only its own
neighbourhood) and a seed closer to the fibre. It has three consumers:

1. **The band sum** (roadmap §4.2, production since task
   `band-sum-renderer`): the events are quadrature nodes.
2. **Phase II contour tracing** (scrum `phase2-contour-quadrature`,
   M2): a band event lies within half a band width of the level set
   $\{D_P = \delta\}$; it is a seed for Newton onto the contour.
3. **Phase I seeds** (sub-task `phase1-seeds-from-store`): band poses
   replace the prescan candidates; `PrescanTable` and its k-d tree go if a
   32-pixel probe shows no component lost. The same events also give Phase
   I a completeness cross-check it does not have (contract C11 is
   procedural): every band event should lie near one of the traced fibres,
   and an event far from all of them is a missed component. The check is
   statistical (a component with no event in the band is missed), but its
   miss probability is bounded by $N$ times the component's measure in the
   band, which the density survey behind `DEFAULT_SAMPLE_COUNT` does not
   give. The certificate proper remains Phase II's critical-point analysis
   (roadmap §4.1(c)).

Because the store does not depend on the sun, consumer 3 also removes the
per-sun rebuild that the prescan table needs.

## 3. Ring invariance and the cost of each route

Let $Q$ be a rotation about $\hat{\mathbf s}$. If a pose $R$ sends the ray
to $\mathbf d$, then $QR$ sends it to $Q\mathbf d$, with the same
$\mathbf u = (QR)^{-1}\hat{\mathbf s} = R^{-1}\hat{\mathbf s}$. So the fibre
of the pixel direction $Q\mathbf d$ is $Q$ times the fibre of $\mathbf d$,
and all pixels at one deviation $\delta$ (one ring about the sun) share
one level set $\{D_P = \delta\}$ on $S^2$, one window along it and one
$1/\lvert\nabla D_P\rvert$. The azimuth $\alpha$ enters only through
$\rho(R(\mathbf u, \psi(\mathbf u, \alpha)))$. Phase I retraces the same
curve at every azimuth because it works on $\mathrm{SO}(3)$ pixel by pixel.

| route | once, resolution-independent | per $\delta$ ring | per pixel | accuracy |
|---|---|---|---|---|
| Phase I (SO(3) continuation) | prescan table | — | discovery, trace, integrate: `0.1-0.3 s` (measured) | pointwise, adaptive error estimate |
| band sum | event store, $N$ events (`80 s` at $10^8$, measured) | — | $K$ band events: pose + $\rho$, `3.3 ms` CPU (measured) | band average in $\delta$, point in $\alpha$; sampling error $\sim 1/\sqrt{K_{\mathrm{eff}}}$ |
| contour (M2) | $D_P$ field and critical points | extract and refine the level set | $\rho$ along the stored nodes | pointwise, deterministic, high order |

The contour row is **design**; its per-pixel cost (node count, sub-ms or
ms) is to be measured by sub-task `s2-contour-quadrature`.

Scaling with resolution:

- Phase I: cost ∝ pixels × a large constant.
- Band sum: cost ∝ pixels × $K_{\mathrm{target}}$, a small constant. The
  band width follows the pixel size, so keeping $K_{\mathrm{eff}}$ fixed
  when the linear resolution doubles needs twice the $N$: the store grows
  linearly with resolution. With the current renderer that growth lands in
  memory (section 4); after section 4 it lands on disk.
- Contour: the expensive part (finding curves) ∝ rings ∝ linear
  resolution, batched as a field computation; only the final line integral
  ∝ pixels. No sampling noise, and no store that grows with resolution
  (the store only seeds).
- A $\rho$ invariant about $\hat{\mathbf s}$ (the random density) makes the
  pixel value a function of $\delta$ alone: both Phase II routes need one
  number per ring, independent of the azimuthal resolution. The other four
  families of chapter 11 refer to the c axis and are not invariant about
  $\hat{\mathbf s}$ unless the sun is at the zenith.
- The two differ in what a pixel means: Phase I and the contour route give
  the value at the pixel centre; the band sum averages over the pixel's
  $\delta$ extent (the worst M1 regression pixels at the inner-edge
  caustic are this difference, not an error). Cross-validation compares
  them in one convention.

## 4. Organising the band sum by deviation

The production renderer is a *gather*: for each pixel, find its band,
`searchsorted` the events, rebuild their poses at the pixel azimuth,
evaluate $\rho$, sum. Jobs are whole columns, and a column spans the whole
$\delta$ range, so every spawned worker loads every store of the plan in
full (`band_sum._worker_init`); the store build keeps every $w>0$ event in
memory before one `argsort` (`s2_store.build_event_store`). The memory
ceiling of section 3 comes from this organisation, not from the band sum.

Turning the loops inside out (the author's proposal, likened to swapping
the loop indices of a matrix product) keeps the sum and changes the order
of work: organise events by deviation, find the pixels a range of
deviations serves, and let a batch of events scatter into all of them.

**Memory (design; task `s2-store-schema-3` for storage, task
`band-sum-scatter-renderer` for rendering).**

- Store: one `.npy` per array (a `.npz` cannot be mapped partially),
  opened with `mmap_mode="r"`; the build buckets events by $D$ on disk in
  one pass over the lattice and sorts per bucket.
- Rendering: sort pixels by band centre, cut the $D$ axis into segments
  (padded by the largest band width), and give each worker a
  $\delta$ segment instead of a column. A worker touches only its segment's
  pages, so the total is about one store, not one store per worker.
- Classes: a pixel value is a sum over path classes; accumulating class by
  class bounds the memory by the largest class instead of the sum of all
  classes (the case that matters for chapter 11's table).
- What remains after this: compute ∝ pixels × $K$ and sampling noise, the
  band sum's intrinsic properties. Disk ∝ $N$ (tens of GB at $10^9$).

**Compute: the block is a matrix product (derived; task
`band-sum-scatter-renderer`).** By ring invariance an event's pose at
azimuth $\alpha$ is $R_i(\alpha) = Q_\alpha R_i(\alpha_0)$ (the pose is
rebuilt from the event's own $D_i$, roadmap §4.2). The five pose densities
depend on the pose only through body axes measured against the zenith
$\hat{\mathbf z}$: `ZenithGaussianPoseDensity` through the c axis,
`ZenithRollGaussianPoseDensity` through the c axis and one side axis (the
roll). Hence

$$
\rho\big(Q_\alpha R_i\big) = f\big((Q_\alpha^{\mathsf T}\hat{\mathbf z})\cdot(R_i\hat{\mathbf c}),\ \dots\big),
$$

a dot product between a pixel vector $Q_\alpha^{\mathsf T}\hat{\mathbf z}$
and an event vector $R_i\hat{\mathbf c}$. For a block of $M$ pixels and $K$
events the contributions are one $(M\times 3)(3\times K)$ product, an
elementwise $f$, the band mask, and a weighted reduction with $w$: a GEMM
followed by a GEMV; the roll families need two products. Events sorted by
$D$ and pixels sorted by $\delta$ make the mask banded, so the work tiles
into (deviation segment × intersecting pixel block). The $D_{6h}$
transport $L_g R g^{\mathsf T}$ (task `band-sum-full-symmetry`) is linear
in the body axes and folds into the same products. The random density has
$f \equiv$ const: each ring is one histogram value.

Not yet known: the speed-up; the axis-vector interface `pose_density` needs
(its `evaluate_batch` takes rotation matrices and stays the defining
oracle); the mask's agreement with `pixel_band` at the caustic, bit for
bit.

Today's canonical strip (`169 s`, about `6 GB` on four workers) has no
memory problem. The reorganisation is for chapter 11's table (many classes
× five families), many sun elevations, full-sky or finer images.

## 5. Divergent light

A nearby point source (a street lamp at $L$) breaks one assumption, and it
is not in the crystal kernel. Under parallel light every crystal on a view
ray sees the same pair (incident, outgoing), so a pixel is one pose
integral. Under a point source the incident direction changes along the
ray. The kernel and the store (section 1) are reused unchanged; the image
formation changes.

**Geometry (derived; the three closed forms checked numerically on 2000 random $(\theta, t)$, finite-difference agreement `6e-6`).** Let the observer be at $O$, $d = |OL|$, the view
ray $\mathbf x(t) = O + t\mathbf v$ at angle $\theta$ from the direction to
the lamp, and $\beta(t)$ the angle at $L$ in the triangle $O L \mathbf x$.
The deviation of the ray scattered at $\mathbf x$ toward $O$ is the
exterior angle

$$
\delta(t) = \theta + \beta(t),
$$

which grows monotonically from $\theta$ (crystals at the observer) to $\pi$
(crystals far behind, back-scattering). The law of sines gives the whole
ray in closed form:

$$
t(\delta) = \frac{d\,\sin(\delta-\theta)}{\sin\delta},\qquad
r(\delta) = |\mathbf x - L| = \frac{d\,\sin\theta}{\sin\delta},\qquad
\frac{dt}{d\delta} = \frac{d\,\sin\theta}{\sin^2\delta}.
$$

For an isotropic source of intensity $J$, single scattering and no
extinction, the pixel radiance is
$\int n(\mathbf x)\,J/r^2\,I_{\hat{\mathbf s}(t)}(\delta(t),\alpha)\,dt$,
where $I_{\hat{\mathbf s}}$ is the parallel-light value for a source in
direction $\hat{\mathbf s}$ (here the local direction toward the lamp) and
$n$ the crystal number density. Changing the variable to $\delta$, the
factor $(dt/d\delta)/r^2 = 1/(d\sin\theta)$ is **constant along the ray**:

$$
L(\theta,\alpha) = \frac{J}{d\,\sin\theta}\int_{\theta}^{\pi}
  n\big(\mathbf x(\delta)\big)\, I_{\hat{\mathbf s}(\delta)}(\delta,\alpha)\,d\delta .
$$

$d\sin\theta$ is the distance from the lamp to the view line. The scattering
plane (through $O$, $L$ and the ray) is the same for every $t$, so the
azimuth frame is fixed along the ray; only the source direction
$\hat{\mathbf s}(\delta)$ turns within that plane, and it matters only
through $\rho$.

**Band sum (derived).** With roadmap §4.2's coarea identity and $N$
equal-area events, the $\delta$ integral becomes a sum over **all** events
with $D_i \ge \theta$, each mapped to the unique point
$t_i = t(D_i)$ on the ray, with no band width and no ray marching:

$$
\hat L(\theta,\alpha) = \frac{J}{d\,\sin\theta}\cdot\frac{1}{2\pi N}
  \sum_{i:\,D_i\ge\theta} \frac{n(\mathbf x(D_i))\, w_i\,\rho(R_i)}{\sin D_i},
$$

where $R_i$ is built as in the parallel case with $\hat{\mathbf s}$
replaced by the direction from $\mathbf x(D_i)$ to the lamp. The locus of
fixed $D$ in space is Minnaert's spindle ("cigar") about the axis $OL$;
the note in the Ice Halo repository (`doc/research/inverse-rendering.md`,
"Extension: Divergent-Light Halos") contrasts ray marching with Gislén's
cigar method, and in the event form they are the same computation.
$\sin D_i \to 0$ near back-scattering corresponds to $t \to \infty$; a
cloud of finite extent ($n = 0$ beyond $t_{\mathrm{far}}$) cuts it off.

- For the random density $I$ depends on $\delta$ only, and with a uniform
  cloud $L \propto \frac{1}{\sin\theta}\int_\theta^\pi I(\delta)\,d\delta$:
  a cumulative integral of the sun halo's radial profile. With events
  sorted by $D$ that is a suffix sum, one lookup per pixel.
- Per pixel the event set is every $D_i \ge \theta$ instead of one band:
  the cost rises by roughly the ratio of the deviation range to the band
  width (the note estimates `10-100×`). The scatter organisation of
  section 4 matters more here.
- The contour route integrates all level sets $\delta \ge \theta$, so the
  coarea step runs backward and the pixel is an area integral over
  $\{D_P \ge \theta\}$ on $S^2$, with no $1/\lvert\nabla D_P\rvert$ and no
  fold singularity; edges come from the integration limit $\theta$.
- Phase I can take any (incident, outgoing) pair, but a pixel needs a
  quadrature over $t$ (a fibre per node): a pointwise reference at a few
  $(\text{pixel}, t)$, not a renderer.

**What is hard is not the algorithm.**

1. A scene model this project does not have: lamp position and intensity,
   observer, crystal cloud $n(\mathbf x)$ and extent, perhaps extinction,
   and the conventions for them (camera coordinates are currently sky
   directions relative to the sun).
2. **No oracle.** Lumice's `light_source.type` is `"sun"` only. Available
   independent checks: the far-lamp limit ($d \to \infty$ at fixed
   $\theta$ recovers parallel light), a brute-force Monte Carlo inside this
   project (it shares the kernel, so its failure modes are not
   independent), Gislén's papers (shapes, qualitatively).
3. Finite source size (the sun's `0.5°` too, not modelled today) is a
   convolution on the sky, a separate axis; it should not be mixed into
   the divergent-light work.

Trigger: Lumice supports a point source (an oracle), or the writing series
needs street-lamp halos. Recorded in `scratchpad/backlog.md`.

## 6. Where each piece lands

| piece | status | task |
|---|---|---|
| store independent of the source; `.npy` + mmap; bucketed build | design, probe measured | `s2-store-schema-3` (21) |
| $\delta$-segment workers, class-by-class accumulation, GEMM tiles | design | `band-sum-scatter-renderer` (22), after 21 |
| Phase I seeds and completeness cross-check from the store | design | `phase1-seeds-from-store` (scrum 24, sub-task 5) |
| contour quadrature, critical points, certificate, cost model of section 3 | design | scrum `phase2-contour-quadrature` (24), sub-tasks 1-4 |
| chapter 10 verdicts (22° inner edge, Liljequist 142°, parhelic circle, parallel-face degeneracy) | open (roadmap §4.1(g)) | scrum 24, sub-task 6 |
| absolute scale against Lumice after its projected-area fix | open | `lumice-area-weighting-recheck` (23), after Ice Halo #597 |
| divergent light | derived | backlog, not scheduled |
