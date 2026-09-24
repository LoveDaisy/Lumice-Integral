# Lumice Integral Roadmap

> Since 2026-09-24 this file holds the project's status, near-term queue,
> writing-project coupling and decisions log. The design moved to
> [overview.md](overview.md), [phase1.md](phase1.md) and
> [phase2.md](phase2.md); the old section numbers below are kept as pointers
> so that references in code, provenance strings and earlier artifacts
> ("roadmap section 4.2", ...) still resolve.

## 0. Status and near-term queue

| phase / milestone | status | design |
|---|---|---|
| Phase I: $\mathrm{SO}(3)$ continuation renderer | closed 2026-09-23 (strip agrees with Lumice in shape and absolute scale) | [phase1.md](phase1.md) |
| M1: $S^2$ event store + band-sum renderer in production; $D_{6h}$ transport; conventions and symmetry authority | done 2026-09-23 / 2026-09-24 | [phase2.md](phase2.md) §1-§3, §5 |
| M2: contour quadrature, critical points, completeness certificate, cross-validation, chapter-10 verdicts | bootstrapped (scrum 24) | [phase2.md](phase2.md) §3.1, §4, §10 |

Queue (tasks in `scratchpad/tasks.md`; dispatch order 20 ∥ 21, then 22 ∥ 24, 23 when unblocked):

| # | task | depends on |
|---|---|---|
| 20 | chore `band-sum-small-fixes`: `pixels.csv` value repr, a docstring escape, two missing regression tests | — |
| 21 | `s2-store-schema-3`: store independent of the source, `.npy` + mmap, bucketed build ([phase2.md](phase2.md) §1.1, §8) — **done 2026-09-24** (schema 3) | — |
| 22 | `band-sum-scatter-renderer`: band sum organised by deviation, class accumulation, GEMM tiles ([phase2.md](phase2.md) §8) | 21 |
| 23 | `lumice-area-weighting-recheck`: absolute scale after Lumice's projected-area fix | Ice Halo #597 merged |
| 24 | scrum `phase2-contour-quadrature` (M2): `dp-field-topology` → `dp-field-layer` → `s2-contour-extraction` → `s2-contour-quadrature` → `phase1-seeds-from-store` → `ch10-numerical-verdicts` | 21 |

Deferred: the chapter-11 table (path classes × pose families) after 22 and
M2; divergent light ([phase2.md](phase2.md) §9, backlog); finite solar disk;
multiple wavelengths (one store per wavelength); GPU kernels. Dropped:
pixel-space adaptive sampling (its premise, an expensive pixel, is gone with
the band sum).

## 1. Project Positioning

Moved to [overview.md](overview.md) (introduction, §1).

## 2. Mathematical Model

Moved: the $\mathrm{SO}(3)$ coarea form to [phase1.md](phase1.md) §1, the
common integral to [overview.md](overview.md) §1.

## 3. Phase I: SO(3) Continuation

Moved to [phase1.md](phase1.md). The Phase I contract stays
[phase1-math-contract.md](phase1-math-contract.md).

### 3.1 Core algorithm

[phase1.md](phase1.md) §2.

### 3.2 Reconstruction milestone

[phase1.md](phase1.md) §2 (summary) and appendix "3.2" (the milestone list
with its evidence, verbatim).

### 3.3 From one fiber to a renderer

[phase1.md](phase1.md) §3-§4 and appendix "3.3" (status record, verbatim).

### 3.4 Orientation distributions

[phase1.md](phase1.md) §5 (last paragraph).

### 3.5 Near-term plan (2026-09-18, after scrum `strip-pipeline-v2`)

All items done or moved; [phase1.md](phase1.md) §4 (key turns) and
appendix "3.5" (verbatim record, items 1-5 and the Phase I closeout).

### 3.6 Writing-project coupling

- W1 `geometry-depend-on-lumice-integral` (writing repo task 16): the notes
  depend on this package (path dependency, Python `3.12`), delete their
  `halo_notes.geometry` copy, and pass `pbd_orbit` as the `symmetry_orbit`
  callback. Prerequisite here: `lumice_integral/__init__.py` must not import
  JAX eagerly, so that `import lumice_integral.geometry` stays numpy-only.
- W2: the notes consume figure-data v3 for the chapter-6 state-space figure;
  the chapter-6 strip remake waits for defect 2 and the rerender.
- W3 (writing repo task 33): the notes import `lumice_integral.symmetry` in
  place of `halo_notes.math.{group,reflection_group,signature,ground_truth,attitude}`
  (`sun_vector` becomes `camera.sun_direction`) and reduce their `check_*.py`
  scripts to thin wrappers. Nothing here waits for it.

## 4. Phase II: Fiber Reduction

Moved to [phase2.md](phase2.md).

### 4.1 Design findings (2026-09-20 discussion, before the Phase II scrum)

(a) weights on $S^2$ → [phase2.md](phase2.md) §1; (b) the level-set
integral and boundaries → §2; (c) topology and completeness → §3.1; (d)
layered invariance → §3.2 (symmetry: §3.3); (e) fixtures → §4; (f) design
constraints → §4; (g) open points → §10.

### 4.2 Band-sum quadrature (precomputed $S^2$ events)

[phase2.md](phase2.md) §5 (correspondence with the Ice Halo note,
estimator, pose rebuild, corrections, pixel model, division of labour), §3.3
(precomputation view, mirrors, $K_{\mathrm{eff}}$), §9 (divergent light);
the dated probe and production records are the appendix of phase2.md,
verbatim.

## 5. Proposed Responsibility Boundaries

Moved to [overview.md](overview.md) §4.

## 6. Relationship to Other Projects

Moved to [overview.md](overview.md) §5.

## 7. Validation Strategy

Moved to [overview.md](overview.md) §6.

## 8. Explicit Non-goals for the First Phase

Moved to [overview.md](overview.md) §3.

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
  numbering; corrected 2026-09-24: Lumice numbers the faces the same way,
  so it is Lumice's roll too, `docs/conventions.md`). Provenance grows a trailing `family` key; the column block is
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
- **2026-09-23**: band sum on the narrow families (section 4.2, task
  `narrow-density-band-sum-probe`): class `[3,5]`, one store per PBD
  member, Phase I class pixels as reference (band averages on steep
  pixels). Verdict: rendering backend for plate, Parry and Lowitz at
  $N = 10^8$ (lit RMS `8e-4` / `4e-3` / `4e-3`); $K_{\mathrm{eff}}/N$
  `1.2e-5`-`1.2e-4`, never collapsed, so the rho-aware store is not
  needed. The Fibonacci lattice usually beats $1/\sqrt{K_{\mathrm{eff}}}$
  by 7-9× but aliases (Parry at $10^7$), so stores are sized by
  $K_{\mathrm{eff}} \ge 10^4$ (all families $< 10^9$). With task
  `band-sum-quadrature-probe`, Phase II's S² field layer and band-sum
  renderer can target all five families of chapter 11 with one uniform
  store per member (measured at sun 15°, class `[3,5]`, one profile per
  family; other classes and elevations are unmeasured).
- **2026-09-23**: band-sum symmetry by the precomputation view (section
  4.2, task `band-sum-full-symmetry`): one store on $S^2$ is every member's
  store on a fundamental domain, so symmetry saves evaluation and makes no
  samples. All of $D_{6h}$ transports on $S^2$ (mirrors included, pose
  $L_g R g^{\mathsf T}$), one store per class; $K$ / $K_{\mathrm{eff}}$
  count distinct events (`k_eff_semantics = "per_event"`, values unchanged
  bit for bit), which the i.i.d. two-seed test confirms as the noise
  predictor. Earlier renders' $K_{\mathrm{eff}}$ (no `k_eff_semantics`
  field) are `per_transport_sample` and not comparable. If more such
  semantics tags accumulate in the provenance, fold them into one
  provenance schema version instead of one tag per field.
- **2026-09-24**: conventions follow fixed authorities (task
  `notation-alignment`, owner ruling of 2026-09-23): coordinates, face
  numbering, pose chain, azimuth, light source, camera and pixels follow the
  Lumice documentation (`doc/coordinate-convention.md` first); mathematical
  notation Lumice does not cover follows the writing series'
  `docs/framework.md`; the rest is decided in `docs/conventions.md`, the
  single table of every convention with its check. Disagreements are settled
  by these rules, not item by item. Consequences: the public sun direction is
  $\hat{\mathbf s}$, toward the sun (`camera.sun_direction`); the solver's
  propagation direction keeps the contract's `s = -ŝ` and is converted only
  by `camera.incident_direction_from_sun`; Phase II's $\mathbf u$ is
  $R^{-1}\hat{\mathbf s}$ (framework theorem 8) and the event store moves to
  schema 2, whose antipodal default lattice keeps every pose, event and pixel
  value of schema 1 bit for bit while schema 1 caches are refused. Lumice,
  Lumice Integral and the writing series' simulation layer share one pose
  chain and face numbering; the writing series' `column_attitude` is the
  chapter parametrisation with $\theta = $ roll $- 180°$. The fiber twist
  keeps its symbol $\psi$ (the writing series' $\theta$ would collide with
  the zenith), and `D6h` elements are identified across projects by matrix,
  not by index, until task `symmetry-authority` migrates the published
  table.
- **2026-09-24**: the symmetry and combinatorial-classification authority
  moves from the writing project's `halo_notes.math` into
  `lumice_integral.symmetry` (task `symmetry-authority`; `group`,
  `reflection_group`, `signature`, `ground_truth` with its data file, and
  `attitude` without `sun_vector`, which is `camera.sun_direction`),
  migrated near-verbatim with public names kept 1:1 and the writing
  project's tests alongside. The repository now holds one `D6h` table:
  `path_class.hexprism_symmetry_matrices()` returns `signature.D6H` instead
  of its own generator closure (same 24 matrices; the order and ten elements'
  last bits change, no key or cached artifact records either). The only
  change of arithmetic is the wedge angle, which shares `geometry`'s
  `atan2` kernel instead of the writing project's `arccos` (no rounded class
  wedge moves). Dependency direction `symmetry -> geometry` is checked
  statically; the chapter-8 / chapter-9 tables are reproduced byte for byte.
- **2026-09-24**: the $S^2$ event store is independent of the light source:
  it depends on crystal × path ($\Phi$ group) × refractive index × sampling
  (owner probe, three sun directions, $\mathbf u$ bit for bit, the rest
  within `1.6e-11`; [phase2.md](phase2.md) §1.1). Schema 3 drops
  `sun_direction` from the spec and the cache key (task 21). The Phase I
  prescan table is the same presampling before the twist is quotiented out;
  Phase I's seeds move to the store with a statistical completeness
  cross-check (M2 sub-task 5). The band sum is to be organised by deviation
  (scatter instead of gather; event × pixel blocks as matrix products, task
  22). Divergent light reuses the store unchanged; its event form needs no
  ray marching ([phase2.md](phase2.md) §9), deferred for lack of an oracle.
  M2 = scrum `phase2-contour-quadrature`; writing order chapter 10 before
  chapter 11.
- **2026-09-24**: documentation restructured (author's request): an
  overview, one design document per phase (English and Chinese; measured
  records English only), and this file reduced to status, queue, coupling
  and decisions. Content moved verbatim where it is a record; old section
  numbers kept as pointers.
- **2026-09-24**: the $S^2$ event store moves to schema 3 (task 21
  `s2-store-schema-3`): no sun direction in `S2StoreSpec` or the cache key
  (the build aligns to one fixed reference direction, numerically the
  canonical sun, so the canonical build is the schema 2 one bit for bit);
  one `.npy` per array with per-array SHA-256 and size, read-only
  `mmap_mode="r"` loading that checks sizes only plus an explicit
  `S2EventStore.verify` (the default load still hashes; mapping and a full
  hash are incompatible, so the hash moved out of the mapped path rather
  than being dropped); the build buckets events by $D$ on disk and writes
  straight into the cache, halving the $N = 10^8$ build peak. Schema 1 and 2
  caches are refused, not converted. Rendering keeps its gather
  organisation and the workers' hashed full load until task 22.
  Measurements: [phase2.md](phase2.md) appendix.
