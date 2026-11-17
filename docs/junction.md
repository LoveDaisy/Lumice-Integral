# S6 junctions: opportunistic structural intersections

[中文](junction_zh.md) · [Phase II](phase2.md#31b-s6-junctions) · [Parity fixtures](analytic-parity-fixtures.md#65-s6-junctions-wave-3-second-batch)

## Object and scope

For one crystal, literal face sequence P and refractive index n, a junction
is an intersection of **existing structural strata in the body-frame sphere**
S², clipped to the closure of U_P. It is not a brightness label, a visibility
certificate, a new critical-set discovery algorithm, or an SO(3) target-fiber
intersection (the last is the separate C16 `family_target` problem).

`lumice_integral.junction.enumerate_junctions` consumes:

- **kind 1:** the existing `DPField.interior_critical_points`, plus the known
  slab axis points and analytic crease `axis·u=0`. An isolated point is tested
  for membership, never replaced by a supposed regular equation `|grad D|²=0`.
  A general nonisolated critical manifold is not discovered here.
- **kind 2:** boundary pieces and their coincident validity gates. Each source
  keeps the original margin name and upstream piece identities. All original
  validity gates remain authoritative, including gates removed as identities
  from boundary walking's active list.
- **kind 3:** each internal reflection's TIR-onset curve, retaining reflection
  step, margin identity, arc identities and upstream failed-seed count. This is
  a Fresnel weight kink, **not** an additional validity gate. Equivalent
  reflection margins retain their separate identities.
- **exact family image:** the four `ExactFamily` supports from
  [the restricted-family contract](family-restricted.md). Intersections with
  a one-dimensional image are points or coincidences. A curve intersecting a
  two-dimensional image generally remains a curve; its samples are not a
  finite junction set. All actual pose lifts above each u are retained.

The top-level result always has `exhaustive=false`,
`coverage="opportunistic_existing_strata_only"`, and explicit omissions for
corridor A=0, unrepresented critical manifolds and undiscovered upstream
components. A closed kink loop or `KinkCurve.complete` does not strengthen
that claim. The shared `family_restricted.circle_intersections`, `margin_plane`
and `unit_sun` are public single-authority primitives for these adapters;
there is no second circle solver. There is no new A>0 condition: geometric existence and the finite
crystal's radiometric weight remain different questions.

## Inputs and identities

`enumerate_junctions(field, pairs=None, family=None, sun=None, ...)` takes an
already constructed `DPField`. `pairs` selects pairs of source IDs; omitted
selection means every existing structural pair and, if requested, every
source/family pair. Reversed or repeated selected pairs are materialized once.
An unknown ID is a caller error, not a normal empty result.

Source IDs are `kind_2:<original-margin>`, `kind_3:<original-margin>`,
`kind_1:interior:<index>`, `kind_1:axis:<index>`, and `kind_1:crease`.
A family selection uses `family:<name>`. Critical-point indices identify an
upstream object **within that wavelength**; they do not pair objects across λ.
Source descriptors include `kind`, stratum `dimension`, original `margin`,
`origin` list, `equivalent_to`, `failed_seeds`, and `model`.
Point records also identify the matching upstream origins; distinct events at
one geometric location are not erased by position deduplication.

`sun` is a finite unit vector **toward** the sun, and rotations map body to
world: `R u = sun`. Family parameter order, sheets, seams and endpoints are
exactly those of `ExactFamily`, not an independently implemented inverse.
All angles and field values are radians; margins and unit-vector components
are dimensionless. Crystal and face numbering follow `conventions.md`.

Default search options:

| option | default | scope |
|---|---:|---|
| `budget` | 64 | maximum local candidates attempted **per source pair** |
| `nearby` | 0.02 | chord-distance seed-neighbourhood, not root acceptance |
| `max_iterations` | 30 | local Newton iterations per candidate |
| `max_step` | 0.001 | maximum tangent-plane Newton step, radians to first order |
| `family_samples` | 32 | existing family clipping/root sampling |
| `family_budget` | 4096 | existing family's numerical root budget |

These bounds do not cover the cost of constructing upstream structures.
`DPField.lattice_n` records that separate input. A budget of zero deliberately
produces unresolved candidates, not a successful empty intersection.

## Evidence required for a point

Candidate generation and local refinement are different operations. Bounded
nearest-curve candidates, sign changes, near-zero local extrema and endpoints
supply seeds. A polyline proximity is never a root certificate. A numerical
search with no seed reports `not_found`, not global nonintersection.

The low-layer `spherical_newton.refine_pair` is the **single** mechanical
Newton implementation shared with `Walker.refine_corner`. The walker adapter
retains the original callback order, floating-point operations, 30-iteration
limit, 0.001 step rejection, 1e-14 determinant guard, 1e-16 post-step stop, and
strictly best-residual point. It returns the same legacy point even after an
unsuccessful refinement. S6 instead requires a successful local termination
(at dimensionless residual 1e-11), then separately checks:

1. Finite unit u (`|u·u-1| <= 1e-12`).
2. **Original** constraints, not just an equivalent plane model, each with
   residual <=1e-11. Every original validity gate must be >=-1e-13, the
   existing walker's closure threshold. A kink thereby also lies on the
   reflecting incidence branch.
3. Local source-arc membership and seed displacement within 0.012 radians
   (chord neighbourhood for sampled arc membership). This is slightly over
   two upstream 0.25° steps, a locality bound, not a geometric root tolerance.
4. For numerical transverse roots, finite tangent-space **original margin**
   Jacobian, with smallest/largest singular-value ratio >1e-7. Rank evidence
   uses the original functions even when a plane model supplies refinement.

A failed, nonfinite, singular, over-budget, or large-step solve cannot pass
merely because its best-so-far point looks plausible. Unsupported numerical
contacts remain unresolved. Tangencies require the existing analytic
plane-circle intersection proof (or the existing family jet/two-sided
checks), never a small determinant alone. Analytic positive-radicand twin
roots are not proximity-merged. Duplicate numerical seeds are merged only
at 1e-10 angular separation after acceptance; near rank loss remains unresolved.

Field `value` is the existing walker's D_P closure value. At exit Snell,
rounding-negative discriminants use the established dropped-root limit;
there is no epsilon inward displacement and no fabricated singular gradient.
`differential.available=false` records `analytic_evidence_no_differential`
for analytic circle evidence, or `upstream_stratum_membership_no_differential`
for point-stratum membership. These mean no numerical differential was
computed, not that the geometry is nonsmooth or its derivative is zero.

## Continuous objects and diagnostic outcomes

Continuous coincidence requires an identity: identical upstream margins,
identical analytic circle planes, or equality of squared unfolded incidence
normals. For example, on `3-5-6-7`, internal-1 TIR and exit-Snell have opposite
sign discriminants and the **same zero set**. Their validated upstream runs
are continuous objects with `dimension=1`, not many independent roots.
Clipping still has upstream sampling coverage. Cyclic accepted runs are joined
across the sampling seam; `closed=true` distinguishes a retained whole loop
from an open clipped arc. A surviving singleton cannot certify a continuous
arc or tangency: it produces `dropped_short_run` with its position rather
than disappearing. `3-5-6-7-3` internal-1 and
internal-3 retain both event identities on their coincident arcs.

Each selected pair has `sources`, `status`, `coverage`, `records`, `curves`
and `diagnostics`; numerical pairs additionally report candidate, attempted,
converged and rejected counts. Successful points carry u, source identities,
raw residuals, minimum validity margin, D_P, intersection classification,
evidence type and differential availability. Continuous records carry their
upstream arc / family sheet, ordered samples, dimension and evidence.

| status | meaning |
|---|---|
| `computed` | locally accepted isolated point(s), not an exhaustive set |
| `continuous` | one-dimensional intersection objects, not junction samples |
| `disjoint` | analytic circle models have no sphere root; scope stated in coverage |
| `not_found` | no accepted intersection in the existing selected strata/search |
| `unresolved` | candidate failure or exhausted budget; no accepted point |
| `partial` | accepted objects **and** explicit unresolved/upstream failures |
| `unsupported` | named exact-family or critical-stratum regime has no adapter |
| `unsupported_continuous` | known critical-circle coincidence lacks a supported clipping adapter |

The last two are not computed empties. Known `RestrictedEscape` regimes are
reported with their name and detail. Other exceptions propagate; programming
errors are not broadly caught and reclassified as normal refusals. Partial
success never discards upstream failed seeds or omitted candidates. An
analytic `disjoint` proof is independent of upstream arc-sampling failures:
those diagnostics stay visible but do not downgrade a proved model-level
nonintersection to `unresolved`. Point membership performs no Newton attempt;
its candidate/attempted/converged counters remain zero even when a point is
accepted.

## Physical coverage and parity

The bounded M1 survey used 11 existing paths at complete n(400/550/700),
33 cases, lattice 20000, at most eight independent scipy roots per pair and
100 function evaluations per root, under a 900-second total search budget.
It completed in about 21 seconds; no second search was required. Coverage is
an existence witness, not a global absence claim for other types/paths.

| fixture cell | path | witness |
|---|---|---|
| `mc_s6_corner` | 3-5 | old gate×gate corner positions |
| `mc_s6_gate_kink` | 3-1-6 | entry gate×basal TIR onset |
| `mc_s6_kink_pair` | 4-8-1-7-5 | distinct internal reflection 1×2, regular interior root |
| `mc_s6_continuous` | 3-5-6-7 | exit-Snell/internal-1 coincidence |
| `mc_s6_plate_disjoint` | 3-1-6 | analytically disjoint family/kink circles |
| `mc_s6_column` | 3-1-6 | continuous family image intersection, both pose sheets |
| `mc_s6_iteration_failure` | 4-8-1-7-5 | controlled zero-iteration failure, no fake root |

At n(550), the independent three-equation root for the internal 1×2 witness
is approximately `[-0.2824757039268523, 0.44883472231481775,
-0.8477941193094051]`: maximum raw residual 2.1e-16, minimum gate about
0.02699, and smallest original tangent-Jacobian singular value about 0.2766.
No fixture relies on A/T positivity on singular sets.

The additive `junction` parity kind preserves schema v1 and all old recipes.
Each of the seven files contains independent complete-index 400/550/700 nm
segments. Detailed envelope, validation and comparison rules are in the
[fixture contract](analytic-parity-fixtures.md#65-s6-junctions-wave-3-second-batch).
Recompute agreement alone is not sufficient: the verifier also checks
original optical margins, all gates, sphere membership and actual SO(3)
family lifts independently of candidate generation and refinement.
