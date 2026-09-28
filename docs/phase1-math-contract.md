# Phase I Mathematical and Numerical Contract

## 1. Status and purpose

This document is the normative Phase I contract for tracing and integrating a
single regular inverse-image component of a fixed ray path in `SO(3)`. It fixes
the mathematical meanings shared by the Python/JAX reference solver and any
future replacement backend. Language bindings, storage formats, algorithms,
and default tolerances are conforming choices only when they preserve these
meanings.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe requirements
on a conforming implementation. Statements explicitly labeled *evidence*,
*current strategy*, or *open* are non-normative.

The contract covers:

- a fixed-path direction map from crystal pose to outgoing direction;
- one regular, connected fiber component reached from one supplied seed;
- local differentiation, continuation, closure, and diagnostic semantics;
- the coarea measure and independently observable physical weight factors;
- backend-independent problem, options, result, event, and termination data;
- seed search and component discovery for one target direction (section 9.5),
  under a weaker contract: procedural, not a completeness certificate.

It does not prove that every component was found, cross a
singular topology change, define a full renderer, calibrate absolute radiometry,
or reconstruct historical ch06 data. Phase II's reduction to contours on `S2`
is an independent cross-check and possible acceleration, not a replacement for
this Phase I `SO(3)` contract. Lumice is not a dependency of the contract or
solver; it may appear only as an external validation oracle.

## 2. Normative objects and terminology

| Symbol or term | Meaning |
|---|---|
| `B`, `W` | Right-handed orthonormal body and world frames. |
| `R in SO(3)` | Active crystal pose mapping body components to world components: `v_W = R v_B`. |
| `P` | An ordered, fixed ray-path branch, including its face and interaction choices. |
| `s in S2` | World-space unit propagation direction from the light source toward the crystal. |
| (boundary note) | `s` is the *opposite* of the writing series' `s` and of Lumice's sun position vector, both of which point toward the sun; the project calls that one `ŝ = -s` (`camera.sun_direction`, Phase II's `u = R^-1 ŝ`). This contract's `s` is unchanged; the one conversion is `camera.incident_direction_from_sun`, and `docs/conventions.md` is the table of all conventions. |
| `d in S2` | World-space unit propagation direction from the crystal toward the observer. |
| `F_P(R)` | Outgoing world-space unit direction produced by path `P` at pose `R`. |
| `V_P` | Open subset on which the selected path branch is feasible and smooth. |
| `X_(P,d)` | Fiber `{R in V_P : F_P(R) = d}`. |
| `C` | The one connected regular component of `X_(P,d)` reached from the supplied seed. |
| `[delta]_x` | Skew matrix satisfying `[delta]_x v = delta cross v`. |
| `dH^1_g` | One-dimensional Hausdorff/arclength measure induced by the declared `SO(3)` metric. |

`S2` always means the unit sphere in the declared world frame. Angles and Lie
algebra coordinates are in radians. Directions, rotations, Fresnel factors,
visibility factors, and geometric cosines are dimensionless. Lengths and
wavelengths, when introduced by an optical evaluator, MUST declare their unit;
this contract does not choose one. Radiometric quantities MUST declare their
units at the source/pixel boundary rather than inheriting an implicit scale
from a fixture.

An image pipeline that stores a camera-to-scene direction MUST negate it at an
explicit adapter boundary before supplying `d`. A path implementation that uses
surface-normal or incident-direction signs different from this section MUST do
the same at its evaluator boundary. It MUST NOT silently redefine `s`, `d`, or
the meaning of `R`.

## 3. Pose composition and local coordinates

Local derivatives and updates use right-trivialized body coordinates:

```text
R(delta) = R exp([delta]_x),    delta in R^3.
```

Consequently, applying an increment composes it on the right; changing to a
left-trivialized implementation requires the appropriate adjoint conversion at
the interface. The contract does not require rotation matrices as storage.
Quaternion storage is conforming only if `q` and `-q` represent the same pose,
and pose proximity is measured on `SO(3)` rather than by a raw quaternion
difference.

The Phase I reference path MUST evaluate continuation and final quadrature in
float64. Float32 is permitted only for an explicitly approximate and separately
validated candidate or prefilter path; it MUST NOT enter the reference path by
implicit dtype conversion. This is a numerical reference requirement, not a
mathematical property of the fiber.

Exponential-map primitives used under AD MUST have a derivative-safe zero
limit. In particular, a finite function value at `delta = 0` is insufficient if
the differentiated expression passes through the derivative of `norm(delta)`.
Near-identity pose distance MUST use a stable group-angle construction such as
`atan2(sin(theta), cos(theta))`, or an equivalent method with demonstrated
accuracy; raw `acos((trace(R)-1)/2)` is not adequate as the sole closure metric.

## 4. Authority, evidence, and ownership

This document is the single source for Phase I mathematical terms, interface
semantics, and conformance invariants. The following sources are evidence or
context, not competing specifications:

| Source | Role | Normative status |
|---|---|---|
| [`phase1.md`](phase1.md), [`overview.md`](overview.md), [`roadmap.md`](roadmap.md) | Phase I design and history; project overview, responsibility boundaries and validation strategy; status and decisions. | Context; links here for detailed Phase I semantics. |
| [`0001-phase-i-python-jax.md`](decisions/0001-phase-i-python-jax.md) | Accepted reference stack, dtype, and execution boundary. | Normative for architecture; numerical observations are evidence. |
| `scratchpad/explore-ad-continuation-stack/SUMMARY.md` | AD, closure-distance, precision, and batching experiments. | Evidence only. |
| `src/lumice_integral/{analytic,so3,continuation,optics}.py` | Current analytic and synthetic probes. | Evidence/current strategy only. |
| `tests/test_analytic_fiber.py`, `tests/test_optics.py` | Existing positive regression fixtures. | Evidence only; exact fixed-step counts are not contract terms. |
| `tests/test_reference_core_conformance.py` | Consumer-facing schema, invariant, convergence, and termination checks mapped to C01–C14. | Durable conformance evidence for the currently supported reference core. |
| `tests/test_discovery.py` | Discovery fixtures mapped to C15–C21. | Durable conformance evidence for section 9.5. |

Downstream ownership is explicit:

| Item | Owner | Contract status before owner completes |
|---|---|---|
| Structured adaptive single-component solver and event gate | `reference-continuation-core` | Semantics fixed here; implementation pending. |
| Default tolerances and their convergence evidence | `reference-core-conformance` | Open numerical values. |
| Basis-invariance, failure, event, and step-size perturbation tests | `reference-core-conformance` | Required matrix rows; evidence pending. |
| Seed/component discovery for one fixed path and one target direction | `lumice_integral.discovery` (`strip-component-discovery`, `phase1-seeds-from-store`; contract `discovery-contract`) | Semantics fixed in section 9.5 (`reference-discovery-v1`); strip-level neighbour continuation and full-image completeness are the strip driver's (`strip-image-driver`). |
| Singular topology changes and branch continuation | Future exploration | Open; no support claim. |
| Historical ch06 projection, normalization, and provenance | `ch06-reference-fixture` | Outside this contract. |

## 5. Metric, map, residual, and differential contract

### 5.1 Measures and metric normalization

Identify a right-trivialized tangent `R [omega]_x` with `omega in R^3` and use
the standard bi-invariant metric

```text
g_R(R [omega]_x, R [eta]_x)
    = (1/2) trace([omega]_x^T [eta]_x)
    = omega dot eta.
```

Thus a one-parameter rotation `R exp(t [omega]_x)` has speed `|omega|`, and
geodesic distance is the principal rotation angle in `[0, pi]`. With this
normalization,

```text
Vol_g(SO(3)) = 8 pi^2,
d mu_Haar = dVol_g / (8 pi^2),
```

where `mu_Haar` is Haar probability measure. `S2` uses the metric induced by
the Euclidean inner product, surface-area measure `dA`, and its usual arclength
measure. A result MUST identify whether its pose density is relative to
`dVol_g` or `d mu_Haar`; the factor `1 / (8 pi^2)` MUST remain explicit when
converting from Haar probability density to `dVol_g`. It MUST NOT be hidden in
a Fresnel, projection, or radiometric factor.

### 5.2 Fixed-path smooth domains

For fixed path data `P`, wavelength, material parameters, and incident
direction `s`, the evaluator defines

```text
F_P : V_P subset SO(3) -> S2.
```

`V_P` is partitioned into open regions on which the ordered faces,
reflection/refraction choices, visibility state used by the branch, and all
other discrete path choices are fixed and `F_P` is smooth. `F_P` MUST return a
unit direction within a declared numerical tolerance. Domain predicates and
branch diagnostics MUST be evaluated before unsafe expressions such as a
square root across a TIR boundary.

For target `d`, a regular fiber point satisfies both `F_P(R) = d` and
`rank(D F_P|_R) = 2`. Phase I continuation applies only within one such smooth
regular region. Approaching the boundary of that region is observable event
data, not an ordinary residual NaN.

### 5.3 Target-local residual

Choose `E_d in R^(3 x 2)` with orthonormal columns spanning `T_d S2`:

```text
E_d^T E_d = I_2,    E_d^T d = 0.
```

On a declared target neighborhood `U_d` that contains `d` but excludes its
antipode `-d`, define

```text
r_d(R) = E_d^T (F_P(R) - d) in R^2,
R in V_P and F_P(R) in U_d.
```

The neighborhood restriction is required because this projected residual also
vanishes at `F_P(R) = -d`. An evaluator MUST NOT accept such an antipodal point
as a root. A sufficient policy is a declared positive lower bound on
`F_P(R) dot d`; the bound itself is a numerical option, not a mathematical
constant.

Replacing `E_d` by `E_d Q`, `Q in O(2)`, left-multiplies the residual and its
Jacobian by `Q^T`. It therefore preserves the local zero set, rank, tangent
kernel, singular values, and normal Jacobian. A general sphere chart MAY be
used, but its coordinate metric MUST be included. If `A_chart` is the chart
Jacobian and `G` is the target metric matrix in those coordinates, then

```text
J_perp F_P = sqrt(det(A_chart A_chart^T) det(G))
```

at a root, when the domain coordinates are orthonormal under `g`. Treating an
arbitrary chart as Euclidean without this correction is non-conforming.

### 5.4 Local Jacobian, normal Jacobian, and tangent

At `R`, define the right-trivialized residual Jacobian

```text
A(R) = D_delta r_d(R exp([delta]_x)) at delta = 0,
A(R) in R^(2 x 3).
```

At a root, an orthonormal target basis makes `A` a matrix representation of the
normal differential of `F_P`. A regular point has `rank(A) = 2`, two positive
singular values `sigma_1 >= sigma_2 > 0`, and

```text
J_perp F_P(R) = sqrt(det(A A^T)) = sigma_1 sigma_2 > 0.
```

The unit fiber tangent `t(R) in R^3` satisfies

```text
A(R) t(R) = 0,    |t(R)| = 1.
```

Its sign is not intrinsic. A trace MUST choose a seed orientation explicitly
or record a deterministic initial convention, then orient subsequent tangents
so they are continuous with the accepted previous tangent. Reversing this
initial choice reverses sample order but MUST NOT change arclength or a scalar
line integral. Singular-value thresholds and condition-number limits used to
declare numerical rank are options supported by evidence; they are not the
exact rank definition.

## 6. Single-component continuation contract

### 6.1 Preconditions and invariant

The input seed MUST be in `V_P`, satisfy the target-local neighborhood gate,
and meet the configured residual and regularity checks. Otherwise no regular
trace starts, and the result reports the responsible event or failure reason.
Every accepted pose MUST remain on the same selected smooth branch and satisfy
the configured root, finiteness, unit-direction, and rank gates.

The solver traces only the component reachable from this seed. Neither a
closed result nor multiple successful seeds proves that every component of
`X_(P,d)` was found. Component discovery and deduplication are reported
separately by the discovery layer (section 9.5).

### 6.2 Predictor and corrector

Given accepted `(R_k, t_k)` and positive step `h_k`, a conforming predictor has
the group semantics

```text
R_pred = R_k exp([h_k t_k]_x).
```

The corrector then solves two target constraints plus one local phase
condition. For example, with a correction `delta` based at `R_pred`, it may
solve

```text
r_d(R_pred exp([delta]_x)) = 0,
t_k dot delta = 0.
```

A pseudo-arclength or equivalent well-posed formulation is conforming if it
selects the nearby root on the predicted branch and exposes the same
diagnostics. A corrected step is accepted only when all configured conditions
hold:

- the corrector converged within its iteration budget;
- the final residual norm meets its absolute/relative tolerance contract;
- all values, derivatives, updates, and diagnostics are finite;
- the bordered or least-squares system meets declared rank/conditioning gates;
- the correction magnitude and accepted advance meet declared trust limits;
- the path remains valid and no unresolved event boundary was crossed;
- the new tangent can be continuously oriented and passes the tangent-change
  gate.

Failure to accept a trial step is not by itself terminal. The solver MAY reduce
the step and retry up to declared retry and minimum-step limits. It MUST retain
the failed-attempt reason and diagnostics.

### 6.3 Adaptive step control

Step reduction MUST be possible in response to corrector non-convergence,
large correction, residual degradation, loss of rank or conditioning, excessive
tangent rotation, an event bracket, or non-finite evaluation. Step growth MAY
occur only after accepted steps with adequate corrector margin, residual,
conditioning, tangent continuity, and event clearance. Event clearance is a
statement about the *approach* to a domain boundary, not about the instantaneous
value of a margin: a margin below the slowdown threshold that is stable or
receding over the last accepted edge MUST NOT by itself prevent growth, while a
margin that is shrinking MUST bound the next step by a fraction of the
arclength at which it would reach zero at the observed rate, so that the
terminating pose of a genuine event lands close to the boundary. Options MUST
bound step size above and below and bound retries, corrector iterations,
accepted steps, and/or arclength.

No particular controller or threshold is a mathematical invariant. The result
MUST expose enough accepted and rejected step diagnostics for conformance tests
to explain why the controller changed `h`.

### 6.4 Closure and self-intersection

A trace is `closed` only when all of the following hold:

1. a minimum accepted-step count and a minimum accumulated arclength exclude
   an immediate return to the seed; both MUST be small relative to the
   problem's own step scale (a few accepted steps, a few initial steps of
   arclength) and MUST NOT encode an absolute loop length, because a loop
   shorter than an absolute bound is otherwise traversed repeatedly until the
   bound is met and its length and integral are multiplied accordingly;
2. stable `SO(3)` distance to the seed is within the closure tolerance,
   measured on the accepted edge that crosses the section (item 3) at the
   section's zero on that edge, not at the edge's end: a step longer than the
   closure tolerance otherwise carries the end outside it although the edge
   passes through the seed. The reference bisects the edge's geodesic
   `R_{k-1} exp(t log(R_{k-1}^T R_k))` to `float64` resolution in `t` and
   starts the final closure corrector (item 5) from that pose. This assumes at
   most one section sign change per accepted edge (section 12);
3. the trace crosses a declared local transverse section through the seed,
   with crossing direction recorded;
4. the terminal tangent agrees with the initial oriented tangent within the
   tangent tolerance;
5. an optional final closure corrector satisfies the ordinary root,
   finiteness, branch, and regularity acceptance gates.

The transverse section MUST be local and nonsingular at the seed. A nearby pose
without a section crossing is not closure. An intersection visible only in a
plot or non-injective coordinate projection is not an `SO(3)` self-intersection.
A return with incompatible tangent, a repeated remote segment, or an encounter
between branches is reported as a self-intersection/topology diagnostic and
MUST NOT be silently promoted to normal closure.

For quaternion storage, closure distance MUST minimize over the `q`/`-q`
equivalence or otherwise compute the same group angle.

### 6.5 Arclength and samples

Each accepted edge carries a nonnegative `SO(3)` arclength increment computed
under `g`; the accumulated length is the sum of those increments, including a
separately identified final closure correction when it contributes to the
quadrature path. `h_k` MAY approximate an edge length but MUST NOT be reported
as exact arclength when correction or curvature makes that untrue beyond the
declared accuracy. Samples MUST remain ordered in the chosen trace orientation.

## 7. Coarea and weight-factor contract

For a nonnegative or integrable function `phi` on a regular smooth branch, the
coarea identity under the measures in section 5.1 is

```text
integral_(V_P) phi(R) J_perp F_P(R) dVol_g(R)
  = integral_(S2) [integral_(F_P^-1(d)) phi(R) dH^1_g(R)] dA(d).
```

If `rho_H` is pose density with respect to Haar probability and `W_P` is the
product of all other declared factors, the contribution density with respect
to sphere area from one regular component `C` is

```text
I_(P,C)(d)
  = (1 / (8 pi^2)) integral_C
      rho_H(R) W_P(R) / J_perp F_P(R) dH^1_g(R).
```

If pose density is instead declared with respect to `dVol_g`, omit the
`1 / (8 pi^2)` factor. A complete path contribution sums this expression over
all regular components; a single-component solver MUST label its output
partial until an external discovery layer establishes completeness. The
formula is not applicable at `J_perp = 0` without a separately justified
singular treatment.

A regular component `C` need not be a closed loop. A single fixed-path
solution set is often an *open arc*: a piece of the fiber cut at both ends by
a named event of section 8 (the smooth branch ends where TIR, a face change,
path infeasibility, visibility, or the chart ends it), and only the sum over
symmetric paths closes into the full halo. The integral above applies to an
arc verbatim over the traced extent; the integrand is continuous down to zero
at a TIR boundary (`fresnel_transmission -> 0`) and simply stops at the other
events, so an arc is a legitimate partial contribution, not a failure. What
an arc does not include is the untraced tail between the last accepted pose
and the event on each side; the quadrature reports that as a per-end
truncation estimate (terminal integrand value times the linear-rate arclength
to the event, `quadrature.ResampledQuadratureResult.endpoint_truncation_estimate`
and `start_endpoint_truncation_estimate`), never added to the value. A
consumer that sums components MUST keep the closed / arc kind and the two end
events observable next to the value (`strip_pixel.ComponentRecord`,
task-pixel-pipeline-v2).

Every multiplicative factor MUST be independently observable before forming
`W_P`. Missing or unimplemented factors MUST be marked unavailable; they MUST
NOT be silently substituted by one in a result claiming physical completeness.

| Factor | Required meaning and normalization | Typical units | Phase I status |
|---|---|---|---|
| `rho_pose` | Density relative to explicitly named `d mu_Haar` or `dVol_g`; normalization state reported. | Dimensionless for Haar probability; inverse metric-volume for `dVol_g`. | Caller-supplied implementation available: `pose_density.ZenithGaussianPoseDensity` (Haar-relative, integrates to one; `task-single-fiber-physical-integrand`). |
| `entry_measure` | Projected/realizable entry measure for the selected path, with sign/clamping and reference area declared. | Dimensionless if area-normalized; otherwise area. | Available: `geometry.entry_measure` via `weights.entry_measure_weight` (absolute area, `length^2`, no reference-area normalization). |
| `visibility` | Obstruction/visibility result and, if soft, its declared transmittance model. | Dimensionless. | Interface/event required; not implemented (`visibility_boundary` stays a margin diagnostic). |
| `fresnel_transmission` | Product of polarization-resolved or explicitly unpolarized interface power factors; convention declared. | Dimensionless. | Available: `optics.fresnel_transmission_3_5` (unpolarized s/p average, entry times exit). |
| `path_validity` | Boolean feasibility of the exact ordered path, separate from numerical convergence. | Boolean gate, not a gain. | Available: `weights.path_validity_weight` (`path_3_5_domain` valid and `entry_measure > 0`). |
| `source_factor` | Source angular/spectral/radiometric quantity and sampling density. | Declared by source model. | Open until renderer/source contract. |
| `pixel_factor` | Pixel response/filter and solid-angle or projected-area conversion. | Declared by camera model. | Open until renderer/camera contract. |
| `other_radiometric` | Any absorption, phase, spectral, or exposure factor not represented above, individually named. | Explicit per factor. | Open; no anonymous catch-all in final results. |

`J_perp F_P` is a geometric coarea denominator, not a physical gain, and MUST
remain separately observable. A boolean path gate MAY exclude a sample; it
does not erase the event that caused exclusion.

## 8. Events, degeneracies, and unresolved topology

Event detection belongs to the value/evaluator path, not to AD through a
discontinuous branch. A result records event kind, the last accepted pose, the
rejected or bracketed trial pose when available, relevant scalar margins,
branch identifiers, and whether localization was attempted.

| Condition | Detection source | Required semantics |
|---|---|---|
| Total internal reflection | Snell discriminant and its signed margin, checked before square root. | `tir_boundary`; stop or localize. Never report only NaN/corrector failure. |
| Face/path branch transition | Explicit face-intersection and ordered-path predicates. | `branch_boundary`; do not change `P` silently. |
| Infeasible path | Entry/exit orientation, intersection ordering, or other named feasibility predicate. | `path_infeasible`; distinguish a bad seed from a boundary reached after accepted steps. |
| Obstruction/visibility boundary | Explicit obstruction predicate or signed clearance. | `visibility_boundary`; report even if its final weight would be zero. |
| Target-chart boundary or antipode | Declared `U_d` membership/margin. | `chart_boundary`; rebuild an equivalent chart only through an observable strategy, never accept the antipode as a root. |
| Jacobian rank loss/critical point | `sigma_2`, `J_perp`, and conditioning diagnostics relative to configured gates. | `rank_loss`; regular continuation/coarea stops. Exact singularity versus threshold detection must be distinguished. |
| Non-finite evaluator output | Finiteness checks plus any preceding physical margin. | Preserve a known physical event; otherwise `non_finite` numerical failure. |
| Corrector cannot converge | Iteration, residual, update, linear-solve, and retry diagnostics. | `corrector_failure` only after bounded recovery is exhausted. |
| Step underflow | Required safe step falls below configured minimum. | `step_underflow`, retaining the triggering condition. |
| Near return/self-intersection | Group distance, transverse section, tangent agreement, and segment history. | Diagnostic or explicit topology event; not closure unless every closure gate passes. |

Crossing TIR, branch changes, rank loss, bifurcations, or singular component
intersections is open. The reference solver MUST stop honestly at the supported
boundary rather than extrapolate the smooth branch. Discovering other connected
components and deduplicating them are external to a single `FiberResult`
(section 9.5); proving seed coverage is open.

An event is only evidence of a boundary when it is met by a corrector iterate
inside the acceptance trust region (`maximum_correction`, `maximum_advance`
of section 6.2). A Newton iterate outside it that lands in an invalid domain
could never have been accepted where it stands, so `_correct_trial` reports
it as a rejected trial (`corrector_failure`, the step shrinks) rather than an
event (task-pixel-pipeline-v2: a `0.17` loop whose first `0.04` predictor
sent the corrector `1.18 rad` away into a TIR region was reported as
`tir_boundary` with no accepted step).

Standard downstream use of an event-terminated result (`discovery`): a trace
that ends on one of the five named boundary events (`tir_boundary`,
`branch_boundary`, `path_infeasible`, `visibility_boundary`,
`chart_boundary`) is traced once more from the same seed with
`initial_tangent_sign = -1`; when that also ends on a named event the two
traces are stitched (`resample.stitch_open_arc`) into one *open arc*
component whose curve runs from the backward end through the seed to the
forward end (reversed backward samples with negated tangents, the seed knot
carrying the forward tangent). The event classification itself is unchanged;
`rank_loss` and `topology_ambiguity` stay "open" and never form an arc, and
a backward trace that fails, or that closes where the forward trace did not
(a contradiction the data cannot resolve), leaves the candidate `incomplete`
with both traces kept.

## 9. Backend-independent interface semantics

Names in this section are descriptive schema names, not required Python class
names. A backend MAY reorganize fields, but serialized diagnostics and public
documentation MUST preserve their meanings, shapes, units, and availability.

### 9.1 `FiberProblem`

| Field semantics | Requirement |
|---|---|
| `path` | Immutable identifier and parameters for the ordered path branch `P`, including wavelength/material data needed by the evaluator. |
| `incident_direction` | Finite world-space `(3)` unit vector `s`, with convention version. |
| `target_direction` | Finite world-space `(3)` unit vector `d`, with convention version. |
| `direction_evaluator` | Maps a pose to outgoing `(3)` direction plus named branch/physical margins; differentiable only inside its declared smooth domain. |
| `domain_and_event_evaluator` | Reports validity and typed event candidates without relying on differentiating discrete predicates. It may be combined with the value evaluator if outputs remain distinct. |
| `target_chart` | Target neighborhood plus orthonormal tangent basis, or a general chart with its metric density. |
| `pose_metric_and_measure` | The section 5.1 metric and either `dVol_g` or `d mu_Haar`; alternative normalizations require an explicit conversion. |
| `seed` | Initial pose with storage representation declared; it denotes a pose in `SO(3)`, not a unique quaternion representative. |
| `weight_evaluators` | Optional independently named factor evaluators (`weights.WeightEvaluator`: callable plus declared unit and normalization). Absence is reported and does not prevent geometry-only tracing; evaluation happens on accepted poses after continuation and never alters termination. |

Problem construction MUST validate finite values, unit directions, convention
compatibility, and required evaluator capabilities before continuation. It MUST
not import or invoke Lumice. A backend-specific callable or AD mechanism is an
adapter behind these semantics, not part of the contract.

### 9.2 `ContinuationOptions`

Options MUST make the following numerical policy observable; no value is fixed
by this document:

- reference dtype and unit-direction validation tolerance;
- seed/root residual absolute and relative tolerances;
- target-neighborhood/antipode margin;
- singular-value, normal-Jacobian, rank, and condition-number gates;
- initial, minimum, and maximum step; shrink/growth limits and retry budget;
- corrector iteration, residual, update, and trust limits;
- tangent-change and accepted-advance limits;
- event prediction/localization margins and iteration budget;
- maximum accepted steps, evaluations, wall-independent work units, and
  accumulated arclength;
- closure minimum steps/arclength, pose distance, transverse-section crossing,
  tangent agreement, and final correction tolerances;
- requested diagnostic level and sample retention policy.

Any default MUST be documented as a reference implementation strategy and
linked to convergence evidence. Relaxing a default MUST NOT alter the exact
meaning of a root, regularity, path validity, or closure.

### 9.3 `FiberResult`

Let `N` be the number of accepted pose samples. A result MUST contain:

| Field semantics | Shape or requirement |
|---|---|
| `status` | Exactly one of `closed`, `event_terminated`, `numerical_failure`, or `budget_exhausted`. |
| `reason` | A reason code compatible with `status`; unknown extension codes retain their original string/payload. |
| `component_scope` | States that this is one component reached from one seed; component completeness is `unknown` unless established externally (for example by the post-hoc check of section 9.5.6a). |
| `poses` | `N` ordered `SO(3)` poses in the declared representation, with representation validity diagnostics. |
| `arclength_increments` | `N - 1` nonnegative metric edge lengths, plus any separately represented closing edge if the storage convention omits a repeated seed. |
| `residual_norms` | `(N)` norms and the norm definition/tolerances used. |
| `tangents` | `(N, 3)` oriented right-trivialized unit tangents where available. |
| `jacobian_diagnostics` | Per-sample singular values, `J_perp`, rank/condition estimates, and availability flags. |
| `step_diagnostics` | Accepted/rejected trial counts, step sizes, corrector iterations, residual/update outcomes, tangent changes, and reason codes. |
| `branch_diagnostics` | Path identifier and named validity/event margins at accepted and terminal evaluations. |
| `closure_diagnostics` | Accumulated length, stable seed distance, section values/crossing, tangent agreement, and final-correction outcome. |
| `terminal_payload` | Last accepted state and relevant rejected/bracketed state, event/failure scalars, iteration/budget counters, and message. |
| `conventions` | Coordinate/sign version, pose representation, metric/measure, dtype, units, and solver/options version. |
| `weight_observables` | Each requested factor and availability status separately (`weights.WeightObservable`: status, unit, normalization, `(N)` values when available); never only an opaque product. |

Unavailable data MUST be explicit rather than encoded as a plausible zero,
one, empty success value, or NaN without a reason. Partial samples on any
non-closed termination MAY support diagnostics, but MUST NOT be reported as a
closed-component integral.

### 9.4 Status and reason mapping

| `status` | Required interpretation | Representative `reason` values |
|---|---|---|
| `closed` | Every closure and ordinary acceptance gate passed for the traced component. | `closed_loop` |
| `event_terminated` | A known physical, chart, regularity, or topology boundary ended supported smooth continuation. | `tir_boundary`, `branch_boundary`, `path_infeasible`, `visibility_boundary`, `chart_boundary`, `rank_loss`, `topology_ambiguity` |
| `numerical_failure` | Bounded numerical recovery failed without a more specific known domain event. | `corrector_failure`, `linear_solve_failure`, `non_finite`, `step_underflow`, `invalid_numerical_input` |
| `budget_exhausted` | Inputs remained meaningful but a declared work/extent budget ended the trace before another terminal condition. | `step_budget`, `arclength_budget`, `evaluation_budget` |

Status precedence is based on the best available causal evidence. For example,
a non-finite refraction result preceded by a negative Snell discriminant is
`event_terminated/tir_boundary`, not `numerical_failure/non_finite`. Exhausting
retries while localizing a known branch boundary retains the boundary event and
records localization failure in its payload.

### 9.5 Discovery interface semantics

Sections 5–9.4 trace one component from one supplied seed. Discovery is the
layer above them: for one fixed path `P`, crystal, refractive index, incident
direction `s` and target `d`, it produces the seeds, traces each distinct
component once, and classifies what it traced. Its contract is weaker than the
single-component one in exactly one respect, stated first: **discovery does not
prove that every connected component of `X_(P,d)` was found** (section 9.5.6).
Everything else below (the sampling measure, the candidate construction, the
clustering, gates and deduplication, the classification and the output) is
normative, so that two conforming backends given the same sample return the
same components and the same counters.

The pipeline fixed here is the `reference-discovery-v1` strategy of
`lumice_integral.discovery`. Its step order and gate semantics are normative
because a parity fixture must be able to reproduce them from a shared sample;
its numerical defaults (section 9.5.9) are strategy, as in section 10.1. A
change of the pipeline itself (another clustering rule, another gate, another
dedup test) is a change of this section and MUST be recorded here with a new
strategy version, not only in code. As in the rest of section 9, the names
below (`DiscoveryProblem`, `DiscoveryResult`, `Component`, ...) are schema
names for reference in this document, not required class names; the Python
reference spells them `discover_components(...)`, `ComponentDiscoveryResult`,
`DiscoveredComponent` and `IncompleteCandidate`.

#### 9.5.1 `DiscoveryProblem`

| Field semantics | Requirement |
|---|---|
| `path` | One concrete ordered face sequence `P` (the numbering of `docs/conventions.md` #1), no symmetry parameter: a caller that wants a family of paths calls discovery once per member. |
| `crystal` | The finite crystal of `P`'s geometry. It enters discovery twice: through the face normals (direction map and path domain, section 5.2) and through the finite-crystal `entry_measure` gate (section 7). |
| `refractive_index` | The ice index at the wavelength of the call. |
| `incident_direction` | `s`, as in section 9.1. The sun direction `ŝ = -s` is derived, never supplied independently. |
| `target_direction` | `d`, as in section 9.1, with deviation `delta = angle(s, d)` in the open interval `(0, pi)` (section 9.5.3 needs a component of `d` normal to `s`). |
| `sample` | A point set `u_1 ... u_N` on `S2`, uniform with respect to `dA(u)/(4 pi)` (section 9.5.2), or its band and fields for this `delta` (section 9.5.3). A backend MAY generate it inside the call. It MUST NOT be required in any persisted store format. |
| `extra_seeds` | Optional poses (warm starts, typically the converged seeds of a neighbouring target). Used only as Gauss-Newton starts of their cluster (section 9.5.4). They are never traced on their own and never count toward completeness. |
| `band_half_width` | Positive angle `b`: the sample events whose deviation lies within `b` of `delta` are the candidates. |
| `cluster_radius` | Positive `SO(3)` geodesic radius `r_c` of the candidate clustering. |
| `distance_threshold` | Positive `SO(3)` geodesic distance `eta` under which a corrected candidate lies on an already accepted component. A non-positive value MUST be rejected. |
| `continuation` | One `ContinuationOptions` (section 9.2) used by every trace of the call. There is no separate discovery budget. |

The problem is determined by `delta` up to a world rotation: rotating `d` about
`s` by an angle `alpha` maps the fiber by left multiplication with that
rotation. A backend MAY exploit this. A conformance fixture MUST still state
`d` itself.

#### 9.5.2 The sampling measure

Discovery samples the path's fields on the sphere of `u = R^-1 ŝ`, the sun in
the crystal frame (`docs/phase2.md` section 1). With the Haar decomposition of
section 5.1,

```text
d mu_Haar(R) = dA(u)/(4 pi) · dpsi/(2 pi),
```

where `psi` is the twist of `R` about `ŝ`. The fields discovery uses (validity,
entry measure `A`, Fresnel factor `T`, body-frame outgoing propagation
direction `phi = Phi_P(-u)`, and the deviation `D(u) = angle(phi, -u)`) depend
on `R` only through `u` (`docs/phase2.md` section 4.1(a)). So a uniform sample
of `SO(3)` quotiented by `psi` is a uniform sample of `u` on `S2`, and that is
the only input discovery needs.

- The sample MUST be uniform in `dA(u)/(4 pi)`: either i.i.d. uniform points or
  a deterministic quasi-uniform set with equal-area cells. A sampler with a
  non-uniform density (inverse weights) MUST NOT be used as a discovery sample,
  and the reference refuses one (`s2_store.StoreSeeds`).
- An event `i` is kept only where `w_i = A(u_i) T(u_i) > 0`: the path is valid
  at `u_i` and the finite crystal admits a positive entry measure. Only kept
  events are candidates.
- The reference sampler is the antipodal Fibonacci lattice,
  `u_i = -(sqrt(1 - z_i^2) cos theta_i, sqrt(1 - z_i^2) sin theta_i, z_i)`
  with `z_i = 1 - (2 i + 1)/N` and `theta_i = pi (1 + sqrt 5)(i + 1/2)`,
  `i = 0 ... N-1` (`s2_store.store_lattice`). Its points are indexed by `i`, and
  that index orders ties (section 9.5.3). The i.i.d. alternative
  (`s2_store.RandomSphereSampler`) draws per chunk of points from a seeded
  generator, so its first `N_1` points are the first `N_1` points of any larger
  draw. Nested samples are what section 9.5.7 talks about.
- The sample is independent of `ŝ` and of `d`. One sample serves every target
  and every sun direction of a scene. For a member `P' = g P` of a symmetry
  class, a backend MAY transport `P`'s sample by the crystal symmetry `g`
  (`u' = g u`, `phi' = g phi`, the same `D` and `w`; `docs/phase2.md` section
  4.1(d), `s2_store.transported_rotations`) instead of sampling `P'` again.

#### 9.5.3 Band and candidate poses

For the target `d` at deviation `delta`, the band is every kept event with
`|D_i - delta| <= b`, inclusive at both ends, taken in increasing `D_i` and,
for equal `D_i`, in increasing sample index. Its size is `pool_count`. The
candidate pose of event `i` puts `u_i` on `ŝ` and `phi_i` in the azimuth of `d`:

```text
R_i = W F_i^T,
W   = [ŝ, e, ŝ x e],      e = unit(d - (d·ŝ) ŝ),
F_i = [u_i, f_i, u_i x f_i], f_i = unit(phi_i + cos(D_i) u_i),
```

with columns as listed (`s2_store.event_rotations`). Then `R_i u_i = ŝ` and
`R_i phi_i` lies at deviation `D_i` in the half-plane of `d`, so the outgoing
direction of candidate `i` is exactly `|D_i - delta|` from `d`. The candidate's
residual is one-dimensional, and its *offset* is `|D_i - delta|`. `e` is
undefined for `d = ±s`. That is why `delta` is required to lie in `(0, pi)`,
and the reference does not guard this case (open item, section 12).

The candidate pool is `extra_seeds` in the caller's order, followed by the band
in the order above. Pool indices are positions in this concatenation.

#### 9.5.4 Clustering, correction, gates, and deduplication

1. **Clustering.** Greedy geodesic clustering of the whole pool. Repeatedly,
   the unassigned member with the lowest pool index becomes a centre, and its
   cluster is every unassigned member (the centre included) whose `SO(3)`
   geodesic distance to the centre is strictly below `r_c`. Membership is
   distance to the centre, not transitive. Clusters are processed in creation
   order, and their number is `raw_cluster_count`. (The reference realises "lowest index" through the
   iteration order of a Python `set` of small integers. The rule is pinned by
   `test_geodesic_cluster_centres_are_the_lowest_unassigned_index_and_membership_is_not_transitive`.)
2. **Representative.** If the cluster contains extra seeds, the one with the
   lowest pool index represents it. Otherwise the band member with the smallest
   offset does, with ties going to the lowest pool index. Only the
   representative goes further. The cluster's other members are not corrected.
3. **Gauss-Newton correction.** Starting from the representative, iterate at
   most 30 times: evaluate the residual `r_d(R)` of section 5.3 and its `2 x 3`
   right-trivialized Jacobian `A` (section 5.4). If `|r_d| <= tau / 100`, stop
   at the current pose. Otherwise update
   `R <- R exp([ -A^T (A A^T)^-1 r_d ]_x)` (the minimum-norm step). Here
   `tau = residual_tolerance + relative_residual_tolerance` of the call's
   continuation options (the reference adds the two; with the defaults of
   section 10.1, `tau = 1e-11`).
4. **Admissibility gate.** The corrected pose is *admissible* iff its residual
   norm is at most `tau`, the path domain (section 5.2, entry and exit Snell
   discriminants and every internal incidence) is valid there, and the
   finite-crystal `entry_measure` there is positive. The sample passed through
   `w > 0` before correction. The corrected pose is gated again. The number of
   admissible representatives is `admissible_count`. An inadmissible
   representative is dropped silently, and the other members of its cluster
   are not tried in its place. The target-chart gate (the antipode exclusion,
   section 5.3) is not part of admissibility. It is the seed precondition of
   the trace (section 6.1).
5. **Deduplication before tracing.** An admissible pose whose `SO(3)`
   geodesic distance to some accepted pose of an *already accepted component*
   is strictly below `eta` is that component. It is folded without a trace and
   counted `dedup_merged`. The distance is taken to the stored poses of the
   component's whole curve, both sides of an arc included. Incomplete
   candidates (section 9.5.5) are not deduplication targets, so a later
   candidate on an unresolved piece of the fiber is traced again.

#### 9.5.5 Trace and classification

Every admissible, non-folded pose is traced once with the call's
`continuation`, and the outcome is classified:

| Forward trace | Backward trace (same seed, `initial_tangent_sign` negated) | Outcome | Counter |
|---|---|---|---|
| `closed` | not run | component, `kind = closed` | none |
| `event_terminated` by one of the five *arc events* `tir_boundary`, `branch_boundary`, `path_infeasible`, `visibility_boundary`, `chart_boundary` | `event_terminated` by an arc event | component, `kind = arc`, the two traces stitched (below) | `arc_stitched` |
| same | `closed` | incomplete (the forward trace of the same one-dimensional fiber should have closed first) | `arc_backward_closed_anomaly` |
| same | any other outcome | incomplete | `arc_backward_failed` |
| `event_terminated` by another reason (`rank_loss`, `topology_ambiguity`, ...) | not run | incomplete | `incomplete_unnamed_event` |
| `numerical_failure` or `budget_exhausted` | not run | incomplete | `incomplete_not_converged` |

The stitched arc (section 8, `resample.stitch_open_arc`) runs from the backward
end through the seed to the forward end. Its poses are the backward poses
after the seed in reverse order, followed by all forward poses. The tangents
are the reversed backward tangents after the seed, negated, followed by the
forward tangents, so the seed carries the forward tangent. The increments are
the reversed backward increments followed by the forward increments. The seed
index is the number of backward poses after the seed. The arc's `reason` is
the forward end's event and its `start_reason` the backward end's. Either
trace MAY have no accepted step (its only pose is the seed). An arc whose two
traces both stopped at the seed is a one-pose arc of length zero (section
9.5.6).

A component's `arclength` is the sum of its increments. A closed component's
`status`/`reason` are those of its trace. An arc's status is
`event_terminated` and its reason is the forward end's event.

#### 9.5.6 `DiscoveryResult` and what `completeness` means

| Field semantics | Requirement |
|---|---|
| `components` | The accepted components in trace order. Each has `kind` (`closed`/`arc`), the corrected seed, the curve (one closed trace, or the stitched arc with its two traces), `arclength`, `status`/`reason`, and for an arc `start_reason`. |
| `incomplete` | The candidates that were traced but not classified: seed, `cause` (one of the four incomplete counters), the forward trace, and the backward trace when one was run. They are evidence of nothing conclusive: neither a component nor a duplicate of one. |
| `completeness` | `complete` iff `incomplete` is empty, otherwise `unknown`. No other value. |
| `pool_count`, `extra_seed_count`, `raw_cluster_count`, `admissible_count` | The funnel counts of sections 9.5.3–9.5.4. |
| `events` | The six counters `dedup_merged`, `arc_stitched`, `arc_backward_failed`, `arc_backward_closed_anomaly`, `incomplete_unnamed_event`, `incomplete_not_converged` (`discovery.DISCOVERY_EVENT_NAMES`), every one present, zeros included. |

A conforming result satisfies the funnel identities
`admissible_count = dedup_merged + len(components) + len(incomplete)`,
`arc_stitched = number of arc components`, and `len(incomplete)` equals the sum
of the four incomplete counters. A backend MAY add diagnostics such as the
reference's `trace_seconds`, but they are not part of parity.

`completeness` is **procedural, not a certificate**. `complete` means only that
every admissible, non-folded candidate of this pool closed or stitched. A target
with no admissible candidate is therefore `complete` with zero components. It
does not mean every connected component of `X_(P,d)` was found. Three
mechanisms, at three stages of the pipeline, make a component invisible to it:

1. **No band event** (sampling, before section 9.5.3). The component's region
   of `u` receives no event of the pool, so it has no candidate at all.
2. **Clustered into another component** (section 9.5.4, step 1, radius `r_c`).
   Every one of its raw candidates lies within `r_c` of a cluster centre on
   another component. Only the representative is corrected, so the component
   is never corrected, gated or traced. The comparison is between raw
   candidate poses, before any correction.
3. **Deduplicated into a traced neighbour** (section 9.5.4, step 5, radius
   `eta`). The component has its own cluster, its representative corrects and
   passes the admissibility gate, but the corrected pose lies within `eta` of
   the stored curve of a component that has already been accepted and traced.
   It is folded as that component and counted `dedup_merged`, and it leaves no
   `incomplete` record. The comparison is between a corrected, admissible pose
   and a traced curve: the two true curves are closer than `eta`, although
   events, clustering and correction are all sufficient. On path `3-5` inside
   the certificate interval `(42.99086°, 43.46516°)` (`n_open = 4`), the
   missing fourth component is a one-pose arc whose pose lies `1.224e-3 rad`
   from a neighbour's curve, below the default `eta = closure_distance`, and
   discovery returns 3 components at production gate radii.

`completeness` MUST NOT be reported as component completeness. The
single-component `component_scope` of section 9.3 stays `unknown` and remains
authoritative for quadrature (section 7). Section 9.5.6a gives the optional
check that can establish the component count externally.

The one quantitative statement available is probabilistic. For `N` i.i.d.
uniform points, a region of `u` of area `mu` receives no event with probability
`(1 - mu/(4 pi))^N <= exp(-N mu/(4 pi))` (`discovery.miss_probability`). The
cross-check `discovery.check_band_coverage` revisits every band event, not one
per cluster. An event posed within `near_radius` (default `closure_distance`,
the default `eta`) of a traced curve is covered. Every
other event is corrected and gated exactly as in section 9.5.4. An admissible
pose farther than `eta` from every traced curve is a *suspect*, which is
evidence of a missed component or of an under-traced one. With `k_min` the
fewest band events assigned to any traced component, `exp(-k_min)` estimates
the chance that a component of that band measure receives no event. On the
deterministic Fibonacci lattice this is the Monte Carlo reading of the same
density, not a bound. The cross-check is a diagnostic and a test tool, not a
rendering step, and it does not change `completeness`.

**Arc extent depends on the seed**, a property of the reference a caller MUST
allow for. When a seed lies within about one
`initial_step` of a named boundary, the first predictor step of that side
crosses it and the trace ends there with no accepted step (section 8: an
event met by an iterate inside the trust region). That side of the arc is
then truncated at the seed. On path `1-3` at 65 deg the same two arcs measure
0.258/0.259 rad from one sample and 0.196/0.171 rad from a denser one. When
both sides stop at the seed, the component is a **one-pose arc** of length
zero. It is counted, because it is a distinct piece of the fiber, but its
extent is not traced. `check_band_coverage` reports the events of its
untraced extent as suspects. An arc's length is therefore not a parity
quantity across samples or backends; its seed, kind and end events are. The
quadrature's per-end truncation estimate (section 7) is the only account of
an arc's untraced tail.

#### 9.5.6a Post-hoc certification

**There is no unconditional completeness certificate for discovery**: no
statement independent of the crystal and the path guarantees that a discovery
call found every component of `X_(P,d)`. The counterexample is mechanism 3 of
section 9.5.6 on path `3-5`: the `dp_field` certificate says 4 components on
`(42.99086°, 43.46516°)`, discovery stably returns 3, and the missing one is
a one-pose arc `1.224e-3 rad` from a traced neighbour.

What the contract provides instead has three levels.

1. **`completeness` keeps its meaning.** It stays procedural (section 9.5.6),
   with the values `complete` and `unknown` only. The check below does not
   add a value, does not rewrite `completeness`, and changes no return value
   of `discover_components`.
2. **An optional post-hoc check, run by the caller outside discovery.** For
   the deviation `delta` of the target `(path, d)`, the caller builds
   `dp_field.DPField.build(crystal, faces, index)` independently and reads the
   interval of `DPField.interval_partition()` that contains `delta`. Its
   structural preconditions are explicit, never a silent downgrade:
   - the field exists only for `halo_map_rank != 0`; a rank-0 path is a point
     mass at the sun and `DPField.build` raises `ValueError`;
   - the partition exists only inside its disk reasoning: `U_P` a disk, at
     most one interior critical point and that one an extremum, no slab crease
     through `U_P`, alternating loop extrema, and an even number of boundary
     crossings. Outside it `interval_partition` raises `TopologyEscape`, and
     the check is unavailable (not failed);
   - `delta` at a critical value lies on no open interval and has no count.

   The rule is one-directional. If discovery's component count equals the
   interval's `n_components`, that is **sufficient evidence** of component
   completeness for this call (the topological truth matches the sample).
   If they differ, the `dp_field` count is the **topological truth** and the
   discovery result is **undersampled**: the caller reseeds or reruns with a
   smaller `distance_threshold`, and does not accept
   `completeness == "complete"` at face value. The counts are compared as a
   whole; the check does not say which component is missing.
3. **Excluded: a constructive threshold rule.** No rule computes, from the
   certificate's macroscopic data (the `CriticalSet`, the spacing of the
   boundary crossings), a `distance_threshold` or `cluster_radius_rad` that
   always avoids undersampling. On path `3-5` the eight boundary crossings are
   at least `0.441 rad` apart, and the collapse that hides the fourth component
   is `1.224e-3 rad`, nearly 400 times smaller: the macroscopic data do not
   predict the local degeneracy. Tightening the threshold is not a monotone
   fix either. From `0.02` to `0.08 rad` the count is a stable 3; a threshold
   small enough to separate the one-pose arc also splits redundant candidates
   of one curve (corrected `1e-7`–`1e-10 rad` apart) into new "components",
   and the count jumps to 6, 7 or 24. The two scales do not overlap, so no
   single threshold resolves both. The one-pose arc exists because its seed is
   already within one predictor step of a boundary or a neighbour, a scale set
   by the path's local geometry.
4. **Excluded: a diagnostic inside the dedup step.** Flagging one-pose arcs
   in `discover_components` was evaluated (explore
   `discovery-one-pose-arc-dedup-heuristic`) and rejected; this is not an
   open item. A classification field on the result never fires: on path
   `3-5` at `43.22801°` the arc's seed is `1.208e-3 rad` from a traced curve
   at the `distance_threshold` check, so it is merged (`dedup_merged`) and
   never traced. The merge distances do not separate it either: the other
   three merges of that call, redundant samples of one curve, are `3.3e-3`,
   `1.08e-2` and `1.50e-2 rad`, and the dangerous one is the smallest. The one
   clean discriminant, `n_poses == 1` after tracing each merged candidate,
   costs a trace per candidate, which undoes the dedup. The check of item 2
   costs one `DPField` per `(path, delta)` and answers the same question.

#### 9.5.7 Densification (the low-then-dense calling pattern)

A consumer that first discovers at a low `N` and later densifies (for example
an interactive low-resolution view) needs to know whether a denser sample can
lose what a sparser one found. **It is not guaranteed.** The greedy clustering
of section 9.5.4 is not monotone in the pool: added events move cluster
centres and representatives. Only one representative per cluster is corrected,
and an inadmissible representative is not replaced, so a component found
through one cluster at low `N` can fall into a cluster represented by a pose of
another component at higher `N`.

Measured (bounded evidence, nested i.i.d. samples, each the prefix of the
next):

- On 8 fibers (path `3-5` at the canonical pixel `(150, 150)`, the caustic
  pixel `(49, 0)` and the boundary-hugging pixel `(700, 150)`; path `1-3` at
  60, 63.5 and 65 deg; path `3-1` at 64.7434 deg; the `D3h` prism's `5-3` at
  43.0347 deg), over `N = 1e4, 3e4, 1e5, 3e5, 1e6`, every component with at
  least one accepted step found at a lower `N` was found again at the next `N`,
  meaning a pose within `eta` of a returned curve. The component count never
  dropped, except in the case below.
- Counterexample: the `D3h` prism, path `5-3` at 43.0347 deg, `3e5 -> 1e6`.
  The three long arcs are kept, but a one-pose arc found at `3e5` is not near
  any curve at `1e6`, and the `1e6` cross-check reports suspects.

So a consumer MUST NOT assume that densification only adds components. Where
it matters, it SHOULD run `check_band_coverage` on the denser result, or
retain the sparser components as `extra_seeds` of the denser call.
Components with an accepted step have been monotone on every fiber measured
so far, but that is evidence, not a property.

#### 9.5.8 v0 output subset (the Lumice `liblumice_analytic` module A)

Lumice `doc/analytic-api.md` section 4.5 draft returns per trace only the
kinematic `LUMICE_ANALYTIC_FiberResult` fields. Discovery's output maps onto
them without new trace fields:

| Discovery item | v0 representation |
|---|---|
| Closed component | Its seed (`double[9]`, row-major, body to world), `kind = closed`, and one `FiberResult`: `status`, `reason`, `poses` `(N, 9)`, `crystal_frame_sun_directions` `u = R^T (-s)` `(N, 3)`, `arclength_increments` `(N-1)`, `residual_norms` `(N)`, `tangents` `(N, 3)`. |
| Arc component | Its seed, `kind = arc`, and the two `FiberResult`s (forward, then backward with the sign reversed). The stitched curve, its seed index, `reason` and `start_reason` follow from them by section 9.5.5 and need not be stored. |
| Incomplete candidate | Its seed, its `cause`, the forward `FiberResult`, and the backward one when run. |
| Result | `completeness`, the four funnel counts, and the six counters of section 9.5.6. |

Jacobian, step, branch and closure diagnostics, terminal payloads and weight
observables (section 9.3) are **not** in the v0 subset. They are the wave-2
diagnostics contract (`explore-fiber-diagnostics-contract`). The `check_band_coverage`
cross-check is likewise not part of v0.

A parity fixture for discovery SHOULD carry the band itself (`u_i`, `phi_i`,
`D_i` in pool order) next to `d`, so that a backend's steps 9.5.3–9.5.5 are
compared on an identical pool. The sampler is compared separately against
the formula of section 9.5.2. Parity quantities are the funnel counts and
counters, each component's kind, seed and end events, and the traced poses
within the continuation tolerances. Arc lengths are compared only within the
same seed (section 9.5.6).

#### 9.5.9 Reference defaults and evidence

The `reference-discovery-v1` defaults. Like section 10.1, they are strategy,
not mathematical constants.

| Parameter | Default | Evidence |
|---|---|---|
| Sampler | antipodal Fibonacci lattice (section 9.5.2) | The Haar mean of `w` on the lattice agrees with independent uniform quaternions (`s2_store.self_check_haar_mean`); `psi` invariance is checked by `self_check_psi_invariance` (`docs/phase2.md` section 4.1(a)). |
| `N` | `1e6` (`s2_store.DEFAULT_SEED_STORE_N`) | Task `phase1-seeds-from-store`: 32 strip pixels found every component already at `N = 1e5` with a `0.02` deg band. The store density survey (`scripts/store_seed_density_survey.py`, `docs/ch06-reference-fixture.md` section 7) pins the production size, and `check_band_coverage` found 0 suspects on the production configuration. |
| `band_half_width` | `0.2` deg | With `N = 1e6`, a pool the size of the retired prescan's. The same probe as `N`. |
| `cluster_radius` | `0.3` rad | `explore-component-discovery` (34+ pixels): distinct components of the surveyed fibers lie farther apart. The two `1-3` arcs at 60 deg are `0.82` rad apart (`test_path_1_3_at_60_deg_is_two_distinct_components`). |
| `distance_threshold` | `closure_distance = 0.08` | The same "is this pose on that curve" scale as closure. It is above half the largest accepted chord (`maximum_step / 2 = 0.06`), so a pose on a traced curve is never farther than that from its nearest sample. |
| Gauss-Newton | 30 iterations, stop at `tau / 100`, accept at `tau` | Reference constants of `discovery._newton_correct` / `_admissible_seed`. No failure attributed to them has been observed. |
| `continuation` | section 10.1 | Discovery adds no trace policy of its own. |

#### 9.5.10 Boundaries

- Crystals: the reference samples and discovers on the closed-form hexagonal
  prism of any `face_distance` (`geometry.HexPrism`). Pyramids are refused
  before discovery by the sample store, which waits for task
  `pyramid-lumice-semantics`
  (`test_pyramid_is_refused_with_a_forward_pointer_to_its_own_task`). On the
  prism the direction map and the path domain depend on the face normals
  only. The finite extent enters only through `entry_measure`. On the `D3h`
  fixture of section 11 the discovered fiber is the regular prism's, and only
  the entry measure differs.
- Discovery covers one target direction. Neighbour continuation across
  targets, full-image completeness and the strip driver's policies are outside
  it (`strip_pixel`, `strip_driver`).
- It inherits the single-component boundaries of section 8. No trace crosses
  TIR, a branch change, rank loss or a singular intersection. A fiber that
  changes topology between two targets is discovered afresh at each.

## 10. Truth, tolerance, and strategy separation

| Mathematical truth | Numerical tolerance or evidence | Permitted implementation strategy |
|---|---|---|
| `SO(3)` has the metric/volume normalization in section 5.1. | Orthonormality and determinant residuals have configured finite tolerances. | Matrix, quaternion, or another faithful pose representation. |
| A regular root has `F_P(R)=d`, rank two, a one-dimensional tangent kernel, and positive `J_perp`. | Residual, `sigma_2`, `J_perp`, and conditioning gates approximate these facts. | AD, analytic derivatives, or verified numerical derivatives. |
| An orthogonal target-basis change does not alter the physical fiber or `J_perp`. | Conformance compares results within declared errors. | Deterministic, transported, or reconstructed orthonormal basis. |
| Arclength and coarea use the declared Riemannian measures. | Quadrature and edge-length convergence are reported. | Adaptive or fixed quadrature once independently converged; corrected edge geometry may vary. |
| Closure is a return through the seed's local section with compatible orientation. | Distance, section, tangent, minimum-length, and correction tolerances are options. | Bordered Newton, pseudo-arclength, or an equivalent corrector. |
| A domain boundary is not a smooth-root solver failure. | Event margins and localization tolerances approximate its location. | Bracketing, step clipping, dense output, or honest termination before the boundary. |

Current observations such as 145 fixed steps for the synthetic 3-5 trace,
specific residual magnitudes, and float32 drift are evidence for selecting and
testing defaults. They MUST NOT appear as universal pass criteria. Similarly,
one successful closed loop establishes only that one seeded trace under one
configuration converged.

### 10.1 Reference defaults and bounded convergence evidence

The following values are the `reference-continuation-v1` strategy as exercised
by the conformance suite. They are not mathematical constants and do not widen
the root, regularity, path-validity, or closure definitions above.

| Policy group | Current reference defaults |
|---|---|
| Precision and root | `dtype=float64`, `unit_tolerance=1e-10`, `rotation_tolerance=1e-10`, `residual_tolerance=1e-11`, `relative_residual_tolerance=0` |
| Regularity | `singular_value_tolerance=1e-8`, `condition_limit=1e8` |
| Step controller | `initial_step=0.04`, `minimum_step=1e-5`, `maximum_step=0.12`, `shrink_factor=0.5`, `growth_factor=1.25`, `maximum_retries=8` |
| Corrector and trust gates | `corrector_maximum_iterations=10`, phase/update tolerances `1e-12`, `maximum_correction=0.2`, `maximum_advance=0.2`, `minimum_tangent_dot=0.8` |
| Seed orientation | `initial_tangent_sign=+1` (section 5.4 explicit choice; `-1` reverses the SVD sign of the seed tangent and hence the sample order; that sign is deterministic per LAPACK build, not across builds, so an open arc is traced in both orientations) |
| Work bounds | `maximum_accepted_steps=4000`, `maximum_evaluations=100000`, `maximum_arclength=20` |
| Event approach | `event_slowdown_margin=0.02`; a shrinking margin below it bounds the next step by `_EVENT_APPROACH_STEP_FRACTION` (code constant, `0.5`) of the linear arclength to zero; a stable or receding margin imposes no bound |
| Closure | minimum `closure_minimum_steps=3` accepted steps and arclength `max(closure_minimum_arclength=0, _CLOSURE_ARCLENGTH_STEP_MULTIPLIER * initial_step)` with the code constant `_CLOSURE_ARCLENGTH_STEP_MULTIPLIER = 2.0` (the constant in `lumice_integral.continuation` is the single source of that value), distance `0.08`, tangent dot `0.8`, section tolerance `1e-11`, at most 10 final-corrector iterations |

On the analytic circle, initial steps `0.04`, `0.08`, and `0.12` all terminate
as `closed/closed_loop`, have maximum residual no greater than `1e-11`, and
recover length `2 pi` within `1e-12`; the test separately verifies the
Haar-to-sphere identity `(2 pi)/(8 pi^2) = 1/(4 pi)`. With `initial_step` and
`maximum_step` both `0.2` the circle still closes on its first traversal with
exactly one sign change of the seed-relative transverse coordinate. On the
synthetic 3-5 branch, initial steps `0.03`, `0.04`, and `0.08` all close with
maximum residual below `1e-11` and closure gap below `2e-13`. Their discrete
metric lengths lie between `0.9643243178593905` and `0.9647199228642988`, a
span below `0.0017`. (The `3.857976632802349` recorded before
2026-09-17 was this same loop traversed four times: the absolute closure
bound of `pi` arclength could only be met on the fourth return.) An additional
3-5 run holds `initial_step=0.04` fixed while changing `minimum_step` to
`2e-5`, `maximum_step` to `0.10`, `shrink_factor` to `0.4`, `growth_factor` to
`1.15`, and `maximum_retries` to `10`. It also closes with the declared
residual and closure bounds; its length differs from the reference by less than
`0.0015`, and the bidirectional sampled-pose set distance is below `0.012` rad
(`0.0093` observed on one traversal; the earlier `0.008` bound was measured on
four interleaved traversals). This exercises controller thresholds in addition
to initial-step selection.

Short loops: the analytic conjugation circle `F(R) = normalize((cos phi,
sin(phi) u . a, 1))`, whose fiber through `Rot(u0, pi/2)` has the closed-form
length `4 pi sin(pi/4) sin(beta)`, closes on its first traversal at lengths
`1.0` and `2.0` with capped steps `0.01`, `0.02`, `0.04`, and `0.2`; the
chord-polygon length converges to the closed form at second order (error
ratio between 3 and 5 per step halving, finest error below `2e-4` relative).
Under the retired absolute gate (`closure_minimum_steps=40`,
`closure_minimum_arclength=pi`, passed literally) the same fibers close at
four and two times their length. On the ch06 strip, column 126 rows 100, 150,
and 224 close at `1.645239`, `2.375620`, and `3.111244` (recorded to `1e-6`,
Mac float64) and at twice those lengths under the retired gate; column 150
rows 700 and 780, whose loops run parallel to the exit TIR boundary with
`exit_snell_discriminant` near `0.0175` for half their length, close normally
without an accepted step at `minimum_step` where the previous unconditional
slowdown exhausted the 4000-step budget.

Step-controller defaults explored on 2026-09-17 and not adopted:
`maximum_step` `0.12 -> 0.2`, `growth_factor` `1.25 -> 1.5`, and the easy-gate
tangent threshold `0.98 -> 0.95` leave every optical trace above bit-identical,
because the easy gate's `corrector iterations <= 2` condition never holds on
them (three iterations are typical), so the step never grows past
`initial_step`. History: on the analytic circle `maximum_step=0.2` closed only
on the third traversal (accumulated arclength `2.9999999999999973 x 2 pi`),
because the closure trigger measured the seed distance at the end of the
accepted edge and a step larger than `closure_distance` landed that end outside
the closure distance. Since task step-aware-closure-implementation
(2026-09-29) the distance is measured at the edge's section zero (section
6.4), and the same configuration closes on the first traversal
(`1.0000000000000002 x 2 pi`, seed distance `3e-17`,
`test_analytic_circle_with_a_growing_step_closes_on_the_first_traversal`).
The step-growth observation remains an open item in section 12.

This is bounded configuration evidence, not a proof of global controller
convergence. The accepted sample counts differ and are deliberately not a pass
criterion; in particular, the historical observation of 145 steps is not
encoded in the tests.

### 10.2 Reproducible validation record

The post-review 2026-09-16 conformance run used source commit
`8cc921c1308d24561acca06301fcf2b5505208f5`. The configured `home-wsl`
rsync target intentionally excludes `.git`, so it has no meaningful remote
`HEAD`. Before the remote test, SHA-256 was compared after synchronization for
the contract, continuation core, and conformance test; the local and remote
hashes respectively matched as `4de16d71...`, `600051e5...`, and
`f269be99...`. This content check is the remote source-version evidence rather
than a fabricated Git revision.

| Environment | Runtime | Command and result |
|---|---|---|
| Mac | macOS 14.7 arm64; uv Python 3.12.11; uv 0.8.14; JAX 0.11.1 on `CpuDevice(id=0)` | `uv run pytest -q` -> `61 passed in 93.25s` |
| `home-wsl` | Python 3.12.3; uv 0.8.14; JAX 0.11.1 on `CudaDevice(id=0)` | `XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q` -> `61 passed in 488.88s` |

The required 3-5 diagnostic run reported `closed/closed_loop`, 192 accepted
steps, maximum residual `4.070836767583417e-16`, metric length
`3.857976632802349`, and closure gap `1.054769277841672e-14` (under the
absolute closure gate of that date; the same seed now closes after 48 accepted
steps at `0.9643243178593905`, one traversal). These values are
observations, not additional pass criteria. The precision comparison measured
direction maximum absolute error `4.4773031837586075e-08` and Jacobian relative
error `1.885098415478209e-07` for its float32 probe, while the reference trace
remained float64 and rejected float32 construction explicitly.

## 11. Conformance evidence matrix

“Verified” identifies durable automated evidence for current public behavior.
“Partial” keeps the supported subset precise. “Open” is not an implementation
failure: the named prerequisite is outside the current reference core.

| ID | Fixture or counterexample | Required invariant | Status and durable evidence |
|---|---|---|---|
| C01 | Analytic `F(R) = R e3`, target `e3` | Rank two; singular values `(1, 1)` and `J_perp = 1` at the identity under the declared bases/metric. | Verified by `test_regular_state_reports_analytic_singular_values_and_tangent` and `test_analytic_circle_truth_closure_and_haar_normalization`. |
| C02 | Same analytic fiber | Closed component length converges to `2 pi`; uniform Haar pose density pushes forward to sphere density `1 / (4 pi)` because `(2 pi)/(8 pi^2) = 1/(4 pi)`. | Verified by `test_analytic_circle_truth_closure_and_haar_normalization`. |
| C03 | Analytic and synthetic 3-5 roots with several `Q in O(2)` basis changes | Root poses, tangent line, rank, singular values, `J_perp`, and converged geometry agree; tangent order may reverse only with seed orientation. | Verified by `test_analytic_circle_is_invariant_under_orthogonal_target_basis` and `test_synthetic_3_5_is_invariant_under_orthogonal_target_basis`; basis changes are metamorphic evidence, not an independent optical oracle. |
| C04 | Target antipode for the projected residual | Algebraic zero at `-d` is rejected by the target-neighborhood gate. | Verified by `test_antipode_algebraic_root_is_rejected_by_chart_gate` and the public termination conformance cases. |
| C05 | Smooth synthetic 3-5 branch | Unit outgoing direction, positive branch margins, local rank two, and one seeded component closes under independently converged settings. Exact 145 steps is not asserted. | Verified by `test_synthetic_3_5_trace_matches_independent_direct_ray_oracle` and `test_synthetic_3_5_safe_step_sweep_converges_without_fixed_step_count`. The oracle independently derives incident direction, target, tangent basis, path gate, and per-pose ray constraints rather than reading them from `path_3_5_problem`; this is not historical ch06 validation. |
| C06 | Initial step sizes and controller thresholds perturbed around reference defaults | Accepted traces converge to the same component geometry, length, quadrature-ready orientation, and terminal status within reported errors; integral comparison follows when C14 factors exist. | Partial: geometry is verified by the analytic/3-5 safe-step tests and `test_synthetic_3_5_controller_threshold_perturbation_converges_consistently` over the bounded configurations in section 10.1; first-traversal closure of loops shorter than `pi` and second-order length convergence on the analytic conjugation circle by `test_analytic_conjugation_loops_shorter_than_pi_close_on_the_first_traversal`, `test_analytic_circle_closes_on_the_first_traversal_with_a_large_step` and `test_analytic_circle_with_a_growing_step_closes_on_the_first_traversal` (the step-aware closure trigger of section 6.4, with the edge bisection unit-tested by `test_closure_crossing_pose_bisects_the_section_zero_on_the_edge`), the strip short loops by `test_strip_short_loops_close_at_their_single_traversal_length`, and the retired absolute gate's repeated traversal as a counterexample by `test_legacy_absolute_closure_gate_traverses_the_analytic_short_loops_repeatedly` and `test_legacy_absolute_closure_gate_doubles_the_strip_short_loops`; the event-approach rule of section 6.3 by `test_strip_boundary_hugging_loops_no_longer_exhaust_the_step_budget` (rows 700/780) and the `_adapt_accepted_step` unit tests. An orientation-independent physical integral is now checked on the canonical pixel fiber (`tests/test_resample_quadrature.py`): `test_canonical_pixel_integral_is_invariant_under_initial_step` (`0.03`, `0.08` against `0.04`), `..._under_tolerance` (`1e-3`, `1e-5` against `1e-4`) and `..._under_reversed_orientation` (`initial_tangent_sign = -1`) agree within the sum of the reported error estimates without comparing sample counts; `test_reversed_seed_orientation_keeps_the_integral` covers the analytic circle. A global controller-convergence claim is still not made. |
| C07 | Constructed rank-deficient map | Terminates as `event_terminated/rank_loss` with singular-value and `J_perp` diagnostics; no regular coarea value is emitted. | Verified by `test_rank_deficient_direction_map_has_typed_event_and_diagnostics` and the public termination conformance cases. |
| C08 | TIR or explicit path-domain boundary | Terminates with the typed event and signed margin before unsafe evaluation; not merely NaN or corrector failure. | Verified by `test_known_event_precedes_unsafe_direction_evaluation`, `test_3_5_tir_is_reported_before_the_unsafe_exit_square_root`, and the public termination conformance cases. Event crossing/localization remains open. |
| C09 | Corrector non-convergence and ill-conditioned linear solve without known physical event | Bounded retries end in the matching `numerical_failure` reason with trial history. | Partial: public conformance verifies bounded corrector non-convergence and rejected-trial history. Rank/conditioning rejection is covered by C07; a deterministic public `linear_solve_failure` fixture remains open. |
| C10 | Non-finite input/evaluator output | Rejects or terminates deterministically with source and reason; never returns apparent closure. | Verified by constructor regressions and `test_public_nonfinite_evaluator_output_never_appears_closed`. |
| C11 | Too-small step and short step/arclength/evaluation budgets | Distinguishes `step_underflow` from each `budget_exhausted` reason and retains partial diagnostics. | Verified by `test_public_corrector_nonconvergence_and_step_underflow_are_distinct` and `test_public_budget_termination_retains_partial_geometry`. |
| C12 | Near self-approach or incompatible-tangent return | Does not close unless distance, section crossing, tangent, minimum extent, and final correction all pass. | Verified at the closure boundary by `test_incompatible_tangent_cannot_pass_final_closure_correction`; discovery of remote self-intersections remains open. |
| C13 | Quaternion `q` versus `-q` storage | Represents the same samples and produces zero pose distance, identical closure, length, and integral diagnostics. | Open until a quaternion storage adapter exists; the reference result currently declares rotation-matrix storage. Cosmetic open item: no result depends on it, because every production path stores and compares rotation matrices, and the continuous-sign quaternions of `resample.fiber_spline` are derived from them. |
| C14 | Named factor audit | Every requested factor has value/unit/normalization/availability; the coarea denominator and Haar conversion remain separate. | Partial: values, units, and normalization are exposed for `rho_pose`, `entry_measure`, `fresnel_transmission`, and `path_validity` on the canonical pixel fiber (`test_canonical_pixel_fiber_exposes_four_available_factors_pointwise`, `test_figure_data_exports_available_weight_arrays_for_the_canonical_pixel`); `test_public_result_schema_preserves_units_shapes_dtype_and_availability` verifies unregistered factors stay unavailable rather than silently one, and `conventions` carries `1/(8 pi^2)` and `J_perp` separately. Quadrature (`lumice_integral.quadrature`): an adaptive composite Simpson line integral over corrector-retracted, chord-parametrised edges with the exact `dH^1_g` speed reports method, refinements, node count, `value`, `error_estimate`, `epsilon` (`J_perp -> J_perp + epsilon`, default `1e-6`, the raw `J_perp` array stays separate) and a convergence-order estimate; `test_constant_weight_recovers_the_haar_identity_on_the_circle` and `test_trigonometric_weight_matches_the_analytic_integral_on_every_grid` verify analytic values, and `test_default_options_align_with_the_adaptive_reference_within_1e_4` / `test_node_count_doubles_until_the_estimate_meets_the_tolerance` / `test_hitting_maximum_node_count_is_reported_not_passed_off_as_converged` verify the converged `partial` canonical value and its non-silent failure (`tests/test_resample_quadrature.py`; the adaptive method and its test names were retired by task-resample-and-integrate) (`docs/ch06-reference-fixture.md` section 4.1), and `test_figure_data_exports_the_quadrature_block_and_pointwise_integrand` the exported block. Component completeness stays `unknown`, so the value remains partial. `visibility` (finite-face obstruction) is contained in `entry_measure` for the convex crystal: the corridor intersection admits only entry points whose internal segment reaches every next face's finite polygon, and a convex body obstructs no incoming or outgoing ray. It stays an unregistered name rather than a separate factor. `source_factor` / `pixel_factor` are not evaluated in code. Their conversion to Lumice's `raw / emitted_energy` is derived and numerically checked: `raw[p] / E = K_p V(w_p)`, `K_p = N_sym ybar(550) Omega_p / (S/2)`, with `S` the crystal's total surface area, because Lumice (since `6fc48bb4`, Ice Halo #597) weighs every ray by its projected area over `S/2` at entry. The bright band of columns `106 / 126 / 146` agrees within `0.3 %` at matched refractive index, and the plate and Parry families' total flux within `0.01 %` (`scripts/probe_absolute_scale.py`, `scripts/compare_lumice_family.py`, `docs/ch06-reference-fixture.md` section 7, stage 4). Against Lumice before `6fc48bb4` the denominator was the pixel-dependent `A_eff(w_p)`, the fiber-weighted harmonic mean of the projected silhouette (same section, history bullet). |
| C15 | Discovery, single closed component: canonical 3-5 pixels, including the caustic short loops | One closed component, the other admissible candidates folded by distance, `complete`; the funnel counts and loop lengths are pinned on the production sample. | Verified by `test_canonical_pixel_has_one_closed_component`, `test_rows_225_and_226_are_one_continuous_branch` and `test_caustic_edge_pixels_are_single_short_closed_loops` (`tests/test_discovery.py`). |
| C16 | Discovery near a domain boundary: pixels whose loop runs along the exit TIR boundary (`exit_snell_discriminant` near `event_slowdown_margin`) | Every candidate folds into one closed loop, with no budget-exhausted candidate. | Verified by `test_boundary_hugging_pixels_fold_every_candidate_into_one_closed_loop` (rows 700/780). |
| C17 | Discovery, several components: path `1-3` on the canonical column at `delta = 60` deg, `N = 1e5` | Two distinct components (`0.82` rad apart, far above `eta` and `r_c`), each traced once, one candidate folded, `complete`, and no band event off them (`check_band_coverage`). | Verified by `test_path_1_3_at_60_deg_is_two_distinct_components`. |
| C18 | Discovery, TIR-truncated open arc on a real path (the same fixture) and every classification branch of section 9.5.5 | Each `1-3` component is an arc cut by exit TIR (margin within the Snell event tolerance) at one end and by the entry ray leaving face 1 (`path_infeasible`) at the other, with valid accepted poses. On the analytic capped circle, a backward trace that fails, closes or meets an unnamed event, and a forward budget exhaustion, each leave the candidate incomplete with its cause. | Verified by `test_path_1_3_components_are_arcs_cut_by_tir_and_path_infeasibility`, `test_two_named_events_stitch_into_an_arc_component`, `test_backward_trace_that_does_not_end_on_a_named_event_leaves_the_candidate_incomplete`, `test_backward_trace_that_closes_is_an_anomaly_not_a_component`, `test_unnamed_event_is_incomplete_and_not_traced_backward`, `test_budget_exhausted_forward_trace_is_incomplete_not_converged` and `test_a_reversed_caller_orientation_still_traces_the_other_way_for_the_arc`. The one-pose arc of section 9.5.6 is pinned as a known limitation by `test_a_seed_within_one_initial_step_of_two_events_is_a_single_pose_arc` (path `3-1`, 64.7434 deg). |
| C19 | Discovery outside the canonical path and crystal: a class member served by symmetry transport, a member with no lit event, a prism-face path on the `D3h` prism, and a pyramid | `3-7` through the transported `3-5` sample and `3-1-2-5` (empty sample, zero components) run the same pipeline; on the `D3h` prism the crystal reaches the entry-measure gate and the component equals the regular prism's (section 9.5.10); a pyramid is refused before discovery. | Verified by `test_discovery_runs_on_another_member_of_the_class`, `test_discovery_on_a_low_symmetry_prism` and `tests/test_s2_store.py::test_pyramid_is_refused_with_a_forward_pointer_to_its_own_task`. Discovery on pyramids is open (section 12). |
| C20 | Pool, clustering, warm seeds, dedup threshold and the funnel counters | Lowest-index cluster centres and non-transitive membership; a warm seed is a Gauss-Newton start, not an extra trace, and a warm seed far from the fiber adds nothing; a non-positive `eta` is rejected; the funnel identities of section 9.5.6 hold on closed, arc, starved and warm results; one continuation policy governs every trace. | Verified by `test_geodesic_cluster_centres_are_the_lowest_unassigned_index_and_membership_is_not_transitive`, `test_geodesic_cluster_separates_two_tight_clusters`, `test_warm_seed_from_the_row_above_is_a_newton_start_not_a_separate_trace`, `test_warm_seed_far_from_every_fiber_neither_poisons_nor_adds_a_component`, `test_dedup_threshold_must_be_positive`, `test_discovery_funnel_identities_hold_on_closed_arc_starved_and_warm_results` and `test_continuation_options_are_the_single_trace_policy`. |
| C21 | Procedural completeness and densification | A target with no admissible candidate is `complete` with zero components; an incomplete candidate makes it `unknown`; the band cross-check finds no suspect on a complete pixel and every event of a removed component as a suspect. Under nested densification every component with an accepted step is kept on the measured `1-3` fibers, and the `D3h` `5-3` counterexample loses a one-pose arc. | Partial, by design: the procedural semantics are verified by `test_dark_pixel_has_no_admissible_candidate_and_is_procedurally_complete`, `test_continuation_options_are_the_single_trace_policy`, `test_band_coverage_of_a_complete_pixel_has_no_suspect`, `test_band_coverage_reports_a_component_discovery_did_not_return` and `test_miss_probability_is_the_poisson_void_probability`; densification by `test_random_sampler_stores_are_nested_prefixes`, `test_nested_densification_keeps_every_traced_component` and `test_nested_densification_can_lose_a_single_pose_arc`. A completeness certificate and monotone densification are open (section 12). |

## 12. Explicit open items

- Reference defaults have the bounded evidence in section 10.1. Broader
  controller sweeps and event localization tolerances remain open; the current
  solver terminates honestly at supported event boundaries rather than crossing
  or localizing them.
- Step growth never triggers on the optical fixtures because the easy gate
  requires at most two corrector iterations; the controller runs them at
  `initial_step`, and the defaults (`maximum_step`, the easy gate) are
  unchanged. The closure trigger is no longer what couples growth to closure:
  it is step-aware since 2026-09-29 (section 6.4; before, a step larger than
  `closure_distance` could carry the post-crossing pose outside the closure
  distance and skip the attempt, and the analytic circle with
  `maximum_step=0.2` closed on its third traversal). Enabling growth on
  optical fibers still waits on two unmeasured points of that trigger and on
  one of the domain:
  - the bisection assumes at most one section sign change per accepted edge;
    it holds for steps well below the loop's half period (`maximum_step=0.12`
    and the analytic sweeps up to `0.8`) and MUST be re-checked if the step
    bound approaches that scale;
  - the edge geodesic is a chord of the fiber, exact on the analytic circle
    (one-parameter subgroup); its deviation from the true trajectory at the
    section has not been measured on optical fibers, where the final closure
    corrector absorbs it;
  - whether periodic or bounded narrow domain features narrower than a grown
    step occur on optical fibers (they would be stepped over, unlike the
    monotone TIR, branch and visibility half-spaces) is the open explore
    thin-domain-feature-skip-risk-on-optical-fibers.
- A deterministic consumer-level fixture for `linear_solve_failure` remains a
  conformance-infrastructure gap. Corrector non-convergence and rank/condition
  rejection are covered without private monkeypatching.
- Discovery (section 9.5) has a normative, procedural contract; its open
  parts are:
  - no completeness certificate, and none exists in unconditional form: the
    path `3-5` counterexample (a one-pose arc deduplicated into a traced
    neighbour `1.224e-3 rad` away) rules out a crystal- and path-independent
    certificate and any threshold rule built from the certificate's
    macroscopic data. `completeness` stays procedural and the band-coverage
    `exp(-k_min)` is a Monte Carlo reading, not a bound (section 9.5.6). The
    optional post-hoc check of section 9.5.6a compares the component count
    with `DPField.interval_partition()`: a match is sufficient evidence, a
    mismatch marks the discovery result undersampled with the `dp_field`
    count as the topological truth, and it is unavailable when
    `halo_map_rank == 0` (`ValueError`) or outside the partition's structural
    preconditions (`TopologyEscape`);
  - densification is not monotone: one-pose arcs can be lost between nested
    samples; components with an accepted step have been kept on every fiber
    measured (section 9.5.7);
  - arc extent depends on the seed, and a seed within one `initial_step` of
    events on both sides gives a one-pose arc (section 9.5.6). Resolving this
    means a first-step retry before an event is declared, which changes arc
    lengths and counts and needs its own evidence; it is not scheduled;
  - the lowest-index cluster centre of the reference rests on the iteration
    order of a Python `set` of small integers (pinned by a test, section
    9.5.4); an explicit `min` is the behaviour-preserving spelling;
  - `d = ±s` (deviation `0` or `pi`) is not guarded in the reference
    (section 9.5.3);
  - pyramid crystals wait for task `pyramid-lumice-semantics` (section 9.5.10).
  History: until 2026-09-25 the pool came from a Haar prescan table indexed by
  outgoing direction (4M samples, 2 deg cone); task `phase1-seeds-from-store`
  replaced it after a 32-pixel probe found the same components. The retired
  small discovery budget, production retrace, arclength-fingerprint dedup,
  arclength-jump gate and periodic cold check of the strip driver no longer
  exist; every pixel always queries the sample, so a warm seed cannot hide a
  component. No open arc exists in the ch06 `3-5` picture (task-pixel-pipeline-v2
  Step 0: every 10th row and column, 2106 pixels, 1954 lit, all closed); real
  arcs are those of section 11 C17 (path `1-3`), and the analytic two-sided
  circle fixture still covers the classification branches that no optical
  fiber reaches (`tests/test_discovery.py`, `tests/test_resample_quadrature.py`).
- Continuation through rank loss, bifurcation, singular intersections, TIR, or
  path-branch changes is unsupported pending dedicated exploration.
- Absolute source radiometry, wavelength/polarization integration, pixel solid
  angle/filtering, finite-crystal entry measure, visibility, and all complete
  weight implementations remain separate contracts/tasks.
- Historical ch06 coordinates, projection, normalization, dynamic range, and
  data provenance require an explicit adapter after their reconstruction.
- Phase II must derive its own measure conversion and demonstrate agreement;
  this document does not assume that reduction in the Phase I algorithm.

## 13. Contract audit and acceptance crosswalk

This crosswalk records the scope audit performed for the initial contract. A
checked row means the semantic requirement is present and internally assigned;
it does not claim that a pending solver or conformance test already exists.

| Acceptance area | Contract location | Audit result |
|---|---|---|
| Pose action, frames, composition, propagation signs, units, and local `SO(3)` coordinates | Sections 2–3 | Defined; evaluator sign conversions must be explicit. |
| `F_P`, smooth domain, local two-dimensional residual, antipode exclusion, and chart/basis invariance | Sections 5.2–5.4 | Defined; general charts carry their metric correction. |
| `2 x 3` Jacobian, tangent orientation, predictor-corrector, adaptive signals, and closure | Sections 5.4 and 6 | Defined for one regular seeded component only. |
| Coarea measure, normal Jacobian, Haar conversion, and named physical factors | Sections 5.1 and 7 | Defined; absolute radiometry and unimplemented factors remain explicit. |
| Rank loss, critical points, TIR, branch/path/visibility boundaries, self-approach, and multiple components | Sections 8 and 9.5, section 12 | Observable termination/open semantics defined; unsupported crossings are not claimed. |
| Backend-independent problem/options/result and four terminal statuses | Section 9 | Required fields, diagnostics, availability, and causal status precedence defined. |
| Seed search and component discovery: sampling measure, candidates, clustering, gates, dedup, classification, output, v0 subset | Section 9.5 | Defined as the procedural `reference-discovery-v1` contract; completeness is not certified and densification is not monotone (section 12). |
| Mathematical truth versus tolerances/default strategies | Section 10 | Separated; default numerical values await conformance evidence. |
| Analytic circle, synthetic 3-5, basis changes, and failure counterexamples | Section 11 | C01–C14 assigned to current or downstream evidence owners; C15–C21 cover discovery. |
| float64, zero-limit AD, and stable rotation distance | Section 3 | Incorporated as reference numerical requirements, not universal mathematical constants. |
| Roadmap linkage, Phase II boundary, Lumice independence, and downstream backfill | Sections 1, 4, and 12; [`roadmap.md`](roadmap.md) | Single detailed authority retained here; open ownership is explicit. |

Initial self-audit conclusion: every issue acceptance area maps to a normative
section or a named open item. No solver implementation, historical fixture
normalization, complete-component claim, exact numerical result, or Lumice
runtime dependency is introduced by this contract.
