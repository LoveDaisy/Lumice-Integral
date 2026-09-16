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
- backend-independent problem, options, result, event, and termination data.

It does not provide seed search, prove that every component was found, cross a
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
| [`roadmap.md`](roadmap.md) | Project staging, responsibility boundaries, and validation strategy. | Context; links here for detailed Phase I semantics. |
| [`0001-phase-i-python-jax.md`](decisions/0001-phase-i-python-jax.md) | Accepted reference stack, dtype, and execution boundary. | Normative for architecture; numerical observations are evidence. |
| `scratchpad/explore-ad-continuation-stack/SUMMARY.md` | AD, closure-distance, precision, and batching experiments. | Evidence only. |
| `src/lumice_integral/{analytic,so3,continuation,optics}.py` | Current analytic and synthetic probes. | Evidence/current strategy only. |
| `tests/test_analytic_fiber.py`, `tests/test_optics.py` | Existing positive regression fixtures. | Evidence only; exact fixed-step counts are not contract terms. |
| `tests/test_reference_core_conformance.py` | Consumer-facing schema, invariant, convergence, and termination checks mapped to C01–C14. | Durable conformance evidence for the currently supported reference core. |

Downstream ownership is explicit:

| Item | Owner | Contract status before owner completes |
|---|---|---|
| Structured adaptive single-component solver and event gate | `reference-continuation-core` | Semantics fixed here; implementation pending. |
| Default tolerances and their convergence evidence | `reference-core-conformance` | Open numerical values. |
| Basis-invariance, failure, event, and step-size perturbation tests | `reference-core-conformance` | Required matrix rows; evidence pending. |
| Single-pixel seed/component discovery for the 3-5 path | `strip-component-discovery` (`lumice_integral.discovery`) | Implemented for one pixel's target direction (section 12); strip-level neighbour continuation and full-image completeness remain open (`strip-image-driver`). |
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
`X_(P,d)` was found. Global component discovery and deduplication MUST be
reported separately by a future caller.

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
conditioning, tangent continuity, and event clearance. Options MUST bound step
size above and below and bound retries, corrector iterations, accepted steps,
and/or arclength.

No particular controller or threshold is a mathematical invariant. The result
MUST expose enough accepted and rejected step diagnostics for conformance tests
to explain why the controller changed `h`.

### 6.4 Closure and self-intersection

A trace is `closed` only when all of the following hold:

1. a configured minimum accepted-step count and/or minimum accumulated
   arclength excludes immediate return to the seed;
2. stable `SO(3)` distance to the seed is within the closure tolerance;
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
components, proving seed coverage, and deduplicating components are also open
and external to a single `FiberResult`.

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
| `component_scope` | States that this is one component reached from one seed; component completeness is `unknown` unless established externally. |
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
| Seed orientation | `initial_tangent_sign=+1` (section 5.4 explicit choice; `-1` reverses the deterministic SVD sign of the seed tangent and hence the sample order) |
| Work bounds | `maximum_accepted_steps=4000`, `maximum_evaluations=100000`, `maximum_arclength=20` |
| Closure | minimum 40 steps and `pi` arclength, distance `0.08`, tangent dot `0.8`, section tolerance `1e-11`, at most 10 final-corrector iterations |

On the analytic circle, initial steps `0.04`, `0.08`, and `0.12` all terminate
as `closed/closed_loop`, have maximum residual no greater than `1e-11`, and
recover length `2 pi` within `1e-12`; the test separately verifies the
Haar-to-sphere identity `(2 pi)/(8 pi^2) = 1/(4 pi)`. On the synthetic 3-5
branch, initial steps `0.03`, `0.04`, and `0.08` all close with maximum residual
below `1e-11` and closure gap below `2e-13`. Their discrete metric lengths lie
between `3.857976632802349` and `3.859621360919519`, a span below `0.0017`.
An additional 3-5 run holds `initial_step=0.04` fixed while changing
`minimum_step` to `2e-5`, `maximum_step` to `0.10`, `shrink_factor` to `0.4`,
`growth_factor` to `1.15`, and `maximum_retries` to `10`. It also closes with
the declared residual and closure bounds; its length differs from the reference
by less than `0.0015`, and the bidirectional sampled-pose set distance is below
`0.008` rad. This exercises controller thresholds in addition to initial-step
selection.
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
`3.857976632802349`, and closure gap `1.054769277841672e-14`. These values are
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
| C06 | Initial step sizes and controller thresholds perturbed around reference defaults | Accepted traces converge to the same component geometry, length, quadrature-ready orientation, and terminal status within reported errors; integral comparison follows when C14 factors exist. | Partial: geometry is verified by the analytic/3-5 safe-step tests and `test_synthetic_3_5_controller_threshold_perturbation_converges_consistently` over the bounded configurations in section 10.1. An orientation-independent physical integral is now checked on the canonical pixel fiber: `test_canonical_pixel_integral_is_invariant_under_initial_step` (`0.03`, `0.08` against `0.04`), `..._under_refinement_tolerance` (`1e-6`, `1e-9` against `1e-8`) and `..._under_reversed_orientation` (`initial_tangent_sign = -1`) agree within the sum of the reported error estimates without comparing sample counts; `test_reversed_seed_orientation_keeps_arclength_and_integral` covers the analytic circle. A global controller-convergence claim is still not made. |
| C07 | Constructed rank-deficient map | Terminates as `event_terminated/rank_loss` with singular-value and `J_perp` diagnostics; no regular coarea value is emitted. | Verified by `test_rank_deficient_direction_map_has_typed_event_and_diagnostics` and the public termination conformance cases. |
| C08 | TIR or explicit path-domain boundary | Terminates with the typed event and signed margin before unsafe evaluation; not merely NaN or corrector failure. | Verified by `test_known_event_precedes_unsafe_direction_evaluation`, `test_3_5_tir_is_reported_before_the_unsafe_exit_square_root`, and the public termination conformance cases. Event crossing/localization remains open. |
| C09 | Corrector non-convergence and ill-conditioned linear solve without known physical event | Bounded retries end in the matching `numerical_failure` reason with trial history. | Partial: public conformance verifies bounded corrector non-convergence and rejected-trial history. Rank/conditioning rejection is covered by C07; a deterministic public `linear_solve_failure` fixture remains open. |
| C10 | Non-finite input/evaluator output | Rejects or terminates deterministically with source and reason; never returns apparent closure. | Verified by constructor regressions and `test_public_nonfinite_evaluator_output_never_appears_closed`. |
| C11 | Too-small step and short step/arclength/evaluation budgets | Distinguishes `step_underflow` from each `budget_exhausted` reason and retains partial diagnostics. | Verified by `test_public_corrector_nonconvergence_and_step_underflow_are_distinct` and `test_public_budget_termination_retains_partial_geometry`. |
| C12 | Near self-approach or incompatible-tangent return | Does not close unless distance, section crossing, tangent, minimum extent, and final correction all pass. | Verified at the closure boundary by `test_incompatible_tangent_cannot_pass_final_closure_correction`; discovery of remote self-intersections remains open. |
| C13 | Quaternion `q` versus `-q` storage | Represents the same samples and produces zero pose distance, identical closure, length, and integral diagnostics. | Open until a quaternion storage adapter exists; the reference result currently declares rotation-matrix storage. |
| C14 | Named factor audit | Every requested factor has value/unit/normalization/availability; the coarea denominator and Haar conversion remain separate. | Partial: values, units, and normalization are exposed for `rho_pose`, `entry_measure`, `fresnel_transmission`, and `path_validity` on the canonical pixel fiber (`test_canonical_pixel_fiber_exposes_four_available_factors_pointwise`, `test_figure_data_exports_available_weight_arrays_for_the_canonical_pixel`); `test_public_result_schema_preserves_units_shapes_dtype_and_availability` verifies unregistered factors stay unavailable rather than silently one, and `conventions` carries `1/(8 pi^2)` and `J_perp` separately. Quadrature (`lumice_integral.quadrature`): an adaptive composite Simpson line integral over corrector-retracted, chord-parametrised edges with the exact `dH^1_g` speed reports method, refinements, node count, `value`, `error_estimate`, `epsilon` (`J_perp -> J_perp + epsilon`, default `1e-6`, the raw `J_perp` array stays separate) and a convergence-order estimate; `test_constant_weight_recovers_the_haar_identity_within_epsilon_and_error` and `test_trigonometric_weight_matches_the_analytic_integral_within_error` verify analytic values, `test_canonical_pixel_quadrature_converges_without_silent_degradation` the converged `partial` canonical value (`docs/ch06-reference-fixture.md` section 4.1), and `test_figure_data_exports_the_quadrature_block_and_pointwise_integrand` the exported block. Component completeness stays `unknown`; `visibility` and the radiometric factors stay unavailable, so the value remains partial. |

## 12. Explicit open items

- Reference defaults have the bounded evidence in section 10.1. Broader
  controller sweeps and event localization tolerances remain open; the current
  solver terminates honestly at supported event boundaries rather than crossing
  or localizing them.
- A deterministic consumer-level fixture for `linear_solve_failure` remains a
  conformance-infrastructure gap. Corrector non-convergence and rank/condition
  rejection are covered without private monkeypatching.
- Seed search, component discovery, completeness certificates, and component
  deduplication are outside the single-component interface.
  `lumice_integral.discovery` provides them for one pixel of the 3-5 path as
  a separate module with its own, weaker contract:
  - `discover_components(target_direction, incident_direction,
    refractive_index, crystal, *, rng_seed, prescan_samples=400000,
    discovery_step_budget=250, angle_tolerance_deg=2.0,
    cluster_radius_rad=0.3, arclength_rtol=1e-3)` returns
    `ComponentDiscoveryResult(components, incomplete, completeness,
    pool_count, raw_cluster_count, admissible_count)`.  It Haar-samples
    `SO(3)`, keeps the pool within the angular tolerance that passes both
    refraction discriminants, clusters the *whole* pool geodesically, Gauss-
    Newton-corrects one representative per cluster, applies the
    `path_3_5_domain` and `entry_measure > 0` gates, and traces each
    admissible candidate with `trace_fiber` under
    `maximum_accepted_steps = discovery_step_budget`.  `rng_seed` is
    required; batch callers decide explicitly whether pixels share it.
  - Two closed traces are the same component iff `(status, reason)` agree and
    their arclengths agree within `arclength_rtol = 1e-3` (`atol = 1e-6`).
    Accepted pose counts are not part of the fingerprint: the survey observed
    one loop traced with `173 / 180 / 172` poses from different entry points.
    Traces with `status != closed` never form or join a component; they are
    returned as `incomplete` with their truncated `FiberResult`.
  - `completeness` is procedural, not a certificate: `"complete"` means every
    admissible candidate of this pool closed and no `status != closed`
    evidence was seen (a dark pixel with no admissible candidate is
    `"complete"` with zero components); `"unknown"` means at least one
    candidate did not close.  It does not prove that every connected
    component of `X_(P,d)` was found, so the single-component result's
    `component_completeness = "unknown"` stays authoritative for quadrature.
  - The discovery budget is independent of the production
    `ContinuationOptions` default (`4000`); the caller retraces a discovered
    seed with production options for quadrature.
  - `hot_start_component(converged_seed, target_direction, ...)` runs the
    same correction, gates, trace, and classification on one caller-supplied
    seed (no prescan), and `detect_arclength_jump(arclengths,
    relative_threshold=0.2)` flags neighbouring-pixel arclength jumps as
    topology-boundary evidence; the `0.2` default is calibrated on the single
    observed boundary (canonical strip rows `225 -> 226`, about `50 %`).
  - Defaults and regression baselines come from
    `scratchpad/scrum-ch06-direct-integration/explore-component-discovery`
    (400k samples stable to 1.6M, 0.3 rad cluster radius, 34+ pixels) and are
    locked by `tests/test_discovery.py`.
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
| Rank loss, critical points, TIR, branch/path/visibility boundaries, self-approach, and multiple components | Section 8 and section 12 | Observable termination/open semantics defined; unsupported crossings are not claimed. |
| Backend-independent problem/options/result and four terminal statuses | Section 9 | Required fields, diagnostics, availability, and causal status precedence defined. |
| Mathematical truth versus tolerances/default strategies | Section 10 | Separated; default numerical values await conformance evidence. |
| Analytic circle, synthetic 3-5, basis changes, and failure counterexamples | Section 11 | C01–C14 assigned to current or downstream evidence owners. |
| float64, zero-limit AD, and stable rotation distance | Section 3 | Incorporated as reference numerical requirements, not universal mathematical constants. |
| Roadmap linkage, Phase II boundary, Lumice independence, and downstream backfill | Sections 1, 4, and 12; [`roadmap.md`](roadmap.md) | Single detailed authority retained here; open ownership is explicit. |

Initial self-audit conclusion: every issue acceptance area maps to a normative
section or a named open item. No solver implementation, historical fixture
normalization, complete-component claim, exact numerical result, or Lumice
runtime dependency is introduced by this contract.
