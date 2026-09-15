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

Downstream ownership is explicit:

| Item | Owner | Contract status before owner completes |
|---|---|---|
| Structured adaptive single-component solver and event gate | `reference-continuation-core` | Semantics fixed here; implementation pending. |
| Default tolerances and their convergence evidence | `reference-core-conformance` | Open numerical values. |
| Basis-invariance, failure, event, and step-size perturbation tests | `reference-core-conformance` | Required matrix rows; evidence pending. |
| Seed discovery and completeness over all components | Future task/exploration | Open and outside this scrum. |
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
| `rho_pose` | Density relative to explicitly named `d mu_Haar` or `dVol_g`; normalization state reported. | Dimensionless for Haar probability; inverse metric-volume for `dVol_g`. | Interface required; model supplied by caller. |
| `entry_measure` | Projected/realizable entry measure for the selected path, with sign/clamping and reference area declared. | Dimensionless if area-normalized; otherwise area. | Interface required; not implemented by current probe. |
| `visibility` | Obstruction/visibility result and, if soft, its declared transmittance model. | Dimensionless. | Interface/event required; not implemented. |
| `fresnel_transmission` | Product of polarization-resolved or explicitly unpolarized interface power factors; convention declared. | Dimensionless. | Interface required; current Snell probe omits it. |
| `path_validity` | Boolean feasibility of the exact ordered path, separate from numerical convergence. | Boolean gate, not a gain. | Event interface required. |
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

