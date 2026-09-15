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

