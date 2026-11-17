# Pose-family → sky differential

`lumice_integral.family_sky` measures the first differential of a caller-defined
pose family into the sky sphere. It is a geometric-optical instrument, not a
renderer or a caustic classifier. It complements `focusing`'s $u$-sphere
critical-set and pose-density analysis without changing its classifications.

## Inputs and conventions

```python
from lumice_integral.family_sky import sky_differential, diagnose_sky

raw = sky_differential(
    rotations, tangent_generators, faces, sun_direction,
    crystal=None, refractive_index=1.31,
)
diagnostic = diagnose_sky(raw, atol=1e-12, rtol=1e-10, generator_rtol=1e-12)
```

- `rotations`: explicit `(N,3,3)` active body→world matrices, in SO(3).
- `tangent_generators`: explicit `(N,K,3)`, $K\geq1$. These are **body angular
  velocities**, not unit axes or skew matrices. Each parameter $a$ defines
  $\partial_aR=R[\Omega_a]_\times$; at the point the differential is evaluated
  along $R\exp(t[\Omega_a]_\times)$. The finite-distance family need not be
  that one-parameter subgroup. Magnitudes are preserved, not normalized.
- `faces`: integer sequence `(entry, *internal_reflections, exit)` present on
  the supplied crystal; default crystal is the canonical hexagonal prism.
  Actual crystal face normals and the supplied refractive index are used.
- `sun_direction`: a unit world vector $\hat s$ **toward** the sun, not ray
  propagation. `camera.incident_direction_from_sun` converts it once to
  $-\hat s$. The sky map is
  $x(R)=-\operatorname{trace\_path}(R,\mathrm{normals},-\hat s,n).\mathrm{direction}$.

All numeric inputs must be real finite. The host API converts to float64,
requires rotation orthogonality and determinant residuals ≤`1e-10` and sun
squared-norm residual ≤`1e-12`; it never repairs inputs. The refractive index
must be a positive finite scalar. No broadcasting is implicit. Empty batches
are allowed. The reference package enables JAX float64; this validation and
valid-subset selection API is not itself a JAX-transformable function.

Parameters are in **radians**: `dx` has units sky radians per parameter radian,
numerically also degrees/degrees. A generator per degree is not silently
converted to a generator per radian. For a general differentiable curve,
compute $\Omega_a=\operatorname{vee}(R^T\partial_a R)$; for a world left angular
velocity $w_a$, use $\Omega_a=R^Tw_a$. For example:

```python
import numpy as np
from lumice_integral.camera import sun_direction
from lumice_integral.family_sky import sky_differential, diagnose_sky

# One caller-supplied pose, and two independent world angular velocities.
R = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
world_axes = np.array([[0., 0., 1.], [1., 0., 0.]])
body_omega = world_axes @ R               # row form of R.T @ w
raw = sky_differential(R[None], body_omega[None], (3, 5), sun_direction(15))
d = diagnose_sky(raw)
assert raw.available[0]
assert d.mechanism[0] == "parallel_collapse"
# d.signed_area[0] is roundoff-scale, not snapped to literal zero.
```

Family parameterizations (azimuth/zenith/roll chains, column, Lowitz, Parry,
plate) belong to callers, not to a family registry in this API. This same
right-trivialized $\Omega$ contract is the reuse point for later family
restrictions on the $u$ sphere; a new left/right convention is unnecessary.

## Domain and failure semantics

`optics.path_domain_batch` is the sole domain authority. Entry/exit incidence
and Snell gates, and internal-face incidence gates, must be positive. An
internal reflection retains its reflected branch even when partial; its TIR
discriminant is evidence, **not** a gate. The returned `margins` map retains
all original names and values (including non-finite evidence outside the
branch). No finite-crystal corridor, nonzero entry area, or power is certified.

Only valid rows enter AD on `so3.exp`'s zero-safe local chart and the existing
`optics.trace_path` graph. No independent production Snell implementation,
Lumice dependency, or change to NumPy `symmetry.signature` is introduced.
Normals and refractive index are numerical JIT arguments, not crystal-object
identity closures. Compilation can still specialize on batch/path shapes.

`SkyDifferential` returns:

| Field | Shape | Meaning |
|---|---|---|
| `sky` | `(N,3)` | Unit sky point $x$ where available |
| `dx` | `(N,K,3)` | $\partial_a x$ per supplied generator |
| `tangent_generators` | `(N,K,3)` | Copy of the caller's body velocities |
| `valid` | `(N,)` bool | Optical open-domain membership |
| `available` | `(N,)` bool | Valid **and** finite sky/AD result |
| `margins` | name → `(N,)` | Original optical domain evidence |

Off domain, `valid=False`, `available=False`, `sky/dx=NaN`. Non-finite AD on
an optically valid row instead retains `valid=True`, sets `available=False`
and `sky/dx=NaN`. A failed row cannot contaminate another row's SVD/eigensolve.
There is no across-event differencing or square-root clamp. Close to an exit
critical angle derivatives may be very large: finiteness is not a promise of
uniform relative accuracy. Strong FD/roundoff assertions use positive-margin
interiors; a separate near-critical fixture tests failure semantics.

## Parameter-space diagnostics, not a 3×3 surrogate

Let $A=(x_a,x_b)$ be the **3×2** differential matrix. The output stores its
columns as `dx[n, k, :]`. The Gram matrix $G=A^TA$ is 2×2. A 3×3 scatter or
symmetrized outer-product matrix always has a zero eigenvalue for two input
directions and does not detect the rank loss of interest.

`diagnose_sky` accepts $K=1,2$ only (raw differentials accept larger $K$):

| Field | Shape | Definition |
|---|---|---|
| `speeds` | `(N,K)` | $\|x_i\|$ |
| `gram` | `(N,K,K)` | $A^TA$ |
| `gram_eigenvalues` | `(N,K)` | Ascending symmetric eigenvalues of $G$ |
| `singular_values` | `(N,K)` | Descending singular values of $A$, **direct SVD**, not $\sqrt{\lambda(G)}$ |
| `null_direction` | `(N,K)` | Least right singular vector, unit, sign arbitrary |
| `signed_area` | `(N,)` or `None` | $T=x\cdot(x_a\times x_b)$ for $K=2$; absent for $K=1$ |
| `rank` | `(N,)` int | Number of $\sigma_i>\mathrm{atol}+\mathrm{rtol}\,\sigma_{\max}$ |
| `generator_rank` | `(N,)` int | Rank of independently length-normalized nonzero $\Omega_i$ |
| `rank_threshold` | `(N,)` | The derivative-unit threshold actually used |
| `mechanism` | `(N,)` string | Coordinate-dependent first-order label below |

The least singular vector is a null line **only** at rank deficiency; at rank
zero the whole parameter space is null and this vector is not a unique line.
Compare lines through absolute dot products or projectors, not LAPACK signs.

Default `atol=1e-12`, `rtol=1e-10` distinguish float64 symmetric-point residuals
($\sigma_{min}<7\cdot10^{-17}$ in the four-height fixture) from ordinary
valleys ($\sigma_{min}>0.0087$ in this sample). They are numerical diagnostics,
not historical-grid or finite-density physics thresholds. `atol` is in
derivative units; strong parameter rescaling may change numerical rank. For
input tangent rank only, each nonzero generator is normalized separately,
with dimensionless `generator_rtol=1e-12` times its largest singular value;
parameter-unit disparities alone do not create false linear dependence. Zero
or dependent input directions are explicitly degenerate. All tolerances are
finite nonnegative scalars and may be overridden; none changes `dx` or `T`.

| Condition (nondegenerate input directions) | `mechanism` |
|---|---|
| $K=1$, rank 1 / 0 | `passing` / `stationary` |
| $K=2$, rank 2 | `regular` |
| rank 1, at least one speed ≤ rank threshold | `axis_stationary` |
| rank 1, both speeds > rank threshold | `parallel_collapse` |
| rank 0 | `rank_zero` |

Deficient input tangent rank overrides these with `degenerate_generators`:
one cannot attribute such a loss to the optical map. Off-domain rows have
`invalid`; valid but unusable differential or overflowing quadratic diagnostics
have `unavailable`. Their continuous diagnostics are NaN and ranks are -1.
The latter can occur even if raw `available=True` when finite, enormous `dx`
would overflow $G$.

Exact rank is invariant under invertible reparameterization, but the
axis-stationary/parallel distinction is **coordinate-dependent**. Swapping the
two parameter directions reverses $T$; scaling them by $c_a,c_b$ multiplies
$T$ by $c_ac_b$ (Gram spectra need not be invariant).

## Why one scalar is enough — and when it is not

For a smooth **unit-sphere-valued** map, $x\cdot x=1$ implies
$x\cdot x_a=x\cdot x_b=0$. Both derivatives lie in the same two-dimensional
tangent plane, hence their cross product is parallel to $x$ or zero:

$$
T^2=|x_a\times x_b|^2
=|x_a|^2|x_b|^2-(x_a\cdot x_b)^2=\det G.
$$

Thus $T=0\iff\operatorname{rank}A<2$ in exact mathematics. On a general
$\mathbb R^3$ map this fails: $x(a,b)=(a,b,0)$ at $(1,0)$ has independent
derivatives $(1,0,0),(0,1,0)$ and nonzero cross product but $T=0$. For a
one-parameter family there is no area factor; use speed, not a fabricated
zero `T`. Numerically small quantities are reported, never forcibly zeroed.

Rank loss is only a **fold candidate**. A stationary derivative does not prove
a turn, and rank-one loss alone does not distinguish fold from cusp or give a
brightness certificate. Callers can locate candidates by bounded roots of the
public `signed_area` on their own two-parameter slices. The column regression
uses `scipy.optimize.brentq` and independently minimizes NumPy deviation;
it does not supply a general zero-set tracker or promise complete discovery.

### The Lowitz special point: exact symmetry, measured residuals

For path 3→5 at caller coordinates azimuth=270°, zenith=90°, roll=0, the
principal section perpendicular to the c axis contains the incoming and
outgoing rays and both side-face normals. Both world angular-velocity axes
(azimuth and zenith) lie in that plane. Let $H$ be reflection in it. For either
axis $w$ in the plane and a base face normal $n$ in the plane,
$H\operatorname{Rot}(w,t)n=\operatorname{Rot}(w,-t)n$. Snell's vector equations
are orthogonally equivariant and the incoming ray is fixed by $H$, so
$x(-t)=Hx(t)$. Differentiating gives $-x_i=Hx_i$: **each derivative is normal
to the same plane**, therefore the two are exactly collinear and $T=0$.
This proof applies at all tested sun heights 10/15/20/30° in the smooth domain.
It is not a claim that a floating-point backend must return literal `0.0`.

Tests separately check the plane/axes/ray premises, positive domain margins,
nonzero speeds and unmodified AD residuals. At h=15°, speeds are
`0.2771075674 / 0.4661150252`, the null parameter line is approximately
`(1.682,-1)`. Along the actual mixed-parameter curve, doubling small steps
multiplies displacement by approximately four; along azimuth alone it doubles.
Parry keeps only that azimuth direction and passes through the same point.
These displacement checks, not the first-order label alone, support the
fixture-specific quadratic-law/turning statements.

## Self-contained conformance and historical precision

`tests/test_family_sky.py` owns the caller fixture chain and independent
NumPy `phi_batch`/Snell/mirror/rotation displacement oracles; it imports no
writing-repository script or data. Coverage includes variable $n$, non-star
crystal normals, internal partial reflection, invalid mixed batches, $K=1/2/3$,
input degeneration, FD step ladders and random positive-margin rotations.
No dependencies are added, no `focusing` classifier or `signature` behavior is
changed. Run:

```bash
uv run --python 3.12 pytest -q -m "slow or not slow" tests/test_family_sky.py
```

Historical numbers carry search-grid precision, not exact-root status:

- The original Lowitz search (azimuth step 3°, zenith step 0.25°, endpoint
  inset 1°) returns **51 candidates: 50 general + 1 special**. All 50 general
  points are regular; $|T|$ min/median/max is about
  `0.00189718 / 0.04043054 / 0.13232419`. Zenith speeds are about
  `0.02064–0.46260`, median `0.23558` degrees/degrees. The source probe's
  display formula erroneously multiplied an already radian-normalized speed
  by `degrees(1)/(2*0.02) ≈ 1432.39`; these are independently recomputed units.
- Column historical `(az,roll)=(90°,115.90°)` gives $T≈8.92012\cdot10^{-6}$;
  `(210°,5.10°)` gives $T≈1.24164\cdot10^{-4}$, **not** strictly ≤`1e-4`.
  Both are the stated `1e-4`-scale neighborhood, inherited from a 0.05° grid.
  Fine roots are approximately `115.919649955° / 5.102702405°`; the independent
  deviation minima agree within `6e-9 rad` in the recorded run. Tests retain
  both historical observations, rather than moving coordinates or snapping
  derivatives to satisfy a provisional bound.
- On 525 positive-margin Haar samples the recorded maximum absolute
  $T^2-\det G$ residual is `1.84e-15`, unit-norm residual `4.45e-16`, and
  tangency residual `1.78e-15`; the permanent identity assertion is the
  operation-scaled bound `512*eps*max(1, ||G||²)`, not an assertion that these
  observed maxima repeat bit for bit on every BLAS/JAX backend.

The consumer handoff is a pinned LI revision plus `family_sky` calls for
computation; the writing repository retains its parameterization, plotting
and comparison glue. Updating that repository or its dependency pin is a
separate consumer action, not performed here.
