# Exact-family restricted structures

This is the LI research contract for `family_restricted`; it does not change
`focusing.classify`, finite-width pose densities, continuation, or rendering.
The crystal, path, refractive index, public **toward-sun** direction, and
`ExactFamily` are explicit inputs. The exact supports and angle order are in
[conventions](conventions.md#exact-family-supports). No Lumice code is called.

## Geometry before measure

For a family pose `R(q)`, `u(q)=R(q).T sun` and
`x(q)=-R(q) Phi_P(-u(q))`. One body direction can have multiple poses, and
those poses can have different sky positions. `ExactFamily.lifts` solves all
canonical inverse branches, not an arbitrary `align_rotations` replacement.
A necessary projection constraint is never a target-space hit certificate.

For a one-circle support, let `a=R(0).T z_world` and `h=sun_z`. Its u image is
`a.u=h`. Plate uses body z; Parry roll=0 uses body x. Column and Lowitz have
two parameters, not one circle. Column inverse branches solve
`(Rz(roll)u)_x=h`; Lowitz solves
`sin(zenith)(Rz(roll)u)_x + cos(zenith)u_z=h`, retaining every solution on
`[0,π]`. The residual `R u-sun` is checked on each returned lift.

## S1: restriction, not intersection with the ambient critical set

`restricted_critical` returns `records`, `coverage`, and `omitted`.
One-dimensional regular critical points solve `(D_P(u(q)))'=0` in the optical
interior. JAX differentiates the **composition**, including curvature, and
classifies a nonzero second derivative. Records contain radians `value`,
`location`, `source`, `profile`, `multiplicity`, parameter/rotation/u/sky,
geometric tangent `du/dq` (3 by family dimension), derivative residual and
minimum validity gate. The witness is optically valid, not necessarily positive
finite-crystal area (the corridor boundary is outside this module's scope).

For smooth positive image density, a one-dimensional Morse extremum has
pushforward density proportional to `|D-D0|^-1/2` (`inverse_sqrt`). Integrating
on a local interval gives mass proportional to `sqrt(epsilon)`; the regression
checks this independently by dense interval measure. A 2D Morse extremum has
`finite_jump`, a 2D saddle `log_divergence`. A slab whose fold matrix commutes
with rotations about the circle axis is constant by rotational invariance of
`u.T M u`. It is one `constant_set` with `delta_mass` and
`multiplicity=null`, not an arbitrary finite number of isolated roots. The
constant-set witness and invariance are distinct from domain clipping.

Numerical root enumeration is **not certified exhaustive**. One-circle results enumerate optical domain endpoints separately in
`endpoints`, with the binding `margins`. Their `profile` and derivative are
explicitly unavailable (`null` with a reason), rather than pretending a
closed-domain limiting value has a finite jet. Constant sets carry clipped
`parameter_intervals`, assembled from those numerical gate roots. The domain
root search remains non-exhaustive, even when rotational invariance is proved.
Two-dimensional results concern
regular image interior strata only; projection rank boundaries and parameter
endpoints are not silently classified as interior extrema. Degenerate jets,
continuous lifts, unsupported strata and exhausted budgets are named refusals.
These omissions must be displayed by a downstream consumer, not promoted into
a full-support certification or a brightness-peak prediction.

## S2: margin intersections

`intersect_margin` takes the **original** domain margin name. Kind 2 means a
validity gate; kind 3 means an internal TIR discriminant (a weight kink, not a
validity gate). Every candidate is clipped against **all** validity gates and,
for a kink, its reflecting incidence branch. Output contains isolated
`records` with `intersection=transverse|tangent`, or continuous `curves` with
`intersection=coincident|curve`, ordered pose witnesses, closure and endpoint
labels. Residuals are dimensionless margin residuals. Empty means no returned
intersection under the stated coverage, never proof of an unsearched set.

The entry gate is the plane `n_entry.u=0`. When unfolded internal incidence
normal `m` is orthogonal to entry, a kink is the circle
`m.u=-sqrt(n^2-1)`. Intersecting two plane sections on S² gives zero, one
(tangent), two (transverse), or an identical circle. Coincidence requires this
structural plane identity, not a near-zero scan. General scalar roots also
bracket stationary points so an even contact is not discarded by a sign-only
search. Numerical enumeration keeps a non-exhaustive coverage token.

Two-dimensional exact supports lift upstream boundary/kink arcs. On circular
margins the family-chart cuts are inserted analytically: column
`u_z=+-cos(h_sun)`; Lowitz `(Rz(roll)u)_y=+-cos(h_sun)` and the endpoint
circles `u_z=+-sin(h_sun)`. Each constant-rank interval is lifted separately
on both sheets. Branch tracking matches **rotations** outward from an interior
witness, so an azimuth seam cannot connect unrelated sheets. A chart cut is
labelled `family_image_boundary`. General marched-margin clipping is refused
until it has a comparable stratum construction. Existing marched-kink
incompleteness is never upgraded by `KinkCurve.complete`.

## Oriented kink sky images

`kink_sky` uses those actual poses in the optical sky map and reports the image
dimension: points, curves, or empty. One-dimensional supports generally give
points at a transverse kink intersection; coincident constraints can retain an
arc. Two-dimensional supports generally retain a curve, but **do not guarantee
one**: `3-1-6` on column is a useful counterexample. Its basal mirror normal
is the horizontal c axis; the kink fixes its dot product with sun, so it has
only two world directions and each connected kink branch maps to a point.
A near-constant numerical image without an available structural identity is
refused (`unclassified_sky_collapse`), not automatically labelled a curve.
Nontrivial computed examples are `3-1-6` on Lowitz and `1-3-2` on column.
Sun/index changes rebuild the constraints and poses; no translated cached sky
curve is reused. Cross-wavelength fixtures are independent per-index results,
not rank-based pairing across a topology change.

Optical closure values use the established dropped-root exit-Snell limit.
There is no epsilon inward displacement. `routed_nonfinite` counts this
specific defined limit route, separate from an algorithm failure. Unsupported
regimes raise `RestrictedEscape(regime,message)`, a `TopologyEscape` subtype;
parity fixtures retain a named refusal rather than serializing a NaN as an
ordinary `null`.

## C11 and C16 evidence

C11 is the azimuth/deviation distinction. A same-altitude target at relative
azimuth ±120° has
`cos(delta)=sin(h)^2+cos(h)^2 cos(120°)`. At sun altitude 9° this gives
117.599764°; the current sun20 `1-4-5-2` plate scene gives 108.937304°.
The tests evaluate both this spherical identity and actual optical sky poses.
They do not conflate the two scene calibrations.

C16 concerns **target fiber intersected with exact family in SO(3)**, not just
an S² margin intersection or a common delta. A one-dimensional fiber can lie
inside the two-dimensional column support without making the whole family
pinned. The original near double root target `(elevation40°,azimuth30°)` is
not a tangent. `family_target.intersect_target` is deliberately narrow:
regular-prism literal `3-5`, explicit local seeds, full sky-vector hit within
`1e-10`, positive A and T, and full SO(3) direction rank 2. A least-squares
stop or a rounded scratch coordinate is not accepted as a tangent. The
`locate_elevation_witness` helper independently recomputes the column fold
on a requested elevation by solving sky elevation and restricted Jacobian
area=0 (Parry solves only the requested elevation). At full n(550), sun20,
the computed column target azimuth is 29.680392011006834°; the regular Parry
witness is 29.835011461218983°. These differ from old scratch locators by
about 1.4e-6°, so callers must explicitly use the returned full target, not
silently snap an input target or widen its hit tolerance.

For two-parameter supports, the family constraint is differentiated **along
the target fiber**, including its curvature. In a right Lie chart let
`r(x)` be the full target residual, `J=r'(0)`, `v` its unit null vector and
`a` solve `J a=-r''(v,v)`, `v.a=0`. Then the constraint jet is
`g'=grad(g).v`, `g''=g''(v,v)+grad(g).a`. A tangent requires a vanishing
first derivative, a nonzero second derivative, and corrected full-target
poses on both sides with same-sign quadratic constraint growth. The sign
of `v` is canonicalized by its largest component; the geometric tangent is
a line regardless of LAPACK sign. Near-fold candidates without sufficient
residual-to-separation evidence are unsupported rather than merged into a
spurious tangent.

For the mirrored-sun column target `(-20°,0°)`, tangential Snell conservation
for side-face chains gives `(target-sun).R e_z=0`; a nonzero vertical
difference forces `Rzz=0` on the whole fiber. Bounded seed correction plus
SO(3) continuation witnesses one closed component, all positive AT. Closure
does **not** certify every component was found. Lowitz regular witnesses
retain free zenith; endpoint strata, other crystals/paths and unresolved
searches return `status=unsupported`, never a certified empty set.

The companion parity data carry full-direction residuals, contact jets,
corrected offsets and finite-crystal area witnesses separately. Reflecting
the Parry/Lowitz witness target's y coordinate leaves its deviation and u
projection eligibility unchanged but fails the actual sky hit (residual
about 0.7622); this is the persistent projection-false-positive case.

## Reproducibility and availability

Use complete `spectrum.refractive_index(lambda_nm)` values, not rounded n.
Profiles and intersections are local geometry under explicit coverage, not an
MC visibility claim. Arrays are float64; angular values in radians, rotation,
u, sky and geometric tangents dimensionless/radian per parameter radian.
Derived witnesses at an inverse-projection fold lose pose digits at
`sqrt(eps)`; the forward residual and constraint jets, not an exact live
coordinate pin, carry correctness. See the additive family fixture kinds in
[analytic parity fixtures](analytic-parity-fixtures.md).
