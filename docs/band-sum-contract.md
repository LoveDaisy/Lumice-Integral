# Band-Sum Contract (Module B)

The backend-independent specification of module B of the shared analytic
library: the single-path $S^2$ event store and the band sum that turns it into
a brightness map. It serves Lumice Analyze function 2, the single-path
whole-sky map of the path view (Lumice `doc/raypath-analysis.md` §5.1.2,
§5.1.8, the "analytic result" of phase two). The rollout is that of
[overview.md](overview.md) §5.3: in wave 2 Lumice implements module B and LI
only certifies it by parity, without switching to it. Module A's contract is
[phase1-math-contract.md](phase1-math-contract.md) §9–§9.5; this document
follows its form, and the parity fixtures are
[analytic-parity-fixtures.md](analytic-parity-fixtures.md) §3.4.

A backend is written from this document and
[conventions.md](conventions.md). The design narrative behind the estimator
(why it is correct, what it costs, how it compares with the other
quadratures) is [phase2.md](phase2.md) §5 and §8. Names such as
`band_sum.scatter_store` are the Python reference, cited in footnotes only.
The keywords MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Scope of v1

| | v1 |
|---|---|
| Path | One concrete ordered face sequence `P` (Lumice face numbers, conventions #1), entry, internal reflections, exit. |
| Wavelength | One refractive index per call. |
| Symmetry | `symmetry_semantics = "none"`: no reduction, no transport. |
| Crystal | The closed-form prism of any `face_distance` and the Lumice pyramid, the crystals of module A (fixtures §2.1). |
| Pose density | The five families of §2.2. |
| Pixels | A table of directions, any projection (§2.3). |
| Output | Per pixel a value and its sampling diagnostics (§6). |

**Symmetry and the path list.** Lumice's raypath list shows L2-folded rows
(conventions #21): one row per physical equivalence class of the shape and the
pose ensemble. The path view of a selected row shows that row's light. A
caller obtains it by calling v1 once per member of the row and adding the
member maps. This is a performance choice, not a semantic one: LI's own
renderer serves a class from one store through `G_true` transports
([phase2.md](phase2.md) §3.3), and the two forms agree to the sampling noise.
Measured on the class `[3,5]` (12 members, $N = 10^6$, 2400 pixels of the
canonical strip): the median of $|\mathrm{rel}|\sqrt{K_{\mathrm{eff}}}$ is
`0.52` for the random and the column (`0.5°`) density, and the sums over the
lit pixels agree to `0.99997` and `1.0003` (the lattice is not closed under
`G_true`, so the two forms sample different points and cannot agree bit for
bit). The per-member form costs one store and one pass per member: a PBD row
of the regular prism has at most 24 members, 12 for `3-5`, and the pass took
`1.8 s` instead of `0.2 s` there (task `band-sum-module-spec`, 2026-09-29).

**Not in v1, and how a caller composes it.**

- Class accumulation and `G_true` transport: per-member calls, summed (above).
- Several wavelengths: one call per distinct refractive index, weighted and
  summed by the caller (LI's `spectrum.xyz_band_sum` does exactly that on top
  of the monochrome band sum).
- Divergent light ([phase2.md](phase2.md) §9), multi-scattering chains,
  a GPU backend, float32 stores ([phase2.md](phase2.md) appendix: float32
  moves a pixel by up to `5.5e-3`), non-uniform samplers.
- Pose densities outside §2.2 (§2.2 lists what Lumice can configure and v1
  cannot).

## 2. Inputs

| Field | Requirement |
|---|---|
| `crystal` | The `LUMICE_ANALYTIC_Crystal` scalars of module A (fixtures §2.1). |
| `faces` | `P`, as in module A. A path with fewer than two faces MUST be rejected. |
| `refractive_index` | The ice index of the call. |
| `incident_direction` | `s`, the world propagation direction of the sunlight (sun → crystal). The sun direction `ŝ = −s` is derived, never supplied independently (conventions #4, #5). |
| `pixels` | The pixel table of §2.3. |
| `n` | `N`, the number of sample points (§3). |
| `pose_density` | One family of §2.2 with its parameters. |

### 2.1 Directions

Every direction is a world unit vector. `s` and the pixel directions are
propagation directions; the sky point a pixel shows is the negative of its
direction (conventions #8). A direction `x` has the deviation
$\delta(x) = \angle(x, s)$, in $[0, \pi]$. Deviation `0` is the sun's own
direction (outgoing light that continues along `s`), deviation $\pi$ the
antisolar point.

### 2.2 Pose densities

The pose density $\rho$ is a density relative to the Haar probability measure
on $\mathrm{SO}(3)$ (it integrates to one, without the $1/8\pi^2$ of
$d\mathrm{Vol}_g$). It reads a pose $R$ (body → world, $v_W = R v_B$) only
through the zenith components of the body axes, the third row
$e_j = R_{3j}$ (1-based rows, $j = 1, 2, 3$). With the c axis zenith
$\theta = \arccos(\mathrm{clip}(e_3, -1, 1))$ and the roll
$\psi = \operatorname{atan2}(-e_2, e_1)$:

| Family | Parameters (degrees) | $\rho(R)$ |
|---|---|---|
| `random` | none | $1$ |
| `column`, `plate` | `zenith_mean_deg` $\mu$ (default 90 / 0), `zenith_std_deg` $\sigma$ (required) | $2\,g(\theta)/I$ |
| `parry`, `lowitz` | `zenith_mean_deg` $\mu$ (default 90 / 0), `zenith_std_deg` $\sigma$, `roll_mean_deg` $\mu_r$ (default 0), `roll_std_deg` $\sigma_r$ (both widths required) | $\big(2\,g(\theta)/I\big)\,\big(2\pi\,h(\psi)/Q\big)$ |

with

$$
g(\theta) = e^{-(\theta-\mu)^2/2\sigma^2}\ \text{on } [0,\pi],\qquad
I = \int_0^\pi g(\theta)\sin\theta\,d\theta,
$$

$$
h(\psi) = e^{-\tilde\psi^2/2\sigma_r^2},\quad
\tilde\psi = \big((\psi - \mu_r + \pi) \bmod 2\pi\big) - \pi,\qquad
Q = \int_{\mu_r-\pi}^{\mu_r+\pi} h\,d\psi .
$$

`mod` is the floored modulo (result in $[0, 2\pi)$), so $\tilde\psi$ is the
offset reduced to one period. $I$ and $Q$ MUST be accurate to `1e-12`
relative: they scale every value of the call (LI integrates $\pm 12\sigma$
windows clipped to the domain with 400-point Gauss–Legendre,
`pose_density.zenith_marginal_integral`, `roll_marginal_integral`; a fixture
carries LI's values as `normalization_informative`). A family with a
parameter it does not use (a roll width on `column`) MUST be rejected. The
derivation of these densities (why $\sin\theta$ appears in $I$ and not in
$g$) is `pose_density`'s module docstring, and the roll reference (`ψ = 0`
puts body `+x` in the vertical plane of the c axis, on the upper side) is
conventions #2 and #3.

**Lumice configuration.** The one-to-one table of the five families and
Lumice's `axis` blocks (zenith, azimuth and roll distributions and the
`kFullSphere` / `kRollLockedGauss` paths) is
[ch11-pose-density-families.md](ch11-pose-density-families.md) §1; this
document does not repeat it. Lumice configurations that v1 cannot express,
and for which the analytic view is therefore not available: a zenith of type
`zigzag`, `laplacian`, `uniform` over less than the full range, or
`gauss_legacy` (no $\sin\theta$); a fixed (scalar) zenith; an azimuth that is
not uniform over 360°; a Gaussian roll combined with anything but a Gaussian
zenith. These are Lumice-only until LI models them (ch11 §1, "Lowitz zenith
type").

### 2.3 The pixel table

A pixel is given by directions, so that the contract does not depend on a
camera model:

| Field | Requirement |
|---|---|
| `centre` | `(P, 3)`: the direction of each pixel's centre. It fixes the pixel's azimuth about `s` (§4.3) and its deviation $\delta_c$. |
| `corners` | `(P, 4, 3)`: the four corner directions of each pixel, **in cyclic order** around it (either orientation). |
| `solid_angle` | `(P,)`: $\Omega_p$ in steradians. Used by the rank-0 point mass (§5) and the absolute scale (§8), not by the band sum. |
| `labels` | `(P, 2)`: the caller's pixel indices, copied to the output. |

The pixel, as a region, is the spherical quadrilateral with great-circle edges
through its corners. On a linear lens this is exact (a straight pixel edge on
the tangent plane is a great circle); on a curved projection it is the
approximation the band is built on (§4.1).

### 2.4 Examples of pixel tables (informative)

Neither expansion is part of the contract; a caller supplies the table.

- **Lumice linear lens** (`camera.linear_pixel_outgoing_direction`, from
  Lumice's `MakeCameraRotation` / `ProjectExitToPixel`): pixel `(row, column)`
  covers `u ∈ [column, column + 1)`, `v ∈ [row, row + 1)`, centre at
  `+0.5`, corners at `±0.5`; the solid angle LI uses is the centre
  approximation $\cos^3\theta_c/\mathrm{scale}^2$
  (`path_class.pixel_solid_angle`).
- **Single-disk Lambert azimuthal equal-area** (Analyze, `raypath-analysis.md`
  §5.1.8 D2; the prototype `doc/prototypes/analyze-workspace.html`,
  `makeView` / `vInv`): a view centre $c$ (sky direction), `up` the zenith
  projected on the tangent plane at $c$ (`[-1, 0, 0]` when $|c_z| > 0.999$),
  `right = c × up`, a disk of radius `0.492 size` pixels reaching the field
  radius $t$, so $k = 0.492\,\mathrm{size} / (2\sin(t/2))$ pixels per unit
  $\rho = 2\sin(\vartheta/2)$. Screen point `(x, y)` (`y` down) maps to
  $X = (x - \mathrm{size}/2)/k$, $Y = (\mathrm{size}/2 - y)/k$,
  $\vartheta = 2\arcsin(\min(1, \sqrt{X^2+Y^2}/2))$ from $c$ toward
  `right X + up Y`. The map preserves area, so every pixel's solid angle is
  exactly $1/k^2$. The pixel direction is the negative of that sky point.
  (`parity_export.lambert_pixel_table`.)

## 3. The sample

The sample and its fields are those of module A's discovery,
[phase1-math-contract.md](phase1-math-contract.md) §9.5.2, without change:
`N` points $u_i$ of the antipodal Fibonacci lattice (the sun in the crystal
frame, $u = R^{-1}\hat s$), at each the path's validity, entry measure $A$,
Fresnel factor $T$ (entry and exit transmittances times every internal
reflectance, `1` under TIR, conventions #18), the body-frame outgoing
propagation direction $\varphi_i = \Phi_P(-u_i)$ and the deviation
$D_i = \angle(\varphi_i, -u_i)$. An event is **kept** iff
$w_i = A(u_i)\,T(u_i) > 0$. Only kept events enter §4 and §5; `N` counts every
lattice point, kept or not.

- For parity the sample MUST be the §9.5.2 lattice with the fixture's `N`. A
  production backend MAY use another uniform point set; §9 gives the sizing.
- The sample does not depend on `s` or on the pixels: one sample serves every
  sun direction and every view of a (crystal, path, index). A backend SHOULD
  keep it across calls that change only those.
- The kept events are ordered by increasing $D_i$, ties in lattice order.
  The order matters only for summation round-off.

## 4. The band sum

### 4.1 The band of a pixel, and singular pixels

For each pixel, $\delta_c = \delta(\text{centre})$ and

$$
\delta_{\mathrm{lo}} = \min_k \delta(\text{corner}_k),\qquad
\delta_{\mathrm{hi}} = \max_k \delta(\text{corner}_k),\qquad
\Delta\delta = \delta_{\mathrm{hi}} - \delta_{\mathrm{lo}},
$$

with $\delta(x) = \arccos(\mathrm{clip}(x\cdot s, -1, 1))$. The band is the
extremes of the corners, not of the whole pixel. The two differ where a pixel
edge passes closer to `s` (or `−s`) than both its corners, which happens only
next to the sun and the antisun: the pixel above the sun pixel of the fixture
views has the band `[1.14°, 2.55°]` while its edge midpoint is `0.73°` from
the sun. That is part of the estimator's pixel model (§6), not an error to be
corrected by a backend.

A pixel is **singular** iff it contains `s` or `−s`. The containment test of
a direction `x` in a pixel with cyclic corners $c_0 \dots c_3$ is

$$
x\cdot\textstyle\sum_k c_k > 0
\quad\text{and}\quad
\operatorname{sign}\det(c_k, c_{k+1}, x)\ \text{is the same for}\ k = 0 \dots 3
\ (\text{zero allowed}, c_4 = c_0).
$$

The first condition excludes the antipode, which passes the sign test too. In
a singular pixel $\sin\delta_c$ may vanish and the corner band misses the part
of the pixel below its smallest corner deviation, so the estimator has no
meaning there: a backend MUST return `status = "singular"` and no value.
(Measured: the fixture view's sun pixel has $\delta_c = 0$ and the band
`[1.14°, 1.14°]`.) A caller that needs light in the sun pixel is asking for
the rank-0 point masses of §5, or for a pixel model outside v1.

### 4.2 Band membership

The band's events are the kept events with

$$
\delta_{\mathrm{lo}} \le D_i < \delta_{\mathrm{hi}}
$$

(left-closed, right-open; `band_sum.band_poses`: `searchsorted(D, [lo, hi])`
on the sorted $D$). $K$ is their number.

### 4.3 The pose and the density of an event

Event $i$ is posed at the pixel's azimuth exactly as module A poses a
discovery candidate ([phase1-math-contract.md](phase1-math-contract.md)
§9.5.3) with the target $d$ = the pixel's `centre`:

$$
R_i = W F_i^{\mathsf T},\quad
W = [\hat s,\ e,\ \hat s\times e],\ e = \mathrm{unit}\big(d - (d\cdot\hat s)\,\hat s\big),\quad
F_i = [u_i,\ f_i,\ u_i\times f_i],\ f_i = \mathrm{unit}\big(\varphi_i + \cos(D_i)\,u_i\big),
$$

columns as listed. Then $R_i u_i = \hat s$ and $R_i\varphi_i$ lies at the
event's own deviation $D_i$ in the half-plane of $d$. The pixel's $\delta_c$
in place of $D_i$ does not give a rotation ([phase2.md](phase2.md) §5). $e$
exists because the pixel is not singular. The density needs only the third
row of $R_i$:

$$
(R_i)_{3j} = W_{3,\cdot}\cdot (F_i)_{j,\cdot},\qquad j = 1, 2, 3,
$$

the dot product of the pixel vector $W_{3,\cdot}$ (the zenith components of
$\hat s$, $e$, $\hat s\times e$) and the event vector
$(F_i)_{j,\cdot} = (u_{i,j}, f_{i,j}, (u_i\times f_i)_j)$
([phase2.md](phase2.md) §8). A backend MAY build $R_i$ in full or only this
row; the fixtures compare the result, not the route. Then
$\rho_i = \rho(R_i)$ by §2.2.

### 4.4 The value

With the contributions $c_i = w_i\,\rho_i$ of the band's events and
$S = \sum_i c_i$,

$$
\hat I = \frac{S}{2\pi N\,\Delta\delta\,\sin\delta_c}\quad\text{if } S \ne 0,
\qquad \hat I = 0\quad\text{if } S = 0 .
$$

`N` is the lattice size of §3. The constant is derived, not fitted: $N$
equal-area points ($4\pi/N$ each) and the Haar split
$d\mu = dA/4\pi\cdot d\psi/2\pi$ turn the $S^2$ level-set integral into this
sum ([phase2.md](phase2.md) §5; measured median ratio `1.0000` against the
Phase I point values at $N = 10^8$). $\hat I$ estimates the band average of
$I\sin\delta'$ divided by $\sin\delta_c$, in the Phase I pixel normalisation
(§6). An empty band is `0` and never divides.

### 4.5 Sampling diagnostics

| Quantity | Definition |
|---|---|
| `K` | Number of band events (§4.2). |
| `K_rho_pos` | Number of band events with $c_i > 0$, evaluated in IEEE-754 double **with gradual underflow** (no flush-to-zero): a narrow density's tail is subnormal long before it is zero. |
| `K_eff` | Kish's effective sample size $S^2 / \sum_i c_i^2$; `0` for an empty band or $\sum c_i^2 = 0$. |

`K_eff` makes $1/\sqrt{K_{\mathrm{eff}}}$ the relative sampling noise of the
value for scattered points. It is a noise proxy, not a confidence interval:
on the Fibonacci lattice a thin band usually beats it by 7–9×, but the
lattice can alias (Parry at $N = 10^7$ shows a sawtooth in $K_{\mathrm{eff}}$
and in the error, [phase2.md](phase2.md) §5 correction 1 and appendix). A
narrow density lowers $K_{\mathrm{eff}}/K$ pixel by pixel, not the estimate
(§9). In v1 every event contributes once (no transports), so LI's
`K_EFF_SEMANTICS = "per_event"` is trivially satisfied.

## 5. Rank-0 paths: a point mass

A path is **rank 0** iff its fold matrix is the identity and its wedge angle
is zero: $M = S_{m_k}\cdots S_{m_1} = I$ to `1e-12` per entry
($S_n = I - 2nn^{\mathsf T}$ over the internal faces' outward normals, the
identity without internal reflections) and
$\angle(n_a, -M^{\mathsf T} n_b) \le 10^{-9}$ degrees for the entry and exit
normals $n_a$, $n_b$ (`geometry.halo_map_rank`; `1-2`, `3-6`, ch8 A0-06).
Every pose of such a path sends the light on along `s`: its whole
contribution is a point mass in the sun direction,

$$
m = E_{\mathrm{Haar}}\big[\,[R \in V_P]\ \rho(R)\,A_P(R)\,T_P(R)\,\big],
$$

in the normalisation of §4.4. The band sum does not apply (every $D_i$ is 0).
The output of a rank-0 call is `m`, and per pixel the value $m/\Omega_p$ with
`status = "point_mass"` on the pixel that contains `s` (§4.1's test), `0`
with `status = "ok"` on every other pixel. The singular rule of §4.1 does not
apply. A sun direction exactly on a pixel edge or corner belongs to several
pixels by the closed test; the backend puts the mass on the first of them in
table order (the fixtures keep the sun strictly inside a pixel).

- **Random density.** $\rho = 1$ and the fields depend on $u$ only, so
  $m = \int w\,dA/4\pi$, and on the sample of §3 $m = \sum_{\text{kept}} w_i / N$.
  For parity a backend MUST compute it this way (deterministic).
- **Other densities.** Any consistent estimator of $m$ MAY be used, with its
  standard error reported. LI's reference is a Monte Carlo mean over a fixed
  Haar stream (`path_class.estimate_rank0_contribution`; `numpy` generator
  `20260916`, $4\times10^6$ poses), which a backend cannot reproduce, so the
  comparison is statistical (fixtures §4). A deterministic alternative exists
  and is recorded, not required: with $R_\psi$ the twist of a pose
  $R_0(u)$ ($R_0 u = \hat s$) by $\psi$ about $\hat s$, the average
  $\langle\rho\rangle_\psi(u) = \frac{1}{2\pi}\oint \rho(R_\psi)\,d\psi$ is a
  one-dimensional periodic integral (for the zenith families,
  $\cos\theta(\psi) = \cos\alpha\cos\beta + \sin\alpha\sin\beta\cos(\psi - \psi_0)$
  with $\alpha = \angle(u, e_3)$ and $\beta$ the sun's zenith angle), and
  $m = \frac1N\sum_{\text{kept}} w_i\,\langle\rho\rangle_\psi(u_i)$.

Measured (canonical column, $n = 1.31$, sun at 15°): `3-6` under the random
density has $m = 0.11816635$ at $N = 10^5$ and $0.11816641$ at $10^6$,
against the Haar stream's $0.11825 \pm 0.00015$ ($4\times10^6$, $z = 0.58$);
under the plate family (1°) the stream gives $0.157 \pm 0.011$.

## 6. Output

Per pixel, in table order:

| Field | Rank 2 | Rank 0 |
|---|---|---|
| `label` | copied | copied |
| `status` | `ok` or `singular` | `point_mass` or `ok` |
| `value` | $\hat I$ (§4.4); none when singular | $m/\Omega_p$ or `0` |
| `delta`, `delta_lo`, `delta_hi` | $\delta_c$ and the band (§4.1), also for a singular pixel | — |
| `K`, `K_rho_pos`, `K_eff` | §4.5; none when singular | — |

A rank-0 call also returns `m` (and its standard error when it is not the
lattice mean).

**Units and normalisation.** A value is LI's Phase I pixel value: power per
steradian sent along `P` by one crystal of the pose ensemble, per unit
incident irradiance, with the crystal at hexagon edge `a = 1`
(conventions #17). §8 converts it to Lumice's brightness.

**What a value is.** A band average in deviation and a point in azimuth
(the pixel centre's). Phase I and the contour quadrature give point values;
next to a steep edge or an inner caustic the two differ by the pixel model,
not by sampling error ([phase2.md](phase2.md) §5 "Pixel model"). The sampling
error is described by `K_eff` (§4.5); there is no error bound.

## 7. Conformance

A backend is certified by the `band_sum` fixtures
([analytic-parity-fixtures.md](analytic-parity-fixtures.md) §3.4) in two
layers.

### 7.1 Layer 1: the estimator on LI's events

A fixture carries the kept events of every non-singular pixel's band (the
union, in LI's order: $u_i$, $\varphi_i$, $D_i$, $w_i$) and, for a rank-0
path under the random density, every kept $w_i$. The backend runs §4 (or §5)
on them. The statuses and `K` are exact. `K_rho_pos` is exact up to the
pixel's count of events whose $c_i$ is subnormal in LI (a flush-to-zero
backend would count them 0; §4.5 forbids that, the allowance only keeps the
comparison honest). The value and `K_eff` agree to `1e-10` relative: the same
sum over bit-identical events differs only by summation order and the
rounding of `arccos`, `atan2` and `exp`, which LI's two forms of the sum
(scatter and gather) show at `1e-12`, `3e-12` for `K_eff` near one
([phase2.md](phase2.md) §8).

### 7.2 Layer 2: the whole call

The backend regenerates the sample (§3) and runs the call end to end. Two
float64 implementations of the same fields differ by rounding, so an event
can change sides where it is within float reach of a decision:

- a band end: $|D_i - \delta_{\mathrm{lo}}|$ or $|D_i - \delta_{\mathrm{hi}}|$ at most $\varepsilon_D = 10^{-9}$ rad;
- a validity gate: the smallest validity margin of module A (fixtures §3.1, `branch_margins`) within $\varepsilon_g = 10^{-9}$ of zero;
- the $w > 0$ gate: $0 < w_i \le \varepsilon_w = 10^{-9}$.

At every validity gate $w \to 0$ continuously (the entry measure goes like the
incidence cosine at entry and at every face the corridor projects through,
$T_{\mathrm{exit}} \to 0$ at exit TIR), so a flip moves the value by at most
the event's own small contribution. LI counts, per pixel, the lattice points
in each of the three sets that lie in the band widened by $\varepsilon_D$
(`allowance.candidates`); `K` and `K_rho_pos` may then differ by that count,
the value by the candidates' contributions (and $\varepsilon_w\rho_{\max}$
for each candidate LI did not keep), and `K_eff` by the Kish range they span
(`allowance.*_layer2`). The float64 deviation of the same closed-form chain
differs by `1e-13` rad or less away from $D = 0, \pi$, so $\varepsilon_D$
leaves four orders of magnitude. The expected count of candidates is about
$N\,\varepsilon/(4\pi)$ per unit of curve length, and on the v1 fixtures it
is zero in every pixel: layer 2 is then as tight as layer 1. For a rank-0
path the regenerated lattice mean agrees to `1e-10` relative; under another
density the comparison is $|m - m_{\mathrm{LI}}| \le 5\sqrt{\sigma_{\mathrm{LI}}^2 + \sigma^2}$.

Layer 1 isolates the estimator; layer 2 adds the sampler and the fields, which
module A's `evaluate_path` fixtures certify pose by pose. A backend passes
both.

## 8. Absolute scale against Lumice

LI's value is not Lumice's brightness. The conversion is derived and has been
checked against Lumice exports with nothing fitted
([ch06-reference-fixture.md](ch06-reference-fixture.md) §7 stage 4;
conventions #17):

$$
\frac{\mathrm{raw}[p]}{E} = K_p\,\hat I_p,\qquad
K_p = N_{\mathrm{sym}}\;\bar y(\lambda)\;\frac{\Omega_p}{S/2},
$$

with `raw / E` Lumice's pixel over its emitted energy, $\bar y(\lambda)$ the
CIE 1931 Y matching function Lumice multiplies into the Y channel at the
call's wavelength ($\bar y(550) $ for the monochrome 550 nm scene), $S$ the
crystal's total surface area at `a = 1` (Lumice weighs every ray by its
projected area over $S/2$ at entry), and $N_{\mathrm{sym}}$ the number of raw
raypaths the Lumice filter folds into the compared image. **In v1
$N_{\mathrm{sym}} = 1$**: the value is one concrete path, compared with a
Lumice image filtered to exactly that raypath; for an L2 row the caller's
member sum is compared with the row's filter, still with
$N_{\mathrm{sym}} = 1$ (LI's `compare_lumice_family.py` uses
$K_p = \bar y\,\Omega_p/(S/2)$ on class sums for the same reason). The
`N_sym = 12` of `probe_absolute_scale.py` belongs to a single representative
compared with a `PBD`-folded Lumice image.

Evidence: at matched refractive index the bright band of columns
106 / 126 / 146 agrees within `0.3 %`, the plate and Parry families' total
flux within `0.01 %` (`scripts/probe_absolute_scale.py`,
`scripts/compare_lumice_family.py`).

## 9. Reference defaults and evidence

Strategy, not mathematics, as in contract §9.5.9.

| Parameter | Default | Evidence |
|---|---|---|
| Sampler | antipodal Fibonacci lattice (§3) | The lattice's Haar mean of `w` agrees with independent uniform quaternions (`s2_store.self_check_haar_mean`); ψ invariance of the fields is checked by `self_check_psi_invariance` ([phase2.md](phase2.md) §4.1(a)). |
| `N` | size by $K_{\mathrm{eff}} \ge 10^4$ at the worst pixel of interest | [phase2.md](phase2.md) §5 correction 1 and appendix: $K_{\mathrm{eff}}$ grows as $N^{1.00}$ in every scene and family; at the canonical pixel size ($0.024°$) $K_{\mathrm{eff}}/N$ at the median lit pixel is $1.2\times10^{-4}$ (plate), $1.25\times10^{-5}$ (Parry), $2.6\times10^{-5}$ (Lowitz); the band keeps 65 % (column, on the sun vertical), 17 % (off it), 3 %, 0.45 %, 0.58 % of its events as $\rho > 0$. $N = 10^8$ serves all five chapter-11 families on the canonical strip. $K_{\mathrm{eff}}$ scales with the band width, so a pixel $s$ times wider needs about $s$ times fewer points. |
| Band | the pixel's corner extremes (§4.1) | Definition; the pixel model of §6. |
| Precision | float64 events and sums | float32 stores move a pixel by up to `5.5e-3` ([phase2.md](phase2.md) appendix, task `s2-event-store`). |
| Cost (reference, CPU) | — | Store: `80 s` to build at $N = 10^8$ (built once per crystal × path × index). Canonical strip (251 × 801) at $10^8$: `30.8 s` on 4 workers, `73.7 s` single process, worker RSS `441 MB` ([phase2.md](phase2.md) §8). Analyze-sized: a $161\times81$ window at $N = 10^6$, `3.4 s` for the full chain ([overview.md](overview.md) §5.3). |
| Rank-0 stream (other densities) | $4\times10^6$ Haar poses, seed `20260916` | `path_class.RANK0_SAMPLE_COUNT`; `3-6` plate: 7 % standard error (§5). |

**Not measured** (gaps, not claims): $N$ against $K_{\mathrm{eff}}$ on a
Lambert whole-sky view (every measurement above is on the linear ch06 strip or
its profiles); the band width and aliasing when Analyze zooms the path view to
about 1.5° of field (a pixel far narrower than the canonical one, so $N$ grows
with the zoom at fixed $K_{\mathrm{eff}}$, [phase2.md](phase2.md) §7).

## 10. Boundaries and open items

- **What the fixtures certify.** The LI side of every fixture, Lambert tables
  included, is computed by both of LI's forms of the sum: the production
  scatter (`band_sum.scatter_store` over `pixel_bands_from_directions`) and
  the per-pixel gather oracle (`band_sum.class_band_sum_of_band`). The camera
  enters LI's estimator only through a pixel's directions
  (`band_of_pixel_directions`), so a Lambert table is not a weaker case than
  a linear one.
- **The spec-only reference implementation**
  (`tests/test_band_sum_spec_reference.py`) was written from this document
  alone and recomputes the layer-1 fixtures within their tolerances. It shows
  that the text is enough to encode the estimator. It does not show that
  Lumice will read the text the same way: it shares its author's reading, and
  its input events are LI's. **A disagreement found by Lumice's first
  implementation is a revision of this document** (and of the fixtures), not a
  local fix in either code base.
- **Fiber and level set.** The estimator is a quadrature of the $S^2$
  integral and does not assume that a level set $\{D_P = \delta\}$ is one
  fiber. Whether the two correspond one-to-one on the pyramid and on paths
  with internal reflections is the open question of
  [overview.md](overview.md) §5.3 and is not decided here.
- **Sun-adjacent pixels.** The corner band (§4.1) misses the part of the
  pixels next to the sun and the antisun that is closer than their corners. A
  finer model there (an exact per-pixel deviation range) would be a v2
  change of this document.
- **Rank-0 under a non-random density** is statistical (§5). A deterministic
  ψ-quadrature is written down but not implemented in LI.
- **Unknown fields.** A consumer MUST ignore top-level manifest keys it does
  not know (later waves add fixture kinds under new keys), and fields of a
  fixture it does not use (`total`, `square`, `normalization_informative`,
  `projection` are informative).
