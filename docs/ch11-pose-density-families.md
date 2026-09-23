# Chapter 11 Pose-Density Families

> Status: implemented 2026-09-20 (task `pose-density-families`). The five
> families of chapter 11 — random, plate, column, Parry, Lowitz — are densities
> relative to Haar on SO(3), evaluated on the *same* fibers; nothing in
> discovery, continuation or the prescan table reads them. This document is
> the alignment table against Lumice's axis presets, the record of this
> renderer's own conventions, and the diagnostics that show what the families
> do on the canonical `3-5` scene — including the finding that, on the
> labelled path `3-5` and the ch06 strip, only column and random carry mass.

## 1. The five families

Implementation: `src/lumice_integral/pose_density.py` (three classes cover the
five families) and `build_pose_density(family, *, zenith_mean_deg=None,
zenith_std_deg=None, roll_mean_deg=None, roll_std_deg=None)`, the one place
that maps a family name to a class and fills the family defaults. Widths have
no defaults: `zenith_std_deg` is required for every family but `random`,
`roll_std_deg` for parry and lowitz; a missing or foreign parameter raises
`ValueError` naming it.

| Family | Class | Zenith of the c axis `theta` | Azimuth | Spin `psi` about the c axis | Lumice `kAxisPresets` (`src/gui/axis_presets.hpp`, evidence only) |
|---|---|---|---|---|---|
| `random` | `HaarUniformPoseDensity` | uniform (area measure) | uniform | uniform | zenith / azimuth / roll all `uniform` over `360 deg` → `kFullSphere` path: Haar itself |
| `plate` | `ZenithGaussianPoseDensity(mean 0 deg, std)` | Gaussian about `0 deg` *as a sphere density* | uniform | uniform | zenith `gauss(0, 1)`, azimuth full, roll free |
| `column` | `ZenithGaussianPoseDensity(mean 90 deg, std)` | Gaussian about `90 deg` as a sphere density | uniform | uniform | zenith `gauss(90, 1)`, azimuth full, roll free (the ch06 canonical scene uses std `0.5 deg`) |
| `parry` | `ZenithRollGaussianPoseDensity(90 deg, std_z, 0 deg, std_r)` | Gaussian about `90 deg` | uniform | Gaussian about `0 deg`, width `std_r` | zenith `gauss(90, 1)`, azimuth full, roll `kRollLockedGauss = gauss(0, 1)` |
| `lowitz` | `ZenithRollGaussianPoseDensity(0 deg, std_z, 0 deg, std_r)` | Gaussian about `0 deg`, wide | uniform | Gaussian about `0 deg`, width `std_r` | zenith `gauss(0, 40)` (v11: gauss preferred over zigzag), azimuth full, roll `kRollLockedGauss` |

Alignment notes, per column of the table:

- **Zenith, `sin theta` weighting.** Lumice samples the c axis from
  `p_sphere(theta) ~ p(theta) sin(theta)` (`doc/crystal-orientation-sampling.md`
  section 4.1, the unified inverse-CDF LUT of section 4.3), i.e. `p(theta)`
  is a density *on the sphere* and the area Jacobian is applied by the
  sampler. Our `zenith_gaussian` is that same `p(theta)`; the normalisation
  `I = int_0^pi g(theta) sin(theta) d theta` carries the Jacobian. The
  `gauss_legacy` type (no `sin theta`) is *not* modelled; it is Lumice's
  pre-fix reproduction mode only.
- **Azimuth.** Uniform in every family on both sides; it cancels in the Haar
  density.
- **Spin / roll.** Lumice's `roll` is the innermost `Rz` of the chain below,
  a spin about the crystal's own c axis; ours is `c_axis_roll(R)`, the same
  angle of the same chain (section 2). The locked families use a Gaussian on
  a single period (section 3); Lumice samples a Gaussian angle and applies
  it modulo `2 pi` (a wrapped Gaussian) — identical for the locked widths
  (`~1 deg`), different only when `roll_std` approaches the period.
- **Widths.** Lumice's preset widths (`1 deg`, `40 deg`) are reference values
  for the comparisons here, not defaults of `build_pose_density`.
- **Lowitz zenith type.** Only the Gaussian is implemented; Lumice's
  classifier also accepts `zigzag` / `uniform` / `laplacian` zenith types for
  the Lowitz preset. Recorded as a simplification, not planned.

## 2. Rotation chain and the roll reference

Lumice builds every crystal pose as

    R = Rz(az - pi) . Ry(-zenith) . Rz(roll)

(`src/core/simulator.cpp::BuildCrystalRotation`, mirrored by
`gui/axis_presets.hpp::ChainRotationToMatrix`; read as evidence, not
imported). The c axis is `R e3 = (cos(az) sin(zenith), sin(az) sin(zenith),
cos(zenith))` (the `-pi` of the outer `Rz` cancels the sign of `Ry(-zenith) e3`'s
horizontal component), so
`zenith = arccos(R[2, 2])` (`c_axis_zenith`, unchanged). The third row of
`Ry(-zenith) . Rz(roll)` is `(sin(zenith) cos(roll), -sin(zenith) sin(roll),
cos(zenith))` and the outer `Rz` leaves it alone, hence

    roll = atan2(-R[2, 1], R[2, 0])        (c_axis_roll, valid for sin(zenith) > 0)

`tests/test_pose_density.py::test_c_axis_roll_round_trips_the_chain_rotation_off_the_poles`
constructs the chain on a `(az, zenith, roll)` grid (zenith `10-170 deg`) and
recovers `roll` to `1e-12` modulo `2 pi`; a further test spins a pose about
its own c axis by `delta` and checks the extracted roll moves by exactly
`delta`. At the gimbal poles (`zenith = 0` or `pi`) only `az +- roll` is
defined and the value is arbitrary — the Parry / Lowitz zenith windows keep
their mass away from the poles, and the five-family pixel test (section 5)
checks every `rho_pose` on the diagnostic fibers is finite.

**What `roll = 0` means here.** With `roll = 0` the body `e1` axis — the
outward normal of face `3` of `geometry.core.HexPrism` — lies in the vertical
plane through the c axis, on the upper side; for a horizontal c axis face `3`
is the horizontal top face (the classical Parry orientation, one prism face
up). Lumice numbers the faces the same way (face `3` = body `+x`, face
`3 + i` at `i * 60 deg`: `src/core/geo3d_closedform.hpp` `kHexFaceCos/Sin`,
read as evidence; `doc/coordinate-convention.md` sections 1 and 5.3), so
this is also Lumice's `roll = 0`: the same labelled face is on top in both
renderers (corrected 2026-09-24, task `notation-alignment`; this paragraph
used to say the numberings were not aligned). The writing series'
chapter parametrisation `column_attitude(psi, theta)` puts face `3` at the
bottom for `theta = 0`: `theta = roll - 180 deg`, the same physical pose
with the labels `3` and `6` swapped (`docs/conventions.md` rows 2-3,
`tests/test_conventions.py`). Consequences for labelled paths are in
section 6.

## 3. Densities and their normalisation

Haar probability on SO(3) in the ZYZ chart is `sin(zenith) d az d zenith
d roll / (8 pi^2)`; the push-forward to the c axis is the uniform area
measure and the conditional spin on every fiber is uniform (module docstring,
steps 1-3). With `g(theta) = exp(-(theta - mean)^2 / (2 std^2))` on `[0, pi]`
and `h(psi) = exp(-(psi - roll_mean)^2 / (2 roll_std^2))` on the single period
`[roll_mean - pi, roll_mean + pi]`:

| Family | `rho_H(R)` | Normalisation constants |
|---|---|---|
| random | `1` | — |
| plate / column | `2 g(theta) / I` | `I = int_0^pi g sin theta d theta` (Gauss-Legendre, 400 nodes, `+-12 sigma` window clipped to `[0, pi]`) |
| parry / lowitz | `(2 g(theta) / I) . (2 pi h(psi) / Q)` | `Q = int h d psi` over one period (same rule, no `sin` weight) |

Every `rho_H` integrates to one against Haar; the `1 / (8 pi^2)`
Haar-to-`dVol_g` factor is applied by the quadrature, not here. Checks
(`tests/test_pose_density.py`):

- **Independent quadrature.** `scipy.integrate.nquad` over the ZYZ chart
  (`sin beta d beta d gamma / (4 pi)`, QUADPACK, independent of the classes'
  Gauss-Legendre constants): random, column (`0.5 deg`), plate (`0.5 deg`),
  parry (`1 / 1 deg`), lowitz (`40 / 1 deg`) and a wide-roll case
  (`roll_std 60 deg`) all give `1` to `1e-6`.
- **Independent Haar Monte Carlo** (uniform unit quaternions): parry with
  `4e6` samples `1.017 +- 0.029`, lowitz with `1e6` samples `0.986 +- 0.016`
  (the variance of the narrow-narrow parry density is `~ 1 / (sigma_roll
  sigma_zenith) ~ 3.3e3` in radians, so the Monte Carlo bound is loose by
  design and the tight check is the quadrature one).
- **Reductions.** Plate at `mean = 0` reproduces a fine trapezoid of `int_0
  g sin theta`; the parry density is the column density times a spin factor
  whose average over a flat spin is `1` (`1e-9`); the spin factor is
  periodic (`psi` and `psi + 2 pi` agree).

## 4. Provenance schema

`src/lumice_integral/pose_density_provenance.py::pose_density_provenance(family,
**parameters)` is the one writer of the `pose_density` block
(`canonical_scene.canonical_fixture_metadata` flat; `strip_io.scene_block`
wraps it in its `{"value": ..., "provenance": "canonical-new"}` envelope like
every other field there). The parameters recorded are the resolved ones, from
the same `resolve_pose_density_parameters` the factory uses, so what is
recorded is what was built.

| Family | Block |
|---|---|
| column (unchanged keys, new trailing `family`) | `{"model": "zenith-gaussian column", "zenith_mean_deg": 90.0, "zenith_std_deg": 0.5, "family": "column"}` |
| plate | `{"model": "zenith-gaussian plate", "zenith_mean_deg": 0.0, "zenith_std_deg": ..., "family": "plate"}` |
| random | `{"model": "haar-uniform random", "family": "random"}` |
| parry | `{"model": "zenith-roll-gaussian parry", "zenith_mean_deg": 90.0, "zenith_std_deg": ..., "roll_mean_deg": 0.0, "roll_std_deg": ..., "family": "parry"}` |
| lowitz | `{"model": "zenith-roll-gaussian lowitz", "zenith_mean_deg": 0.0, ...same keys as parry..., "family": "lowitz"}` |

Archived strips (`lumice-integral.strip/v2`) read unchanged: the three
historical column keys keep their order and values (two golden tests, one
per writer). No renderer CLI takes a family yet (`render_ch06_strip.py` stays
column-only); `canonical_strip_scene(pose_density=...)` is the diagnostic
entry point.

## 5. Diagnostics on the canonical `3-5` scene

Generated by

    uv run --with matplotlib python scripts/compare_pose_density_families.py \
        --output-dir scratchpad/task-pose-density-families/artifacts

(`pose_density_families.png` / `.json`, git-ignored; ~1 min on a laptop with
the default 400k-sample prescan and 19 profile rows). Families at the
widths of section 1's table (column and plate at the canonical `0.5 deg`).

### 5.1 Column-126 profile: only column and random carry mass

`render_pixel` on column `126`, rows `150-600` every `25`, per family
(figure, top-left, log scale):

| row | random | plate | column | parry | lowitz |
|---|---|---|---|---|---|
| 150 | `0.0927` | `0` | `3.23` | `0` | `0` |
| 300 | `0.0289` | `0` | `1.82` | `0` | `0` |
| 450 | `0.0102` | `0` | `0.832` | `0` | `0` |
| 600 | `0.00361` | `0` | `0.138` | `0` | `0` |

Plate, parry and lowitz are *exactly* zero at every rendered pixel. This is
not a solver failure: the five families share the fibers (same component
count, pose count and arclength — asserted by
`tests/test_strip_pixel.py::test_every_family_renders_the_diagnostic_pixels_on_the_same_fibers`),
and on these fibers the c-axis zenith is `88-92 deg` and the roll is
`90-155 deg` — face `3` is a *side* face, as it must be for the `3-5`
tangent-arc path of a horizontal column. A plate density (zenith about `0`)
and the roll-locked densities (face `3` on *top*) are `exp(-4000)` there and
underflow. The same test pins the three zeros as a regression guard of the
roll reference of section 2.

### 5.2 Landing map: where each family's `3-5` light goes

The prescan's Haar samples of `3-5` (64k valid of 400k) weighted by each
family's `rho_pose` and binned by outgoing sky direction, with the strip's
field of view outlined (figure, five panels). Fraction of a family's `3-5`
weight that lands inside the strip, and where the rest goes (elevation /
azimuth percentiles 1/50/99 of the significant samples):

| Family | inside strip | elevation (deg) | azimuth (deg) | reading |
|---|---|---|---|---|
| random | `3.2 %` | `-24 / 13 / 53` | `-41 / 0 / 41` | the full `22 deg` ring |
| column | `39.8 %` | `-27 / 32 / 55` | `-40 / 0 / 37` | lower tangent arc (in the strip) and upper tangent arc |
| plate | `0` | `14 / 15 / 16` | `24 / 26 / 40` | the right parhelion (the left one is path `3-7`) |
| parry | `0` | `30 / 45 / 48` | `-40 / -1 / 36` | the upper suncave Parry arc |
| lowitz | `0` | `15 / 39 / 56` | `-35 / 10 / 39` | Lowitz arcs above the sun |

So the ch06 strip (the lower tangent-arc region below the sun) is a column
strip on the labelled path `3-5`; the other families' `3-5` light is
elsewhere on the sky, and their contributions *to the strip* come from other
labelled paths of the same class (section 6).

### 5.3 Zenith-width sweep of the column family

`defect2_findings.md` section 4 (task strip-rerender-and-compare) reweighted
the traced integrand of the four probe pixels by `rho(std') / rho(0.5 deg)`
with an independent `scipy.integrate.quad` Gaussian. The same sweep through
the production classes (`scripts/compare_pose_density_families.py::zenith_width_sensitivity`,
pinned cell by cell to `1.5 %` by `tests/test_pose_density_zenith_sensitivity.py`):

| row | `0.25 deg` | `0.5 deg` | `1 deg` | `2 deg` | section-4 values |
|---|---|---|---|---|---|
| 150 | `1.0018` | `1` | `0.9208` | `0.5953` | `1.001 / 1 / 0.921 / 0.595` |
| 300 | `1.1489` | `1` | `0.6512` | `0.3527` | `1.149 / 1 / 0.651 / 0.353` |
| 450 | `1.5263` | `1` | `0.5448` | `0.2788` | `1.526 / 1 / 0.545 / 0.279` |
| 600 | `1.0787` | `1` | `0.8284` | `0.5355` | `1.078 / 1 / 0.828 / 0.536` |

Largest deviation `0.08 %`. The `0.5 deg` of the canonical scene is now a
parameter of `build_pose_density("column", zenith_std_deg=...)`, and the
table's conclusion stands: the width cannot produce the `2.6-4x` of defect 2.

## 6. Known gaps (recorded, not planned here)

1. **One labelled path, one locked peak.** The roll-locked density has a
   single peak, `roll = 0` = face `3` up. The prism's `C6` rotations about
   the c axis (Lumice's `P` symmetry, `x6`) and the mirror through the
   vertical plane containing the c axis (`D`) map the labelled path `3-5` to
   other labelled paths (`4-6`, ..., and `3-7`), each of which is "face `k`
   up" for its own `k` under the same physical Parry orientation. A physical
   Parry (or plate, or Lowitz) image of the *class* is the orbit sum of task
   `path-class-rendering-unit` (roadmap section 3.5 item 3), not a second
   peak in this density: the density is right for its labelled path, and
   section 5.2 shows which sky region that path lights. The group-level
   treatment (convolution of the pose law with the crystal's point group)
   belongs to the writing project's theory chapter.
2. **Single-period vs wrapped roll Gaussian** (section 1): differs only for
   `roll_std` near the period; no locked family is there.
3. **Lowitz zenith is Gaussian only** (no `zigzag`).
4. **Gimbal poles**: `c_axis_roll` is undefined at zenith `0 / pi`; the
   Lowitz zenith mean *is* the pole, but its `40 deg` width puts negligible
   mass in the singular neighbourhood and the spin factor there multiplies a
   finite zenith factor; the pixel test of section 5.1 covers the diagnostic
   fibers only, not every fiber of the image.
5. **No production CLI parameter** for the family (`render_ch06_strip.py`);
   the full-image family renders wait for the per-pixel cost work and the
   class-level renderer.
