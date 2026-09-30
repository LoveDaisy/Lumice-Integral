# Weight kinks and the colour criterion: module C spec draft

Status: implemented in LI (task `chromatic-weight-kink-diagnostic`, #49,
2026-09-30), JAX authoritative. This is the spec input for module C of the
shared library (wave 3, [overview.md](overview.md) §5.3) and for Analyze
function 3's preset points and mechanism labels. The Ice Halo side
(`doc/raypath-analysis.md` §5.1) cites this file. Modules A and B do not
change: no per-pose field is added, and module B's `T` already contains the
partial reflection's `R`.

## 1. Three kinds of critical line

For one face sequence $P$ the field layer (`lumice_integral.dp_field`) has
three kinds of critical structure on the body-frame sphere:

| kind | what it is | LI | what it does to the image |
|---|---|---|---|
| 1 | critical points and values of $D_P$ in $U_P$ (minima, maxima, saddles, the slab set) | `DPField.interior_critical_points` | caustics, the edges of ring halos |
| 2 | the boundary $\partial U_P$: zero sets of the gates (entry and exit incidence cosines and Snell discriminants, internal incidence cosines) | `DPField.boundary` | the support of the path ends |
| 3 | **weight kinks** $C_k = \{\mathrm{disc}_k = 0\}$: the TIR onset of internal reflection $k$ | `DPField.weight_kinks` | the Fresnel weight $R_k$ kinks: 1 on the total side, continuous with an unbounded normal derivative on the partial side |

Kind 3 is not part of $\partial U_P$ or of the topology certificate: a
partial reflection keeps the pose in the domain. Kind 2 depends on $n$ only
through the gates that refract (exit Snell, and an internal incidence cosine
whose unfolded normal is not orthogonal to the entry normal). Kind 3 always
depends on $n$, and on a slab path it is the **only** thing that does,
because the direction map of a slab does not disperse.

**Tracing $C_k$** (`dp_field.weight_kink`). With
$\mathbf m_k = R_{k-1}^{\mathsf T}\mathbf n_k$ the unfolded incidence normal
of step $k$:

- $\mathbf m_k\cdot\mathbf n_a = 0$: the entry refraction keeps the
  tangential component, so $\mathrm{disc}_k = n^2 - 1 - (\mathbf m_k\cdot\mathbf u)^2$
  and $C_k$ is the small circle $\mathbf m_k\cdot\mathbf u = -\sqrt{n^2-1}$.
  It is clipped to $U_P$ by bisection. Closed form, residual `< 1e-14`.
  This covers `3-1-6`, `1-3-2`, `3-1-5` and `1-3-5-2`, but not only slabs.
- otherwise: predictor-corrector on the zero set with the boundary walker's
  steppers, both ways from lattice seeds, until a gate stops it or the
  walk closes. The arcs found are not certified to be all of $C_k$.
  `3-7-5`, `3-5-1-7` and `3-5-6-7-3` have residual `< 1e-12`, and the
  Liljequist onset maximum 153.0697° of `ch10_verdicts` is on the walked
  arcs.

On a single-mirror slab (`3-1-6`, `1-3-2`), $D_P = 2\arcsin|\mathbf m\cdot\mathbf u|$,
so the whole of $C_k$ lies on one level, $D = 2\arcsin\sqrt{n^2-1}$
(114.61° / 115.61° / 117.97° at $n$ = 1.307 / 1.31 / 1.317). Under random
orientation this is the rim of a dark hole around the antisolar point with
radius $180° - D$ (65.4° / 64.4° / 62.0°). Inside the hole the path still
exists but reflects only partially.

## 2. The criterion (`lumice_integral.chromatic`)

Every verdict compares two indices, `N_RED = 1.307` and `N_BLUE = 1.317`
(the ends of Lumice's pool; both are parameters).

**Colour sign.** On a kink, $\partial\,\mathrm{disc}_k/\partial n$ at fixed
$\mathbf u$ (JAX AD). A positive value means blue reflects totally on a
larger set, so the energy moves toward blue. For an internal reflection it
is positive at every point of every kink found (5 paths, checked also by
counting 2e4 directions at both indices): **an internal-reflection kink can
only make blue**. A gate can have either sign (the exit Snell gate of
`3-1-5` is red, the internal incidence gate of `3-7-5` is blue).

**Random orientation** (`diagnose(crystal, faces)`). The image is a function
of $\delta = D$ alone. One `ChromaticFeature` is produced per kink, and one
per gate that bounds $U_P$ at both indices and moves with $n$:

| field | meaning |
|---|---|
| `kind` | `edge` (a kink) or `gate_edge` (a moving gate) |
| `source` | the margin name |
| `color`, `positive_fraction` | sign of $\partial m/\partial n$ on the line (fraction of points with $> 0$) |
| `delta_red`, `delta_blue` | median $D$ of the line at each index: its $\delta$ |
| `shift` $\Delta$ | median over the blue line of $D_b(\mathbf u) - D_r(\text{nearest red point})$ |
| `spread` $\sigma$ | $\max D - \min D$ on the line (the larger of the two indices) |
| `direction_dispersion` | median $\lvert\partial D_P/\partial n\rvert\,(n_b - n_r)$ on the line (0 for a slab) |
| `contrast` | edge: $1 - R$ of the disfavoured colour on the favoured colour's kink; gate: the favoured colour's weight on the other colour's gate, relative to its median in $U_P$ |
| `weight`, `lit_fraction` | median $A\,T$ on the favoured line, and the fraction of its points with $A\,T > 0$ |
| `visible` | $\lvert\Delta\rvert \ge 0.5°$ and $\sigma \le \lvert\Delta\rvert$ and `lit_fraction` $> 0$ |

The rule for `visible` comes from two scales:

- The solar disc smears every sky feature over about 0.5°
  (`EDGE_MIN_SHIFT_RAD`), so a smaller shift cannot be seen.
- A line whose own image in $D$ is wider than its red/blue separation
  (`EDGE_SPREAD_PER_SHIFT = 1`) has red and blue images that overlap more
  than they separate.

A line with $\sigma > \lvert\Delta\rvert$ still makes a slope corner of the
profile at its $D$ extrema, as the Liljequist point does, but not a coloured
step.

The path's `kind`, `color`, `visible` and `position` come from the
dominant feature: visible features first, then the largest
$\lvert\Delta\rvert/\sigma$.

**Oriented crystals** (`diagnose_class(..., PlateFamily(sun_altitude_deg))`).
When every pose of a class lands on one sky spot at both indices, the colour
is a **tint** of the spot. It is measured on a sample of the family as the
ratio of blue to red weighted power, $\sum_{\text{members}} A\,T$.

- A tint is declared only if the red and blue outgoing directions of the
  same pose are within 0.5° (`TintMetrics.direction_dispersion`).
- A dispersive class (`3-5`, `1-3`) spreads its colours like a spectrum.
  Its ratio (up to 1.21 / 0.88) is reported, but the verdict is `none` with
  a note.

The tint threshold `TINT_RATIO_MIN = 1.10` was frozen on calibration classes
that are not acceptance fixtures: non-dispersive classes whose reflections
stay total at both indices (`1-3-4-2`, `3-5-6-8`) on three other crystals at
sun altitudes of 5° and 20°. These stay within 0.045 of 1, which is plain
Fresnel dispersion of entry and exit; the threshold is twice the rounded-up
bound. The sky position of an oriented kink is not traced (scope of #49, a
follow-up).

**Kind and colour values.**

| kind | color | when |
|---|---|---|
| `edge` | `blue` (`red` is allowed by the code, never found) | a kink feature dominates (visible or not) |
| `gate_edge` | `red` / `blue` | a moving gate dominates |
| `tint` | `blue` / `red` | plate class, non-dispersive, ratio outside $[1/1.1, 1.1]$ |
| `none` | `white` | plate class, non-dispersive, ratio inside |
| `none` | `none` | no feature / not lit / dispersive class (with a note) |

**Classes** (`diagnose_class`). The input is a representative. Its class is
Lumice's filter orbit (`PBD`, L1: `symmetry.reflection_group.pbd_orbit`,
[conventions.md](conventions.md) #21). The literal sequence is never used
alone: on the rhombic plate `[1.5, 1, 1, 1.5, 1, 1]`, `3-5-6-8` itself is
geometrically impossible, while `4-8-7-5`, `5-7-8-4`, `7-5-4-8` and
`8-4-5-7` carry the class.

- `lit_members[red|blue]`: the members with positive $A\,T$ on the family's
  sample at that index. This is a sampled verdict, recomputed per index,
  not the `n = 1.31` gates of `geometry.feasibility`.
- Under random orientation, one member per `G_true` orbit is diagnosed
  (congruent fields), and the class verdict is the dominant feature over
  those members.
- $A\,T$ is one kernel, `chromatic.weighted_power` (entry measure at the
  same $n$ times the path power with every internal $R_k$). The plate sum
  and the random-orientation feature weights both go through it.

## 3. Evidence

- Fixtures (`tests/test_chromatic.py`, `tests/test_dp_field_weight_kink.py`):
  - `3-1-6` / `1-3-2`: `edge`, `blue`, visible, at
    $\delta = 2\arcsin\sqrt{n_b^2-1}$ (to 1e-12), shift 3.354°, spread 0.
  - `3-1-5` exit gate: `red`, not visible ($\sigma$ = 108.7° against
    $\Delta$ = −0.33°).
  - plate, sun 9°, `[1.5, 1, 1, 1.5, 1, 1]`, summed over the class:
    `1-3-5-2` is `tint` `blue` (1.492 at h/a = 1, 1.528 at h/a = 2);
    `1-3-4-2` is white (0.965 / 1.010); `3-5-6-8` is white (1.029 / 1.026).
    These equal the owner's probe `probe_120_cls` to the printed digits.
- Band-sum render (`scripts/verify_chromatic_kink.py`, the class under
  `G_true`, $N = 10^7$, red / blue monochrome plus D65 at $M = 5$, a
  0.17°-per-pixel column across $\delta$ 100–150°):
  - `3-1-6`: the profile drops at 114.70° / 118.00°, against 114.61° /
    117.97° predicted (0.5 / 0.2 pixel). In the rim, blue/red is 2.03
    against 0.995 on the total side, and the $z$ chromaticity is 0.350
    against 0.274.
  - `3-1-5`: no red edge. The only sharp colour step is a blue one at
    131°, inside the kink's predicted interval.

**Finding that differs from the task's expectation.** The issue expected
"`3-1-5`: no visible colour edge", with the red gate in mind. The gate is
indeed invisible. But `3-1-5`'s own kink is visible by the criterion:

- `3-1-5`'s kink is a closed-form circle that is not on one $D$ level.
  $\sigma$ is 4.9° against $\Delta$ = 7.3°, because the kink moves and the
  direction map disperses by 1.1° as well.
- The render shows the corresponding blue band at $\delta$ ≈ 130–142°,
  with blue/red up to 2.3, at 1–10 % of the profile's peak.
- The path verdict is therefore `edge` / `blue` / visible. Whether that
  band is noticeable next to other paths is a brightness question, outside
  the criterion.

## 4. What module C would carry

- **Inputs:** crystal (face distances), representative face sequence,
  family (random, or plate with sun altitude and tilt width), the index
  pair.
- **Outputs:** the verdict record above, per class and per diagnosed
  member, plus the kink curves (points, $D$ values, the gate at each arc
  end, `closed`).
- **Computation it needs beyond modules A/B:** the margin vector and its
  derivatives in $\mathbf u$ and in $n$ (the `Jet2` template of
  [overview.md](overview.md) §5.3 extends to $\partial/\partial n$ by one
  more dual direction), the closed-form circle, the zero-set walker (the
  boundary walker's steppers), the boundary walk for the gate features,
  and $A\,T$ on samples (module B already has it).

**Not in this draft (follow-ups):**

- the sky curve of a kink for oriented crystals;
- dispersion edges of kind 1 (the red inner edge of the 22° halo is the
  critical value moving with $n$, `focusing.wavelength_critical_table`);
- a brightness-relative visibility, weighing a feature against the other
  paths at the same $\delta$.
