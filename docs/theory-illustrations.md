# Theory Illustration Inventory and Production Contract

This document is the tracked authority for what the explanatory figures in the
Lumice Integral theory documents are meant to explain, where they belong, and
how they should be produced. It records the inventory made on 2026-10-11 from
the English and Chinese overview, Phase I, and Phase II documents. It does not
claim that one inventory proves conceptual completeness, and it does not replace
an author's visual judgment.

The first production batch is fixed at six figures whose content and overall
layout were accepted as prototypes on 2026-10-10. Later figures are separate
proposals with explicit prerequisites; they are not part of that acceptance and
are not started by this document.

## 1. Scope, authority, and sources

The inventory covers the explanatory bodies of:

- [overview.md](overview.md) and [overview_zh.md](overview_zh.md), especially
  “Two ways to render a halo” and “Phase I and Phase II”;
- [phase1.md](phase1.md) and [phase1_zh.md](phase1_zh.md), especially “The
  formulation”, “Tracing one fibre”, and the limits of procedural component
  discovery;
- [phase2.md](phase2.md) and [phase2_zh.md](phase2_zh.md), from the
  $\mathrm{SO}(3)\to S^2$ reduction through both quadratures, topology,
  symmetry, and the mechanism verdicts;
- the normative definitions in [conventions.md](conventions.md),
  [phase1-math-contract.md](phase1-math-contract.md), and
  [band-sum-contract.md](band-sum-contract.md);
- the data and provenance boundaries in
  [ch06-reference-fixture.md](ch06-reference-fixture.md), the pose-family
  distinctions in [ch11-pose-density-families.md](ch11-pose-density-families.md),
  and the three critical structures in
  [chromatic-module-c.md](chromatic-module-c.md).

Long measured appendices were checked as evidence and data sources. A diagnostic
record does not become a figure requirement merely because data for it exists.

The accepted prototypes came from the local, git-ignored discussion record
`scratchpad/discussion-theory-figures/ILLUSTRATION_GUIDE.md`, dated 2026-10-10.
All six named PNGs were inspected directly. Their shared calculation is the
canonical `3-5` scene at LI revision
`98b4a8c7ec70e519f3744279840c2814f871ade7`: hexagonal column $h/a=2$,
$n=1.31$, Sun altitude $15^\circ$, pixel `(row=150, column=150)`, and column
pose density $\sigma=0.5^\circ$. That revision is a provenance anchor, not a
request to keep old code paths alive.

The discussion record remains historical evidence. A clean checkout must be
able to understand every decision below without it, and future production
commands must not require it.

### 1.1 Decision rule

A figure earns a place only when it makes a relationship materially easier to
see than prose and equations alone. The useful relationships in this inventory
are spatial correspondence, continuous variation, constraint departure and
recovery, boundary changes, and competing physical factors.

For every proposed figure, ask:

1. What single reader question does it answer?
2. What is fixed, and what varies?
3. What intuition is added beyond the adjacent text?
4. Is each panel necessary to that answer?
5. Is the underlying scientific quantity already authoritative and observable?
6. Can numerical validity be checked independently of visual quality?

A source module, a document section, or an available array is not by itself a
reason to draw a figure. Explanatory priority and data readiness are recorded as
separate dimensions below.

### 1.2 Language boundary

English documents are authoritative. The same static image is shared by the
English and Chinese prose: mathematical symbols and short English labels are the
default inside the image, while each document supplies its own caption. Two
language-specific image files are not maintained unless a later task establishes
a concrete need.

## 2. First batch: the six accepted figures

The stable IDs below are semantic; the temporary prototype numbers `00`–`03`
do not define reading order. All six canonical PNGs are tracked production
assets; their scripts, caches, and batch commands are recorded in section 6.4.

### 2.1 `world-body-frames`

- **Reader question:** How can a fixed Sun and changing crystal pose in the
  world frame become a fixed crystal and changing direction $\mathbf u$ in the
  body frame?
- **Primary prose location:** immediately after the definition
  $\mathbf u=R^{-1}\hat{\mathbf s}$ in [phase2.md](phase2.md), “From
  $\mathrm{SO}(3)$ to $S^2$”; the corresponding paragraph in
  [phase2_zh.md](phase2_zh.md), “从 $\mathrm{SO}(3)$ 到 $S^2$”.
- **Secondary references:** [overview.md](overview.md), “Phase I and Phase II”,
  and its Chinese counterpart, before the first use of the body-frame sphere.
- **Fixed / varying:** each column is one physical configuration. In the world
  row, $\hat{\mathbf s}$ is fixed and $R$ varies; in the body row, the crystal
  is fixed and $\mathbf u=R^{\mathsf T}\hat{\mathbf s}$ varies.
- **Added intuition:** the reduction is a coordinate change, not a different
  optical scene. The four numbered states establish the correspondence reused
  by the level-set and fibre figures.
- **Prerequisite and caption boundary:** state that $R$ maps body to world. The
  orange arrows point **toward the source**; propagation is their negative.
  The source dots indicate directions, not finite source distances.
- **Cross-figure path:** precedes `deviation-level-set`; its numbered poses also
  match `pose-fiber` and `fiber-contributions`.
- **Existing evidence:** accepted prototype
  `00-world-body-frames-v1.png`; reconstruction errors were
  `6.7e-16` in world direction and `1.1e-15` in body geometry.
- **Production input:** accepted poses from the versioned fibre data, the
  canonical Sun direction, and the canonical crystal. Crystal drawing and
  coordinate transforms are presentation work; no optical quantity is
  recomputed in the drawing layer.
- **Canonical asset:** [world-body-frames.png](figures/theory/world-body-frames.png).
  It is implemented by the first production slice described in section 6.4.

### 2.2 `deviation-level-set`

- **Reader question:** Why does the projection of a pose fibre onto the
  body-frame sphere lie on a level set of $D_P$?
- **Primary prose location:** after the $S^2$ fields and their invariance are
  introduced in [phase2.md](phase2.md), “From $\mathrm{SO}(3)$ to $S^2$”,
  before “One integral, two quadratures”; the same semantic location in
  [phase2_zh.md](phase2_zh.md).
- **Fixed / varying:** path, crystal, index, Sun, and target deviation
  $\delta$ are fixed. $\mathbf u$ moves along the projected fibre while
  $D_P(\mathbf u)=\delta$ stays fixed.
- **Added intuition:** the spherical field and its local orthographic chart are
  two views of the same field; shared colours, contour values, and labels
  `1`–`4` make the equality visible rather than merely asserted.
- **Prerequisite and caption boundary:** read `world-body-frames` first. The
  colour is deviation in degrees, not brightness. Grey is outside the optical
  formula domain $U_P$; it does not represent every point with zero finite-
  crystal weight. A deviation level fixes angular distance, not sky azimuth:
  a fixed sky point still requires the unique twist/pose lift.
- **Cross-figure path:** labels `1`–`4` refer to `pose-fiber`; the thick curve is
  that production fibre's projection, not a completeness-certified contour
  extraction.
- **Existing evidence:** accepted prototype `dp-field-level-set-v2.png`;
  projected-level residual `5.6e-16` rad; its thin background contours are grid
  illustrations and make no completeness claim.
- **Production input:** `DPField` for $U_P$ and $D_P$, plus the versioned fibre
  poses for the highlighted projection. The removed twist panel is not restored;
  the lift condition belongs in the caption and prose.
- **Canonical asset:** [deviation-level-set.png](figures/theory/deviation-level-set.png).
  It is produced by the shared six-figure batch in section 6.4.

### 2.3 `pose-fiber`

- **Reader question:** What is the one-dimensional inverse image
  $F_P^{-1}(\mathbf d)$, and how can changing crystal poses keep the incident
  and outgoing world directions fixed?
- **Primary prose location:** immediately after the inverse-image and coarea
  definition in [phase1.md](phase1.md), “The formulation”, and the corresponding
  part of [phase1_zh.md](phase1_zh.md), “表述”.
- **Secondary references:** [overview.md](overview.md), “Two ways to render a
  halo”, and its Chinese counterpart. The overview may reference this figure
  instead of introducing a separate forward-versus-inverse schematic.
- **Fixed / varying:** crystal, path `3-5`, incident propagation direction, and
  outgoing propagation direction $\mathbf d$ are fixed; $R$ and
  $\mathbf u=R^{\mathsf T}\hat{\mathbf s}$ vary continuously.
- **Added intuition:** one numbered state is simultaneously a point on the
  body-frame sphere, a point on a curve in local $\mathrm{SO}(3)$ coordinates,
  and a real in-crystal path with the same world directions. The representative
  entry point may differ between poses.
- **Prerequisite and caption boundary:** define $\mathbf u$ in one sentence so
  a Phase I reader need not read Phase II first. Orange arrows follow incident
  propagation here, unlike the source-pointing arrows in
  `world-body-frames`; blue follows outgoing propagation and the apparent sky
  point is $-\mathbf d$.
- **Cross-figure path:** supplies labels `1`–`4` to
  `deviation-level-set` and `fiber-contributions`; it is the full-component
  locator for `predictor-corrector`.
- **Existing evidence:** accepted prototype `01-fiber-overview-v3.png`; four
  representative rays were checked against Snell, face membership, and fixed
  world directions to a maximum direction error `3.9e-16`.
- **Scope limit:** this is one known regular closed component. It is not proof
  that every component of the pixel fibre has been found.
- **Production input:** versioned fibre geometry plus a narrow LI-owned export
  of representative ray segments. The historical prototype used
  `halo_notes.draw.raypath.search_raypath` and `verify_path` through an absolute
  Writing-Lab path; production must remove that runtime dependency rather than
  copy the whole drawing package.
- **Canonical asset:** [pose-fiber.png](figures/theory/pose-fiber.png).
  It is produced by the shared six-figure batch in section 6.4.

### 2.4 `predictor-corrector`

- **Reader question:** How does continuation move from one valid pose to the
  next while keeping the target sky point fixed?
- **Primary prose location:** after steps 2–5 in [phase1.md](phase1.md),
  “Tracing one fibre”, and after the corresponding list in
  [phase1_zh.md](phase1_zh.md), “追踪一条纤维”.
- **Fixed / varying:** the target direction and path are fixed. The current pose
  `A`, tangent prediction `P`, corrected pose `B`, and later accepted states
  vary along one fibre component.
- **Added intuition:** the full fibre locates a short segment; the local pose
  view shows predict–correct–repeat; the separate sky tangent chart shows the
  predicted miss and corrected return to the target.
- **Prerequisite and caption boundary:** read `pose-fiber` first. Local
  $\mathrm{SO}(3)$ coordinates and sky angular offsets are different spaces.
  Euclidean distance in the plotted rotation-vector chart is not fibre
  arclength. Production correction uses two outgoing residuals plus a phase
  condition; the orange/green marks are not a single $D_P$-gradient step.
- **Cross-figure path:** the left panel is the same component as `pose-fiber`;
  it does not add a second fibre or make a component-completeness claim.
- **Existing evidence:** accepted prototype `02-predictor-corrector-v5.png`;
  the recorded three production steps used `0.04` rad predictions, the first
  predicted sky miss was `0.0478°`, accepted outgoing residuals were at machine
  precision, and replay of the first corrected pose was exact.
- **Production input and boundary:** process data need a narrow, stable export.
  The prototype monkey-patched private continuation functions and called
  `_correct_trial`; those calls are historical evidence, not an interface to
  preserve. The production task must either export the already-observable
  accepted predictor/corrector record or add only the smallest diagnostic hook
  required by this figure, without changing continuation semantics.
- **Canonical asset:** [predictor-corrector.png](figures/theory/predictor-corrector.png).
  It is produced by the shared six-figure batch in section 6.4.

### 2.5 `fiber-contributions`

- **Reader question:** Which parts of one fibre carry the light, and how do the
  named factors combine?
- **Primary prose location:** after the coarea formula and factor definitions in
  [phase1.md](phase1.md), “The formulation”, and the corresponding paragraph in
  [phase1_zh.md](phase1_zh.md).
- **Fixed / varying:** scene, path, target pixel, and one closed component are
  fixed. Arclength $s$ varies; $\rho$, $A/a^2$, $T$, $K=1/(J_\perp+\epsilon)$,
  and their product vary along it.
- **Added intuition:** one shared logarithmic axis preserves actual relative
  magnitudes and makes the product traceable to its factors. In this fixture
  $\rho$ spans almost six orders of magnitude while $T$ changes by only about
  `0.7%`; the near-flat curve is therefore information, not a plotting defect.
- **Prerequisite and caption boundary:** define every factor and keep the Haar
  constant $1/(8\pi^2)$ separate from $J_\perp$. The displayed
  $\epsilon=10^{-6}$ is this calculation's regularisation, not a replacement
  for the unregularised contract. No factor is individually peak-normalised.
- **Cross-figure path:** numbered positions refer back to `pose-fiber`.
  `entry-area` expands one geometric kink in $A$.
- **Existing evidence:** accepted prototype `03-fiber-contributions-v2.png`;
  product assembly relative error `3.8e-16`; numerical arclength from production
  $ds/dt$; no independent peak normalisation.
- **Scope limit:** $s=0$ and $s=L$ are the same pose, so high contribution at
  both ends may be one region cut by the chosen start. One integrated component
  remains partial until component completeness is established.
- **Production input:** `export_fiber_figure_data` schema
  `lumice-integral.figure-data/v3`, including poses, cumulative arclength,
  named weights, unregularised `normal_jacobian`, quadrature metadata, and the
  integrand. Plotting must not reimplement any factor.
- **Canonical asset:** [fiber-contributions.png](figures/theory/fiber-contributions.png).
  It is produced by the shared six-figure batch in section 6.4.

### 2.6 `entry-area`

- **Reader question:** Why does the effective entry area have a kink even when
  the fibre and optical map remain regular?
- **Primary prose location:** in [phase1.md](phase1.md), “The formulation”,
  after `fiber-contributions` as a mechanism expansion of $A_P$; the same
  semantic location in [phase1_zh.md](phase1_zh.md).
- **Fixed / varying:** path, target pixel, and projection scale are fixed.
  Arclength moves through states `A` (before), `B` (boundary alignment), and
  `C` (after); the active upper and lower limits of the projected-face
  intersection exchange at `B`.
- **Added intuition:** the linear $A(s)$ plot and three equal-scale orthogonal
  projections show that a slope change comes from a clipping-boundary switch,
  not from a halo-map critical point.
- **Prerequisite and caption boundary:** read `fiber-contributions` first.
  The green overlap is $A_\perp$ in the plane perpendicular to the internal ray;
  the incident-beam area is
  $A=A_\perp\cos\theta_i/\cos\theta_t$. State `B` is not a Jacobian
  critical point. The three panels share a physical scale and are not
  perspective views.
- **Existing evidence:** accepted prototype `entry-area-geometry-v1.png`;
  event arclength `1.909876` rad; one-sided slopes converge to equal and
  opposite `0.621` $a^2$/rad; maximum difference from production
  `entry_measure` was zero.
- **Production input and boundary:** versioned fibre poses plus the production
  corridor/projection geometry. The drawing layer may assemble polygons and
  labels but must not independently decide path feasibility or recompute a
  competing entry area.
- **Canonical asset:** [entry-area.png](figures/theory/entry-area.png).
  It is produced by the shared six-figure batch in section 6.4.

### 2.7 First-batch reading paths

The figures are referenced rather than duplicated.

- A Phase I reader can follow `pose-fiber` → `fiber-contributions` →
  `entry-area`, then `predictor-corrector`. The one-sentence definition of
  $\mathbf u$ in `pose-fiber` prevents a dependency on reading Phase II first.
- A Phase II reader can follow `world-body-frames` → `deviation-level-set`,
  then use `pose-fiber` only when the lift back to full poses matters.
- The overview references `pose-fiber` for inverse-image integration and
  `world-body-frames` for the Phase II reduction. It does not maintain smaller
  copies of either image.

Three figures near the Phase I opening are deliberate but should not form one
unbroken wall. `pose-fiber` follows the inverse-image definition;
`fiber-contributions` follows the coarea factor paragraph; `entry-area` follows
as the optional geometric expansion. `predictor-corrector` belongs in the next
section after the algorithm steps.

## 3. Additional reader questions and decisions

Priority is explanatory value, not implementation order. Data readiness records
whether authoritative inputs exist; it is not permission to bypass a scientific
or author decision.

| Stable ID / reader question | Decision and priority | Data readiness and prerequisite | Proposed deliverable and acceptance |
|---|---|---|---|
| `fiber-component-topology`: how do closed loops, event-truncated open arcs, and multiple components differ, and why is one loop not completeness? | **Near-term, P0.** This is the largest limitation not visible in the six regular-component figures. | Phase I component records and Phase II interval/contour certificates exist. Select one real closed, one real open, and one multi-component fixture without implying they share one path; fixture selection and the comparison statement need a scientific review. | A small topology atlas tied to the contract's component kinds, with named end events and procedural-versus-certified status. Verify each curve against production outputs and its reported component count; no schematic component may be presented as measured. |
| `quadrature-equivalence`: how are the Phase I fibre integral, Phase II contour integral, and finite-band event sum three quadratures of one measure? | **Near-term, P0.** The equations are correct but their shared object is difficult to reconstruct from prose. | The closed Jacobian identity, contour geometry, band events, and same-pixel comparisons are already in production and tested. No new physics is required. | One mechanism diagram plus one restrained same-pixel numerical inset: $\mathrm{SO}(3)$ fibre → $S^2$ contour → finite deviation band. Verify constants from the existing implementations, use a shared pixel model, and report residual/convergence rather than call the match exact. |
| `domain-vs-finite-support`: why can a level set exist in $U_P$ while finite-crystal weight is zero? | **Near-term, P1.** It prevents a recurring confusion between formula domain, contour existence, and radiance support. | The pyramid `13-15-26-28` `gated_out` fixtures and contour outputs exist. A compact scene and target deviations must be chosen. | Overlay one level set, $U_P$, and the $A_PT_P>0$ subset using distinct patterns, not merely two similar colours. Acceptance requires the same points to pass `DPField` while production weight reports zero; do not generalise one fixture to all crystals. |
| `critical-structure-mechanisms`: how do a $D_P$ critical set, a domain boundary, and an internal-reflection weight kink differ? | **Near-term, P0**, after the first batch. These three structures can all look like an image edge but have different mathematics. | `DPField`, boundary loops, weight kinks, and the module-C verdicts exist. Closed-form slab fixtures have certified geometry; general walked kink arcs are not certified complete. | A body-sphere mechanism map with three labelled structures and a corresponding one-dimensional profile. Use a fixture whose completeness scope is explicit. Verify residuals, colour sign, and domain membership separately; never label a kink as $\partial U_P$. |
| `focusing-mechanisms`: why can a bright or narrow feature come from Jacobian focusing, a pose-family dimension collapse, or a window kink? | **Near-term, P1.** It is the conceptual bridge from Phase II to `focusing.classify` and pose families. | Production labels and verdict arrays exist for finite jump, fold, slab, and family-pinned examples. The author must choose the smallest set of examples that remains readable. | A comparison with the same observable profile but visibly different controlling objects. Acceptance checks the production mechanism labels and the stated scaling with pose-family width; avoid an encyclopedic grid of every label. |
| `pose-family-reweighting`: how can the same geometric fibre produce different halos when only $\rho$ changes? | **Later, P1**, likely with the chapter-11 table rather than the six-figure batch. | The five densities and diagnostic landing data exist, but the old landing-map probe used a retired prescan and a single labelled path; a class-level, current-data scene must be selected. | A shared-fibre / changed-weight figure or a small class-level landing comparison. Recompute from current production data, identify L1/L2 symmetry scope, and do not infer a physical family image from one labelled path. |
| `pixel-model-near-caustic`: how do point value, deviation-band average, and finite-source blur change a sharp edge? | **Conditional, P2.** Useful when finite-sun or pixel-averaging work becomes a live requirement; not needed to explain the current point-source contract. | Point and band models exist; finite-sun averaging does not. The solar-disc blur must not be faked by relabelling a pixel average. | When triggered, show the same production fixture under separately named averaging operations. Verify each integral and kernel; until then, keep the limitation in prose. |
| `divergent-light-geometry`: how does one line of sight traverse a range of deviation angles for a nearby source? | **Conditional, P2.** The geometry is visually valuable, but the project has no scene model or independent oracle. | Section 9 of Phase II is a checked derivation only. Trigger on a writing need or an oracle/scene implementation. | A ray/source geometry figure and event-domain integral only after the scene contract exists. Marking a derivation as a measured result is a failure. |

### 3.1 Questions intentionally handled without another figure

- **Forward Monte Carlo versus inverse integration:** the overview equation and
  `pose-fiber` already expose the distinction. A second generic arrows diagram
  would add little unless user testing shows the present path is insufficient.
- **The twist needed to lift a $D_P$ contour to a fixed sky point:** a sentence
  in the `deviation-level-set` caption preserves the mathematical condition.
  The rejected twist thumbnail did not make its relation to the other panels
  visible.
- **Newton residual histories:** they are validation evidence, not the mechanism
  of continuation. Keep them in metadata/tests unless a numerical-analysis
  document asks a different question.
- **Cumulative contribution along a closed fibre:** it depends on the arbitrary
  start point and added little in the accepted fixture. The actual factor and
  product curves answer the current reader question.
- **Three nearly identical crystal poses beside the local walk:** the visible
  difference was too small. The local state and sky miss/correction are more
  informative.
- **Precomputation dependencies and ring invariance:** the tables and equations
  in Phase II already state them compactly. A new picture must wait for evidence
  that those representations fail readers.
- **Symmetry transport and $K_{\mathrm{eff}}$:** important for implementation and
  sampling semantics, but not currently a missing spatial relationship. It is
  better served by tests and the existing equations than by replicated event
  clouds that could suggest transported copies are independent samples.
- **A separate contract figure for $A_\perp$:** `entry-area` already gives the
  mechanism and conversion. The normative formula remains prose; no duplicate
  figure is needed merely because the contract has its own section.

## 4. Batch handoff

### B1 — accepted canonical six

Scope: exactly the six figures in section 2. Tasks 58.2 and 58.3 may improve
font, spacing, line weight, and export format while preserving each figure's
reader question, visible relationships, and scientific semantics. A change that
adds a new conceptual panel or alters a core relationship requires a new author
decision; it is not visual cleanup.

### B2 — near-term mechanism gaps

The top-level owner can form follow-up tasks from the P0/P1 rows in section 3,
preferably by shared scientific prerequisite rather than by equal figure counts:

1. component topology and completeness;
2. the equivalence of the three quadratures;
3. $U_P$ versus finite-crystal support;
4. critical structures and focusing mechanisms.

These are briefs, not created tasks. Each task should choose its fixtures before
drawing and state whether its result is certified, numerically observed, or
schematic. The B2 work does not block B1.

### B3 — trigger-based figures

Pose-family reweighting waits for a current class-level scene. Pixel/finite-source
averaging waits for the corresponding source model. Divergent light waits for a
scene contract or an independent oracle. Their trigger is the prerequisite, not
elapsed time or spare plotting capacity.

## 5. Stable drawing principles

The following principles are promoted from the 2026-10-10 prototype review.
They are stable across the six figures; exact colours, font sizes, and export
formats are still subject to visual review.

1. **One primary question per figure.** A related concept that needs a new
   variable or background gets prose or a separate figure.
2. **Name the space.** World direction, body-frame $S^2$, local
   $\mathrm{SO}(3)$ coordinates, and sky tangent offsets must not share an
   unlabeled coordinate system.
3. **State fixed and varying quantities.** Use the image, caption, and metadata
   together; do not fill the canvas with the full parameter record.
4. **Make correspondence redundant.** Reuse numbers, letters, shapes, and
   selected segments in addition to colour.
5. **Give arrows distinct semantics.** Source direction, ray propagation,
   prediction/correction, and view correspondence must be distinguishable by
   labels and local style, not colour alone.
6. **Share scales when comparison is the point.** Views of one $D_P$ field share
   a colour scale; projected polygons share a physical scale; factor curves
   preserve their actual relative amplitudes. Do not stretch data silently.
7. **Use the coordinate background the object needs.** Sphere grids can replace
   a large 3-D box; an independent pose curve benefits from axes and a light
   coordinate plane.
8. **Prefer layout to decorative routing.** Enlarge key objects and move labels
   before drawing long connection curves. A small crossing of secondary lines
   is better than a connector that becomes the main object.
9. **Use production arclength and areas.** Rotation-vector Euclidean distance is
   not $\mathrm{SO}(3)$ arclength. $A_\perp$ and incident-beam $A$ are distinct.
10. **Do not rename mechanisms by appearance.** An area clipping kink, a domain
    boundary, a $D_P$ critical point, and a pose-density concentration are
    different even when their image profiles look sharp.
11. **Separate numerical and visual acceptance.** Residuals, units, source
    revision, convergence, and completeness scope live in data/metadata and
    tests. Legibility at the actual document width and aesthetic judgment are
    checked separately by a human.
12. **Inspect the delivered size.** A label visible in a 16-inch prototype is
    not automatically legible when embedded in a document. Check mathematical
    subscripts, hollow markers, fine dashes, and faint grids in the real layout.

The prototype palette—white, dark blue-grey text, low-saturation blue/orange/
green/purple accents, and light grids—is a starting point, not a cross-figure
law assigning one physical meaning to each colour. Every figure must be readable
without relying on colour alone.

## 6. Lightweight production layout

The production layout is being populated by the six real figures, not by a
generic plotting platform. Task 58.2 implemented the first vertical slice as:

```text
scripts/illustrations/
  regen.py                       # the one explicit ordered step table
  world_body_frames.py           # public recompute/replot entry for the first figure
  _world_body_data.py            # figure-specific schema, production export and validation
  _crystal_view.py               # narrow projection and crystal presentation
  _style.py                      # constants already needed by the accepted six-figure batch

docs/figures/theory/
  world-body-frames.png          # canonical tracked asset

artifacts/theory-illustrations/
  canonical-fiber/               # rebuildable upstream figure-data/v3 export
  world-body-frames/             # rebuildable self-contained figure cache
```

The other five entries are added to this same table only when their real scripts
exist in task 58.3. There are no placeholder scripts, automatic discovery, empty
packages, or a second registry.

### 6.1 Command semantics

Each figure entry should have explicit input and output directories and two
non-overlapping operations:

- **replot:** read an existing internal data export, check the basic consumption
  format and file pairing, and write only the figure. Missing data fails clearly.
  It must not import a solver path that starts calculation as a side effect.
- **recompute:** call LI's production scientific interfaces, write numerical
  arrays plus metadata to the requested data directory, and stop. It must not
  silently overwrite the canonical document figure.
- **regen:** an explicit ordered table invokes recompute and replot as distinct
  steps. A dry-run lists what will execute, which steps are expensive, and what
  outputs they own.

The useful lesson from Writing-Lab is the explicit one-figure script, a small
shared style module only where duplication is real, and a visible regeneration
step table. Its `halo_notes.repo`, simulation wrappers, and drawing framework
are not copied and do not become runtime dependencies.

### 6.2 Scientific data boundaries

- `export_fiber_figure_data` is the authoritative boundary for fibre geometry,
  diagnostic arrays, named factors, cumulative arclength, and quadrature
  metadata. Its JSON/NPZ pair is versioned and records units and SHA-256.
- `export_verdict_figure_data` is a separate schema for chapter-10 mechanism
  verdicts. It is not a new version of the fibre schema.
- `DPField`, contour, geometry, optics, and weight modules own their scientific
  quantities. A plotting script may select, project, annotate, and compose
  those outputs; it must not recreate Snell/Fresnel equations, arclength,
  $J_\perp$, entry-area clipping, or Newton correction.
- The historical world/body and current pose-fibre prototypes inherited drawing
  and ray helpers from Writing-Lab through `sys.path`. The first dependency is
  now replaced by LI-owned narrow presentation code and a self-contained geometry
  record; `pose-fiber` retains the same migration requirement for task 58.3.
- The predictor/corrector prototype records private continuation calls. Formal
  production requires a narrow process-data boundary, not continued monkey-
  patching of `_correct_trial` or `_trial_newton_step_kernel`.
- Replotting has no Lumice dependency. Lumice remains an explicit validation
  oracle, never a hidden data provider for these six figures.

### 6.3 Dependencies and assets

Replotting needs Python 3.12, NumPy, and Matplotlib. Matplotlib should remain an
optional plotting dependency (for example `uv run --with matplotlib ...`), not
a production-solver requirement. Recompute may use the project's existing JAX,
NumPy, and SciPy stack because it invokes LI itself.

Small, stable, irreproducible inputs may be versioned when a later task justifies
them. Large caches remain under `artifacts/` and are rebuildable. The canonical
static figures referenced by the docs belong under `docs/figures/theory/`.
Whether SVG or PDF accompanies PNG is decided after checking the actual document
embedding; this inventory does not multiply formats pre-emptively.

### 6.4 Implemented B1 batch: all six canonical figures

`--data-dir` names the batch data root, not a figure leaf. Recompute first writes
one `canonical-fiber/` export and one `canonical-data/` shared cache, then builds
six self-contained leaf caches. It never writes a PNG. Replot reads only those
six leaves and writes the stable PNG names; it neither imports the solver nor
requires the shared parent caches. Missing, unreadable, or unsupported leaves
fail with an exact recompute command instead of silently calculating.

```bash
# Inspect the exact shared, leaf, and rendering steps without writing.
uv run python scripts/illustrations/regen.py --mode all \
  --data-dir artifacts/theory-illustrations \
  --output-dir docs/figures/theory --dry-run

# Scientific recompute only, then pure rendering only.
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  uv run python scripts/illustrations/regen.py --mode recompute \
  --data-dir artifacts/theory-illustrations \
  --output-dir docs/figures/theory
uv run --with matplotlib python scripts/illustrations/regen.py --mode replot \
  --data-dir artifacts/theory-illustrations \
  --output-dir docs/figures/theory

# One leaf and PNG; the necessary shared recompute is explicit.
uv run --with matplotlib python scripts/illustrations/regen.py --mode all \
  --only entry-area --data-dir artifacts/theory-illustrations \
  --output-dir /tmp/theory-figure-preview
```

The one explicit `regen.STEPS` table has these entries:

| Stable ID | Public entry | Leaf schema | PNG dimensions |
|---|---|---|---:|
| `world-body-frames` | `world_body_frames.py` | `lumice-integral.illustration.world-body-frames/v1` | 2400×1500 |
| `deviation-level-set` | `deviation_level_set.py` | `lumice-integral.illustration.deviation-level-set/v1` | 2175×1200 |
| `pose-fiber` | `pose_fiber.py` | `lumice-integral.illustration.pose-fiber/v1` | 2400×1500 |
| `predictor-corrector` | `predictor_corrector.py` | `lumice-integral.illustration.predictor-corrector/v1` | 2250×1200 |
| `fiber-contributions` | `fiber_contributions.py` | `lumice-integral.illustration.fiber-contributions/v1` | 2100×1140 |
| `entry-area` | `entry_area.py` | `lumice-integral.illustration.entry-area/v1` | 2100×1410 |

The shared schema is
`lumice-integral.illustration.canonical-data/v1`, separate from both the six
leaf schemas and upstream `lumice-integral.figure-data/v3`. It records the
accepted trajectory, the fixed 513-point presentation grid, the final
production quadrature grid, production retractions and $ds/dt$, numerical
arclength, named factors, unregularised $J_\perp$, geometry, conventions,
convergence evidence, component scope, source hashes, and provenance. The two
grids currently both contain 513 nodes, but their names and metadata remain
separate because presentation numbering and adaptive integration have different
contracts. Numbered poses remain `[0, 128, 256, 384]`; they are not relabelled as
equal-arclength quartiles.

Each leaf is an internal, rebuildable intermediate and copies exactly what its
renderer consumes. Loaders use `allow_pickle=False`, check JSON/NPZ identity and
SHA-256, require usable shapes and numeric types, and enforce only consumption
invariants. A loader does not re-certify SO(3), crystal topology, optics, or
quadrature after a synchronously written pair is read back. Scientific claims
are checked during recompute and in the slow production regression rather than
reimplemented as a second solver in the cache loader. Replot was exercised with
only the six leaf directories present and imports of LI, JAX, SciPy, and
`halo_notes` blocked; all six PNGs were produced. A clean source copy without
`.git`, scratchpad, artifacts, or an old virtual environment also rebuilt the
batch. When Git metadata is absent, provenance records that fact; an archive
builder can pass the known revision explicitly as
`LUMICE_INTEGRAL_SOURCE_REVISION`.

The first figure retains the independently implemented narrow LI-owned
perspective helper from task 58.2 after consulting Writing-Lab only as visual
evidence; Writing-Lab is not a runtime dependency. Its explicit camera remains
at azimuth `65 deg`, elevation `24 deg`, distance `7`, with face `3` as the visible
landmark, and the rendered PNG remains byte-identical to its accepted production
asset. `pose-fiber` uses the same helper but constructs each representative ray
from the production corridor, verifies the physical hits with
`Polyhedron.intersect_ray`, and checks Snell tangential conservation
independently. `predictor-corrector` replays the
three accepted production steps from the versioned minimal seed in
`scripts/illustrations/data/predictor_corrector_seed.json`; it does not patch the
continuation implementation. The other leaves consume `DPField`, the shared
quadrature factors, and production corridor geometry without copying their
scientific equations.

Fresh production measurements for this batch were: projected $D_P$ level
residual `4.44e-16 rad`; representative-ray direction and independent Snell
residuals `4.20e-16` and `8.06e-16`; accepted predictor/corrector outgoing and
phase residuals `3.40e-16` and `2.75e-17 rad`, with predicted miss
`0.0478352436 deg`; factor-product relative disagreement `3.80e-16`; numerical
loop length `2.3806321109 rad`; and entry-area event
`s=1.9098764199 rad`, with $J_\perp=0.149751$ and one-sided slopes near
`+0.621/-0.621 a^2/rad`. These are regression and assembly observations, not a
claim of exactness or component completeness. The production integral is
`I/a^2=6.5814665434` with internal error estimate `2.69e-4`; tightening the
relative tolerance from `1e-4` to `1e-5` gave `6.5814512657`, inside the combined
reported estimates.

All six tracked images, their accepted prototypes, and individual 960- and
720-pixel reading views were opened directly. Core labels, mathematical
subscripts, markers, dashed boundaries, and geometry remained legible without
clipping. Secondary footer text on the denser figures approaches the lower
reading-size limit; task 58.4 must repeat the check at the actual English and
Chinese content widths. The accepted blue/orange/green/purple palette does not
pass an all-pairs colour-separation validator, so numbers, letters, marker
shapes, line styles, and direct labels remain mandatory redundant channels. This
is a recorded accessibility limitation, not a claim of author aesthetic
approval.

## 7. Acceptance and handoff

### 7.1 Scientific acceptance for each B1 figure

- Record the LI revision, fixture, units, data schema, and exact recompute and
  replot commands.
- Check every displayed production quantity against the data export rather than
  against pixels in the old PNG.
- Keep residuals and convergence statements quantitative. “Machine-level
  assembly agreement” is not an independent accuracy proof.
- State the component scope. One closed loop is never reported as proof that all
  components were found.
- Check direction semantics: $\hat{\mathbf s}$ and $\mathbf u$ point toward the
  source; incident propagation is their negative; apparent sky direction is
  negative outgoing propagation.
- Check that cached replot performs no scientific recomputation, missing cache
  fails, and explicit recompute does not overwrite the canonical image.

### 7.2 Visual acceptance

- Render at the width used by the English and Chinese documents, not only at the
  source canvas size.
- Inspect text, mathematical notation, line types, markers, light grids, and
  colour-independent correspondence.
- Provide the author one local comparison view containing all six figures at
  their real reading widths. Numerical validity does not imply aesthetic
  approval; that final judgment remains with the author.

### 7.3 Inputs to subsequent scrum items

- **58.2** has completed `world-body-frames`, proving the directory and command
  split and the smallest LI-owned crystal drawing/data boundary.
- **58.3** has completed the other five figures and the shared 6/6 batch. It
  removed the Writing-Lab ray runtime dependency, added a versioned minimal
  process seed, and delivered self-contained leaf caches plus production
  science/reading-size evidence. Stable paths and caption facts are in sections
  2 and 6.4.
- **58.4** embeds the single canonical image per stable ID at the primary and
  secondary locations above, writes separate English and Chinese captions, and
  checks real document widths and links.
- **Top-level owner** may turn B2/B3 rows into later tasks only after their named
  prerequisites hold. This document neither allocates new task IDs nor starts
  those calculations.

## 8. Coverage limits and unresolved decisions

This inventory reviewed the current explanatory bodies and the named supporting
contracts. It does not establish that every future reader question has been
found. In particular:

- the six accepted figures cover one normal closed `3-5` component, not open
  arcs, multiple components, rank loss, exact low-dimensional pose measures, or
  finite-source averaging;
- a general walked weight-kink set is not yet completeness-certified;
- the best B2 fixtures and the final visual composition of mechanism-comparison
  figures still require scientific and author review;
- B1 now has one canonical PNG per stable ID with DejaVu Sans and the accepted
  prototype palette; author aesthetic approval and a final palette decision
  remain open, and extra SVG/PDF formats still require a demonstrated need;
- prose, tables, and equations retain default priority where they communicate a
  relation more directly than a new image.

The inventory should be revisited when a new theory section introduces a new
reader relationship, not merely when a new module or output array appears.
