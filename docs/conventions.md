# Conventions

The single authority for the coordinate, pose, sun, sign and symbol
conventions of Lumice Integral, and for how they map onto Lumice and onto the
writing series *现代冰晕研究漫谈* (`~/Codes/Writing-Lab/现代冰晕研究漫谈`).
Every row names the test that checks it numerically. Established by task
`notation-alignment` (2026-09-24); roadmap section 9 records the decision.

## Authority rules (owner, 2026-09-23)

1. **Coordinates, face numbering, pose chain, azimuth, light source, camera
   and pixels** follow the Lumice documentation: `doc/coordinate-convention.md`
   (main), `crystal-orientation-sampling.md`,
   `crystal-geometry-representation.md`, `raypath-symmetry.md` in the Ice
   Halo Simulation repository. This file cites their sections and does not
   restate them; it records only how Lumice Integral maps onto them and what
   Lumice Integral adds.
2. **Mathematical notation Lumice does not cover** (fiber coordinates,
   $\Phi$, $D$, $M$, $W$, signatures, ...) follows the writing series'
   `docs/framework.md`.
3. **What neither covers** is decided here.

Any future disagreement is settled by these rules, not re-discussed item by
item. Lumice and the writing series stay read-only evidence: nothing is
imported from either (`AGENTS.md`).

## The table

| # | item | Lumice | writing series | Lumice Integral | check |
|---|---|---|---|---|---|
| 1 | crystal frame, face numbers | §1: `N1` = +z (c axis), `N3` = +x; `geo3d_closedform.hpp` `kHexFaceCos/Sin`: face 3+i at i·60° | the same: `halo_notes.draw.geometry` imports `lumice_integral.geometry.HexPrism` | `geometry.HexPrism`: faces 1/2 = ±z, face 3+i outward normal at azimuth i·60° | `tests/test_conventions.py::test_face_numbering_is_lumices` |
| 2 | pose chain | §6: `R = Rz(az − 180°) · Ry(−zenith) · Rz(roll)`, active, body → world | simulation layer (`halo_notes.sim`) writes Lumice configs: the same chain. Chapter parametrisation `attitude.column_attitude(ψ, θ) = Rz(ψ) · Ry(90°) · Rz(θ)` | Lumice's chain (`pose_density` module docstring, `c_axis_zenith`, `c_axis_roll`); `v_W = R v_B` (contract §2) | `test_column_attitude_is_the_lumice_chain_with_theta_equal_roll_minus_180` |
| 3 | roll zero | §5.3 (Parry: `N3` up, roll ≈ 0) | `column_attitude`: `θ = roll − 180°` (at θ = 0 face 3 points down; the two differ by `Rz(180°)`, a crystal symmetry, so a class-level result is the same and only the labels 3 ↔ 6 swap) | roll = 0 puts face 3 on top of a horizontal c axis | `test_roll_zero_puts_face_3_up_and_theta_zero_puts_it_down`, `test_column_attitude_...` |
| 4 | sun direction (public) | §4: `(altitude, azimuth)` parametrise the sun **position** vector, toward the sun; azimuth 0 = +x (§3) | `attitude.sun_vector(Σ) = (cos Σ, 0, sin Σ)`, toward the sun; framework theorem 8 `s` | **`ŝ`, toward the sun**: `camera.sun_direction(altitude_deg, azimuth_deg)`, `canonical_scene.canonical_sun_direction()`, `band_sum.BandSumScene.sun_direction`, `s2_store` `sun` (the general functions; a store records no sun, schema 3) | `tests/test_camera.py::test_sun_direction_points_toward_the_sun_and_light_travels_away_from_it`, `test_sun_direction_is_the_writing_series_sun_vector` |
| 5 | sun ray propagation (internal) | §4: `SampleRayDir` emits photons **along** `−ŝ` (sampling-side convention) | `Φ_P` takes the incident *propagation* direction (framework §2, theorem 8 `Φ_P(−R⁻¹s)`) | the solver's `incident_direction` = contract §2 `s` = `−ŝ`, the only input of `optics`, `weights`, `geometry.entry_measure`, `discovery`, `path_class`, `continuation`; converted **only** by `camera.incident_direction_from_sun` (`0.0 − ŝ`, so a zero component stays `+0.0`) | `test_sun_direction_points_toward_...` (bitwise `−ŝ`, no `−0.0`) |
| 6 | Phase II base point `u` | — | theorem 8: `u = R⁻¹s` (the sun in the crystal frame) | `u = R⁻¹ŝ`: `s2_store` events (`SCHEMA_VERSION = 4`), `align_rotations(u, sun)`: `R u = ŝ`; roadmap §4 | `test_store_u_phi_and_d_are_framework_theorem_8`; `tests/test_s2_store.py::test_events_match_the_batch_evaluators_directly` |
| 7 | outgoing map, deviation | — | `D_P(u) = ∠(−Φ_P(−u), u)`; light point `x = −R Φ_P(−u)` | store field `phi = Φ_P(−u)` (body-frame outgoing *propagation*), `D = ∠(phi, −u) = D_P(u)`; the pixel's deviation is the angle between the incoming and outgoing propagation directions (`band_sum.pixel_band`), equal to `∠(x, ŝ)` | `test_store_u_phi_and_d_are_framework_theorem_8` |
| 8 | outgoing direction vs sky point | camera §9: the image shows `−(outgoing propagation)` | light point `x` (sky direction) | solver target `d` = outgoing propagation, crystal → observer (contract §2); camera-side sky direction `−d`; the negation happens only in `camera.linear_pixel_outgoing_direction` | `tests/test_camera.py::test_pixel_center_round_trips_through_the_forward_projection` |
| 9 | camera, pixels | §9 camera; linear lens (`MakeCameraRotation`, `ProjectExitToPixel`, read as evidence) | — | `camera.py` transcription: pixel `(column, row)` covers `[column, column+1) × [row, row+1)`, row 0 at the top | `tests/test_camera.py` |
| 10 | fiber coordinate (twist about the sun) | — | theorem 8: `Rot_s(θ) R_0(u)`, Haar `= du dθ` | roadmap §4.1/§4.2 keep `ψ` (**kept**, see below) | — (notation only) |
| 11 | sun elevation | `altitude` | `Σ` | `altitude_deg` in code, `altitude` in prose (**kept**, see below) | row 4 tests |
| 12 | column azimuth / spin | `azimuth`, `roll` | `column_attitude(ψ, θ)`: ψ azimuth, θ spin | `azimuth`, `roll` (Lumice names); `pose_density` docstrings call the roll `psi` (**kept**, see below) | row 2 tests |
| 13 | `Φ_P`, `M`, `W`, `D_P`, `w_P`, signature | — | framework §0 and theorem 4 (§2): `Φ_P = 𝒮_{n_b} ∘ M ∘ 𝒮_{n_a} = M ∘ W`, `𝒮` refraction, `M` fold matrix, `W` unfolded wedge refraction, `ñ_b = M⁻¹ n_b` | the same letters: `geometry.fold_matrix` = `M` (mirrors left-multiplied in encounter order), `path_class.phi_key` = `(M, n_a, M^T n_b)`, `w_P = A_P T_P` | `tests/test_geometry_unfold.py`, `tests/test_path_class_phi_key.py` |
| 14 | `D6h` / `G` element labels | `raypath-symmetry.md` §2a (`D6h`, P/B/D) | `reflection_group`: the 12 fold matrices `G`, published numbers #1–#12 (`identify(M)`); `signature.D6H`: 24 elements in the order `Rz(60k)`, `sxy(30k)`, then `B·` | the writing series' tables are this repository's `lumice_integral.symmetry` (task 19): `path_class.hexprism_symmetry_matrices()` returns `symmetry.signature.D6H`, the only `D6h` table, in its construction order; `phi_key(...)[0]` is an **index into that tuple**, not a published number (**interface**, see below) | `test_phi_key_indexes_the_d6h_tuple_by_matrix_not_by_published_number`; theorems 1 / 2 / 3 and "five eigenvalue classes" of `G`: `test_symmetry_reflection_group.py::test_closure_is_twelve_with_growth_1_5_10_12_12` / `::test_conjugacy_classes_equal_published_six` / `::test_commuting_with_rz_is_3_4_5_6_11_12` / `::test_eigenvalue_classes_are_five_with_11_merged_into_2`; theorem 5′ (34 = 6 + 12 + 16): `test_symmetry_signature.py::test_phi_class_table_counts_and_ids` |
| 15 | angle units | degrees in configs | degrees in code, radians in formulas | API arguments with `_deg` are degrees, `_rad` radians; stored `D`, pixel deviations and all Lie-algebra coordinates are radians (contract §2) | — |
| 16 | pose storage | — | — | `(3, 3)` matrices; quaternions only as an interpolation chart, `(w, x, y, z)` scalar first (`so3`); right-trivialised increments `R exp([δ]×)` (contract §3) | `tests/test_so3.py` |
| 17 | crystal size, entry weight | `doc/configuration.md`, `ray_allocation` / `proportion` notes: equal-surface-area convention; every ray's weight × `A/(S/2)` at entry (`lm_pcg::entry_weight`, since `6fc48bb4`) | — | one crystal, hexagon edge `a = 1`: values in `length² / sr`; to Lumice's `raw / emitted_energy` through `K_p = N_sym ȳ(550) Ω_p / (S/2)` (`docs/ch06-reference-fixture.md` §7 stage 4) | `tests/test_probe_absolute_scale.py`; `scripts/probe_absolute_scale.py`, `scripts/compare_lumice_family.py` against Lumice exports |
| 18 | interface power split (entry, internal reflections, exit) | `src/core/optics.cpp::HitSurface`: every hit splits the weight deterministically, reflected `R w`, refracted `(1 - R) w` (none under TIR); `R` from `src/core/shared/optics_shared.h::GetReflectRatio`, per interface, unpolarized s/p average, no polarization state carried between faces | — | path energy factor `T_entry × Π_k R_k × T_exit` with `R_k = 1` under TIR (`optics.fresnel_transmission_path(_batch)`, same s/p average as `fresnel_unpolarized_transmittance`); a non-total internal reflection is a weight, not a domain boundary (task `optics-partial-reflection`, 2026-09-25) | `tests/test_optics_path_generic.py` (internal `R` against Lumice's `GetReflectRatio` form); `scripts/compare_lumice_family.py` on the A60-10 family |

## Boundaries and kept notation

**The Phase I contract's `s` is not renamed.** `docs/phase1-math-contract.md`
§2 defines `s` as the propagation direction (sun → crystal) and forbids
redefining it silently; its solver, fixtures and every `incident_direction`
argument keep that meaning. It is the writing series' `s` with the opposite
sign — the same letter, the opposite direction — so the public interfaces use
`ŝ` / `sun` / `sun_direction` for the direction toward the sun and
`incident_direction` for the propagation direction, and
`camera.incident_direction_from_sun` is the one place the sign changes
(explicit at the boundary, never two meanings under one name).

**Phase II store schema.** `s2_store.SCHEMA_VERSION = 4` (scrum `internal-partial-reflection`, 2026-09-25): the arrays of schema 3 with `w` carrying the internal reflectances (#18) and the events of partial-reflection branches, which schema 3 dropped. Schema 3 (task `s2-store-schema-3`) records `u = R⁻¹ŝ`
and no light source: the events depend on crystal × path × refractive index ×
sampling only, the build aligns `R u = ŝ` to one fixed reference direction
(numerically the canonical sun, which keeps schema 2 builds bit for bit) and
the spec and cache key carry no `sun_direction`
(`test_events_are_independent_of_the_reference_direction`); each array is its
own `.npy`. Schema 2 (task `notation-alignment`) had the same arrays with the
sun direction in the key. Schema 1 (tasks 13–17) recorded `u = R⁻¹s` with the propagation `s`, i.e. the
same events with `u` negated. The schema 2 default lattice is the antipode of
the schema 1 lattice (`s2_store.store_lattice`, `RandomSphereSampler` negated
too), and every Phase II formula is the schema 1 one with `(u, s)` replaced by
`(−u, −ŝ)`, which leaves frames, cross and outer products unchanged bit for
bit: every pose, `phi`, `D`, `w` and band-sum value is the schema 1 value
exactly (`tests/test_s2_store.py::test_rebuilds_task13_store_bit_for_bit`
against the task 13 artifacts). Schema 1 and 2 cache directories are refused by
`S2EventStore.load` (`test_cache_refuses_an_older_schema`); legacy flat
artifacts are read through the one named conversion
`s2_store.events_from_schema1` (u negated).

**Kept symbols (rows 10–12).** The writing series uses `θ` both for the
fiber coordinate (theorem 8) and for the column spin (`column_attitude`), and
`ψ` for the column azimuth; Lumice Integral already uses `theta` for the
c-axis zenith (`pose_density`) and `ψ` for the fiber twist (roadmap §4, the
`s2_store` psi-invariance self-check). Renaming the fiber coordinate to `θ`
would collide with the zenith, and renaming it to anything the writing series
uses would collide there. So the symbols stay, scoped by module, and this
table is the dictionary: roadmap `ψ` (fiber twist about `ŝ`) = theorem 8 `θ`;
`pose_density` `psi` / `roll` (spin about the c axis) = `column_attitude` `θ`
+ 180°; `altitude` = `Σ`. The pixel coordinates `(u, v)` of `camera.py` are
screen coordinates, unrelated to Phase II's `u`.

**Interface settled by task 19 (`symmetry-authority`, 2026-09-24).** Element
identity across projects is the **matrix**, never an index. The writing
series' element tables moved into `lumice_integral.symmetry`, and
`hexprism_symmetry_matrices()` now returns `symmetry.signature.D6H` (the
generator closure it used to build is gone, so the repository holds one `D6h`
table). The order changed from closure order to the table's construction
order, and `phi_key`'s first entry is still an index into it, not a published
number: every consumer compares keys for equality only (`s2_store` checks that
members share a key; no cache key, file or provenance records the index), so
the reordering changes no value. Published numbers #1–#12 of `G` come from
`symmetry.reflection_group.identify(M)`.

**Camera.** `camera.py` is a transcription of Lumice's linear lens (§9 and the
source files named in its docstring); the direction toward the sky it computes
is negated once into the solver's `d` (row 8).
