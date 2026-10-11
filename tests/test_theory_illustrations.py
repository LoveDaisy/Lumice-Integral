from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys

import numpy as np
import pytest


REPO = Path(__file__).parents[1]
REFERENCE = Path(__file__).parent / "data" / "theory_illustrations" / "world_body_reference.json"
ILLUSTRATIONS = REPO / "scripts" / "illustrations"
sys.path.insert(0, str(ILLUSTRATIONS))


def _load_module(name: str):
    path = ILLUSTRATIONS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


world_body_data = _load_module("_world_body_data")
crystal_view = _load_module("_crystal_view")
regen = _load_module("regen")


def _valid_arrays() -> dict[str, np.ndarray]:
    rotations = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    source_world = np.array([1.0, 0.0, 0.0])
    source_body = np.repeat(source_world[None, :], 4, axis=0)
    vertices_body = np.zeros((12, 3))
    vertices_body[:3] = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    face_normals_body = np.array([[0.0, 0.0, 1.0]])
    face_centers_body = np.array([[0.0, -1.0 / 3.0, 0.0]])
    return {
        "selected_dense_indices": world_body_data.SELECTED_INDICES,
        "rotations": rotations,
        "source_world": source_world,
        "source_body": source_body,
        "target_direction": np.array([0.0, 1.0, 0.0]),
        "outgoing_directions": np.repeat(np.array([[0.0, 1.0, 0.0]]), 4, axis=0),
        "vertices_body": vertices_body,
        "vertices_world": np.repeat(vertices_body[None, :, :], 4, axis=0),
        "face_numbers": np.array([3], dtype=np.int64),
        "face_normals_body": face_normals_body,
        "face_normals_world": np.repeat(face_normals_body[None, :, :], 4, axis=0),
        "face_centers_body": face_centers_body,
        "face_centers_world": np.repeat(face_centers_body[None, :, :], 4, axis=0),
    }


def _valid_metadata() -> dict[str, object]:
    return {
        "schema": world_body_data.SCHEMA_VERSION,
        "figure_id": world_body_data.FIGURE_ID,
        "fixture": {"name": "synthetic"},
        "sampling": {
            "dense_node_count": world_body_data.DENSE_NODE_COUNT,
            "retraction_iterations": 2,
            "selected_indices": world_body_data.SELECTED_INDICES.tolist(),
        },
        "geometry": {
            "face_numbers": [3],
            "face_vertex_ids": [[0, 1, 2]],
            "edges": [[0, 1], [1, 2], [0, 2]],
        },
    }


def _write_valid(root: Path) -> Path:
    world_body_data._write_data(root, _valid_arrays(), _valid_metadata())
    return world_body_data.data_directory(root)


def _rewrite_metadata(directory: Path, update) -> None:
    path = directory / world_body_data.METADATA_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    update(metadata)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_array(directory: Path, name: str, value: np.ndarray) -> None:
    arrays_path = directory / world_body_data.ARRAYS_FILE
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: np.array(payload[key], copy=True) for key in payload.files}
    arrays[name] = value
    with arrays_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    digest = sha256(arrays_path.read_bytes()).hexdigest()

    def update(metadata):
        metadata["payload"]["sha256"] = digest
        metadata["payload"]["arrays"][name]["shape"] = list(value.shape)
        metadata["payload"]["arrays"][name]["dtype"] = str(value.dtype)

    _rewrite_metadata(directory, update)


def _rewrite_arrays(directory: Path, update) -> None:
    arrays_path = directory / world_body_data.ARRAYS_FILE
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: np.array(payload[key], copy=True) for key in payload.files}
    update(arrays)
    with arrays_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    digest = sha256(arrays_path.read_bytes()).hexdigest()
    _rewrite_metadata(
        directory,
        lambda metadata: metadata["payload"].__setitem__("sha256", digest),
    )


def test_world_body_data_round_trip_is_self_contained(tmp_path):
    directory = _write_valid(tmp_path)

    loaded = world_body_data.load_data(tmp_path)

    assert loaded.directory == directory
    assert loaded.metadata["schema"] == world_body_data.SCHEMA_VERSION
    np.testing.assert_array_equal(loaded.arrays["rotations"], np.repeat(np.eye(3)[None], 4, axis=0))


def test_world_body_data_missing_payload_reports_exact_recompute_command(tmp_path):
    data_root = tmp_path / "data with spaces"
    with pytest.raises(world_body_data.IllustrationDataError) as caught:
        world_body_data.load_data(data_root)

    command = world_body_data.recompute_command(data_root)
    assert "missing metadata.json, arrays.npz" in str(caught.value)
    assert command in str(caught.value)
    assert shlex.split(command) == [
        "uv",
        "run",
        "--project",
        str(REPO.resolve()),
        "python",
        str((ILLUSTRATIONS / "world_body_frames.py").resolve()),
        "--recompute",
        "--data-dir",
        str(data_root.resolve()),
    ]


def test_world_body_data_reports_unreadable_files(tmp_path):
    directory = _write_valid(tmp_path)
    metadata_path = directory / world_body_data.METADATA_FILE
    metadata_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(world_body_data.IllustrationDataError, match="cannot read metadata.json"):
        world_body_data.load_data(tmp_path)

    _write_valid(tmp_path)
    arrays_path = directory / world_body_data.ARRAYS_FILE
    arrays_path.write_bytes(b"not an npz")
    with pytest.raises(world_body_data.IllustrationDataError, match="cannot read arrays.npz"):
        world_body_data.load_data(tmp_path)


def test_world_body_data_reports_truncated_npz_with_recompute_command(tmp_path):
    directory = _write_valid(tmp_path)
    arrays_path = directory / world_body_data.ARRAYS_FILE
    arrays_path.write_bytes(arrays_path.read_bytes()[:-16])

    with pytest.raises(world_body_data.IllustrationDataError) as caught:
        world_body_data.load_data(tmp_path)

    assert "cannot read arrays.npz" in str(caught.value)
    assert world_body_data.recompute_command(tmp_path) in str(caught.value)


def test_world_body_data_reports_binary_corruption_with_recompute_command(tmp_path):
    directory = _write_valid(tmp_path)
    metadata_path = directory / world_body_data.METADATA_FILE
    metadata_path.write_bytes(b"\xff")

    with pytest.raises(world_body_data.IllustrationDataError) as caught:
        world_body_data.load_data(tmp_path)
    assert "cannot read metadata.json" in str(caught.value)
    assert world_body_data.recompute_command(tmp_path) in str(caught.value)

    _write_valid(tmp_path)
    arrays_path = directory / world_body_data.ARRAYS_FILE
    arrays_path.write_bytes(b"")

    with pytest.raises(world_body_data.IllustrationDataError) as caught:
        world_body_data.load_data(tmp_path)
    assert "cannot read arrays.npz" in str(caught.value)
    assert world_body_data.recompute_command(tmp_path) in str(caught.value)


def test_world_body_data_rejects_missing_required_array(tmp_path):
    directory = _write_valid(tmp_path)
    _rewrite_arrays(directory, lambda arrays: arrays.pop("source_world"))

    with pytest.raises(world_body_data.IllustrationDataError, match="missing required arrays.*source_world"):
        world_body_data.load_data(tmp_path)


def test_world_body_data_rejects_non_object_metadata_root(tmp_path):
    directory = _write_valid(tmp_path)
    path = directory / world_body_data.METADATA_FILE
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(world_body_data.IllustrationDataError, match="root must be an object"):
        world_body_data.load_data(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata["payload"].pop("file"), "payload file declaration"),
        (lambda metadata: metadata.pop("geometry"), "geometry declaration"),
        (lambda metadata: metadata["geometry"].pop("face_vertex_ids"), "face_vertex_ids"),
        (lambda metadata: metadata["geometry"].pop("edges"), "edges"),
    ],
)
def test_world_body_data_rejects_missing_consumption_metadata(tmp_path, mutate, message):
    directory = _write_valid(tmp_path)
    _rewrite_metadata(directory, mutate)

    with pytest.raises(world_body_data.IllustrationDataError, match=message):
        world_body_data.load_data(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata["geometry"].__setitem__("edges", []), "non-empty list"),
        (
            lambda metadata: metadata["geometry"].__setitem__("face_vertex_ids", [[0, 0, 2]]),
            "repeated vertex indices",
        ),
        (
            lambda metadata: metadata["geometry"].__setitem__(
                "edges", [[0, 1], [1, 2], [0, 11]]
            ),
            "no adjacent face",
        ),
        (
            lambda metadata: metadata["geometry"].__setitem__(
                "edges", [[0, 0], [1, 2], [0, 2]]
            ),
            "invalid or repeated",
        ),
    ],
)
def test_world_body_data_rejects_unusable_geometry_indices(tmp_path, mutate, message):
    directory = _write_valid(tmp_path)
    _rewrite_metadata(directory, mutate)

    with pytest.raises(world_body_data.IllustrationDataError, match=message):
        world_body_data.load_data(tmp_path)


def test_world_body_data_ignores_redundant_and_presentation_metadata(tmp_path):
    directory = _write_valid(tmp_path)

    def mutate(metadata):
        metadata["sampling"] = {"selected_indices": [99]}
        metadata["illustration_camera"] = {"distance": -1.0}
        metadata["geometry"]["face_numbers"] = [99]
        metadata["geometry"]["face_landmark"] = 99
        metadata["payload"]["arrays"] = None

    _rewrite_metadata(directory, mutate)

    loaded = world_body_data.load_data(tmp_path)

    assert loaded.metadata["illustration_camera"]["distance"] == -1.0


def test_world_body_data_allows_unrelated_additional_records(tmp_path):
    directory = _write_valid(tmp_path)
    _rewrite_arrays(directory, lambda arrays: arrays.__setitem__("unused", np.array([1.0])))
    _rewrite_metadata(directory, lambda metadata: metadata.__setitem__("unused", {"note": "ignored"}))

    loaded = world_body_data.load_data(tmp_path)

    np.testing.assert_array_equal(loaded.arrays["unused"], [1.0])


def test_world_body_data_does_not_recompute_scientific_invariants_on_load(tmp_path):
    directory = _write_valid(tmp_path)
    _rewrite_array(directory, "rotations", np.zeros((4, 3, 3)))
    _rewrite_array(directory, "source_body", np.zeros((4, 3)))

    loaded = world_body_data.load_data(tmp_path)

    np.testing.assert_array_equal(loaded.arrays["rotations"], 0.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "unsupported/v9", "unsupported schema"),
        ("figure_id", "other-figure", "unexpected figure_id"),
    ],
)
def test_world_body_data_rejects_wrong_identity(tmp_path, field, value, message):
    directory = _write_valid(tmp_path)
    _rewrite_metadata(directory, lambda metadata: metadata.__setitem__(field, value))

    with pytest.raises(world_body_data.IllustrationDataError, match=message):
        world_body_data.load_data(tmp_path)


def test_world_body_data_rejects_hash_mismatch(tmp_path):
    directory = _write_valid(tmp_path)
    arrays_path = directory / world_body_data.ARRAYS_FILE
    arrays_path.write_bytes(arrays_path.read_bytes() + b"corrupt")

    with pytest.raises(world_body_data.IllustrationDataError, match="SHA-256"):
        world_body_data.load_data(tmp_path)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("rotations", np.zeros((3, 3, 3)), "rotations has shape"),
        ("rotations", np.zeros((4, 3, 3), dtype=np.int64), "floating dtype"),
        ("source_world", np.array([np.nan, 0.0, 0.0]), "contains non-finite"),
    ],
)
def test_world_body_data_rejects_unusable_numeric_payload(tmp_path, name, value, message):
    directory = _write_valid(tmp_path)
    _rewrite_array(directory, name, value)

    with pytest.raises(world_body_data.IllustrationDataError, match=message):
        world_body_data.load_data(tmp_path)


def test_crystal_camera_projection_and_visibility_are_deterministic():
    camera = crystal_view.Camera(azimuth_deg=0.0, elevation_deg=0.0, distance=7.0)

    projected = camera.project_xy(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(projected, [[1.0, 0.0], [0.0, 1.0]], atol=1e-15)
    near = camera.project_xy(np.array([[1.0, 1.0, 0.0]]))[0]
    assert near[0] == pytest.approx(7.0 / 6.0)
    with pytest.raises(ValueError, match="behind the perspective camera"):
        camera.project_xy(np.array([[7.0, 0.0, 0.0]]))

    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    centers = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    visible = crystal_view.face_visibility(normals, centers, camera)
    np.testing.assert_array_equal(visible, [True, False])
    assert crystal_view.edge_visibility([[0, 1]], [[0, 1, 2], [0, 1, 3]], visible).tolist() == [True]


def test_world_body_replot_has_eight_panels_and_does_not_modify_data(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    directory = _write_valid(tmp_path / "data")
    world_body_frames = _load_module("world_body_frames")
    metadata_path = directory / world_body_data.METADATA_FILE
    arrays_path = directory / world_body_data.ARRAYS_FILE
    before = {
        path: (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (metadata_path, arrays_path)
    }

    output = world_body_frames.replot_data(tmp_path / "data", tmp_path / "figures")

    image = matplotlib.image.imread(output)
    assert image.shape[:2] == (1500, 2400)
    assert before == {
        path: (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (metadata_path, arrays_path)
    }
    data = world_body_data.load_data(tmp_path / "data")
    figure = world_body_frames.render_figure(data)
    try:
        assert len(figure.axes) == 8
        assert [text.get_text() for text in figure.texts].count("1") == 2
        assert [text.get_text() for text in figure.texts].count("4") == 2
        assert sum(text.get_text() == "3" for axis in figure.axes for text in axis.texts) == 8
        assert sum("source" in text.get_text() for axis in figure.axes for text in axis.texts) == 4
        assert sum("Sun" in text.get_text() for axis in figure.axes for text in axis.texts) == 4
        lower_polygons = [figure.axes[index].patches[0].get_xy() for index in (1, 3, 5, 7)]
        for polygon in lower_polygons[1:]:
            np.testing.assert_allclose(polygon, lower_polygons[0], atol=0.0)
    finally:
        matplotlib.pyplot.close(figure)


def test_world_body_replot_closes_figure_when_save_fails(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _write_valid(tmp_path / "data")
    world_body_frames = _load_module("world_body_frames")
    before = plt.get_fignums()

    def fail_save(self, *args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="synthetic write failure"):
        world_body_frames.replot_data(tmp_path / "data", tmp_path / "figures")

    assert plt.get_fignums() == before


def test_world_body_replot_closes_figure_when_render_fails(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    directory = _write_valid(tmp_path / "data")
    _rewrite_array(directory, "face_numbers", np.array([4], dtype=np.int64))
    world_body_frames = _load_module("world_body_frames")
    before = plt.get_fignums()

    with pytest.raises(ValueError, match="highlight face 3"):
        world_body_frames.replot_data(tmp_path / "data", tmp_path / "figures")

    assert plt.get_fignums() == before


def test_world_body_data_import_does_not_load_presentation_layer(tmp_path):
    code = f"""
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == '_crystal_view':
            raise RuntimeError(f'forbidden import: {{fullname}}')
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {str(ILLUSTRATIONS)!r})
import _world_body_data
assert '_crystal_view' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_world_body_replot_import_chain_excludes_solver_modules(tmp_path):
    pytest.importorskip("matplotlib")
    directory = _write_valid(tmp_path / "data")
    output_directory = tmp_path / "output with spaces"
    script = ILLUSTRATIONS / "world_body_frames.py"
    code = f"""
import importlib.abc
import runpy
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax', 'scipy', 'lumice_integral', 'halo_notes'}}:
            raise RuntimeError(f'forbidden import: {{fullname}}')
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {str(ILLUSTRATIONS)!r})
sys.argv = [{str(script)!r}, '--replot', '--data-dir', {str(tmp_path / 'data')!r}, '--output-dir', {str(output_directory)!r}]
runpy.run_path({str(script)!r}, run_name='__main__')
"""

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert (output_directory / "world-body-frames.png").is_file()
    assert directory.is_dir()


def test_regen_uses_one_ordered_command_table():
    assert [(step.figure_id, step.output_name) for step in regen.STEPS] == [
        ("world-body-frames", "world-body-frames.png"),
        ("deviation-level-set", "deviation-level-set.png"),
        ("pose-fiber", "pose-fiber.png"),
        ("predictor-corrector", "predictor-corrector.png"),
        ("fiber-contributions", "fiber-contributions.png"),
        ("entry-area", "entry-area.png"),
    ]
    planned = regen.commands(
        mode="all",
        data_root=Path("data root"),
        output_directory=Path("output root"),
        only="world-body-frames",
    )

    assert [(step.figure_id, operation) for step, operation, *_ in planned] == [
        ("canonical-data", "recompute"),
        ("world-body-frames", "recompute"),
        ("world-body-frames", "replot"),
    ]
    assert [category for *_, category, _ in planned] == [
        "shared scientific recompute",
        "leaf data assembly",
        "pure rendering",
    ]
    assert all(Path(command[1]).is_absolute() for _, _, command, _, _ in planned)
    assert "--canonical-data" in planned[1][2]
    assert "--canonical-data" not in planned[2][2]
    assert "--output-dir" not in planned[0][2]
    assert "--output-dir" not in planned[1][2]
    assert planned[2][2][-2:] == ["--output-dir", "output root"]


def test_regen_dry_run_is_side_effect_free_from_arbitrary_cwd(tmp_path):
    data_root = tmp_path / "data root"
    output_directory = tmp_path / "output root"
    completed = subprocess.run(
        [
            sys.executable,
            str(ILLUSTRATIONS / "regen.py"),
            "--mode",
            "all",
            "--data-dir",
            str(data_root),
            "--output-dir",
            str(output_directory),
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Implemented batch: 6/6 B1 figures" in completed.stdout
    assert "canonical-data:recompute" in completed.stdout
    assert "world-body-frames:recompute" in completed.stdout
    assert "world-body-frames:replot" in completed.stdout
    assert "shared scientific recompute" in completed.stdout
    assert "leaf data assembly" in completed.stdout
    assert "pure rendering" in completed.stdout
    assert str(data_root / "canonical-fiber") in completed.stdout
    assert str(data_root / world_body_data.FIGURE_ID) in completed.stdout
    assert not data_root.exists()
    assert not output_directory.exists()


def test_regen_replot_handles_space_paths_and_preserves_child_failure(tmp_path):
    pytest.importorskip("matplotlib")
    data_root = tmp_path / "data root"
    output_directory = tmp_path / "output root"
    _write_valid(data_root)
    command = [
        sys.executable,
        str(ILLUSTRATIONS / "regen.py"),
        "--mode",
        "replot",
        "--only",
        "world-body-frames",
        "--data-dir",
        str(data_root),
        "--output-dir",
        str(output_directory),
    ]

    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (output_directory / "world-body-frames.png").is_file()

    failed = subprocess.run(
        [
            *command[: command.index("--data-dir") + 1],
            str(tmp_path / "missing data"),
            "--output-dir",
            str(tmp_path / "unused output"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "Recompute with:" in failed.stderr
    assert not (tmp_path / "unused output").exists()


def test_regen_rejects_unknown_figure_id(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ILLUSTRATIONS / "regen.py"),
            "--mode",
            "replot",
            "--only",
            "not-a-figure",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "unknown figure ID" in completed.stderr
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_tracked_six_figure_batch_has_stable_png_names_and_sizes():
    expected = {
        "world-body-frames.png": (2400, 1500),
        "deviation-level-set.png": (2175, 1200),
        "pose-fiber.png": (2400, 1500),
        "predictor-corrector.png": (2250, 1200),
        "fiber-contributions.png": (2100, 1140),
        "entry-area.png": (2100, 1410),
    }
    directory = REPO / "docs" / "figures" / "theory"
    assert expected.keys() <= {path.name for path in directory.glob("*.png")}
    for name, dimensions in expected.items():
        payload = (directory / name).read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", payload[16:24]) == dimensions


def test_reproduction_contract_records_the_complete_six_figure_batch():
    contract = (REPO / "docs" / "theory-illustrations.md").read_text(encoding="utf-8")
    for name in (
        "world-body-frames",
        "deviation-level-set",
        "pose-fiber",
        "predictor-corrector",
        "fiber-contributions",
        "entry-area",
    ):
        assert f"[{name}.png](figures/theory/{name}.png)" in contract
        assert f"lumice-integral.illustration.{name}/v1" in contract
    assert "Implemented B1 batch: all six canonical figures" in contract
    assert "LUMICE_INTEGRAL_SOURCE_REVISION" in contract
    assert "does not\npass an all-pairs colour-separation validator" in contract
    assert "58.3** has completed" in contract


def test_tracked_world_body_asset_and_reproduction_contract_are_linked():
    image = REPO / "docs" / "figures" / "theory" / "world-body-frames.png"
    payload = image.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", payload[16:24]) == (2400, 1500)

    contract = (REPO / "docs" / "theory-illustrations.md").read_text(encoding="utf-8")
    assert "[world-body-frames.png](figures/theory/world-body-frames.png)" in contract
    assert world_body_data.SCHEMA_VERSION in contract
    assert "scripts/illustrations/regen.py --mode all" in contract
    assert "internal, rebuildable" in contract
    assert "re-certify SO(3)" in contract
    assert "azimuth `65 deg`" in contract
    assert "independently implemented" in contract
    assert "/Users/" not in contract

    for script in ILLUSTRATIONS.glob("*.py"):
        source = script.read_text(encoding="utf-8")
        assert "scratchpad/" not in source
        assert "/Users/" not in source
        assert "halo_notes" not in source or script.name == "_world_body_data.py"


def test_world_body_reference_records_the_accepted_sampling_and_source_hashes():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    assert reference["figure_id"] == world_body_data.FIGURE_ID
    assert reference["sampling"] == {
        "dense_node_count": 513,
        "retraction_iterations": 2,
        "selected_indices": [0, 128, 256, 384],
    }
    assert reference["source"]["prototype_sha256"] == world_body_data.ORIGINAL_PROTOTYPE_SHA256
    assert np.asarray(reference["rotations"]).shape == (4, 3, 3)


# slow: traces and retracts the canonical fiber, including its JAX compilation.
@pytest.mark.slow
def test_recomputed_world_body_data_matches_accepted_four_poses(tmp_path):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    data_root = tmp_path / "data with spaces"
    command = world_body_data.recompute_command(data_root)

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        shell=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = world_body_data.load_data(data_root)
    arrays = result.arrays

    from lumice_integral.canonical_scene import (
        canonical_crystal,
        canonical_pixel_problem,
        canonical_sun_direction,
        canonical_target_direction,
    )
    from lumice_integral.weights import evaluate_weights_batch
    import jax

    rotations = arrays["rotations"]
    np.testing.assert_allclose(
        rotations,
        np.asarray(reference["rotations"]),
        rtol=0.0,
        atol=reference["rotation_atol"],
    )
    np.testing.assert_array_equal(arrays["selected_dense_indices"], [0, 128, 256, 384])

    crystal = canonical_crystal()
    vertices_body = np.asarray(crystal.vertices)
    face_numbers = np.array([face.number for face in crystal.faces])
    face_normals_body = np.stack([crystal.normal(face) for face in crystal.faces])
    face_centers_body = np.stack([crystal.centroid(face) for face in crystal.faces])
    np.testing.assert_array_equal(arrays["vertices_body"], vertices_body)
    np.testing.assert_array_equal(arrays["face_numbers"], face_numbers)
    np.testing.assert_array_equal(arrays["face_normals_body"], face_normals_body)
    np.testing.assert_array_equal(arrays["face_centers_body"], face_centers_body)
    assert result.metadata["geometry"]["face_vertex_ids"] == [
        list(face.vertex_ids) for face in crystal.faces
    ]
    assert result.metadata["geometry"]["edges"] == [list(edge) for edge in crystal.edges]

    source_world = np.asarray(canonical_sun_direction())
    target_direction = np.asarray(canonical_target_direction())
    np.testing.assert_array_equal(arrays["source_world"], source_world)
    np.testing.assert_array_equal(arrays["target_direction"], target_direction)
    np.testing.assert_allclose(
        arrays["source_body"],
        np.einsum("nji,j->ni", rotations, source_world),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        arrays["vertices_world"],
        np.stack([vertices_body @ rotation.T for rotation in rotations]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        arrays["face_normals_world"],
        np.stack([face_normals_body @ rotation.T for rotation in rotations]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        arrays["face_centers_world"],
        np.stack([face_centers_body @ rotation.T for rotation in rotations]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.einsum("nij,nj->ni", rotations, arrays["source_body"]),
        np.broadcast_to(source_world, (4, 3)),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotations.transpose(0, 2, 1) @ rotations,
        np.broadcast_to(np.eye(3), (4, 3, 3)),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, rtol=0.0, atol=1e-10)

    problem = canonical_pixel_problem()
    outgoing = np.asarray(jax.vmap(problem.direction_evaluator)(rotations))
    np.testing.assert_allclose(arrays["outgoing_directions"], outgoing, rtol=0.0, atol=1e-12)
    assert np.max(np.linalg.norm(outgoing - target_direction, axis=1)) <= 1e-10
    validity = evaluate_weights_batch(
        problem.weight_evaluators,
        rotations,
        ("path_validity",),
    ).values["path_validity"]
    np.testing.assert_array_equal(validity, np.ones_like(validity))

    world_body_frames = _load_module("world_body_frames")
    assert world_body_frames.LANDMARK_FACE_NUMBER in face_numbers
    assert world_body_frames.DEFAULT_CAMERA == {
        "azimuth_deg": 65.0,
        "elevation_deg": 24.0,
        "distance": 7.0,
        "projection": "perspective",
    }
    assert result.metadata["source"]["component_scope"] == "one component reached from one seed"
    assert result.metadata["source"]["component_completeness"] == "unknown"
    assert result.metadata["sampling"]["selected_indices"] == [0, 128, 256, 384]
    assert result.metadata["validation"]["dense_outgoing_direction_max_l2"] < 1e-10
    assert result.metadata["validation"]["dense_retraction_residual_max_l2"] < 1e-10
    upstream = data_root / "canonical-fiber" / "metadata.json"
    assert json.loads(upstream.read_text(encoding="utf-8"))["schema"] == "lumice-integral.figure-data/v3"
