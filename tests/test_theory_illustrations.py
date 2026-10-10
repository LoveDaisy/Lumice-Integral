from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
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
    face_normals_body = np.array([[0.0, 0.0, 1.0]])
    face_centers_body = np.array([[0.0, 0.0, 0.0]])
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
        "illustration_camera": world_body_data.ILLUSTRATION_CAMERA,
        "geometry": {
            "face_numbers": [3],
            "face_vertex_ids": [[0, 1, 2]],
            "edges": [[0, 1], [1, 2], [0, 2]],
            "face_landmark": 3,
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


def test_world_body_data_round_trip_is_self_contained(tmp_path):
    directory = _write_valid(tmp_path)

    loaded = world_body_data.load_data(tmp_path)

    assert loaded.directory == directory
    assert loaded.metadata["schema"] == world_body_data.SCHEMA_VERSION
    np.testing.assert_array_equal(loaded.arrays["rotations"], np.repeat(np.eye(3)[None], 4, axis=0))


def test_world_body_data_missing_payload_reports_exact_recompute_command(tmp_path):
    with pytest.raises(world_body_data.IllustrationDataError) as caught:
        world_body_data.load_data(tmp_path / "data with spaces")

    assert "missing metadata.json, arrays.npz" in str(caught.value)
    assert world_body_data.recompute_command(tmp_path / "data with spaces") in str(caught.value)


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
        ("source_world", np.array([np.nan, 0.0, 0.0]), "contains non-finite"),
        ("rotations", np.zeros((4, 3, 3)), "not elements of SO\\(3\\)"),
        ("source_body", np.zeros((4, 3)), "coordinate reconstruction"),
        ("outgoing_directions", np.zeros((4, 3)), "outgoing-direction residual"),
    ],
)
def test_world_body_data_rejects_invalid_numeric_payload(tmp_path, name, value, message):
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
    planned = regen.commands(
        mode="all",
        data_root=Path("data root"),
        output_directory=Path("output root"),
        only="world-body-frames",
    )

    assert [(step.figure_id, operation) for step, operation, *_ in planned] == [
        ("world-body-frames", "recompute"),
        ("world-body-frames", "replot"),
    ]
    assert [category for *_, category, _ in planned] == ["scientific recompute", "pure rendering"]
    assert all(Path(command[1]).is_absolute() for _, _, command, _, _ in planned)
    assert "--output-dir" not in planned[0][2]
    assert planned[1][2][-2:] == ["--output-dir", "output root"]


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
    assert "Implemented batch: 1/6 B1 figures" in completed.stdout
    assert "world-body-frames:recompute" in completed.stdout
    assert "world-body-frames:replot" in completed.stdout
    assert "scientific recompute" in completed.stdout
    assert "pure rendering" in completed.stdout
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

    result = world_body_data.recompute_data(tmp_path)

    np.testing.assert_allclose(
        result.arrays["rotations"],
        np.asarray(reference["rotations"]),
        rtol=0.0,
        atol=reference["rotation_atol"],
    )
    assert result.metadata["source"]["component_scope"] == "one component reached from one seed"
    assert result.metadata["source"]["component_completeness"] == "unknown"
    assert result.metadata["sampling"]["selected_indices"] == [0, 128, 256, 384]
    assert result.metadata["validation"]["dense_outgoing_direction_max_l2"] < 1e-10
    assert result.metadata["validation"]["dense_retraction_residual_max_l2"] < 1e-10
    upstream = tmp_path / "canonical-fiber" / "metadata.json"
    assert json.loads(upstream.read_text(encoding="utf-8"))["schema"] == "lumice-integral.figure-data/v3"
