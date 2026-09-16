import numpy as np
import pytest

from lumice_integral.geometry import HexPrism, Pyramid, fold_matrix, unfold_faces

from _geometry_oracles import NORMALS, path_matrix, refraction_cancels

# ch2 已发表的三条光路（六棱柱），首尾折射抵消的平行光路
CH2_PATHS = [(1, 3, 2), (3, 1, 5, 7, 4), (1, 2, 3, 5, 1)]


@pytest.mark.parametrize("faces", CH2_PATHS)
def test_body_count_and_chain(faces):
    c = HexPrism(1.0, 0.8)
    bodies, _ = unfold_faces(c, faces)
    assert len(bodies) == len(faces) - 1
    assert bodies[0] is c
    # 第 j 个幽灵 = 第 j-1 个幽灵关于其面 m_j 的镜像（级联，而非都对原晶体镜像）
    for j, number in enumerate(faces[1:-1], start=1):
        prev = bodies[j - 1]
        assert np.allclose(bodies[j].vertices, prev.mirrored(prev.face(number)).vertices)
        # 幽灵 j 的面 m_j 与幽灵 j-1 的面 m_j 共面、法向相反（镜面两侧）
        assert np.allclose(bodies[j].normal(bodies[j].face(number)), -prev.normal(prev.face(number)))
        assert np.allclose(bodies[j].centroid(bodies[j].face(number)), prev.centroid(prev.face(number)))


@pytest.mark.parametrize("faces", CH2_PATHS)
def test_fold_matrix_equals_oracle_path_matrix(faces):
    c = HexPrism(1.0, 0.8)
    assert np.allclose(fold_matrix(c, faces), path_matrix(faces))


@pytest.mark.parametrize("faces", CH2_PATHS)
def test_exit_normal_is_M_inverse_n_b(faces):
    """展开空间里出射面法向 = M^{-1} n_b（用 oracle 的解析 NORMALS / path_matrix 独立算）。"""
    c = HexPrism(1.0, 0.8)
    _, n_tilde_b = unfold_faces(c, faces)
    M = path_matrix(faces)
    assert np.allclose(n_tilde_b, M.T @ NORMALS[faces[-1]])


@pytest.mark.parametrize("faces", CH2_PATHS)
def test_parallel_path_exit_normal_is_minus_entry_normal(faces):
    """平行光路（refraction_cancels）：展开后的出射面与入射面反向平行，ñ_b = -n_a。"""
    assert refraction_cancels(faces)
    c = HexPrism(1.0, 0.8)
    _, n_tilde_b = unfold_faces(c, faces)
    assert np.allclose(n_tilde_b, -NORMALS[faces[0]])


def test_non_parallel_path_exit_normal_not_minus_entry():
    faces = (3, 1, 4)          # 出射面 4 ≠ M^{-1} 搬回的 -n_3 方向
    assert not refraction_cancels(faces)
    _, n_tilde_b = unfold_faces(HexPrism(1.0, 0.8), faces)
    assert not np.allclose(n_tilde_b, -NORMALS[3])
    assert np.allclose(n_tilde_b, path_matrix(faces).T @ NORMALS[4])


def test_no_reflection():
    c = HexPrism(1.0, 0.8)
    bodies, n_tilde_b = unfold_faces(c, (1, 2))
    assert bodies == [c]
    assert np.allclose(n_tilde_b, c.normal(c.face(2)))
    assert np.allclose(fold_matrix(c, (1, 2)), np.eye(3))


def test_too_short_sequence_rejected():
    with pytest.raises(ValueError):
        unfold_faces(HexPrism(), (3,))
    with pytest.raises(ValueError):
        fold_matrix(HexPrism(), ())


def test_ghosts_keep_outward_normals():
    """每一步镜像 det<0，顶点环序须同步反转；三次级联后外法向仍全部朝外。"""
    bodies, _ = unfold_faces(HexPrism(1.0, 0.8), (3, 1, 5, 7, 4))
    for body in bodies:
        c = body.centroid()
        for f in body.faces:
            assert body.normal(f) @ (body.centroid(f) - c) > 0


def test_pyramid_path_consistency():
    """锥晶上没有面置换群 oracle 可对照，改用矩阵恒等式自证：M 正交、ñ_b = M^T n_b。"""
    p = Pyramid(1.0, 0.5)
    faces = (13, 3, 2, 26, 16)
    bodies, n_tilde_b = unfold_faces(p, faces)
    M = fold_matrix(p, faces)
    assert len(bodies) == 4
    assert np.allclose(M @ M.T, np.eye(3))
    assert np.allclose(n_tilde_b, M.T @ p.normal(p.face(faces[-1])))
    for body in bodies:
        assert isinstance(body, Pyramid)
