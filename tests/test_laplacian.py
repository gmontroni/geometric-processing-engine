import numpy as np
from numpy.linalg import eigvals
import pytest
from buildoperators.laplaceoperator import build_laplace_operators

@pytest.fixture(scope="module")
def laplacians():
    # These should return lap3D_heat, Lc_heat as numpy arrays
    lap3D_heat, Lc_heat = build_laplace_operators(mesh_path="meshes/sphere.obj")
    return {"lap3D_heat": lap3D_heat, "Lc_heat": Lc_heat}

def test_symmetry(laplacians):
    for name, mat in laplacians.items():
        assert np.allclose(mat, mat.T, atol=1e-12), f"{name} is not symmetric"

def test_row_sum_zero(laplacians):
    for name, mat in laplacians.items():
        row_sums = mat.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-12), f"{name} row sums not zero"

def test_constant_nullspace(laplacians):
    for name, mat in laplacians.items():
        ones = np.ones(mat.shape[0])
        res = mat @ ones
        assert np.allclose(res, 0, atol=1e-12), f"{name} doesn't annihilate constants"

def test_negative_semidefinite(laplacians):
    for name, mat in laplacians.items():
        vals = eigvals(mat)
        max_val = np.max(np.real(vals))
        assert max_val <= 1e-12, f"{name} has positive eigenvalues {max_val}"

# Optional: check if lap3D_heat and Lc_heat have similar action on a sinusoidal test function
def test_compare_on_wave(laplacians):
    lap3D_heat = laplacians["lap3D_heat"]
    Lc_heat = laplacians["Lc_heat"]
    n = lap3D_heat.shape
    x = np.linspace(0, 2*np.pi, n)
    test_func = np.sin(x)
    r1 = lap3D_heat @ test_func
    r2 = Lc_heat @ test_func
    err = np.linalg.norm(r1 - r2) / np.linalg.norm(r2)
    assert err < 0.3, f"Laplacians disagree on test wave: err={err}"

# Optional: check sparsity (skip if using dense)
def test_sparsity(laplacians):
    for name, mat in laplacians.items():
        density = np.count_nonzero(mat) / mat.size
        assert density < 0.2, f"{name} is not sparse enough: density={density}"

