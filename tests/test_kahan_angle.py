"""Kahan's 2*atan2 angle formula used by DataLoader.getAlgularityOfLocations."""
import ast
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "fore_swe", "extended_dl_forecasting.py")


def _load_formula():
    """Exec only `pairwise_vector_angle` (the module itself imports sentence-transformers at top level)."""
    with open(SRC) as f:
        tree = ast.parse(f.read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pairwise_vector_angle"]
    ns = {"np": np}
    exec(compile(ast.Module(body=fn, type_ignores=[]), SRC, "exec"), ns)
    return ns["pairwise_vector_angle"]


angle = _load_formula()


def _arccos_angle(v):
    """The formula that was replaced, for comparison."""
    m = np.linalg.norm(v, axis=1)
    return np.arccos(np.clip(v @ v.T / np.outer(m, m), -1.0, 1.0))


def test_finite_and_symmetric_for_nonzero_vectors():
    v = np.random.default_rng(0).normal(size=(64, 3)) * 6371.0
    a = angle(v)
    assert a.shape == (64, 64)
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, a.T)  # exactly symmetric: |u|v-|v|u flips sign only
    assert np.all((a >= 0) & (a <= np.pi))


def test_identical_vectors_give_zero():
    v = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])
    a = angle(v)
    assert np.all(np.diag(a) == 0.0)
    assert a[0, 1] == 0.0 and a[1, 0] == 0.0


def test_orthogonal_and_opposite_vectors():
    v = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-7.0, 0.0, 0.0]])
    a = angle(v)
    assert a[0, 1] == pytest.approx(np.pi / 2, rel=1e-15)
    assert a[0, 2] == pytest.approx(np.pi, rel=1e-15)
    assert a[1, 2] == pytest.approx(np.pi / 2, rel=1e-15)


@pytest.mark.parametrize("delta", [1e-4, 1e-8, 1e-12])
def test_near_coincident_vectors_keep_small_nonzero_angle(delta):
    r = 6371.0
    u = np.array([r, 0.0, 0.0])
    v = np.array([r * np.cos(delta), r * np.sin(delta), 0.0])
    a = angle(np.stack([u, v]))[0, 1]
    assert a > 0
    assert a == pytest.approx(delta, rel=1e-6)
    if delta <= 1e-8:
        # arccos cannot resolve these: near cos=1 it is quantised at ~sqrt(2*eps) ~ 2.1e-8 rad (or collapses
        # to 0), so its relative error is O(1) while Kahan's stays at rounding level
        old = _arccos_angle(np.stack([u, v]))[0, 1]
        assert abs(old - delta) / delta > 0.5


def test_near_antipodal_vectors_resolve_small_deficit():
    delta = 1e-9
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([-np.cos(delta), np.sin(delta), 0.0])
    a = angle(np.stack([u, v]))[0, 1]
    assert np.pi - a == pytest.approx(delta, rel=1e-6)


def test_magnitude_invariance():
    v = np.random.default_rng(1).normal(size=(16, 3))
    scaled = v * np.random.default_rng(2).uniform(1e-3, 1e4, size=(16, 1))
    np.testing.assert_allclose(angle(v), angle(scaled), rtol=1e-12, atol=1e-15)


def test_agrees_with_arccos_away_from_singular_ends():
    v = np.random.default_rng(3).normal(size=(32, 3))
    a, b = angle(v), _arccos_angle(v)
    mask = (b > 1e-3) & (b < np.pi - 1e-3)
    np.testing.assert_allclose(a[mask], b[mask], rtol=1e-10)


def test_dataloader_integration_degrees_shape_order():
    """Real station metadata through the unchanged DataLoader method (needs the full ForeSWE stack)."""
    pytest.importorskip("sentence_transformers")
    pd = pytest.importorskip("pandas")
    import sys
    sys.path.insert(0, os.path.join(REPO, "fore_swe"))
    import extended_dl_forecasting as DL

    info = pd.read_csv(os.path.join(REPO, "data", "snotel_location_info_2.csv"))
    locs = info.loc[:, ["Station Name", "Elevation", "Latitude", "Longitude"]]
    deg = DL.DataLoader.getAlgularityOfLocations(None, locs)
    n = len(locs)
    assert deg.shape == (n, n)
    assert np.all(np.isfinite(deg)) and np.array_equal(deg, deg.T) and np.all(np.diag(deg) == 0)
    assert 0 <= deg.min() and deg.max() <= 180  # degrees, not radians
    # same station order: row i still corresponds to station i (compare against the formula on row-subsets)
    sub = DL.DataLoader.getAlgularityOfLocations(None, locs.iloc[[5, 1, 9]])
    np.testing.assert_array_equal(sub, deg[np.ix_([5, 1, 9], [5, 1, 9])])
