"""
The IDL references were made with ``iris_get_mg_features_lev2`` (IDL 9.2, SolarSoft of
2026-09-28) on a whole level 2 file; the ``mg_features/*_test.fits`` file is cut from
it.
"""

import copy

import numpy as np
import pytest

import astropy.units as u
from astropy.table import Table

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.utils.mg_features import _spline, calculate_mg_features

WINDOWS = {"k": "Mg II k 2796", "h": "Mg II h 2803"}


@pytest.fixture(scope="module")
def raster():
    # Steps 13, 32 and 57 of the file, with its Mg II window split into one window per line.
    return read_spectrograph_lvl2(
        get_test_filepath("mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits")
    )


@pytest.fixture(scope="module")
def idl():
    return np.load(get_test_filepath("mg_features/iris_get_mg_features_lev2_4000005156_r00000.npz"))


@pytest.mark.parametrize(("index", "line"), list(enumerate(WINDOWS)))
def test_matches_idl(raster, idl, index, line):
    cube = raster[WINDOWS[line]][0]
    features = calculate_mg_features(cube, lines=(line,))
    missing = (cube.data == -200).any(axis=-1)
    for feature, key, tolerance in [("3", "lc", 0.01), ("2v", "bp", 0.7), ("2r", "rp", 0.7)]:
        ours = np.stack([features[f"{line}{feature}_{kind}"].data for kind in ("velocity", "intensity")], axis=-1)
        theirs = idl[key][..., index]
        assert np.isnan(ours[missing]).all()  # IDL interpolates through the -200 fill
        assert not np.any(np.isfinite(ours[..., 0]) & np.isnan(theirs[..., 0]))
        both = np.isfinite(ours[..., 0]) & np.isfinite(theirs[..., 0])
        assert np.count_nonzero(np.isnan(ours[..., 0]) & np.isfinite(theirs[..., 0]) & ~missing) <= 10
        # IDL's peak refinement extrapolates one spline interval, which moves its peaks by up to 0.6 km/s.
        velocity = np.abs(ours[..., 0] - theirs[..., 0])[both]
        assert np.mean(velocity <= tolerance) >= 0.998
        intensity = np.abs(ours[..., 1] / theirs[..., 1] - 1)[both]
        assert np.percentile(intensity, 99) < 0.02


def test_output(raster):
    features = calculate_mg_features(raster["Mg II k 2796"][0], lines=("k",))
    assert list(features.keys()) == [
        f"k{feature}_{kind}" for feature in ("2v", "3", "2r") for kind in ("velocity", "intensity")
    ]
    velocity = features["k3_velocity"]
    assert velocity.unit == u.km / u.s
    assert velocity.data.shape == (3, 771)
    np.testing.assert_array_equal(velocity.mask, np.isnan(velocity.data))
    assert features["k3_intensity"].unit == raster["Mg II k 2796"][0].unit


def test_missing_data(raster):
    cube = copy.deepcopy(raster["Mg II k 2796"][0])
    cube.data[1, 300, 10] = -200
    features = calculate_mg_features(cube, lines=("k",))
    assert np.isnan(features["k3_velocity"].data[1, 300])
    assert np.isfinite(features["k3_velocity"].data[1, 299])


def test_window_must_cover_line(raster):
    with pytest.raises(ValueError, match="does not cover Mg II h"):
        calculate_mg_features(raster["Mg II k 2796"][0], lines=("h",))


def test_spline_matches_idl():
    idl = Table.read(get_test_filepath("mg_features/idl_spline.ecsv"))
    x = np.linspace(-2.5, 2.5, 21)
    y = np.exp(-(x**2)) + 0.1 * x
    for tension in (0, 1):
        np.testing.assert_allclose(_spline(x, y, idl["t"], tension=tension), idl[f"tension_{tension}"], atol=1e-8)
