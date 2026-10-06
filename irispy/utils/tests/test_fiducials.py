import numpy as np
import pooch
import pytest

import astropy.units as u

from irispy.data.test import get_test_filepath
from irispy.io import read_files
from irispy.spectrograph import SpectrogramCubeSequence
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.fiducials import find_fiducials

# 89.7 rows apart, the Level 2 separation at the 1 arcsec pixels of make_test_spectrogram_cube
MARKS = np.array([40.3, 130.0])
MG_FEATURES_FILE = "mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits"


def make_window(marks=MARKS, *, darkest=0.9, rows=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(rows)
    profile = 1 - sum((darkest * np.exp(-((y - mark) ** 2) / (2 * 1.3**2)) for mark in marks), np.zeros(rows))
    data = 100 * profile[np.newaxis, :, np.newaxis] * (1 + 0.01 * rng.standard_normal((4, rows, 8)))
    cube = make_test_spectrogram_cube(data, np.linspace(279.5, 279.7, 8) * u.nm)
    cube.meta["OBSID"] = "3893010094"  # SpectrogramCubeSequence wants one OBSID
    return cube


def test_find_fiducials_sub_pixel():
    positions, depths = find_fiducials(make_window())
    np.testing.assert_allclose(positions, MARKS, atol=0.05)
    np.testing.assert_allclose(depths, 0.9, atol=0.03)


def test_find_fiducials_averages_a_sequence():
    # Half the exposures show the marks, so the mean profile dips by 0.45: below the default min_depth
    sequence = SpectrogramCubeSequence([make_window(seed=1), make_window(marks=[], seed=2)], common_axis=0)
    assert find_fiducials(sequence)[0].size == 0
    np.testing.assert_allclose(find_fiducials(sequence, min_depth=0.4)[0], MARKS, atol=0.05)


@pytest.mark.parametrize("hide", ["mask", "nan"])
def test_find_fiducials_ignores_masked_and_missing_rows(hide):
    cube = make_window()
    if hide == "mask":
        cube.mask = np.zeros(cube.data.shape, dtype=bool)
        cube.mask[:, 125:136] = True
    else:
        cube.data[:, 125:136] = np.nan
    np.testing.assert_allclose(find_fiducials(cube)[0], MARKS[:1], atol=0.05)


def test_find_fiducials_needs_a_deep_dip():
    assert find_fiducials(make_window(darkest=0.3))[0].size == 0
    assert find_fiducials(make_window(darkest=0.3), min_depth=0.2)[0].size == 2


def test_find_fiducials_prefers_the_pair_at_the_mark_separation():
    # A deeper dip between the marks does not displace the pair
    cube = make_window(marks=[*MARKS, 75.0])
    cube.data[:, 74:77] *= 0.1
    np.testing.assert_allclose(find_fiducials(cube)[0], MARKS, atol=0.05)


def test_find_fiducials_takes_the_most_significant_dip_without_a_pair():
    cube = make_window(marks=MARKS[:1])
    cube.data[:, 99:102] *= 0.4
    np.testing.assert_allclose(find_fiducials(cube)[0], MARKS[:1], atol=0.05)


def test_find_fiducials_skips_single_dark_rows():
    # The PSF spreads a mark over its neighbours; a lone dark row is a bad pixel row
    cube = make_window(marks=[])
    cube.data[:, 100] = 0
    assert find_fiducials(cube)[0].size == 0


def test_find_fiducials_skips_dips_below_zero():
    # Background subtraction can take FUV rows below zero: blocking more than all the light is not a mark
    cube = make_window(marks=[])
    cube.data[:, 99:102] = -300
    assert find_fiducials(cube)[0].size == 0


def test_find_fiducials_rejects_unscaled_data():
    raster = read_files(get_test_filepath(MG_FEATURES_FILE), memmap=True)
    with pytest.raises(ValueError, match="unscaled"):
        find_fiducials(raster["Mg II k 2796"])


def test_level2_marks():
    # 80.97 and 620.01 in the whole Mg II k window of this file
    raster = read_files(get_test_filepath(MG_FEATURES_FILE))
    for window in ("Mg II k 2796", "Mg II h 2803"):
        positions, depths = find_fiducials(raster[window])
        np.testing.assert_allclose(positions, [81.0, 620.0], atol=0.1)
        assert np.all(depths > 0.85)


def test_level2_fuv_without_marks():
    # OBS 3824262996 near the limb: the upper mark is off the disk and these FUV cuts are too noisy
    raster = read_files(
        get_test_filepath(
            "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
        )
    )
    np.testing.assert_allclose(find_fiducials(raster["Mg II k 2796"])[0], [238.0], atol=0.1)
    for window in ("O I 1356", "Si IV 1394"):
        assert find_fiducials(raster[window])[0].size == 0


@pytest.mark.remote_data
def test_level2_fuv_and_nuv_marks():
    raster = read_files(
        pooch.retrieve(
            "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20150130_055150_3893010094_cutout_raster.fits.gz",
            known_hash="603aa2a5dbe0cf9738e3628451dd24361da05b8eeecd42962ac1253db05888eb",
        )
    )
    # Rows 22.85, 22.88 and 23.02: the Level 2 pipeline aligns the FUV to the NUV
    for window in ("C II 1336", "Si IV 1394", "Mg II k 2796"):
        np.testing.assert_allclose(find_fiducials(raster[window])[0], [23.0], atol=0.2)
