"""
The IDL references were made with ``iris_prep_wavecorr_l2`` (IDL 9.2, SolarSoft of
2026-09-28) on whole level 2 files; the ``wavelength_drift/*_test.fits`` file is cut
from one of them.
"""

import numpy as np
import pooch
import pytest

import astropy.units as u
from astropy.table import QTable

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.utils.wavelength_drift import _fit_drift, calculate_wavelength_drift

ARCHIVE = "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed"
LINES = ["Ni I", "Mn I", "Fe I", "O I", "Fe II"]
CROP = get_test_filepath(
    "wavelength_drift/iris_l2_20130902_182935_4000005156_raster_t000_r00000_wavelength_drift_test.fits"
)
CROP_STEPS = [5, 22, 30, 40, 49, 60]


def idl_reference(obsid):
    return QTable.read(get_test_filepath(f"wavelength_drift/iris_prep_wavecorr_l2_{obsid}_r00000.ecsv"))


def assert_shifts_match_idl(table, idl):
    """
    Our Gaussian fits converge further than IDL's CURVEFIT, which moves the centres by
    up to 2 mÅ.
    """
    assert all(np.abs((table["time"] - idl["time"]).to_value(u.s)) < 1e-6)
    for name in LINES:
        ours, theirs = table[name].to_value(u.AA), idl[name].to_value(u.AA)
        difference = np.abs(ours - theirs)[np.isfinite(ours) & np.isfinite(theirs)]
        if name == "Fe II":
            # Fits to the weak Fe II line can land on spikes, and are kept or rejected near the range limit.
            assert np.mean(np.isnan(ours) == np.isnan(theirs)) >= 0.95
            assert np.median(difference) < 1e-3
        else:
            np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
            assert difference.max() < 2.5e-3


def seconds(table):
    return (table["time"] - table["time"][0]).to_value(u.s)


def test_shifts_match_idl():
    table = calculate_wavelength_drift(read_spectrograph_lvl2(CROP))
    assert table.colnames == ["time", "raster", "step", *LINES, "nuv", "fuv"]
    assert list(table["step"]) == list(range(len(CROP_STEPS)))
    assert_shifts_match_idl(table, idl_reference("4000005156")[CROP_STEPS])
    assert np.isfinite(table["nuv"]).all()
    assert np.isfinite(table["fuv"]).all()


def test_drift_fit_matches_idl():
    idl = idl_reference("3824262996")
    for drift, line, outlier in [("nuv", "Ni I", 0.08), ("fuv", "O I", 0.05)]:
        fitted = _fit_drift(seconds(idl), idl[line].to_value(u.AA), outlier, drift)
        np.testing.assert_allclose(fitted, idl[drift].to_value(u.AA), atol=1e-8)


def test_drift_fit_leaves_out_missing_shifts():
    # IDL keeps the exposure without a Ni I shift in its fit, so MPFIT returns its starting guess
    idl = idl_reference("4000005156")
    guess = 0.1 + 0.01 * np.sin(2 * np.pi * seconds(idl) / 5856 + 1)
    np.testing.assert_allclose(idl["nuv"].to_value(u.AA), guess, atol=1e-10)
    shifts = idl["Ni I"].to_value(u.AA)
    fitted = _fit_drift(seconds(idl), shifts, 0.08, "nuv")
    assert np.sqrt(np.nanmean((shifts - fitted) ** 2)) < 1e-3
    assert np.sqrt(np.nanmean((shifts - guess) ** 2)) > 0.1


def test_missing_lines():
    nuv_only = read_spectrograph_lvl2(CROP, spectral_windows=["Ni I 2799", "Mn I 2802", "Fe I 2805"])
    with pytest.warns(UserWarning, match="Too few shifts to fit the FUV drift"):
        table = calculate_wavelength_drift(nuv_only)
    assert np.isnan(table["O I"]).all()
    assert np.isnan(table["fuv"]).all()
    assert np.isfinite(table["nuv"]).all()
    other = read_spectrograph_lvl2(
        get_test_filepath("sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits"),
        spectral_windows="C II 1336",
    )
    with pytest.raises(ValueError, match="no window with any of the reference lines"):
        calculate_wavelength_drift(other)


@pytest.mark.remote_data
@pytest.mark.parametrize(
    ("path", "known_hash"),
    [
        (
            "2013/09/02/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_raster.tar.gz",
            "91211a52e278fb6e535242d4d6064facf9f93cf24f0a433c276ace1b2d621e7d",
        ),
        (
            "2014/07/08/20140708_114109_3824262996/iris_l2_20140708_114109_3824262996_raster.tar.gz",
            "21cff86fd0064936ce6807b1334ea1d3d50f3d358e5888810fba8e9bc1118567",
        ),
    ],
)
def test_whole_file_matches_idl(path, known_hash):
    first_raster = path.split("/")[-1].replace(".tar.gz", "_t000_r00000.fits")
    (filename,) = pooch.retrieve(f"{ARCHIVE}/{path}", known_hash=known_hash, processor=pooch.Untar([first_raster]))
    table = calculate_wavelength_drift(read_spectrograph_lvl2(filename))
    idl = idl_reference(first_raster.split("_")[4])
    assert_shifts_match_idl(table, idl)
    if first_raster.startswith("iris_l2_20140708"):
        # The shift differences carry over to the drifts.
        for drift in ["nuv", "fuv"]:
            np.testing.assert_allclose(table[drift].to_value(u.AA), idl[drift].to_value(u.AA), atol=2.5e-3)
