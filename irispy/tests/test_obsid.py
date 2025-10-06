import pytest

import astropy.units as u

from irispy.obsid import ObsID

OBSIDS = [
    3400109162,
    3677508065,
    3880903651,
    4050607445,
]
INVALID_OBSID = [
    4643502010,
    4050607495,
    3880903650,
    3680903685,
    335987081297,
    40,
]
TEST_DATA = {}
TEST_DATA["exptime"] = [
    8 * u.s,
    8 * u.s,
    30 * u.s,
    4 * u.s,
]
TEST_DATA["raster_desc"] = [
    "Very large coarse 64-step raster",
    "Very large dense 96-step raster",
    "Small sit-and-stare",
    "Very large dense raster (tight timing)",
]
TEST_DATA["sjis"] = [
    "C II   Si IV   Mg II h/k",
    "C II   Si IV   Mg II h/k   Mg II w",
    "Si IV",
    "Si IV   Mg II h/k",
]
TEST_DATA["binning"] = [
    "Spatial x 2, Spectral x 2",
    "Spatial x 1, Spectral x 1",
    "Spatial x 2, Spectral x 8",
    "Spatial x 2, Spectral x 2",
]
TEST_DATA["fuv_binning"] = [
    "FUV binned same as NUV",
    "FUV spectrally rebinned x 4",
    "FUV spectrally rebinned x 8",
    "FUV spectrally rebinned x 4",
]
TEST_DATA["sji_cadence"] = [
    "SJI cadence default",
    "SJI cadence 0.5x faster",
    "SJI cadence default",
    "SJI cadence 10s",
]
TEST_DATA["linelist"] = [
    "Large linelist",
    "Flare linelist 1",
    "Full readout",
    "Small linelist",
]


@pytest.mark.parametrize(
    ("attr_name", "test_input", "expected_output"),
    [(name, obs, output[i]) for (name, output) in TEST_DATA.items() for i, obs in enumerate(OBSIDS)],
)
def test_attribute(attr_name, test_input, expected_output):
    assert ObsID(test_input)[attr_name] == expected_output


@pytest.mark.parametrize("test_input", [INVALID_OBSID])
def test_invalid_obsid(test_input):
    with pytest.raises(ValueError, match=r"Invalid OBS ID: must have 10 digits."):
        ObsID(test_input)
