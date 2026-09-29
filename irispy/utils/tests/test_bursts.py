"""
IDL references come from ``iris_burst_check`` and ``iris_sji_burst_check`` (IDL 9.2,
SolarSoft of 2026-09-28) run on whole level 2 files, from which the
``bursts/*_test.fits`` files are cut.
"""

import copy

import numpy as np
import pooch
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.spectrograph import SpectrogramCubeSequence
from irispy.utils.bursts import find_si_iv_bursts, find_sji_bursts
from irispy.utils.constants import DN_UNIT

IRISPY_DATA = "https://github.com/LM-SAL/irispy-data/releases/download/v1"
SI_IV_FILE = get_test_filepath("bursts/iris_l2_20130902_182935_4000005156_raster_t000_r00000_si_iv_test.fits")
SJI_FILE = get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits")
SJI_SUMMARY = get_test_filepath("bursts/iris_sji_burst_check_4000255147_summary.ecsv")


@pytest.fixture(scope="module")
def si_iv_raster():
    return read_spectrograph_lvl2(SI_IV_FILE)


@pytest.fixture
def si_iv_cube(si_iv_raster):
    return copy.deepcopy(si_iv_raster["Si IV 1403"][0])


@pytest.fixture(scope="module")
def idl_si_iv():
    return Table.read(get_test_filepath("bursts/iris_burst_check_4000005156_r00000_thr40.ecsv"))


@pytest.fixture(scope="module")
def sji_1400():
    return read_sji_lvl2(SJI_FILE)


def assert_same_events(labels, pixels, groups):
    """
    The labelled pixels are ``pixels``, and they split into events as ``groups`` does.
    """
    assert labels[pixels].all()
    assert np.count_nonzero(labels) == len(groups)
    pairs = set(zip(groups, labels[pixels], strict=True))
    assert len(pairs) == len(set(groups)) == len(np.unique(labels[labels > 0]))


def assert_si_iv_matches_idl(labels, events, idl, y_offset):
    steps, rows = np.asarray(idl["step"]), np.asarray(idl["y"]) - y_offset
    assert_same_events(labels, (steps, rows), idl["group"])
    for event in events:
        at_peak = (steps == event["step"]) & (rows == event["y"])
        group = idl[idl["group"] == idl["group"][at_peak][0]]
        assert event["npix"] == len(group)
        assert np.isclose(event["intensity"].value, group["intensity"].max(), rtol=1e-6)


@pytest.mark.parametrize("threshold", [80, 80 * u.DN / u.s, 4800 * DN_UNIT["FUV"] / u.min])
def test_si_iv_matches_idl(si_iv_raster, idl_si_iv, threshold):
    # IDL ran with 40 DN/s on data not summed in wavelength, which is 80 DN/s here
    labels, events = find_si_iv_bursts(si_iv_raster, threshold=threshold)
    assert isinstance(labels, SpectrogramCubeSequence)
    assert events.meta["threshold"].value == 40
    assert_si_iv_matches_idl(labels.data[0].data, events, idl_si_iv, idl_si_iv.meta["y_offset"])
    cube = si_iv_raster["Si IV 1403"][0]
    step = events["step"]
    assert isinstance(events["time"], Time)
    assert all(events["time"] == cube.meta["auxiliary times"][step] + cube.meta["exposure time"][step] / 2)
    assert isinstance(events["coordinate"], SkyCoord)
    assert all(events["coordinate"] == cube.wcs.array_index_to_world(step, events["y"], 0)[1])


def test_si_iv_default_threshold(si_iv_cube, idl_si_iv):
    labels, events = find_si_iv_bursts(si_iv_cube)
    # IDL's threshold, halved because the data are not summed in wavelength
    assert np.isclose(events.meta["threshold"].value, idl_si_iv.meta["default_threshold"] / 2, rtol=1e-6)
    assert len(events) == 0
    assert not labels.data.any()


def test_si_iv_sequence_numbers_events_on(si_iv_cube):
    later = copy.deepcopy(si_iv_cube)
    later.wcs.wcs.dateobs = "2013-09-02T18:40:00"  # its own obstime, as a separate read gives
    labels, events = find_si_iv_bursts(SpectrogramCubeSequence([si_iv_cube, later]), threshold=80)
    assert list(events["label"]) == [1, 2, 3, 4, 5, 6]
    assert list(events["raster"]) == [0, 0, 0, 1, 1, 1]
    first, second = (cube.data for cube in labels.data)
    np.testing.assert_array_equal(second, np.where(first > 0, first + 3, 0))
    assert all(events["coordinate"][3:] == events["coordinate"][:3])


def test_si_iv_threshold_scales_with_summing(si_iv_cube):
    si_iv_cube.meta["SUMSPAT"] = 2
    si_iv_cube.meta["SUMSPTRF"] = 4
    _, events = find_si_iv_bursts(si_iv_cube, threshold=80)
    assert events.meta["threshold"] == 80 * 2 * 4 / 2 * DN_UNIT["FUV"] / u.s


def test_si_iv_median_test_rejects_spikes(si_iv_cube):
    si_iv_cube.data[5, 100, 20] = 1e5  # a one-bin spike, like a particle hit
    _, events = find_si_iv_bursts(si_iv_cube, threshold=80)
    assert 5 not in events["step"]
    _, events = find_si_iv_bursts(si_iv_cube, threshold=80, median_factor=None)
    assert list(events[events["step"] == 5]["y"]) == [100]


def test_si_iv_ignores_missing_data(si_iv_cube):
    si_iv_cube.data[3] = -200
    si_iv_cube.meta["exposure time"][3:5] = 0 * u.s  # step 4 keeps its data
    si_iv_cube.data[10, :, ::2] = -150  # iris_getwindata.pro treats values below -10 as missing too
    si_iv_cube.data[10, :, 21] = np.nan
    labels, _ = find_si_iv_bursts(si_iv_cube, threshold=1)
    assert not labels.data[3:5].any()
    assert labels.data[10].any()


def test_si_iv_errors(si_iv_raster):
    other = read_spectrograph_lvl2(
        get_test_filepath("sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits"),
        spectral_windows="C II 1336",
    )
    with pytest.raises(ValueError, match="No spectral window covers"):
        find_si_iv_bursts(other)
    with pytest.raises(ValueError, match="no wavelength bins"):
        find_si_iv_bursts(si_iv_raster["Si IV 1403"][0][:, :, :3])
    with pytest.raises(ValueError, match="unscaled"):
        find_si_iv_bursts(read_spectrograph_lvl2(SI_IV_FILE, memmap=True))
    with pytest.raises(ValueError, match="slice with a range"):
        find_si_iv_bursts(si_iv_raster["Si IV 1403"][0][3])
    with pytest.raises(ValueError, match="must be in DN"):
        find_si_iv_bursts(si_iv_raster["Si IV 1403"][0].apply_exposure_time_correction())


def test_sji_matches_idl(sji_1400):
    idl = Table.read(get_test_filepath("bursts/iris_sji_burst_check_4000255147_pixels.ecsv"))
    summary = Table.read(SJI_SUMMARY)
    labels, events = find_sji_bursts(sji_1400)
    for frame in range(len(sji_1400.data)):
        pixels = idl[idl["frame"] == frame]
        assert_same_events(labels.data[frame].ravel(), pixels["pixel"], pixels["group"])
        idl_npix = np.unique(pixels["group"], return_counts=True)[1]
        assert sorted(events["npix"][events["frame"] == frame]) == sorted(idl_npix)
    # Frame 1 (69 in the file) holds a 2100.50 DN pixel 0.04 DN below the threshold; IDL rejects it too.
    idl_frames = np.unique(idl["idl_frame"])
    np.testing.assert_array_equal(np.bincount(events["frame"], minlength=3), summary["nevents"][idl_frames])
    np.testing.assert_array_equal(events["intensity"].value, sji_1400.data[events["frame"], events["y"], events["x"]])
    assert np.all(events["intensity"] >= events["threshold"])
    assert events.colnames == ["label", "frame", "npix", "threshold", "y", "x", "time", "coordinate", "intensity"]
    assert isinstance(events["time"], Time)
    assert isinstance(events["coordinate"], SkyCoord)
    coordinate, time = sji_1400.wcs.pixel_to_world(events["x"], events["y"], events["frame"])
    assert all(events["coordinate"] == coordinate)
    assert all(events["time"] == time)
    assert set(labels.extra_coords.keys()) == set(sji_1400.extra_coords.keys())


def test_sji_edge_and_small_events(sji_1400):
    sji = copy.deepcopy(sji_1400)
    sji.data[0, :2, :2] = 1e5  # IDL's REGION_GROW leaves edge pixels out of every region
    sji.mask[0, :2, :2] = False  # they are fill in this file
    sji.data[0, 200, 200] = 1e5
    labels, _ = find_sji_bursts(sji)
    assert labels.data[0, :2, :2].all()
    assert labels.data[0, 200, 200] == 0
    labels, _ = find_sji_bursts(sji, min_pixels=1)
    assert labels.data[0, 200, 200] > 0


def test_sji_frames_without_valid_pixels(sji_1400):
    sji = copy.deepcopy(sji_1400)
    sji.mask[1] = True
    _, events = find_sji_bursts(sji)
    assert 1 not in events["frame"]
    sji.data[0, 100, 100] = np.inf  # spoils the frame's statistics, so it must warn
    with pytest.warns(RuntimeWarning, match="invalid value"):
        find_sji_bursts(sji)


def test_sji_errors(sns_sjicube_1330, sji_1400):
    with pytest.raises(ValueError, match="not 1330"):
        find_sji_bursts(sns_sjicube_1330)
    with pytest.raises(ValueError, match="unscaled"):
        find_sji_bursts(read_sji_lvl2(SJI_FILE, memmap=True))
    with pytest.raises(ValueError, match="slice with a range"):
        find_sji_bursts(sji_1400[0])


@pytest.mark.remote_data
def test_si_iv_whole_file_matches_idl(idl_si_iv):
    # Only the Si IV 1403 window of the raster
    filename = pooch.retrieve(
        f"{IRISPY_DATA}/iris_l2_20130902_182935_4000005156_raster_t000_r00000_si_iv.fits.gz",
        known_hash="ac50a0255b73af1610702653e17d3b3b9c8fc37bc487a313e1c9fb3a2983428a",
    )
    labels, events = find_si_iv_bursts(read_spectrograph_lvl2(filename), threshold=80)
    assert_si_iv_matches_idl(labels.data[0].data, events, idl_si_iv, 0)


@pytest.mark.remote_data
def test_sji_first_frames_match_idl():
    # The first 50 of the 400 frames; IDL treats each frame on its own
    filename = pooch.retrieve(
        f"{IRISPY_DATA}/iris_l2_20130902_163935_4000255147_SJI_1400_t000_f050.fits.gz",
        known_hash="b9a0b8cf2d98f5e1121000a14168079b113fdb0b169668ec1213411915afdb8f",
    )
    summary = Table.read(SJI_SUMMARY)[:50]
    labels, events = find_sji_bursts(read_sji_lvl2(filename))
    np.testing.assert_array_equal(np.bincount(events["frame"], minlength=50), summary["nevents"])
    np.testing.assert_array_equal(np.count_nonzero(labels.data, axis=(1, 2)), summary["npix"])
