"""
The IDL references were made with ``iris_burst_check`` and ``iris_sji_burst_check`` (IDL
9.2, SolarSoft of 2026-09-28) on whole level 2 files; the ``bursts/*_test.fits`` files
are cut from them.
"""

import copy

import numpy as np
import pooch
import pytest

import astropy.units as u
from astropy.table import Table

from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.spectrograph import SpectrogramCubeSequence
from irispy.utils.bursts import find_si_iv_bursts, find_sji_bursts

ARCHIVE = "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2013/09/02"


@pytest.fixture(scope="module")
def si_iv_raster():
    return read_spectrograph_lvl2(
        get_test_filepath("bursts/iris_l2_20130902_182935_4000005156_raster_t000_r00000_si_iv_test.fits")
    )


@pytest.fixture
def si_iv_cube(si_iv_raster):
    return copy.deepcopy(si_iv_raster["Si IV 1403"][0])


@pytest.fixture(scope="module")
def idl_si_iv():
    return Table.read(get_test_filepath("bursts/iris_burst_check_4000005156_r00000_thr40.ecsv"))


@pytest.fixture(scope="module")
def sji_1400():
    return read_sji_lvl2(get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits"))


def assert_same_events(labels, pixels, groups):
    """
    The labelled pixels are ``pixels``, and they split into events as ``groups`` does.
    """
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


def test_si_iv_matches_idl(si_iv_raster, idl_si_iv):
    # IDL ran with 40 DN/s on data not summed in wavelength, which is 80 DN/s here
    labels, events = find_si_iv_bursts(si_iv_raster, threshold=80)
    assert isinstance(labels, SpectrogramCubeSequence)
    assert events.meta["threshold"].value == 40
    assert_si_iv_matches_idl(labels.data[0].data, events, idl_si_iv, idl_si_iv.meta["y_offset"])
    cube = si_iv_raster["Si IV 1403"][0]
    assert all(events["time"] == cube.axis_world_coords("time", wcs=cube.extra_coords)[0][events["step"]])


def test_si_iv_default_threshold(si_iv_cube, idl_si_iv):
    labels, events = find_si_iv_bursts(si_iv_cube)
    # IDL's threshold, halved because the data are not summed in wavelength
    assert np.isclose(events.meta["threshold"].value, idl_si_iv.meta["default_threshold"] / 2, rtol=1e-6)
    assert len(events) == 0
    assert not labels.data.any()


def test_si_iv_sequence_numbers_events_on(si_iv_cube):
    labels, events = find_si_iv_bursts(SpectrogramCubeSequence([si_iv_cube, si_iv_cube]), threshold=80)
    assert list(events["label"]) == [1, 2, 3, 4, 5, 6]
    assert list(events["raster"]) == [0, 0, 0, 1, 1, 1]
    first, second = (cube.data for cube in labels.data)
    np.testing.assert_array_equal(second, np.where(first > 0, first + 3, 0))


def test_si_iv_median_test_rejects_spikes(si_iv_cube):
    si_iv_cube.data[5, 100, 20] = 1e5  # one bright bin, as a particle hit gives
    _, events = find_si_iv_bursts(si_iv_cube, threshold=80)
    assert 5 not in events["step"]
    _, events = find_si_iv_bursts(si_iv_cube, threshold=80, median_factor=None)
    assert list(events[events["step"] == 5]["y"]) == [100]


def test_si_iv_ignores_missing_data(si_iv_cube):
    si_iv_cube.data[3] = -200
    si_iv_cube.meta["exposure time"][3] = 0 * u.s
    si_iv_cube.data[10, :, ::2] = -150  # iris_getwindata.pro treats values below -10 as missing too
    labels, _ = find_si_iv_bursts(si_iv_cube, threshold=1)
    assert not labels.data[3].any()
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
    memmap = read_spectrograph_lvl2(
        get_test_filepath("bursts/iris_l2_20130902_182935_4000005156_raster_t000_r00000_si_iv_test.fits"),
        memmap=True,
    )
    with pytest.raises(ValueError, match="unscaled"):
        find_si_iv_bursts(memmap)


def test_sji_matches_idl(sji_1400):
    idl = Table.read(get_test_filepath("bursts/iris_sji_burst_check_4000255147_pixels.ecsv"))
    summary = Table.read(get_test_filepath("bursts/iris_sji_burst_check_4000255147_summary.ecsv"))
    labels, events = find_sji_bursts(sji_1400)
    for frame in range(len(sji_1400.data)):
        pixels = idl[idl["frame"] == frame]
        assert_same_events(labels.data[frame].ravel(), pixels["pixel"], pixels["group"])
    # Frame 1 (69 in the file) holds a 2100.50 DN pixel 0.04 DN below the threshold; IDL rejects it too.
    idl_frames = np.unique(idl["idl_frame"])
    np.testing.assert_array_equal(np.bincount(events["frame"], minlength=3), summary["nevents"][idl_frames])
    np.testing.assert_array_equal(events["intensity"].value, sji_1400.data[events["frame"], events["y"], events["x"]])
    assert np.all(events["intensity"] >= events["threshold"])


def test_sji_edge_and_small_events(sji_1400):
    sji = copy.deepcopy(sji_1400)
    sji.data[0, :2, :2] = 1e5  # IDL's REGION_GROW never grows from edge pixels
    sji.mask[0, :2, :2] = False  # they are fill in this file
    sji.data[0, 200, 200] = 1e5
    labels, _ = find_sji_bursts(sji)
    assert labels.data[0, :2, :2].all()
    assert labels.data[0, 200, 200] == 0
    labels, _ = find_sji_bursts(sji, min_pixels=1)
    assert labels.data[0, 200, 200] > 0


def test_sji_errors(sns_sjicube_1330):
    with pytest.raises(ValueError, match="not 1330"):
        find_sji_bursts(sns_sjicube_1330)
    memmap = read_sji_lvl2(
        get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits"), memmap=True
    )
    with pytest.raises(ValueError, match="unscaled"):
        find_sji_bursts(memmap)


@pytest.mark.remote_data
def test_si_iv_whole_file_matches_idl(idl_si_iv):
    (filename,) = pooch.retrieve(
        f"{ARCHIVE}/20130902_182935_4000005156/iris_l2_20130902_182935_4000005156_raster.tar.gz",
        known_hash="91211a52e278fb6e535242d4d6064facf9f93cf24f0a433c276ace1b2d621e7d",
        processor=pooch.Untar(members=["iris_l2_20130902_182935_4000005156_raster_t000_r00000.fits"]),
    )
    raster = read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403")
    labels, events = find_si_iv_bursts(raster, threshold=80)
    assert_si_iv_matches_idl(labels.data[0].data, events, idl_si_iv, 0)


@pytest.mark.remote_data
def test_sji_whole_file_matches_idl():
    filename = pooch.retrieve(
        f"{ARCHIVE}/20130902_163935_4000255147/iris_l2_20130902_163935_4000255147_SJI_1400_t000.fits.gz",
        known_hash="1f424de4420b729385e81b00df4ba4d868a121486686f17cd1ecdbe7754ee78b",
    )
    summary = Table.read(get_test_filepath("bursts/iris_sji_burst_check_4000255147_summary.ecsv"))
    labels, events = find_sji_bursts(read_sji_lvl2(filename))
    np.testing.assert_array_equal(np.bincount(events["frame"], minlength=400), summary["nevents"])
    np.testing.assert_array_equal(np.count_nonzero(labels.data, axis=(1, 2)), summary["npix"])
