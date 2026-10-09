import copy

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.wcs import WCS

from sunpy.coordinates import Helioprojective

from irispy._wcs import _RasterWCS
from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import _create_tabular_wcs, _nuv_t_obs_from_source_filenames, read_spectrograph_lvl2
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED


def test_spectral_windows_order_read_spectrograph_lvl2(sns_sg_file):
    # Requesting windows in an order different from the file order must not swap the data.
    all_windows = read_spectrograph_lvl2(sns_sg_file)
    subset = read_spectrograph_lvl2(sns_sg_file, spectral_windows=["Mg II k 2796", "Si IV 1394"])
    assert set(subset) == {"Mg II k 2796", "Si IV 1394"}
    for key in subset:
        np.testing.assert_array_equal(subset[key][0].data, all_windows[key][0].data)


def test_sns_read_spectrograph_lvl2(sns_sg_file):
    raster_collection = read_spectrograph_lvl2(sns_sg_file)
    assert list(raster_collection.keys()) == [
        "C II 1336",
        "Fe XII 1349",
        "O I 1356",
        "Si IV 1394",
        "Si IV 1403",
        "2832",
        "2814",
        "Mg II k 2796",
    ]
    # Simple repr check
    assert str(raster_collection)
    # We do not expect any metadata to be present on the collection
    assert raster_collection.meta is None

    si_iv = raster_collection["Si IV 1403"]
    # Simple repr check
    assert str(si_iv)
    # Test data only has a sequence of 1 long
    assert len(si_iv) == 1
    # The primary fits header is attached to the sequence
    assert si_iv.meta is not None
    # Meta is attached one level down to the individual cube for now.
    assert si_iv[0].meta is not None
    meta = si_iv[0].meta
    assert si_iv[0].data.shape == (187, 40, 29)  # (lambda, y, x)
    assert np.all(si_iv[0].data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SPEC" == meta.instrument
    assert meta.detector == "FUV2"
    assert meta.spectral_band == "FUV"
    assert meta.automatic_exposure_control_enabled is True
    assert meta.date_end.isot == "2021-09-05T05:07:27.400"
    assert meta.date_reference.isot == "2021-09-05T00:18:33.810"
    assert meta.date_start.isot == "2021-09-05T00:18:33.810"
    assert_quantity_allclose(meta.distance_to_sun, 1.00827638 * u.AU)
    assert meta.exposure_control_triggers_in_observation == 0
    assert meta.exposure_control_triggers_in_raster == 0
    assert len(meta.fits_header) == 380 == (len(meta.keys()) + 13)  # History is missing
    assert meta.fov_center == SkyCoord(
        Tx=meta.get("XCEN"),
        Ty=meta.get("YCEN"),
        unit=u.arcsec,
        frame=Helioprojective,
    )
    assert meta.key_comments == {}
    assert meta.number_of_unique_raster_positions == 1
    assert meta.number_of_raster_positions == 1
    assert meta.observation_includes_saa is True
    assert meta.observatory_at_high_latitude is False
    assert meta.observing_campaign_start.isot == "2021-09-05T00:18:33.640"
    assert meta.observing_mode_description == "Medium sit-and-stare 0.3x60 1s  C II   Si IV   Mg II h/k   Mg II w s"
    assert meta.observing_mode_id == 3620258102
    assert meta.processing_level == 2
    assert meta.raster_fov_width_x == 0.16635 * u.arcsec
    assert meta.raster_fov_width_y == 66.54 * u.arcsec
    assert meta.satellite_rotation == 8.09432e-05 * u.deg
    assert meta.spatial_summing_factor == 1
    assert_quantity_allclose(meta.spectral_range, (1398.60550787, 1406.03398787) * u.angstrom)
    assert meta.spectral_summing_factor == 2
    assert meta.tracking_mode_enabled is False

    # TODO: Decide if I want to set observer_location, observer_radial_velocity, rsun_angular, run_meters
    # These are more WCS properties...
    assert meta.observer_location is None
    assert meta.rsun_angular is None
    assert meta.rsun_meters is None


def test_raster_all_files_read_spectrograph_lvl2(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    assert list(raster_collection.keys()) == [
        "C II 1336",
        "1343",
        "Fe XII 1349",
        "O I 1356",
        "Si IV 1403",
        "2832",
        "2826",
        "2814",
        "Mg II k 2796",
    ]
    # Simple repr check
    assert str(raster_collection)
    # We do not expect any metadata to be present on the collection
    assert raster_collection.meta is None

    si_iv = raster_collection["Si IV 1403"]
    # Simple repr check
    assert str(si_iv)
    # One raster per file.
    assert len(si_iv) == len(raster_sg_files)
    # The primary fits header is attached to the sequence
    assert si_iv.meta is not None
    # Meta is attached one level down to the individual cube for now.
    assert si_iv[0].meta is not None
    meta = si_iv[0].meta
    assert si_iv[0].data.shape == (8, 109, 29)  # (lambda, y, x)
    assert np.all(si_iv[0].data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SPEC" == meta.instrument
    assert meta.detector == "FUV2"
    assert meta.spectral_band == "FUV"
    assert meta.automatic_exposure_control_enabled is True
    assert meta.date_end.isot == "2014-03-29T14:10:44.500"
    assert meta.date_reference.isot == "2014-03-29T14:09:39.000"
    assert meta.date_start.isot == "2014-03-29T14:09:39.000"
    assert_quantity_allclose(meta.distance_to_sun, 0.99849015 * u.AU)
    assert meta.exposure_control_triggers_in_observation == 526
    assert meta.exposure_control_triggers_in_raster == 0
    assert len(meta.fits_header) == 412 == (len(meta.keys()) + 12)  # History is missing
    assert meta.fov_center == SkyCoord(
        Tx=meta.get("XCEN"),
        Ty=meta.get("YCEN"),
        unit=u.arcsec,
        frame=Helioprojective,
    )
    assert meta.key_comments == {}
    assert meta.number_of_unique_raster_positions == 8
    assert meta.number_of_raster_positions == 180
    assert meta.observation_includes_saa is True
    assert meta.observatory_at_high_latitude is False
    assert meta.observing_campaign_start.isot == "2014-03-29T14:09:38.830"
    assert meta.observing_mode_description == "Very large coarse 8-step raster 14x175 8s  Si IV   Mg II h/k   Mg II"
    assert meta.observing_mode_id == 3860258481
    assert meta.processing_level == 2
    assert meta.raster_fov_width_x == 13.9680814743 * u.arcsec
    assert meta.raster_fov_width_y == 181.987 * u.arcsec
    assert meta.satellite_rotation == -0.000540529 * u.deg
    assert meta.spatial_summing_factor == 1
    assert_quantity_allclose(meta.spectral_range, (1398.63094787, 1405.95766787) * u.angstrom)
    assert meta.spectral_summing_factor == 2
    assert meta.tracking_mode_enabled is False

    # TODO: Decide if I want to set observer_location, observer_radial_velocity, rsun_angular, run_meters
    # These are more WCS properties...
    assert meta.observer_location is None
    assert meta.rsun_angular is None
    assert meta.rsun_meters is None


@pytest.mark.parametrize("file_fixture", ["sns_sg_file", "raster_sg_file", "raster_sg_files"])
def test_smoke_read_spectrograph_lvl2(request, file_fixture):
    assert read_spectrograph_lvl2(request.getfixturevalue(file_fixture))


def test_read_spectrograph_lvl2_uses_auxiliary_pointing(raster_sg_file):
    windows = ["C II 1336", "Mg II k 2796"]
    raster = read_spectrograph_lvl2(raster_sg_file, spectral_windows=windows)

    with fits.open(raster_sg_file) as hdulist:
        window_indices = {hdulist[0].header[f"TDESC{i}"]: i for i in range(1, hdulist[0].header["NWIN"] + 1)}
        auxiliary_hdu = hdulist[-2]
        expected_longitude = auxiliary_hdu.data[:, auxiliary_hdu.header["XCENIX"]] / 3600
        expected_latitude = auxiliary_hdu.data[:, auxiliary_hdu.header["YCENIX"]] / 3600
        for window in windows:
            header = hdulist[window_indices[window]].header
            step_pixels = np.arange(header["NAXIS3"])
            _, latitude, longitude = raster[window][0].wcs.pixel_to_world_values(
                np.full_like(step_pixels, header["CRPIX1"] - 1, dtype=float),
                np.full_like(step_pixels, header["CRPIX2"] - 1, dtype=float),
                step_pixels,
            )
            assert list(raster[window][0]._fits_wcs.ctype)[1:] == ["HPLT-TAB", "HPLN-TAB"]
            assert raster[window][0]._fits_wcs.aux.dsun_obs > 1e11
            np.testing.assert_allclose(latitude, expected_latitude)
            np.testing.assert_allclose(longitude, expected_longitude)

    fuv_time = raster[windows[0]][0].axis_world_coords("time", wcs=raster[windows[0]][0].extra_coords)[0]
    nuv_time = raster[windows[1]][0].axis_world_coords("time", wcs=raster[windows[1]][0].extra_coords)[0]
    assert fuv_time[0].isot == "2014-03-29T14:09:43.000"
    assert nuv_time[0].isot == "2014-03-29T14:09:42.940"
    assert raster[windows[0]][0].meta["auxiliary times"][0].isot == "2014-03-29T14:09:39.000"


def test_read_spectrograph_lvl2_reports_missing_spectral_window(sns_sg_file):
    with pytest.raises(ValueError, match=r"Spectral windows \['NOPE'\] not in file"):
        read_spectrograph_lvl2(sns_sg_file, spectral_windows=["C II 1336", "NOPE"])


def test_nuv_times_reject_invalid_source_filename(raster_sg_file):
    with fits.open(raster_sg_file) as hdulist:
        source_data = hdulist[-1].data.copy()
        exposure_times = hdulist[-2].data[:, hdulist[-2].header["EXPTIMEN"]] * u.s
        auxiliary_times = Time(np.arange(len(exposure_times)), format="unix")
    source_data["NUVfilename"][0] = ""

    with pytest.raises(ValueError, match="Invalid timestamp in NUV source filenames"):
        _nuv_t_obs_from_source_filenames(source_data, exposure_times, auxiliary_times, filename=raster_sg_file)


def test_read_spectrograph_requires_one_source_row_per_step(raster_sg_file, tmp_path):
    filename = tmp_path / "missing_source_row.fits"
    with fits.open(raster_sg_file, memmap=False) as hdulist:
        hdulist[-1].header["TFIELDS"] = 9
        hdulist[-1].data = hdulist[-1].data[:-1]
        hdulist.writeto(filename)

    with pytest.raises(ValueError, match=r"Expected 8 NUV source filename rows.*found 7"):
        read_spectrograph_lvl2(filename, spectral_windows="Mg II k 2796")


def test_read_spectrograph_retains_missing_nuv_exposure(raster_sg_file, tmp_path, caplog):
    filename = tmp_path / "missing_nuv_exposure.fits"
    with fits.open(raster_sg_file, memmap=False) as hdulist:
        hdulist[-2].data[0, hdulist[-2].header["EXPTIMEN"]] = 0
        window_index = next(
            i for i in range(1, hdulist[0].header["NWIN"] + 1) if hdulist[0].header[f"TDESC{i}"] == "Mg II k 2796"
        )
        hdulist[window_index].data[0] = BAD_PIXEL_VALUE_SCALED
        hdulist.writeto(filename)

    with caplog.at_level("WARNING", logger="sunpy"):
        raster = read_spectrograph_lvl2(filename, spectral_windows="Mg II k 2796")

    cube = raster["Mg II k 2796"][0]
    times = cube.axis_world_coords("time", wcs=cube.extra_coords)[0]
    assert str(filename) in caplog.text
    assert "EXPTIMEN is 0 s at row(s) [0]" in caplog.text
    assert cube.meta["exposure time"][0] == 0 * u.s
    assert np.all(cube.mask[0])
    assert not np.all(cube.mask[1])
    assert abs((times[0] - cube.meta["auxiliary times"][0]).to_value(u.s)) < 1e-6


def test_read_spectrograph_flips_v34_mask_uncertainty_and_meta(raster_sg_file, tmp_path):
    # STEPS_AV < -0.01 marks a V34 raster, which is read flipped along the step axis
    filename = tmp_path / "v34.fits"
    windows = ["C II 1336", "Mg II k 2796"]
    with fits.open(raster_sg_file, memmap=False) as hdulist:
        hdulist[0].header["STEPS_AV"] = -hdulist[0].header["STEPS_AV"]
        names = [hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)]
        for window in windows:
            hdulist[names.index(window) + 1].data[0] = BAD_PIXEL_VALUE_SCALED
        hdulist.writeto(filename)

    flipped = read_spectrograph_lvl2(filename, spectral_windows=windows, uncertainty=True)
    original = read_spectrograph_lvl2(filename, spectral_windows=windows, uncertainty=True, revert_v34=True)
    for window in windows:
        cube, reference = flipped[window][0], original[window][0]
        np.testing.assert_array_equal(cube.data, reference.data[::-1])
        np.testing.assert_array_equal(cube.mask, cube.data == BAD_PIXEL_VALUE_SCALED)
        assert np.all(cube.mask[-1])
        assert not np.all(cube.mask[0])
        np.testing.assert_array_equal(cube.uncertainty.array, reference.uncertainty.array[::-1])
        times = cube.axis_world_coords("time", wcs=cube.extra_coords)[0]
        reference_times = reference.axis_world_coords("time", wcs=reference.extra_coords)[0]
        np.testing.assert_array_equal(times.jd, reference_times[::-1].jd)
        np.testing.assert_array_equal(cube.meta["auxiliary times"].jd, reference.meta["auxiliary times"][::-1].jd)
        for key in ("exposure time", "observer radial velocity", "orbital phase"):
            np.testing.assert_array_equal(cube.meta[key], reference.meta[key][::-1])
        center, reference_center = cube.meta["exposure FOV center"], reference.meta["exposure FOV center"][::-1]
        np.testing.assert_array_equal(center.Tx, reference_center.Tx)
        np.testing.assert_array_equal(center.Ty, reference_center.Ty)


def test_raster_wcs_steps_have_no_index_vector(raster_sg_file):
    # wcslib searches a -TAB index vector linearly, so a 1..N step index made each lookup O(step)
    cube = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336")["C II 1336"][0]
    assert "PS3_2" not in cube._fits_wcs.to_header()

    steps = np.array([0, 0.5, 1, 6.25, 7])
    world = cube.wcs.pixel_to_world_values(np.zeros(5), np.full(5, 50.0), steps)
    np.testing.assert_allclose(world[2][1], (world[2][0] + world[2][2]) / 2)
    np.testing.assert_allclose(cube.wcs.world_to_pixel_values(*world)[2], steps, atol=1e-6)


def _sit_and_stare_wcs(sns_sg_file):
    # A sit-and-stare tracks solar rotation in 0.05 arcsec jumps every few exposures and jitters
    # by about 1 mas in between, so several exposures' slits pass through each point
    with fits.open(sns_sg_file) as hdulist:
        header, auxiliary_hdu = hdulist[1].header, hdulist[-2].copy()
    longitude = auxiliary_hdu.data[:, auxiliary_hdu.header["XCENIX"]]
    steps = np.arange(len(longitude))
    longitude[:] = longitude[0] + 0.05 * (steps // 6) + np.random.default_rng(0).uniform(-1e-3, 1e-3, len(steps))
    return _create_tabular_wcs(header, auxiliary_hdu, date_obs="2021-09-05T00:18:33")


def test_raster_wcs_inverse_matches_wcslib(sns_sg_file):
    wcs = _sit_and_stare_wcs(sns_sg_file)
    assert type(wcs) is type(copy.copy(wcs)) is type(copy.deepcopy(wcs)) is _RasterWCS
    rng = np.random.default_rng(1)
    # Points from a little past the ends of the table, some of them beside the slit
    pixel = [rng.uniform(-0.2, size - 0.8, 500) for size in wcs.pixel_shape]
    wavelength, *position = wcs.pixel_to_world_values(*pixel)
    world = (wavelength, *(coordinate + rng.normal(0, 2e-6, 500) for coordinate in position))

    inverse = np.array(wcs.world_to_pixel_values(*world))
    wcslib = np.array(WCS.world_to_pixel_values(wcs, *world))

    assert 0 < np.isnan(inverse[2]).sum() < 100
    # The first exposure whose slit passes through the point, as wcslib finds it
    np.testing.assert_array_equal(np.floor(inverse[2] + 1e-6), np.floor(wcslib[2] + 1e-6))
    # wcslib stops within 1e-10 degrees, a good part of a step where these slits are about 3e-7 degrees apart,
    # so its step is only compared by cell, and the world round trip shows the inverse is exact
    np.testing.assert_allclose(inverse[:2], wcslib[:2], atol=1e-3)
    inside = np.isfinite(inverse[2])
    back = wcs.pixel_to_world_values(*inverse[:, inside])
    np.testing.assert_allclose(back[1:], np.array(world)[1:, inside], rtol=0, atol=1e-12)


def test_raster_wcs_inverse_falls_back_to_wcslib(sns_sg_file):
    celestial = _sit_and_stare_wcs(sns_sg_file).sub([2, 3])
    world = celestial.pixel_to_world_values([3.0, 20.5], [4.0, 100.25])
    np.testing.assert_array_equal(celestial.world_to_pixel_values(*world), WCS.world_to_pixel_values(celestial, *world))


def test_raster_wcs_inverse_after_astropy_slicing(sns_sg_file):
    # astropy's slicing moves CRPIX on every axis
    wcs = _sit_and_stare_wcs(sns_sg_file).slice((slice(50, 150), slice(10, 30), slice(5, None)))
    assert type(wcs) is _RasterWCS
    world = wcs.pixel_to_world_values(*(np.linspace(0, size - 1, 7) for size in wcs.pixel_shape))
    inverse = np.array(wcs.world_to_pixel_values(*world))
    wcslib = np.array(WCS.world_to_pixel_values(wcs, *world))
    np.testing.assert_allclose(inverse[:2], wcslib[:2], atol=1e-3)
    # wcslib's step is only good to a part of a step here, and differs most on the slit lines themselves
    np.testing.assert_allclose(inverse[2], wcslib[2], atol=0.5)
    np.testing.assert_allclose(wcs.pixel_to_world_values(*inverse)[1:], world[1:], rtol=0, atol=1e-12)


def test_read_spectrograph_memmap_has_no_uncertainty(raster_sg_file):
    # memmap data are unscaled integers, so no uncertainty is computed from them
    memmap = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336", memmap=True, uncertainty=True)
    assert memmap["C II 1336"][0].uncertainty is None
    scaled = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336", uncertainty=True)
    assert scaled["C II 1336"][0].uncertainty is not None


@pytest.mark.parametrize("memmap", [False, True])
def test_read_spectrograph_masks_both_fill_values(tmp_path, sns_sg_file, memmap):
    filename = tmp_path / "raster_fill.fits"
    with fits.open(sns_sg_file, memmap=False) as hdulist:
        data = np.full(hdulist[1].data.shape, 7.0)
        data.flat[:4] = [-200, -199, -198, 7]
        expected_mask = np.isin(data, [-200, -199])
        hdulist[1].data = data
        hdulist[1].scale("int16", bscale=0.25, bzero=7992)
        raw = hdulist[1].data.copy()
        hdulist.writeto(filename)

    cube = read_spectrograph_lvl2(filename, spectral_windows="C II 1336", memmap=memmap)["C II 1336"].data[0]

    np.testing.assert_array_equal(cube.mask, expected_mask)
    if memmap:
        np.testing.assert_array_equal(cube.data, raw)


def test_read_spectrograph_uncertainty_is_a_standard_deviation(raster_sg_file):
    # A bare array would be stored as an UnknownUncertainty (issue #57).
    cube = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336", uncertainty=True)["C II 1336"][0]
    assert isinstance(cube.uncertainty, StdDevUncertainty)
    assert cube.uncertainty.array.shape == cube.data.shape


def test_memmap_records_the_window_scaling():
    # memmap=True keeps the FITS integers, so the window's BSCALE and BZERO go in the metadata
    filename = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    cube = read_spectrograph_lvl2(filename, memmap=True, spectral_windows="Mg II k 2796")["Mg II k 2796"].data[0]
    assert np.issubdtype(cube.data.dtype, np.integer)
    with fits.open(filename) as hdulist:
        header = hdulist[3].header
    assert (cube.meta["BSCALE"], cube.meta["BZERO"]) == (header["BSCALE"], header["BZERO"])
