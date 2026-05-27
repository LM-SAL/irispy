import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from irispy._spectrograph_wcs import _create_raster_gwcs, _sanitize_raster_wcs_tables
from irispy.io._raster_combine import _validate_combinable_raster_cubes
from irispy.io.spectrograph import read_spectrograph_lvl2


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
    meta = si_iv.meta
    assert si_iv.data.shape == (187, 40, 29)
    assert np.all(si_iv.data.shape == meta.data_shape)
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
    assert len(meta.fits_header) == 380 == (len(meta.keys()) + 14)  # History is missing
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
    assert si_iv.wcs.world_n_dim == 5
    assert si_iv.wcs.pixel_n_dim == 3
    assert si_iv.basic_wcs is not None
    assert si_iv.raster_boundaries == (slice(0, 187),)


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
    meta = si_iv.meta
    assert si_iv.data.shape == (13, 8, 109, 29)
    assert np.all(si_iv.data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SPEC" == meta.instrument
    assert meta.detector == "FUV2"
    assert meta.spectral_band == "FUV"
    assert meta.automatic_exposure_control_enabled is True
    assert meta.date_end.isot == "2014-03-29T14:25:43.280"
    assert meta.date_reference.isot == "2014-03-29T14:09:39.000"
    assert meta.date_start.isot == "2014-03-29T14:09:39.000"
    assert_quantity_allclose(meta.distance_to_sun, 0.99849015 * u.AU)
    assert meta.exposure_control_triggers_in_observation == 526
    assert meta.exposure_control_triggers_in_raster == 0
    assert len(meta.fits_header) == 413 == (len(meta.keys()) + 12)  # History is missing
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
    assert si_iv.wcs.world_n_dim == 6
    assert si_iv.wcs.pixel_n_dim == 4
    assert si_iv.wcs.world_axis_physical_types == (
        "em.wl",
        "custom:pos.helioprojective.lon",
        "custom:pos.helioprojective.lat",
        "time",
        "custom:STEP",
        "custom:SCAN",
    )
    assert si_iv.time.format == "isot"
    assert si_iv.time.shape == (13, 8)
    assert si_iv.basic_wcs is None
    assert len(si_iv.raster_boundaries) == 13
    assert si_iv.raster_slice(0).shape == (8, 109, 29)


def test_smoke_read_spectrograph_lvl2(sns_sg_file, raster_sg_file, raster_sg_files):
    read_spectrograph_lvl2(sns_sg_file)
    read_spectrograph_lvl2(raster_sg_file)
    read_spectrograph_lvl2(raster_sg_files)


def test_memmap_mode_never_computes_uncertainty(sns_sg_file, raster_sg_files):
    sit_and_stare = read_spectrograph_lvl2(sns_sg_file, memmap=True, uncertainty=True)["Si IV 1403"]
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True, uncertainty=True)["Si IV 1403"]

    assert sit_and_stare.uncertainty is None
    assert raster.uncertainty is None
    assert isinstance(raster.mask, da.Array)


def test_read_spectrograph_lvl2_reports_all_missing_spectral_windows(raster_sg_file):
    with pytest.raises(ValueError, match=r"Spectral windows .* not in file") as excinfo:
        read_spectrograph_lvl2(raster_sg_file, spectral_windows=["NOPE1", "NOPE2"])
    message = str(excinfo.value)
    assert "NOPE1" in message
    assert "NOPE2" in message


def test_read_spectrograph_lvl2_preserves_requested_spectral_window_order(raster_sg_file):
    requested_windows = ["Mg II k 2796", "C II 1336"]

    raster_collection = read_spectrograph_lvl2(raster_sg_file, spectral_windows=requested_windows)

    assert list(raster_collection.keys()) == requested_windows
    for window in requested_windows:
        assert raster_collection[window].meta.spectral_window == window


def test_read_spectrograph_lvl2_rejects_mismatched_observation(tmp_path, raster_sg_files):
    copied_files = []
    for path in raster_sg_files[:2]:
        destination = tmp_path / Path(path).name
        with fits.open(path) as hdulist:
            hdulist.writeto(destination)
        copied_files.append(destination)

    with fits.open(copied_files[1], mode="update") as hdulist:
        hdulist[0].header["OBSID"] = 9999999999

    with pytest.raises(ValueError, match="OBSID"):
        read_spectrograph_lvl2(copied_files)


def test_v34_flip_reverses_exposure_fov_center(tmp_path, raster_sg_file):
    destination = tmp_path / Path(raster_sg_file).name
    with fits.open(raster_sg_file) as hdulist:
        hdulist.writeto(destination)

    with fits.open(destination, mode="update") as hdulist:
        hdulist[0].header["STEPS_AV"] = -1.0

    with fits.open(destination) as hdulist:
        aux = hdulist[-2]
        expected_x = aux.data[:, aux.header["XCENIX"]] * u.arcsec
        expected_y = aux.data[:, aux.header["YCENIX"]] * u.arcsec

    scan = read_spectrograph_lvl2(destination)["Si IV 1403"]
    fov_center = scan.meta["exposure FOV center"]

    assert_quantity_allclose(fov_center.Tx.to(u.arcsec), expected_x[::-1])
    assert_quantity_allclose(fov_center.Ty.to(u.arcsec), expected_y[::-1])


def test_combined_raster_metadata_shape_and_end_time_are_consistent(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"]

    assert np.all(scan.shape == scan.meta.data_shape)
    assert scan.meta["NAXIS4"] == scan.shape[0]
    assert scan.meta["NAXIS3"] == scan.shape[1]
    assert scan.meta["DATE_END"] == scan.split_rasters()[-1].meta["DATE_END"]
    assert scan.meta["ENDOBS"] == scan.split_rasters()[-1].meta["ENDOBS"]
    if "NAXIS3" in scan.meta.fits_header:
        assert scan.meta.fits_header["NAXIS3"] == scan.shape[1]
    assert scan.meta.fits_header["NAXIS4"] == scan.shape[0]


def test_gwcs_crop_supports_full_world_component_api(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"]

    spectral_coord = scan.spectral_axis[len(scan.spectral_axis) // 2]
    spectral_crop = scan.crop(
        [SpectralCoord(spectral_coord), None, None, None, None],
        [SpectralCoord(spectral_coord), None, None, None, None],
    )
    assert spectral_crop.data.ndim == 3
    spectrum = scan.crop(
        scan.wcs.array_index_to_world(0, 3, 50, 0),
        scan.wcs.array_index_to_world(0, 3, 50, scan.data.shape[-1] - 1),
    )
    assert spectrum.data.ndim == 1

    spectrum_by_values = scan.crop_by_values(
        scan.wcs.array_index_to_world_values(0, 3, 50, 0),
        scan.wcs.array_index_to_world_values(0, 3, 50, scan.data.shape[-1] - 1),
        units=(u.nm, u.arcsec, u.arcsec, u.s, u.pix, u.pix),
    )
    assert spectrum_by_values.data.ndim == 1


def test_gwcs_crop_rejects_removed_two_component_shorthand(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"]
    spectral_coord = scan.spectral_axis[len(scan.spectral_axis) // 2]
    frame = wcs_to_celestial_frame(scan.raster_slice(0).basic_wcs.celestial)
    target = SkyCoord(-8 * u.arcsec, 370 * u.arcsec, unit=u.arcsec, frame=frame)

    with pytest.raises(ValueError, match="do not match WCS"):
        scan.crop([SpectralCoord(spectral_coord), None], [SpectralCoord(spectral_coord), None])

    with pytest.raises(ValueError, match="do not match WCS"):
        scan.crop([None, target], [None, target])


def test_gwcs_inverse_enables_official_crop_api(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"]

    start = scan.wcs.array_index_to_world(0, 3, 50, 10)
    stop = scan.wcs.array_index_to_world(0, 4, 50, 10)

    assert scan.wcs.world_to_array_index(*start) == (0, 3, 50, 10)
    assert scan.wcs.world_to_array_index(*stop) == (0, 4, 50, 10)

    cropped = scan.crop(start, stop)
    assert cropped.data.shape == (2,)


def test_gwcs_inverse_roundtrips_sit_and_stare_exposures(sns_sg_file):
    raster_collection = read_spectrograph_lvl2(sns_sg_file)
    scan = raster_collection["Si IV 1403"]

    start = scan.wcs.array_index_to_world(3, 20, 10)
    stop = scan.wcs.array_index_to_world(4, 20, 10)

    assert scan.wcs.world_to_array_index(*start) == (3, 20, 10)
    assert scan.wcs.world_to_array_index(*stop) == (4, 20, 10)

    cropped = scan.crop(start, stop)
    assert cropped.data.shape == (2,)


def test_gwcs_inverse_uses_explicit_step_when_time_is_not_monotonic(raster_sg_file):
    raster_collection = read_spectrograph_lvl2(raster_sg_file)
    scan = raster_collection["Si IV 1403"]

    pc_all = np.concatenate([scan._raster_pc_table, scan._raster_pc_table], axis=0)
    crval_all = np.concatenate([scan._raster_crval_table, scan._raster_crval_table], axis=0)
    dt_all = (
        np.concatenate(
            [
                np.arange(scan.shape[0], dtype=float),
                np.arange(scan.shape[0] - 1, -1, -1, dtype=float),
            ]
        )
        * u.s
    )

    wcs = _create_raster_gwcs(
        scan._raster_wcs_header,
        pc_all,
        crval_all,
        dt_all,
        scan.time[0],
        scan._raster_observer,
    )

    for scan_index in (1, scan.shape[0] + 1):
        point = wcs.array_index_to_world(scan_index, 20, 10)
        assert wcs.world_to_array_index(*point) == (scan_index, 20, 10)


def test_raster_gwcs_matches_basic_wcs_forward_world_coordinates(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"].raster_slice(0)

    for array_index in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
        spectral, sky, _, _ = scan.wcs.array_index_to_world(*array_index)
        basic_spectral, basic_sky = scan.basic_wcs.array_index_to_world(*array_index)

        assert_quantity_allclose(spectral.to(u.nm), basic_spectral.to(u.nm))
        assert_quantity_allclose(sky.Tx.to(u.arcsec), basic_sky.Tx.to(u.arcsec), atol=10 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), basic_sky.Ty.to(u.arcsec), atol=1 * u.arcsec)


def test_sanitize_raster_wcs_tables_all_bad_rows_uses_fallback():
    """
    When every row is all-zero, fallback values must be applied.
    """
    pc = np.zeros((3, 2, 2)) * u.pix
    crval = np.zeros((3, 2)) * u.arcsec
    fallback_pc = np.array([[1.0, 0.1], [0.2, 0.9]]) * u.pix
    fallback_crval = np.array([150.0, 250.0]) * u.arcsec

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pc_out, crval_out = _sanitize_raster_wcs_tables(pc.copy(), crval, fallback_pc, fallback_crval)
        assert len(w) == 2
        assert "all-zero" in str(w[0].message)
        assert "fallback" in str(w[1].message).lower()

    expected_pc = np.repeat(fallback_pc.to_value(u.pix)[None, ...], 3, axis=0)
    expected_crval = np.repeat(fallback_crval.to_value(u.arcsec)[None, ...], 3, axis=0)
    np.testing.assert_allclose(pc_out.to_value(u.pix), expected_pc)
    np.testing.assert_allclose(crval_out.to_value(u.arcsec), expected_crval)


def test_sanitize_raster_wcs_tables_all_bad_rows_without_fallback_raises():
    """
    When every row is all-zero and no fallback is given, a clear error is raised.
    """
    pc = np.zeros((2, 2, 2)) * u.pix
    crval = np.zeros((2, 2)) * u.arcsec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="All WCS table rows are bad"):
            _sanitize_raster_wcs_tables(pc.copy(), crval)


def test_combined_raster_rejects_mismatched_observer(raster_sg_files):
    """
    Combining rasters with different observers should raise ValueError.
    """
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    cube = raster_collection["Si IV 1403"]
    cubes = cube.split_rasters()
    if len(cubes) > 1:
        # Replace observer with a completely different one
        cubes[1]._raster_observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime="2020-01-01")
        with pytest.raises(ValueError, match="same observer"):
            _validate_combinable_raster_cubes(cubes)


def test_combined_raster_rejects_empty_cubes():
    """
    Combining empty raster list should raise ValueError.
    """
    with pytest.raises(ValueError, match="empty"):
        _validate_combinable_raster_cubes([])


def test_combined_raster_rejects_mismatched_shape(raster_sg_files):
    """
    Combining rasters with different shapes should raise ValueError.
    """
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    cube = raster_collection["Si IV 1403"]
    cubes = list(cube.split_rasters())
    if len(cubes) > 1:
        # Truncate one cube's slit dimension
        truncated = cubes[1][:, :-1, :]
        with pytest.raises(ValueError, match="same step, slit, and wavelength"):
            _validate_combinable_raster_cubes([cubes[0], truncated])
