import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs.utils import wcs_to_celestial_frame
import dask.array as da

from sunpy.coordinates import Helioprojective

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
    assert si_iv.data.shape == (104, 109, 29)
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
    assert si_iv.wcs.world_n_dim == 5
    assert si_iv.wcs.pixel_n_dim == 3
    assert si_iv.time.format == "isot"
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


def test_gwcs_crop_supports_full_world_component_api(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"]

    spectral_coord = scan.spectral_axis[len(scan.spectral_axis) // 2]
    spectral_crop = scan.crop(
        [SpectralCoord(spectral_coord), None, None, None],
        [SpectralCoord(spectral_coord), None, None, None],
    )
    assert spectral_crop.data.ndim == 2
    spectrum = scan.crop(
        scan.wcs.array_index_to_world(3, 50, 0),
        scan.wcs.array_index_to_world(3, 50, scan.data.shape[-1] - 1),
    )
    assert spectrum.data.ndim == 1

    spectrum_by_values = scan.crop_by_values(
        scan.wcs.array_index_to_world_values(3, 50, 0),
        scan.wcs.array_index_to_world_values(3, 50, scan.data.shape[-1] - 1),
        units=(u.nm, u.arcsec, u.arcsec, u.s, u.pix),
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

    start = scan.wcs.array_index_to_world(3, 50, 10)
    stop = scan.wcs.array_index_to_world(4, 50, 10)

    assert scan.wcs.world_to_array_index(*start) == (3, 50, 10)
    assert scan.wcs.world_to_array_index(*stop) == (4, 50, 10)

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


def test_raster_gwcs_matches_basic_wcs_forward_world_coordinates(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"].raster_slice(0)

    for array_index in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
        spectral, sky, _, _ = scan.wcs.array_index_to_world(*array_index)
        basic_spectral, basic_sky = scan.basic_wcs.array_index_to_world(*array_index)

        assert_quantity_allclose(spectral.to(u.nm), basic_spectral.to(u.nm))
        assert_quantity_allclose(sky.Tx.to(u.arcsec), basic_sky.Tx.to(u.arcsec), atol=10 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), basic_sky.Ty.to(u.arcsec), atol=1 * u.arcsec)
