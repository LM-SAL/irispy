import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose

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
    # Test data only has a sequence of 13 long
    assert len(si_iv) == 13
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
    assert len(meta.fits_header) == 412 == (len(meta.keys()) + 13)  # History is missing
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


def test_smoke_read_spectrograph_lvl2(sns_sg_file, raster_sg_file, raster_sg_files):
    read_spectrograph_lvl2(sns_sg_file)
    read_spectrograph_lvl2(raster_sg_file)
    read_spectrograph_lvl2(raster_sg_files)
