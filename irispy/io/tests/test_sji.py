import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose

from sunpy.coordinates import Helioprojective

from irispy.io.sji import read_sji_lvl2


def test_sns_read_sji_lvl2(sns_sji_2832_file):
    sji_2832_cube = read_sji_lvl2(sns_sji_2832_file)
    # Simple repr check
    assert str(sji_2832_cube)
    assert sji_2832_cube.meta is not None
    meta = sji_2832_cube.meta
    assert sji_2832_cube.data.shape == (10, 40, 37)  # (time, y, x)
    assert np.all(sji_2832_cube.data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SJI" == meta.instrument
    assert meta.detector == "SJI"
    assert meta.spectral_band == "NUV"
    assert meta.automatic_exposure_control_enabled is True
    assert meta.date_end.isot == "2021-09-05T05:05:17.950"
    assert meta.date_reference.isot == "2021-09-05T00:19:01.890"
    assert meta.date_start.isot == "2021-09-05T00:19:01.890"
    assert_quantity_allclose(meta.distance_to_sun, 1.00827638 * u.AU)
    assert meta.exposure_control_triggers_in_observation == 0
    assert meta.exposure_control_triggers_in_raster == 0
    assert len(meta.fits_header) == 164 == (len(meta.keys()) + 16)  # History is missing
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
    assert meta.raster_fov_width_x == 61.2168 * u.arcsec
    assert meta.raster_fov_width_y == 66.54 * u.arcsec
    assert meta.satellite_rotation == 4.1483e-05 * u.deg
    assert meta.spatial_summing_factor == 1
    assert_quantity_allclose(meta.spectral_range, (2830.0, 2834.0) * u.angstrom)
    assert meta.spectral_summing_factor is None
    assert meta.tracking_mode_enabled is False

    # TODO: Decide if I want to set these, they are more WCS properties...
    assert meta.observer_location is None
    assert meta.rsun_angular is None
    assert meta.rsun_meters is None


def test_raster_read_sji_lvl2(raster_sji_1400_file):
    sji_1400_cube = read_sji_lvl2(raster_sji_1400_file)
    # Simple repr check
    assert str(sji_1400_cube)
    assert sji_1400_cube.meta is not None
    meta = sji_1400_cube.meta
    assert sji_1400_cube.data.shape == (54, 109, 110)  # (time, y, x)
    assert np.all(sji_1400_cube.data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SJI" == meta.instrument
    assert meta.detector == "SJI"
    assert meta.spectral_band == "FUV"
    assert meta.automatic_exposure_control_enabled is True
    assert meta.date_end.isot == "2014-03-29T17:54:08.106"
    assert meta.date_reference.isot == "2014-03-29T14:09:38.930"
    assert meta.date_start.isot == "2014-03-29T14:09:38.930"
    assert_quantity_allclose(meta.distance_to_sun, 0.99849015 * u.AU)
    assert meta.exposure_control_triggers_in_observation == 526
    assert meta.exposure_control_triggers_in_raster == 475
    assert len(meta.fits_header) == 163 == (len(meta.keys()) + 15)  # History is missing
    assert meta["XCEN"] == 505.392 == meta.fov_center.Tx.to_value(u.arcsec)
    assert meta["YCEN"] == 279.88 == meta.fov_center.Ty.to_value(u.arcsec)
    assert meta.key_comments == {}
    assert meta.number_of_unique_raster_positions == 3
    assert meta.number_of_raster_positions == 1
    assert meta.observation_includes_saa is True
    assert meta.observatory_at_high_latitude is False
    assert meta.observing_campaign_start.isot == "2014-03-29T14:09:38.830"
    assert meta.observing_mode_description == "Very large coarse 8-step raster 14x175 8s  Si IV   Mg II h/k   Mg II"


def test_smoke_read_sji_lvl2(
    sns_sji_1330_file,
    sns_sji_1400_file,
    sns_sji_2796_file,
    sns_sji_2832_file,
    raster_sji_1400_file,
    raster_sji_2796_file,
    raster_sji_2832_file,
):
    read_sji_lvl2(sns_sji_1330_file)
    read_sji_lvl2(sns_sji_1400_file)
    read_sji_lvl2(sns_sji_2796_file)
    read_sji_lvl2(sns_sji_2832_file)
    read_sji_lvl2(raster_sji_1400_file)
    read_sji_lvl2(raster_sji_2796_file)
    read_sji_lvl2(raster_sji_2832_file)
