import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose

from sunpy.coordinates import Helioprojective

from irispy.io.sji import read_sji_lvl2
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED


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
    assert len(meta.fits_header) == 162 == (len(meta.keys()) + 14)  # History is missing
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
    assert sji_1400_cube.data.shape == (2, 109, 178)  # (time, y, x)
    assert np.all(sji_1400_cube.data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SJI" == meta.instrument
    assert meta.detector == "SJI"
    assert meta.spectral_band == "FUV"
    assert meta.automatic_exposure_control_enabled is True
    assert meta.date_end.isot == "2023-04-08T11:42:01.050"
    assert meta.date_reference.isot == "2023-04-08T11:09:57.690"
    assert meta.date_start.isot == "2023-04-08T11:09:57.690"
    assert_quantity_allclose(meta.distance_to_sun, 1.0011105057794114 * u.AU)
    assert meta.exposure_control_triggers_in_observation == 0
    assert meta.exposure_control_triggers_in_raster == 0
    assert len(meta.fits_header) == 162 == (len(meta.keys()) + 14)  # History is missing
    assert meta["XCEN"] == -2.73951
    assert meta["YCEN"] == 945.279
    assert_quantity_allclose(meta.fov_center.Tx, -2.73951 * u.arcsec)
    assert_quantity_allclose(meta.fov_center.Ty, 945.279 * u.arcsec)
    assert meta.key_comments == {}
    assert meta.number_of_unique_raster_positions == 16
    assert meta.number_of_raster_positions == 1
    assert meta.observation_includes_saa is True
    assert meta.observatory_at_high_latitude is False
    assert meta.observing_campaign_start.isot == "2023-04-08T11:08:21.730"
    assert meta.observing_mode_description == "Very large coarse 64-step raster 126x175 64s   Deep x 30"
    observation_times = sji_1400_cube.axis_world_coords("time")[0]
    assert observation_times.isot.tolist() == ["2023-04-08T11:10:12.690", "2023-04-08T11:42:16.050"]
    assert [wcs.wcs.dateobs for wcs in sji_1400_cube.basic_wcs] == observation_times.isot.tolist()


def test_read_sji_lvl2_masks_scaled_float_bad_pixels(sns_sji_1330_file):
    expected_bad_pixels = np.count_nonzero(fits.getdata(sns_sji_1330_file, 0) == BAD_PIXEL_VALUE_SCALED)

    cube = read_sji_lvl2(sns_sji_1330_file)

    assert expected_bad_pixels > 0
    assert cube.mask.sum() == expected_bad_pixels


def test_read_sji_lvl2_masks_explicit_float_bad_pixels(tmp_path, sns_sji_1330_file):
    with fits.open(sns_sji_1330_file) as hdul:
        data = hdul[0].data.astype("float32")
        data.flat[:2] = BAD_PIXEL_VALUE_SCALED
        expected_bad_pixels = np.count_nonzero(data == BAD_PIXEL_VALUE_SCALED)
        hdul[0].data = data
        float_file = tmp_path / "sji_float_bad_pixels.fits"
        hdul.writeto(float_file)

    cube = read_sji_lvl2(float_file)

    assert expected_bad_pixels > 0
    assert cube.mask.sum() == expected_bad_pixels


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


def test_read_sji_lvl2_unrotated_pointing(tmp_path, sns_sji_1330_file):
    filename = tmp_path / "unrotated.fits"
    with fits.open(sns_sji_1330_file, memmap=False) as hdulist:
        for key in ("PC1_2IX", "PC2_1IX"):
            hdulist[1].data[:, hdulist[1].header[key]] = 0.0
        hdulist.writeto(filename)

    cube = read_sji_lvl2(filename)
    for key in ("PC1_2IX", "PC2_1IX"):
        for header in cube._basic_wcs:
            assert header[f"PC{key[2]}_{key[4]}"] == 0.0


def test_read_sji_lvl2_fills_dropped_pointing_rows(tmp_path, sns_sji_1330_file):
    # A dropped exposure zeroes its whole pointing row; those rows are filled
    # with the average of the neighbouring exposures.
    keys = ("XCENIX", "YCENIX", "PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX")
    filename = tmp_path / "dropped_row.fits"
    with fits.open(sns_sji_1330_file, memmap=False) as hdulist:
        columns = [hdulist[1].header[key] for key in keys]
        expected = {key: hdulist[1].data[[0, 2], column].mean() for key, column in zip(keys, columns, strict=True)}
        hdulist[1].data[1, columns] = 0.0
        hdulist.writeto(filename)

    cube = read_sji_lvl2(filename)
    header = cube._basic_wcs[1]
    np.testing.assert_allclose(header["CRVAL1"], expected["XCENIX"])
    np.testing.assert_allclose(header["CRVAL2"], expected["YCENIX"])
