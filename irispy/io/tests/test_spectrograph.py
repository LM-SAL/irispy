import warnings

import dask.array as da
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.units.errors import UnitsWarning
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs.utils import wcs_to_celestial_frame

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
    assert str(raster_collection)
    assert raster_collection.meta is None

    si_iv = raster_collection["Si IV 1403"]
    assert str(si_iv)
    assert len(si_iv) == 1
    assert si_iv.meta is not None
    assert si_iv[0].meta is not None
    meta = si_iv[0].meta
    assert si_iv[0].data.shape == (187, 40, 29)
    assert np.all(si_iv[0].data.shape == meta.data_shape)
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
    assert len(meta.fits_header) == 380
    assert len(meta.keys()) < len(meta.fits_header)
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
    assert meta.observer_location is None
    assert meta.rsun_angular is None
    assert meta.rsun_meters is None
    assert si_iv[0].wcs.world_n_dim == 5
    assert si_iv[0].wcs.pixel_n_dim == 3
    assert si_iv[0].basic_wcs is not None
    assert si_iv[0].wcs.world_axis_physical_types == (
        "em.wl",
        "custom:pos.helioprojective.lon",
        "custom:pos.helioprojective.lat",
        "time",
        "custom:STEP",
    )


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
    assert str(raster_collection)
    assert raster_collection.meta is None

    si_iv = raster_collection["Si IV 1403"]
    assert str(si_iv)
    assert len(si_iv) == 13
    assert si_iv.meta is not None
    assert si_iv[0].meta is not None
    meta = si_iv[0].meta
    assert si_iv[0].data.shape == (8, 109, 29)
    assert np.all(si_iv[0].data.shape == meta.data_shape)
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
    assert len(meta.fits_header) == 412
    assert len(meta.keys()) < len(meta.fits_header)
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
    assert meta.observer_location is None
    assert meta.rsun_angular is None
    assert meta.rsun_meters is None
    assert si_iv[0].wcs.world_n_dim == 5
    assert si_iv[0].wcs.pixel_n_dim == 3
    assert si_iv.time.format == "isot"


def test_smoke_read_spectrograph_lvl2(sns_sg_file, raster_sg_file, raster_sg_files):
    read_spectrograph_lvl2(sns_sg_file)
    read_spectrograph_lvl2(raster_sg_file)
    read_spectrograph_lvl2(raster_sg_files)


def test_memmap_read_spectrograph_lvl2(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files, memmap=True)
    cube = raster_collection["Si IV 1403"][0]
    assert isinstance(cube.data, da.Array)
    assert cube.data.shape == (8, 109, 29)
    assert isinstance(cube.mask, da.Array)
    assert cube.mask.shape == cube.data.shape
    frame = cube.data[0].compute()
    assert frame.shape == (109, 29)
    assert frame.dtype == np.float32


def test_gwcs_crop_is_compatible_with_original_raster_api(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"][0]

    assert scan.time.format == "isot"
    assert scan.basic_wcs is not None
    spectral_coord = scan.spectral_axis[len(scan.spectral_axis) // 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        warnings.simplefilter("ignore", UnitsWarning)
        spectral_crop = scan.crop([SpectralCoord(spectral_coord), None], [SpectralCoord(spectral_coord), None])
    assert spectral_crop.data.ndim == 2

    frame = wcs_to_celestial_frame(scan.wcs.celestial)
    target = SkyCoord(-8 * u.arcsec, 370 * u.arcsec, unit=u.arcsec, frame=frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        warnings.simplefilter("ignore", UnitsWarning)
        spectrum = scan.crop([None, target], [None, target])
    assert spectrum.data.ndim == 1


def test_gwcs_inverse_enables_official_crop_api(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"][0]

    start = scan.wcs.array_index_to_world(3, 50, 10)
    stop = scan.wcs.array_index_to_world(4, 50, 10)

    assert scan.wcs.world_to_array_index(*start) == (3, 50, 10)
    assert scan.wcs.world_to_array_index(*stop) == (4, 50, 10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        warnings.simplefilter("ignore", UnitsWarning)
        cropped = scan.crop(start, stop)
    assert cropped.data.shape == (2,)


def test_scalar_slice_preserves_promoted_global_coords(sns_sg_file):
    raster_collection = read_spectrograph_lvl2(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    sliced = cube[10, 20]

    assert "time" in sliced.global_coords
    assert "exposure time" in sliced.global_coords
    assert "helioprojective" in sliced.global_coords
    assert sliced.global_coords["time"] == cube.time[10]
    assert sliced.exposure_time == cube.exposure_time[10]


def test_raster_gwcs_legacy_basic_wcs_bridge_is_limited_to_spectral_and_sky(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"][0]

    spectral, sky, _, _ = scan.wcs.array_index_to_world(3, 50, 10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        np.testing.assert_allclose(
            scan.wcs.world_to_pixel(spectral, sky),
            scan.basic_wcs.world_to_pixel(spectral, sky),
        )
        assert scan.wcs.world_to_array_index(spectral, sky) == scan.basic_wcs.world_to_array_index(spectral, sky)
