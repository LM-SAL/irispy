import warnings
from pathlib import Path

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

import irispy.io._raster_combine as raster_combine
import irispy.io.spectrograph as spectrograph_io
from irispy._spectrograph_wcs import (
    SIT_AND_STARE_CDELT3_PLACEHOLDER,
    _create_raster_gwcs,
    _raster_wcs_bad_row_mask,
    _sanitize_raster_wcs_tables,
)
from irispy.io._raster_combine import _combine_raster_cubes, _validate_combinable_raster_cubes
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
    assert len(meta.fits_header) == 380 == (len(meta.keys()) + 10)  # History is missing
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
    assert si_iv.fits_wcs is not None
    assert len(si_iv.split_rasters()) == 1
    assert si_iv.raster_slice(0).shape == si_iv.shape


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
    assert len(meta.fits_header) == 413 == (len(meta.keys()) + 10)  # History is missing
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
    assert si_iv.fits_wcs is None
    assert len(si_iv.split_rasters()) == 13
    assert si_iv.raster_slice(0).shape == (8, 109, 29)


def test_smoke_read_spectrograph_lvl2(sns_sg_file, raster_sg_file, raster_sg_files):
    read_spectrograph_lvl2(sns_sg_file)
    read_spectrograph_lvl2(raster_sg_file)
    read_spectrograph_lvl2(raster_sg_files)


def test_read_spectrograph_lvl2_skips_window_with_bad_wcs(sns_sg_file, monkeypatch):
    real_create_raster_gwcs = spectrograph_io._create_raster_gwcs
    calls = {"count": 0}

    def fail_first_window(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            msg = "bad test WCS"
            raise ValueError(msg)
        return real_create_raster_gwcs(*args, **kwargs)

    monkeypatch.setattr(spectrograph_io, "_create_raster_gwcs", fail_first_window)

    raster_collection = spectrograph_io.read_spectrograph_lvl2(
        sns_sg_file,
        spectral_windows=["C II 1336", "Fe XII 1349"],
    )

    assert list(raster_collection.keys()) == ["Fe XII 1349"]


def test_multifile_read_builds_only_combined_gwcs(raster_sg_files, monkeypatch):
    real_create_raster_gwcs = spectrograph_io._create_raster_gwcs
    pc_shapes = []

    def count_gwcs_builds(*args, **kwargs):
        pc_shapes.append(args[1].shape)
        return real_create_raster_gwcs(*args, **kwargs)

    monkeypatch.setattr(spectrograph_io, "_create_raster_gwcs", count_gwcs_builds)
    monkeypatch.setattr(raster_combine, "_create_raster_gwcs", count_gwcs_builds)

    raster_collection = read_spectrograph_lvl2(raster_sg_files[:2], spectral_windows="Si IV 1403")

    assert raster_collection["Si IV 1403"].data.ndim == 4
    assert pc_shapes == [(2, 8, 2, 2)]


def test_multifile_read_materializes_gwcs_when_one_cube_remains(raster_sg_files, monkeypatch):
    real_validate_raster_wcs_inputs = spectrograph_io._validate_raster_wcs_inputs
    real_create_raster_gwcs = spectrograph_io._create_raster_gwcs
    validate_calls = {"count": 0}
    pc_shapes = []

    def fail_second_file_validation(*args, **kwargs):
        validate_calls["count"] += 1
        if validate_calls["count"] == 2:
            msg = "bad test WCS"
            raise ValueError(msg)
        return real_validate_raster_wcs_inputs(*args, **kwargs)

    def count_gwcs_builds(*args, **kwargs):
        pc_shapes.append(args[1].shape)
        return real_create_raster_gwcs(*args, **kwargs)

    monkeypatch.setattr(spectrograph_io, "_validate_raster_wcs_inputs", fail_second_file_validation)
    monkeypatch.setattr(spectrograph_io, "_create_raster_gwcs", count_gwcs_builds)
    monkeypatch.setattr(raster_combine, "_create_raster_gwcs", count_gwcs_builds)

    raster_collection = read_spectrograph_lvl2(raster_sg_files[:2], spectral_windows="Si IV 1403")
    si_iv = raster_collection["Si IV 1403"]

    assert si_iv.data.ndim == 3
    assert si_iv.wcs is not si_iv.fits_wcs
    assert pc_shapes == [(8, 2, 2)]


@pytest.mark.filterwarnings("ignore:uncertainty is not computed when memmap=True:UserWarning")
def test_memmap_mode_never_computes_uncertainty(sns_sg_file, raster_sg_files):
    sit_and_stare = read_spectrograph_lvl2(sns_sg_file, memmap=True, uncertainty=True)["Si IV 1403"]
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True, uncertainty=True)["Si IV 1403"]

    assert sit_and_stare.uncertainty is None
    assert raster.uncertainty is None
    assert raster.mask is None


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


def test_combined_raster_meta_stacks_new_scan_aligned_keys(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    cubes = list(raster_collection["Si IV 1403"].split_rasters()[:2])
    for index, cube in enumerate(cubes):
        cube.meta.add("synthetic scan value", np.arange(cube.shape[0]) + 100 * index, axes=0)

    combined = _combine_raster_cubes(cubes)

    assert np.array_equal(combined.meta._axes["synthetic scan value"], [0, 1])
    np.testing.assert_array_equal(combined.meta["synthetic scan value"][0], np.arange(cubes[0].shape[0]))
    np.testing.assert_array_equal(combined.meta["synthetic scan value"][1], np.arange(cubes[1].shape[0]) + 100)


def test_combined_raster_exposes_time_on_scan_and_step_axes(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"]

    time = scan.time

    assert time.shape == scan.shape[:2]
    assert time[0, 0].isot == scan.raster_slice(0).time[0].isot
    assert time[-1, -1].isot == scan.raster_slice(-1).time[-1].isot


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
    frame = wcs_to_celestial_frame(scan.raster_slice(0).fits_wcs.celestial)
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
        scan.meta.observer,
        sit_and_stare=scan.meta["sit_and_stare"],
    )

    for scan_index in (1, scan.shape[0] + 1):
        point = wcs.array_index_to_world(scan_index, 20, 10)
        assert wcs.world_to_array_index(*point) == (scan_index, 20, 10)


def test_raster_gwcs_matches_fits_wcs_forward_world_coordinates(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    scan = raster_collection["Si IV 1403"].raster_slice(0)
    assert not scan.meta["sit_and_stare"]

    for array_index in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
        spectral, sky, _, _ = scan.wcs.array_index_to_world(*array_index)
        basic_spectral, basic_sky = scan.fits_wcs.array_index_to_world(*array_index)

        assert_quantity_allclose(spectral.to(u.nm), basic_spectral.to(u.nm))
        # The raster gWCS uses per-step AUX pointing tables, while fits_wcs is
        # the single linear FITS WCS. This fixture differs by up to ~8.05" in Tx.
        assert_quantity_allclose(sky.Tx.to(u.arcsec), basic_sky.Tx.to(u.arcsec), atol=8.2 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), basic_sky.Ty.to(u.arcsec), atol=0.05 * u.arcsec)


@pytest.fixture
def v34_raster_file(tmp_path, raster_sg_file):
    destination = tmp_path / Path(raster_sg_file).name
    with fits.open(raster_sg_file) as hdulist:
        hdulist.writeto(destination)
    with fits.open(destination, mode="update") as hdulist:
        hdulist[0].header["STEPS_AV"] = -1.0
    return destination


@pytest.mark.parametrize("revert_v34", [False, True])
def test_v34_gwcs_matches_fits_wcs_forward_world_coordinates(v34_raster_file, revert_v34):
    scan = read_spectrograph_lvl2(v34_raster_file, revert_v34=revert_v34)["Si IV 1403"]
    assert not scan.meta["sit_and_stare"]
    assert scan.meta["flipped"] is not revert_v34

    for array_index in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
        spectral, sky, _, _ = scan.wcs.array_index_to_world(*array_index)
        basic_spectral, basic_sky = scan.fits_wcs.array_index_to_world(*array_index)

        assert_quantity_allclose(spectral.to(u.nm), basic_spectral.to(u.nm))
        # Same per-step-table vs linear-FITS-WCS difference as the non-v34
        # raster test above.
        assert_quantity_allclose(sky.Tx.to(u.arcsec), basic_sky.Tx.to(u.arcsec), atol=8.2 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), basic_sky.Ty.to(u.arcsec), atol=0.05 * u.arcsec)


@pytest.mark.parametrize("revert_v34", [False, True])
def test_v34_gwcs_inverse_roundtrips_array_indices(v34_raster_file, revert_v34):
    scan = read_spectrograph_lvl2(v34_raster_file, revert_v34=revert_v34)["Si IV 1403"]

    for array_index in ((3, 50, 10), (4, 50, 10), (0, 80, 20)):
        point = scan.wcs.array_index_to_world(*array_index)
        assert scan.wcs.world_to_array_index(*point) == array_index


def test_basic_wcs_is_deprecated_alias_of_fits_wcs(sns_sg_file):
    cube = read_spectrograph_lvl2(sns_sg_file)["Si IV 1403"]

    with pytest.warns(DeprecationWarning, match="fits_wcs"):
        deprecated = cube.basic_wcs

    assert deprecated is cube.fits_wcs


def test_celestial_frame_is_slicing_invariant(raster_sg_files):
    cube = read_spectrograph_lvl2(raster_sg_files)["Si IV 1403"]

    frame = cube.celestial_frame
    assert isinstance(frame, Helioprojective)
    # Combined cube has fits_wcs None; slices must still agree.
    assert cube.raster_slice(0).celestial_frame == frame
    assert cube[0, 2].celestial_frame == frame
    assert cube[:, 1:3].celestial_frame == frame
    assert cube[0, 2, 50].celestial_frame == frame

    wcs_frame = wcs_to_celestial_frame(cube.raster_slice(0).fits_wcs.celestial)
    assert_quantity_allclose(
        frame.observer.cartesian.xyz.to(u.km),
        wcs_frame.observer.cartesian.xyz.to(u.km),
        rtol=1e-6,
    )
    # The FITS WCS frame must carry an obstime, otherwise coordinates built in
    # celestial_frame cannot be transformed into it (they become NaN).
    assert wcs_frame.obstime is not None
    roundtripped = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame).transform_to(wcs_frame)
    assert np.isfinite(roundtripped.Tx.value)
    assert np.isfinite(roundtripped.Ty.value)


def test_celestial_frame_on_sji_cube(sns_sjicube_1330):
    frame = sns_sjicube_1330.celestial_frame

    assert isinstance(frame, Helioprojective)
    assert sns_sjicube_1330[0].celestial_frame == frame


def test_v34_flip_and_revert_agree_on_world_coordinates(v34_raster_file):
    """
    The same physical exposure must map to the same sky position whether the cube is
    read flipped (default) or with revert_v34=True.
    """
    flipped = read_spectrograph_lvl2(v34_raster_file)["Si IV 1403"]
    reverted = read_spectrograph_lvl2(v34_raster_file, revert_v34=True)["Si IV 1403"]
    n_steps = flipped.shape[0]

    for step, slit, wavelength in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
        spectral, sky, time, _ = flipped.wcs.array_index_to_world(step, slit, wavelength)
        mirrored_spectral, mirrored_sky, mirrored_time, _ = reverted.wcs.array_index_to_world(
            n_steps - 1 - step, slit, wavelength
        )

        assert_quantity_allclose(spectral.to(u.nm), mirrored_spectral.to(u.nm))
        assert_quantity_allclose(sky.Tx.to(u.arcsec), mirrored_sky.Tx.to(u.arcsec), atol=0.01 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), mirrored_sky.Ty.to(u.arcsec), atol=0.01 * u.arcsec)
        assert time.isclose(mirrored_time)


def test_sit_and_stare_gwcs_matches_fits_wcs_forward_world_coordinates(sns_sg_file):
    raster_collection = read_spectrograph_lvl2(sns_sg_file)
    scan = raster_collection["Si IV 1403"]
    assert scan.meta["sit_and_stare"]

    for array_index in ((3, 20, 10), (4, 25, 10), (10, 30, 5)):
        spectral, sky, _, _ = scan.wcs.array_index_to_world(*array_index)
        basic_spectral, basic_sky = scan.fits_wcs.array_index_to_world(*array_index)

        assert_quantity_allclose(spectral.to(u.nm), basic_spectral.to(u.nm))
        assert_quantity_allclose(sky.Tx.to(u.arcsec), basic_sky.Tx.to(u.arcsec), atol=0.05 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), basic_sky.Ty.to(u.arcsec), atol=0.02 * u.arcsec)


def test_raster_gwcs_crpix_reference_pixel_is_zero_based():
    header = {
        "CUNIT1": "nm",
        "CDELT1": 0.1,
        "CRVAL1": 140.0,
        "CRPIX1": 1.0,
        "CDELT2": 0.5,
        "CDELT3": 2.0,
        "CRPIX2": 3.0,
        "CRPIX3": 4.0,
    }
    crval = np.repeat([[10.0, 20.0]], 6, axis=0) * u.arcsec
    pc = np.repeat(np.eye(2)[np.newaxis, :, :], 6, axis=0) * u.pix
    dt = np.arange(6) * u.s
    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime="2020-01-01")

    wcs = _create_raster_gwcs(header, pc, crval, dt, "2020-01-01", observer, sit_and_stare=False)
    _, sky, _, _ = wcs.array_index_to_world(3, 2, 0)

    assert_quantity_allclose(sky.Tx.to(u.arcsec), 10 * u.arcsec)
    assert_quantity_allclose(sky.Ty.to(u.arcsec), 20 * u.arcsec)


def test_tiny_nonzero_cdelt3_is_not_sit_and_stare():
    header = {
        "CUNIT1": "nm",
        "CDELT1": 0.1,
        "CRVAL1": 140.0,
        "CRPIX1": 1.0,
        "CDELT2": 0.5,
        "CDELT3": SIT_AND_STARE_CDELT3_PLACEHOLDER,
        "CRPIX2": 3.0,
        "CRPIX3": 4.0,
    }
    crval = np.repeat([[10.0, 20.0]], 6, axis=0) * u.arcsec
    pc = np.repeat(np.eye(2)[np.newaxis, :, :], 6, axis=0) * u.pix
    dt = np.arange(6) * u.s
    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime="2020-01-01")

    wcs = _create_raster_gwcs(header, pc, crval, dt, "2020-01-01", observer, sit_and_stare=False)
    _, sky, _, _ = wcs.array_index_to_world(3, 2, 0)

    assert_quantity_allclose(sky.Tx.to(u.arcsec), 10 * u.arcsec)
    assert_quantity_allclose(sky.Ty.to(u.arcsec), 20 * u.arcsec)


def test_separate_raster_step_slice_slices_wcs_tables(raster_sg_files):
    raster_collection = read_spectrograph_lvl2(raster_sg_files)
    cube = raster_collection["Si IV 1403"]

    sliced = cube[:, 0:4]

    assert sliced.data.shape == (13, 4, 109, 29)
    assert sliced._raster_pc_table.shape == (13, 4, 2, 2)
    assert sliced._raster_crval_table.shape == (13, 4, 2)
    assert sliced._raster_wcs_header["NAXIS3"] == 4

    split_cubes = sliced.split_rasters()
    assert all(split_cube._raster_pc_table.shape == (4, 2, 2) for split_cube in split_cubes)
    assert all(split_cube._raster_crval_table.shape == (4, 2) for split_cube in split_cubes)

    recombined = _combine_raster_cubes(split_cubes)
    assert recombined.shape == sliced.shape
    assert recombined._raster_pc_table.shape == sliced._raster_pc_table.shape

    for array_index in ((0, 0, 50, 3), (12, 3, 80, 20)):
        spectral, sky, _, _, _ = recombined.wcs.array_index_to_world(*array_index)
        expected_spectral, expected_sky, _, _, _ = sliced.wcs.array_index_to_world(*array_index)

        assert_quantity_allclose(spectral.to(u.nm), expected_spectral.to(u.nm))
        assert_quantity_allclose(sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
        assert_quantity_allclose(sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


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


def test_sit_and_stare_zero_pc_rows_are_interpolated():
    pc = (
        np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[3.0, 0.0], [0.0, 3.0]],
            ]
        )
        * u.pix
    )
    crval = np.repeat([[10.0, 20.0]], 3, axis=0) * u.arcsec

    assert not _raster_wcs_bad_row_mask(pc, crval).any()
    bad_rows = _raster_wcs_bad_row_mask(pc, crval, pc_only=True)

    with pytest.warns(UserWarning, match="all-zero WCS tables"):
        pc_out, crval_out = _sanitize_raster_wcs_tables(pc.copy(), crval.copy(), bad_rows=bad_rows)

    np.testing.assert_array_equal(bad_rows, [False, True, False])
    np.testing.assert_allclose(pc_out[1].to_value(u.pix), [[2.0, 0.0], [0.0, 2.0]])
    assert_quantity_allclose(crval_out[1], [10.0, 20.0] * u.arcsec)


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
        with pytest.raises(ValueError, match="same slit and wavelength"):
            _validate_combinable_raster_cubes([cubes[0], truncated])


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
def test_combined_raster_pads_short_final_raster(tmp_path, raster_sg_files):
    copied_files = []
    for source in raster_sg_files[:2]:
        destination = tmp_path / Path(source).name
        with fits.open(source, memmap=False) as hdulist:
            hdulist.writeto(destination)
        copied_files.append(destination)

    with fits.open(copied_files[1], mode="update", memmap=False) as hdulist:
        windows = np.array([hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)])
        window_ext = int(np.where(windows == "Si IV 1403")[0][0]) + 1
        hdulist[window_ext].data = hdulist[window_ext].data[:-1]
        hdulist[-2].data = hdulist[-2].data[:-1]

    with pytest.warns(UserWarning, match="mismatched step counts"):
        combined = read_spectrograph_lvl2(copied_files, spectral_windows="Si IV 1403", uncertainty=True)["Si IV 1403"]

    assert combined.shape == (2, 8, 109, 29)
    assert np.isnan(combined.data[1, -1]).all()
    assert combined.mask[1, -1].all()
    assert combined.uncertainty.array.shape == combined.shape
    assert np.isnan(combined.uncertainty.array[1, -1]).all()
    assert combined._raster_pc_table.shape == (2, 8, 2, 2)
    assert combined._raster_crval_table.shape == (2, 8, 2)
    np.testing.assert_allclose(
        combined._raster_pc_table[1, -1].to_value(u.pix),
        combined._raster_pc_table[1, -2].to_value(u.pix),
    )
    assert combined.time.shape == combined.shape[:2]
    assert combined.time[1, -1].isot == combined.time[1, -2].isot
