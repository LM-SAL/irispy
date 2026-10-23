from contextlib import contextmanager

import dask.array as da
import numpy as np
import pytest

import asdf
import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates import Helioprojective

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import _nuv_t_obs_from_source_filenames, read_spectrograph_lvl2
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, SATURATION_LIMIT

SG_METADATA = {
    "sns_sg_file": {
        "keys": ["C II 1336", "Fe XII 1349", "O I 1356", "Si IV 1394", "Si IV 1403", "2832", "2814", "Mg II k 2796"],
        "shape": (187, 40, 29),
        "campaign_start_start_end": ("2021-09-05T00:18:33.640", "2021-09-05T00:18:33.810", "2021-09-05T05:07:27.400"),
        "distance_to_sun": 1.00827638 * u.AU,
        "exposure_control_triggers_in_observation": 0,
        "header_length": 380,
        "meta_length": 368,
        "unique_and_total_raster_positions": (1, 1),
        "observing_mode": ("Medium sit-and-stare 0.3x60 1s  C II   Si IV   Mg II h/k   Mg II w s", 3620258102),
        "raster_fov_width": (0.16635 * u.arcsec, 66.54 * u.arcsec),
        "satellite_rotation": 8.09432e-05 * u.deg,
        "spectral_range": (1398.60550787, 1406.03398787) * u.angstrom,
    },
    "raster_sg_files": {
        "keys": ["C II 1336", "1343", "Fe XII 1349", "O I 1356", "Si IV 1403", "2832", "2826", "2814", "Mg II k 2796"],
        "shape": (8, 109, 29),
        "campaign_start_start_end": ("2014-03-29T14:09:38.830", "2014-03-29T14:09:39.000", "2014-03-29T14:13:14.280"),
        "distance_to_sun": 0.99849015 * u.AU,
        "exposure_control_triggers_in_observation": 526,
        "header_length": 412,
        "meta_length": 401,
        "unique_and_total_raster_positions": (8, 180),
        "observing_mode": ("Very large coarse 8-step raster 14x175 8s  Si IV   Mg II h/k   Mg II", 3860258481),
        "raster_fov_width": (13.9680814743 * u.arcsec, 181.987 * u.arcsec),
        "satellite_rotation": -0.000540529 * u.deg,
        "spectral_range": (1398.63094787, 1405.95766787) * u.angstrom,
    },
}
V34_WINDOWS = ["C II 1336", "Mg II k 2796"]
# Raster indices are (step, slit, wavelength); sit-and-stare has 187 exposures of 40 slit pixels.
RASTER_INDICES = ((0, 50, 3), (3, 50, 10), (4, 50, 10), (7, 80, 20), (0, 80, 20))
SIT_AND_STARE_INDICES = ((0, 0, 10), (3, 20, 10), (4, 25, 10), (10, 30, 5), (93, 39, 5), (186, 0, 5))


def test_read_spectrograph_orders_files_by_date_obs(raster_sg_files):
    ordered = read_spectrograph_lvl2(raster_sg_files, spectral_windows="C II 1336")["C II 1336"]
    unordered = read_spectrograph_lvl2(raster_sg_files[::-1], spectral_windows="C II 1336")["C II 1336"]
    np.testing.assert_array_equal(unordered.data, ordered.data)
    assert (unordered.time == ordered.time).all()
    dates = [raster.fits_wcs.wcs.dateobs for raster in unordered]
    assert dates == sorted(dates)
    assert dates == [raster.fits_wcs.wcs.dateobs for raster in ordered]


@contextmanager
def _edited_copy(source, filename):
    """
    Open ``source`` for editing and write the edited HDUs to ``filename``.
    """
    with fits.open(source, memmap=False) as hdulist:
        yield hdulist
        hdulist.writeto(filename)


def _window_ext(hdulist, window):
    return [hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)].index(window) + 1


def _window_header_and_aux(filename, window):
    with fits.open(filename) as hdulist:
        return hdulist[_window_ext(hdulist, window)].header.copy(), hdulist[-2].data.copy(), hdulist[-2].header.copy()


@pytest.mark.parametrize("files", ["sns_sg_file", "raster_sg_files"])
def test_read_spectrograph_lvl2_metadata(request, files):
    expected = SG_METADATA[files]
    filenames = request.getfixturevalue(files)
    n_rasters = len(filenames) if isinstance(filenames, list) else 1
    raster_collection = read_spectrograph_lvl2(filenames)
    assert list(raster_collection.keys()) == expected["keys"]
    assert {type(key) for key in raster_collection} == {str}
    assert str(raster_collection)
    # We do not expect any metadata to be present on the collection
    assert raster_collection.meta is None

    si_iv = raster_collection["Si IV 1403"]
    assert str(si_iv)
    assert si_iv.time.ravel()[0].isot in str(si_iv[0])
    meta = si_iv.meta
    assert si_iv.data.shape == (n_rasters, *expected["shape"])
    assert np.all(si_iv.data.shape == meta.data_shape)
    # Meta is both a dict with the fits header keys but also provides
    # helper functions for specific values
    assert meta["TELESCOP"] == "IRIS" == meta.observatory
    assert meta["INSTRUME"] == "SPEC" == meta.instrument
    assert meta.detector == "FUV2"
    assert meta.spectral_band == "FUV"
    assert meta.automatic_exposure_control_enabled is True
    dates = (meta.observing_campaign_start.isot, meta.date_start.isot, meta.date_end.isot)
    assert dates == expected["campaign_start_start_end"]
    assert meta.date_reference.isot == meta.date_start.isot
    assert_quantity_allclose(meta.distance_to_sun, expected["distance_to_sun"])
    assert meta.exposure_control_triggers_in_observation == expected["exposure_control_triggers_in_observation"]
    assert meta.exposure_control_triggers_in_raster == 0
    assert len(meta.fits_header) == expected["header_length"]
    assert len(meta.keys()) == expected["meta_length"]  # History is missing
    assert meta.fov_center == SkyCoord(
        Tx=meta.get("XCEN"),
        Ty=meta.get("YCEN"),
        unit=u.arcsec,
        frame=Helioprojective,
        observer="earth",
        obstime=meta.date_reference,
    )
    assert meta.key_comments == {}
    positions = (meta.number_of_unique_raster_positions, meta.number_of_raster_positions)
    assert positions == expected["unique_and_total_raster_positions"]
    assert meta.observation_includes_saa is True
    assert meta.observatory_at_high_latitude is False
    assert (meta.observing_mode_description, meta.observing_mode_id) == expected["observing_mode"]
    assert meta.processing_level == 2
    assert (meta.raster_fov_width_x, meta.raster_fov_width_y) == expected["raster_fov_width"]
    assert meta.satellite_rotation == expected["satellite_rotation"]
    assert meta.spatial_summing_factor == 1
    assert_quantity_allclose(meta.spectral_range, expected["spectral_range"])
    assert meta.spectral_summing_factor == 2
    assert meta.tracking_mode_enabled is False
    assert si_iv.wcs.pixel_n_dim == si_iv.data.ndim
    physical_types = ("em.wl", "custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat", "time")
    assert si_iv.wcs.world_axis_physical_types == (*physical_types, "custom:STEP", "custom:SCAN")
    assert si_iv.time.format == "isot"
    assert si_iv.time.shape == si_iv.shape[:-2]
    assert si_iv[0].fits_wcs is not None


def test_combined_raster_metadata_and_time_match_its_rasters(raster_sg_files):
    scan = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Si IV 1403")["Si IV 1403"]

    time = scan.time
    assert time[0, 0].isot == scan[0].time[0].isot
    assert time[-1, -1].isot == scan[-1].time[-1].isot
    assert (time.isot[1] == scan[1].time.isot).all()
    # A file read alone gives its times from its time table, and the combined cube from its gWCS
    last = read_spectrograph_lvl2(raster_sg_files[-1], spectral_windows="Si IV 1403")["Si IV 1403"]
    assert_quantity_allclose((time[-1] - last.time).to(u.s), 0 * u.s, atol=1 * u.ns)
    assert scan.meta["DATE_END"] == last.meta["DATE_END"]


@pytest.mark.parametrize("container", [list, np.array])
@pytest.mark.parametrize("files", ["sns_sg_file", "raster_sg_file"])
def test_read_spectrograph_lvl2_keeps_requested_window_order_and_data_and_reports_missing(request, files, container):
    filename = request.getfixturevalue(files)
    requested_windows = ["Mg II k 2796", "C II 1336"]

    all_windows = read_spectrograph_lvl2(filename)
    raster_collection = read_spectrograph_lvl2(filename, spectral_windows=container(requested_windows))

    assert list(raster_collection.keys()) == requested_windows
    for window in requested_windows:
        assert raster_collection[window].meta.spectral_window == window
        np.testing.assert_array_equal(raster_collection[window].data, all_windows[window].data)
    with pytest.raises(ValueError, match=r"Spectral windows \['NOPE1', 'NOPE2'\] not in .*, which has \['C II 1336', "):
        read_spectrograph_lvl2(filename, spectral_windows=[*requested_windows, "NOPE1", "NOPE2"])


def test_each_window_has_the_spectral_axis_of_its_header(sns_sg_file):
    # The windows of a file share one built FITS WCS, each with its own spectral axis; the windows of the
    # wavelength drift file have CRPIX1 -116, -61 and -243
    drift = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    for filename in (sns_sg_file, drift):
        with fits.open(filename) as hdulist:
            for index, cube in enumerate(read_spectrograph_lvl2(filename).values(), start=1):
                header = hdulist[index].header
                assert cube[0].fits_wcs.pixel_shape == cube[0].data.shape[::-1]
                expected = header["CRVAL1"] + header["CDELT1"] * (np.arange(header["NAXIS1"]) + 1 - header["CRPIX1"])
                (wavelength,) = cube.axis_world_coords_values("em.wl")
                np.testing.assert_allclose(wavelength.to_value(u.AA), expected, rtol=1e-12)
                fits_wavelength = cube[0].fits_wcs.pixel_to_world_values(np.arange(header["NAXIS1"]), 0, 0)[0] * u.m
                np.testing.assert_allclose(fits_wavelength.to_value(u.AA), expected, rtol=1e-12)


def test_each_window_meta_is_of_the_primary_header(sns_sg_file):
    with fits.open(sns_sg_file) as hdulist:
        header = hdulist[0].header
        for window, cube in read_spectrograph_lvl2(hdulist).items():
            assert cube.meta.fits_header is header
            assert cube.meta.original_meta == header  # a read-only view of it, made anew at each access
            assert {key: cube.meta[key] for key in header} == dict(header.items())
            assert cube.meta.spectral_window == window


@pytest.mark.parametrize("files", ["raster_sg_file", "sns_sg_file"], ids=["raster", "sit_and_stare"])
def test_read_spectrograph_lvl2_uses_auxiliary_pointing(request, files):
    filename = request.getfixturevalue(files)
    windows = ["C II 1336", "Mg II k 2796"]
    raster = read_spectrograph_lvl2(filename, spectral_windows=windows)

    for window in windows:
        header, aux, aux_header = _window_header_and_aux(filename, window)
        expected_longitude = aux[:, aux_header["XCENIX"]] / 3600
        expected_latitude = aux[:, aux_header["YCENIX"]] / 3600
        steps = np.arange(header["NAXIS3"])
        pixels = (np.full(steps.shape, header["CRPIX1"] - 1.0), np.full(steps.shape, header["CRPIX2"] - 1.0), steps)
        cube = raster[window][0]
        assert list(cube.fits_wcs.wcs.ctype)[1:] == ["HPLT-TAB", "HPLN-TAB"]
        assert cube.fits_wcs.wcs.aux.dsun_obs > 1e11
        sky = cube.fits_wcs.pixel_to_world(*pixels)[1]
        np.testing.assert_allclose(sky.Tx.to_value(u.deg), expected_longitude)
        np.testing.assert_allclose(sky.Ty.to_value(u.deg), expected_latitude)

    if files == "raster_sg_file":
        assert raster[windows[0]][0].time[0].isot == "2014-03-29T14:09:43.000"
        assert raster[windows[1]][0].time[0].isot == "2014-03-29T14:09:42.940"
        assert raster[windows[0]][0].meta["auxiliary times"][0].isot == "2014-03-29T14:09:39.000"


def test_read_spectrograph_lvl2_reads_a_repeated_window_once(sns_sg_file):
    raster = read_spectrograph_lvl2(sns_sg_file, spectral_windows=["C II 1336", "Si IV 1394", "C II 1336"])
    assert list(raster) == ["C II 1336", "Si IV 1394"]
    assert raster["C II 1336"].shape[0] == 1  # one raster, not the window stacked twice


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
    with _edited_copy(raster_sg_file, filename) as hdulist:
        hdulist[-1].header["TFIELDS"] = 9
        hdulist[-1].data = hdulist[-1].data[:-1]

    with pytest.raises(ValueError, match=r"Expected 8 NUV source filename rows.*found 7"):
        read_spectrograph_lvl2(filename, spectral_windows="Mg II k 2796")


def test_read_spectrograph_retains_missing_nuv_exposure(raster_sg_file, tmp_path, caplog):
    filename = tmp_path / "missing_nuv_exposure.fits"
    with _edited_copy(raster_sg_file, filename) as hdulist:
        hdulist[-2].data[0, hdulist[-2].header["EXPTIMEN"]] = 0
        hdulist[_window_ext(hdulist, "Mg II k 2796")].data[0] = BAD_PIXEL_VALUE_SCALED

    with caplog.at_level("WARNING", logger="sunpy"):
        cube = read_spectrograph_lvl2(filename, spectral_windows="Mg II k 2796")["Mg II k 2796"][0]

    assert str(filename) in caplog.text
    assert "EXPTIMEN is 0 s at row(s) [0]" in caplog.text
    assert cube.meta["exposure time"][0] == 0 * u.s
    assert np.all(cube.mask[0])
    assert not np.all(cube.mask[1])
    assert abs((cube.time[0] - cube.meta["auxiliary times"][0]).to_value(u.s)) < 1e-6


def test_read_spectrograph_reads_files_with_no_data_or_no_nuv_exposure(caplog):
    # Real files of a data gap, r00066 with no data and r00067 with no NUV exposure (see data_gap/overview.txt).
    # Given in reverse, the file with no data is ordered first.
    no_data, no_nuv = (
        get_test_filepath(f"data_gap/iris_l2_20140329_140938_3860258481_raster_t000_r0006{step}_test.fits")
        for step in (6, 7)
    )
    with caplog.at_level("WARNING", logger="sunpy"):
        raster = read_spectrograph_lvl2([no_nuv, no_data], spectral_windows=["C II 1336", "2832"])

    assert sum(str(no_data) in record.getMessage() for record in caplog.records) == 1
    auxiliary = fits.getdata(no_data, -2)[:, fits.getheader(no_data, -2)["TIME"]]
    times = Time(fits.getheader(no_data)["STARTOBS"]) + auxiliary * u.s
    for window, has_data in [("C II 1336", True), ("2832", False)]:
        cube = raster[window]
        empty, after = cube[0], cube[1]
        assert np.all(empty.mask)
        assert_quantity_allclose((empty.time - times).to(u.s), 0 * u.s, atol=1 * u.us)
        assert cube.meta["DATE_OBS"] == times[0].isot < after.time[0].isot
        steps = np.arange(empty.shape[0])
        assert np.all(np.isnan(cube.wcs.pixel_to_world_values(0, 0, steps, 0)[1:3]))
        # r00067 has one FUV exposure, at step 2, so the other steps have no pointing
        has_pointing = np.isfinite(cube.wcs.pixel_to_world_values(0, 0, steps, 1)[1:3])
        np.testing.assert_array_equal(has_pointing, [steps == 2] * 2)
        assert np.all(np.isnan(empty.meta["exposure FOV center"].Tx))
        assert np.isnan(cube.meta.fov_center.Tx)
        assert np.isnan(cube.meta.satellite_rotation)
        assert np.all(after.mask) != has_data


def test_read_spectrograph_reads_an_unquoted_nan():
    # STEPT_DV = NAN, as files made before June 2018 write a NaN (see unparsable_card/overview.txt)
    filename = get_test_filepath("unparsable_card/iris_l2_20140917_050324_3860356063_raster_t000_r00000_test.fits")
    assert read_spectrograph_lvl2(filename)["C II 1336"][0].meta.step_time_stddev == "NAN"


def test_read_spectrograph_flips_v34_data_meta_and_world_coordinates(v34_sg_file):
    """
    Everything per exposure is reversed when a V34 raster is read flipped (default), so
    the same physical exposure keeps its world coordinates whether or not
    ``revert_v34=True``.
    """
    flipped = read_spectrograph_lvl2(v34_sg_file, spectral_windows=V34_WINDOWS, uncertainty=True)
    original = read_spectrograph_lvl2(v34_sg_file, spectral_windows=V34_WINDOWS, uncertainty=True, revert_v34=True)
    _, aux, aux_header = _window_header_and_aux(v34_sg_file, V34_WINDOWS[0])
    for window in V34_WINDOWS:
        cube, reference = flipped[window][0], original[window][0]
        np.testing.assert_array_equal(cube.data, reference.data[::-1])
        np.testing.assert_array_equal(cube.mask, reference.mask[::-1])
        np.testing.assert_array_equal(cube.uncertainty.array, reference.uncertainty.array[::-1])
        assert_quantity_allclose((cube.time - reference.time[::-1]).to(u.s), 0 * u.s, atol=1 * u.ns)
        np.testing.assert_array_equal(cube.meta["auxiliary times"].jd, reference.meta["auxiliary times"][::-1].jd)
        for key in ("exposure time", "observer radial velocity", "orbital phase"):
            np.testing.assert_array_equal(cube.meta[key], reference.meta[key][::-1])
        center, reference_center = cube.meta["exposure FOV center"], reference.meta["exposure FOV center"][::-1]
        np.testing.assert_array_equal(center.Tx, reference_center.Tx)
        np.testing.assert_array_equal(center.Ty, reference_center.Ty)
        assert_quantity_allclose(center.Tx.to(u.arcsec), aux[::-1, aux_header["XCENIX"]] * u.arcsec)
        assert_quantity_allclose(center.Ty.to(u.arcsec), aux[::-1, aux_header["YCENIX"]] * u.arcsec)

        for step, slit, wavelength in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
            spectral, sky, time, _ = cube.wcs.array_index_to_world(step, slit, wavelength)
            mirrored_spectral, mirrored_sky, mirrored_time, _ = reference.wcs.array_index_to_world(
                cube.shape[0] - 1 - step, slit, wavelength
            )
            assert_quantity_allclose(spectral.to(u.nm), mirrored_spectral.to(u.nm))
            assert_quantity_allclose(sky.Tx.to(u.arcsec), mirrored_sky.Tx.to(u.arcsec), atol=0.01 * u.arcsec)
            assert_quantity_allclose(sky.Ty.to(u.arcsec), mirrored_sky.Ty.to(u.arcsec), atol=0.01 * u.arcsec)
            assert time.isclose(mirrored_time)
        # Away from an exposure's center, the sky also depends on the sign of the step column of PC.
        _, sky, _, _ = cube.wcs.pixel_to_world(10, 50, 3.3)
        _, mirrored_sky, _, _ = reference.wcs.pixel_to_world(10, 50, cube.shape[0] - 1 - 3.3)
        assert_quantity_allclose(sky.Tx.to(u.arcsec), mirrored_sky.Tx.to(u.arcsec), atol=0.01 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), mirrored_sky.Ty.to(u.arcsec), atol=0.01 * u.arcsec)


def test_memmap_v34_rasters_read_together_are_flipped(v34_sg_file):
    cube = read_spectrograph_lvl2([v34_sg_file] * 2, spectral_windows="Mg II k 2796", memmap=True)["Mg II k 2796"]
    with fits.open(v34_sg_file, do_not_scale_image_data=True) as hdulist:
        expected = hdulist[_window_ext(hdulist, "Mg II k 2796")].data[::-1]
    for scan in range(2):
        np.testing.assert_array_equal(np.asarray(cube.data[scan]), expected)


def test_read_spectrograph_flips_a_real_v34_raster(v34_sg_file):
    # OBS 3460104433 steps the slit by -2": solar x falls along the file's steps and rises along the cube's
    windows = ["C II 1336", "Mg II k 2796"]
    with fits.open(v34_sg_file) as hdulist:
        assert hdulist[0].header["STEPS_AV"] < -0.01
        auxiliary = hdulist[-2]
        x_center = auxiliary.data[:, auxiliary.header["XCENIX"]]
        file_data = [hdulist[index].data for index in (1, 2)]
        reference_rows = [hdulist[index].header["CRPIX2"] - 1 for index in (1, 2)]
    assert np.all(np.diff(x_center) < 0)

    flipped = read_spectrograph_lvl2(v34_sg_file, spectral_windows=windows)
    reverted = read_spectrograph_lvl2(v34_sg_file, spectral_windows=windows, revert_v34=True)
    steps = np.arange(len(x_center))
    for window, data, row in zip(windows, file_data, reference_rows, strict=True):
        cube = flipped[window][0]
        np.testing.assert_array_equal(cube.data, data[::-1])
        np.testing.assert_array_equal(reverted[window][0].data, data)
        assert np.all(np.diff(cube.time.jd) < 0)
        assert_quantity_allclose(cube.meta["exposure FOV center"].Tx, x_center[::-1] * u.arcsec)
        sky = cube.wcs.array_index_to_world(steps, np.full(len(steps), row), np.zeros(len(steps)))[1]
        assert_quantity_allclose(sky.Tx.to(u.arcsec), x_center[::-1] * u.arcsec, atol=0.01 * u.arcsec)


def test_fits_wcs_interpolates_between_steps_and_inverts(raster_sg_file):
    fits_wcs = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336")["C II 1336"][0].fits_wcs
    steps = np.array([0, 0.5, 1, 6.25, 7])
    world = fits_wcs.pixel_to_world_values(np.zeros(5), np.full(5, 50.0), steps)
    np.testing.assert_allclose(world[2][1], (world[2][0] + world[2][2]) / 2)
    np.testing.assert_allclose(fits_wcs.world_to_pixel_values(*world)[2], steps, atol=1e-6)


@pytest.mark.parametrize("files", ["sns_sg_file", "raster_sg_file", "raster_sg_files"])
def test_memmap_mode_never_computes_uncertainty(request, files):
    # memmap data are unscaled integers, so no uncertainty is computed from them, and the fill mask is lazy
    files = request.getfixturevalue(files)
    memmap = read_spectrograph_lvl2(files, memmap=True, uncertainty=True)["Si IV 1403"]
    assert memmap.uncertainty is None
    assert isinstance(memmap.mask, da.Array)
    scaled = read_spectrograph_lvl2(files, spectral_windows="Si IV 1403", uncertainty=True)["Si IV 1403"]
    assert scaled.uncertainty is not None


@pytest.mark.parametrize("files", ["raster_sg_file", "raster_sg_files", "sns_sg_file"])
def test_world_coordinates_have_finite_corners_invert_crop_and_survive_asdf(request, tmp_path, files):
    cube = read_spectrograph_lvl2(request.getfixturevalue(files), spectral_windows="Si IV 1403")["Si IV 1403"]
    assert np.isfinite(cube.axis_world_coords("time", pixel_corners=True)[0].jd).all()
    assert np.isfinite(cube.axis_world_coords("custom:pos.helioprojective.lon", pixel_corners=True)[0].Tx).all()

    start, stop = (cube.wcs.array_index_to_world(0, step, 20, 10) for step in (3, 4))
    assert cube.wcs.world_to_array_index(*start) == (0, 3, 20, 10)
    assert cube.wcs.world_to_array_index(*stop) == (0, 4, 20, 10)
    assert cube.crop(start, stop).data.shape == (2,)

    pixel = [np.array([3.0, 10.5]), np.array([5.0, 20.25]), np.array([1.0, 2.0]), np.array([0.0, cube.shape[0] - 1])]
    asdf.AsdfFile({"wcs": cube.wcs}).write_to(tmp_path / "wcs.asdf")
    with asdf.open(tmp_path / "wcs.asdf") as asdf_file:
        wcs = asdf_file["wcs"]
        world = wcs.pixel_to_world_values(*pixel)
        np.testing.assert_array_equal(world, cube.wcs.pixel_to_world_values(*pixel))
        np.testing.assert_allclose(wcs.world_to_pixel_values(*world), pixel)


def test_crop_with_every_world_coordinate(raster_sg_files):
    scan = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Si IV 1403")["Si IV 1403"]
    spectral_coord = SpectralCoord(scan.spectral_axis[len(scan.spectral_axis) // 2])
    last = scan.data.shape[-1] - 1

    assert scan.crop([spectral_coord, None, None, None, None], [spectral_coord, None, None, None, None]).data.ndim == 3
    assert scan.crop(*(scan.wcs.array_index_to_world(0, 3, 50, i) for i in (0, last))).data.ndim == 1
    values = (scan.wcs.array_index_to_world_values(0, 3, 50, i) for i in (0, last))
    assert scan.crop_by_values(*values, units=(u.nm, u.arcsec, u.arcsec, u.s, u.pix, u.pix)).data.ndim == 1


@pytest.mark.parametrize("case", ["raster", "combined", "v34_flip", "v34_revert", "sit_and_stare"])
def test_slit_center_is_at_the_exposure_pointing_and_agrees_with_fits_wcs(request, case):
    """
    Each exposure's slit center must be at its auxiliary FOV center (XCENIX, YCENIX),
    and the WCS must agree with the FITS -TAB ``fits_wcs`` and invert back to its array
    indices.
    """
    fixture = {"raster": "raster_sg_file", "combined": "raster_sg_files", "sit_and_stare": "sns_sg_file"}
    filename = request.getfixturevalue(fixture.get(case, "v34_sg_file"))
    window = "Mg II k 2796"
    cube = read_spectrograph_lvl2(filename, spectral_windows=window, revert_v34=case == "v34_revert")[window][0]
    if case == "combined":
        filename = filename[0]
    header, aux, aux_header = _window_header_and_aux(filename, window)
    if case == "v34_flip":
        aux = aux[::-1]
    n_steps, n_slit = cube.shape[:2]
    slit_center = header["CRPIX2"] - 1

    steps = np.arange(n_steps)
    _, sky, _, _ = cube.wcs.pixel_to_world(np.zeros(n_steps), np.full(n_steps, slit_center), steps)
    # Measured residual is ~1e-10" for every case; 0.01" is far below the smallest real error (~0.17").
    assert_quantity_allclose(sky.Tx, aux[:, aux_header["XCENIX"]] * u.arcsec, atol=0.01 * u.arcsec)
    assert_quantity_allclose(sky.Ty, aux[:, aux_header["YCENIX"]] * u.arcsec, atol=0.01 * u.arcsec)

    if case == "sit_and_stare":
        # Along the slit, each exposure's pixels are CDELT2 apart in its own roll direction.
        steps, rows = np.meshgrid([0, n_steps // 2, n_steps - 1], [0, n_slit - 1])
        _, edge, _, _ = cube.wcs.pixel_to_world(np.zeros(steps.shape), rows, steps)
        along_slit = header["CDELT2"] * (rows - slit_center) * u.arcsec
        assert_quantity_allclose(
            edge.Tx - sky.Tx[steps], along_slit * aux[steps, aux_header["PC3_2IX"]], atol=0.001 * u.arcsec
        )
        assert_quantity_allclose(
            edge.Ty - sky.Ty[steps], along_slit * aux[steps, aux_header["PC2_2IX"]], atol=0.001 * u.arcsec
        )

    for array_index in SIT_AND_STARE_INDICES if case == "sit_and_stare" else RASTER_INDICES:
        spectral, sky, _, _ = cube.wcs.array_index_to_world(*array_index)
        basic_spectral, basic_sky = cube.fits_wcs.array_index_to_world(*array_index)
        assert_quantity_allclose(spectral.to(u.nm), basic_spectral.to(u.nm))
        assert_quantity_allclose(sky.Tx.to(u.arcsec), basic_sky.Tx.to(u.arcsec), atol=0.001 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), basic_sky.Ty.to(u.arcsec), atol=0.001 * u.arcsec)
        assert cube.wcs.world_to_array_index(*cube.wcs.array_index_to_world(*array_index)) == array_index


def test_sit_and_stare_pointing_does_not_ramp_between_exposures(sns_sg_file):
    """
    A sit-and-stare slit does not move between exposures, so a fractional step takes the
    pointing of the nearest exposure instead of ramping at the virtual step scale, which
    made plotted longitude ticks repeat.
    """
    wcs = read_spectrograph_lvl2(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"][0].wcs.low_level_wcs
    steps = np.array([4.0, 4.25, 4.49, 4.51, 5.0])
    _, longitude, latitude, _, _ = wcs.pixel_to_world_values(np.zeros(5), np.zeros(5), steps)
    np.testing.assert_array_equal(longitude, longitude[[0, 0, 0, 4, 4]])
    np.testing.assert_array_equal(latitude, latitude[[0, 0, 0, 4, 4]])
    assert longitude[4] != longitude[0]


def test_celestial_frame_is_slicing_invariant(raster_sg_files):
    cube = read_spectrograph_lvl2(raster_sg_files)["Si IV 1403"]

    frame = cube.celestial_frame
    assert isinstance(frame, Helioprojective)
    # Combined cube has fits_wcs None; slices must still agree.
    assert cube[0].celestial_frame == frame
    assert cube[0, 2].celestial_frame == frame
    assert cube[:, 1:3].celestial_frame == frame
    assert cube[0, 2, 50].celestial_frame == frame

    wcs_frame = wcs_to_celestial_frame(cube[0].fits_wcs.celestial)
    # The FITS WCS frame must carry an obstime, otherwise coordinates built in
    # celestial_frame cannot be transformed into it (they become NaN).
    assert wcs_frame.obstime is not None
    roundtripped = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame).transform_to(wcs_frame)
    assert np.isfinite(roundtripped.Tx.value)
    assert np.isfinite(roundtripped.Ty.value)


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

    cube = read_spectrograph_lvl2(filename, spectral_windows="C II 1336", memmap=memmap)["C II 1336"][0]

    np.testing.assert_array_equal(cube.mask, expected_mask)
    if memmap:
        np.testing.assert_array_equal(cube.data, raw)


def test_read_spectrograph_uncertainty_is_a_standard_deviation(raster_sg_file):
    # A bare array would be stored as an UnknownUncertainty (issue #57).
    cube = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336", uncertainty=True)["C II 1336"][0]
    assert isinstance(cube.uncertainty, StdDevUncertainty)
    assert cube.uncertainty.array.shape == cube.data.shape


def test_an_exposure_without_pointing_takes_it_from_its_neighbors():
    # Step 0 of this raster has no exposure, and so no pointing (see data_gap/overview.txt)
    filename = get_test_filepath("data_gap/iris_l2_20150917_073915_3630104144_raster_t000_r00102_test.fits")
    with pytest.warns(UserWarning, match=r"Found 1 step\(s\) with no pointing"):
        cube = read_spectrograph_lvl2(filename)["Si IV 1403"][0]
    steps = np.arange(cube.shape[0])
    slit_center = np.full(steps.size, fits.getheader(filename, 1)["CRPIX2"] - 1)
    sky = cube.wcs.pixel_to_world(np.zeros(steps.size), slit_center, steps)[1]
    fov_center = cube.meta["exposure FOV center"]

    assert np.isnan(fov_center.Tx[0])
    assert_quantity_allclose(sky.Tx[1:], fov_center.Tx[1:], atol=1e-6 * u.arcsec)
    assert_quantity_allclose(sky.Ty[1:], fov_center.Ty[1:], atol=1e-6 * u.arcsec)
    # Before the first exposure with pointing, on the line through the first two
    assert_quantity_allclose(sky.Tx[0], 2 * sky.Tx[1] - sky.Tx[2])
    assert_quantity_allclose(sky.Ty[0], 2 * sky.Ty[1] - sky.Ty[2])


def test_world_coordinates_of_a_later_raster_invert(raster_sg_files):
    cube = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Si IV 1403")["Si IV 1403"]
    for index in ((1, 0, 20, 10), (1, 7, 100, 5), (2, 3, 50, 0)):
        assert cube.wcs.world_to_array_index(*cube.wcs.array_index_to_world(*index)) == index


def test_memmap_records_the_window_scaling():
    # memmap=True keeps the FITS integers, so the window's BSCALE and BZERO go in the metadata
    filename = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    cube = read_spectrograph_lvl2(filename, memmap=True, spectral_windows="Mg II k 2796")["Mg II k 2796"]
    assert np.issubdtype(cube.data.dtype, np.integer)
    with fits.open(filename) as hdulist:
        header = hdulist[3].header
    assert (cube.meta["BSCALE"], cube.meta["BZERO"]) == (header["BSCALE"], header["BZERO"])


@pytest.mark.parametrize(("window", "band"), [("O I 1356", "FUV"), ("Mg II k 2796", "NUV")])
def test_memmap_unit_is_unscaled(window, band, raster_sg_file):
    filename = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    assert read_spectrograph_lvl2(filename, memmap=True)[window][0].unit == DN_UNIT[f"{band}_UNSCALED"]
    assert read_spectrograph_lvl2(filename)[window][0].unit == DN_UNIT[band]
    # Float data hold DN already
    assert read_spectrograph_lvl2(raster_sg_file, memmap=True)["C II 1336"][0].unit == DN_UNIT["FUV"]


def test_unscaled_opened_file_needs_memmap():
    filename = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    with fits.open(filename, do_not_scale_image_data=True) as hdulist:
        with pytest.raises(ValueError, match="needs memmap=True"):
            read_spectrograph_lvl2(hdulist, spectral_windows="Mg II k 2796")
        cube = read_spectrograph_lvl2(hdulist, spectral_windows="Mg II k 2796", memmap=True)["Mg II k 2796"][0]
    assert cube.unit == DN_UNIT["NUV_UNSCALED"]


def test_read_spectrograph_masks_clipped_samples(tmp_path, sns_sg_file):
    with fits.open(sns_sg_file) as hdul:
        window = hdul[0].header["TDESC1"]
        data = hdul[1].data.astype("float32")
        data.flat[7] = SATURATION_LIMIT.value
        hdul[1].data = data
        hdul.writeto(tmp_path / "clipped.fits")
    raster = read_spectrograph_lvl2(tmp_path / "clipped.fits", uncertainty=True)
    cube = raster[window][0]
    # The clipped sample keeps its value, masked and marked saturated
    assert np.flatnonzero(cube.meta["saturated"]).tolist() == [7]
    assert cube.data.flat[7] == SATURATION_LIMIT.value
    assert cube.mask.flat[7]
    assert np.isfinite(cube.uncertainty.array.flat[7])
    # Windows without clipped samples have no saturated meta
    assert not any("saturated" in other.meta for name, other in raster.items() if name != window)


def test_combined_rasters_mark_clipped_samples(tmp_path, raster_sg_files):
    # Only the second file has a clipped sample
    clipped = tmp_path / "clipped.fits"
    with _edited_copy(raster_sg_files[1], clipped) as hdulist:
        ext = _window_ext(hdulist, "Si IV 1403")
        data = hdulist[ext].data.astype("float32")
        data[2, 50, 10] = SATURATION_LIMIT.value
        hdulist[ext].data = data

    cube = read_spectrograph_lvl2([raster_sg_files[0], clipped], spectral_windows="Si IV 1403")["Si IV 1403"]

    expected = np.zeros(cube.shape, dtype=bool)
    expected[1, 2, 50, 10] = True
    np.testing.assert_array_equal(cube.meta["saturated"], expected)
    assert cube.mask[1, 2, 50, 10]
    np.testing.assert_array_equal(cube[1].meta["saturated"], expected[1])
