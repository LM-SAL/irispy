from contextlib import contextmanager

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

from ndcube import NDCube
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

import irispy.io.spectrograph as spectrograph_io
from irispy._spectrograph_wcs import (
    _create_raster_gwcs,
    _raster_wcs_bad_row_mask,
    _sanitize_raster_times,
    _sanitize_raster_wcs_tables,
)
from irispy.io.spectrograph import _nuv_t_obs_from_source_filenames, read_spectrograph_lvl2
from irispy.spectrograph import SpectrogramCube
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED

SG_METADATA = {
    "sns_sg_file": {
        "keys": ["C II 1336", "Fe XII 1349", "O I 1356", "Si IV 1394", "Si IV 1403", "2832", "2814", "Mg II k 2796"],
        "shape": (187, 40, 29),
        "n_rasters": 1,
        "campaign_start_start_end": ("2021-09-05T00:18:33.640", "2021-09-05T00:18:33.810", "2021-09-05T05:07:27.400"),
        "distance_to_sun": 1.00827638 * u.AU,
        "exposure_control_triggers_in_observation": 0,
        "header_length": 380,
        "meta_length": 369,
        "unique_and_total_raster_positions": (1, 1),
        "observing_mode": ("Medium sit-and-stare 0.3x60 1s  C II   Si IV   Mg II h/k   Mg II w s", 3620258102),
        "raster_fov_width": (0.16635 * u.arcsec, 66.54 * u.arcsec),
        "satellite_rotation": 8.09432e-05 * u.deg,
        "spectral_range": (1398.60550787, 1406.03398787) * u.angstrom,
    },
    "raster_sg_files": {
        "keys": ["C II 1336", "1343", "Fe XII 1349", "O I 1356", "Si IV 1403", "2832", "2826", "2814", "Mg II k 2796"],
        "shape": (13, 8, 109, 29),
        "n_rasters": 13,
        "campaign_start_start_end": ("2014-03-29T14:09:38.830", "2014-03-29T14:09:39.000", "2014-03-29T14:25:43.280"),
        "distance_to_sun": 0.99849015 * u.AU,
        "exposure_control_triggers_in_observation": 526,
        "header_length": 412,
        "meta_length": 402,
        "unique_and_total_raster_positions": (8, 180),
        "observing_mode": ("Very large coarse 8-step raster 14x175 8s  Si IV   Mg II h/k   Mg II", 3860258481),
        "raster_fov_width": (13.9680814743 * u.arcsec, 181.987 * u.arcsec),
        "satellite_rotation": -0.000540529 * u.deg,
        "spectral_range": (1398.63094787, 1405.95766787) * u.angstrom,
    },
}
V34_WINDOWS = ["Si IV 1403", "Mg II k 2796"]
# Raster indices are (step, slit, wavelength); sit-and-stare has 187 exposures of 40 slit pixels.
RASTER_INDICES = ((0, 50, 3), (3, 50, 10), (4, 50, 10), (7, 80, 20), (0, 80, 20))
SIT_AND_STARE_INDICES = ((0, 0, 10), (3, 20, 10), (4, 25, 10), (10, 30, 5), (93, 39, 5), (186, 0, 5))


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
    raster_collection = read_spectrograph_lvl2(request.getfixturevalue(files))
    assert list(raster_collection.keys()) == expected["keys"]
    assert str(raster_collection)
    # We do not expect any metadata to be present on the collection
    assert raster_collection.meta is None

    si_iv = raster_collection["Si IV 1403"]
    assert str(si_iv)
    assert si_iv.time.ravel()[0].isot in str(si_iv[0])
    meta = si_iv.meta
    assert si_iv.data.shape == expected["shape"]
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
    assert meta.fov_center == SkyCoord(Tx=meta.get("XCEN"), Ty=meta.get("YCEN"), unit=u.arcsec, frame=Helioprojective)
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

    # TODO: Decide if I want to set observer_location, observer_radial_velocity, rsun_angular, run_meters
    # These are more WCS properties...
    assert meta.observer_location is None
    assert meta.rsun_angular is None
    assert meta.rsun_meters is None
    assert si_iv.wcs.pixel_n_dim == si_iv.data.ndim
    physical_types = ("em.wl", "custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat", "time")
    physical_types += ("custom:STEP", "custom:SCAN")[: si_iv.data.ndim - 2]
    assert si_iv.wcs.world_axis_physical_types == physical_types
    assert si_iv.time.format == "isot"
    assert si_iv.time.shape == si_iv.shape[:-2]
    n_rasters = expected["n_rasters"]
    assert (si_iv.fits_wcs is None) == (n_rasters > 1)
    assert len(si_iv.split_rasters()) == n_rasters
    assert si_iv.raster_slice(0).shape == si_iv.raster_slice(-1).shape == si_iv.shape[-3:]
    with pytest.raises(IndexError, match=r"Raster index out of range."):
        si_iv.raster_slice(-n_rasters - 1)
    with pytest.raises(TypeError, match="integer"):
        si_iv.raster_slice("0")


def test_combined_raster_metadata_and_time_match_its_rasters(raster_sg_files):
    scan = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Si IV 1403")["Si IV 1403"]

    time = scan.time
    assert time[0, 0].isot == scan.raster_slice(0).time[0].isot
    assert time[-1, -1].isot == scan.raster_slice(-1).time[-1].isot
    assert (time.isot[1] == scan.raster_slice(1).time.isot).all()
    # Exact, so partial time crops by a per-file time find their exposure.
    last = read_spectrograph_lvl2(raster_sg_files[-1], spectral_windows="Si IV 1403")["Si IV 1403"]
    assert (time[-1] == last.time).all()
    assert scan.meta["DATE_END"] == last.meta["DATE_END"]


@pytest.mark.parametrize("files", ["sns_sg_file", "raster_sg_file"])
def test_read_spectrograph_lvl2_keeps_requested_window_order_and_data_and_reports_missing(request, files):
    filename = request.getfixturevalue(files)
    requested_windows = ["Mg II k 2796", "C II 1336"]

    all_windows = read_spectrograph_lvl2(filename)
    raster_collection = read_spectrograph_lvl2(filename, spectral_windows=requested_windows)

    assert list(raster_collection.keys()) == requested_windows
    for window in requested_windows:
        assert raster_collection[window].meta.spectral_window == window
        np.testing.assert_array_equal(raster_collection[window].data, all_windows[window].data)
    with pytest.raises(ValueError, match=r"Spectral windows \['NOPE1', 'NOPE2'\] not in file"):
        read_spectrograph_lvl2(filename, spectral_windows=[*requested_windows, "NOPE1", "NOPE2"])


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
        cube = raster[window]
        assert list(cube.fits_wcs.wcs.ctype)[1:] == ["HPLT-TAB", "HPLN-TAB"]
        assert cube.fits_wcs.wcs.aux.dsun_obs > 1e11
        sky = cube.fits_wcs.pixel_to_world(*pixels)[1]
        np.testing.assert_allclose(sky.Tx.to_value(u.deg), expected_longitude)
        np.testing.assert_allclose(sky.Ty.to_value(u.deg), expected_latitude)

    if files == "raster_sg_file":
        assert raster[windows[0]].time[0].isot == "2014-03-29T14:09:43.000"
        assert raster[windows[1]].time[0].isot == "2014-03-29T14:09:42.940"
        assert raster[windows[0]].meta["auxiliary times"][0].isot == "2014-03-29T14:09:39.000"


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
        cube = read_spectrograph_lvl2(filename, spectral_windows="Mg II k 2796")["Mg II k 2796"]

    assert str(filename) in caplog.text
    assert "EXPTIMEN is 0 s at row(s) [0]" in caplog.text
    assert cube.meta["exposure time"][0] == 0 * u.s
    assert np.all(cube.mask[0])
    assert not np.all(cube.mask[1])
    assert abs((cube.time[0] - cube.meta["auxiliary times"][0]).to_value(u.s)) < 1e-6


@pytest.fixture
def v34_raster_file(tmp_path, raster_sg_file):
    """
    The raster marked V34 (STEPS_AV < -0.01), which is read flipped along the step axis,
    with the first exposure of each of ``V34_WINDOWS`` marked bad.
    """
    filename = tmp_path / "v34.fits"
    with _edited_copy(raster_sg_file, filename) as hdulist:
        hdulist[0].header["STEPS_AV"] = -1.0
        for window in V34_WINDOWS:
            hdulist[_window_ext(hdulist, window)].data[0] = BAD_PIXEL_VALUE_SCALED
    return filename


def test_read_spectrograph_flips_v34_data_meta_and_world_coordinates(v34_raster_file):
    """
    Everything per exposure is reversed when a V34 raster is read flipped (default), so
    the same physical exposure keeps its world coordinates whether or not
    ``revert_v34=True``.
    """
    flipped = read_spectrograph_lvl2(v34_raster_file, spectral_windows=V34_WINDOWS, uncertainty=True)
    original = read_spectrograph_lvl2(v34_raster_file, spectral_windows=V34_WINDOWS, uncertainty=True, revert_v34=True)
    _, aux, aux_header = _window_header_and_aux(v34_raster_file, V34_WINDOWS[0])
    for window in V34_WINDOWS:
        cube, reference = flipped[window], original[window]
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
        centre, reference_centre = cube.meta["exposure FOV center"], reference.meta["exposure FOV center"][::-1]
        np.testing.assert_array_equal(centre.Tx, reference_centre.Tx)
        np.testing.assert_array_equal(centre.Ty, reference_centre.Ty)
        assert_quantity_allclose(centre.Tx.to(u.arcsec), aux[::-1, aux_header["XCENIX"]] * u.arcsec)
        assert_quantity_allclose(centre.Ty.to(u.arcsec), aux[::-1, aux_header["YCENIX"]] * u.arcsec)

        for step, slit, wavelength in ((0, 50, 3), (3, 50, 10), (7, 80, 20)):
            spectral, sky, time, _ = cube.wcs.array_index_to_world(step, slit, wavelength)
            mirrored_spectral, mirrored_sky, mirrored_time, _ = reference.wcs.array_index_to_world(
                cube.shape[0] - 1 - step, slit, wavelength
            )
            assert_quantity_allclose(spectral.to(u.nm), mirrored_spectral.to(u.nm))
            assert_quantity_allclose(sky.Tx.to(u.arcsec), mirrored_sky.Tx.to(u.arcsec), atol=0.01 * u.arcsec)
            assert_quantity_allclose(sky.Ty.to(u.arcsec), mirrored_sky.Ty.to(u.arcsec), atol=0.01 * u.arcsec)
            assert time.isclose(mirrored_time)
        # Away from an exposure's centre, the sky also depends on the sign of the step column of PC.
        _, sky, _, _ = cube.wcs.pixel_to_world(10, 50, 3.3)
        _, mirrored_sky, _, _ = reference.wcs.pixel_to_world(10, 50, cube.shape[0] - 1 - 3.3)
        assert_quantity_allclose(sky.Tx.to(u.arcsec), mirrored_sky.Tx.to(u.arcsec), atol=0.01 * u.arcsec)
        assert_quantity_allclose(sky.Ty.to(u.arcsec), mirrored_sky.Ty.to(u.arcsec), atol=0.01 * u.arcsec)


def test_memmap_combined_v34_rasters_read_flipped_chunks(tmp_path, raster_sg_files, monkeypatch):
    files = [tmp_path / f"v34_{index}.fits" for index in range(2)]
    for source, filename in zip(raster_sg_files[:2], files, strict=True):
        with _edited_copy(source, filename) as hdulist:
            hdulist[0].header["STEPS_AV"] = -1.0
    # Chunks of 3 rows split the 8 steps unevenly, so each chunk must map back to its own rows.
    monkeypatch.setattr("irispy.io._raster_combine._lazy_raster_scan_chunk_rows", lambda _: 3)

    cube = read_spectrograph_lvl2(files, spectral_windows="Si IV 1403", memmap=True)["Si IV 1403"]

    for scan, filename in enumerate(files):
        with fits.open(filename, do_not_scale_image_data=True) as hdulist:
            np.testing.assert_array_equal(
                np.asarray(cube.data[scan]), hdulist[_window_ext(hdulist, "Si IV 1403")].data[::-1]
            )


def test_raster_wcs_steps_have_no_index_vector(raster_sg_file):
    # wcslib searches a -TAB index vector linearly, so a 1..N step index made each lookup O(step)
    fits_wcs = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336")["C II 1336"].fits_wcs
    assert "PS3_2" not in fits_wcs.to_header()

    steps = np.array([0, 0.5, 1, 6.25, 7])
    world = fits_wcs.pixel_to_world_values(np.zeros(5), np.full(5, 50.0), steps)
    np.testing.assert_allclose(world[2][1], (world[2][0] + world[2][2]) / 2)
    np.testing.assert_allclose(fits_wcs.world_to_pixel_values(*world)[2], steps, atol=1e-6)


def test_read_spectrograph_lvl2_raises_when_a_window_wcs_fails(sns_sg_file, monkeypatch):
    def fail(*_args, **_kwargs):
        msg = "bad test WCS"
        raise ValueError(msg)

    monkeypatch.setattr(spectrograph_io, "_create_raster_gwcs", fail)

    with pytest.raises(ValueError, match="bad test WCS") as excinfo:
        spectrograph_io.read_spectrograph_lvl2(sns_sg_file, spectral_windows=["C II 1336", "Fe XII 1349"])
    assert excinfo.value.__notes__ == [f"While building the WCS of spectral window 'C II 1336' in {sns_sg_file}."]


@pytest.mark.parametrize("column", ["PC2_2IX", "PC3_2IX", "XCENIX"])
@pytest.mark.parametrize("files", ["raster_sg_file", "sns_sg_file"])
def test_read_spectrograph_lvl2_interpolates_non_finite_aux_rows(request, tmp_path, files, column):
    filename = tmp_path / "nan_row.fits"
    with _edited_copy(request.getfixturevalue(files), filename) as hdulist:
        hdulist[-2].data[2, hdulist[-2].header[column]] = np.nan

    with pytest.warns(UserWarning, match=r"Found 1 step\(s\) with all-zero or non-finite WCS tables"):
        cube = read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403")["Si IV 1403"]

    _, sky, _, _ = cube.wcs.pixel_to_world(np.zeros(3), np.full(3, 20), [1, 2, 3])
    assert np.isfinite(sky.Tx).all()
    assert np.isfinite(sky.Ty).all()


@pytest.mark.parametrize("column", ["TIME", "EXPTIMEF", "EXPTIMEN"])
@pytest.mark.parametrize("files", ["raster_sg_file", "sns_sg_file"])
def test_read_spectrograph_lvl2_interpolates_non_finite_times(request, tmp_path, caplog, files, column):
    filename = tmp_path / "nan_time.fits"
    with _edited_copy(request.getfixturevalue(files), filename) as hdulist:
        hdulist[-2].data[2, hdulist[-2].header[column]] = np.nan
    window = "Mg II k 2796" if column == "EXPTIMEN" else "Si IV 1403"

    with (
        caplog.at_level("WARNING", logger="sunpy"),
        pytest.warns(UserWarning, match=r"Found 1 step\(s\) with non-finite times"),
    ):
        cube = read_spectrograph_lvl2(filename, spectral_windows=window)[window]
    assert "EXPTIMEN is 0 s" not in caplog.text

    assert (np.diff(cube.time.jd) > 0).all()
    # TIME sets the auxiliary (start) times, and EXPTIMEF/EXPTIMEN the midpoints derived from them.
    times = cube.meta["auxiliary times"] if column == "TIME" else cube.time
    assert abs((times[2] - times[1] - (times[3] - times[2])).to_value(u.s)) < 1e-6
    # cube.time comes from the gWCS; the time extra coordinate is the stored T_OBS.
    t_obs = cube.axis_world_coords("time", wcs=cube.extra_coords)[0]
    assert abs((t_obs - cube.time).to_value(u.s)).max() < 1e-6


def test_read_spectrograph_lvl2_names_the_file_when_every_time_is_non_finite(tmp_path, raster_sg_file):
    filename = tmp_path / "all_nan_time.fits"
    with _edited_copy(raster_sg_file, filename) as hdulist:
        hdulist[-2].data[:, hdulist[-2].header["TIME"]] = np.nan

    with pytest.raises(ValueError, match="Every exposure has non-finite times") as excinfo:
        read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403")
    assert excinfo.value.__notes__ == [f"While reading the auxiliary times in {filename}."]


@pytest.mark.parametrize(("column", "window"), [("EXPTIMEF", "Si IV 1403"), ("EXPTIMEN", "Mg II k 2796")])
def test_read_spectrograph_lvl2_uses_planned_times_when_every_exposure_time_is_non_finite(
    tmp_path, raster_sg_file, column, window
):
    filename = tmp_path / "all_nan_exposure_time.fits"
    with _edited_copy(raster_sg_file, filename) as hdulist:
        hdulist[-2].data[:, hdulist[-2].header[column]] = np.nan

    with pytest.warns(UserWarning, match="Using the planned start times"):
        cube = read_spectrograph_lvl2(filename, spectral_windows=window)[window]
    assert abs((cube.time - cube.meta["auxiliary times"]).to_value(u.s)).max() < 1e-6


def test_read_spectrograph_lvl2_requires_startobs(tmp_path, raster_sg_file):
    filename = tmp_path / "no_startobs.fits"
    with _edited_copy(raster_sg_file, filename) as hdulist:
        del hdulist[0].header["STARTOBS"]

    with pytest.raises(KeyError, match="STARTOBS"):
        read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        ({"CDELT1": None}, "Missing WCS header key 'CDELT1'"),
        ({"CDELT3": None}, "Missing WCS header key 'CDELT3'"),
        ({"CDELT1": "abc"}, "'CDELT1' must be numeric"),
        ({"CDELT1": "NAN"}, "'CDELT1' must be finite"),
        # wcslib can abort the process on a bad CRPIX2, so it must be caught before the FITS-TAB WCS is built.
        ({"CRPIX2": "NAN"}, "'CRPIX2' must be finite"),
        # Extrapolating past the last exposure from extreme finite pointings overflows.
        ({"XCENIX": [-1.7e308, 1.7e308, np.nan]}, "CRVAL table must contain only finite values"),
    ],
    ids=["missing", "missing_cdelt3", "non_numeric", "non_finite", "crpix2_before_fits_wcs", "overflow"],
)
@pytest.mark.filterwarnings("ignore:Found 1 step:UserWarning")
def test_read_spectrograph_lvl2_rejects_bad_wcs_inputs(tmp_path, raster_sg_file, edit, match):
    filename = tmp_path / "bad.fits"
    with _edited_copy(raster_sg_file, filename) as hdulist:
        header = hdulist[_window_ext(hdulist, "Si IV 1403")].header
        for key, value in edit.items():
            if key in hdulist[-2].header:
                hdulist[-2].data[-len(value) :, hdulist[-2].header[key]] = value
            elif value is None:
                del header[key]
            else:
                header[key] = value

    with pytest.raises(ValueError, match=match) as excinfo:
        read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403")
    assert excinfo.value.__notes__ == [f"While building the WCS of spectral window 'Si IV 1403' in {filename}."]


@pytest.mark.parametrize("files", ["sns_sg_file", "raster_sg_file", "raster_sg_files"])
def test_memmap_mode_never_computes_uncertainty(request, files):
    # memmap data are unscaled integers, so no uncertainty (or mask) is computed from them
    files = request.getfixturevalue(files)
    with pytest.warns(UserWarning, match="uncertainty is not computed when memmap=True"):
        memmap = read_spectrograph_lvl2(files, memmap=True, uncertainty=True)["Si IV 1403"]
    assert memmap.uncertainty is None
    assert memmap.mask is None
    scaled = read_spectrograph_lvl2(files, spectral_windows="Si IV 1403", uncertainty=True)["Si IV 1403"]
    assert scaled.uncertainty is not None


@pytest.mark.parametrize(("key", "value"), [("OBSID", 9999999999), ("STARTOBS", "2014-03-29T15:00:00.000")])
def test_read_spectrograph_lvl2_rejects_mismatched_observation(tmp_path, raster_sg_files, key, value):
    other_observation = tmp_path / "other_observation.fits"
    with _edited_copy(raster_sg_files[1], other_observation) as hdulist:
        hdulist[0].header[key] = value

    with pytest.raises(ValueError, match=f"same {key}"):
        read_spectrograph_lvl2([raster_sg_files[0], other_observation])


def test_read_spectrograph_lvl2_rejects_mismatched_window_shape(tmp_path, raster_sg_files):
    narrow = tmp_path / "narrow.fits"
    with _edited_copy(raster_sg_files[1], narrow) as hdulist:
        window_ext = _window_ext(hdulist, "Si IV 1403")
        hdulist[window_ext].data = hdulist[window_ext].data[:, :, :-1]

    with pytest.raises(ValueError, match="same slit and wavelength dimensions"):
        read_spectrograph_lvl2([raster_sg_files[0], narrow], spectral_windows="Si IV 1403")


def test_combined_raster_pads_short_final_raster(tmp_path, raster_sg_files):
    short_raster = tmp_path / "short_raster.fits"
    with _edited_copy(raster_sg_files[1], short_raster) as hdulist:
        window_ext = _window_ext(hdulist, "Si IV 1403")
        hdulist[window_ext].data = hdulist[window_ext].data[:-1]
        hdulist[-2].data = hdulist[-2].data[:-1]
    files = [raster_sg_files[0], short_raster]

    with pytest.warns(UserWarning, match="mismatched step counts"):
        combined = read_spectrograph_lvl2(files, spectral_windows="Si IV 1403", uncertainty=True)["Si IV 1403"]

    assert combined.shape == (2, 8, 109, 29)
    assert combined.data.dtype == np.float32
    assert combined.mask.dtype == bool
    assert np.isnan(combined.data[1, -1]).all()
    assert combined.mask[1, -1].all()
    assert combined.uncertainty.array.shape == combined.shape
    assert np.isnan(combined.uncertainty.array[1, -1]).all()
    # The short raster's missing step repeats its last exposure's pointing and roll.
    last, padded = (combined.wcs.array_index_to_world(1, step, 0, 0)[1] for step in (6, 7))
    assert_quantity_allclose(padded.Tx, last.Tx)
    assert_quantity_allclose(padded.Ty, last.Ty)
    assert combined.time.shape == combined.shape[:2]
    assert combined.time[1, -1].isot == combined.time[1, -2].isot

    with pytest.raises(ValueError, match="memmap=True does not support raster files with mismatched step counts"):
        read_spectrograph_lvl2(files, spectral_windows="Si IV 1403", memmap=True)


@pytest.mark.parametrize("files", ["raster_sg_file", "raster_sg_files", "sns_sg_file"])
def test_gwcs_corners_inverse_crop_and_asdf_round_trip(request, tmp_path, files):
    cube = read_spectrograph_lvl2(request.getfixturevalue(files), spectral_windows="Si IV 1403")["Si IV 1403"]
    assert np.isfinite(cube.axis_world_coords("time", pixel_corners=True)[0].jd).all()
    assert np.isfinite(cube.axis_world_coords("custom:pos.helioprojective.lon", pixel_corners=True)[0].Tx).all()

    prefix = (0,) if cube.data.ndim == 4 else ()
    start, stop = (cube.wcs.array_index_to_world(*prefix, step, 20, 10) for step in (3, 4))
    assert cube.wcs.world_to_array_index(*start) == (*prefix, 3, 20, 10)
    assert cube.wcs.world_to_array_index(*stop) == (*prefix, 4, 20, 10)
    assert cube.crop(start, stop).data.shape == (2,)

    pixel = [np.array([3.0, 10.5]), np.array([5.0, 20.25]), np.array([1.0, 2.0]), np.array([0.0, 2.0])]
    pixel = pixel[: cube.data.ndim]
    asdf.AsdfFile({"wcs": cube.wcs}).write_to(tmp_path / "wcs.asdf")
    with asdf.open(tmp_path / "wcs.asdf") as asdf_file:
        wcs = asdf_file["wcs"]
        world = wcs.pixel_to_world_values(*pixel)
        np.testing.assert_array_equal(world, cube.wcs.pixel_to_world_values(*pixel))
        np.testing.assert_allclose(wcs.world_to_pixel_values(*world), pixel)


def test_gwcs_crop_supports_full_world_components(raster_sg_files):
    scan = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Si IV 1403")["Si IV 1403"]
    spectral_coord = SpectralCoord(scan.spectral_axis[len(scan.spectral_axis) // 2])
    last = scan.data.shape[-1] - 1

    assert scan.crop([spectral_coord, None, None, None, None], [spectral_coord, None, None, None, None]).data.ndim == 3
    assert scan.crop(*(scan.wcs.array_index_to_world(0, 3, 50, i) for i in (0, last))).data.ndim == 1
    values = (scan.wcs.array_index_to_world_values(0, 3, 50, i) for i in (0, last))
    assert scan.crop_by_values(*values, units=(u.nm, u.arcsec, u.arcsec, u.s, u.pix, u.pix)).data.ndim == 1


# TODO: remove once irispy requires an ndcube release with the crop-bounds hook.
requires_crop_hook = pytest.mark.skipif(
    not hasattr(NDCube, "_get_crop_bounds"), reason="partial raster crops need ndcube with the crop-bounds hook"
)


@pytest.fixture(params=[False, True], ids=["single", "combined"])
def partial_raster_tables(request):
    """
    The data shape and the (scan,) step-indexed PC, CRVAL and time tables of a synthetic
    raster.
    """
    shape = (2, 6, 4, 5) if request.param else (6, 4, 5)
    table_shape = shape[:-2]
    pc = np.broadcast_to(np.eye(2), (*table_shape, 2, 2)) * u.pix
    crval = np.zeros((*table_shape, 2)) * u.arcsec
    crval[..., 0] = np.arange(table_shape[-1]) * u.arcsec  # slit moves CDELT3 per step
    # Uneven cadence rules out a two-sample linear inverse.
    dt = np.arange(np.prod(table_shape)).reshape(table_shape) ** 2 * u.s
    return shape, pc, crval, dt


def _partial_raster_cube(shape, pc, crval, dt):
    wcs = _synthetic_raster_gwcs(pc, crval, dt, CDELT2=1, CDELT3=1, CRPIX2=1)
    return SpectrogramCube(np.arange(np.prod(shape)).reshape(shape), wcs=wcs, uncertainty=None, unit=u.DN, meta={})


@pytest.fixture
def partial_raster_cube(partial_raster_tables):
    return _partial_raster_cube(*partial_raster_tables)


@requires_crop_hook
@pytest.mark.parametrize("coordinate", ["sky", "time", "step", "sky_step", "spectral_step"])
def test_gwcs_partial_crop(partial_raster_cube, coordinate):
    cube = partial_raster_cube
    wcs = cube.wcs
    combined = cube.data.ndim == 4
    objects, axes = {
        "sky": ((1,), (1, 2)),
        "time": ((2,), (3,)),
        "step": ((3,), (4,)),
        "sky_step": ((1, 3), (1, 2, 4)),
        "spectral_step": ((0, 3), (0, 4)),
    }[coordinate]
    prefix = (0,) if combined else ()
    points = []
    values = []
    wavelength_bounds = (1, 3) if coordinate == "spectral_step" else (0, 4)
    for array_index in ((*prefix, 1, 1, wavelength_bounds[0]), (*prefix, 3, 2, wavelength_bounds[1])):
        world = wcs.array_index_to_world(*array_index)
        assert wcs.world_to_array_index(*world) == array_index
        points.append([value if i in objects else None for i, value in enumerate(world)])
        values.append(
            [value if i in axes else None for i, value in enumerate(wcs.array_index_to_world_values(*array_index))]
        )
    expected = cube.data[..., 1:4, 1:3, :] if "sky" in coordinate else cube.data[..., 1:4, :, :]
    if combined and coordinate == "time":
        expected = expected[0]
    if coordinate == "spectral_step":
        expected = expected[..., 1:4]
    cropped = cube.crop(*points)
    np.testing.assert_array_equal(cropped.data, expected)
    np.testing.assert_array_equal(cube.crop_by_values(*values, units=wcs.world_axis_units).data, expected)
    origin = (0,) * cropped.data.ndim
    assert cropped.wcs.world_to_array_index(*cropped.wcs.array_index_to_world(*origin)) == origin


@requires_crop_hook
def test_gwcs_partial_time_crop_keeps_repeated_matches(partial_raster_tables):
    shape, pc, crval, dt = partial_raster_tables
    # Both rasters have the same nonmonotonic timestamps; step 2 lies between matches.
    cube = _partial_raster_cube(shape, pc, crval, np.broadcast_to([0, 1, 4, 1, 0, 9], dt.shape) * u.s)
    point = [None] * len(cube.wcs.world_axis_object_classes)
    point[2] = cube.wcs.array_index_to_world(*((0,) if cube.data.ndim == 4 else ()), 1, 0, 0)[2]
    np.testing.assert_array_equal(cube.crop(point).data, cube.data[..., 1:4, :, :])


@requires_crop_hook
def test_gwcs_partial_sky_crop_uses_each_pointing(partial_raster_tables):
    shape, pc, crval, dt = partial_raster_tables
    if len(shape) == 4:
        # Only the second raster covers the target sky position.
        crval[1, :, 0] += 100 * u.arcsec
    else:
        # Exposure 4 is pointed back between exposures 1 and 3, so it covers the target too.
        crval[4, 0] = 2 * u.arcsec
    cube = _partial_raster_cube(shape, pc, crval, dt)
    prefix = (1,) if cube.data.ndim == 4 else ()
    world = cube.wcs.array_index_to_world(*prefix, 1, 1, 0)
    other = cube.wcs.array_index_to_world(*prefix, 3, 2, 0)
    points = [[value if i == 1 else None for i, value in enumerate(point)] for point in (world, other)]
    expected = cube.data[1, 1:4, 1:3, :] if prefix else cube.data[1:5, 1:3, :]
    np.testing.assert_array_equal(cube.crop(*points).data, expected)


@requires_crop_hook
@pytest.mark.parametrize("coordinate", ["sky", "time", "step"])
def test_gwcs_partial_crop_after_slicing(partial_raster_cube, coordinate):
    cube = partial_raster_cube[..., 1:5, 1:4, 1:]
    index = {"sky": 1, "time": 2, "step": 3}[coordinate]
    prefix = (0,) if cube.data.ndim == 4 else ()
    world = cube.wcs.array_index_to_world(*prefix, 1, 1, 0)
    point = [value if i == index else None for i, value in enumerate(world)]
    expected = cube.data[..., 1:2, 1:2, :] if coordinate == "sky" else cube.data[..., 1:2, :, :]
    if prefix and coordinate == "time":
        expected = expected[:1]
    np.testing.assert_array_equal(cube.crop(point, keepdims=True).data, expected)


@requires_crop_hook
def test_gwcs_partial_sky_crop_with_rotation(partial_raster_tables):
    shape, pc, crval, dt = partial_raster_tables
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cube = _partial_raster_cube(shape, np.broadcast_to(rotation, pc.shape) * u.pix, crval, dt)
    prefix = (0,) if cube.data.ndim == 4 else ()
    world = cube.wcs.array_index_to_world(*prefix, 2, 1, 0)
    point = [value if i == 1 else None for i, value in enumerate(world)]
    np.testing.assert_array_equal(cube.crop(point).data, cube.data[..., 2, 1, :])


@requires_crop_hook
@pytest.mark.parametrize(
    ("components", "match"),
    [
        ({3: lambda _: 100 * u.pix}, "No raster pixels"),
        ({0: lambda _: SpectralCoord(500 * u.nm), 2: lambda w: w[2]}, "outside the range"),
        ({1: lambda w: SkyCoord(np.nan * u.arcsec, w[1].Ty, frame=w[1].frame)}, "finite scalars"),
    ],
    ids=["step", "wavelength", "nan"],
)
def test_gwcs_partial_crop_outside_data(partial_raster_cube, components, match):
    cube = partial_raster_cube
    world = cube.wcs.array_index_to_world(*[0] * cube.data.ndim)
    point = [components[i](world) if i in components else None for i in range(len(world))]
    with pytest.raises(ValueError, match=match):
        cube.crop(point)


@requires_crop_hook
def test_gwcs_crop_by_values_needs_both_sky_components(partial_raster_cube):
    values = [None] * partial_raster_cube.wcs.world_n_dim
    values[1] = 0
    with pytest.raises(ValueError, match="Both celestial components"):
        partial_raster_cube.crop_by_values(values, units=partial_raster_cube.wcs.world_axis_units)


@requires_crop_hook
def test_gwcs_partial_sky_crop_skips_exposure_whose_slit_misses(partial_raster_tables):
    shape, pc, crval, dt = partial_raster_tables
    # Step 3's slit is moved along itself, so it no longer reaches slit pixel 1 of step 2.
    crval[..., 3, 1] = 10 * u.arcsec
    cube = _partial_raster_cube(shape, pc, crval, dt)
    prefix = (0,) if cube.data.ndim == 4 else ()
    a, b = (cube.wcs.array_index_to_world(*prefix, step, 1, 0)[1] for step in (2, 3))
    b = SkyCoord(b.Tx, a.Ty, frame=a.frame)
    rest = [None] * (len(cube.wcs.world_axis_object_classes) - 2)
    np.testing.assert_array_equal(cube.crop([None, a, *rest], [None, b, *rest]).data, cube.data[..., 2, 1, :])


@requires_crop_hook
def test_extra_coords_crop_bypasses_raster_crop(sns_sg_file):
    cube = read_spectrograph_lvl2(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"]
    times = cube.axis_world_coords(wcs=cube.extra_coords)[0]
    np.testing.assert_array_equal(cube.crop([times[3]], [times[4]], wcs=cube.extra_coords).data, cube.data[3:5])


@requires_crop_hook
@pytest.mark.parametrize("partial_raster_tables", [True], indirect=True, ids=["combined"])
def test_gwcs_partial_crop_with_scan_selection(partial_raster_cube):
    cube = partial_raster_cube
    world = cube.wcs.array_index_to_world(1, 2, 1, 0)
    point = [value if i in (1, 4) else None for i, value in enumerate(world)]
    np.testing.assert_array_equal(cube.crop(point).data, cube.data[1, 2, 1, :])
    # Time varies with scan, so this crop depends on the integer-indexed scan being kept.
    np.testing.assert_array_equal(cube[1].crop([None, None, world[2], None]).data, cube.data[1, 2])


@requires_crop_hook
@pytest.mark.parametrize("coordinate", ["sky", "time", "step"])
@pytest.mark.parametrize("source", ["raster_sg_files", "sns_sg_file"])
def test_gwcs_partial_crop_real_rasters(request, source, coordinate):
    files = request.getfixturevalue(source)
    cube = read_spectrograph_lvl2(files[:2] if isinstance(files, list) else files, spectral_windows="Si IV 1403")[
        "Si IV 1403"
    ]
    selected = {"sky": 1, "time": 2, "step": 3}[coordinate]
    prefix = (0,) if cube.data.ndim == 4 else ()
    indices = [(*prefix, step, 20, 10) for step in (3, 4)]
    world = [cube.wcs.array_index_to_world(*index) for index in indices]
    points = [[value if i == selected else None for i, value in enumerate(point)] for point in world]
    cropped = cube.crop(*points, keepdims=True)
    for index, point in zip(indices, world, strict=True):
        cropped_index = cropped.wcs.world_to_array_index(*point)
        assert all(0 <= i < size for i, size in zip(cropped_index, cropped.shape, strict=True))
        assert cropped.data[cropped_index] == cube.data[index]
    if coordinate == "time":
        np.testing.assert_array_equal(cropped.data, cube.data[:1, 3:5] if prefix else cube.data[3:5])
    elif coordinate == "step":
        np.testing.assert_array_equal(cropped.data, cube.data[..., 3:5, :, :])
    else:
        assert cropped.shape[-2] < cube.shape[-2]


@requires_crop_hook
def test_gwcs_partial_sky_crop_sit_and_stare_uses_slit_width(tmp_path, sns_sg_file):
    """
    A sit-and-stare exposure covers the slit width across the slit, even when summing
    along the slit makes its pixels (and so its virtual step scale) coarser than that.
    """
    filename = tmp_path / "summed.fits"
    with _edited_copy(sns_sg_file, filename) as hdulist:
        for hdu in hdulist[1:-2]:
            hdu.header["CDELT2"] *= 4
    cube = read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403")["Si IV 1403"]
    _, sky, _, _ = cube.wcs.array_index_to_world(100, 20, 0)
    corners = [
        SkyCoord(sky.Tx + dx * u.arcsec, sky.Ty + dy * u.arcsec, frame=sky.frame)
        for dx, dy in ((-0.01, -0.01), (0.01, 0.01))
    ]

    cropped = cube.crop(*[[None, corner, None, None] for corner in corners], keepdims=True)

    # The pointing drifts 0.23" per exposure, so only exposure 100's 0.33"-wide slit covers the box.
    np.testing.assert_array_equal(cropped.data, cube.data[100:101, 20:21])


@pytest.mark.parametrize("case", ["raster", "combined", "v34_flip", "v34_revert", "sit_and_stare"])
def test_gwcs_slit_centre_matches_aux_pointing_and_fits_wcs(request, case):
    """
    Each exposure's slit centre must be at its AUX FOV centre (XCENIX, YCENIX), and the
    gWCS must agree with the per-exposure FITS-TAB ``fits_wcs`` and invert back to its
    array indices.
    """
    fixture = {"raster": "raster_sg_file", "combined": "raster_sg_files", "sit_and_stare": "sns_sg_file"}
    filename = request.getfixturevalue(fixture.get(case, "v34_raster_file"))
    cube = read_spectrograph_lvl2(filename, spectral_windows="Si IV 1403", revert_v34=case == "v34_revert")[
        "Si IV 1403"
    ]
    if case == "combined":
        cube, filename = cube.raster_slice(0), filename[0]
    assert cube.meta["sit_and_stare"] == (case == "sit_and_stare")
    header, aux, aux_header = _window_header_and_aux(filename, "Si IV 1403")
    if case == "v34_flip":
        aux = aux[::-1]
    n_steps, n_slit = cube.shape[:2]
    slit_centre = header["CRPIX2"] - 1

    steps = np.arange(n_steps)
    _, sky, _, _ = cube.wcs.pixel_to_world(np.zeros(n_steps), np.full(n_steps, slit_centre), steps)
    # Measured residual is ~1e-10" for every case; 0.01" is far below the smallest real error (~0.17").
    assert_quantity_allclose(sky.Tx, aux[:, aux_header["XCENIX"]] * u.arcsec, atol=0.01 * u.arcsec)
    assert_quantity_allclose(sky.Ty, aux[:, aux_header["YCENIX"]] * u.arcsec, atol=0.01 * u.arcsec)

    if case == "sit_and_stare":
        # Along the slit, each exposure's pixels are CDELT2 apart in its own roll direction.
        steps, rows = np.meshgrid([0, n_steps // 2, n_steps - 1], [0, n_slit - 1])
        _, edge, _, _ = cube.wcs.pixel_to_world(np.zeros(steps.shape), rows, steps)
        along_slit = header["CDELT2"] * (rows - slit_centre) * u.arcsec
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
    wcs = read_spectrograph_lvl2(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"].wcs
    steps = np.array([4.0, 4.25, 4.49, 4.51, 5.0])
    _, longitude, latitude, _, _ = wcs.pixel_to_world_values(np.zeros(5), np.zeros(5), steps)
    np.testing.assert_array_equal(longitude, longitude[[0, 0, 0, 4, 4]])
    np.testing.assert_array_equal(latitude, latitude[[0, 0, 0, 4, 4]])
    assert longitude[4] != longitude[0]


@pytest.mark.parametrize("n_steps", [4, 5], ids=["even", "odd"])
def test_sit_and_stare_slit_stays_at_the_exposure_pointing(n_steps):
    """
    With every exposure at the same pointing, the slit centre is there for any step
    pixel, including half pixels and both edges, however dkist rounds a fractional step.
    """
    pc = np.broadcast_to(np.eye(2), (n_steps, 2, 2)) * u.pix
    crval = np.zeros((n_steps, 2)) * u.arcsec
    wcs = _synthetic_raster_gwcs(
        pc, crval, np.arange(n_steps) * u.s, sit_and_stare=True, CDELT2=0.5, CDELT3=0, CRPIX2=1
    )
    steps = np.arange(-0.5, n_steps, 0.5)
    sky = wcs.pixel_to_world(np.zeros(steps.size), np.zeros(steps.size), steps)[1]
    assert_quantity_allclose(sky.Tx, 0 * u.arcsec, atol=1e-9 * u.arcsec)
    assert_quantity_allclose(sky.Ty, 0 * u.arcsec, atol=1e-9 * u.arcsec)


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
    # The FITS WCS frame must carry an obstime, otherwise coordinates built in
    # celestial_frame cannot be transformed into it (they become NaN).
    assert wcs_frame.obstime is not None
    roundtripped = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame).transform_to(wcs_frame)
    assert np.isfinite(roundtripped.Tx.value)
    assert np.isfinite(roundtripped.Ty.value)


def test_read_spectrograph_uncertainty_is_a_standard_deviation(raster_sg_file):
    # A bare array would be stored as an UnknownUncertainty (issue #57).
    cube = read_spectrograph_lvl2(raster_sg_file, spectral_windows="C II 1336", uncertainty=True)["C II 1336"][0]
    assert isinstance(cube.uncertainty, StdDevUncertainty)
    assert cube.uncertainty.array.shape == cube.data.shape


def _synthetic_raster_gwcs(pc, crval, dt, *, sit_and_stare=False, **header):
    header = {"CUNIT1": "nm", "CDELT1": 0.1, "CRVAL1": 140, "CRPIX1": 1, **header}
    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime="2020-01-01")
    return _create_raster_gwcs(header, pc, crval, dt, "2020-01-01", observer, sit_and_stare=sit_and_stare)


def test_raster_gwcs_crpix_is_zero_based():
    pc = np.repeat(np.eye(2)[np.newaxis, :, :], 6, axis=0) * u.pix
    crval = np.repeat([[10.0, 20.0]], 6, axis=0) * u.arcsec
    wcs = _synthetic_raster_gwcs(pc, crval, np.arange(6) * u.s, CDELT2=0.5, CDELT3=2.0, CRPIX2=3.0, CRPIX3=4.0)

    _, sky, _, _ = wcs.array_index_to_world(3, 2, 0)
    assert_quantity_allclose(sky.Tx.to(u.arcsec), 10 * u.arcsec)
    assert_quantity_allclose(sky.Ty.to(u.arcsec), 20 * u.arcsec)


def test_raster_gwcs_far_edge_of_an_even_step_count_is_half_a_step_past_the_last_centre():
    pc = np.repeat(np.eye(2)[np.newaxis, :, :], 6, axis=0) * u.pix
    crval = np.repeat([[10.0, 20.0]], 6, axis=0) * u.arcsec
    wcs = _synthetic_raster_gwcs(pc, crval, np.arange(6) * u.s, CDELT2=0.5, CDELT3=2.0, CRPIX2=3.0, CRPIX3=4.0)

    lon = [wcs.pixel_to_world_values(0, 2, step)[1] for step in (5, 5.25, 5.5)]
    np.testing.assert_allclose(lon[2] - lon[0], 2 * (lon[1] - lon[0]))


def test_sit_and_stare_zero_pc_rows_are_interpolated_unless_every_row_is_bad():
    pc = np.array([np.eye(2), np.zeros((2, 2)), 3 * np.eye(2)]) * u.pix
    crval = np.repeat([[10.0, 20.0]], 3, axis=0) * u.arcsec

    assert not _raster_wcs_bad_row_mask(pc, crval).any()
    bad_rows = _raster_wcs_bad_row_mask(pc, crval, pc_only=True)
    np.testing.assert_array_equal(bad_rows, [False, True, False])
    with pytest.warns(UserWarning, match="all-zero or non-finite WCS tables"):
        pc_out, crval_out = _sanitize_raster_wcs_tables(pc.copy(), crval.copy(), pc_only=True)
    np.testing.assert_allclose(pc_out[1].to_value(u.pix), [[2.0, 0.0], [0.0, 2.0]])
    assert_quantity_allclose(crval_out[1], [10.0, 20.0] * u.arcsec)

    # When every row is all-zero there is nothing to interpolate from, so reading fails.
    with pytest.raises(ValueError, match="Every exposure has all-zero or non-finite"):
        _sanitize_raster_wcs_tables(np.zeros((2, 2, 2)) * u.pix, np.zeros((2, 2)) * u.arcsec)


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [([np.nan, 1, 2, np.nan, 4, np.nan], [0, 1, 2, 3, 4, 5]), ([np.nan, 5, np.nan], [5, 5, 5])],
    ids=["interpolate_and_extrapolate", "one_good_row"],
)
def test_raster_times_are_interpolated_between_and_beyond_good_rows(offsets, expected):
    start = Time("2020-01-01T23:59:58")  # The times cross midnight.
    with pytest.warns(UserWarning, match="non-finite times"):
        times = _sanitize_raster_times(start, offsets * u.s)
    assert_quantity_allclose((times - start).to(u.s), expected * u.s, atol=1e-9 * u.s)


def test_gwcs_inverse_uses_explicit_step_when_time_is_not_monotonic():
    # Two passes over the same 8 slit positions, the second back in time.
    steps = np.arange(8.0)
    pc = np.repeat(np.eye(2)[np.newaxis, :, :], 16, axis=0) * u.pix
    crval = np.zeros((16, 2)) * u.arcsec
    crval[:, 0] = np.tile(steps, 2) * u.arcsec
    wcs = _synthetic_raster_gwcs(pc, crval, np.concatenate([steps, steps[::-1]]) * u.s, CDELT2=1, CDELT3=1, CRPIX2=1)

    for scan_index in (1, 9):
        point = wcs.array_index_to_world(scan_index, 20, 10)
        assert wcs.world_to_array_index(*point) == (scan_index, 20, 10)
