import io
from pathlib import Path

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.wcs import WCS

from sunpy.coordinates import Helioprojective, get_body_heliographic_stonyhurst
from sunpy.map.header_helper import make_fitswcs_header

from irispy.data.test import get_test_filepath
from irispy.io.sji import _create_headers_wcs, _fill_dropped_pointing_rows, _t_obs, read_sji_lvl2
from irispy.sji import AIACube, SOTCube


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
    assert len(meta.fits_header) == 162 == (len(meta.keys()) + 12)  # History is missing
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
    assert len(meta.fits_header) == 162 == (len(meta.keys()) + 12)  # History is missing
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
    assert [wcs.wcs.dateobs for wcs in sji_1400_cube.fits_wcs] == observation_times.isot.tolist()


@pytest.mark.parametrize("from_bytes", [False, True])
def test_read_sji_lvl2_masks_explicit_float_bad_pixels(tmp_path, sns_sji_1330_file, from_bytes):
    with fits.open(sns_sji_1330_file) as hdul:
        data = hdul[0].data.astype("float32")
        data.flat[:3] = [-200, -199, -198]
        expected_mask = np.isin(data, [-200, -199])
        hdul[0].data = data
        float_file = tmp_path / "sji_float_bad_pixels.fits"
        hdul.writeto(float_file)

    cube = read_sji_lvl2(float_file.read_bytes() if from_bytes else float_file)

    np.testing.assert_array_equal(cube.mask, expected_mask)
    assert np.isnan(cube.data[cube.mask]).all()
    assert cube.data.flat[2] == -198


@pytest.mark.parametrize("memmap", [False, True])
def test_read_sji_lvl2_masks_both_fill_values_without_changing_raw_data(memmap):
    filename = get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits")
    with fits.open(filename, memmap=False, do_not_scale_image_data=True) as hdulist:
        raw = hdulist[0].data.copy()
        scale, offset = hdulist[0].header["BSCALE"], hdulist[0].header["BZERO"]
    expected_mask = np.isin(raw, [-32768, -32764])
    assert np.any(raw == -32768)
    assert np.any(raw == -32764)

    cube = read_sji_lvl2(filename, memmap=memmap, uncertainty=True)

    np.testing.assert_array_equal(cube.mask, expected_mask)
    assert cube.meta["scaled"] is not memmap
    if memmap:
        np.testing.assert_array_equal(cube.data, raw)
        assert cube.uncertainty is None
    else:
        expected = raw.astype(float) * scale + offset
        expected[expected_mask] = np.nan
        np.testing.assert_allclose(cube.data, expected)
        assert isinstance(cube.uncertainty, StdDevUncertainty)
        assert cube.uncertainty.array.shape == cube.data.shape


@pytest.mark.parametrize("input_kind", ["bytes", "file-like", "hdulist"])
@pytest.mark.parametrize("memmap", [False, True])
def test_read_sji_lvl2_accepts_decompressed_data(input_kind, memmap):
    filename = get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits")
    expected = read_sji_lvl2(filename, memmap=memmap)
    content = Path(filename).read_bytes()
    if input_kind == "hdulist":
        with fits.open(filename, mode="denywrite", memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            original = hdulist[0].data
            cube = read_sji_lvl2(hdulist, memmap=memmap)
            assert not hdulist.fileinfo(0)["file"].closed
            if memmap:
                assert np.shares_memory(cube.data, original)
                assert not cube.data.flags["W"]
    else:
        cube = read_sji_lvl2(content if input_kind == "bytes" else io.BytesIO(content), memmap=memmap)

    np.testing.assert_array_equal(cube.data, expected.data)
    np.testing.assert_array_equal(cube.mask, expected.mask)
    assert cube.unit == expected.unit
    assert cube.meta["scaled"] == expected.meta["scaled"]
    np.testing.assert_allclose(cube.wcs.pixel_to_world_values(2, 3, 0), expected.wcs.pixel_to_world_values(2, 3, 0))


def test_read_sji_lvl2_is_exported():
    import irispy.io as iris_io  # NOQA: PLC0415

    assert iris_io.read_sji_lvl2 is read_sji_lvl2
    assert "read_sji_lvl2" in iris_io.__all__


@pytest.mark.parametrize(
    "file_fixture",
    [
        "sns_sji_1330_file",
        "sns_sji_1400_file",
        "sns_sji_2796_file",
        "sns_sji_2832_file",
        "raster_sji_1400_file",
        "raster_sji_2796_file",
        "raster_sji_2832_file",
    ],
)
def test_smoke_read_sji_lvl2(request, file_fixture):
    assert read_sji_lvl2(request.getfixturevalue(file_fixture)).data.ndim == 3


def test_read_sji_lvl2_unrotated_pointing(tmp_path, sns_sji_1330_file):
    filename = tmp_path / "unrotated.fits"
    with fits.open(sns_sji_1330_file, memmap=False) as hdulist:
        for key in ("PC1_2IX", "PC2_1IX"):
            hdulist[1].data[:, hdulist[1].header[key]] = 0.0
        hdulist.writeto(filename)

    cube = read_sji_lvl2(filename)
    for key in ("PC1_2IX", "PC2_1IX"):
        for header in cube.meta["frame_wcs_headers"]:
            assert header[f"PC{key[2]}_{key[4]}"] == 0.0


@pytest.mark.parametrize("from_bytes", [False, True])
def test_read_sji_lvl2_fills_dropped_pointing_rows(tmp_path, sns_sji_1330_file, from_bytes):
    # A dropped exposure zeroes its whole pointing row; those rows are filled
    # with the average of the neighboring exposures.
    keys = ("XCENIX", "YCENIX", "PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX")
    filename = tmp_path / "dropped_row.fits"
    with fits.open(sns_sji_1330_file, memmap=False) as hdulist:
        columns = [hdulist[1].header[key] for key in keys]
        expected = {key: hdulist[1].data[[0, 2], column].mean() for key, column in zip(keys, columns, strict=True)}
        hdulist[1].data[1, columns] = 0.0
        hdulist.writeto(filename)

    cube = read_sji_lvl2(filename.read_bytes() if from_bytes else filename)
    header = cube.meta["frame_wcs_headers"][1]
    np.testing.assert_allclose(header["CRVAL1"], expected["XCENIX"])
    np.testing.assert_allclose(header["CRVAL2"], expected["YCENIX"])


def test_sji_gwcs_matches_the_fits_pointing(sns_sji_1400_file):
    # CRVAL is at the 1-based FITS CRPIX, i.e. the 0-based pixel CRPIX - 1
    cube = read_sji_lvl2(sns_sji_1400_file)
    with fits.open(sns_sji_1400_file) as hdulist:
        header, aux, columns = hdulist[0].header, hdulist[1].data, hdulist[1].header
    n_frames, ny, nx = cube.data.shape
    x = np.array([0, nx - 1, 0, nx - 1, (nx - 1) / 2])
    y = np.array([0, 0, ny - 1, ny - 1, (ny - 1) / 2])
    for frame in (0, n_frames - 1):
        fits_wcs = WCS(naxis=2)
        fits_wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
        fits_wcs.wcs.cunit = ["arcsec", "arcsec"]
        fits_wcs.wcs.crpix = [header["CRPIX1"], header["CRPIX2"]]
        fits_wcs.wcs.cdelt = [header["CDELT1"], header["CDELT2"]]
        fits_wcs.wcs.crval = [aux[frame, columns["XCENIX"]], aux[frame, columns["YCENIX"]]]
        fits_wcs.wcs.pc = [
            [aux[frame, columns["PC1_1IX"]], aux[frame, columns["PC1_2IX"]]],
            [aux[frame, columns["PC2_1IX"]], aux[frame, columns["PC2_2IX"]]],
        ]
        longitude, latitude = (value * 3600 for value in fits_wcs.pixel_to_world_values(x, y))
        world = cube.wcs.pixel_to_world_values(x, y, np.full_like(x, frame))
        # the gWCS wraps longitude to [0, 360) degrees and astropy to (-180, 180]
        np.testing.assert_allclose((world[0] - longitude + 648000) % 1296000 - 648000, 0, atol=1e-6)
        np.testing.assert_allclose(world[1], latitude, atol=1e-6)


def test_sji_extra_coordinate_units(sns_sji_1400_file):
    extra_coords = read_sji_lvl2(sns_sji_1400_file).extra_coords.wcs
    units = dict(zip(extra_coords.world_axis_names, extra_coords.world_axis_units, strict=True))
    assert units["slit x position"] == units["slit y position"] == "pixel"
    assert units["ophaseix"] == ""


def test_read_aia_cube(tmp_path, sns_sji_1330_file):
    filename = tmp_path / "aia.fits"
    with fits.open(sns_sji_1330_file) as hdulist:
        hdulist[0].header["INSTRUME"] = "AIA"
        hdulist.writeto(filename)

    aia = read_sji_lvl2(filename)

    assert isinstance(aia, AIACube)
    assert aia[-3:].shape == (3, 40, 37)
    assert isinstance(aia[:, 10:, 20:].fits_wcs, list)


def test_sji_first_and_last_frames_round_trip(sns_sjicube_1400):
    cube = sns_sjicube_1400
    assert np.isfinite(cube.axis_world_coords("time", pixel_corners=True)[0].jd).all()
    for index in ((0, 5, 3), (cube.shape[0] - 1, 39, 36)):
        assert cube.wcs.world_to_array_index(*cube.wcs.array_index_to_world(*index)) == index


def test_frame_wcs_headers_match_per_frame_make_fitswcs_header(sns_sji_1400_file):
    # The headers are built from one template; each frame must equal a header made for that frame alone.
    with fits.open(sns_sji_1400_file) as hdulist:
        hdulist.verify("silentfix")
        t_obs = _t_obs(hdulist)
        _fill_dropped_pointing_rows(hdulist)
        headers = _create_headers_wcs(hdulist, t_obs)
        aux = hdulist[1]
        for i in (1, len(headers) - 1):
            pointing = Helioprojective(
                aux.data[i, aux.header["XCENIX"]] * u.arcsec,
                aux.data[i, aux.header["YCENIX"]] * u.arcsec,
                observer=get_body_heliographic_stonyhurst("Earth", t_obs[i].isot),
                obstime=t_obs[i],
            )
            pc = [aux.data[i, aux.header[key]] for key in ("PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX")]
            expected = make_fitswcs_header(
                data=hdulist[0].data[i].shape,
                coordinate=pointing,
                scale=[hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]] * u.arcsec / u.pixel,
                rotation_matrix=np.asanyarray([[pc[0], pc[1]], [pc[2], pc[3]]]),
                instrument="SJI",
                telescope="IRIS",
                observatory="IRIS",
                wavelength=int(hdulist[0].header["TWAVE1"]) * u.AA,
                exposure=aux.data[i, aux.header["EXPTIMES"]] * u.second,
                unit=u.DN,
            )
            assert dict(headers[i]) == dict(expected)


@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize(
    ("kind", "unit", "date"),
    [
        ("FG", u.DN, "2015-08-30T10:04:31.477"),
        ("MG", u.dimensionless_unscaled, "2016-01-08T19:04:35.322"),
        # The SOT-SP maps give SolarSoft dates, "8-Jan-2016 19:13:48.658"
        ("SP", u.G, "2016-01-08T19:13:48.658"),
    ],
    ids=["FG", "MG", "SP"],
)
def test_read_sot_cube(kind, unit, date, memmap, request):
    filename = request.getfixturevalue(f"sot_{kind.lower()}_file")
    cube = read_sji_lvl2(filename, memmap=memmap)
    with fits.open(filename) as hdulist:
        header, raw = hdulist[0].header, hdulist[0].data
        aux, columns = np.atleast_2d(hdulist[1].data), hdulist[1].header

    assert isinstance(cube, SOTCube)
    assert "SOTCube" in str(cube)
    assert cube.unit == unit
    assert cube.meta["scaled"]
    # The floats are unscaled and NaN marks missing pixels, with or without memmap
    np.testing.assert_array_equal(cube.data, raw)
    np.testing.assert_array_equal(cube.mask, np.isnan(raw))
    assert np.isnan(raw).any() == (kind != "MG")
    assert cube.meta.date_reference.isot == date
    # DSUN_OBS and TWAVE1 are 0, so the solar distance and wavelength are missing
    assert cube.meta.distance_to_sun is None
    assert cube.meta.mu is None
    assert cube.meta.rest_wavelength is None
    times = Time(header["STARTOBS"]) + (aux[:, columns["TIME"]] + aux[:, columns["EXPTIMES"]] / 2) * u.s
    assert_quantity_allclose((cube.axis_world_coords("time")[0] - times).to(u.s), 0 * u.s, atol=1 * u.us)
    # CRVAL is the per-frame XCENIX/YCENIX at the 1-based FITS CRPIX
    frames = np.arange(len(aux))
    pointing = cube.wcs.pixel_to_world(header["CRPIX1"] - 1, header["CRPIX2"] - 1, frames)[0]
    assert_quantity_allclose(pointing.Tx, aux[:, columns["XCENIX"]] * u.arcsec, atol=1e-6 * u.arcsec)
    assert_quantity_allclose(pointing.Ty, aux[:, columns["YCENIX"]] * u.arcsec, atol=1e-6 * u.arcsec)
    frame = int(header["CRPIX3"]) - 1
    assert_quantity_allclose(pointing[frame].Tx, header["CRVAL1"] * u.arcsec, atol=1e-6 * u.arcsec)
    assert_quantity_allclose(pointing[frame].Ty, header["CRVAL2"] * u.arcsec, atol=1e-6 * u.arcsec)
    step = cube.wcs.pixel_to_world(header["CRPIX1"], header["CRPIX2"] - 1, frame)[0]
    assert_quantity_allclose(
        np.hypot(step.Tx - pointing[frame].Tx, step.Ty - pointing[frame].Ty), header["CDELT1"] * u.arcsec, rtol=1e-4
    )


@pytest.mark.parametrize("memmap", [False, True])
def test_read_sot_magnetic_values_are_not_fill_values(sot_sp_file, memmap):
    with fits.open(sot_sp_file) as hdulist:
        hdulist[0].data.flat[:3] = [-200, -199, np.nan]
        expected = hdulist[0].data.copy()
        cube = read_sji_lvl2(hdulist, memmap=memmap)

    np.testing.assert_array_equal(cube.data, expected)
    np.testing.assert_array_equal(cube.mask, np.isnan(expected))


@pytest.mark.parametrize("kind", ["FG", "MG", "SP"])
@pytest.mark.parametrize("conversion", ["frame", "sequence", "2d"])
def test_sot_cube_to_maps(kind, conversion, request):
    cube = read_sji_lvl2(request.getfixturevalue(f"sot_{kind.lower()}_file"))
    index = cube.shape[0] - 1
    if conversion == "2d":
        sot_map = cube[index].to_maps()
    elif conversion == "sequence":
        sot_map = cube.to_maps([index])[0]
    else:
        sot_map = cube.to_maps(index)
    assert sot_map.unit == cube.unit
    assert sot_map.instrument == cube.meta["INSTRUME"]
    assert sot_map.observatory == cube.meta["TELESCOP"]
    assert sot_map.exposure_time == cube.meta["EXPTIME"] * u.s
    assert sot_map.meta["TWAVE1"] == cube.meta["TWAVE1"]
    assert sot_map.date.utc.isot == cube.wcs.pixel_to_world(0, 0, index)[-1].utc.isot
    # The SOT cubes have no wavelength (TWAVE1 is 0), nor an IRIS colormap to plot with
    assert sot_map.wavelength is None
    sot_map.plot()
