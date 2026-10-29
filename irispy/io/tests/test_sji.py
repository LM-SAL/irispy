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
from sunpy.map.sources import AIAMap, SOTMap

from irispy.data.test import get_test_filepath
from irispy.io.sji import _create_headers_wcs, _fill_dropped_pointing_rows, _t_obs, read_sji_lvl2
from irispy.sji import AIACube, SOTCube
from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED, DN_UNIT, SATURATION_LIMIT

SJI_FILES = [
    "sns_sji_1330_file",
    "sns_sji_1400_file",
    "sns_sji_2796_file",
    "sns_sji_2832_file",
    "raster_sji_1400_file",
    "raster_sji_2796_file",
    "raster_sji_2832_file",
]


@pytest.mark.parametrize("file_fixture", SJI_FILES)
def test_read_sji_lvl2_matches_the_file(request, file_fixture):
    filename = request.getfixturevalue(file_fixture)
    cube = read_sji_lvl2(filename)
    with fits.open(filename) as hdulist:
        header, data = hdulist[0].header, hdulist[0].data
        aux, columns = hdulist[1].data, hdulist[1].header

    assert str(cube)
    assert cube.shape == (header["NAXIS3"], header["NAXIS2"], header["NAXIS1"])
    assert tuple(cube.meta.data_shape) == cube.shape
    assert cube.unit == DN_UNIT[header["TDESC1"]]
    assert cube.mask.sum() == (np.isin(data, BAD_PIXEL_VALUES_SCALED) | (data >= SATURATION_LIMIT.value)).sum()
    # The times are the exposure midpoints
    times = cube.axis_world_coords("time")[0]
    expected_times = Time(header["STARTOBS"]) + (aux[:, columns["TIME"]] + aux[:, columns["EXPTIMES"]] / 2) * u.s
    assert_quantity_allclose((times[[0, -1]] - expected_times[[0, -1]]).to(u.s), 0 * u.s, atol=1 * u.us)
    assert [wcs.wcs.dateobs for wcs in cube.fits_wcs] == times.isot.tolist()
    # The reference pixel of the first frame points at that frame's XCENIX and YCENIX
    reference = cube.wcs.pixel_to_world(header["CRPIX1"] - 1, header["CRPIX2"] - 1, 0)[0]
    assert_quantity_allclose(reference.Tx, aux[0, columns["XCENIX"]] * u.arcsec, atol=1e-6 * u.arcsec)
    assert_quantity_allclose(reference.Ty, aux[0, columns["YCENIX"]] * u.arcsec, atol=1e-6 * u.arcsec)

    meta = cube.meta
    assert len(meta.fits_header) == len(header)
    assert meta.fov_center == SkyCoord(
        Tx=header["XCEN"],
        Ty=header["YCEN"],
        unit=u.arcsec,
        frame=Helioprojective,
        observer="earth",
        obstime=header["DATE_OBS"],
    )
    assert_quantity_allclose(meta.spectral_range, [header["TWMIN1"], header["TWMAX1"]] * u.AA)
    values = {
        "observatory": header["TELESCOP"],
        "instrument": header["INSTRUME"],
        "detector": header["TDET1"],
        "spectral_window": header["TDESC1"].removeprefix("SJI_"),
        "spectral_band": "FUV" if header["TWAVE1"] < 2000 else "NUV",
        "observing_mode_id": int(header["OBSID"]),
        "observing_mode_description": header["OBS_DESC"],
        "processing_level": header["DATA_LEV"],
        "number_of_unique_raster_positions": header["NRASTERP"],
        "number_of_raster_positions": header["RASNRPT"],
        "exposure_control_triggers_in_observation": header["AECNOBS"],
        "exposure_control_triggers_in_raster": header["AECNRAS"],
        "spatial_summing_factor": header["SUMSPAT"],
        "spectral_summing_factor": None,
        "key_comments": {},
    }
    for name, expected in values.items():
        assert getattr(meta, name) == expected, name
    flags = {
        "automatic_exposure_control_enabled": "IAECFLAG",
        "observation_includes_saa": "SAA",
        "observatory_at_high_latitude": "HLZ",
        "tracking_mode_enabled": "TR_MODE",
    }
    for name, keyword in flags.items():
        assert getattr(meta, name) is bool(header[keyword]), name
    quantities = {
        "distance_to_sun": ("DSUN_OBS", u.m),
        "raster_fov_width_x": ("FOVX", u.arcsec),
        "raster_fov_width_y": ("FOVY", u.arcsec),
        "satellite_rotation": ("SAT_ROT", u.deg),
        "exposure_time": ("EXPTIME", u.s),
    }
    for name, (keyword, unit) in quantities.items():
        assert_quantity_allclose(getattr(meta, name), header[keyword] * unit, err_msg=name)
    dates = {
        "date_reference": "DATE_OBS",
        "date_start": "DATE_OBS",
        "date_end": "DATE_END",
        "observing_campaign_start": "STARTOBS",
        "observing_campaign_end": "ENDOBS",
    }
    for name, keyword in dates.items():
        assert getattr(meta, name).isot == Time(header[keyword]).isot, name


def test_read_sji_lvl2_masks_explicit_float_bad_pixels(tmp_path, sns_sji_1330_file):
    with fits.open(sns_sji_1330_file) as hdul:
        data = hdul[0].data.astype("float32")
        data.flat[:3] = [-200, -199, -198]
        expected_mask = np.isin(data, [-200, -199])
        hdul[0].data = data
        float_file = tmp_path / "sji_float_bad_pixels.fits"
        hdul.writeto(float_file)

    cube = read_sji_lvl2(float_file)

    np.testing.assert_array_equal(cube.mask, expected_mask)
    assert np.isnan(cube.data[cube.mask]).all()
    assert cube.data.flat[2] == -198


def test_read_sji_lvl2_reads_an_unquoted_nan():
    # CADPL_DV = NAN, as files made before June 2018 write a NaN (see unparsable_card/overview.txt)
    filename = get_test_filepath("unparsable_card/iris_l2_20210905_001833_3620258102_SJI_1400_t000_nan_test.fits")
    assert read_sji_lvl2(filename).meta["CADPL_DV"] == "NAN"


def test_read_sji_lvl2_keeps_the_image_of_a_supplied_hdulist(bursts_sji_1400_file):
    with fits.open(bursts_sji_1400_file) as hdulist:
        original = hdulist[0].data.copy()
        cube = read_sji_lvl2(hdulist)
        np.testing.assert_array_equal(hdulist[0].data, original)
    # The fill values were replaced in the returned cube only
    assert np.isnan(cube.data).any()
    assert not np.isnan(original).any()


def test_read_sji_lvl2_unscaled_opened_file_needs_memmap(bursts_sji_1400_file):
    with (
        fits.open(bursts_sji_1400_file, do_not_scale_image_data=True) as hdulist,
        pytest.raises(ValueError, match="needs memmap=True"),
    ):
        read_sji_lvl2(hdulist)


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

    assert cube.meta["scaled"] is not memmap
    if memmap:
        np.testing.assert_array_equal(cube.mask, expected_mask)
        np.testing.assert_array_equal(cube.data, raw)
        assert cube.uncertainty is None
    else:
        expected = raw.astype(float) * scale + offset
        expected[expected_mask] = np.nan
        clipped = expected >= SATURATION_LIMIT.value  # the clipped pixels of this flare, which keep their value
        assert clipped.any()
        np.testing.assert_array_equal(cube.mask, expected_mask | clipped)
        np.testing.assert_array_equal(cube.meta["saturated"], clipped)
        np.testing.assert_allclose(cube.data, expected)
        assert isinstance(cube.uncertainty, StdDevUncertainty)
        assert cube.uncertainty.array.shape == cube.data.shape


@pytest.mark.parametrize(("file_fixture", "photons_per_dn"), [("sns_sji_1400_file", 12), ("sns_sji_2796_file", 18)])
def test_read_sji_lvl2_uncertainty_uses_the_photons_per_dn_of_the_passband(request, file_fixture, photons_per_dn):
    # Wülser et al. (2018) Table 1: 12 photons per DN in the FUV slit-jaw passbands, 18 in the NUV ones
    cube = read_sji_lvl2(request.getfixturevalue(file_fixture), uncertainty=True)
    assert cube.unit.to(u.photon) == photons_per_dn
    brightest = np.unravel_index(np.nanargmax(cube.data), cube.shape)
    # Photon noise and 1.2 DN of read noise
    photons = cube.data[brightest] * photons_per_dn
    expected = np.sqrt(photons + (1.2 * photons_per_dn) ** 2) / photons_per_dn
    assert cube.uncertainty.array[brightest] == pytest.approx(expected, rel=1e-6)


def test_read_sji_lvl2_other_passband_stays_in_dn(tmp_path, sns_sji_1400_file):
    # SJI_5000W and SJI_1600W have no published photons per DN
    filename = tmp_path / "sji_5000w.fits"
    with fits.open(sns_sji_1400_file, memmap=False) as hdulist:
        hdulist[0].header["TDESC1"] = "SJI_5000W"
        hdulist.writeto(filename)
    assert read_sji_lvl2(filename).unit == u.DN
    with pytest.raises(ValueError, match="SJI_5000W"):
        read_sji_lvl2(filename, uncertainty=True)


@pytest.mark.parametrize("input_kind", ["file-like", "hdulist"])
@pytest.mark.parametrize("memmap", [False, True])
def test_read_sji_lvl2_accepts_decompressed_data(input_kind, memmap):
    filename = get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits")
    expected = read_sji_lvl2(filename, memmap=memmap)
    if input_kind == "hdulist":
        with fits.open(filename, mode="denywrite", memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            original = hdulist[0].data
            cube = read_sji_lvl2(hdulist, memmap=memmap)
            assert not hdulist.fileinfo(0)["file"].closed
            if memmap:
                assert np.shares_memory(cube.data, original)
                assert not cube.data.flags["W"]
    else:
        cube = read_sji_lvl2(io.BytesIO(Path(filename).read_bytes()), memmap=memmap)

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


def test_read_sji_lvl2_fills_dropped_pointing_rows(tmp_path, sns_sji_1330_file):
    # A dropped exposure zeroes its whole pointing row; those rows are filled
    # with the average of the neighboring exposures.
    keys = ("XCENIX", "YCENIX", "PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX")
    filename = tmp_path / "dropped_row.fits"
    with fits.open(sns_sji_1330_file, memmap=False) as hdulist:
        columns = [hdulist[1].header[key] for key in keys]
        expected = {key: hdulist[1].data[[0, 2], column].mean() for key, column in zip(keys, columns, strict=True)}
        hdulist[1].data[1, columns] = 0.0
        hdulist.writeto(filename)

    cube = read_sji_lvl2(filename)
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
        expected = fits_wcs.pixel_to_world(x, y)
        sky = cube.wcs.pixel_to_world(x, y, np.full_like(x, frame))[0]
        assert_quantity_allclose(sky.Tx, expected.Tx, atol=1e-6 * u.arcsec)
        assert_quantity_allclose(sky.Ty, expected.Ty, atol=1e-6 * u.arcsec)


def test_sji_extra_coordinate_units(sns_sji_1400_file):
    extra_coords = read_sji_lvl2(sns_sji_1400_file).extra_coords.wcs
    units = dict(zip(extra_coords.world_axis_names, extra_coords.world_axis_units, strict=True))
    assert units["slit x position"] == units["slit y position"] == "pixel"
    assert units["ophaseix"] == ""


def test_sji_slit_positions_are_zero_based_pixels(sns_sji_1400_file):
    # The file stores the slit at 1-based FITS pixels; the extra coordinates follow the cube's 0-based pixels
    cube = read_sji_lvl2(sns_sji_1400_file)
    with fits.open(sns_sji_1400_file) as hdulist:
        aux, columns = hdulist[1].data, hdulist[1].header
    for name, column in (("slit x position", "SLTPX1IX"), ("slit y position", "SLTPX2IX")):
        position = cube.axis_world_coords(wcs=cube.extra_coords[name])[0]
        assert_quantity_allclose(position, (aux[:, columns[column]] - 1) * u.pix)


@pytest.mark.parametrize("memmap", [False, True])
def test_read_aia_cube(aia_1700_file, memmap):
    # Three frames of the AIA 1700 Å cube the IRIS team co-aligned with OBS 3620106076
    with fits.open(aia_1700_file) as hdulist:
        data = hdulist[0].data
        aux, columns = hdulist[1].data, hdulist[1].header
        expected_times = Time(hdulist[0].header["STARTOBS"]) + (
            (aux[:, columns["TIME"]] + aux[:, columns["EXPTIMES"]] / 2) * u.s
        )

    # The file stores DN as integers without scaling, so memmap=True maps the same values
    aia = read_sji_lvl2(aia_1700_file, memmap=memmap)

    assert isinstance(aia, AIACube)
    assert "AIACube" in str(aia)
    assert aia.shape == (3, 382, 320)
    assert aia.unit == u.DN
    assert aia.meta["scaled"]
    assert aia.meta.instrument == "AIA_3"
    assert aia.meta.observing_mode_id == 3620106076
    # The cube keeps the file's integers: no fill values, so nothing to replace or mask
    np.testing.assert_array_equal(aia.data, data)
    assert not aia.mask.any()
    times = aia.axis_world_coords("time")[0]
    assert times.isot.tolist() == expected_times.isot.tolist()
    assert_quantity_allclose((times[1:] - times[:-1]).to(u.s), 24 * u.s, atol=1e-3 * u.s)
    # The frame headers carry the file's instrument, so sunpy picks its AIA map class
    aia_map = aia.to_maps(1)
    assert isinstance(aia_map, AIAMap)
    assert aia_map.date.isot == expected_times[1].isot
    assert aia_map.wavelength == 1700 * u.angstrom
    assert aia_map.exposure_time == aux[1, columns["EXPTIMES"]] * u.s
    assert_quantity_allclose(aia_map.reference_coordinate.Tx, aux[1, columns["XCENIX"]] * u.arcsec)
    assert_quantity_allclose(aia_map.reference_coordinate.Ty, aux[1, columns["YCENIX"]] * u.arcsec)
    np.testing.assert_array_equal(aia_map.data, data[1])
    assert len(aia.to_maps()) == 3
    # Slicing keeps the frame times and the FITS WCS of the frames
    assert aia[1:].shape == (2, 382, 320)
    assert aia[1:].axis_world_coords("time")[0].isot.tolist() == expected_times[1:].isot.tolist()
    sub = aia[:, 10:, 20:]
    assert isinstance(sub.fits_wcs, list)
    assert sub.fits_wcs[0].wcs.crpix.tolist() == [141.0, 182.0]
    assert isinstance(aia[0].to_maps(), AIAMap)


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
                reference_pixel=[hdulist[0].header["CRPIX1"] - 1, hdulist[0].header["CRPIX2"] - 1] * u.pixel,
                scale=[hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]] * u.arcsec / u.pixel,
                rotation_matrix=np.asanyarray([[pc[0], pc[1]], [pc[2], pc[3]]]),
                instrument=hdulist[0].header["INSTRUME"],
                telescope=hdulist[0].header["TELESCOP"],
                wavelength=int(hdulist[0].header["TWAVE1"]) * u.AA,
                exposure=aux.data[i, aux.header["EXPTIMES"]] * u.second,
            )
            expected["twave1"] = hdulist[0].header["TWAVE1"]
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
    # One FITS WCS per frame
    assert [wcs.wcs.ctype[0] for wcs in cube.fits_wcs] == [header["CTYPE1"]] * cube.shape[0]
    assert cube.unit == unit
    assert cube.meta["scaled"]
    # The floats are unscaled and NaN marks missing pixels, with or without memmap
    np.testing.assert_array_equal(cube.data, raw)
    np.testing.assert_array_equal(cube.mask, np.isnan(raw))
    assert np.isnan(raw).any() == (kind != "MG")
    assert cube.meta.date_reference.isot == date
    # DSUN_OBS and TWAVE1 are 0, so the solar distance and wavelength are missing
    assert cube.meta.distance_to_sun is None
    assert cube.meta.rest_wavelength is None
    # mu needs only the pointing and the date
    assert 0 < cube.meta.mu < 1
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
    # sunpy's SOTMap knows the narrowband files ("SOT/NB") and spells the observatory "Hinode"
    assert isinstance(sot_map, SOTMap) == (cube.meta["INSTRUME"] in SOTMap.Instruments)
    assert sot_map.unit == cube.unit
    assert sot_map.instrument == cube.meta["INSTRUME"]
    assert sot_map.observatory.upper() == cube.meta["TELESCOP"]
    assert sot_map.exposure_time == cube.axis_world_coords(wcs=cube.extra_coords["exposure time"])[0][index]
    assert sot_map.meta["TWAVE1"] == cube.meta["TWAVE1"]
    assert sot_map.date.utc.isot == cube.wcs.pixel_to_world(0, 0, index)[-1].utc.isot
    # The SOT cubes have no wavelength (TWAVE1 is 0), nor an IRIS colormap to plot with
    assert sot_map.wavelength is None
    sot_map.plot()


def test_length_one_time_slice_prints_and_maps(sns_sjicube_1330):
    # A one-frame cube only arises from slicing; no level 2 file holds a single exposure
    cube = sns_sjicube_1330[0:1]
    assert cube.data.shape[0] == 1
    assert "Obs ID" in str(cube)
    assert len(cube.to_maps()) == 1
