import copy
from types import SimpleNamespace

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS

from ndcube.utils.exceptions import NDCubeUserWarning

import irispy.io._raster_combine as raster_combine
from irispy.io._raster_combine import _lazy_raster_scan_chunk_rows
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.utils import read_files
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import BAD_PIXEL_VALUE_UNSCALED, SLIT_WIDTH


def test_fits_data_comparison(sns_sg_file):
    """
    Make sure the data is the same in pyfits and irispy.
    """
    iris_l2_test_raster = read_spectrograph_lvl2(sns_sg_file)
    with fits.open(sns_sg_file) as hdulist:
        spectral_window1 = hdulist[0].header["TDESC1"]
        spectral_window2 = hdulist[0].header["TDESC2"]
        spectral_window3 = hdulist[0].header["TDESC3"]
        data1 = copy.deepcopy(hdulist[1].data)
        data2 = copy.deepcopy(hdulist[2].data)
        data3 = copy.deepcopy(hdulist[3].data)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window1].data, data1)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window2].data, data2)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window3].data, data3)


def test_spectrogram_cube_remove_cosmic_rays(sns_sg_file, monkeypatch):
    captured = {}

    def fake_remove_cosmic_rays(cube, *, method, sigma, max_iters, method_kwargs):
        captured["method"] = method
        captured["mask"] = cube.mask.copy()
        captured["sigma"] = sigma
        captured["max_iters"] = max_iters
        captured["method_kwargs"] = method_kwargs
        return cube.to_nddata(
            data=cube.data + 2,
            mask="copy",
            nddata_type=type(cube),
            extra_coords="copy",
            global_coords="copy",
        )

    monkeypatch.setattr("irispy.spectrograph.remove_cosmic_rays", fake_remove_cosmic_rays)

    raster = read_spectrograph_lvl2(sns_sg_file)
    key = next(iter(raster.keys()))
    cube = raster[key]
    cleaned_cube = cube.remove_cosmic_rays(
        method="astroscrappy",
        sigma=5.0,
        max_iters=5,
        method_kwargs={"batch_size": 16},
    )

    np.testing.assert_array_equal(cleaned_cube.data, cube.data + 2)
    np.testing.assert_array_equal(cleaned_cube.mask, cube.mask)
    assert list(cleaned_cube.extra_coords.keys()) == list(cube.extra_coords.keys())
    assert captured["method"] == "astroscrappy"
    assert captured["sigma"] == 5.0
    assert captured["max_iters"] == 5
    assert captured["method_kwargs"]["batch_size"] == 16
    np.testing.assert_array_equal(captured["mask"], cube.mask)
    array_index = (3, 20, 10)
    world = cube.wcs.array_index_to_world(*array_index)
    assert cleaned_cube.wcs.world_to_array_index(*world) == array_index
    assert list(cleaned_cube.global_coords) == list(cube.global_coords)
    assert cleaned_cube.unit == cube.unit
    # meta is an NDMeta subclass that may contain arrays; compare keys only
    assert set(cleaned_cube.meta.keys()) == set(cube.meta.keys())


def test_spectrogram_cube_slice_slices_fits_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    scan_index = cube.shape[0] // 2
    step_index = cube.shape[1] // 2
    segment_cube = cube.raster_slice(scan_index)

    sliced_cube = cube[scan_index, step_index]
    row_index = sliced_cube.shape[0] // 2
    column_index = sliced_cube.shape[1] // 2

    assert sliced_cube.fits_wcs is not None
    assert sliced_cube.fits_wcs.pixel_n_dim == sliced_cube.wcs.pixel_n_dim == 2
    sliced_spectral, sliced_sky = sliced_cube.fits_wcs.array_index_to_world(row_index, column_index)
    expected_spectral, expected_sky = segment_cube.fits_wcs.array_index_to_world(step_index, row_index, column_index)
    assert_quantity_allclose(sliced_spectral.to(u.nm), expected_spectral.to(u.nm))
    assert_quantity_allclose(sliced_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(sliced_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_crop_slices_fits_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"].raster_slice(0)
    wavelength_index = len(cube.spectral_axis) // 2
    wavelength = cube.spectral_axis[wavelength_index]
    image = cube.crop(
        [SpectralCoord(wavelength), None, None, None],
        [SpectralCoord(wavelength), None, None, None],
    )
    row_index = image.shape[0] // 2
    column_index = image.shape[1] // 2

    assert image.fits_wcs is not None
    assert image.fits_wcs.pixel_n_dim == image.wcs.pixel_n_dim == 2
    image_sky = image.fits_wcs.array_index_to_world(row_index, column_index)
    _, expected_sky = cube.fits_wcs.array_index_to_world(row_index, column_index, wavelength_index)
    assert_quantity_allclose(image_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(image_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_exposes_raster_grouping_helpers(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    assert cube.raster_slice(0).shape == (8, 109, 29)
    assert cube.raster_slice(-1).shape == (8, 109, 29)
    assert len(cube.split_rasters()) == 13
    with pytest.raises(IndexError, match=r"Raster index out of range."):
        cube.raster_slice(-14)
    with pytest.raises(TypeError, match="integer"):
        cube.raster_slice("0")


def test_spectrogram_cube_supports_crop_apis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    spectrum = cube.crop(
        cube.wcs.array_index_to_world(1, 2, 50, 0),
        cube.wcs.array_index_to_world(1, 2, 50, cube.shape[-1] - 1),
    )
    spectrum_by_values = cube.crop_by_values(
        cube.wcs.array_index_to_world_values(1, 2, 50, 0),
        cube.wcs.array_index_to_world_values(1, 2, 50, cube.shape[-1] - 1),
        units=(u.nm, u.arcsec, u.arcsec, u.s, u.pix, u.pix),
    )

    assert spectrum.data.ndim == 1
    assert spectrum_by_values.data.ndim == 1


def test_spectrogram_cube_scan_slice_recovers_segment_fits_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    subcube = cube[1]
    array_index = (2, 50, 10)

    expected_segment = cube.raster_slice(1)
    assert subcube.shape == expected_segment.shape
    assert subcube.fits_wcs is not None
    sub_spectral, sub_sky = subcube.fits_wcs.array_index_to_world(*array_index)
    expected_spectral, expected_sky = expected_segment.fits_wcs.array_index_to_world(*array_index)
    assert_quantity_allclose(sub_spectral.to(u.nm), expected_spectral.to(u.nm))
    assert_quantity_allclose(sub_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(sub_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_memmap_split_rasters_returns_lazy_subcubes(raster_sg_files):
    with pytest.warns(UserWarning, match="uncertainty is not computed when memmap=True"):
        raster = read_spectrograph_lvl2(raster_sg_files, memmap=True, uncertainty=True)
    cube = raster["Si IV 1403"].split_rasters()[0]

    assert isinstance(cube.data, da.Array)
    assert cube.mask is None
    assert cube.uncertainty is None
    assert cube.fits_wcs is not None
    assert cube.shape == (8, 109, 29)


def test_memmap_raster_returns_lazy_combined_cube(raster_sg_files):
    with pytest.warns(UserWarning, match="uncertainty is not computed when memmap=True"):
        raster = read_spectrograph_lvl2(raster_sg_files, memmap=True, uncertainty=True)
    cube = raster["Si IV 1403"]
    image = cube.crop(
        [SpectralCoord(cube.spectral_axis[len(cube.spectral_axis) // 2]), None, None, None, None],
        [SpectralCoord(cube.spectral_axis[len(cube.spectral_axis) // 2]), None, None, None, None],
    )

    assert isinstance(cube.data, da.Array)
    assert cube.mask is None
    assert len(cube.split_rasters()) == 13
    assert cube.fits_wcs is None
    assert cube.uncertainty is None
    assert image.data.ndim == 3


def test_memmap_raster_values_match_raw_fits_after_reader_returns(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True)
    cube = raster["Si IV 1403"]
    raster0 = cube.raster_slice(0)

    with fits.open(raster_sg_files[0], memmap=True, do_not_scale_image_data=True) as hdulist:
        windows = np.array([hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)])
        window_ext = int(np.where(windows == "Si IV 1403")[0][0]) + 1
        expected_data = np.array(hdulist[window_ext].data, copy=True)

    np.testing.assert_array_equal(raster0.data.compute(), expected_data)
    np.testing.assert_array_equal(
        (raster0.data == BAD_PIXEL_VALUE_UNSCALED).compute(),
        expected_data == BAD_PIXEL_VALUE_UNSCALED,
    )

    np.testing.assert_array_equal(raster0[3, 50, :].data.compute(), expected_data[3, 50])


def test_memmap_raster_uses_subfile_scan_chunks(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True)
    cube = raster["Si IV 1403"]
    expected_rows = _lazy_raster_scan_chunk_rows(cube.raster_slice(0))

    assert isinstance(cube.data, da.Array)
    assert max(cube.data.chunks[1]) == expected_rows
    assert len(cube.data.chunks[0]) == len(cube.split_rasters())


def test_memmap_raster_single_slice_opens_one_chunk(raster_sg_files, monkeypatch):
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True)
    cube = raster["Si IV 1403"]
    with fits.open(raster_sg_files[0], memmap=True, do_not_scale_image_data=True) as hdulist:
        windows = np.array([hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)])
        window_ext = int(np.where(windows == "Si IV 1403")[0][0]) + 1
        expected = np.array(hdulist[window_ext].data[0, 0], copy=True)

    open_calls = []
    real_open = raster_combine.fits.open

    def count_open(*args, **kwargs):
        open_calls.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(raster_combine.fits, "open", count_open)

    np.testing.assert_array_equal(cube.data[0, 0, 0].compute(), expected)
    assert open_calls == [raster_sg_files[0]]


def test_default_raster_animation_plots(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"].raster_slice(0)
    fig = plt.figure()
    animator = cube.plot(fig=fig)

    assert animator.axes
    plt.close(fig)


def test_raster_animation_reapplies_axis_properties_after_update(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"].raster_slice(0)
    fig = plt.figure()
    animator = cube.plot(fig=fig)
    slider = SimpleNamespace(cval=0)

    initial_colors = {coord.get_axislabel(): coord._axislabels.get_color() for coord in animator.axes.coords}
    animator.update_plot_2d(0, animator.im, slider)
    updated_colors = {coord.get_axislabel(): coord._axislabels.get_color() for coord in animator.axes.coords}

    assert initial_colors["Helioprojective Latitude [arcsec]"] == "red"
    assert initial_colors["Helioprojective Longitude [arcsec]"] == "black"
    assert updated_colors["Helioprojective Latitude [arcsec]"] == "red"
    assert updated_colors["Helioprojective Longitude [arcsec]"] == "black"
    plt.close(fig)


def test_raster_animation_can_put_requested_longitude_on_bottom(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Mg II k 2796")
    cube = raster["Mg II k 2796"][0]
    fig = plt.figure()
    with pytest.warns(NDCubeUserWarning, match="does not support transposing"):
        animator = cube.plot(
            plot_axes=["x", "y", None],
            axes_coordinates=["custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat", None],
            aspect="auto",
            fig=fig,
        )
    slider = SimpleNamespace(cval=0)
    lon = animator.axes.coords["custom:pos.helioprojective.lon"]
    lat = animator.axes.coords["custom:pos.helioprojective.lat"]
    time = animator.axes.coords["time"]

    assert "b" in lon.get_axislabel_position()
    assert "l" in lat.get_axislabel_position()
    assert time.get_axislabel() == ""

    animator.update_plot_2d(0, animator.im, slider)

    assert "b" in lon.get_axislabel_position()
    assert "l" in lat.get_axislabel_position()
    assert time.get_axislabel() == ""
    plt.close(fig)


def test_raster_animation_time_axis_label_does_not_repeat_unit(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Mg II k 2796")
    cube = raster["Mg II k 2796"][0]
    fig = plt.figure()
    with pytest.warns(NDCubeUserWarning, match="does not support transposing"):
        animator = cube.plot(plot_axes=["x", "y", None], aspect="auto", fig=fig)
    slider = SimpleNamespace(cval=0)
    time = animator.axes.coords["time"]

    assert time.get_axislabel() == "Seconds from Start [$\\mathrm{s}$]"

    animator.update_plot_2d(0, animator.im, slider)

    assert time.get_axislabel() == "Seconds from Start [$\\mathrm{s}$]"
    plt.close(fig)


def test_default_raster_sequence_animation_labels_scan_and_step(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    fig = plt.figure()
    animator = cube.plot(fig=fig, vmin=0, vmax=1000)

    assert animator.slider_labels == ["Scan number", "Raster step"]
    plt.close(fig)


def test_raster_animation_accepts_custom_slider_labels(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    fig = plt.figure()
    animator = cube.plot(fig=fig, slider_labels=["Slit", "Line"], vmin=0, vmax=1000)

    assert animator.slider_labels == ["Slit", "Line"]
    plt.close(fig)


def test_spectrogram_cube_fancy_indexing_strips_raster_metadata(raster_sg_files):
    """
    Non-standard indices (arrays, booleans) cannot preserve per-raster WCS bridges.
    """
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    # _normalize_fits_wcs_item returns None for non-standard indices
    fancy_item = (np.array([0, 2, 4]), slice(None), slice(None), slice(None))
    assert cube._normalize_fits_wcs_item(fancy_item) is None

    # Verify _slice_raster_metadata strips metadata when normalization fails
    sliced = cube[0:1]
    cube._slice_raster_metadata(fancy_item, sliced)
    assert sliced._fits_wcs_segments is None
    assert sliced._raster_boundaries is None


def test_spectrogram_cube_slice_preserves_coordinates(raster_sg_files):
    """
    Slicing a raster cube should preserve global coords and extra coord keys.
    """
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    sliced = cube[0:1]
    assert sliced.global_coords == cube.global_coords
    assert list(sliced.extra_coords.keys()) == list(cube.extra_coords.keys())


def test_spectrogram_cube_to_nddata_preserves_raster_metadata(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    copied = cube.to_nddata(
        data=cube.data.copy(),
        nddata_type=type(cube),
        extra_coords="copy",
        global_coords="copy",
    )

    assert copied._raster_boundaries == cube._raster_boundaries
    assert copied.fits_wcs is cube.fits_wcs
    assert copied._raster_wcs_header is cube._raster_wcs_header
    assert copied._raster_pc_table is cube._raster_pc_table
    assert copied._raster_crval_table is cube._raster_crval_table
    assert copied.meta.observer is cube.meta.observer


def test_spectrogram_cube_arithmetic_preserves_raster_metadata(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    doubled = cube * 2

    assert type(doubled) is type(cube)
    assert doubled._raster_boundaries == cube._raster_boundaries
    assert doubled.fits_wcs is cube.fits_wcs
    assert doubled._raster_pc_table is cube._raster_pc_table
    assert doubled._raster_crval_table is cube._raster_crval_table


def test_spectral_dispersion():
    wavelengths = np.arange(10) * 0.02 * u.nm + 140 * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), wavelengths)
    dispersion = cube.spectral_dispersion
    assert dispersion.unit.is_equivalent(u.nm)
    assert u.isclose(dispersion, 0.02 * u.nm, rtol=0.01)


def test_solid_angle():
    wavelengths = np.arange(10) * 0.02 * u.nm + 140 * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), wavelengths)
    angle = cube.solid_angle
    assert angle.unit.is_equivalent(u.sr)
    expected = 1.0 * u.arcsec * SLIT_WIDTH
    assert u.isclose(angle.to(u.sr), expected.to(u.sr), rtol=0.01)


def test_wavelength_axis():
    wavelengths = np.arange(10) * 0.02 * u.nm + 140 * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), wavelengths)
    assert cube.wavelength_axis == 2


def test_wavelength_axis_raises_without_wave():
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 5
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLT-TAN"
    header["CTYPE2"] = "HPLN-TAN"
    header["CDELT1"] = 0.1
    header["CRVAL1"] = 0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "arcsec"
    header["CDELT2"] = 0.1
    header["CRVAL2"] = 0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "arcsec"
    wcs = WCS(header)
    cube = SpectrogramCube(np.ones((2, 5)), wcs=wcs, uncertainty=None, unit=u.DN, meta={}, mask=None)
    with pytest.raises(ValueError, match="wavelength axis"):
        _ = cube.wavelength_axis


def test_spectral_dispersion_missing_wave_raises():
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 5
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLT-TAN"
    header["CTYPE2"] = "HPLN-TAN"
    header["CDELT1"] = 0.1
    header["CRVAL1"] = 0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "arcsec"
    header["CDELT2"] = 0.1
    header["CRVAL2"] = 0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "arcsec"
    wcs = WCS(header)
    cube = SpectrogramCube(np.ones((2, 5)), wcs=wcs, uncertainty=None, unit=u.DN, meta={}, mask=None)
    with pytest.raises(ValueError, match="no WAVE ctype"):
        _ = cube.spectral_dispersion


def test_solid_angle_missing_hplt_raises():
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 5
    header["NAXIS2"] = 2
    header["NAXIS3"] = 1
    header["CTYPE1"] = "WAVE"
    header["CTYPE2"] = "TIME"
    header["CTYPE3"] = "UTC"
    header["CDELT1"] = 0.02
    header["CRVAL1"] = 140.0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "nm"
    header["CDELT2"] = 1.0
    header["CRVAL2"] = 0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "s"
    header["CDELT3"] = 1.0
    header["CRVAL3"] = 0
    header["CRPIX3"] = 1
    header["CUNIT3"] = "s"
    wcs = WCS(header)
    cube = SpectrogramCube(np.ones((1, 2, 5)), wcs=wcs, uncertainty=None, unit=u.DN, meta={}, mask=None)
    with pytest.raises(ValueError, match="no HPLT ctype"):
        _ = cube.solid_angle


def test_spectral_dispersion_real_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"]
    dispersion = cube.spectral_dispersion
    assert dispersion.unit.is_equivalent(u.nm)
    assert dispersion.value > 0


def test_solid_angle_real_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"]
    angle = cube.solid_angle
    assert angle.unit.is_equivalent(u.sr)
    assert angle.value > 0
