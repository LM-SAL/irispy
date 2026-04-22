import copy
import warnings
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.exceptions import AstropyUserWarning
from astropy.io import fits
from astropy.units import UnitsWarning
from ndcube.utils.exceptions import NDCubeUserWarning

from irispy.io.spectrograph import read_spectrograph_lvl2


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
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window1].data[0].data, data1)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window2].data[0].data, data2)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window3].data[0].data, data3)


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
    cube = raster[key][0]
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


def test_spectrogram_cube_sequence_remove_cosmic_rays(sns_sg_file, monkeypatch):
    captured = {}

    def fake_remove_cosmic_rays(cube, *, method, sigma, max_iters, method_kwargs):
        captured["cube"] = cube
        captured["method"] = method
        captured["sigma"] = sigma
        captured["max_iters"] = max_iters
        captured["method_kwargs"] = method_kwargs
        return cube

    monkeypatch.setattr("irispy.spectrograph.remove_cosmic_rays", fake_remove_cosmic_rays)

    raster = read_spectrograph_lvl2(sns_sg_file)
    key = next(iter(raster.keys()))
    sequence = raster[key]
    cleaned_sequence = sequence.remove_cosmic_rays(
        method="astroscrappy",
        sigma=5.0,
        max_iters=5,
        method_kwargs={"batch_size": 16},
    )

    assert cleaned_sequence is sequence
    assert captured["cube"] is sequence
    assert captured["method"] == "astroscrappy"
    assert captured["sigma"] == 5.0
    assert captured["max_iters"] == 5
    assert captured["method_kwargs"] == {"batch_size": 16}


def test_spectrogram_cube_slice_slices_basic_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"][0]
    slice_index = cube.shape[0] // 2

    sliced_cube = cube[slice_index]
    row_index = sliced_cube.shape[0] // 2
    column_index = sliced_cube.shape[1] // 2

    assert sliced_cube.basic_wcs is not None
    assert sliced_cube.basic_wcs.pixel_n_dim == sliced_cube.wcs.pixel_n_dim == 2
    sliced_spectral, sliced_sky = sliced_cube.basic_wcs.array_index_to_world(row_index, column_index)
    expected_spectral, expected_sky = cube.basic_wcs.array_index_to_world(slice_index, row_index, column_index)
    assert_quantity_allclose(sliced_spectral.to(u.nm), expected_spectral.to(u.nm))
    assert_quantity_allclose(sliced_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(sliced_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_crop_slices_basic_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"][0]
    wavelength_index = len(cube.spectral_axis) // 2
    wavelength = cube.spectral_axis[wavelength_index]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        warnings.simplefilter("ignore", UnitsWarning)
        image = cube.crop(
            [SpectralCoord(wavelength), None, None, None],
            [SpectralCoord(wavelength), None, None, None],
        )
    row_index = image.shape[0] // 2
    column_index = image.shape[1] // 2

    assert image.basic_wcs is not None
    assert image.basic_wcs.pixel_n_dim == image.wcs.pixel_n_dim == 2
    image_sky = image.basic_wcs.array_index_to_world(row_index, column_index)
    _, expected_sky = cube.basic_wcs.array_index_to_world(row_index, column_index, wavelength_index)
    assert_quantity_allclose(image_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(image_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_sequence_slice_preserves_type_and_common_axis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    sequence = raster["Si IV 1403"]

    sliced_sequence = sequence[:3]

    assert isinstance(sliced_sequence, type(sequence))
    assert len(sliced_sequence) == 3
    assert getattr(sliced_sequence, "_common_axis", None) == getattr(sequence, "_common_axis", None)
    assert sliced_sequence[0].shape == sequence[0].shape


def _get_coord(ax, default_label):
    for coord in ax.coords:
        if coord.default_label.lower() == default_label:
            return coord
    pytest.fail(f"Coordinate '{default_label}' not found.")


def test_raster_animation_defaults_to_longitude_axis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"][0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NDCubeUserWarning)
        fig = plt.figure()
        animator = cube.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=1000)
    ax = animator.axes

    longitude = _get_coord(ax, "helioprojective longitude")
    latitude = _get_coord(ax, "helioprojective latitude")
    time = _get_coord(ax, "seconds from start (s)")

    assert longitude.get_ticks_position() == ["b", "r"]
    assert longitude.get_axislabel_position() == ["b", "r"]
    assert latitude.get_ticks_position() == ["l"]
    assert latitude.get_axislabel_position() == ["l"]
    assert time.get_ticks_position() == ["#"]
    assert time.get_axislabel() == ""
    plt.close(fig)


def test_default_raster_animation_keeps_wavelength_on_bottom(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"][0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NDCubeUserWarning)
        fig = plt.figure()
        animator = cube.plot(fig=fig)
    ax = animator.axes

    wavelength = _get_coord(ax, "wavelength")

    assert wavelength.get_ticks_position() == ["b", "#"]
    assert wavelength.get_axislabel_position() == ["b", "#"]
    plt.close(fig)


def test_raster_animation_can_show_time_axis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"][0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NDCubeUserWarning)
        fig = plt.figure()
        animator = cube.plot(
            fig=fig,
            plot_axes=["x", "y", None],
            axes_coordinates=["time", "custom:pos.helioprojective.lat", None],
            vmin=0,
            vmax=1000,
        )
    ax = animator.axes

    longitude = _get_coord(ax, "helioprojective longitude")
    latitude = _get_coord(ax, "helioprojective latitude")
    time = _get_coord(ax, "seconds from start (s)")

    assert time.get_ticks_position() == ["b"]
    assert time.get_axislabel_position() == ["b"]
    assert latitude.get_ticks_position() == ["l"]
    assert latitude.get_axislabel_position() == ["l"]
    assert longitude.get_ticks_position() == ["r"]
    assert longitude.get_axislabel_position() == ["r"]
    plt.close(fig)


def test_raster_animation_keeps_longitude_axis_after_slider_update(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"][0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NDCubeUserWarning)
        fig = plt.figure()
        animator = cube.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=1000)

    class _DummyText:
        def set_text(self, _):
            return None

    slider = SimpleNamespace(cval=0, slider_ind=0, valtext=_DummyText())
    animator.update_plot(1, animator.im, slider)
    ax = animator.axes

    longitude = _get_coord(ax, "helioprojective longitude")
    latitude = _get_coord(ax, "helioprojective latitude")
    time = _get_coord(ax, "seconds from start (s)")

    assert longitude.get_ticks_position() == ["b", "r"]
    assert longitude.get_axislabel_position() == ["b", "r"]
    assert latitude.get_ticks_position() == ["l"]
    assert time.get_ticks_position() == ["#"]
    assert time.get_axislabel() == ""
    plt.close(fig)


def test_raster_sequence_animation_defaults_to_longitude_axis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    sequence = raster["Si IV 1403"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NDCubeUserWarning)
        fig = plt.figure()
        animator = sequence.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=1000)
    ax = animator.axes

    longitude = _get_coord(ax, "helioprojective longitude")
    latitude = _get_coord(ax, "helioprojective latitude")
    time = _get_coord(ax, "seconds from start (s)")

    assert longitude.get_ticks_position() == ["b", "r"]
    assert longitude.get_axislabel_position() == ["b", "r"]
    assert latitude.get_ticks_position() == ["l"]
    assert latitude.get_axislabel_position() == ["l"]
    assert time.get_ticks_position() == ["#"]
    assert time.get_axislabel() == ""
    plt.close(fig)


def test_raster_sequence_animation_keeps_requested_axis_after_slider_update(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    sequence = raster["Si IV 1403"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NDCubeUserWarning)
        fig = plt.figure()
        animator = sequence.plot(
            fig=fig,
            plot_axes=["x", "y", None],
            axes_coordinates=["time", "custom:pos.helioprojective.lat", None],
            vmin=0,
            vmax=1000,
        )

    class _DummyText:
        def set_text(self, _):
            return None

    slider = SimpleNamespace(cval=0, slider_ind=0, valtext=_DummyText())
    animator.update_plot(1, animator.im, slider)
    ax = animator.axes

    longitude = _get_coord(ax, "helioprojective longitude")
    latitude = _get_coord(ax, "helioprojective latitude")
    time = _get_coord(ax, "seconds from start (s)")

    assert time.get_ticks_position() == ["b"]
    assert time.get_axislabel_position() == ["b"]
    assert latitude.get_ticks_position() == ["l"]
    assert latitude.get_axislabel_position() == ["l"]
    assert longitude.get_ticks_position() == ["r"]
    assert longitude.get_axislabel_position() == ["r"]
    plt.close(fig)
