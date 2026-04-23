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


def test_spectrogram_cube_slice_slices_basic_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    slice_index = cube.shape[0] // 2
    segment_index, segment_slice = next(
        (i, boundary) for i, boundary in enumerate(cube.raster_boundaries) if boundary.start <= slice_index < boundary.stop
    )
    segment_cube = cube.raster_slice(segment_index)
    local_index = slice_index - segment_slice.start

    sliced_cube = cube[slice_index]
    row_index = sliced_cube.shape[0] // 2
    column_index = sliced_cube.shape[1] // 2

    assert sliced_cube.basic_wcs is not None
    assert sliced_cube.basic_wcs.pixel_n_dim == sliced_cube.wcs.pixel_n_dim == 2
    sliced_spectral, sliced_sky = sliced_cube.basic_wcs.array_index_to_world(row_index, column_index)
    expected_spectral, expected_sky = segment_cube.basic_wcs.array_index_to_world(local_index, row_index, column_index)
    assert_quantity_allclose(sliced_spectral.to(u.nm), expected_spectral.to(u.nm))
    assert_quantity_allclose(sliced_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(sliced_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_crop_slices_basic_wcs(raster_sg_files):
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

    assert image.basic_wcs is not None
    assert image.basic_wcs.pixel_n_dim == image.wcs.pixel_n_dim == 2
    image_sky = image.basic_wcs.array_index_to_world(row_index, column_index)
    _, expected_sky = cube.basic_wcs.array_index_to_world(row_index, column_index, wavelength_index)
    assert_quantity_allclose(image_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(image_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_spectrum_at_returns_nearest_raster_spectrum(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    _, target, _, _ = cube.wcs.array_index_to_world(3, 50, 10)

    expected = cube.crop(
        cube.wcs.array_index_to_world(3, 50, 0),
        cube.wcs.array_index_to_world(3, 50, cube.shape[-1] - 1),
    )
    spectrum = cube.spectrum_at(target)

    assert spectrum.shape == expected.shape
    np.testing.assert_array_equal(spectrum.data, expected.data)


def test_spectrogram_cube_exposes_raster_grouping_helpers(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    assert len(cube.raster_boundaries) == 13
    assert cube.raster_slice(0).shape == (8, 109, 29)
    assert len(cube.split_rasters()) == 13


def test_spectrogram_cube_supports_crop_apis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    spectrum = cube.crop(
        cube.wcs.array_index_to_world(10, 50, 0),
        cube.wcs.array_index_to_world(10, 50, cube.shape[-1] - 1),
    )
    spectrum_by_values = cube.crop_by_values(
        cube.wcs.array_index_to_world_values(10, 50, 0),
        cube.wcs.array_index_to_world_values(10, 50, cube.shape[-1] - 1),
        units=(u.nm, u.arcsec, u.arcsec, u.s, u.pix),
    )

    assert spectrum.data.ndim == 1
    assert spectrum_by_values.data.ndim == 1


def test_spectrogram_cube_spectrum_at_works_without_basic_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    _, target, _, _ = cube.wcs.array_index_to_world(10, 50, 0)

    expected = cube.crop(
        cube.wcs.array_index_to_world(10, 50, 0),
        cube.wcs.array_index_to_world(10, 50, cube.shape[-1] - 1),
    )
    spectrum = cube.spectrum_at(target)

    assert cube.basic_wcs is None
    assert spectrum.shape == expected.shape
    np.testing.assert_array_equal(spectrum.data, expected.data)


def test_spectrogram_cube_scan_slice_recovers_segment_basic_wcs(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]

    subcube = cube[8:16]
    array_index = (2, 50, 10)

    expected_segment = cube.raster_slice(1)
    assert subcube.shape == expected_segment.shape
    assert subcube.basic_wcs is not None
    sub_spectral, sub_sky = subcube.basic_wcs.array_index_to_world(*array_index)
    expected_spectral, expected_sky = expected_segment.basic_wcs.array_index_to_world(*array_index)
    assert_quantity_allclose(sub_spectral.to(u.nm), expected_spectral.to(u.nm))
    assert_quantity_allclose(sub_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(sub_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_memmap_split_rasters_returns_lazy_subcubes(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True, uncertainty=True)
    cube = raster["Si IV 1403"].split_rasters()[0]

    assert isinstance(cube.data, da.Array)
    assert isinstance(cube.mask, da.Array)
    assert cube.uncertainty is None
    assert cube._memmap is True
    assert cube.basic_wcs is not None
    assert cube.shape == (8, 109, 29)


def test_memmap_raster_returns_lazy_combined_cube(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True, uncertainty=True)
    cube = raster["Si IV 1403"]
    _, target, _, _ = cube.wcs.array_index_to_world(10, 50, 0)
    spectrum = cube.spectrum_at(target)
    image = cube.crop(
        [SpectralCoord(cube.spectral_axis[len(cube.spectral_axis) // 2]), None, None, None],
        [SpectralCoord(cube.spectral_axis[len(cube.spectral_axis) // 2]), None, None, None],
    )

    assert isinstance(cube.data, da.Array)
    assert isinstance(cube.mask, da.Array)
    assert cube._memmap is True
    assert len(cube.raster_boundaries) == 13
    assert cube.basic_wcs is None
    assert cube.uncertainty is None
    assert spectrum.data.ndim == 1
    assert image.data.ndim == 2


def test_memmap_raster_uses_subfile_scan_chunks(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files, memmap=True)
    cube = raster["Si IV 1403"]

    assert isinstance(cube.data, da.Array)
    assert len(cube.data.chunks[0]) > len(cube.raster_boundaries)
    assert max(cube.data.chunks[0]) < cube.raster_slice(0).shape[0]


def _get_coord(ax, *, coord_type=None, coord_unit=None):
    for coord in ax.coords:
        if coord_type is not None and coord.coord_type != coord_type:
            continue
        if coord_unit is not None and coord.coord_unit != coord_unit:
            continue
        return coord
    pytest.fail(f"Coordinate with type={coord_type!r} and unit={coord_unit!r} not found.")


def test_default_raster_animation_keeps_wavelength_on_bottom(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    fig = plt.figure()
    animator = cube.plot(fig=fig)
    ax = animator.axes
    wavelength = _get_coord(ax, coord_type="scalar", coord_unit=u.nm)

    assert "b" in wavelength.get_ticks_position()
    assert "b" in wavelength.get_axislabel_position()
    plt.close(fig)


def test_raster_animation_can_show_time_axis(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    fig = plt.figure()
    with pytest.warns(
        NDCubeUserWarning,
        match="Animating a NDCube does not support transposing the array",
    ):
        animator = cube.plot(
            fig=fig,
            plot_axes=["x", "y", None],
            axes_coordinates=["time", "custom:pos.helioprojective.lat", None],
            vmin=0,
            vmax=1000,
        )
    ax = animator.axes
    longitude = _get_coord(ax, coord_type="longitude")
    latitude = _get_coord(ax, coord_type="latitude")
    time = _get_coord(ax, coord_type="scalar", coord_unit=u.s)

    assert "b" in time.get_ticks_position()
    assert "b" in time.get_axislabel_position()
    assert "l" in latitude.get_ticks_position()
    assert "b" not in longitude.get_ticks_position()
    plt.close(fig)


def test_raster_animation_keeps_longitude_axis_after_slider_update(raster_sg_files):
    raster = read_spectrograph_lvl2(raster_sg_files)
    cube = raster["Si IV 1403"]
    fig = plt.figure()
    with pytest.warns(
        NDCubeUserWarning,
        match="Animating a NDCube does not support transposing the array",
    ):
        animator = cube.plot(fig=fig, plot_axes=["x", "y", None], vmin=0, vmax=1000)

    class _DummyText:
        def set_text(self, _):
            return None

    slider = SimpleNamespace(cval=0, slider_ind=0, valtext=_DummyText())
    animator.update_plot(1, animator.im, slider)
    ax = animator.axes
    longitude = _get_coord(ax, coord_type="longitude")
    latitude = _get_coord(ax, coord_type="latitude")
    time = _get_coord(ax, coord_type="scalar", coord_unit=u.s)

    assert "b" in longitude.get_ticks_position()
    assert "b" in longitude.get_axislabel_position()
    assert "l" in latitude.get_ticks_position()
    assert "b" not in time.get_ticks_position()
    assert time.get_axislabel() == ""
    plt.close(fig)

