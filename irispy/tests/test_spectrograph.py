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

import irispy.io._raster_combine as raster_combine
from irispy.io._raster_combine import _lazy_raster_scan_chunk_rows
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.utils import read_files
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import BAD_PIXEL_VALUE_UNSCALED, SLIT_WIDTH

LON, LAT = "custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"
SPATIAL_AXES = [("HPLT-TAN", "arcsec", 0.1, 0), ("HPLN-TAN", "arcsec", 0.1, 0)]
NO_LATITUDE_AXES = [("WAVE", "nm", 0.02, 140.0), ("TIME", "s", 1.0, 0), ("UTC", "s", 1.0, 0)]


@pytest.fixture
def combined_si_iv(raster_sg_files):
    return read_spectrograph_lvl2(raster_sg_files, spectral_windows="Si IV 1403")["Si IV 1403"]


def _raw_si_iv(filename):
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as hdulist:
        header = hdulist[0].header
        windows = [header[f"TDESC{i}"] for i in range(1, header["NWIN"] + 1)]
        return np.array(hdulist[windows.index("Si IV 1403") + 1].data, copy=True)


def test_fits_data_comparison(sns_sg_file):
    raster = read_spectrograph_lvl2(sns_sg_file)
    with fits.open(sns_sg_file) as hdulist:
        for ext in (1, 2, 3):
            np.testing.assert_array_almost_equal(raster[hdulist[0].header[f"TDESC{ext}"]].data, hdulist[ext].data)


def test_spectrogram_cube_remove_cosmic_rays(sns_sg_file, monkeypatch):
    captured = {}

    def fake_remove_cosmic_rays(cube, **kwargs):
        captured.update(kwargs, mask=cube.mask.copy())
        return cube.to_nddata(
            data=cube.data + 2, mask="copy", nddata_type=type(cube), extra_coords="copy", global_coords="copy"
        )

    monkeypatch.setattr("irispy.spectrograph.remove_cosmic_rays", fake_remove_cosmic_rays)
    raster = read_spectrograph_lvl2(sns_sg_file)
    cube = raster[next(iter(raster))]
    cleaned_cube = cube.remove_cosmic_rays(
        method="astroscrappy", sigma=5.0, max_iters=5, method_kwargs={"batch_size": 16}
    )

    np.testing.assert_array_equal(captured.pop("mask"), cube.mask)
    assert captured == {"method": "astroscrappy", "sigma": 5.0, "max_iters": 5, "method_kwargs": {"batch_size": 16}}
    np.testing.assert_array_equal(cleaned_cube.data, cube.data + 2)
    np.testing.assert_array_equal(cleaned_cube.mask, cube.mask)
    assert list(cleaned_cube.extra_coords.keys()) == list(cube.extra_coords.keys())
    array_index = (3, 20, 10)
    assert cleaned_cube.wcs.world_to_array_index(*cube.wcs.array_index_to_world(*array_index)) == array_index
    assert list(cleaned_cube.global_coords) == list(cube.global_coords)
    assert cleaned_cube.unit == cube.unit
    # meta is an NDMeta subclass that may contain arrays; compare keys only
    assert set(cleaned_cube.meta.keys()) == set(cube.meta.keys())


@pytest.mark.parametrize(
    ("item", "scan", "index", "segment_index"),
    [(1, 1, (2, 50, 10), (2, 50, 10)), ((6, 4), 6, (50, 10), (4, 50, 10))],
    ids=["scan", "scan_and_step"],
)
def test_integer_index_slices_segment_fits_wcs(combined_si_iv, item, scan, index, segment_index):
    sliced, segment = combined_si_iv[item], combined_si_iv.raster_slice(scan)

    assert sliced.shape == segment.shape[-len(index) :]
    assert sliced.fits_wcs is not None
    assert sliced.fits_wcs.pixel_n_dim == sliced.wcs.pixel_n_dim == len(index)
    sliced_spectral, sliced_sky = sliced.fits_wcs.array_index_to_world(*index)
    expected_spectral, expected_sky = segment.fits_wcs.array_index_to_world(*segment_index)
    assert_quantity_allclose(sliced_spectral.to(u.nm), expected_spectral.to(u.nm))
    assert_quantity_allclose(sliced_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(sliced_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_spectrogram_cube_crop_slices_fits_wcs(combined_si_iv):
    cube = combined_si_iv.raster_slice(0)
    wavelength_index = len(cube.spectral_axis) // 2
    wavelength = SpectralCoord(cube.spectral_axis[wavelength_index])
    image = cube.crop([wavelength, None, None, None], [wavelength, None, None, None])
    row_index, column_index = image.shape[0] // 2, image.shape[1] // 2

    assert image.fits_wcs is not None
    assert image.fits_wcs.pixel_n_dim == image.wcs.pixel_n_dim == 2
    image_sky = image.fits_wcs.array_index_to_world(row_index, column_index)
    _, expected_sky = cube.fits_wcs.array_index_to_world(row_index, column_index, wavelength_index)
    assert_quantity_allclose(image_sky.Tx.to(u.arcsec), expected_sky.Tx.to(u.arcsec))
    assert_quantity_allclose(image_sky.Ty.to(u.arcsec), expected_sky.Ty.to(u.arcsec))


def test_memmap_raster_and_split_rasters_are_lazy(raster_sg_files):
    cube = read_spectrograph_lvl2(raster_sg_files, memmap=True)["Si IV 1403"]
    wavelength = SpectralCoord(cube.spectral_axis[len(cube.spectral_axis) // 2])
    rasters = cube.split_rasters()

    assert cube.crop([wavelength, None, None, None, None], [wavelength, None, None, None, None]).data.ndim == 3
    assert len(rasters) == 13
    assert cube.fits_wcs is None
    assert rasters[0].fits_wcs is not None
    assert rasters[0].shape == (8, 109, 29)
    for lazy in (cube, rasters[0]):
        assert isinstance(lazy.data, da.Array)


def test_memmap_raster_reads_raw_fits_one_scan_chunk_at_a_time(raster_sg_files, monkeypatch):
    cube = read_spectrograph_lvl2(raster_sg_files, memmap=True)["Si IV 1403"]
    raster0 = cube.raster_slice(0)
    expected = _raw_si_iv(raster_sg_files[0])

    assert max(cube.data.chunks[1]) == _lazy_raster_scan_chunk_rows(raster0.data)
    assert len(cube.data.chunks[0]) == len(cube.split_rasters())
    np.testing.assert_array_equal(raster0.data.compute(), expected)
    np.testing.assert_array_equal(
        (raster0.data == BAD_PIXEL_VALUE_UNSCALED).compute(), expected == BAD_PIXEL_VALUE_UNSCALED
    )
    np.testing.assert_array_equal(raster0[3, 50, :].data.compute(), expected[3, 50])

    open_calls = []
    real_open = raster_combine.fits.open

    def count_open(*args, **kwargs):
        open_calls.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(raster_combine, "fits", SimpleNamespace(open=count_open))
    np.testing.assert_array_equal(cube.data[0, 0, 0].compute(), expected[0, 0])
    assert open_calls == [raster_sg_files[0]]


@pytest.fixture
def mg_ii_raster(raster_sg_file):
    return read_spectrograph_lvl2(raster_sg_file, spectral_windows="Mg II k 2796")["Mg II k 2796"]


@pytest.fixture
def mg_ii_image(mg_ii_raster):
    wavelength = SpectralCoord(mg_ii_raster.spectral_axis[len(mg_ii_raster.spectral_axis) // 2])
    return mg_ii_raster.crop([wavelength, None, None, None], [wavelength, None, None, None])


def test_spectrogram_animation_layout_and_label_colors_survive_update(mg_ii_raster):
    """
    Latitude is on the left and longitude on the right, time is hidden, and the latitude
    label is red, before and after a slider update rebuilds the coordinates.
    """
    fig = plt.figure()
    animator = mg_ii_raster.plot(fig=fig)

    def assert_layout():
        coords = animator.axes.coords
        assert coords[LAT].get_ticklabel_position() == ["l"]
        assert coords[LON].get_ticklabel_position() == ["r"]
        assert not coords["time"].get_ticklabel_visible()
        # WCSAxes has no public getter for an axis label's color.
        assert (coords[LAT]._axislabels.get_color(), coords[LON]._axislabels.get_color()) == ("red", "black")

    assert_layout()
    animator.update_plot_2d(0, animator.im, SimpleNamespace(cval=0))
    assert_layout()
    plt.close(fig)


@pytest.mark.parametrize(
    ("axes_coordinates", "edges", "hidden"),
    [
        (None, {LON: "b", LAT: "l"}, ["time", "raster_step"]),
        ([LON, LAT, None], {LON: "b", LAT: "l"}, ["time", "raster_step"]),
        ([LON, None, None], {LON: "b"}, [LAT, "time", "raster_step"]),
        ([LON, "time", None], {LON: "b"}, [LAT, "raster_step"]),
    ],
    ids=["default", "longitude_and_latitude", "longitude_only", "longitude_and_time"],
)
@pytest.mark.filterwarnings("ignore:Animating a NDCube does not support transposing")
def test_raster_animation_axis_layout_survives_update(raster_sg_files, axes_coordinates, edges, hidden):
    """
    Longitude follows the raster step axis and latitude the slit axis; time and step are
    only shown when asked for.

    A hidden coordinate must not take an edge from a shown coordinate.
    """
    cube = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Mg II k 2796")["Mg II k 2796"][0]
    fig = plt.figure()
    animator = cube.plot(plot_axes=["x", "y", None], axes_coordinates=axes_coordinates, aspect="auto", fig=fig)

    def assert_layout():
        # A slider update replaces the coordinates, so look them up each time.
        coords = animator.axes.coords
        assert coords["time"].get_axislabel() == "Seconds from Start [$\\mathrm{s}$]"
        for name, edge in edges.items():
            assert coords[name].get_ticklabel_position() == [edge]
            assert coords[name].get_axislabel_position() == [edge]
        for name in hidden:
            assert not coords[name].get_ticklabel_visible()
        if "time" not in hidden:
            assert coords["time"].get_ticklabel_visible()

    assert_layout()
    animator.update_plot_2d(0, animator.im, SimpleNamespace(cval=0))
    assert_layout()
    plt.close(fig)


@pytest.mark.parametrize(
    ("plot_axes", "edges"),
    [(None, {LAT: "b", LON: "l"}), (["x", "y"], {LON: "b", LAT: "l"})],
    ids=["slit_on_x", "step_on_x"],
)
def test_raster_image_labels_latitude_on_slit_edge_and_longitude_on_step_edge(
    mg_ii_raster, mg_ii_image, plot_axes, edges
):
    assert mg_ii_image.shape == mg_ii_raster.shape[:2]
    fig = plt.figure()
    ax = mg_ii_image.plot(plot_axes=plot_axes)
    fig.canvas.draw()

    for name, edge in edges.items():
        assert ax.coords[name].get_ticklabel_position() == [edge]
    for name in ("time", "raster_step"):
        assert not ax.coords[name].get_ticklabel_visible()
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:Animating a NDCube does not support transposing")
def test_requested_time_coordinate_joins_the_celestial_layout(mg_ii_raster, mg_ii_image):
    """
    A single-file cube carries a "time" extra coordinate, so with ``axes_coordinates``
    ndcube plots through its combined WCS, whose pixel axis names it cannot join.

    The layout must still come from the cube's own pixel axes, with time shown as well.
    """
    fig = plt.figure()
    ax = mg_ii_image.plot(plot_axes=["x", "y"], axes_coordinates=[LON, LAT, "time"])
    fig.canvas.draw()

    assert ax.coords[LON].get_ticklabel_position() == ["b"]
    assert ax.coords[LAT].get_ticklabel_position() == ["l"]
    assert ax.coords["Seconds from Start (s)"].get_ticklabel_visible()
    assert not ax.coords["raster_step"].get_ticklabel_visible()
    plt.close(fig)

    fig = plt.figure()
    animator = mg_ii_raster.plot(plot_axes=["x", "y", None], axes_coordinates=[LON, LAT, "time"], fig=fig)
    animator.update_plot_2d(1, animator.im, SimpleNamespace(cval=1))
    assert animator.axes.coords[LON].get_ticklabel_position() == ["b"]
    assert animator.axes.coords[LAT].get_ticklabel_position() == ["l"]
    assert not animator.axes.coords["raster_step"].get_ticklabel_visible()
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:Animating a NDCube does not support transposing")
def test_fixed_wavelength_raster_spatial_slider_label(raster_sg_files):
    cube = read_spectrograph_lvl2(raster_sg_files, spectral_windows="Mg II k 2796")["Mg II k 2796"]
    wavelength = SpectralCoord(cube.spectral_axis[len(cube.spectral_axis) // 2])
    fixed_wavelength = cube.crop([wavelength, None, None, None, None], [wavelength, None, None, None, None])
    fig = plt.figure()
    animator = fixed_wavelength.plot(plot_axes=["x", "y", None], aspect="auto", fig=fig)

    assert animator.slider_axes == [2]
    assert animator.slider_ranges == [[0, fixed_wavelength.shape[2]]]
    assert animator.slider_labels == ["Spatial pixel"]
    plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "labels"),
    [({}, ["Scan number", "Raster step"]), ({"slider_labels": ["Slit", "Line"]}, ["Slit", "Line"])],
    ids=["default_scan_and_step", "custom"],
)
def test_raster_sequence_animation_slider_labels(combined_si_iv, kwargs, labels):
    fig = plt.figure()
    assert combined_si_iv.plot(fig=fig, vmin=0, vmax=1000, **kwargs).slider_labels == labels
    plt.close(fig)


def test_spectrogram_cube_rejects_fancy_indexing(combined_si_iv):
    with pytest.raises((IndexError, TypeError, ValueError)):
        combined_si_iv[np.array([0, 2, 4])]


def test_spectrogram_cube_slice_preserves_coordinates(combined_si_iv):
    sliced = combined_si_iv[0:1]
    assert sliced.global_coords == combined_si_iv.global_coords
    assert list(sliced.extra_coords.keys()) == list(combined_si_iv.extra_coords.keys())


@pytest.mark.parametrize(
    "derive",
    [
        lambda cube: cube * 2,
        lambda cube: cube.to_nddata(
            data=cube.data.copy(), nddata_type=type(cube), extra_coords="copy", global_coords="copy"
        ),
    ],
    ids=["arithmetic", "to_nddata"],
)
def test_derived_combined_cube_keeps_raster_behaviour(combined_si_iv, derive):
    out = derive(combined_si_iv)

    assert type(out) is type(combined_si_iv)
    assert out.raster_slice(0).shape == combined_si_iv.raster_slice(0).shape
    assert out.raster_slice(0).fits_wcs is not None
    assert len(out.split_rasters()) == 13


def _fits_wcs_sample(cube):
    wcs = cube.fits_wcs
    return wcs.array_shape, wcs.array_index_to_world_values(1, 5, 7)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (lambda c: len(c[:, 3].split_rasters()), lambda _: 13),
        (lambda c: c[:, 3].raster_slice(1).shape, lambda c: c.shape[2:]),
        (lambda c: len(c.rebin((1, 2, 1, 1)).split_rasters()), lambda _: 13),
        (lambda c: c.rebin((1, 2, 1, 1)).raster_slice(0).fits_wcs, lambda _: None),
        (lambda c: _fits_wcs_sample(c[:, 2:6][3]), lambda c: _fits_wcs_sample(c[3][2:6])),
        (
            lambda c: _fits_wcs_sample(c[:, 2:5, 10:60][1]),
            lambda c: (
                (3, 50, 29),
                c[1].fits_wcs.slice((slice(2, 5), slice(10, 60))).array_index_to_world_values(1, 5, 7),
            ),
        ),
    ],
    ids=[
        "step_slice_split",
        "step_slice_raster_slice",
        "rebin_split",
        "rebin_fits_wcs",
        "slice_order",
        "chained_slice",
    ],
)
def test_combined_raster_derived_state(combined_si_iv, actual, expected):
    np.testing.assert_equal(actual(combined_si_iv), expected(combined_si_iv))


@pytest.fixture(params=["raster_sg_file", "raster_sg_files", "sns_sg_file", "sns_sjicube_1330"])
def any_cube(request):
    source = request.getfixturevalue(request.param)
    if request.param == "sns_sjicube_1330":
        return source
    return read_spectrograph_lvl2(source, spectral_windows="Si IV 1403")["Si IV 1403"]


def _slice_summary(cube):
    low = cube.wcs.low_level_wcs
    fits_wcs = cube.fits_wcs if isinstance(cube.fits_wcs, list) else [cube.fits_wcs]
    return (
        cube.data,
        low.pixel_to_world_values(*[0] * low.pixel_n_dim),
        getattr(low, "dropped_world_dimensions", {}).get("value"),
        [wcs.pixel_to_world_values(*[0] * wcs.pixel_n_dim) for wcs in fits_wcs if wcs is not None],
    )


@pytest.mark.parametrize(
    ("negative", "positive"),
    [
        (lambda c: c[-1], lambda c: c[c.shape[0] - 1]),
        (lambda c: c[1:5][-1], lambda c: c[4]),
        (lambda c: c[-3:], lambda c: c[c.shape[0] - 3 :]),
        (lambda c: c[:-2][-2:], lambda c: c[c.shape[0] - 4 : c.shape[0] - 2]),
        (lambda c: c[:, -5:], lambda c: c[:, c.shape[1] - 5 :]),
    ],
    ids=["index", "nested_index", "slice", "nested_slice", "second_axis"],
)
def test_negative_indices_match_positive(any_cube, negative, positive):
    np.testing.assert_equal(_slice_summary(negative(any_cube)), _slice_summary(positive(any_cube)))


@pytest.mark.parametrize(
    "item",
    [-9, np.s_[::2], (3, 10, 5), np.s_[5:2]],
    ids=["out_of_range", "step", "scalar", "empty"],
)
def test_failed_slice_raises_and_keeps_meta(raster_sg_file, item):
    cube = read_spectrograph_lvl2(raster_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"]

    with pytest.raises((IndexError, ValueError)):
        cube[item]
    assert cube.fits_wcs is not None


def test_synthetic_cube_pixel_scales_and_wavelength_axis():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), np.arange(10) * 0.02 * u.nm + 140 * u.nm)

    assert cube.wavelength_axis == 2
    assert cube.spectral_dispersion.unit.is_equivalent(u.nm)
    assert u.isclose(cube.spectral_dispersion, 0.02 * u.nm, rtol=0.01)
    assert cube.solid_angle.unit.is_equivalent(u.sr)
    assert u.isclose(cube.solid_angle.to(u.sr), (1.0 * u.arcsec * SLIT_WIDTH).to(u.sr), rtol=0.01)
    assert "Unknown -- Unknown" in str(cube)


@pytest.mark.parametrize(
    ("shape", "axes", "attribute", "match"),
    [
        ((2, 5), SPATIAL_AXES, "wavelength_axis", "wavelength axis"),
        ((2, 5), SPATIAL_AXES, "spectral_dispersion", "no WAVE ctype"),
        ((1, 2, 5), NO_LATITUDE_AXES, "solid_angle", "no HPLT ctype"),
    ],
)
def test_missing_wcs_axis_raises(shape, axes, attribute, match):
    header = {"NAXIS": len(shape)}
    for i, (ctype, cunit, cdelt, crval) in enumerate(axes, start=1):
        header |= {f"NAXIS{i}": shape[-i], f"CTYPE{i}": ctype, f"CUNIT{i}": cunit}
        header |= {f"CDELT{i}": cdelt, f"CRVAL{i}": crval, f"CRPIX{i}": 1}
    cube = SpectrogramCube(np.ones(shape), wcs=WCS(header), uncertainty=None, unit=u.DN, meta={}, mask=None)

    with pytest.raises(ValueError, match=match):
        getattr(cube, attribute)


def test_pixel_scales_need_a_fits_wcs(combined_si_iv):
    cube = SpectrogramCube(combined_si_iv.data, wcs=combined_si_iv.wcs, uncertainty=None, unit=u.DN, meta={})

    with pytest.raises(ValueError, match="no FITS WCS"):
        _ = cube.spectral_dispersion


def test_pixel_scales_real_data(sns_sg_file):
    cube = read_files(sns_sg_file)["Si IV 1403"]

    assert cube.spectral_dispersion.unit.is_equivalent(u.nm)
    assert cube.spectral_dispersion.value > 0
    assert cube.solid_angle.unit.is_equivalent(u.sr)
    assert cube.solid_angle.value > 0


@pytest.mark.parametrize(("files", "bins"), [("sns_sg_file", (1, 2, 1)), ("raster_sg_files", (1, 2, 1, 1))])
def test_rebinned_cube_keeps_native_pixel_scales(request, files, bins):
    cube = read_files(request.getfixturevalue(files), spectral_windows="Si IV 1403")["Si IV 1403"]
    rebinned = cube.rebin(bins)

    assert rebinned.spectral_dispersion == cube.raster_slice(0).spectral_dispersion
    assert rebinned.solid_angle == cube.raster_slice(0).solid_angle


@pytest.mark.parametrize(
    "derive",
    [
        lambda c: c[1:3],
        lambda c: c[1],
        lambda c: c.rebin((2, 1, 1)),
        lambda c: c.crop([None, c.wcs.array_index_to_world(1, 2, 0)[1]]),
    ],
    ids=["slice", "index", "rebin", "crop"],
)
def test_fits_wcs_cube_keeps_native_pixel_scales(derive):
    cube = make_test_spectrogram_cube(np.ones((4, 5, 6)), np.linspace(1400, 1405, 6) * u.AA)

    assert derive(cube).spectral_dispersion == cube.spectral_dispersion
    assert derive(cube).solid_angle == cube.solid_angle


@pytest.mark.parametrize(
    ("item", "kwargs", "cmaps"),
    [(0, {}, ["viridis"]), (0, {"cmap": "gray"}, ["gray"]), ((0, 20), {"cmap": "gray"}, [])],
    ids=["default", "cmap", "1d_drops_cmap"],
)
def test_spectrogram_cube_plot_defaults(sns_sg_file, item, kwargs, cmaps):
    ax = read_files(sns_sg_file)["Si IV 1403"][item].plot(**kwargs)
    assert [image.get_cmap().name for image in ax.images] == cmaps
    plt.close(ax.figure)


def test_raster_collection_aligned_axis_physical_types_are_sorted(sns_sg_file):
    # NDCollection builds these from sets, so without sorting the order changes between runs.
    types = read_files(sns_sg_file).aligned_axis_physical_types
    assert any(len(axis_types) > 1 for axis_types in types)
    assert types == [tuple(sorted(axis_types)) for axis_types in types]


def test_spectrogram_cube_from_data_and_wcs():
    # For example, a map of fitted line parameters with the WCS of the cube they came from.
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLT-TAN", "HPLN-TAN"]
    cube = SpectrogramCube(np.ones((4, 5)) * u.km / u.s, wcs)
    assert cube.unit == u.km / u.s
    assert cube.uncertainty is None
    assert str(cube)
