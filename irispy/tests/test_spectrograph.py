import io
import re
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.utils import read_files
from irispy.tests.helpers import figure_test
from irispy.utils.constants import BAD_PIXEL_VALUE_UNSCALED

LON, LAT = "custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"


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
            np.testing.assert_array_almost_equal(raster[hdulist[0].header[f"TDESC{ext}"]][0].data, hdulist[ext].data)


def test_raster_of_one_file_indexes_as_in_0_9_1(sns_sg_file):
    # 0.9.1 gave a sequence with one raster for each file, so raster[window][0] was the raster
    window = read_files(sns_sg_file)["Si IV 1403"]
    raster = window[0]
    assert window.shape == (1, *raster.shape)
    assert len(list(window)) == 1
    assert raster.data.ndim == 3
    assert raster.wavelength_axis == 2
    assert raster.time.shape == raster.meta["exposure time"].shape == raster.shape[:1]
    assert raster.fits_wcs.pixel_n_dim == 3


def test_time_is_the_exposure_midpoint_of_any_slice(sns_sg_file, combined_si_iv):
    cube = read_files(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"][0]
    midpoints = cube.meta["auxiliary times"] + cube.meta["exposure time"] / 2
    assert_quantity_allclose((cube.time - midpoints).to(u.s), 0 * u.s, atol=1 * u.us)
    for piece, times in ((cube[:, 5], cube.time), (cube[1:3, 4:9, 2:20], cube.time[1:3]), (cube[0:1], cube.time[:1])):
        assert_quantity_allclose((piece.time - times).to(u.s), 0 * u.s, atol=1 * u.ns)
    # One exposure, of one raster file or of several read together
    for piece, time in ((cube[-1], cube.time[-1]), (combined_si_iv[1, 3], combined_si_iv.time[1, 3])):
        assert piece.time.shape == ()
        assert piece.time == time
        assert f"{time.isot} -- {time.isot}" in str(piece)
    assert combined_si_iv[1][3].time == combined_si_iv[1, 3, 5].time == combined_si_iv.time[1, 3]
    # A whole cube gives the times its gWCS gives
    assert (combined_si_iv.time == combined_si_iv.axis_world_coords("time")[0]).all()


def _wcs_summary(wcs):
    return wcs.pixel_n_dim, wcs.pixel_shape, wcs.pixel_to_world_values(*[1] * wcs.pixel_n_dim)


@pytest.mark.parametrize(
    ("sliced", "rasters", "item"),
    [
        (lambda c: c[2, 4], 2, (4,)),
        (lambda c: c[1:3, 2:4], slice(1, 3), (slice(2, 4),)),
        (lambda c: c[:, 2:6][2], 2, (slice(2, 6),)),
        (lambda c: c[:, 2:5, 10:60][1], 1, (slice(2, 5), slice(10, 60))),
        (lambda c: c[1, 3, 10:20, 5], 1, (3, slice(10, 20), 5)),
    ],
    ids=["raster_and_step", "ranges", "chained_ranges_then_raster", "chained_two_axes", "integers_and_a_range"],
)
def test_fits_wcs_follows_slicing(combined_si_iv, sliced, rasters, item):
    # The FITS WCS of each raster left, sliced as the cube is after the raster axis
    cube = sliced(combined_si_iv)
    roots = combined_si_iv.fits_wcs[rasters]
    if isinstance(roots, list):
        assert len(cube.fits_wcs) == len(roots)
        pairs, n_dim = zip(cube.fits_wcs, roots, strict=True), cube.wcs.pixel_n_dim - 1  # the raster axis stays
    else:
        pairs, n_dim = [(cube.fits_wcs, roots)], cube.wcs.pixel_n_dim
    for wcs, root in pairs:
        assert wcs.pixel_n_dim == n_dim
        np.testing.assert_equal(_wcs_summary(wcs), _wcs_summary(root.slice(item, numpy_order=True)))


def test_spectrogram_cube_crop_slices_fits_wcs(combined_si_iv):
    cube = combined_si_iv[0]
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


def test_memmap_rasters_read_together_hold_the_raw_fits_integers(raster_sg_files):
    cube = read_spectrograph_lvl2(raster_sg_files, memmap=True)["Si IV 1403"]
    raster0 = cube[0]
    expected = _raw_si_iv(raster_sg_files[0])

    np.testing.assert_array_equal(raster0.data.compute(), expected)
    np.testing.assert_array_equal(
        (raster0.data == BAD_PIXEL_VALUE_UNSCALED).compute(), expected == BAD_PIXEL_VALUE_UNSCALED
    )
    np.testing.assert_array_equal(raster0[3, 50, :].data.compute(), expected[3, 50])
    np.testing.assert_array_equal(cube.data[0, 0, 0].compute(), expected[0, 0])


@pytest.fixture
def mg_ii_raster(raster_sg_file):
    return read_spectrograph_lvl2(raster_sg_file, spectral_windows="Mg II k 2796")["Mg II k 2796"][0]


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
        ([LON, LAT, "time"], {LON: "b", LAT: "l"}, ["raster_step"]),
        ([LON, None, None], {LON: "b"}, [LAT, "time", "raster_step"]),
        ([LON, "time", None], {LON: "b"}, [LAT, "raster_step"]),
    ],
    ids=["default", "longitude_latitude_and_time", "longitude_only", "longitude_and_time"],
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
        assert coords["time"].get_axislabel() == "Time (UTC) [$\\mathrm{s}$]"
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
    ("plot_axes", "axes_coordinates", "edges", "time_shown"),
    [
        (["y", "x"], None, {LAT: "b", LON: "l"}, False),
        (None, None, {LON: "b", LAT: "l"}, False),
        (["x", "y"], [LON, LAT, "time"], {LON: "b", LAT: "l"}, True),
    ],
    ids=["slit_on_x", "step_on_x_by_default", "time_requested"],
)
def test_raster_image_labels_latitude_on_slit_edge_and_longitude_on_step_edge(
    mg_ii_raster, mg_ii_image, plot_axes, axes_coordinates, edges, time_shown
):
    # The time and raster step are shown only when asked for, and longitude and latitude keep their edges
    assert mg_ii_image.shape == mg_ii_raster.shape[:2]
    fig = plt.figure()
    ax = mg_ii_image.plot(plot_axes=plot_axes, axes_coordinates=axes_coordinates)
    fig.canvas.draw()

    for name, edge in edges.items():
        assert ax.coords[name].get_ticklabel_position() == [edge]
    time = ax.coords[list(ax.wcs.world_axis_physical_types).index("time")]
    assert time.get_ticklabel_visible() == time_shown
    assert not ax.coords["raster_step"].get_ticklabel_visible()
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
def test_combined_raster_animation_slider_labels(combined_si_iv, kwargs, labels):
    fig = plt.figure()
    assert combined_si_iv.plot(fig=fig, vmin=0, vmax=1000, **kwargs).slider_labels == labels
    plt.close(fig)


def test_spectrogram_cube_rejects_fancy_indexing(combined_si_iv):
    with pytest.raises(IndexError, match="only integer or range slices"):
        combined_si_iv[np.array([0, 2, 4])]


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
def test_derived_combined_cube_keeps_raster_behavior(combined_si_iv, derive):
    out = derive(combined_si_iv)

    assert type(out) is type(combined_si_iv)
    assert out[0].shape == combined_si_iv[0].shape
    assert out[0].fits_wcs is not None


@pytest.fixture(params=["raster_sg_files", "sns_sg_file", "sns_sjicube_1330"])
def any_cube(request):
    source = request.getfixturevalue(request.param)
    if request.param == "sns_sjicube_1330":
        return source
    cube = read_spectrograph_lvl2(source, spectral_windows="Si IV 1403")["Si IV 1403"]
    # The raster of one file, or the cube of several
    return cube if request.param == "raster_sg_files" else cube[0]


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
        (lambda c: c[1:3][-1], lambda c: c[2]),
        (lambda c: c[-3:], lambda c: c[c.shape[0] - 3 :]),
        (lambda c: c[:-1][-2:], lambda c: c[c.shape[0] - 3 : c.shape[0] - 1]),
        (lambda c: c[:, -5:], lambda c: c[:, c.shape[1] - 5 :]),
    ],
    ids=["index", "nested_index", "slice", "nested_slice", "second_axis"],
)
def test_negative_indices_match_positive(any_cube, negative, positive):
    np.testing.assert_equal(_slice_summary(negative(any_cube)), _slice_summary(positive(any_cube)))


@pytest.mark.parametrize(
    ("item", "match"),
    [(-9, "out of range"), (np.s_[::2], "with a step"), (np.s_[5:2], "length-0 axis")],
    ids=["out_of_range", "step", "empty"],
)
def test_failed_slice_raises_and_keeps_meta(raster_sg_file, item, match):
    cube = read_spectrograph_lvl2(raster_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"]
    meta = cube.meta

    with pytest.raises(IndexError, match=match) as error:
        _ = cube[item]
    assert type(error.value) is IndexError  # not numpy's AxisError, which is also a ValueError
    assert cube.meta is meta


def test_length_one_step_axis(sns_sg_file):
    # The combined WCS of a length-1 step axis raises in ndcube
    cube = read_spectrograph_lvl2(sns_sg_file)["C II 1336"][0][0:1]
    assert cube.wavelength_axis == 2
    assert "('em.wl',)" in str(cube)


def test_a_map_has_no_wavelength_axis(sns_sg_file):
    image = read_files(sns_sg_file)["Si IV 1403"][0, :, :, 10]
    with pytest.raises(ValueError, match="wavelength axis"):
        _ = image.wavelength_axis


def test_pixel_scales_real_data(sns_sg_file):
    cube = read_files(sns_sg_file)["Si IV 1403"]

    assert cube.spectral_dispersion.unit.is_equivalent(u.nm)
    assert cube.spectral_dispersion.value > 0
    assert cube.solid_angle.unit.is_equivalent(u.sr)
    assert cube.solid_angle.value > 0


def test_rebinned_cube_keeps_native_pixel_scales_and_has_no_fits_wcs(sns_sg_file):
    cube = read_files(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"]
    rebinned = cube.rebin((1, 1, 2, 1))

    assert rebinned.spectral_dispersion == cube.spectral_dispersion
    assert rebinned.solid_angle == cube.solid_angle
    assert rebinned.fits_wcs is None


@pytest.mark.parametrize(
    ("item", "kwargs", "cmaps"),
    [(0, {}, ["irissjiFUV"]), (0, {"cmap": "gray"}, ["gray"]), ((0, 20), {"cmap": "gray"}, [])],
    ids=["default", "cmap", "1d_drops_cmap"],
)
def test_spectrogram_cube_plot_defaults(sns_sg_file, item, kwargs, cmaps):
    ax = read_files(sns_sg_file)["Si IV 1403"][0][item].plot(**kwargs)
    assert [image.get_cmap().name for image in ax.images] == cmaps
    plt.close(ax.figure)


@pytest.mark.parametrize(
    ("kwargs", "labels"),
    [({}, ["283.20", "283.35"]), ({"aspect": "auto"}, ["283.20", "283.25", "283.30", "283.35", "283.40"])],
)
def test_spectrogram_plot_labels_narrow_window(sns_sg_file, kwargs, labels):
    # The 2832 window spans 0.23 nm, which the default equal aspect draws 92 pixels wide
    ax = read_files(sns_sg_file)["2832"][0][0].plot(**kwargs)
    svg = io.StringIO()
    with mpl.rc_context({"svg.fonttype": "none"}):
        ax.figure.savefig(svg, format="svg")
    plt.close(ax.figure)
    assert re.findall(r">(283\.\d+)</text>", svg.getvalue()) == labels


@pytest.mark.parametrize(
    ("file_fixture", "plot_axes", "expected"),
    [
        # The 2" steps of this raster are 12 times its 0.17" slit pixels: CDELT2 / CDELT3 is 0.0834
        ("bursts_si_iv_file", None, 0.0834),
        ("bursts_si_iv_file", ["y", "x"], 1 / 0.0834),
        # The steps of a sit-and-stare are times, so its maps keep square pixels
        ("sns_sg_file", None, 1),
    ],
)
def test_raster_map_is_drawn_with_its_true_proportions(request, file_fixture, plot_axes, expected):
    raster_map = read_files(request.getfixturevalue(file_fixture))["Si IV 1403"][0][:, :, 10]
    assert raster_map.plot(plot_axes=plot_axes).get_aspect() == pytest.approx(expected, rel=0.01)
    plt.close("all")


def test_image_along_the_raster_axis_fills_the_axes(combined_si_iv):
    # The rasters are times, often only a few, which square pixels would draw as a sliver
    assert combined_si_iv[:, 2, :, 10].plot().get_aspect() == "auto"
    plt.close("all")


@pytest.mark.parametrize("raster", ["r00066", "r00067"])
@pytest.mark.filterwarnings("ignore:Found 7 step:UserWarning")
def test_raster_map_with_a_data_gap_plots(raster):
    """
    None of the 8 steps of r00066 has an exposure, so none has a pointing, and r00067
    has one, too few to measure the step spacing from.
    """
    collection = read_files(
        get_test_filepath(f"data_gap/iris_l2_20140329_140938_3860258481_raster_t000_{raster}_test.fits")
    )
    for window in ("C II 1336", "2832"):
        assert collection[window][0][:, :, 5].plot().get_aspect() == 1
        plt.close("all")


@figure_test
def test_spectrogram_figure(sns_sg_file):
    # One exposure of the Si IV 1403 window: slit position against wavelength
    read_files(sns_sg_file)["Si IV 1403"][0][0].plot()


def test_raster_collection_aligned_axis_physical_types_are_sorted(sns_sg_file):
    # NDCollection builds these from sets, so without sorting the order changes between runs.
    types = read_files(sns_sg_file).aligned_axis_physical_types
    assert any(len(axis_types) > 1 for axis_types in types)
    assert types == [tuple(sorted(axis_types)) for axis_types in types]
