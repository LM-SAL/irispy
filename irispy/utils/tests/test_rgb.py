import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u

from irispy.io.utils import read_files
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.rgb import calculate_rgb, plot_rgb

colorsynth = pytest.importorskip("colorsynth")


@pytest.fixture
def si_iv_cube(sns_sg_file):
    return read_files(sns_sg_file)["Si IV 1403"][0]


@pytest.fixture
def shifted_line_cube():
    """
    Two synthetic Si IV 1403 spectra, one shifted into each wing.
    """
    si_iv_rest = 1402.77 * u.AA
    wavelengths = si_iv_rest + np.linspace(-2, 2, 81) * u.AA
    centers = si_iv_rest + [-0.5, 0.5] * u.AA
    data = np.exp(-(((wavelengths - centers[:, np.newaxis]) / (0.15 * u.AA)) ** 2)).value
    return make_test_spectrogram_cube(100 * data[:, np.newaxis, :], wavelengths)


def test_calculate_rgb_shape_and_range(si_iv_cube):
    rgb, (intensity, wavelength, rgb_colorbar) = calculate_rgb(si_iv_cube)
    assert rgb.shape == (*si_iv_cube.shape[:-1], 3)
    assert np.all(rgb >= 0)
    assert np.all(np.isfinite(rgb))
    assert rgb_colorbar.shape == (intensity.shape[0], si_iv_cube.shape[-1], 3)
    assert wavelength.shape == intensity.shape
    assert wavelength.unit.is_equivalent(u.AA)


def test_calculate_rgb_ignores_wavelengths_outside_the_window(si_iv_cube):
    (wavelength,) = si_iv_cube.axis_world_coords("em.wl")
    wavelength = wavelength.to(u.AA)
    beyond = wavelength.max() + 10 * u.AA
    rgb, _ = calculate_rgb(si_iv_cube, wavelength_min=beyond, wavelength_max=beyond + 1 * u.AA)
    assert np.all(rgb == 0)


def test_calculate_rgb_vmax_accepts_a_quantity(si_iv_cube):
    expected, _ = calculate_rgb(si_iv_cube, vmax=100)
    result, _ = calculate_rgb(si_iv_cube, vmax=100 * si_iv_cube.unit)
    np.testing.assert_allclose(result, expected)


def test_calculate_rgb_stretch_survives_data_below_vmin(si_iv_cube):
    """
    `numpy.sqrt` would return NaN for the samples that ``vmin`` maps below zero.
    """
    rgb, _ = calculate_rgb(si_iv_cube, vmin=np.nanmedian(si_iv_cube.data), stretch=np.sqrt)
    assert np.all(np.isfinite(rgb))


def test_calculate_rgb_colors_the_short_wavelength_line_blue(shifted_line_cube):
    rgb, _ = calculate_rgb(shifted_line_cube)
    blue_shifted, red_shifted = rgb[0, 0], rgb[1, 0]
    assert blue_shifted[2] > blue_shifted[0]
    assert red_shifted[0] > red_shifted[2]


def test_calculate_rgb_rejects_a_cube_that_is_not_three_dimensional(si_iv_cube):
    with pytest.raises(ValueError, match="three dimensional cube"):
        calculate_rgb(si_iv_cube[0])


def test_calculate_rgb_entirely_masked_cube():
    """
    Without a finite sample there is no percentile; an explicit ``vmax`` still works.
    """
    wavelengths = 1402.77 * u.AA + np.linspace(-2, 2, 41) * u.AA
    cube = make_test_spectrogram_cube(np.full((4, 4, wavelengths.size), np.nan), wavelengths)
    with pytest.raises(ValueError, match="no intensity range to map"):
        calculate_rgb(cube)
    rgb, _ = calculate_rgb(cube, vmax=1)
    assert np.all(rgb == 0)


@pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
def test_plot_rgb_rejects_a_singleton_spatial_axis(shape):
    """
    One pixel across leaves no spacing to size the cells from.
    """
    wavelengths = 1402.77 * u.AA + np.linspace(-2, 2, 41) * u.AA
    data = np.random.default_rng(0).uniform(1, 100, size=(*shape, wavelengths.size))
    cube = make_test_spectrogram_cube(data, wavelengths)
    # The colors are still well defined, only the cell edges are not.
    rgb, _ = calculate_rgb(cube)
    assert rgb.shape == (*shape, 3)
    with pytest.raises(ValueError, match=r"only one pixel.*calculate_rgb"):
        plot_rgb(cube)


def test_plot_rgb(si_iv_cube):
    fig, (ax, ax_auto) = plt.subplots(ncols=2)
    result = plot_rgb(si_iv_cube, ax=ax)
    assert result is ax
    assert ax.get_aspect() == 1
    assert "Longitude" in ax.get_xlabel()
    plot_rgb(si_iv_cube, ax=ax_auto, aspect="auto")
    assert ax_auto.get_aspect() == "auto"
    plt.close(fig)


def test_plot_rgb_creates_its_own_figure(si_iv_cube):
    ax = plot_rgb(si_iv_cube)
    assert ax.get_figure().get_layout_engine() is not None
    assert ax.collections
    plt.close(ax.get_figure())


def test_plot_rgb_with_axes_outside_a_grid(si_iv_cube):
    """
    Axes from ``Figure.add_axes`` have no grid cell to split, so ``cax`` is required.
    """
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.1, 0.6, 0.8))
    assert ax.get_subplotspec() is None
    with pytest.raises(ValueError, match="pass `cax` explicitly"):
        plot_rgb(si_iv_cube, ax=ax)
    plt.close(fig)


def test_plot_rgb_without_a_rest_wavelength_in_the_metadata(si_iv_cube):
    # SGMeta reads TWAVE<iwin>, not TWAVE1.
    si_iv_cube.meta.pop(f"TWAVE{si_iv_cube.meta._iwin}")
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax)
    assert _velocity_axis(fig) is None
    plt.close(fig)


def test_plot_rgb_against_time(si_iv_cube):
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, coordinates="time")
    assert "Time" in ax.get_xlabel()
    assert ax.get_aspect() == "auto"
    plt.close(fig)


def test_plot_rgb_rejects_an_unknown_coordinate_choice(si_iv_cube):
    with pytest.raises(ValueError, match="must be 'helioprojective' or 'time'"):
        plot_rgb(si_iv_cube, coordinates="pixel")


def _velocity_axis(fig):
    # A secondary axis lives in its parent's child_axes, not in fig.axes, and only
    # takes on its limits at draw time.
    fig.canvas.draw()
    children = [child for ax in fig.axes for child in ax.child_axes]
    return next((ax for ax in children if "Velocity" in ax.get_ylabel()), None)


def test_plot_rgb_colorbar_velocity_axis(si_iv_cube):
    rest_wavelength = 1402.77 * u.AA
    doppler = u.doppler_optical(rest_wavelength)
    wavelength_min = (-60 * u.km / u.s).to(u.AA, equivalencies=doppler)
    wavelength_max = (40 * u.km / u.s).to(u.AA, equivalencies=doppler)

    fig, ax = plt.subplots()
    plot_rgb(
        si_iv_cube,
        ax=ax,
        rest_wavelength=rest_wavelength,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )
    cax_velocity = _velocity_axis(fig)
    assert cax_velocity is not None
    np.testing.assert_allclose(cax_velocity.get_ylim(), (-60, 40))
    plt.close(fig)


def test_plot_rgb_colorbar_zooms_to_the_mapped_range(si_iv_cube):
    fig, ax = plt.subplots()
    cax = fig.add_axes((0.8, 0.1, 0.05, 0.8))
    plot_rgb(si_iv_cube, ax=ax, cax=cax, wavelength_min=1399 * u.AA, wavelength_max=1399.2 * u.AA)
    np.testing.assert_allclose(cax.get_ylim(), (1399, 1399.2))
    plt.close(fig)


def test_plot_rgb_uses_the_given_cax(si_iv_cube):
    fig, ax = plt.subplots()
    cax = fig.add_axes((0.8, 0.1, 0.05, 0.8))
    before = list(fig.axes)
    plot_rgb(si_iv_cube, ax=ax, cax=cax)
    assert cax.collections, "the colorbar was not drawn into the given axes"
    # No axes were carved out; the velocity axis rides on the given cax.
    assert fig.axes == before
    assert _velocity_axis(fig) is not None
    plt.close(fig)


def test_plot_rgb_panels_do_not_collide(si_iv_cube):
    """
    ``axes_grid1`` hid the colorbar from the layout engine, so panels overlapped.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6), layout="constrained")
    panels = []
    for ax in axes:
        seen = set(map(id, fig.axes))
        plot_rgb(si_iv_cube, ax=ax)
        # Image, colorbar and its velocity axis all belong to this panel.
        new = [a for a in fig.axes if id(a) not in seen or a is ax]
        panels.append(new + [child for a in new for child in a.child_axes])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [
        [art.get_window_extent(renderer) for a in panel for art in (a.yaxis.label, a.xaxis.label) if art.get_text()]
        for panel in panels
    ]
    collisions = [
        (one, other)
        for panel, neighbour in itertools.combinations(labels, 2)
        for one in panel
        for other in neighbour
        if one.overlaps(other)
    ]
    assert not collisions
    plt.close(fig)


def test_plot_rgb_colorbar_matches_the_image_height(si_iv_cube):
    """
    A fixed aspect shrinks the image inside its cell; the colorbar must follow.
    """
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    plot_rgb(si_iv_cube, ax=ax, aspect="equal")
    fig.canvas.draw()
    cax = next(a for a in fig.axes if a.get_ylabel().startswith("Wavelength"))
    np.testing.assert_allclose(cax.get_window_extent().height, ax.get_window_extent().height)
    plt.close(fig)


def test_plot_rgb_colorbar_defaults_to_the_metadata_rest_wavelength(si_iv_cube):
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax)
    cax_velocity = _velocity_axis(fig)
    assert cax_velocity is not None
    # Wavelength limits converted through the metadata TWAVE.
    doppler = u.doppler_optical(si_iv_cube.meta.rest_wavelength)
    (wavelength,) = si_iv_cube.axis_world_coords("em.wl")
    expected = u.Quantity([wavelength.min(), wavelength.max()]).to_value(u.km / u.s, equivalencies=doppler)
    np.testing.assert_allclose(cax_velocity.get_ylim(), expected, rtol=1e-6)
    plt.close(fig)


def test_plot_rgb_colorbar_label_clears_the_image(si_iv_cube):
    """
    Regression: a small default ``cbar_pad`` pushed the wavelength label onto the image.
    """
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax)
    fig.canvas.draw()
    label = next(a for a in fig.axes if a.get_ylabel().startswith("Wavelength")).yaxis.label
    gap = label.get_window_extent(fig.canvas.get_renderer()).x0 - ax.get_window_extent().x1
    assert gap > 0
    plt.close(fig)


def test_plot_rgb_without_a_velocity_axis(si_iv_cube):
    """
    ``rest_wavelength=False`` wins over the metadata.
    """
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, rest_wavelength=False)
    assert _velocity_axis(fig) is None
    plt.close(fig)


def test_plotter_plot_rgb_matches_the_function(si_iv_cube):
    fig, (ax_plotter, ax_function) = plt.subplots(ncols=2)
    si_iv_cube.plotter.plot_rgb(ax=ax_plotter)
    plot_rgb(si_iv_cube, ax=ax_function)
    plotter, function = ax_plotter.collections[0], ax_function.collections[0]
    np.testing.assert_allclose(plotter.get_facecolors(), function.get_facecolors())
    plt.close(fig)
