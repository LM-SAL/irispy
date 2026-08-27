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
    A synthetic Si IV 1403 window whose two rows hold the line shifted to opposite
    wings.
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


def test_calculate_rgb_num_intensity(si_iv_cube):
    _, (intensity, _, rgb_colorbar) = calculate_rgb(si_iv_cube, num_intensity=25)
    assert intensity.shape[0] == 25
    assert rgb_colorbar.shape[0] == 25


def test_calculate_rgb_ignores_wavelengths_outside_the_window(si_iv_cube):
    """
    Wavelengths mapped outside the human visible range contribute no color.
    """
    (wavelength,) = si_iv_cube.axis_world_coords("em.wl")
    wavelength = wavelength.to(u.AA)
    beyond = wavelength.max() + 10 * u.AA
    rgb, _ = calculate_rgb(si_iv_cube, wavelength_min=beyond, wavelength_max=beyond + 1 * u.AA)
    assert np.all(rgb == 0)


def test_calculate_rgb_vmax_accepts_a_quantity(si_iv_cube):
    expected, _ = calculate_rgb(si_iv_cube, vmax=100)
    result, _ = calculate_rgb(si_iv_cube, vmax=100 * si_iv_cube.unit)
    np.testing.assert_allclose(result, expected)


def test_calculate_rgb_norm_survives_data_below_vmin(si_iv_cube):
    """
    `numpy.sqrt` would return NaN for the samples that ``vmin`` maps below zero.
    """
    rgb, _ = calculate_rgb(si_iv_cube, vmin=np.nanmedian(si_iv_cube.data), norm=np.sqrt)
    assert np.all(np.isfinite(rgb))


def test_calculate_rgb_colors_the_short_wavelength_line_blue(shifted_line_cube):
    """
    The short wavelength end of the window maps to blue and the long end to red.
    """
    rgb, _ = calculate_rgb(shifted_line_cube)
    blue_shifted, red_shifted = rgb[0, 0], rgb[1, 0]
    assert blue_shifted[2] > blue_shifted[0]
    assert red_shifted[0] > red_shifted[2]


def test_calculate_rgb_rejects_a_cube_that_is_not_three_dimensional(si_iv_cube):
    with pytest.raises(ValueError, match="three dimensional cube"):
        calculate_rgb(si_iv_cube[0])


def test_calculate_rgb_rejects_an_entirely_masked_cube():
    """
    Without a usable sample there is no percentile to scale by, and the image would
    otherwise come out uniformly black behind an "All-NaN slice" warning.
    """
    wavelengths = 1402.77 * u.AA + np.linspace(-2, 2, 41) * u.AA
    cube = make_test_spectrogram_cube(np.full((4, 4, wavelengths.size), np.nan), wavelengths)
    with pytest.raises(ValueError, match="no intensity range to map"):
        calculate_rgb(cube)


def test_calculate_rgb_accepts_an_entirely_masked_cube_with_an_explicit_vmax():
    wavelengths = 1402.77 * u.AA + np.linspace(-2, 2, 41) * u.AA
    cube = make_test_spectrogram_cube(np.full((4, 4, wavelengths.size), np.nan), wavelengths)
    rgb, _ = calculate_rgb(cube, vmax=1)
    assert np.all(rgb == 0)


@pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
def test_plot_rgb_rejects_a_singleton_spatial_axis(shape):
    """
    One pixel across leaves no spacing to measure a cell width from, so refuse to invent
    one and point at the function that works without a plot.
    """
    wavelengths = 1402.77 * u.AA + np.linspace(-2, 2, 41) * u.AA
    data = np.random.default_rng(0).uniform(1, 100, size=(*shape, wavelengths.size))
    cube = make_test_spectrogram_cube(data, wavelengths)
    # The colors themselves are still well defined, only the cell edges are not.
    rgb, _ = calculate_rgb(cube)
    assert rgb.shape == (*shape, 3)
    with pytest.raises(ValueError, match=r"only one pixel.*calculate_rgb"):
        plot_rgb(cube)


def test_plot_rgb(si_iv_cube):
    fig, ax = plt.subplots()
    result = plot_rgb(si_iv_cube, ax=ax)
    assert result is ax
    assert ax.get_aspect() == 1
    assert "Longitude" in ax.get_xlabel()
    plt.close(fig)


def test_plot_rgb_method_matches_the_function(si_iv_cube):
    fig, (ax_method, ax_function) = plt.subplots(ncols=2)
    si_iv_cube.plot_rgb(ax=ax_method)
    plot_rgb(si_iv_cube, ax=ax_function)
    method, function = ax_method.collections[0], ax_function.collections[0]
    np.testing.assert_allclose(method.get_facecolors(), function.get_facecolors())
    plt.close(fig)


@pytest.mark.remote_data
def test_rgb_on_a_complete_si_iv_window(remote_archive_sunspot_tar):
    """
    The packaged windows are truncated slices that miss their own line, so check a full
    Si IV 1403 window that actually brackets the 1402.77 A rest wavelength.
    """
    cube = read_files(remote_archive_sunspot_tar, spectral_windows="Si IV 1403")["Si IV 1403"][0]
    # The full OBS is 64 x 771 x 536; a cutout keeps the peak memory reasonable.
    cube = cube[:, 200:500]
    (wavelength,) = cube.axis_world_coords("em.wl")
    wavelength = wavelength.to(u.AA)
    assert wavelength.min() < 1402.77 * u.AA < wavelength.max()

    doppler = u.doppler_optical(1402.77 * u.AA)
    rgb, _ = calculate_rgb(
        cube,
        wavelength_min=(-100 * u.km / u.s).to(u.AA, equivalencies=doppler),
        wavelength_max=(100 * u.km / u.s).to(u.AA, equivalencies=doppler),
        norm=np.sqrt,
    )
    assert rgb.shape == (*cube.shape[:-1], 3)
    assert np.all(np.isfinite(rgb))
    # Restricting to the line core has to leave color behind, not a gray image.
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    assert np.nanmax(np.abs(red - blue)) > 0.1
    assert np.nanmax(green) > 0.1

    fig, ax = plt.subplots()
    plot_rgb(cube, ax=ax, norm=np.sqrt)
    assert "Longitude" in ax.get_xlabel()
    plt.close(fig)


def test_plot_rgb_aspect(si_iv_cube):
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, aspect="auto")
    assert ax.get_aspect() == "auto"
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
    return next((ax for ax in fig.axes if "Velocity" in ax.get_ylabel()), None)


def test_plot_rgb_colorbar_velocity_axis(si_iv_cube):
    """
    The velocity axis has to agree with the wavelength axis beside it.
    """
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
    """
    The colorbar shows the wavelengths that carry color, not the whole window.
    """
    fig, ax = plt.subplots()
    cax = fig.add_axes((0.8, 0.1, 0.05, 0.8))
    plot_rgb(si_iv_cube, ax=ax, cax=cax, wavelength_min=1399 * u.AA, wavelength_max=1399.2 * u.AA)
    np.testing.assert_allclose(cax.get_ylim(), (1399, 1399.2))
    plt.close(fig)


def test_plot_rgb_uses_the_given_cax(si_iv_cube):
    """
    An explicit ``cax`` is drawn into as given, with no extra axes carved out.
    """
    fig, ax = plt.subplots()
    cax = fig.add_axes((0.8, 0.1, 0.05, 0.8))
    before = list(fig.axes)
    plot_rgb(si_iv_cube, ax=ax, cax=cax)
    assert cax.collections, "the colorbar was not drawn into the given axes"
    # Only the velocity twin is new; the image axes were not re-homed into a grid.
    assert [a for a in fig.axes if a not in before] == [_velocity_axis(fig)]
    assert ax.get_position().bounds == before[0].get_position().bounds
    plt.close(fig)


def test_plot_rgb_uses_the_given_cax_without_a_velocity_axis(si_iv_cube):
    fig, ax = plt.subplots()
    cax = fig.add_axes((0.8, 0.1, 0.05, 0.8))
    plot_rgb(si_iv_cube, ax=ax, cax=cax, velocity=False)
    assert fig.axes == [ax, cax]
    plt.close(fig)


def test_plot_rgb_panels_do_not_collide(si_iv_cube):
    """
    The colorbar is carved out of the grid cell so the layout engine counts its labels.

    Stealing the space with ``axes_grid1`` hid them, and the labels of neighbouring
    panels then overlapped.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6), layout="constrained")
    panels = []
    for ax in axes:
        seen = set(map(id, fig.axes))
        plot_rgb(si_iv_cube, ax=ax)
        # The image, its colorbar and the velocity twin all belong to this panel.
        panels.append([a for a in fig.axes if id(a) not in seen or a is ax])
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
    A fixed aspect ratio shrinks the image inside its cell, and the colorbar has to
    follow it down instead of keeping the full height of the cell.
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
    # This window is truncated well blueward of the line, so every velocity is negative.
    assert max(cax_velocity.get_ylim()) < 0
    plt.close(fig)


def test_plot_rgb_colorbar_label_clears_the_image(si_iv_cube):
    """
    The wavelength ticks and label live in the gap between the image and the colorbar,
    and too small a ``cbar_pad`` used to push the label onto the image.
    """
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax)
    fig.canvas.draw()
    label = next(a for a in fig.axes if a.get_ylabel().startswith("Wavelength")).yaxis.label
    gap = label.get_window_extent(fig.canvas.get_renderer()).x0 - ax.get_window_extent().x1
    assert gap > 0
    plt.close(fig)


def test_plot_rgb_without_a_velocity_axis(si_iv_cube):
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, velocity=False)
    assert _velocity_axis(fig) is None
    plt.close(fig)
