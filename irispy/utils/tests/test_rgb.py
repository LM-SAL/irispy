import copy

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SpectralCoord

from irispy.io.utils import read_files
from irispy.tests.helpers import figure_test, make_test_spectrogram_cube
from irispy.utils.constants import DN_UNIT
from irispy.utils.rgb import calculate_rgb, plot_rgb
from irispy.visualization import LON_AXIS_LABEL

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
    cube = make_test_spectrogram_cube(100 * data[:, np.newaxis, :], wavelengths)
    cube.meta["TWAVE1"] = si_iv_rest.to_value(u.AA)
    return cube


def test_calculate_rgb_shape_and_range(si_iv_cube):
    rgb, (intensity, wavelength, rgb_colorbar) = calculate_rgb(si_iv_cube)
    assert rgb.shape == (*si_iv_cube.shape[:-1], 3)
    assert np.nanmin(rgb) >= 0
    assert np.nanmax(rgb) <= 1
    assert rgb_colorbar.shape == (intensity.shape[0], si_iv_cube.shape[-1], 3)
    assert wavelength.shape == intensity.shape
    assert wavelength.unit.is_equivalent(u.AA)


def test_calculate_rgb_leaves_spectra_without_data_blank(si_iv_cube):
    """
    A spectrum with no unmasked, finite sample within ``velocity_range`` is NaN, not black
    like a spectrum without signal: the 748 spectra of this sit-and-stare without data, and
    one whose samples within the range are masked.
    """
    no_data = np.all(si_iv_cube.mask | ~np.isfinite(si_iv_cube.data), axis=-1)
    assert no_data.sum() == 748
    lower, upper = ([-100, 100] * u.km / u.s).to(u.AA, equivalencies=u.doppler_optical(si_iv_cube.meta.rest_wavelength))
    wavelength = si_iv_cube.spectral_axis.to(u.AA)
    si_iv_cube.mask[100, 20, (wavelength >= lower) & (wavelength <= upper)] = True
    assert not no_data[100, 20]
    no_data[100, 20] = True
    rgb, _ = calculate_rgb(si_iv_cube)
    np.testing.assert_array_equal(np.isnan(rgb), np.repeat(no_data[..., np.newaxis], 3, axis=-1))
    assert np.all(np.isfinite(calculate_rgb(si_iv_cube, velocity_range=None)[0][100, 20]))


def test_calculate_rgb_metadata_defaults(si_iv_cube):
    expected, (_, _, expected_colorbar) = calculate_rgb(
        si_iv_cube, velocity_range=[-100, 100] * u.km / u.s, rest_wavelength=si_iv_cube.meta.rest_wavelength
    )
    actual, (_, _, colorbar) = calculate_rgb(si_iv_cube)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(colorbar, expected_colorbar)


@pytest.mark.parametrize("render", [calculate_rgb, plot_rgb])
def test_rgb_requires_rest_wavelength_without_metadata(si_iv_cube, render):
    si_iv_cube.meta.pop(f"TWAVE{si_iv_cube.meta._iwin}")
    with pytest.raises(ValueError, match="Pass rest_wavelength explicitly"):
        render(si_iv_cube)


@pytest.mark.parametrize("render", [calculate_rgb, plot_rgb])
def test_rgb_explicit_rest_wavelength_without_metadata(si_iv_cube, render):
    rest_wavelength = si_iv_cube.meta.pop(f"TWAVE{si_iv_cube.meta._iwin}") * u.AA
    render(si_iv_cube, rest_wavelength=rest_wavelength)
    plt.close("all")


def test_rgb_defaults_to_the_line_the_cube_covers(raster_sg_file):
    """
    A crop of the Mg II k window to Mg II h is colored around Mg II h, and a crop ending
    in the pixel that holds Mg II k, whose center is short of it, around Mg II k.
    """
    cube = read_files(raster_sg_file)["Mg II k 2796"][0]
    for limits, rest_wavelength in (((2802, 2805), 2803.53 * u.AA), ((2793, 2796.352), 2796.352 * u.AA)):
        crop = cube.crop(*([SpectralCoord(wavelength * u.AA), None, None, None] for wavelength in limits))
        expected, _ = calculate_rgb(crop, rest_wavelength=rest_wavelength)
        np.testing.assert_allclose(calculate_rgb(crop)[0], expected)
        fig, ax = plt.subplots()
        plot_rgb(crop, ax=ax)
        np.testing.assert_allclose(_velocity_axis(fig).get_ylim(), (-100, 100))
        plt.close(fig)


@pytest.mark.parametrize("render", [calculate_rgb, plot_rgb])
def test_rgb_rejects_a_velocity_range_outside_the_window(si_iv_cube, render):
    """
    The window ends 630 km/s to the red of Si IV 1402.77, so this range would be black.
    """
    with pytest.raises(
        ValueError, match=r"velocity_range=\[700\. 800\.\] km / s of rest_wavelength=1402\.770 Angstrom; pass"
    ):
        render(si_iv_cube, velocity_range=[700, 800] * u.km / u.s)


@pytest.mark.parametrize(
    ("velocity_range", "error"),
    [
        ((-100, 100), TypeError),
        ([-100, 100] * u.AA, u.UnitsError),
        ([100, -100] * u.km / u.s, ValueError),
        (100 * u.km / u.s, ValueError),
    ],
)
def test_calculate_rgb_rejects_an_invalid_velocity_range(si_iv_cube, velocity_range, error):
    with pytest.raises(error, match="velocity_range"):
        calculate_rgb(si_iv_cube, velocity_range=velocity_range)


def test_calculate_rgb_intensity_limits_in_dn(si_iv_cube):
    expected, _ = calculate_rgb(si_iv_cube, vmin=5 * si_iv_cube.unit, vmax=100 * si_iv_cube.unit)
    result, _ = calculate_rgb(si_iv_cube, vmin=5 * u.DN, vmax=100 * u.DN)
    np.testing.assert_allclose(result, expected)
    for name in ("vmin", "vmax"):
        with pytest.raises(TypeError, match=name):
            calculate_rgb(si_iv_cube, **{name: 100})


def test_plot_rgb_rejects_a_rest_wavelength_without_units(si_iv_cube):
    with pytest.raises(TypeError, match="rest_wavelength"):
        plot_rgb(si_iv_cube, rest_wavelength=1402.77)


def test_calculate_rgb_stretch_survives_data_below_vmin(si_iv_cube):
    """
    `numpy.sqrt` would return NaN for the samples that ``vmin`` maps below zero.
    """
    rgb, _ = calculate_rgb(si_iv_cube, vmin=np.nanmedian(si_iv_cube.data) * si_iv_cube.unit, stretch=np.sqrt)
    np.testing.assert_array_equal(np.isnan(rgb), np.isnan(calculate_rgb(si_iv_cube)[0]))


def test_calculate_rgb_colors_the_short_wavelength_line_blue(shifted_line_cube):
    rgb, _ = calculate_rgb(shifted_line_cube)
    blue_shifted, red_shifted = rgb[0, 0], rgb[1, 0]
    assert blue_shifted[2] > blue_shifted[0]
    assert red_shifted[0] > red_shifted[2]


def test_calculate_rgb_rejects_a_cube_that_is_not_three_dimensional(si_iv_cube):
    with pytest.raises(ValueError, match="three dimensional cube"):
        calculate_rgb(si_iv_cube[0])


def test_calculate_rgb_rejects_unscaled_data():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm, unit=DN_UNIT["NUV_UNSCALED"])
    with pytest.raises(ValueError, match="unscaled"):
        calculate_rgb(cube)


def test_calculate_rgb_entirely_masked_cube():
    """
    Without a finite sample there is no percentile; an explicit ``vmax`` still works.
    """
    wavelengths = 1402.77 * u.AA + np.linspace(-2, 2, 41) * u.AA
    cube = make_test_spectrogram_cube(np.full((4, 4, wavelengths.size), np.nan), wavelengths)
    with pytest.raises(ValueError, match="no intensity range to map"):
        calculate_rgb(cube)
    rgb, _ = calculate_rgb(cube, vmax=1 * u.DN)
    assert np.all(np.isnan(rgb))


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


@pytest.mark.parametrize(
    "overrides",
    [
        lambda cube: {"rest_wavelength": cube.meta.rest_wavelength + 0.3 * u.AA},
        lambda _: {"velocity_norm": lambda v: np.arcsinh(v.to_value(u.km / u.s) / 25)},
    ],
    ids=["rest_wavelength", "velocity_norm"],
)
def test_plot_rgb_options_update_image_and_colorbar(si_iv_cube, overrides):
    kwargs = {"vmax": 100 * u.DN, **overrides(si_iv_cube)}
    rgb, (_, _, colorbar) = calculate_rgb(si_iv_cube, **kwargs)
    default_rgb, (_, _, default_colorbar) = calculate_rgb(si_iv_cube, vmax=100 * u.DN)
    assert not np.allclose(rgb, default_rgb, equal_nan=True)
    assert not np.allclose(colorbar, default_colorbar)

    fig, (ax, cax) = plt.subplots(ncols=2)
    plot_rgb(si_iv_cube, ax=ax, cax=cax, **kwargs)
    np.testing.assert_allclose(ax.collections[0].get_array(), rgb)
    np.testing.assert_allclose(cax.collections[0].get_array(), colorbar)
    plt.close(fig)


def test_plot_rgb_whole_window(si_iv_cube):
    """
    ``velocity_range=None`` maps the ends of the window, here a slice that ends 3.7 Å
    short of the rest wavelength.
    """
    si_iv_cube = si_iv_cube[:, :, :3]
    wavelength = si_iv_cube.spectral_axis.to(u.AA)
    assert si_iv_cube.meta.rest_wavelength > wavelength.max()
    with pytest.raises(ValueError, match="No wavelength of the window"):
        calculate_rgb(si_iv_cube)
    expected, _ = calculate_rgb(si_iv_cube, velocity_range=None)
    assert np.any(expected > 0)
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, velocity_range=None)
    np.testing.assert_allclose(ax.collections[0].get_array(), expected)
    cax = next(a for a in fig.axes if a.get_ylabel().startswith("Wavelength"))
    np.testing.assert_allclose(cax.get_ylim(), (wavelength.value.min(), wavelength.value.max()))
    assert _velocity_axis(fig) is not None
    plt.close(fig)


def test_plot_rgb_against_time(si_iv_cube):
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, coordinates="time")
    assert "Time" in ax.get_xlabel()
    plt.close(fig)


def test_plot_rgb_defaults_to_time_for_a_sit_and_stare(si_iv_cube, raster_sg_file):
    ax = plot_rgb(si_iv_cube)
    assert ax.get_xlabel() == "Time [UTC]"
    assert isinstance(ax.xaxis.get_major_formatter(), matplotlib.dates.ConciseDateFormatter)
    ax = plot_rgb(read_files(raster_sg_file)["Si IV 1403"][0])
    assert ax.get_xlabel() == LON_AXIS_LABEL
    plt.close("all")


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
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, rest_wavelength=1402.77 * u.AA, velocity_range=[-60, 40] * u.km / u.s)
    cax_velocity = _velocity_axis(fig)
    assert cax_velocity is not None
    np.testing.assert_allclose(cax_velocity.get_ylim(), (-60, 40))
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


def _colorbar_geometry(cube, **kwargs):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_rgb(cube, ax=ax, vmax=100 * u.DN, **kwargs)
    fig.canvas.draw()
    cax = next(a for a in fig.axes if a.get_ylabel().startswith("Wavelength"))
    width = cax.get_window_extent().width
    gap = cax.get_window_extent().x0 - ax.get_window_extent().x1
    plt.close(fig)
    return width, gap


def test_plot_rgb_colorbar_fraction_and_pad(si_iv_cube):
    """
    `cbar_fraction` widens the colorbar, `cbar_pad` the gap before it.
    """
    narrow, _ = _colorbar_geometry(si_iv_cube, cbar_fraction=0.1)
    wide, _ = _colorbar_geometry(si_iv_cube, cbar_fraction=0.3)
    assert narrow < wide
    _, tight = _colorbar_geometry(si_iv_cube, cbar_pad=0.05)
    _, loose = _colorbar_geometry(si_iv_cube, cbar_pad=0.5)
    assert tight < loose


def test_plot_rgb_colorbar_defaults_to_the_metadata_rest_wavelength(si_iv_cube):
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax)
    cax_velocity = _velocity_axis(fig)
    assert cax_velocity is not None
    np.testing.assert_allclose(cax_velocity.get_ylim(), (-100, 100))
    plt.close(fig)


def test_plot_rgb_without_a_velocity_axis(si_iv_cube):
    """
    ``rest_wavelength=False`` wins over the metadata.
    """
    fig, ax = plt.subplots()
    plot_rgb(si_iv_cube, ax=ax, rest_wavelength=False)
    assert _velocity_axis(fig) is None
    plt.close(fig)


@figure_test
def test_plot_rgb_figure():
    wavelengths = 1402.77 * u.AA + np.linspace(-1, 1, 81) * u.AA
    centers = 1402.77 + np.linspace(-0.5, 0.5, 21)
    intensity = np.linspace(5, 100, 16)
    profiles = np.exp(-(((wavelengths.to_value(u.AA)[None, :] - centers[:, None]) / 0.15) ** 2))
    cube = make_test_spectrogram_cube(intensity[None, :, None] * profiles[:, None, :], wavelengths)
    cube.meta["TWAVE1"] = 1402.77

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained")
    plot_rgb(cube, ax=axes[0], vmax=100 * u.DN)
    axes[0].set_title("Default: +/-100 km/s, linear velocity")
    plot_rgb(cube, ax=axes[1], vmax=100 * u.DN, stretch=np.sqrt, velocity_range=[-50, 50] * u.km / u.s)
    axes[1].set_title("+/-50 km/s, sqrt stretch")
    return fig


def test_calculate_rgb_blanks_clipped_samples(si_iv_cube):
    clipped, masked = copy.deepcopy(si_iv_cube), copy.deepcopy(si_iv_cube)
    clipped.data[0, 0, 5] = np.inf  # blank, like a masked sample
    masked.mask[0, 0, 5] = True
    np.testing.assert_array_equal(calculate_rgb(clipped)[0], calculate_rgb(masked)[0])
