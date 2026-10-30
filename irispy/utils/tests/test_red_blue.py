import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.coordinates import SpectralCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty, VarianceUncertainty
from astropy.tests.helper import assert_quantity_allclose

import irispy.utils.red_blue as red_blue_module
from irispy.io.utils import read_files
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.tests.helpers import figure_test, make_test_spectrogram_cube, mark_saturated
from irispy.utils.constants import DN_UNIT, SATURATION_LIMIT
from irispy.utils.red_blue import RBAQualityFlag, calculate_red_blue_asymmetry
from irispy.utils.spectrograph import subtract_background

REST_WAVELENGTH = 140.277 * u.nm


def _wavelengths_from_velocity(velocity):
    return REST_WAVELENGTH * (1 + velocity / constants.c.to(u.km / u.s))


def _flat_wing_profile(velocity, *, red_excess=0, blue_excess=0):
    profile = np.ones(velocity.shape, dtype=float)
    profile[np.isclose(velocity.to_value(u.km / u.s), 0)] = 10
    red = (velocity >= 50 * u.km / u.s) & (velocity <= 150 * u.km / u.s)
    blue = (velocity >= -150 * u.km / u.s) & (velocity <= -50 * u.km / u.s)
    profile[red] += red_excess
    profile[blue] += blue_excess
    return profile


def test_calculate_red_blue_asymmetry_red_and_blue_signs():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    red_profile = _flat_wing_profile(velocity, red_excess=2)
    blue_profile = _flat_wing_profile(velocity, blue_excess=3)
    data = np.stack([red_profile, blue_profile]).reshape(1, 2, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)

    assert isinstance(result, RasterCollection)
    assert set(result.keys()) == {"red_blue_asymmetry", "quality", "interpolated_profile"}
    assert_quantity_allclose(result["red_blue_asymmetry"].data[0, 0] * u.one, 0.2 * u.one)
    assert_quantity_allclose(result["red_blue_asymmetry"].data[0, 1] * u.one, -0.3 * u.one)
    assert result["quality"].unit == u.dimensionless_unscaled
    assert RBAQualityFlag(result["quality"].data[0, 0]) is RBAQualityFlag.OK
    assert RBAQualityFlag(result["quality"].data[0, 0]).description == "ok"


def test_calculate_red_blue_asymmetry_uses_masked_bins():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=1000)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.mask = np.zeros(cube.shape, dtype=bool)
    red = (velocity >= 50 * u.km / u.s) & (velocity <= 150 * u.km / u.s)
    cube.mask[..., red] = True

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    assert_quantity_allclose(result["red_blue_asymmetry"].data[0, 0] * u.one, 0 * u.one, atol=1e-12 * u.one)


def test_calculate_red_blue_asymmetry_output_does_not_inherit_first_wavelength_mask():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.stack([profile, profile]).reshape(1, 2, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.mask = np.zeros(cube.shape, dtype=bool)
    cube.mask[0, 0, 0] = True
    cube.mask[0, 1, :] = True

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])
    assert not result["red_blue_asymmetry"].mask[0, 0]
    assert result["quality"].data[0, 0] == 0
    assert result["quality"].mask[0, 1]
    assert RBAQualityFlag(result["quality"].data[0, 1]) is RBAQualityFlag.NO_FINITE_DATA


def test_calculate_red_blue_asymmetry_requires_wavelength_axis():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm)
    cube.wcs.wcs.ctype[0] = "TIME"
    cube.wcs.wcs.cunit[0] = "s"
    cube.wcs.wcs.set()
    with pytest.raises(ValueError, match="Could not identify a spectral wavelength axis"):
        calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH)


def test_calculate_red_blue_asymmetry_rejects_unscaled_data():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm, unit=DN_UNIT["FUV_UNSCALED"])
    with pytest.raises(ValueError, match="unscaled"):
        calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH)


def test_calculate_red_blue_asymmetry_with_uncertainty():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths, uncertainty=StdDevUncertainty(np.full(data.shape, 0.1)))

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        spline_degree=1,
    )
    assert set(result) == {"red_blue_asymmetry", "quality", "interpolated_profile"}
    assert isinstance(result["red_blue_asymmetry"].uncertainty, StdDevUncertainty)
    numerator = 2
    peak = 10
    weight = np.r_[0.5, np.ones(9), 0.5]  # the bins at 50 and 150 km/s are half in the wing
    wing_error = 0.1 * np.sqrt(np.sum(weight**2)) / weight.sum()
    expected_propagated_error = np.sqrt((np.sqrt(2) * wing_error / peak) ** 2 + (numerator * 0.1 / peak**2) ** 2)
    assert_quantity_allclose(
        result["red_blue_asymmetry"].uncertainty.array[0, 0] * u.one, expected_propagated_error * u.one
    )
    assert result["red_blue_asymmetry"].meta["rba_rest_wavelength"] == 140.277
    assert result["red_blue_asymmetry"].meta["rba_spline_degree"] == 1

    symmetric_cube = make_test_spectrogram_cube(
        _flat_wing_profile(velocity).reshape(1, 1, -1),
        wavelengths,
        uncertainty=StdDevUncertainty(np.full(data.shape, 0.1)),
    )
    symmetric_result = calculate_red_blue_asymmetry(
        symmetric_cube,
        rest_wavelength=REST_WAVELENGTH,
        spline_degree=1,
    )
    expected_zero_error = np.sqrt(2) * wing_error / 10
    assert_quantity_allclose(symmetric_result["red_blue_asymmetry"].data[0, 0] * u.one, 0 * u.one)
    assert_quantity_allclose(
        symmetric_result["red_blue_asymmetry"].uncertainty.array[0, 0] * u.one, expected_zero_error * u.one
    )


@pytest.mark.parametrize(
    "uncertainty",
    [VarianceUncertainty(np.full((1, 1, 41), 0.01)), StdDevUncertainty(np.full((1, 1, 41), 100), unit=u.DN / 1000)],
)
def test_calculate_red_blue_asymmetry_error_from_other_uncertainties(uncertainty):
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    cube = make_test_spectrogram_cube(
        _flat_wing_profile(velocity, red_excess=2).reshape(1, 1, -1),
        _wavelengths_from_velocity(velocity),
        uncertainty=StdDevUncertainty(np.full((1, 1, velocity.size), 0.1)),
    )
    expected = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH)["red_blue_asymmetry"]
    cube.uncertainty = uncertainty
    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH)["red_blue_asymmetry"]
    np.testing.assert_allclose(result.uncertainty.array, expected.uncertainty.array)


def test_calculate_red_blue_asymmetry_flags_error_interpolation_failure(monkeypatch):
    original_make_interp_spline = red_blue_module.make_interp_spline

    def fail_on_error_spline(*args, **kwargs):
        if np.allclose(np.asarray(args[1]), 0.1):
            msg = "error spline failed"
            raise ValueError(msg)
        return original_make_interp_spline(*args, **kwargs)

    monkeypatch.setattr(red_blue_module, "make_interp_spline", fail_on_error_spline)
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    cube = make_test_spectrogram_cube(
        _flat_wing_profile(velocity, red_excess=2).reshape(1, 1, -1),
        wavelengths,
        uncertainty=StdDevUncertainty(np.full((1, 1, velocity.size), 0.1)),
    )

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        spline_degree=1,
    )
    assert RBAQualityFlag(result["quality"].data[0, 0]) is RBAQualityFlag.INTERP_FAILED
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_centers_profiles_on_peak():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = np.ones(velocity.shape, dtype=float)
    peak_wvl = _wavelengths_from_velocity(30 * u.km / u.s)
    sigma_wvl = abs(_wavelengths_from_velocity(10 * u.km / u.s) - peak_wvl).value
    profile += 9 * np.exp(-0.5 * ((wavelengths.value - peak_wvl.value) / sigma_wvl) ** 2)
    profile[(velocity >= 50 * u.km / u.s) & (velocity <= 150 * u.km / u.s)] += 2
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        velocity_range=(50, 150) * u.km / u.s,
        spline_degree=1,
    )
    assert result["red_blue_asymmetry"].data[0, 0] > 0
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])
    centered_profile = result["interpolated_profile"][0, 0]
    centered_velocity = centered_profile.axis_world_coords(0)[0]
    assert centered_velocity[np.nanargmax(centered_profile.data)].to_value(u.km / u.s) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("velocity_range", [(25, 75), (50, 150)])
def test_calculate_red_blue_asymmetry_symmetric_line_between_samples(velocity_range):
    # The 5.44 km/s samples of the Si IV gallery cutout, and lines shifted across one sample
    velocity = np.arange(-80, 81) * 5.44
    shift = 30 + 5.44 * np.linspace(0, 1, 5)[:, None, None]
    width = np.array([20, 30, 40])[:, None]
    data = 100 * np.exp(-(((velocity - shift) / width) ** 2))
    cube = make_test_spectrogram_cube(data, _wavelengths_from_velocity(velocity * u.km / u.s))

    result = calculate_red_blue_asymmetry(
        cube, rest_wavelength=REST_WAVELENGTH, velocity_range=velocity_range * u.km / u.s
    )
    assert np.abs(result["red_blue_asymmetry"].data).max() < 0.002


def test_calculate_red_blue_asymmetry_return_profiles():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.stack([profile, profile + 1]).reshape(1, 2, -1)
    cube = make_test_spectrogram_cube(data, wavelengths, uncertainty=StdDevUncertainty(np.full(data.shape, 0.1)))

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    interp_cube = result["interpolated_profile"]
    assert isinstance(interp_cube, SpectrogramCube)
    assert interp_cube.data.shape == (1, 2, 33)
    assert "spect.dopplerVeloc" in interp_cube.wcs.world_axis_physical_types
    assert interp_cube.uncertainty.array.shape == interp_cube.shape
    assert np.isfinite(interp_cube[0, 1].data).all()


def test_calculate_red_blue_asymmetry_return_profiles_on_sliced_cube():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.tile(profile, (2, 3, 1))
    cube = make_test_spectrogram_cube(data, wavelengths)[:, :1, :]

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    assert result["interpolated_profile"].shape == (2, 1, 33)
    assert_quantity_allclose(
        result["interpolated_profile"][0, 0].axis_world_coords(0)[0],
        np.arange(-160, 161, 10) * u.km / u.s,
    )


def test_calculate_red_blue_asymmetry_can_skip_profile_outputs():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    cube = make_test_spectrogram_cube(profile.reshape(1, 1, -1), wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        spline_degree=1,
        return_profiles=False,
    )
    assert set(result) == {"red_blue_asymmetry", "quality"}
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_continuum_subtraction():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = (profile + 5).reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result_without = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)

    continuum_wavelengths = (
        u.Quantity([[wavelengths[0].value, wavelengths[2].value], [wavelengths[-3].value, wavelengths[-1].value]])
        * wavelengths.unit
    )
    result_with = calculate_red_blue_asymmetry(
        subtract_background(cube, continuum_wavelengths, degree=0),
        rest_wavelength=REST_WAVELENGTH,
        spline_degree=1,
    )
    rba_without = result_without["red_blue_asymmetry"].data[0, 0]
    rba_with = result_with["red_blue_asymmetry"].data[0, 0]
    assert np.isfinite(rba_with)
    assert_quantity_allclose(rba_with * u.one, (2 / 9) * u.one)
    assert not np.isclose(rba_without, rba_with)


@figure_test
def test_calculate_red_blue_asymmetry_continuum_figure():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    cube = make_test_spectrogram_cube((profile + 5).reshape(1, 1, -1), wavelengths)
    continuum_wavelengths = (
        u.Quantity([[wavelengths[0].value, wavelengths[2].value], [wavelengths[-3].value, wavelengths[-1].value]])
        * wavelengths.unit
    )

    subtracted = subtract_background(cube, continuum_wavelengths, degree=0)

    result_without = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    result_with = calculate_red_blue_asymmetry(subtracted, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    rba_without = float(result_without["red_blue_asymmetry"].data[0, 0])
    rba_with = float(result_with["red_blue_asymmetry"].data[0, 0])
    assert_quantity_allclose(rba_without * u.one, (2 / 15) * u.one)
    assert_quantity_allclose(rba_with * u.one, (2 / 9) * u.one)

    velocities = velocity.to_value(u.km / u.s)
    v_low, v_high = 50, 150

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].plot(velocities, cube.data[0, 0], "o-", color="0.2", markersize=3, label="Input + continuum")
    axes[0].plot(velocities, subtracted.data[0, 0], "o-", color="C3", markersize=3, label="Continuum subtracted")
    axes[0].axvspan(-v_high, -v_low, color="C0", alpha=0.12)
    axes[0].axvspan(v_low, v_high, color="C3", alpha=0.12)
    axes[0].set(xlabel="Velocity [km/s]", ylabel="Intensity [DN]", title="Profiles")
    axes[0].legend(fontsize=8)

    axes[1].bar(["Raw", "Subtracted"], [rba_without, rba_with], color=["0.5", "C3"])
    axes[1].axhline(0, color="0.2", linewidth=0.8)
    axes[1].set(ylabel="Red-Blue Asymmetry", title="Continuum correction")
    axes[1].set_ylim(0, 0.26)

    return fig


def test_calculate_red_blue_asymmetry_flags_incomplete_wings():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = np.ones(velocity.shape, dtype=float)
    profile += 9 * np.exp(-0.5 * ((velocity.to_value(u.km / u.s) - 80) / 10) ** 2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        velocity_range=(50, 150) * u.km / u.s,
        spline_degree=1,
    )
    assert RBAQualityFlag(result["quality"].data[0, 0]) is RBAQualityFlag.INCOMPLETE_WINGS
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_flags_low_signal():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    cube = make_test_spectrogram_cube(_flat_wing_profile(velocity, red_excess=2).reshape(1, 1, -1), wavelengths)
    result = calculate_red_blue_asymmetry(
        cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1, min_intensity=15 * u.DN
    )
    assert RBAQualityFlag(result["quality"].data[0, 0]) is RBAQualityFlag.LOW_SIGNAL
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_min_intensity_in_dn(sns_sg_file):
    cube = read_files(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"][0]
    with pytest.raises(TypeError, match="min_intensity"):
        calculate_red_blue_asymmetry(cube, min_intensity=75, return_profiles=False)
    for data, threshold in ((cube, 75 * u.DN), (cube.apply_exposure_time_correction(), 75 * u.DN / u.s)):
        quality = calculate_red_blue_asymmetry(data, min_intensity=threshold, return_profiles=False)["quality"]
        expected = calculate_red_blue_asymmetry(data, min_intensity=75 * data.unit, return_profiles=False)
        assert 0 < np.count_nonzero(quality.data == RBAQualityFlag.LOW_SIGNAL) < quality.data.size
        np.testing.assert_array_equal(quality.data, expected["quality"].data)


def test_calculate_red_blue_asymmetry_flags_saturated():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    data = np.tile(_flat_wing_profile(velocity, red_excess=2), (1, 2, 1))
    mask = np.zeros(data.shape, dtype=bool)
    data[0, 0, 20] = data[0, 1, 5] = SATURATION_LIMIT.value
    mask[0, 1, 5] = True  # masked, but not marked, so not saturated
    cube = make_test_spectrogram_cube(data, wavelengths, mask=mask)
    saturated = np.zeros(data.shape, dtype=bool)
    saturated[0, 0, 20] = True  # the peak, marked as the readers mark clipped samples
    mark_saturated(cube, saturated)
    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    assert RBAQualityFlag(result["quality"].data[0, 0]) is RBAQualityFlag.SATURATED
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])
    assert RBAQualityFlag(result["quality"].data[0, 1]) is RBAQualityFlag.OK


def _varied_cube():
    """
    Six noisy Gaussians at different velocities, with gaps, a masked sample and a
    missing error.
    """
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 30, (2, 3, 1))
    data = 100 * np.exp(-(((velocity.to_value(u.km / u.s) - centers) / 25) ** 2)) + rng.normal(0, 1, (2, 3, 41))
    data[0, 1, 3] = np.nan  # a gap
    data[1, 2, 30] = -1  # dropped, so a gap too
    errors = np.sqrt(np.abs(data) + 1)
    errors[1, 0, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 1, 7] = True
    return make_test_spectrogram_cube(
        data, _wavelengths_from_velocity(velocity), uncertainty=StdDevUncertainty(errors), mask=mask
    )


def test_calculate_red_blue_asymmetry_pixels_are_independent():
    # Profiles without gaps share one spline per peak sample and those with gaps get their own,
    # so each pixel must come out as it does alone.
    cube = _varied_cube()
    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH)
    assert set(np.unique(result["quality"].data)) == {RBAQualityFlag.OK}
    for i, j in np.ndindex(2, 3):
        alone = calculate_red_blue_asymmetry(cube[i : i + 1, j : j + 1], rest_wavelength=REST_WAVELENGTH)
        for name in result:
            np.testing.assert_allclose(result[name].data[i, j], alone[name].data[0, 0], rtol=1e-12, err_msg=name)
        np.testing.assert_allclose(
            result["interpolated_profile"].uncertainty.array[i, j],
            alone["interpolated_profile"].uncertainty.array[0, 0],
            rtol=1e-12,
        )


@pytest.mark.parametrize("return_profiles", [True, False])
def test_calculate_red_blue_asymmetry_chunks_give_the_same_results(monkeypatch, return_profiles):
    cube = _varied_cube()
    # A peak at 120 km/s has no full red wing (fixture check below), so one pixel is flagged in its chunk
    cube.data[0, 2] = 100 * np.exp(-(((np.arange(-200, 201, 10) - 120) / 25) ** 2)) + 1
    kwargs = {"rest_wavelength": REST_WAVELENGTH, "return_profiles": return_profiles}
    expected = calculate_red_blue_asymmetry(cube, **kwargs)
    assert RBAQualityFlag(expected["quality"].data[0, 2]) is RBAQualityFlag.INCOMPLETE_WINGS
    monkeypatch.setattr(red_blue_module, "_CHUNK_ROWS", 4)
    result = calculate_red_blue_asymmetry(cube, **kwargs)
    assert list(result) == list(expected)
    for name in expected:
        np.testing.assert_array_equal(result[name].data, expected[name].data, err_msg=name)
        if expected[name].uncertainty is not None:
            np.testing.assert_array_equal(
                result[name].uncertainty.array, expected[name].uncertainty.array, err_msg=name
            )


def test_calculate_red_blue_asymmetry_real_cube_shape_and_coords(sns_sg_file):
    raster = read_files(sns_sg_file, uncertainty=True)
    cube = raster["C II 1336"][0]
    result = calculate_red_blue_asymmetry(cube, spline_degree=1)
    assert set(result) == {"red_blue_asymmetry", "quality", "interpolated_profile"}
    result["interpolated_profile"].meta["exposure time"][0] = 0 * u.s
    assert cube.meta["exposure time"][0] > 0 * u.s
    assert result["red_blue_asymmetry"].shape == cube.shape[:-1]
    np.testing.assert_array_equal(result["red_blue_asymmetry"].time.isot, cube.time.isot)
    error = result["red_blue_asymmetry"].uncertainty.array
    np.testing.assert_array_equal(np.isfinite(error), result["quality"].data == RBAQualityFlag.OK)
    for item in (np.s_[0:3], np.s_[:, 10:20]):
        for name, sliced in result[item].items():
            np.testing.assert_array_equal(sliced.data, result[name].data[item], err_msg=name)


def test_calculate_red_blue_asymmetry_rest_wavelength_required_without_meta():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(1, 6) * u.nm)
    del cube.meta["TWAVE1"]
    with pytest.raises(ValueError, match="rest_wavelength"):
        calculate_red_blue_asymmetry(cube, spline_degree=1)


def _crop(cube, limits):
    return cube.crop(*([SpectralCoord(wavelength * u.AA), None, None, None] for wavelength in limits))


def test_calculate_red_blue_asymmetry_default_rest_wavelength(raster_sg_file):
    """
    A crop of the Mg II k window to Mg II h is measured from Mg II h.
    """
    cube = _crop(read_files(raster_sg_file)["Mg II k 2796"][0], (2802, 2805))
    result = calculate_red_blue_asymmetry(cube, return_profiles=False)["red_blue_asymmetry"]
    expected = calculate_red_blue_asymmetry(cube, rest_wavelength=2803.53 * u.AA, return_profiles=False)
    assert_quantity_allclose(result.meta["rba_rest_wavelength"] * u.nm, 2803.53 * u.AA)
    np.testing.assert_allclose(result.data, expected["red_blue_asymmetry"].data, rtol=1e-12)


def test_calculate_red_blue_asymmetry_rejects_rest_wavelength_outside_cube(raster_sg_file):
    """
    A wing crop of the Mg II k window, which defaults to Mg II k, has no sample to take
    the peak from.
    """
    cube = _crop(read_files(raster_sg_file)["Mg II k 2796"][0], (2800, 2802))
    with pytest.raises(
        ValueError,
        match=r"No wavelength of the cube \(2799.91 to 2801.95 Å\) lies within velocity_range\[1\]=150.0 km/s "
        r"of rest_wavelength=2796.352 Å",
    ):
        calculate_red_blue_asymmetry(cube)


def test_calculate_red_blue_asymmetry_peak_within_velocity_range(v34_sg_file):
    """
    Mg II h measured in the Mg II k window, where Mg II k is brighter, gives what a crop
    to Mg II h gives, min_intensity included.
    """
    cube = read_files(v34_sg_file)["Mg II k 2796"][0]
    kwargs = {
        "rest_wavelength": 2803.53 * u.AA,
        "velocity_range": (25, 75) * u.km / u.s,
        "min_intensity": 300 * u.DN,
        "return_profiles": False,
    }
    result = calculate_red_blue_asymmetry(cube, **kwargs)
    expected = calculate_red_blue_asymmetry(_crop(cube, (2800, 2806)), **kwargs)
    assert set(np.unique(result["quality"].data)) == {RBAQualityFlag.OK, RBAQualityFlag.LOW_SIGNAL}
    np.testing.assert_array_equal(result["quality"].data, expected["quality"].data)
    np.testing.assert_allclose(result["red_blue_asymmetry"].data, expected["red_blue_asymmetry"].data, rtol=1e-10)


@pytest.mark.parametrize("velocity_range", [(25, 75), (50, 150)])
def test_calculate_red_blue_asymmetry_divides_by_profile_at_peak(v34_sg_file, velocity_range):
    """
    The asymmetry is the difference of the wings of the interpolated profile, each
    velocity weighted by the overlap of the 10 km/s about it with the wing, over its
    value at the peak, not its maximum, which is brighter for a few of these Mg II k
    profiles at the defaults.

    The profile holds the peak also when velocity_range[1] is not a multiple of dv.
    """
    cube = read_files(v34_sg_file)["Mg II k 2796"][0]
    low, high = velocity_range
    result = calculate_red_blue_asymmetry(cube, velocity_range=velocity_range * u.km / u.s)
    profiles = result["interpolated_profile"].data
    velocity = result["interpolated_profile"].axis_world_coords(-1)[0].to_value(u.km / u.s)
    np.testing.assert_allclose(velocity, -velocity[::-1], atol=1e-9)

    def wing(low, high):
        overlap = np.clip(np.minimum(velocity + 5, high) - np.maximum(velocity - 5, low), 0, None)
        return np.nansum(profiles * overlap, axis=-1) / (np.isfinite(profiles) * overlap).sum(axis=-1)

    peak = profiles[..., np.isclose(velocity, 0)][..., 0]
    ok = result["quality"].data == RBAQualityFlag.OK
    assert ok.sum() > ok.size / 2
    expected = (wing(low, high) - wing(-high, -low)) / peak
    np.testing.assert_allclose(result["red_blue_asymmetry"].data[ok], expected[ok], rtol=1e-12)


@pytest.mark.parametrize(("velocity_range", "atol"), [((52, 58), 0.01), ((50, 150), 0.002)])
def test_calculate_red_blue_asymmetry_wings_do_not_depend_on_dv(v34_sg_file, velocity_range, atol):
    """
    The wings are the means over velocity_range, which change little with dv: on these Mg II k
    profiles, dv = 10 km/s gives a median asymmetry within atol of a 1 km/s grid, about a
    quarter of the difference the bins within the range alone gave.
    """
    cube = read_files(v34_sg_file)["Mg II k 2796"][0]
    kwargs = {"velocity_range": velocity_range * u.km / u.s, "return_profiles": False}
    coarse = calculate_red_blue_asymmetry(cube, **kwargs)
    fine = calculate_red_blue_asymmetry(cube, dv=1 * u.km / u.s, **kwargs)
    ok = (coarse["quality"].data == RBAQualityFlag.OK) & (fine["quality"].data == RBAQualityFlag.OK)
    assert ok.sum() > ok.size / 2
    difference = coarse["red_blue_asymmetry"].data[ok] - fine["red_blue_asymmetry"].data[ok]
    assert np.median(np.abs(difference)) < atol


def test_calculate_red_blue_asymmetry_flags_saturated_near_peak(v34_sg_file):
    """
    A saturated sample flags a pixel within velocity_range[1] of rest_wavelength, where
    the peak is looked for, or within velocity_range[1] + dv of the peak: one at Mg II h
    does not flag the Mg II k measurement.
    """
    cube = read_files(v34_sg_file)["Mg II k 2796"][0]
    velocity = cube.axis_world_coords("em.wl")[0].to_value(
        u.km / u.s, doppler_rest=cube.meta.rest_wavelength, doppler_convention="optical"
    )
    kwargs = {"velocity_range": (10, 30) * u.km / u.s, "return_profiles": False}
    saturated = np.zeros(cube.shape, dtype=bool)
    saturated[0, 10, np.argmin(np.abs(velocity - 766))] = True  # Mg II h
    saturated[0, 20, np.argmin(np.abs(velocity))] = True  # Mg II k
    # Pixel 30 peaks at -15.8 km/s, more than 30 + 10 km/s from its last sample within 30 km/s of rest
    saturated[0, 30, np.flatnonzero(velocity <= 30)[-1]] = True
    # Pixel 40 peaks at -15.8 km/s: its sample at -32 km/s, beyond the search range, is within 30 + 10 km/s of the peak
    saturated[0, 40, np.flatnonzero(velocity >= -30)[0] - 1] = True
    expected = calculate_red_blue_asymmetry(cube, **kwargs)["quality"].data
    assert np.all(expected[0, [10, 20, 30, 40]] == RBAQualityFlag.OK)
    expected[0, [20, 30, 40]] = RBAQualityFlag.SATURATED
    mark_saturated(cube, saturated)
    quality = calculate_red_blue_asymmetry(cube, **kwargs)["quality"].data
    np.testing.assert_array_equal(quality, expected)


def test_calculate_red_blue_asymmetry_flags_saturated_one_sample_beyond_the_wing(v34_sg_file):
    """
    With dv finer than the samples, a saturated sample within one sample of the end of a
    wing, which the spline there depends on, flags the pixel.
    """
    cube = read_files(v34_sg_file)["Mg II k 2796"][0]
    velocity = cube.axis_world_coords("em.wl")[0].to_value(
        u.km / u.s, doppler_rest=cube.meta.rest_wavelength, doppler_convention="optical"
    )
    kwargs = {"velocity_range": (10, 30) * u.km / u.s, "dv": 1 * u.km / u.s, "return_profiles": False}
    expected = calculate_red_blue_asymmetry(cube, **kwargs)["quality"].data
    assert expected[0, 0] == RBAQualityFlag.OK
    expected[0, 0] = RBAQualityFlag.SATURATED
    # Pixel 0 peaks at 16.3 km/s: its sample at 49.8 km/s is beyond 30 + 1 km/s of the peak, but
    # within 30 km/s and one of the 5.5 km/s samples
    saturated = np.zeros(cube.shape, dtype=bool)
    saturated[0, 0, np.argmin(np.abs(velocity - 49.8))] = True
    mark_saturated(cube, saturated)
    np.testing.assert_array_equal(calculate_red_blue_asymmetry(cube, **kwargs)["quality"].data, expected)


def test_calculate_red_blue_asymmetry_bin_in_both_wings_has_no_error(v34_sg_file):
    """
    Within dv / 2 of the peak, the center bin alone is in both wings, so the asymmetry
    and its error are 0.
    """
    cube = read_files(v34_sg_file, uncertainty=True)["Mg II k 2796"][0]
    result = calculate_red_blue_asymmetry(cube, velocity_range=(1, 2) * u.km / u.s, return_profiles=False)
    ok = result["quality"].data == RBAQualityFlag.OK
    assert ok.sum() > 100
    np.testing.assert_array_equal(result["red_blue_asymmetry"].data[ok], 0)
    np.testing.assert_array_equal(result["red_blue_asymmetry"].uncertainty.array[ok], 0)


def test_calculate_red_blue_asymmetry_peak_at_edge_of_velocity_range(v34_sg_file):
    """
    A peak sample at the first or last sample within velocity_range[1] of rest_wavelength is
    measured where the next sample beyond it is lower and flagged where the profile still rises:
    within 10 km/s of Mg II k, profiles have either. A crop to those samples flags them all, as
    the window ends there.
    """
    cube = read_files(v34_sg_file)["Mg II k 2796"][0]
    kwargs = {"velocity_range": (5, 10) * u.km / u.s, "dv": 1 * u.km / u.s, "return_profiles": False}
    velocity = cube.axis_world_coords("em.wl")[0].to_value(
        u.km / u.s, doppler_rest=cube.meta.rest_wavelength, doppler_convention="optical"
    )
    first, last = np.flatnonzero(np.abs(velocity) <= 10)[[0, -1]]
    quality = calculate_red_blue_asymmetry(cube, **kwargs)["quality"].data
    at_edge = calculate_red_blue_asymmetry(cube[..., first : last + 1], **kwargs)["quality"].data
    at_edge = at_edge == RBAQualityFlag.PEAK_AT_EDGE
    at_first = np.argmax(cube.data[..., first : last + 1], axis=-1) == 0
    edge = np.where(at_first, cube.data[..., first], cube.data[..., last])
    beyond = np.where(at_first, cube.data[..., first - 1], cube.data[..., last + 1])
    rising = at_edge & (beyond >= edge)
    assert 0 < np.count_nonzero(rising) < np.count_nonzero(at_edge)
    np.testing.assert_array_equal(quality == RBAQualityFlag.PEAK_AT_EDGE, rising)
    assert np.all(quality[at_edge & ~rising] == RBAQualityFlag.OK)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rest_wavelength": [140, 141] * u.nm}, "rest_wavelength must be a positive finite scalar"),
        ({"rest_wavelength": np.nan * u.nm}, "rest_wavelength must be a positive finite scalar"),
        ({"rest_wavelength": -140 * u.nm}, "rest_wavelength must be a positive finite scalar"),
        ({"velocity_range": [50] * u.km / u.s}, "velocity_range must be two increasing velocities"),
        ({"velocity_range": [-50, 150] * u.km / u.s}, "velocity_range must be positive"),
        ({"velocity_range": [150, 50] * u.km / u.s}, "velocity_range must be two increasing velocities"),
        ({"dv": 0 * u.km / u.s}, "dv must be positive"),
        ({"spline_degree": -1}, "spline_degree must be a non-negative integer"),
    ],
    ids=["rest-vector", "rest-nan", "rest-negative", "range-one", "range-negative", "range-decreasing", "dv", "degree"],
)
def test_calculate_red_blue_asymmetry_rejects_bad_arguments(kwargs, message):
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    cube = make_test_spectrogram_cube(
        _flat_wing_profile(velocity).reshape(1, 1, -1), _wavelengths_from_velocity(velocity)
    )

    with pytest.raises(ValueError, match=message):
        calculate_red_blue_asymmetry(cube, **{"rest_wavelength": REST_WAVELENGTH, **kwargs})


@pytest.mark.parametrize(
    ("kind", "flag"),
    [
        ("rising", RBAQualityFlag.PEAK_AT_EDGE),
        ("two-samples", RBAQualityFlag.TOO_FEW_POINTS),
        ("zero", RBAQualityFlag.PEAK_IS_ZERO),
    ],
)
def test_calculate_red_blue_asymmetry_flags_unusable_profiles(kind, flag):
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    center = velocity.size // 2
    if kind == "rising":
        profile = np.linspace(1, 10, velocity.size)  # the peak is the last sample
    elif kind == "two-samples":
        profile = np.full(velocity.size, np.nan)
        profile[center], profile[center + 1] = 10, 1  # fewer than the cubic spline's four points
    else:
        profile = np.zeros(velocity.size)
        profile[center] = 1e-9  # a peak within np.isclose of zero
    cube = make_test_spectrogram_cube(profile.reshape(1, 1, -1), _wavelengths_from_velocity(velocity))

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH)

    assert RBAQualityFlag(result["quality"].data[0, 0]) is flag
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_explicit_rest_wavelength():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    cube = make_test_spectrogram_cube(profile.reshape(1, 1, -1), wavelengths)
    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, spline_degree=1)
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_auto_detect_rest_wavelength():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    header = fits.Header(cube.wcs.to_header())
    header["INSTRUME"] = "IRIS"
    header["TELESCOP"] = "IRIS"
    header["DATA_LEV"] = 2
    header["OBSID"] = 1
    header["OBS_DESC"] = "Test obs"
    header["NWIN"] = 1
    header["TDESC1"] = "Si IV 1403"
    header["TWAVE1"] = REST_WAVELENGTH.to(u.AA).value
    header["TWMIN1"] = (wavelengths[0] - 0.5 * u.nm).to(u.AA).value
    header["TWMAX1"] = (wavelengths[-1] + 0.5 * u.nm).to(u.AA).value
    header["TDET1"] = "FUV2"
    meta = SGMeta(header, "Si IV 1403", data_shape=data.shape)
    cube.meta = meta

    result = calculate_red_blue_asymmetry(cube, spline_degree=1)
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])
    assert u.isclose(
        result["red_blue_asymmetry"].meta["rba_rest_wavelength"] * u.nm,
        REST_WAVELENGTH,
        rtol=1e-4,
    )


def test_calculate_red_blue_asymmetry_auto_detect_on_real_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["C II 1336"][0]
    result = calculate_red_blue_asymmetry(cube, spline_degree=1)
    assert "red_blue_asymmetry" in result
    assert result["red_blue_asymmetry"].shape == cube.shape[:-1]
