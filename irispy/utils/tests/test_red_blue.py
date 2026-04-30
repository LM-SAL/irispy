import numpy as np

import astropy.units as u
from astropy import constants
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose

from irispy.io.utils import read_files
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.red_blue import RBA_INCOMPLETE_WINGS, calculate_red_blue_asymmetry

REST_WAVELENGTH = 140.277 * u.nm


def _wavelengths_from_velocity(velocity):
    return REST_WAVELENGTH * (1 + velocity / constants.c.to(u.km / u.s))


def _flat_wing_profile(velocity, *, red_excess=0, blue_excess=0):
    profile = np.ones(velocity.shape, dtype=float)
    profile[np.abs(velocity.to_value(u.km / u.s)) <= 20] = 10
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

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
    )

    assert isinstance(result, RasterCollection)
    assert set(result.keys()) == {
        "red_blue_asymmetry",
        "red_wing",
        "blue_wing",
        "peak_intensity",
        "peak_velocity",
        "quality",
        "observed_profile",
        "interpolated_profile",
    }
    assert_quantity_allclose(result["red_blue_asymmetry"].data[0, 0] * u.one, 0.2 * u.one)
    assert_quantity_allclose(result["red_blue_asymmetry"].data[0, 1] * u.one, -0.3 * u.one)
    assert_quantity_allclose(result["red_wing"].data[0, 0] * result["red_wing"].unit, 3 * u.DN)
    assert_quantity_allclose(result["blue_wing"].data[0, 0] * result["blue_wing"].unit, 1 * u.DN)
    assert_quantity_allclose(result["peak_intensity"].data[0, 0] * result["peak_intensity"].unit, 10 * u.DN)
    assert result["peak_velocity"].unit == u.km / u.s
    assert result["quality"].unit == u.dimensionless_unscaled


def test_calculate_red_blue_asymmetry_uses_masked_bins():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=1000)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.mask = np.zeros(cube.shape, dtype=bool)
    red = (velocity >= 50 * u.km / u.s) & (velocity <= 150 * u.km / u.s)
    cube.mask[..., red] = True

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
    )

    assert_quantity_allclose(result["red_blue_asymmetry"].data[0, 0] * u.one, 0 * u.one)


def test_calculate_red_blue_asymmetry_output_does_not_inherit_first_wavelength_mask():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.mask = np.zeros(cube.shape, dtype=bool)
    cube.mask[..., 0] = True

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
    )

    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])
    assert not result["red_blue_asymmetry"].mask[0, 0]
    assert result["quality"].mask is None
    assert result["quality"].data[0, 0] == 0


def test_calculate_red_blue_asymmetry_with_uncertainty():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    uncertainty = np.full(cube.shape, 0.1) * u.DN

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        uncertainty=uncertainty,
        interpolation_kind="linear",
        center_on_peak=False,
    )

    assert "red_blue_asymmetry_error" in result
    assert result["red_blue_asymmetry_error"].unit == u.one
    red_wing = float(result["red_wing"].data[0, 0])
    blue_wing = float(result["blue_wing"].data[0, 0])
    peak = float(result["peak_intensity"].data[0, 0])
    numerator = red_wing - blue_wing
    wing_error = np.sqrt(np.sum(np.full(11, 0.1) ** 2)) / 11
    expected_propagated_error = np.sqrt((np.sqrt(2) * wing_error / peak) ** 2 + (numerator * 0.1 / peak**2) ** 2)
    assert_quantity_allclose(
        result["red_blue_asymmetry_error"].data[0, 0] * u.one,
        expected_propagated_error * u.one,
    )

    symmetric_cube = make_test_spectrogram_cube(_flat_wing_profile(velocity).reshape(1, 1, -1), wavelengths)
    symmetric_result = calculate_red_blue_asymmetry(
        symmetric_cube,
        rest_wavelength=REST_WAVELENGTH,
        uncertainty=uncertainty,
        interpolation_kind="linear",
        center_on_peak=False,
    )
    expected_zero_error = np.sqrt(2) * wing_error / 10
    assert_quantity_allclose(symmetric_result["red_blue_asymmetry"].data[0, 0] * u.one, 0 * u.one)
    assert_quantity_allclose(
        symmetric_result["red_blue_asymmetry_error"].data[0, 0] * u.one,
        expected_zero_error * u.one,
    )

    assert result["red_wing_error"].unit == u.DN
    assert result["blue_wing_error"].unit == u.DN
    # Meta should record the parameters used
    assert result["red_blue_asymmetry"].meta["rba_rest_wavelength"] == 140.277
    assert result["red_blue_asymmetry"].meta["rba_center_on_peak"] is False


def test_calculate_red_blue_asymmetry_center_on_peak():
    """
    Test that center_on_peak=True correctly aligns a Doppler-shifted line.
    """
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    # Create a line peaked at +30 km/s with red excess
    profile = np.ones(velocity.shape, dtype=float)
    peak_wvl = _wavelengths_from_velocity(30 * u.km / u.s)
    sigma_wvl = abs(_wavelengths_from_velocity(10 * u.km / u.s) - peak_wvl).value
    profile += 9 * np.exp(-0.5 * ((wavelengths.value - peak_wvl.value) / sigma_wvl) ** 2)
    # Add red wing excess
    red = (velocity >= 50 * u.km / u.s) & (velocity <= 150 * u.km / u.s)
    profile[red] += 2
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        velocity_range=(50, 150) * u.km / u.s,
        velocity_window=160 * u.km / u.s,
        fit_window=190 * u.km / u.s,
        interpolation_kind="linear",
        center_on_peak=True,
    )

    # Positive RBA because red wing has excess
    assert result["red_blue_asymmetry"].data[0, 0] > 0
    # With center_on_peak=True the peak should be found and aligned, so the
    # blue wing (now centered on the peak) should NOT have the excess.
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_return_profiles():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.stack([profile, profile + 1]).reshape(1, 2, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.uncertainty = StdDevUncertainty(np.full(cube.shape, 0.1))

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
    )

    assert isinstance(result, RasterCollection)
    assert "observed_profile" in result
    assert "interpolated_profile" in result
    observed_cube = result["observed_profile"]
    interp_cube = result["interpolated_profile"]
    assert isinstance(observed_cube, SpectrogramCube)
    assert isinstance(interp_cube, SpectrogramCube)
    assert observed_cube.data.shape == cube.shape
    assert interp_cube.data.shape == (1, 2, 41)
    assert "spect.dopplerVeloc" in observed_cube.wcs.world_axis_physical_types
    assert "spect.dopplerVeloc" in interp_cube.wcs.world_axis_physical_types
    assert interp_cube.uncertainty.array.shape == interp_cube.shape

    observed_profile = observed_cube[0, 0]
    interp_profile = interp_cube[0, 1]
    assert observed_profile.data.shape == (41,)
    assert interp_profile.data.shape == (41,)
    assert_quantity_allclose(observed_profile.axis_world_coords(0)[0], velocity)
    assert_quantity_allclose(interp_profile.axis_world_coords(0)[0], np.arange(-200, 201, 10) * u.km / u.s)
    assert np.isfinite(interp_profile.data).all()


def test_calculate_red_blue_asymmetry_return_profiles_on_sliced_cube():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.tile(profile, (2, 3, 1))
    cube = make_test_spectrogram_cube(data, wavelengths)[:, :1, :]

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
    )

    assert result["observed_profile"].shape == cube.shape
    assert result["interpolated_profile"].shape == (2, 1, 41)
    assert_quantity_allclose(
        result["interpolated_profile"][0, 0].axis_world_coords(0)[0],
        np.arange(-200, 201, 10) * u.km / u.s,
    )


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
        velocity_window=160 * u.km / u.s,
        fit_window=190 * u.km / u.s,
        interpolation_kind="linear",
        center_on_peak=True,
    )

    assert int(result["quality"].data[0, 0]) == RBA_INCOMPLETE_WINGS
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_min_intensity():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
        min_intensity=15,
    )

    assert result["quality"].data[0, 0] == 7  # RBA_LOW_SIGNAL
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_saturation_limit():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        interpolation_kind="linear",
        center_on_peak=False,
        saturation_limit=5,
    )

    assert result["quality"].data[0, 0] == 8  # RBA_SATURATED
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_real_cube_shape_and_coords(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["C II 1336"][0]
    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=133.29 * u.nm,
        velocity_range=(20, 60) * u.km / u.s,
        velocity_window=80 * u.km / u.s,
        interpolation_kind="linear",
    )

    assert result["red_blue_asymmetry"].shape == cube.shape[:-1]
    assert "time" in tuple(result["red_blue_asymmetry"].extra_coords.keys())
