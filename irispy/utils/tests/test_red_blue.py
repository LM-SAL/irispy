import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose

import irispy.utils.red_blue as red_blue_module
from irispy.io.utils import read_files
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.tests.helpers import figure_test, make_test_spectrogram_cube
from irispy.utils.red_blue import RBAQualityFlag, calculate_red_blue_asymmetry

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

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)

    assert isinstance(result, RasterCollection)
    assert set(result.keys()) == {
        "red_blue_asymmetry",
        "quality",
        "observed_profile",
        "interpolated_profile",
    }
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

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)
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

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)
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


def test_calculate_red_blue_asymmetry_with_uncertainty():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.uncertainty = StdDevUncertainty(np.full(cube.shape, 0.1))

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        degree=1,
    )
    assert "red_blue_asymmetry_error" in result
    assert result["red_blue_asymmetry_error"].unit == u.one
    numerator = 2
    peak = 10
    n_wing_bins = int((150 - 50) / 10) + 1
    wing_error = np.sqrt(np.sum(np.full(n_wing_bins, 0.1) ** 2)) / n_wing_bins
    expected_propagated_error = np.sqrt((np.sqrt(2) * wing_error / peak) ** 2 + (numerator * 0.1 / peak**2) ** 2)
    assert_quantity_allclose(result["red_blue_asymmetry_error"].data[0, 0] * u.one, expected_propagated_error * u.one)
    assert result["red_blue_asymmetry"].meta["rba_rest_wavelength"] == 140.277
    assert result["red_blue_asymmetry"].meta["rba_interpolation_degree"] == 1

    symmetric_cube = make_test_spectrogram_cube(_flat_wing_profile(velocity).reshape(1, 1, -1), wavelengths)
    symmetric_cube.uncertainty = StdDevUncertainty(np.full(symmetric_cube.shape, 0.1))
    symmetric_result = calculate_red_blue_asymmetry(
        symmetric_cube,
        rest_wavelength=REST_WAVELENGTH,
        degree=1,
    )
    expected_zero_error = np.sqrt(2) * wing_error / 10
    assert_quantity_allclose(symmetric_result["red_blue_asymmetry"].data[0, 0] * u.one, 0 * u.one)
    assert_quantity_allclose(
        symmetric_result["red_blue_asymmetry_error"].data[0, 0] * u.one, expected_zero_error * u.one
    )


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
    cube = make_test_spectrogram_cube(_flat_wing_profile(velocity, red_excess=2).reshape(1, 1, -1), wavelengths)
    cube.uncertainty = StdDevUncertainty(np.full(cube.shape, 0.1))

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        degree=1,
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
        degree=1,
    )
    assert result["red_blue_asymmetry"].data[0, 0] > 0
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_return_profiles():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.stack([profile, profile + 1]).reshape(1, 2, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)
    cube.uncertainty = StdDevUncertainty(np.full(cube.shape, 0.1))

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)
    observed_cube = result["observed_profile"]
    interp_cube = result["interpolated_profile"]
    assert isinstance(observed_cube, SpectrogramCube)
    assert isinstance(interp_cube, SpectrogramCube)
    assert observed_cube.data.shape == cube.shape
    assert interp_cube.data.shape == (1, 2, 33)
    assert "spect.dopplerVeloc" in observed_cube.wcs.world_axis_physical_types
    assert "spect.dopplerVeloc" in interp_cube.wcs.world_axis_physical_types
    assert interp_cube.uncertainty.array.shape == interp_cube.shape
    assert_quantity_allclose(observed_cube[0, 0].axis_world_coords(0)[0], velocity)
    assert np.isfinite(interp_cube[0, 1].data).all()


def test_calculate_red_blue_asymmetry_return_profiles_on_sliced_cube():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = np.tile(profile, (2, 3, 1))
    cube = make_test_spectrogram_cube(data, wavelengths)[:, :1, :]

    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)
    assert result["observed_profile"].shape == cube.shape
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
        degree=1,
        return_profiles=False,
    )
    assert "observed_profile" not in result
    assert "interpolated_profile" not in result
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_continuum_subtraction():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = (profile + 5).reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result_without = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)

    continuum_wavelengths = (
        u.Quantity([[wavelengths[0].value, wavelengths[2].value], [wavelengths[-3].value, wavelengths[-1].value]])
        * wavelengths.unit
    )
    result_with = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        degree=1,
        continuum_windows=continuum_wavelengths,
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

    result_without = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)
    result_with = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        degree=1,
        continuum_windows=continuum_wavelengths,
    )
    rba_without = float(result_without["red_blue_asymmetry"].data[0, 0])
    rba_with = float(result_with["red_blue_asymmetry"].data[0, 0])
    assert_quantity_allclose(rba_without * u.one, (2 / 15) * u.one)
    assert_quantity_allclose(rba_with * u.one, (2 / 9) * u.one)

    raw_profile = result_without["observed_profile"][0, 0]
    continuum_profile = result_with["observed_profile"][0, 0]
    velocities = raw_profile.axis_world_coords(0)[0].to_value(u.km / u.s)
    v_low, v_high = 50, 150

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].plot(velocities, raw_profile.data, "o-", color="0.2", markersize=3, label="Input + continuum")
    axes[0].plot(velocities, continuum_profile.data, "o-", color="C3", markersize=3, label="Continuum subtracted")
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
        degree=1,
    )
    assert RBAQualityFlag(result["quality"].data[0, 0]) is RBAQualityFlag.INCOMPLETE_WINGS
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


@pytest.mark.parametrize(
    ("option", "value", "expected_flag"),
    [
        ("min_intensity", 15, RBAQualityFlag.LOW_SIGNAL),
        ("saturation_limit", 5, RBAQualityFlag.SATURATED),
    ],
)
def test_calculate_red_blue_asymmetry_quality_thresholds(option, value, expected_flag):
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    data = profile.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wavelengths)

    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=REST_WAVELENGTH,
        degree=1,
        **{option: value},
    )
    assert RBAQualityFlag(result["quality"].data[0, 0]) is expected_flag
    assert np.isnan(result["red_blue_asymmetry"].data[0, 0])


def test_calculate_red_blue_asymmetry_real_cube_shape_and_coords(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["C II 1336"]
    result = calculate_red_blue_asymmetry(
        cube,
        rest_wavelength=133.29 * u.nm,
        velocity_range=(20, 60) * u.km / u.s,
        degree=1,
    )
    assert result["red_blue_asymmetry"].shape == cube.shape[:-1]
    assert "time" in tuple(result["red_blue_asymmetry"].extra_coords.keys())


def test_calculate_red_blue_asymmetry_rest_wavelength_required_without_meta():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm)
    with pytest.raises(ValueError, match="rest_wavelength"):
        calculate_red_blue_asymmetry(cube, degree=1)


def test_calculate_red_blue_asymmetry_explicit_rest_wavelength():
    velocity = np.arange(-200, 201, 10) * u.km / u.s
    wavelengths = _wavelengths_from_velocity(velocity)
    profile = _flat_wing_profile(velocity, red_excess=2)
    cube = make_test_spectrogram_cube(profile.reshape(1, 1, -1), wavelengths)
    result = calculate_red_blue_asymmetry(cube, rest_wavelength=REST_WAVELENGTH, degree=1)
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

    result = calculate_red_blue_asymmetry(cube, degree=1)
    assert np.isfinite(result["red_blue_asymmetry"].data[0, 0])
    assert u.isclose(
        result["red_blue_asymmetry"].meta["rba_rest_wavelength"] * u.nm,
        REST_WAVELENGTH,
        rtol=1e-4,
    )


def test_calculate_red_blue_asymmetry_auto_detect_on_real_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["C II 1336"][0]
    result = calculate_red_blue_asymmetry(cube, degree=1)
    assert "red_blue_asymmetry" in result
    assert result["red_blue_asymmetry"].shape == cube.shape[:-1]
