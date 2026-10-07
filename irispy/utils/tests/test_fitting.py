import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.modeling.fitting import LevMarLSQFitter, TRFLSQFitter, parallel_fit_dask
from astropy.nddata import StdDevUncertainty
from astropy.time import Time

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import ATOMIC_MASS, INSTRUMENTAL_FWHM
from irispy.utils.fitting import (
    FitQualityFlag,
    NonThermalQualityFlag,
    maps_from_fit,
    mg_ii_model,
    non_thermal_velocity,
    profiles_on_background,
    si_iv_1403_model,
)

SHAPE = (20, 20)
WAVELENGTH = (1402.0 + 0.026 * np.arange(60)) * u.AA
MG_FEATURES_FILE = "mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits"
C_KMS = constants.c.to_value(u.km / u.s)


def _fit(model, data, sigma, wavelength=WAVELENGTH, unit=u.ct, *, fitter=None, **kwargs):
    fitter = TRFLSQFitter() if fitter is None else fitter
    fitted = parallel_fit_dask(
        model=model,
        fitter=fitter,
        data=data,
        data_unit=unit,
        weights=1 / sigma,
        world=(wavelength,),
        fitting_axes=2,
        scheduler="single-threaded",
        fit_info=True,
        **kwargs,
    )
    return fitted, fitter


def _errors(fitter):
    covariance = fitter.fit_info.get_property_as_array("param_cov")
    return np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))


def _observe(model, rng, wavelength=WAVELENGTH, unit=u.ct):
    """
    Evaluate a model with array parameters on the wavelength grid and add Poisson-like
    noise.
    """
    clean = np.moveaxis(model(wavelength[:, None, None]).to_value(unit), 0, -1)
    sigma = np.sqrt(clean)
    return clean + rng.normal(size=clean.shape) * sigma, sigma


def _assert_recovered(fitted, truth, error, data, sigma, wavelength=WAVELENGTH, max_local_minima=0.0):
    def chi2(model):
        return np.sum(((data - np.moveaxis(model(wavelength[:, None, None]).value, 0, -1)) / sigma) ** 2, axis=-1)

    # The true parameters are one point the fit could reach, so a fit worse than them is in a local minimum.
    found = chi2(fitted) <= chi2(truth) * (1 + 1e-6)
    assert np.mean(~found) <= max_local_minima
    for index, name in enumerate(fitted.param_names):
        z = (getattr(fitted, name).value - getattr(truth, name).value)[found] / error[..., index][found]
        # The reported 1 sigma should hold the true value in about 68% of the spectra.
        assert 0.6 < np.mean(np.abs(z) < 1) < 0.76, name
        assert np.mean(np.abs(z) < 3) > 0.98, name


def _assert_within_bounds(fitted, model):
    for name, (lower, upper) in model.bounds.items():
        values = getattr(fitted, name).value
        assert lower is None or np.all(values >= lower), name
        assert upper is None or np.all(values <= upper), name


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("background", ["constant", "linear"])
def test_profiles_on_background_recovers_parameters(profile, background):
    rng = np.random.default_rng(42)
    centers = np.stack([1402.70 + rng.normal(0, 0.01, SHAPE), 1402.88 + rng.normal(0, 0.01, SHAPE)]) * u.AA
    amplitudes = rng.uniform(80, 150, (2, *SHAPE)) * u.ct
    widths = rng.uniform(0.03, 0.05, (2, *SHAPE)) * u.AA
    level = np.full(SHAPE, 5.0) * u.ct
    truth = profiles_on_background(
        centers, amplitudes=amplitudes, widths=widths, background=background, background_level=level, profile=profile
    )
    data, sigma = _observe(truth, rng)
    start = profiles_on_background(
        centers + 0.01 * u.AA,
        amplitudes=0.8 * amplitudes,
        widths=1.2 * widths,
        background=background,
        background_level=0.6 * level,
        profile=profile,
    )
    fitted, fitter = _fit(start, data, sigma)
    _assert_recovered(fitted, truth, _errors(fitter), data, sigma)


def test_profiles_on_background_array_starts_broadcast():
    rng = np.random.default_rng(0)
    centers = (1402.77 + rng.normal(0, 0.02, SHAPE)) * u.AA
    truth = profiles_on_background(
        centers[np.newaxis], amplitudes=[100] * u.ct, widths=[0.04] * u.AA, background_level=2 * u.ct
    )
    assert truth.mean_1.shape == SHAPE
    assert truth.amplitude_1.shape == ()
    data, sigma = _observe(truth, rng)
    # Scalar starts mixed with per-spectrum centers.
    start = profiles_on_background(
        centers[np.newaxis] + 0.01 * u.AA, amplitudes=[80] * u.ct, widths=[0.05] * u.AA, background_level=1 * u.ct
    )
    fitted, _ = _fit(start, data, sigma)
    assert fitted.mean_1.shape == SHAPE
    assert fitted.amplitude_1.shape == SHAPE
    np.testing.assert_allclose(fitted.mean_1.quantity.to_value(u.AA), centers.to_value(u.AA), atol=0.01)


def test_profiles_on_background_bounds_hold():
    rng = np.random.default_rng(1)
    truth = profiles_on_background(
        [1402.77] * u.AA, amplitudes=[100] * u.ct, widths=[0.04] * u.AA, background_level=np.full(SHAPE, 2.0) * u.ct
    )
    data, sigma = _observe(truth, rng)
    # Bounds in another unit than the centers are converted to it.
    bounds = {"mean_1": (140.26, 140.27) * u.nm, "amplitude_1": (0, 50) * u.ct, "amplitude_0": (None, 1 * u.ct)}
    start = profiles_on_background([1402.65] * u.AA, amplitudes=[40] * u.ct, widths=[0.04] * u.AA, bounds=bounds)
    np.testing.assert_allclose(start.mean_1.bounds, (1402.6, 1402.7))
    assert start.stddev_1.bounds[0] > 0
    fitted, _ = _fit(start, data, sigma)
    _assert_within_bounds(fitted, start)
    # The true center lies outside the bounds, so most fits stop on the nearer one.
    np.testing.assert_allclose(np.median(fitted.mean_1.value), 1402.7)


@pytest.mark.parametrize(("keyword", "value"), [("profile", "voigt"), ("background", "quadratic")])
def test_profiles_on_background_rejects_unknown_options(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        profiles_on_background([1402.77] * u.AA, **{keyword: value})


def _cube(truth, wavelength, rng):
    data, sigma = _observe(truth, rng, wavelength, u.DN)
    return make_test_spectrogram_cube(data, wavelength, uncertainty=StdDevUncertainty(sigma))


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_si_iv_1403_model_recovers_parameters(profile):
    rng = np.random.default_rng(7)
    truth = profiles_on_background(
        (1402.77 + rng.normal(0, 0.02, SHAPE))[np.newaxis] * u.AA,
        amplitudes=rng.uniform(100, 300, (1, *SHAPE)) * u.DN,
        widths=rng.uniform(0.05, 0.09, (1, *SHAPE)) * u.AA,
        background_level=5 * u.DN,
        profile=profile,
    )
    cube = _cube(truth, WAVELENGTH, rng)
    model = si_iv_1403_model(cube, profile=profile)
    assert model.param_names == truth.param_names
    assert model(WAVELENGTH[:, np.newaxis, np.newaxis]).unit == u.DN
    assert all(getattr(model, name).shape == SHAPE for name in model.param_names)
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, unit=cube.unit)
    _assert_recovered(fitted, truth, _errors(fitter), cube.data, cube.uncertainty.array, max_local_minima=0.01)
    _assert_within_bounds(fitted, model)


def test_mg_ii_model_recovers_parameters():
    rng = np.random.default_rng(8)
    wavelength = (2795.5 + 0.0254 * np.arange(70)) * u.AA
    # The k2v and k2r peaks 32 km/s apart, as observed in the quiet Sun (Ondratschek et al. 2024), overlapping
    # so that the line center is about 60% of the peaks, as in Level 2 data.
    truth = profiles_on_background(
        (np.array([2796.20, 2796.50])[:, None, None] + rng.normal(0, 0.01, (2, *SHAPE))) * u.AA,
        amplitudes=rng.uniform(400, 600, (2, *SHAPE)) * u.DN,
        widths=rng.uniform(0.08, 0.1, (2, *SHAPE)) * u.AA,
        background_level=100 * u.DN,
    )
    cube = _cube(truth, wavelength, rng)
    model = mg_ii_model(cube)
    assert all(getattr(model, name).shape == SHAPE for name in model.param_names)
    assert all(getattr(model, name).unit == u.AA for name in ("mean_1", "stddev_1", "mean_2", "stddev_2"))
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, wavelength, cube.unit)
    _assert_recovered(
        fitted, truth, _errors(fitter), cube.data, cube.uncertainty.array, wavelength, max_local_minima=0.03
    )
    _assert_within_bounds(fitted, model)


def test_mg_ii_model_on_level_2_data():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0]
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    fitted, _ = _fit(
        mg_ii_model(cube), np.nan_to_num(cube.data.clip(min=0)), cube.uncertainty.array, wavelength, cube.unit
    )
    # IRIS k2 peak separations spread about a mean of 33 km/s (Ondratschek et al. 2024, Fig. 5b), so the median
    # of this disk-center raster should fall well inside 20-50 km/s.
    separation = (fitted.mean_2.quantity - fitted.mean_1.quantity) / (2796.352 * u.AA) * C_KMS * u.km / u.s
    assert 20 < np.nanmedian(separation.value) < 50


@pytest.mark.parametrize(
    ("function", "centers", "start"),
    [(si_iv_1403_model, [1402.77], 1402.0), (mg_ii_model, [2796.20, 2796.50], 2795.5)],
)
def test_presets_accept_scalar_mask(function, centers, start):
    wavelength = (start + 0.0254 * np.arange(70)) * u.AA
    truth = profiles_on_background(
        centers * u.AA,
        amplitudes=np.full(len(centers), 500) * u.DN,
        widths=np.full(len(centers), 0.09) * u.AA,
        background_level=100 * u.DN,
    )
    data = np.broadcast_to(truth(wavelength).value, (2, 3, 70)).copy()
    cube = make_test_spectrogram_cube(data, wavelength)
    expected = function(cube)
    cube.mask = False
    np.testing.assert_allclose(function(cube).parameters, expected.parameters)


@pytest.mark.parametrize("function", [si_iv_1403_model, mg_ii_model])
def test_presets_reject_unscaled_data(function):
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), memmap=True)
    with pytest.raises(ValueError, match="unscaled"):
        function(raster["Mg II k 2796"][0])


@pytest.mark.parametrize(
    ("function", "start", "match"),
    [
        (si_iv_1403_model, 1401.0, "O IV 1401.157"),
        (mg_ii_model, 2795.5, "Mg II 2798.754"),
        (mg_ii_model, 2792.0, "neither"),
    ],
)
def test_presets_reject_other_windows(function, start, match):
    wavelength = (start + 0.0254 * np.arange(140)) * u.AA
    cube = make_test_spectrogram_cube(np.ones((2, 3, 140)), wavelength)
    with pytest.raises(ValueError, match=match):
        function(cube)


def _si_iv_cube(rng, profile="gaussian"):
    truth = profiles_on_background(
        (1402.77 + rng.normal(0, 0.02, SHAPE))[np.newaxis] * u.AA,
        amplitudes=rng.uniform(100, 300, (1, *SHAPE)) * u.DN,
        widths=rng.uniform(0.05, 0.09, (1, *SHAPE)) * u.AA,
        background_level=5 * u.DN,
        profile=profile,
    )
    return truth, _cube(truth, WAVELENGTH, rng)


def test_maps_from_fit_recovers_derived_maps():
    rng = np.random.default_rng(7)
    truth, cube = _si_iv_cube(rng)
    fitted, fitter = _fit(si_iv_1403_model(cube), cube.data, cube.uncertainty.array, unit=cube.unit)
    error = _errors(fitter)
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    derived = ["velocity_1", "fwhm_1", "fwhm_velocity_1", "integrated_intensity_1"]
    assert list(maps.keys()) == [*fitted.param_names, *derived, "quality", "residual"]
    for index, name in enumerate(fitted.param_names):
        np.testing.assert_allclose(maps[name].data, getattr(fitted, name).value)
        np.testing.assert_allclose(maps[name].uncertainty.array, error[..., index])
    speed = C_KMS * u.km / u.s / (1402.77 * u.AA)
    expected = {
        "velocity_1": ((truth.mean_1.quantity - 1402.77 * u.AA) * speed, u.km / u.s),
        "fwhm_1": (2 * np.sqrt(2 * np.log(2)) * truth.stddev_1.quantity, u.AA),
        "fwhm_velocity_1": (2 * np.sqrt(2 * np.log(2)) * truth.stddev_1.quantity * speed, u.km / u.s),
        "integrated_intensity_1": (
            np.sqrt(2 * np.pi) * truth.amplitude_1.quantity * truth.stddev_1.quantity,
            u.DN * u.AA,
        ),
    }
    for name, (value, unit) in expected.items():
        assert maps[name].unit == unit
        z = (maps[name].data - value.to_value(unit)) / maps[name].uncertainty.array
        # The propagated 1 sigma should hold the true value in about 68% of the 400 spectra.
        assert 0.6 < np.mean(np.abs(z) < 1) < 0.76, name
    assert np.all(maps["quality"].data == FitQualityFlag.OK)
    residual = maps["residual"]
    assert residual.data.shape == cube.data.shape
    assert residual.wcs is cube.wcs
    np.testing.assert_allclose(
        residual.data,
        cube.data - np.moveaxis(fitted(WAVELENGTH[:, np.newaxis, np.newaxis]).to_value(u.DN), 0, -1),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("profile", "width", "fwhm", "area"),
    [
        ("gaussian", 0.1, 0.1 * 2.3548200450309493, 0.1 * 2.5066282746310002 * 50),
        ("lorentzian", 0.2, 0.2, np.pi * 0.2 * 50 / 2),
    ],
)
def test_maps_from_fit_formulas(profile, width, fwhm, area):
    model = profiles_on_background(
        np.full((1, 2, 3), 1402.77) * u.AA,
        amplitudes=np.full((1, 2, 3), 50) * u.DN,
        widths=np.full((1, 2, 3), width) * u.AA,
        background_level=0 * u.DN,
        profile=profile,
    )
    cube = make_test_spectrogram_cube(np.zeros((2, 3, 60)), WAVELENGTH)
    maps = maps_from_fit(model, cube)
    np.testing.assert_allclose(maps["velocity_1"].data, 0, atol=1e-9)
    np.testing.assert_allclose(maps["fwhm_1"].data, fwhm)
    np.testing.assert_allclose(maps["integrated_intensity_1"].data, area)
    assert all(maps[name].uncertainty is None for name in maps if name != "residual")


def test_maps_from_fit_flags_no_fit_and_at_bound():
    rng = np.random.default_rng(3)
    truth, cube = _si_iv_cube(rng)
    data = cube.data.copy()
    data[0, 0] = np.nan  # no fit
    model = si_iv_1403_model(cube)
    # Capping the amplitude below that of the brighter lines pins their fits to the cap.
    cap = np.median(truth.amplitude_1.value)
    model.amplitude_1.bounds = (0, cap)
    model.amplitude_1 = np.minimum(model.amplitude_1.value, cap) * u.DN
    fitted, fitter = _fit(model, data, cube.uncertainty.array, unit=cube.unit)
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    assert np.isnan(maps["mean_1"].uncertainty.array[0, 0])
    quality = maps["quality"].data
    assert quality[0, 0] == FitQualityFlag.NO_FIT
    bright = truth.amplitude_1.value > 1.2 * cap
    faint = truth.amplitude_1.value < 0.8 * cap
    bright[0, 0] = faint[0, 0] = False
    assert np.mean(quality[bright] == FitQualityFlag.AT_BOUND) > 0.95
    assert np.mean(quality[faint] == FitQualityFlag.OK) > 0.95


@pytest.mark.parametrize(
    ("fitter_class", "expected"),
    # LevMarLSQFitter's fit information has no success, so its fits are never flagged.
    [(TRFLSQFitter, FitQualityFlag.NOT_CONVERGED), (LevMarLSQFitter, FitQualityFlag.OK)],
)
def test_maps_from_fit_flags_convergence(fitter_class, expected):
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, fitter = _fit(
        si_iv_1403_model(cube),
        cube.data,
        cube.uncertainty.array,
        unit=cube.unit,
        fitter=fitter_class(),
        fitter_kwargs={"maxiter": 1},
    )
    assert np.all(maps_from_fit(fitted, cube, fitter=fitter)["quality"].data == expected)


def test_maps_from_fit_bound_tolerance_is_relative():
    # In meters the center is 1e-11 from its bounds, which only a relative tolerance leaves unflagged.
    model = profiles_on_background(
        np.full((1, 2, 3), 1402.77e-10) * u.m,
        amplitudes=np.full((1, 2, 3), 50) * u.DN,
        widths=np.full((1, 2, 3), 0.05e-10) * u.m,
        background_level=5 * u.DN,
        bounds={"mean_1": [1402.67, 1402.87] * u.AA, "stddev_1": [0.01, 0.1] * u.AA, "amplitude_1": [0, 100] * u.DN},
    )
    cube = make_test_spectrogram_cube(np.zeros((2, 3, 60)), WAVELENGTH)
    assert np.all(maps_from_fit(model, cube)["quality"].data == FitQualityFlag.OK)


@pytest.mark.parametrize(("wavelength_axis", "wcs_axes"), [(0, [2, 3, 1]), (1, [2, 1, 3])])
def test_maps_from_fit_aligns_spatial_axes(wavelength_axis, wcs_axes):
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, _ = _fit(si_iv_1403_model(cube), cube.data, cube.uncertainty.array, unit=cube.unit)
    cube = type(cube)(np.moveaxis(cube.data, 2, wavelength_axis), cube.wcs.sub(wcs_axes), unit=cube.unit)
    maps = maps_from_fit(fitted, cube)
    map_axes = (0, 1)
    residual_axes = tuple(axis for axis in range(3) if axis != wavelength_axis)
    assert maps.aligned_axes["mean_1"] == map_axes
    assert maps.aligned_axes["residual"] == residual_axes
    np.testing.assert_allclose(
        maps["residual"].data,
        cube.data - np.moveaxis(fitted(WAVELENGTH[:, None, None]).to_value(cube.unit), 0, wavelength_axis),
        atol=1e-9,
    )


def test_maps_from_fit_preserves_residual_coordinates():
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, _ = _fit(si_iv_1403_model(cube), cube.data, cube.uncertainty.array, unit=cube.unit)
    times = Time("2020-01-01") + np.arange(SHAPE[0]) * u.s
    cube.extra_coords.add("time", 0, times, physical_types="time")
    cube.global_coords.add("observer", "custom:observer", "IRIS")
    residual = maps_from_fit(fitted, cube)["residual"]
    assert residual.global_coords["observer"] == "IRIS"
    assert residual.extra_coords._ndcube is residual
    assert cube.extra_coords._ndcube is cube
    np.testing.assert_array_equal(residual.axis_world_coords("time", wcs=residual.extra_coords)[0], times)


def test_maps_from_fit_flags_masked_input():
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, _ = _fit(si_iv_1403_model(cube), cube.data, cube.uncertainty.array, unit=cube.unit)
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    cube.mask[0, 1, 5] = True
    quality = maps_from_fit(fitted, cube)["quality"].data
    assert quality[0, 1] == FitQualityFlag.MASKED_INPUT
    assert quality[1, 1] == FitQualityFlag.OK
    for mask, expected in ((False, FitQualityFlag.OK), (True, FitQualityFlag.MASKED_INPUT)):  # Scalar masks.
        cube.mask = mask
        assert np.all(maps_from_fit(fitted, cube)["quality"].data == expected)


def test_maps_from_fit_without_any_covariance():
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, fitter = _fit(
        si_iv_1403_model(cube), np.full_like(cube.data, np.nan), cube.uncertainty.array, unit=cube.unit
    )
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    assert np.all(maps["quality"].data == FitQualityFlag.NO_FIT)
    assert maps["mean_1"].uncertainty is None
    assert maps["integrated_intensity_1"].uncertainty is None


def test_maps_from_fit_fixed_parameters_and_rest_wavelength():
    rng = np.random.default_rng(4)
    _, cube = _si_iv_cube(rng)
    model = si_iv_1403_model(cube)
    model.amplitude_0.fixed = True
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, unit=cube.unit)
    maps = maps_from_fit(fitted, cube, fitter=fitter, rest_wavelength=1402.0 * u.AA)
    assert np.all(np.isnan(maps["amplitude_0"].uncertainty.array))
    assert np.all(np.isfinite(maps["mean_1"].uncertainty.array))
    # Against 1402.0 Å instead of the documented 1402.77 Å.
    np.testing.assert_allclose(np.median(maps["velocity_1"].data), 0.77 / 1402.0 * C_KMS, rtol=0.01)
    wider = make_test_spectrogram_cube(np.zeros((2, 3, 400)), (1395.0 + 0.026 * np.arange(400)) * u.AA)
    with pytest.raises(ValueError, match="pass rest_wavelength"):
        maps_from_fit(model, wider)
    with pytest.raises(ValueError, match="no unit"):
        maps_from_fit(profiles_on_background([1402.77] * u.AA), cube)
    with pytest.raises(ValueError, match="fit_info=True"):
        maps_from_fit(fitted, cube, fitter=TRFLSQFitter())


def _width_map(sigmas, *, error=None, mask=None):
    """
    A map of Gaussian full widths at half maximum, in Å, from standard deviations.
    """
    shape = np.shape(sigmas)
    template = make_test_spectrogram_cube(np.zeros((*shape, 2)), WAVELENGTH[:2])[..., 0]
    fwhm = 2 * np.sqrt(2 * np.log(2)) * np.asarray(sigmas, dtype=float)
    uncertainty = None if error is None else StdDevUncertainty(np.full(shape, error))
    return SpectrogramCube(fwhm, template.wcs, uncertainty=uncertainty, unit=u.AA, mask=mask)


def test_non_thermal_velocity_formula():
    # Standard deviations of 0.05 and 0.01 Å, and one whose width is all instrumental.
    sigma = np.array([[0.05, 0.01, 0.02]])
    widths = 2 * np.sqrt(2 * np.log(2)) * sigma
    result = non_thermal_velocity(
        _width_map(sigma, error=1e-3),
        1402.77 * u.AA,
        instrumental_fwhm=widths[0, 2] * u.AA,
        thermal_fwhm=0.02 * u.AA,
    )
    velocity = result["non_thermal_velocity"]
    assert velocity.unit == u.km / u.s
    radicand = widths**2 - widths[0, 2] ** 2 - 0.02**2
    expected = C_KMS / (1402.77 * np.sqrt(4 * np.log(2))) * np.sqrt(np.where(radicand > 0, radicand, np.nan))
    np.testing.assert_allclose(velocity.data, expected)
    # dv/dW by a central difference, for the FWHM error of 2 sqrt(2 ln 2) 1e-3 Å.
    step = 1e-6
    shifted = [
        non_thermal_velocity(
            _width_map(sigma + sign * step),
            1402.77 * u.AA,
            instrumental_fwhm=widths[0, 2] * u.AA,
            thermal_fwhm=0.02 * u.AA,
        )["non_thermal_velocity"].data
        for sign in (1, -1)
    ]
    # The step is in the standard deviation, so dv/dW is the difference over the step in FWHM.
    derivative = (shifted[0] - shifted[1]) / (2 * step * 2 * np.sqrt(2 * np.log(2)))
    np.testing.assert_allclose(velocity.uncertainty.array[0, 0], derivative[0, 0] * 1e-3, rtol=1e-5)
    quality = result["quality"].data
    np.testing.assert_array_equal(
        quality, [[NonThermalQualityFlag.OK, NonThermalQualityFlag.TOO_NARROW, NonThermalQualityFlag.TOO_NARROW]]
    )
    assert np.isnan(velocity.data[0, 1])


def test_non_thermal_velocity_defaults():
    sigma = np.array([[0.05, np.nan]])
    # Si IV peaks at log T = 4.9 in CHIANTI's ionization equilibrium.
    result = non_thermal_velocity(_width_map(sigma), 1402.77 * u.AA, ion="Si IV", temperature=10**4.9 * u.K)
    thermal = (
        np.sqrt(4 * np.log(2))
        * 1402.77
        * u.AA
        * np.sqrt(2 * constants.k_B * 10**4.9 * u.K / ATOMIC_MASS["Si"])
        / constants.c
    )
    expected = non_thermal_velocity(
        _width_map(sigma), 1402.77 * u.AA, instrumental_fwhm=INSTRUMENTAL_FWHM["FUV2"], thermal_fwhm=thermal
    )
    np.testing.assert_allclose(result["non_thermal_velocity"].data, expected["non_thermal_velocity"].data)
    assert result["quality"].data[0, 1] == NonThermalQualityFlag.NO_DATA
    # A hotter ion leaves less width for the non-thermal velocity.
    hotter = non_thermal_velocity(_width_map(sigma), 1402.77 * u.AA, ion="Si IV", temperature=2e5 * u.K)
    assert hotter["non_thermal_velocity"].data[0, 0] < result["non_thermal_velocity"].data[0, 0]


def test_non_thermal_velocity_non_positive_widths():
    widths = _width_map([[0.05, -0.05, 0, np.nan, -0.05]], error=1e-3, mask=[[False, False, False, False, True]])
    result = non_thermal_velocity(widths, 1402.77 * u.AA, thermal_fwhm=0.03 * u.AA)
    velocity = result["non_thermal_velocity"]
    assert np.isfinite(velocity.data[0, 0])
    assert np.isfinite(velocity.uncertainty.array[0, 0])
    assert np.all(np.isnan(velocity.data[0, 1:]))
    assert np.all(np.isnan(velocity.uncertainty.array[0, 1:]))
    np.testing.assert_array_equal(velocity.mask, [[False, True, True, True, True]])
    np.testing.assert_array_equal(
        result["quality"].data,
        [
            [
                NonThermalQualityFlag.OK,
                NonThermalQualityFlag.TOO_NARROW,
                NonThermalQualityFlag.TOO_NARROW,
                NonThermalQualityFlag.NO_DATA,
                NonThermalQualityFlag.NO_DATA,
            ]
        ],
    )


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({}, "Pass ion"),
        ({"ion": "Si IV"}, "temperature"),
        ({"ion": "Xx IV", "temperature": 1e5 * u.K}, "No atomic mass"),
        ({"ion": "Si IV", "wavelength": 1500 * u.AA}, "outside the IRIS passbands"),
    ],
)
def test_non_thermal_velocity_errors(keywords, match):
    wavelength = keywords.pop("wavelength", 1402.77 * u.AA)
    with pytest.raises(ValueError, match=match):
        non_thermal_velocity(_width_map(np.array([[0.05, 0.06]])), wavelength, **keywords)
