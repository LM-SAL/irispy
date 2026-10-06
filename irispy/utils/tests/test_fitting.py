import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.modeling.fitting import TRFLSQFitter, parallel_fit_dask
from astropy.nddata import StdDevUncertainty

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.fitting import gaussians_on_background, mg_ii_model, si_iv_1403_model

SHAPE = (20, 20)
WAVELENGTH = (1402.0 + 0.026 * np.arange(60)) * u.AA
MG_FEATURES_FILE = "mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits"
C_KMS = constants.c.to_value(u.km / u.s)


def _fit(model, data, sigma, wavelength=WAVELENGTH, unit=u.ct):
    fitter = TRFLSQFitter()
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
def test_gaussians_on_background_recovers_parameters(profile, background):
    rng = np.random.default_rng(42)
    centers = np.stack([1402.70 + rng.normal(0, 0.01, SHAPE), 1402.88 + rng.normal(0, 0.01, SHAPE)]) * u.AA
    amplitudes = rng.uniform(80, 150, (2, *SHAPE)) * u.ct
    widths = rng.uniform(0.03, 0.05, (2, *SHAPE)) * u.AA
    level = np.full(SHAPE, 5.0) * u.ct
    truth = gaussians_on_background(
        centers, amplitudes=amplitudes, widths=widths, background=background, background_level=level, profile=profile
    )
    data, sigma = _observe(truth, rng)
    start = gaussians_on_background(
        centers + 0.01 * u.AA,
        amplitudes=0.8 * amplitudes,
        widths=1.2 * widths,
        background=background,
        background_level=0.6 * level,
        profile=profile,
    )
    fitted, fitter = _fit(start, data, sigma)
    _assert_recovered(fitted, truth, _errors(fitter), data, sigma)


def test_gaussians_on_background_array_starts_broadcast():
    rng = np.random.default_rng(0)
    centers = (1402.77 + rng.normal(0, 0.02, SHAPE)) * u.AA
    truth = gaussians_on_background(
        centers[np.newaxis], amplitudes=[100] * u.ct, widths=[0.04] * u.AA, background_level=2 * u.ct
    )
    assert truth.mean_1.shape == SHAPE
    assert truth.amplitude_1.shape == ()
    data, sigma = _observe(truth, rng)
    # Scalar starts mixed with per-spectrum centres.
    start = gaussians_on_background(
        centers[np.newaxis] + 0.01 * u.AA, amplitudes=[80] * u.ct, widths=[0.05] * u.AA, background_level=1 * u.ct
    )
    fitted, _ = _fit(start, data, sigma)
    assert fitted.mean_1.shape == SHAPE
    assert fitted.amplitude_1.shape == SHAPE
    np.testing.assert_allclose(fitted.mean_1.quantity.to_value(u.AA), centers.to_value(u.AA), atol=0.01)


def test_gaussians_on_background_bounds_hold():
    rng = np.random.default_rng(1)
    truth = gaussians_on_background(
        [1402.77] * u.AA, amplitudes=[100] * u.ct, widths=[0.04] * u.AA, background_level=np.full(SHAPE, 2.0) * u.ct
    )
    data, sigma = _observe(truth, rng)
    # Bounds in another unit than the centres are converted to it.
    bounds = {"mean_1": (140.26, 140.27) * u.nm, "amplitude_1": (0, 50) * u.ct, "amplitude_0": (None, 1 * u.ct)}
    start = gaussians_on_background([1402.65] * u.AA, amplitudes=[40] * u.ct, widths=[0.04] * u.AA, bounds=bounds)
    np.testing.assert_allclose(start.mean_1.bounds, (1402.6, 1402.7))
    assert start.stddev_1.bounds[0] > 0
    fitted, _ = _fit(start, data, sigma)
    _assert_within_bounds(fitted, start)
    # The true centre lies outside the bounds, so most fits stop on the nearer one.
    np.testing.assert_allclose(np.median(fitted.mean_1.value), 1402.7)


@pytest.mark.parametrize(("keyword", "value"), [("profile", "voigt"), ("background", "quadratic")])
def test_gaussians_on_background_rejects_unknown_options(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        gaussians_on_background([1402.77] * u.AA, **{keyword: value})


def _cube(truth, wavelength, rng):
    data, sigma = _observe(truth, rng, wavelength, u.DN)
    return make_test_spectrogram_cube(data, wavelength, uncertainty=StdDevUncertainty(sigma))


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_si_iv_1403_model_recovers_parameters(profile):
    rng = np.random.default_rng(7)
    truth = gaussians_on_background(
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
    # so that the line centre is about 60% of the peaks, as in Level 2 data.
    truth = gaussians_on_background(
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
    # of this disk-centre raster should fall well inside 20-50 km/s.
    separation = (fitted.mean_2.quantity - fitted.mean_1.quantity) / (2796.352 * u.AA) * C_KMS * u.km / u.s
    assert 20 < np.nanmedian(separation.value) < 50


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
