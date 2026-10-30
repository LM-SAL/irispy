import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.coordinates import SpectralCoord
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter, LMLSQFitter, TRFLSQFitter, parallel_fit_dask
from astropy.nddata import StdDevUncertainty
from astropy.table import QTable
from astropy.time import Time

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import ATOMIC_MASS, INSTRUMENTAL_FWHM
from irispy.utils.fitting import FitQualityFlag, line_model, maps_from_fit, non_thermal_velocity, starts_from_model

SHAPE = (20, 20)
WAVELENGTH = (1402.0 + 0.026 * np.arange(60)) * u.AA
MG_FEATURES_FILE = "mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits"
V34_FILE = "v34/iris_l2_20260206_210853_3460104433_raster_t000_r00000_test.fits"
C_KMS = constants.c.to_value(u.km / u.s)


def _fit(model, data, sigma, wavelength=WAVELENGTH, unit=u.ct, *, fitter=None, fit_info=True, **kwargs):
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
        fit_info=fit_info,
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


def _cube(truth, wavelength, rng):
    data, sigma = _observe(truth, rng, wavelength, u.DN)
    return make_test_spectrogram_cube(data, wavelength, uncertainty=StdDevUncertainty(sigma))


def test_line_model():
    model = line_model(["Si IV 1394", "C II 1336", "Mg II k", "Mg II h"])
    assert model.submodel_names == ("background", "Si IV 1394", "C II 1336", "k2v", "k2r", "h2v", "h2r")
    k, h = 2796.352, 2803.53
    means = [1393.76, 1335.7079, k - 0.15, k + 0.15, h - 0.15, h + 0.15]
    np.testing.assert_allclose([getattr(model, f"mean_{index}").value for index in range(1, 7)], means)
    for index in range(1, 7):
        assert getattr(model, f"mean_{index}").unit == u.AA
        assert getattr(model, f"stddev_{index}").quantity == 0.05 * u.AA
    assert model.bounds["mean_1"] == model.bounds["mean_2"] == (None, None)
    # Only the Mg II peaks are kept in emission: other lines, such as Ni I 2799, can be absorption lines.
    assert model.bounds["amplitude_1"] == model.bounds["amplitude_2"] == (None, None)
    for index, rest in ((3, k), (4, k), (5, h), (6, h)):
        np.testing.assert_allclose(model.bounds[f"mean_{index}"], rest * (1 + np.array([-40, 40]) / C_KMS))
        assert model.bounds[f"amplitude_{index}"] == (0, None)


@pytest.mark.parametrize("line", ["Si IV", "Si IV 1500", "Xx IV 1403", "C II 1335"])
def test_line_model_rejects_unknown_lines(line):
    with pytest.raises(ValueError, match="No single documented line"):
        line_model([line])


def test_line_model_recovers_si_iv():
    truth, cube = _si_iv_cube(np.random.default_rng(7))
    model = starts_from_model(line_model(["Si IV 1403"]), cube)
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, unit=cube.unit)
    _assert_recovered(fitted, truth, _errors(fitter), cube.data, cube.uncertainty.array, max_local_minima=0.01)


def test_line_model_recovers_mg_ii_peaks():
    rng = np.random.default_rng(8)
    wavelength = (2795.5 + 0.0254 * np.arange(70)) * u.AA
    # The k2v and k2r peaks 32 km/s apart, as observed in the quiet Sun (Ondratschek et al. 2024), overlapping
    # so that the line center is about 60% of the peaks, as in Level 2 data.
    centers = (np.array([2796.20, 2796.50])[:, None, None] + rng.normal(0, 0.01, (2, *SHAPE))) * u.AA
    amplitudes = rng.uniform(400, 600, (2, *SHAPE)) * u.DN
    widths = rng.uniform(0.08, 0.1, (2, *SHAPE)) * u.AA
    truth = models.Const1D(100 * u.DN)
    for amplitude, center, width in zip(amplitudes, centers, widths, strict=True):
        truth += models.Gaussian1D(amplitude, center, width)
    cube = _cube(truth, wavelength, rng)
    model = starts_from_model(line_model(["Mg II k"]), cube)
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, wavelength, cube.unit)
    _assert_recovered(
        fitted, truth, _errors(fitter), cube.data, cube.uncertainty.array, wavelength, max_local_minima=0.03
    )
    _assert_within_bounds(fitted, model)


def test_line_model_on_level_2_data():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0]
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    model = starts_from_model(line_model(["Mg II k"]), cube)
    fitted, _ = _fit(model, np.nan_to_num(cube.data.clip(min=0)), cube.uncertainty.array, wavelength, cube.unit)
    # IRIS k2 peak separations spread about a mean of 33 km/s (Ondratschek et al. 2024, Fig. 5b), so the median
    # of this disk-center raster should fall well inside 20-50 km/s.
    separation = (fitted.mean_2.quantity - fitted.mean_1.quantity) / (2796.352 * u.AA) * C_KMS * u.km / u.s
    assert 20 < np.nanmedian(separation.value) < 50


def _background(kind, level):
    return models.Const1D(level) if kind == "constant" else models.Linear1D(0 * level.unit / u.AA, level)


def _line_cube(rng, *, amplitudes=(100, 300), background="constant", background_level=5, profile=models.Gaussian1D):
    """
    A cube of one noiseless line on a background, its center, amplitude and width
    different in every spectrum.
    """
    center = (1402.77 + rng.normal(0, 0.02, SHAPE)) * u.AA
    amplitude = rng.uniform(*amplitudes, SHAPE) * u.DN
    width = rng.uniform(0.05, 0.09, SHAPE) * u.AA
    level = rng.uniform(background_level, 2 * background_level, SHAPE) * u.DN
    truth = _background(background, level) + profile(amplitude, center, width)
    data = np.moveaxis(truth(WAVELENGTH[:, np.newaxis, np.newaxis]).to_value(u.DN), 0, -1)
    return truth, make_test_spectrogram_cube(data, WAVELENGTH)


@pytest.mark.parametrize("profile", [models.Gaussian1D, models.Lorentz1D])
@pytest.mark.parametrize("background", ["constant", "linear"])
def test_starts_from_model_follows_each_spectrum(profile, background):
    truth, cube = _line_cube(np.random.default_rng(5), background=background, profile=profile)
    model = _background(background, 5 * u.DN) + profile(150 * u.DN, 1402.75 * u.AA, 0.06 * u.AA)
    if background == "linear":
        model.slope_0 = 0.5 * u.DN / u.AA
    starts = starts_from_model(model, cube)
    assert starts.param_names == model.param_names
    assert all(getattr(starts, name).shape == SHAPE for name in starts.param_names)
    # The background passes through each spectrum's 10th percentile, with its slope removed, in the middle of the window.
    slope = model[0](WAVELENGTH) - model[0](WAVELENGTH.mean())
    level = np.percentile(cube.data - slope.to_value(u.DN), 10, axis=-1)
    if background == "constant":
        np.testing.assert_allclose(starts.amplitude_0.value, level)
    else:
        np.testing.assert_allclose(starts.slope_0.value, 0.5)
        np.testing.assert_allclose(starts.intercept_0.value + starts.slope_0.value * WAVELENGTH.mean().value, level)
    # Each line starts at its brightest sample, the nearest to its center, its height above the started background
    # there, with the model's width.
    center, width = truth.param_names[-2:]
    spacing = np.diff(WAVELENGTH.value)[0]
    assert np.all(np.abs(getattr(starts, center).value - getattr(truth, center).value) < spacing / 2)
    background_at_peak = starts[0](getattr(starts, center).quantity).value
    np.testing.assert_allclose(starts.amplitude_1.value, cube.data.max(axis=-1) - background_at_peak)
    np.testing.assert_array_equal(getattr(starts, width).value, getattr(model, width).value)


@pytest.mark.parametrize(
    "model",
    [
        models.Polynomial1D(2, c0=5 * u.DN, c1=0.5 * u.DN / u.AA, c2=1e-4 * u.DN / u.AA**2)
        + models.Gaussian1D(150 * u.DN, 1402.75 * u.AA, 0.06 * u.AA),
        # A polynomial with a domain takes parameters without units, in Å and the unit of the cube.
        models.Polynomial1D(2, c0=5, c1=0.5, c2=1e-4, domain=(1402, 1404)) + models.Gaussian1D(150, 1402.75, 0.06),
    ],
)
def test_starts_from_model_follows_each_spectrum_on_a_polynomial(model):
    _, cube = _line_cube(np.random.default_rng(10))
    starts = starts_from_model(model, cube)
    assert starts.c0_0.unit == model.c0_0.unit
    for name in ("c1_0", "c2_0"):
        np.testing.assert_array_equal(getattr(starts, name).value, getattr(model, name).value)
    # The background passes through each spectrum's 10th percentile, with its higher terms removed, in the middle of
    # the window.
    angstrom = u.AA if model.c0_0.unit else 1
    terms = u.Quantity(model[0](WAVELENGTH.value * angstrom) - model[0](WAVELENGTH.mean().value * angstrom)).value
    level = np.percentile(cube.data - terms, 10, axis=-1)
    np.testing.assert_allclose(u.Quantity(starts[0](WAVELENGTH.mean().value * angstrom)).value, level)
    # The line starts at its height above the started background at the sample it starts at.
    peak = np.searchsorted(WAVELENGTH.value, starts.mean_1.value)
    at_peak = np.take_along_axis(cube.data, peak[..., np.newaxis], axis=-1)[..., 0]
    background_at_peak = u.Quantity(starts[0](starts.mean_1.value * angstrom)).value
    np.testing.assert_allclose(starts.amplitude_1.value, at_peak - background_at_peak)


@pytest.mark.parametrize(
    "background",
    [
        models.Linear1D(slope=1000),
        # A domain centered on the window keeps the coefficients small.
        models.Polynomial1D(2, c1=1000, c2=-400, domain=(1401.77, 1403.77)),
    ],
)
def test_starts_from_model_finds_lines_on_a_steep_background(background):
    truth, cube = _line_cube(np.random.default_rng(12))
    # The lines on a background as steep as that of the model, which would move their peaks and lower its level.
    cube.data += background(WAVELENGTH.value) - background(WAVELENGTH.mean().value)
    starts = starts_from_model(background + models.Gaussian1D(150, 1402.75, 0.06), cube)
    # The started background passes through the true one in the middle of the window.
    np.testing.assert_allclose(starts[0](WAVELENGTH.mean().value), truth.amplitude_0.value)
    # Each line starts at its sample nearest its center, at its true height there.
    assert np.all(np.abs(starts.mean_1.value - truth.mean_1.value) < np.diff(WAVELENGTH.value)[0] / 2)
    np.testing.assert_allclose(starts.amplitude_1.value, truth[1](starts.mean_1.value * u.AA).to_value(u.DN))


def test_starts_from_model_rejects_lines_outside_the_cube():
    _, cube = _line_cube(np.random.default_rng(13))
    with pytest.raises(ValueError, match=r"lines at 1401\.16 Å, outside the cube's 1402\.00-1403\.53 Å"):
        starts_from_model(line_model(["Si IV 1403", "O IV 1401"]), cube)


def test_starts_from_model_takes_absolute_widths():
    # A Lorentzian fit can end with a negative fwhm; reused as a start, it still finds each spectrum's peak.
    truth, cube = _line_cube(np.random.default_rng(14), profile=models.Lorentz1D)
    model = models.Const1D(5 * u.DN) + models.Lorentz1D(150 * u.DN, 1402.75 * u.AA, -0.06 * u.AA)
    starts = starts_from_model(model, cube)
    assert np.all(np.abs(starts.x_0_1.value - truth.x_0_1.value) < np.diff(WAVELENGTH.value)[0] / 2)


def test_starts_from_model_keeps_lines_without_a_peak():
    # A shallow absorption line in spectra whose 10th percentile lies in a deeper one.
    model = models.Const1D(100) + models.Gaussian1D(-80, 1402.4, 0.1) + models.Gaussian1D(-10, 1403.2, 0.06)
    cube = make_test_spectrogram_cube(np.tile(model(WAVELENGTH.value), (2, 3, 1)), WAVELENGTH)
    starts = starts_from_model(model, cube)
    assert np.all(starts.amplitude_1.value < 0)
    # No sample near the shallow line lies below the background, so it keeps its start rather than turning into
    # emission.
    np.testing.assert_array_equal(starts.amplitude_2.value, -10)
    np.testing.assert_array_equal(starts.mean_2.value, 1403.2)


@pytest.mark.parametrize(
    ("model", "operator"),
    [
        (models.Const1D(5) * models.Gaussian1D(150, 1402.77, 0.06), "'*'"),
        (models.Const1D(5) + (models.Shift(0.1) | models.Gaussian1D(150, 1402.77, 0.06)), "'|'"),
    ],
)
def test_starts_from_model_rejects_models_other_than_sums(model, operator):
    _, cube = _line_cube(np.random.default_rng(13))
    with pytest.raises(ValueError, match="sum of backgrounds and lines") as error:
        starts_from_model(model, cube)
    assert operator in str(error.value)


@pytest.mark.parametrize("constraint", ["fixed", "tied"])
def test_starts_from_model_keeps_polynomial_constraints(constraint):
    _, cube = _line_cube(np.random.default_rng(11))
    model = models.Polynomial1D(2, c0=5, c1=0.5, c2=-0.01) + models.Gaussian1D(150, 1402.75, 0.06)
    if constraint == "fixed":
        model.c0_0.fixed = True
    else:
        model.c0_0.tied = lambda model: model.c1_0 + 4.5
    starts = starts_from_model(model, cube)
    assert starts.fixed == model.fixed
    assert starts.tied == model.tied
    for name in ("c0_0", "c1_0", "c2_0"):
        np.testing.assert_array_equal(getattr(starts, name).value, getattr(model, name).value)


def test_starts_from_model_keeps_absorption():
    truth, cube = _line_cube(np.random.default_rng(6), amplitudes=(-60, -20), background_level=100)
    model = models.Const1D(100 * u.DN) + models.Gaussian1D(-40 * u.DN, 1402.75 * u.AA, 0.06 * u.AA)
    starts = starts_from_model(model, cube)
    assert np.all(np.abs(starts.mean_1.value - truth.mean_1.value) < np.diff(WAVELENGTH.value)[0] / 2)
    np.testing.assert_allclose(starts.amplitude_1.value, cube.data.min(axis=-1) - np.percentile(cube.data, 10, axis=-1))
    assert np.all(starts.amplitude_1.value < 0)


def test_starts_from_model_keeps_constraints():
    _, cube = _line_cube(np.random.default_rng(7))
    model = (
        models.Const1D(5 * u.DN)
        + models.Gaussian1D(
            150 * u.DN, 1402.77 * u.AA, 0.06 * u.AA, bounds={"mean": (1402.76, 1402.78), "amplitude": (0, 200)}
        )
        + models.Gaussian1D(20 * u.DN, 1403.2 * u.AA, 0.06 * u.AA)
    )
    model.amplitude_0.fixed = True
    model.mean_2.tied = lambda model: model.mean_1 + 0.43
    starts = starts_from_model(model, cube)
    assert starts.fixed == model.fixed
    assert starts.tied == model.tied
    assert starts.bounds == model.bounds
    np.testing.assert_array_equal(starts.amplitude_0.value, 5)
    np.testing.assert_array_equal(starts.mean_2.value, 1403.2)
    for name in ("mean_1", "amplitude_1"):
        lower, upper = model.bounds[name]
        values = getattr(starts, name).value
        assert np.all((values >= lower) & (values <= upper)), name
        assert np.any(values == upper), name
    assert np.any(starts.mean_1.value == 1402.76)


def test_starts_from_model_leaves_out_missing_samples():
    _, cube = _line_cube(np.random.default_rng(8))
    # Without units, the model is in Å and the unit of the cube.
    model = models.Const1D(5) + models.Gaussian1D(150, 1402.77, 0.06)
    expected = starts_from_model(model, cube)
    peak = np.argmax(cube.data[0, 1])
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    cube.mask[0, 0] = True
    cube.mask[0, 1, peak] = True
    cube.data[0, 2, 25:40] = np.nan
    cube.data[0, 3, np.argmax(cube.data[0, 3])] = np.inf
    starts = starts_from_model(model, cube)
    assert starts.amplitude_0.unit is None
    # A fully masked spectrum keeps the line of the model on a NaN background, which parallel_fit_dask leaves out.
    assert np.isnan(starts.amplitude_0.value[0, 0])
    assert (starts.mean_1.value[0, 0], starts.amplitude_1.value[0, 0]) == (1402.77, 150)
    # A masked peak moves the start to the brightest sample left, beside it, and so does a non-finite one.
    for index in ((0, 1), (0, 3)):
        shift = np.abs(starts.mean_1.value[index] - expected.mean_1.value[index])
        np.testing.assert_allclose(shift, np.diff(WAVELENGTH.value)[0])
        np.testing.assert_allclose(starts.amplitude_0.value[index], expected.amplitude_0.value[index], rtol=0.1)
    # No samples left within two widths of the line.
    assert (starts.mean_1.value[0, 2], starts.amplitude_1.value[0, 2]) == (1402.77, 150)
    for name in model.param_names:
        np.testing.assert_array_equal(getattr(starts, name).value[1:], getattr(expected, name).value[1:])


def test_starts_from_model_units():
    _, cube = _line_cube(np.random.default_rng(9))
    in_angstrom = models.Const1D(5 * u.DN) + models.Gaussian1D(150 * u.DN, 1402.75 * u.AA, 0.06 * u.AA)
    in_nm = models.Const1D(5 * u.DN) + models.Gaussian1D(150 * u.DN, 140.275 * u.nm, 0.006 * u.nm)
    expected, starts = starts_from_model(in_angstrom, cube), starts_from_model(in_nm, cube)
    assert starts.mean_1.unit == u.nm
    for name in in_nm.param_names:
        assert u.allclose(getattr(starts, name).quantity, getattr(expected, name).quantity), name
    with pytest.raises(ValueError, match="array parameters"):
        starts_from_model(expected, cube)


def test_starts_from_model_accepts_a_scalar_mask():
    _, cube = _line_cube(np.random.default_rng(14))
    expected = starts_from_model(line_model(["Si IV 1403"]), cube)
    cube.mask = False
    np.testing.assert_array_equal(starts_from_model(line_model(["Si IV 1403"]), cube).parameters, expected.parameters)


def test_starts_from_model_rejects_unscaled_data():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), memmap=True)
    with pytest.raises(ValueError, match="unscaled"):
        starts_from_model(line_model(["Mg II k"]), raster["Mg II k 2796"][0])


def _k2_peaks(unit):
    """
    Gaussians for the Mg II k2v and k2r peaks on a constant background, to fit to one
    spectrum.
    """
    return (
        models.Const1D(10 * unit)
        + models.Gaussian1D(100 * unit, 2796.20 * u.AA, 0.08 * u.AA)
        + models.Gaussian1D(100 * unit, 2796.50 * u.AA, 0.08 * u.AA)
    )


def test_starts_from_model_starts_blended_lines_at_their_own_peaks():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0]
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    start = _k2_peaks(cube.unit)
    mean = np.nanmean(np.where(cube.mask, np.nan, cube.data), axis=(0, 1))
    starts = starts_from_model(TRFLSQFitter()(start, wavelength, mean * cube.unit), cube)
    # The k2v and k2r Gaussians lie within two widths of each other, and would otherwise share the brighter peak.
    assert np.all(starts.mean_1.value < starts.mean_2.value)


@pytest.mark.parametrize("tied", [False, True])
def test_starts_from_model_shares_the_peak_of_lines_with_one_center(bursts_si_iv_raster, tied):
    cube = bursts_si_iv_raster[0]
    broad = models.Gaussian1D(1 * cube.unit, 1402.77 * u.AA, 0.1 * u.AA)
    one_line = starts_from_model(models.Const1D(0 * cube.unit) + broad, cube)
    # A core 0.4 samples from the broad component, or the broad one 3 samples off with its center tied to the core's,
    # as after a fit with both free.
    core = models.Gaussian1D(3 * cube.unit, (1402.77 if tied else 1402.775) * u.AA, 0.03 * u.AA)
    if tied:
        broad = models.Gaussian1D(1 * cube.unit, (1402.77 + 3 * 0.01272) * u.AA, 0.1 * u.AA)
    model = models.Const1D(0 * cube.unit) + core + broad
    if tied:
        model.mean_2.tied = lambda model: model.mean_1
    starts = starts_from_model(model, cube)
    # The peak that the broad component alone starts at, in the ratio of the model's amplitudes.
    np.testing.assert_allclose(starts.amplitude_1.value, 0.75 * one_line.amplitude_1.value)
    np.testing.assert_allclose(starts.amplitude_2.value, 0.25 * one_line.amplitude_1.value)
    np.testing.assert_array_equal(starts.mean_1.value, one_line.mean_1.value)
    if not tied:
        np.testing.assert_array_equal(starts.mean_2.value, one_line.mean_1.value)


def test_starts_from_model_shares_what_fixed_amplitudes_leave_of_the_peak(bursts_si_iv_raster):
    cube = bursts_si_iv_raster[0]
    broad = models.Gaussian1D(1 * cube.unit, 1402.77 * u.AA, 0.1 * u.AA)
    peak = starts_from_model(models.Const1D(0 * cube.unit) + broad, cube).amplitude_1.value
    model = models.Const1D(0 * cube.unit) + models.Gaussian1D(3 * cube.unit, 1402.77 * u.AA, 0.03 * u.AA) + broad
    model.amplitude_2.fixed = True
    starts = starts_from_model(model, cube)
    # The core starts at the rest of the peak, and keeps its amplitude in model where the fixed broad one takes it all.
    np.testing.assert_allclose(starts.amplitude_1.value, np.where(peak > 1, peak - 1, 3))
    np.testing.assert_array_equal(starts.amplitude_2.value, 1)


def test_starts_from_model_shares_a_peak_only_within_a_sample(bursts_si_iv_raster):
    cube = bursts_si_iv_raster[0]
    # Lines 0.75 and 1.5 samples from the first: the second shares its peak, and the third, a sample and a half from the
    # first, searches its own samples.
    model = models.Const1D(0 * cube.unit)
    for offset in (0, 0.75, 1.5):
        model += models.Gaussian1D(1 * cube.unit, (1402.77 + offset * 0.01272) * u.AA, 0.05 * u.AA)
    starts = starts_from_model(model, cube)
    np.testing.assert_array_equal(starts.mean_1.value, starts.mean_2.value)
    assert np.all(starts.mean_3.value != starts.mean_1.value)


def test_starts_from_model_keeps_components_without_unit_support(bursts_si_iv_raster):
    cube = bursts_si_iv_raster[0]

    def flat(x, level=0.0):
        return level * np.ones_like(x)

    line = models.Const1D(0) + models.Gaussian1D(1, 1402.77, 0.1)
    starts = starts_from_model(line + models.custom_model(flat)(2.0), cube)
    np.testing.assert_array_equal(starts.level_2.value, 2)
    np.testing.assert_array_equal(starts.amplitude_1.value, starts_from_model(line, cube).amplitude_1.value)


@pytest.mark.parametrize("background", ["constant", "polynomial"])
def test_starts_from_model_on_level_2_data(background):
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0][:, 700:740]  # Rows 731 to 739 are masked in full.
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    data = np.where(cube.mask, np.nan, cube.data)
    start = _k2_peaks(cube.unit)
    if background == "polynomial":
        polynomial = models.Polynomial1D(2, c0=10 * cube.unit, c1=0 * cube.unit / u.AA, c2=0 * cube.unit / u.AA**2)
        start = polynomial + start[1] + start[2]
    # A model fitted to one spectrum, then to every spectrum with its masked samples left out.
    model = TRFLSQFitter()(start, wavelength, data[1, 0] * cube.unit)
    fitted, fitter = _fit(
        starts_from_model(model, cube),
        data,
        cube.uncertainty.array,
        wavelength,
        cube.unit,
        fitter_kwargs={"filter_non_finite": True},
    )
    quality = maps_from_fit(fitted, cube, fitter=fitter)["quality"].data
    masked = np.all(cube.mask, axis=-1)
    assert np.all(quality[masked] == FitQualityFlag.NO_FIT)
    assert np.mean(quality[~masked] == FitQualityFlag.OK) > 0.8


def _si_iv_cube(rng, profile=models.Gaussian1D):
    center = (1402.77 + rng.normal(0, 0.02, SHAPE)) * u.AA
    amplitude = rng.uniform(100, 300, SHAPE) * u.DN
    width = rng.uniform(0.05, 0.09, SHAPE) * u.AA
    truth = models.Const1D(5 * u.DN) + profile(amplitude, center, width)
    return truth, _cube(truth, WAVELENGTH, rng)


def test_maps_from_fit_recovers_derived_maps():
    rng = np.random.default_rng(7)
    truth, cube = _si_iv_cube(rng)
    fitted, fitter = _fit(
        starts_from_model(line_model(["Si IV 1403"]), cube), cube.data, cube.uncertainty.array, unit=cube.unit
    )
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
        (models.Gaussian1D, 0.1, 0.1 * 2.3548200450309493, 0.1 * 2.5066282746310002 * 50),
        (models.Lorentz1D, 0.2, 0.2, np.pi * 0.2 * 50 / 2),
    ],
)
def test_maps_from_fit_formulas(profile, width, fwhm, area):
    model = models.Const1D(0 * u.DN) + profile(
        np.full((2, 3), 50) * u.DN, np.full((2, 3), 1402.77) * u.AA, np.full((2, 3), width) * u.AA
    )
    data = np.moveaxis(model(WAVELENGTH[:, None, None]).to_value(u.DN), 0, -1)
    fitted, fitter = _fit(model, data, np.ones_like(data), unit=u.DN)
    maps = maps_from_fit(fitted, make_test_spectrogram_cube(data, WAVELENGTH), fitter=fitter)
    np.testing.assert_allclose(maps["velocity_1"].data, 0, atol=1e-9)
    np.testing.assert_allclose(maps["fwhm_1"].data, fwhm)
    np.testing.assert_allclose(maps["integrated_intensity_1"].data, area)


def test_maps_from_fit_takes_absolute_widths():
    # A Lorentzian is even in its width, so a fit started at a negative one ends at a negative one.
    _, cube = _si_iv_cube(np.random.default_rng(3), models.Lorentz1D)
    maps = []
    for width in (0.15, -0.15):
        start = starts_from_model(models.Const1D(5) + models.Lorentz1D(150, 1402.77, width), cube)
        fitted, fitter = _fit(start, cube.data, cube.uncertainty.array, unit=cube.unit)
        maps.append(maps_from_fit(fitted, cube, fitter=fitter))
    positive, negative = maps
    for name in ("fwhm_1", "fwhm_velocity_1", "integrated_intensity_1"):
        np.testing.assert_allclose(negative[name].data, positive[name].data, rtol=1e-4, err_msg=name)
        np.testing.assert_allclose(
            negative[name].uncertainty.array, positive[name].uncertainty.array, rtol=1e-4, err_msg=name
        )


def test_maps_from_fit_flags_no_fit_and_at_bound():
    rng = np.random.default_rng(3)
    truth, cube = _si_iv_cube(rng)
    data = cube.data.copy()
    data[0, 0] = np.nan  # no fit
    model = starts_from_model(line_model(["Si IV 1403"]), cube)
    # Capping the amplitude below that of the brighter lines pins their fits to the cap.
    cap = np.median(truth.amplitude_1.value)
    model.amplitude_1.bounds = (0, cap)
    model.amplitude_1 = np.minimum(model.amplitude_1.value, cap)
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
        starts_from_model(line_model(["Si IV 1403"]), cube),
        cube.data,
        cube.uncertainty.array,
        unit=cube.unit,
        fitter=fitter_class(),
        fitter_kwargs={"maxiter": 1},
    )
    assert np.all(maps_from_fit(fitted, cube, fitter=fitter)["quality"].data == expected)


def test_maps_from_fit_bound_tolerance_is_relative():
    # In meters the center is 1e-11 from its bounds, which only a relative tolerance leaves unflagged;
    # LMLSQFitter reports no active bounds, which TRFLSQFitter would here, within its own tolerance.
    model = models.Const1D(5 * u.DN) + models.Gaussian1D(
        np.full((2, 3), 50) * u.DN,
        np.full((2, 3), 1402.77e-10) * u.m,
        np.full((2, 3), 0.05e-10) * u.m,
        bounds={"amplitude": (0, 100), "mean": (1402.67e-10, 1402.87e-10), "stddev": (0.01e-10, 0.1e-10)},
    )
    data = np.moveaxis(model(WAVELENGTH[:, None, None]).to_value(u.DN), 0, -1)
    fitted, fitter = _fit(model, data, np.ones_like(data), WAVELENGTH.to(u.m), u.DN, fitter=LMLSQFitter())
    maps = maps_from_fit(fitted, make_test_spectrogram_cube(data, WAVELENGTH), fitter=fitter)
    assert np.all(maps["quality"].data == FitQualityFlag.OK)


# TRFLSQFitter stops just inside a bound and LMLSQFitter beyond it.
@pytest.mark.parametrize("fitter_class", [TRFLSQFitter, LMLSQFitter])
def test_maps_from_fit_flags_bounds_per_spectrum(fitter_class):
    truth, cube = _si_iv_cube(np.random.default_rng(3))
    background = truth.amplitude_0.value
    # Dips, which take the amplitude to its bound of 0, and one line bright enough to fit an amplitude of about 1e10.
    dips = np.zeros(SHAPE, dtype=bool)
    dips[1::4, 1::4] = True
    cube.data[dips] = 2 * background - cube.data[dips]
    cube.data[0, 0] = background + 1e8 * (cube.data[0, 0] - background)
    cube.uncertainty.array[0, 0] *= 1e8
    model = starts_from_model(line_model(["Si IV 1403"]), cube)
    model.amplitude_1.bounds = (0, None)
    for name in ("mean_1", "stddev_1"):
        setattr(model, name, getattr(truth, name).quantity)
        model.fixed[name] = True
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, unit=cube.unit, fitter=fitter_class())
    assert fitted.amplitude_1.value[0, 0] > 1e9
    quality = maps_from_fit(fitted, cube, fitter=fitter)["quality"].data
    assert np.all(quality[dips] == FitQualityFlag.AT_BOUND)
    assert np.all(quality[~dips] == FitQualityFlag.OK)


@pytest.mark.parametrize(("wavelength_axis", "wcs_axes"), [(0, [2, 3, 1]), (1, [2, 1, 3])])
def test_maps_from_fit_aligns_spatial_axes(wavelength_axis, wcs_axes):
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, fitter = _fit(
        starts_from_model(line_model(["Si IV 1403"]), cube), cube.data, cube.uncertainty.array, unit=cube.unit
    )
    cube = type(cube)(np.moveaxis(cube.data, 2, wavelength_axis), cube.wcs.sub(wcs_axes), unit=cube.unit)
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    map_axes = (0, 1)
    residual_axes = tuple(axis for axis in range(3) if axis != wavelength_axis)
    assert maps.aligned_axes["mean_1"] == map_axes
    assert maps.aligned_axes["residual"] == residual_axes
    # The WCS of the moved cube gives WAVELENGTH to within 1e-12 Å.
    np.testing.assert_allclose(
        maps["residual"].data,
        cube.data - np.moveaxis(fitted(WAVELENGTH[:, None, None]).to_value(cube.unit), 0, wavelength_axis),
        atol=1e-8,
    )


def test_maps_from_fit_preserves_residual_coordinates():
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, fitter = _fit(
        starts_from_model(line_model(["Si IV 1403"]), cube), cube.data, cube.uncertainty.array, unit=cube.unit
    )
    times = Time("2020-01-01") + np.arange(SHAPE[0]) * u.s
    cube.extra_coords.add("time", 0, times, physical_types="time")
    cube.global_coords.add("observer", "custom:observer", "IRIS")
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    residual = maps["residual"]
    assert residual.global_coords["observer"] == "IRIS"
    assert residual.extra_coords._ndcube is residual
    assert cube.extra_coords._ndcube is cube
    np.testing.assert_array_equal(residual.axis_world_coords("time", wcs=residual.extra_coords)[0], times)
    residual.meta["note"] = "changed"
    residual.mask[...] = True
    assert "note" not in cube.meta
    assert not cube.mask.any()
    for name in maps:
        assert ("fit_rest_wavelength" in maps[name].meta) == name.startswith(("velocity_", "fwhm_velocity_"))


def test_maps_from_fit_without_residual():
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, fitter = _fit(
        starts_from_model(line_model(["Si IV 1403"]), cube), cube.data, cube.uncertainty.array, unit=cube.unit
    )
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    without = maps_from_fit(fitted, cube, fitter=fitter, residual=False)
    assert list(maps.keys()) == [*without.keys(), "residual"]
    for name in without:
        assert without.aligned_axes[name] == maps.aligned_axes[name]
        assert without[name].unit == maps[name].unit
        np.testing.assert_array_equal(without[name].data, maps[name].data)
        if maps[name].uncertainty is not None:
            np.testing.assert_array_equal(without[name].uncertainty.array, maps[name].uncertainty.array)


def test_maps_from_fit_flags_masked_input():
    _, cube = _si_iv_cube(np.random.default_rng(3))
    fitted, fitter = _fit(
        starts_from_model(line_model(["Si IV 1403"]), cube), cube.data, cube.uncertainty.array, unit=cube.unit
    )
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    cube.mask[0, 1, 5] = True
    quality = maps_from_fit(fitted, cube, fitter=fitter)["quality"].data
    assert quality[0, 1] == FitQualityFlag.MASKED_INPUT
    assert quality[1, 1] == FitQualityFlag.OK
    # Scalar masks; a cube masked in full has no usable sample.
    for mask, expected in ((False, FitQualityFlag.OK), (True, FitQualityFlag.NO_FIT)):
        cube.mask = mask
        assert np.all(maps_from_fit(fitted, cube, fitter=fitter)["quality"].data == expected)


def test_maps_from_fit_flags_spectra_without_usable_samples():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0][:, 700:740]  # Rows 731 to 739 are masked in full.
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    good = np.isfinite(cube.data) & ~cube.mask
    # One start for every spectrum and zero weights for the masked samples, with which the fits of the
    # spectra masked in full end at their start.
    start = TRFLSQFitter()(_k2_peaks(cube.unit), wavelength, cube.data[1, 0] * cube.unit)
    sigma = np.where(good, cube.uncertainty.array, np.inf)
    fitted, fitter = _fit(start, np.where(good, cube.data, 0), sigma, wavelength, cube.unit)
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    empty = ~np.any(good, axis=-1)
    np.testing.assert_array_equal(maps["quality"].data == FitQualityFlag.NO_FIT, empty)
    assert np.all(np.isnan(maps["integrated_intensity_1"].data[empty]))
    assert np.all(np.isnan(maps["mean_1"].uncertainty.array[empty]))


def test_maps_from_fit_raises_only_when_no_spectrum_with_data_was_fitted():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0][:, 700:740]  # Rows 731 to 739 are masked in full.
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)

    def fit(cube, wavelength):
        return _fit(
            starts_from_model(line_model(["Mg II k"]), cube),
            np.where(cube.mask, np.nan, cube.data),
            cube.uncertainty.array,
            wavelength,
            cube.unit,
            fitter_kwargs={"filter_non_finite": True},
        )

    # astropy converts the means to nm but not their bounds, so every fit fails.
    fitted, fitter = fit(cube, wavelength.to(u.nm))
    with pytest.raises(ValueError, match="Every fit of a spectrum with data failed"):
        maps_from_fit(fitted, cube, fitter=fitter)
    missing = cube[:, 31:]
    fitted, fitter = fit(missing, wavelength)
    maps = maps_from_fit(fitted, missing, fitter=fitter)
    assert np.all(maps["quality"].data == FitQualityFlag.NO_FIT)
    assert maps["mean_1"].uncertainty is None


def test_maps_from_fit_takes_fit_info_naming_its_keys():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0][:, 700:740]
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    quality = []
    for fit_info in (True, ("param_cov", "success", "active_mask")):
        fitted, fitter = _fit(
            starts_from_model(line_model(["Mg II k"]), cube),
            np.where(cube.mask, np.nan, cube.data),
            cube.uncertainty.array,
            wavelength,
            cube.unit,
            fit_info=fit_info,
            fitter_kwargs={"filter_non_finite": True},
        )
        maps = maps_from_fit(fitted, cube, fitter=fitter)
        assert np.isfinite(maps["mean_1"].uncertainty.array).any()
        quality.append(maps["quality"].data)
    np.testing.assert_array_equal(*quality)


def test_maps_from_fit_fixed_parameters_and_rest_wavelength():
    rng = np.random.default_rng(4)
    _, cube = _si_iv_cube(rng)
    model = starts_from_model(line_model(["Si IV 1403"]), cube)
    model.amplitude_0.fixed = True
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, unit=cube.unit)
    maps = maps_from_fit(fitted, cube, fitter=fitter, rest_wavelength=1402.0 * u.AA)
    assert np.all(np.isnan(maps["amplitude_0"].uncertainty.array))
    assert np.all(np.isfinite(maps["mean_1"].uncertainty.array))
    # Against 1402.0 Å instead of the documented 1402.77 Å.
    np.testing.assert_allclose(np.median(maps["velocity_1"].data), 0.77 / 1402.0 * C_KMS, rtol=0.01)
    with pytest.raises(ValueError, match="no unit"):
        maps_from_fit(line_model(["Si IV 1403"]), cube, fitter=fitter)
    with pytest.raises(TypeError, match="fit_info=True"):
        maps_from_fit(fitted, cube, fitter=TRFLSQFitter())


def _fit_mg_ii(model):
    """
    ``model`` fitted to every spectrum of Mg II k and h in the Mg II k 2796 window of a
    raster, and that cube.
    """
    raster = read_spectrograph_lvl2(get_test_filepath(V34_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0][:, :48]
    cube = cube.crop([SpectralCoord(2795.4 * u.AA), None, None, None], [SpectralCoord(2804.5 * u.AA), None, None, None])
    good = np.isfinite(cube.data) & ~cube.mask
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    sigma = np.where(good, cube.uncertainty.array, np.inf)
    return (*_fit(starts_from_model(model, cube), np.where(good, cube.data, 0), sigma, wavelength, cube.unit), cube)


def test_maps_from_fit_velocities_against_each_line():
    fitted, fitter, cube = _fit_mg_ii(line_model(["Mg II k", "Mg II h"]))
    maps = maps_from_fit(fitted, cube, fitter=fitter, residual=False)
    for index, rest in zip((1, 2, 3, 4), (2796.352, 2796.352, 2803.53, 2803.53), strict=True):
        speed = C_KMS / rest
        np.testing.assert_allclose(maps[f"velocity_{index}"].data, speed * (maps[f"mean_{index}"].data - rest))
        np.testing.assert_allclose(maps[f"fwhm_velocity_{index}"].data, speed * maps[f"fwhm_{index}"].data)
        assert u.allclose(maps[f"velocity_{index}"].meta["fit_rest_wavelength"], rest * u.AA)
    # A rest wavelength given for one component leaves the others at their lines.
    k2v = maps_from_fit(fitted, cube, fitter=fitter, rest_wavelength={"k2v": 2796.0 * u.AA}, residual=False)
    np.testing.assert_allclose(k2v["velocity_1"].data, (maps["mean_1"].data / 2796.0 - 1) * C_KMS)
    np.testing.assert_array_equal(k2v["velocity_3"].data, maps["velocity_3"].data)


def test_maps_from_fit_needs_a_rest_wavelength_for_each_component():
    # Mg II h as a Gaussian named outside the line catalog and a broad one without a name.
    model = (
        line_model(["Mg II k"])
        + models.Gaussian1D(1, 2803.53 * u.AA, 0.1 * u.AA, name="h")
        + models.Gaussian1D(1, 2803.53 * u.AA, 0.3 * u.AA)
    )
    fitted, fitter, cube = _fit_mg_ii(model)
    # Against Mg II k, the line of the window, they would be 770 km/s off.
    for rest_wavelength, components in ((None, r"3 \('h'\), 4"), ({"h": 2803.53 * u.AA}, "4")):
        with pytest.raises(ValueError, match=rf"rest_wavelength: {components}\.$"):
            maps_from_fit(fitted, cube, fitter=fitter, rest_wavelength=rest_wavelength)
    for key in ("H", None):
        with pytest.raises(
            ValueError, match=rf"no Gaussian or Lorentzian component named {key!r}; those are 'k2v', 'k2r', 'h'\.$"
        ):
            maps_from_fit(fitted, cube, fitter=fitter, rest_wavelength={key: 2803.53 * u.AA, "k2v": 2796.352 * u.AA})
    with pytest.raises(TypeError, match=r"rest_wavelength\['h'\] must be an astropy Quantity"):
        maps_from_fit(fitted, cube, fitter=fitter, rest_wavelength={"h": 2803.53})


def test_maps_from_fit_defaults_to_the_line_of_the_window(raster_sg_file):
    # A broad Gaussian without a name at Mg II k, in a crop of the Mg II k window that covers five documented lines.
    fitted, fitter, cube = _fit_mg_ii(line_model(["Mg II k"]) + models.Gaussian1D(1, 2796.352 * u.AA, 0.3 * u.AA))
    maps = maps_from_fit(fitted, cube, fitter=fitter, residual=False)
    assert u.allclose(maps["velocity_3"].meta["fit_rest_wavelength"], 2796.352 * u.AA)
    raster = read_spectrograph_lvl2(raster_sg_file, uncertainty=True)

    def one_gaussian(window, center):
        """
        The maps of a Gaussian without a name at ``center`` fitted to every spectrum of
        ``window``.
        """
        cube = raster[window][0]
        wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
        good = np.isfinite(cube.data) & ~cube.mask
        model = models.Const1D(0 * cube.unit) + models.Gaussian1D(1 * cube.unit, center, 0.2 * u.AA)
        sigma = np.where(good, cube.uncertainty.array, np.inf)
        fitted, fitter = _fit(
            starts_from_model(model, cube), np.where(good, cube.data, 0), sigma, wavelength, cube.unit
        )
        return maps_from_fit(fitted, cube, fitter=fitter, residual=False)

    # A whole Si IV 1403 window, which covers five documented lines, and the 2832 continuum window, whose rest
    # wavelength is its center.
    for window, rest in (("Si IV 1403", 1402.77 * u.AA), ("2832", raster["2832"][0].meta.rest_wavelength.to(u.AA))):
        assert u.allclose(one_gaussian(window, rest)["velocity_1"].meta["fit_rest_wavelength"], rest)
    # C II 1334.53 in the C II window, whose line is C II 1335.71.
    with pytest.raises(ValueError, match=r"rest_wavelength: 1\.$"):
        one_gaussian("C II 1336", 1334.53 * u.AA)


def test_maps_from_fit_on_one_spectrum():
    raster = read_spectrograph_lvl2(get_test_filepath(MG_FEATURES_FILE), uncertainty=True)
    cube = raster["Mg II k 2796"][0][1:2, 301:302]
    wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)
    model = starts_from_model(line_model(["Mg II k"]), cube)
    fitted, fitter = _fit(model, cube.data, cube.uncertainty.array, wavelength, cube.unit)
    maps = maps_from_fit(fitted, cube, fitter=fitter)
    for name in fitted.param_names:
        np.testing.assert_array_equal(maps[name].data, [[getattr(fitted, name).value]], strict=True)
    assert maps["quality"].data.tolist() == [[FitQualityFlag.OK]]
    assert maps["residual"].data.shape == cube.data.shape
    np.testing.assert_allclose(maps["mean_1"].uncertainty.array, _errors(fitter)[..., 2], strict=True)


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
    velocity = non_thermal_velocity(
        _width_map(sigma, error=1e-3),
        1402.77 * u.AA,
        instrumental_fwhm=widths[0, 2] * u.AA,
        thermal_fwhm=0.02 * u.AA,
    )
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
        ).data
        for sign in (1, -1)
    ]
    # The step is in the standard deviation, so dv/dW is the difference over the step in FWHM.
    derivative = (shifted[0] - shifted[1]) / (2 * step * 2 * np.sqrt(2 * np.log(2)))
    np.testing.assert_allclose(velocity.uncertainty.array[0, 0], derivative[0, 0] * 1e-3, rtol=1e-5)
    np.testing.assert_array_equal(velocity.mask, [[False, True, True]])
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
    np.testing.assert_allclose(result.data, expected.data)
    assert result.mask[0, 1]
    # A hotter ion leaves less width for the non-thermal velocity.
    hotter = non_thermal_velocity(_width_map(sigma), 1402.77 * u.AA, ion="Si IV", temperature=2e5 * u.K)
    assert hotter.data[0, 0] < result.data[0, 0]


def test_non_thermal_velocity_non_positive_widths():
    widths = _width_map(
        [[0.05, -0.05, 0, np.nan, -0.05]], error=1e-3, mask=np.array([[False, False, False, False, True]])
    )
    velocity = non_thermal_velocity(widths, 1402.77 * u.AA, thermal_fwhm=0.03 * u.AA)
    assert np.isfinite(velocity.data[0, 0])
    assert np.isfinite(velocity.uncertainty.array[0, 0])
    assert np.all(np.isnan(velocity.data[0, 1:]))
    assert np.all(np.isnan(velocity.uncertainty.array[0, 1:]))
    np.testing.assert_array_equal(velocity.mask, [[False, True, True, True, True]])
    # The recipe in the docstring for the widths that are too narrow, rather than missing or masked.
    too_narrow = velocity.mask & ~widths.mask & np.isfinite(widths.data)
    np.testing.assert_array_equal(too_narrow, [[False, True, True, False, False]])


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({}, "Pass ion"),
        ({"ion": "O I"}, "no formation temperature"),
        ({"ion": "Xx IV", "temperature": 1e5 * u.K}, "No atomic mass"),
        ({"ion": "Si IV", "rest_wavelength": 1500 * u.AA}, "outside the IRIS passbands"),
    ],
)
def test_non_thermal_velocity_errors(keywords, match):
    with pytest.raises(ValueError, match=match):
        non_thermal_velocity(_width_map(np.array([[0.05, 0.06]])), **{"rest_wavelength": 1402.77 * u.AA, **keywords})


@pytest.mark.parametrize(
    ("keywords", "name"),
    [
        # A log T read as kelvin gave 14.7 km/s instead of 13.0 km/s
        ({"rest_wavelength": 1402.77 * u.AA, "temperature": 4.9}, "temperature"),
        # 140.277 for nm, read as Å, gave 147 km/s instead of 13.0 km/s
        ({"rest_wavelength": 140.277, "instrumental_fwhm": 0.026 * u.AA}, "rest_wavelength"),
    ],
)
def test_non_thermal_velocity_needs_quantities(keywords, name):
    with pytest.raises(TypeError, match=name):
        non_thermal_velocity(_width_map(np.array([[0.05]])), ion="Si IV", **keywords)


def test_non_thermal_velocity_temperature_from_the_catalog():
    widths = _width_map(np.array([[0.05, 0.06]]))
    from_catalog = non_thermal_velocity(widths, 1402.77 * u.AA, ion="Si IV")
    explicit = non_thermal_velocity(widths, 1402.77 * u.AA, ion="Si IV", temperature=10**4.9 * u.K)
    np.testing.assert_allclose(from_catalog.data, explicit.data)


def test_non_thermal_velocity_matches_iris_nonthermalwidth():
    idl = QTable.read(get_test_filepath("nonthermal_width/iris_nonthermalwidth_formula.ecsv"))
    for row in idl:
        result = non_thermal_velocity(
            _width_map([[row["fwhm"].to_value(u.AA) / (2 * np.sqrt(2 * np.log(2)))]]),
            row["wavelength"],
            instrumental_fwhm=row["instrumental_fwhm"],
            temperature=10 ** row["log_t"] * u.K,
            ion=row["ion"],
        )
        # IDL returns -1 where the observed width is too narrow.
        velocity = row["velocity"].to_value(u.km / u.s)
        # IDL's c of 2.9979e5 km/s and its older masses and Boltzmann constant change the velocities by about 1e-5.
        np.testing.assert_allclose(
            result.data[0, 0],
            np.nan if velocity == -1 else velocity,
            rtol=3e-5,
            err_msg=f"{row['ion']}, FWHM {row['fwhm']}",
        )
