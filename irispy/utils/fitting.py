"""
Starting models for fitting IRIS spectral lines with astropy.
"""

import warnings
from copy import deepcopy

import numpy as np

import astropy.units as u
from astropy import constants
from astropy.modeling import models
from astropy.modeling.fitting import FitInfoArrayContainer
from astropy.nddata import StdDevUncertainty

from ndcube import NDCollection

from irispy.spectrograph import SpectrogramCube
from irispy.utils._spectral import (
    _QualityFlag,
    check_scaled,
    make_map_cube,
    make_spatial_template,
    resolve_rest_wavelength,
    standard_deviation,
)
from irispy.utils.constants import ATOMIC_MASS, DOCUMENTED_LINES, INSTRUMENTAL_FWHM, PASSBAND_LIMITS
from irispy.utils.mg_features import calculate_mg_features

__all__ = [
    "FitQualityFlag",
    "NonThermalQualityFlag",
    "maps_from_fit",
    "mg_ii_model",
    "non_thermal_velocity",
    "profiles_on_background",
    "si_iv_1403_model",
]

_PROFILES = {"gaussian": (models.Gaussian1D, "mean", "stddev"), "lorentzian": (models.Lorentz1D, "x_0", "fwhm")}
_BACKGROUNDS = ("constant", "linear")
_SPEED_OF_LIGHT = constants.c.to(u.km / u.s)
# Vacuum rest wavelengths in Å of the lines the presets fit; the documented IRIS lines are in
# `irispy.utils.constants.DOCUMENTED_LINES`, with their sources.
_SI_IV_1403 = 1402.77
_MG_II_K = 2796.352
_MG_II_H = 2803.530


def _profile(profile):
    """
    The astropy model of ``profile`` and the names of its center and width parameters.
    """
    if profile not in _PROFILES:
        msg = f"profile must be one of {tuple(_PROFILES)}, not {profile!r}."
        raise ValueError(msg)
    return _PROFILES[profile]


def _wavelength(cube):
    """
    The wavelengths of ``cube`` in Å.
    """
    return cube.axis_world_coords(cube.wavelength_axis)[0].to(u.AA)


def _covered_lines(wavelength):
    """
    The ``(ion, wavelength in Å)`` pairs of the documented lines that ``wavelength``
    covers.
    """
    low, high = wavelength.min().to_value(u.AA), wavelength.max().to_value(u.AA)
    return [(ion, line) for ion, line in DOCUMENTED_LINES if low <= line <= high]


def _check_window(wavelength, rest_wavelength):
    """
    Raise if ``wavelength`` covers a documented line other than ``rest_wavelength`` Å.
    """
    others = [f"{ion} {line:.3f}" for ion, line in _covered_lines(wavelength) if line != rest_wavelength]
    if others:
        msg = f"The cube also covers {', '.join(others)} Å, which the model leaves out; crop it to the line to fit."
        raise ValueError(msg)


def _spectra(cube):
    """
    The data of ``cube`` with masked samples set to NaN, its spectra along the last
    axis.
    """
    data = np.asarray(cube.data, dtype=float)
    if cube.mask is not None:
        data = np.where(cube.mask, np.nan, data)
    return np.moveaxis(data, cube.wavelength_axis, -1)


def _background_level(spectra):
    """
    A start for the background of each spectrum: its 10th percentile, as in the gallery
    examples.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)  # Fully masked spectra.
        return np.nanpercentile(spectra, 10, axis=-1)


def profiles_on_background(
    centers,
    *,
    amplitudes=None,
    widths=None,
    background="constant",
    background_level=None,
    profile="gaussian",
    bounds=None,
):
    """
    A sum of Gaussian or Lorentzian line profiles on a constant or linear background.

    The model is ready for `astropy.modeling.fitting.parallel_fit_dask`, or for a fitter called on
    one spectrum. Its first component is the background, so the line parameters are numbered from
    1: ``amplitude_1``, ``mean_1`` and ``stddev_1`` for the first Gaussian, ``amplitude_1``,
    ``x_0_1`` and ``fwhm_1`` for the first Lorentzian.

    Parameters
    ----------
    centers : `~astropy.units.Quantity`
        Initial line centers, one per component along the first axis. Further axes give a
        different start in every spectrum of a cube, and must match its spatial axes.
    amplitudes : `~astropy.units.Quantity`, optional
        Initial peak values above the background, shaped like ``centers``. Defaults to 1.
    widths : `~astropy.units.Quantity`, optional
        Initial widths, shaped like ``centers``: the standard deviation of a Gaussian or the full
        width at half maximum of a Lorentzian. Defaults to 1.
    background : `str`, optional
        ``"constant"`` for a `~astropy.modeling.functional_models.Const1D` (``amplitude_0``) or
        ``"linear"`` for a `~astropy.modeling.functional_models.Linear1D` (``slope_0`` and
        ``intercept_0``, the value at zero wavelength) background.
    background_level : `~astropy.units.Quantity`, optional
        Initial background level, a scalar or an array over the spatial axes. Defaults to 0.
    profile : `str`, optional
        The line profile, ``"gaussian"`` or ``"lorentzian"``.
    bounds : `dict`, optional
        ``(lower, upper)`` limits keyed by parameter name, such as ``{"mean_1": (1402, 1403.5) * u.AA}``,
        with `None` for no limit. Widths are kept positive.

    Returns
    -------
    `~astropy.modeling.CompoundModel`

    Notes
    -----
    Pass the wavelengths to the fitter in the unit of ``centers``; see :ref:`irispy-tutorial-fitting`.
    """
    line, center_name, width_name = _profile(profile)
    if background not in _BACKGROUNDS:
        msg = f"background must be one of {_BACKGROUNDS}, not {background!r}."
        raise ValueError(msg)
    centers = u.Quantity(centers)
    if centers.ndim == 0:
        centers = centers[np.newaxis]
    # Unitless starts take the unit of the data when fitted.
    level = 0 if background_level is None else background_level
    slope = 0 * level.unit / centers.unit if isinstance(level, u.Quantity) else 0
    if background == "constant":
        model = models.Const1D(amplitude=level)
    else:
        model = models.Linear1D(slope=slope, intercept=level)
    for index, center in enumerate(centers):
        start = {center_name: center}
        if amplitudes is not None:
            start["amplitude"] = amplitudes[index]
        if widths is not None:
            start[width_name] = widths[index]
        model += line(**start)
        # astropy counts the left-hand parameters by their values, not their names, which breaks
        # array starts once parallel_fit_dask resets them to one spectrum's scalars:
        # https://github.com/astropy/astropy/issues/20554
        model.n_left_params = len(model.left.param_names)
    limits = dict(bounds or {})
    for name in (f"{width_name}_{index}" for index in range(1, len(centers) + 1)):
        lower, upper = limits.get(name, (None, None))
        # Gaussian1D's own lower bound, as zero widths divide by zero.
        limits[name] = (models.Gaussian1D.stddev.bounds[0] if lower is None else lower, upper)
    for name, (lower, upper) in limits.items():
        parameter = getattr(model, name)
        parameter.bounds = tuple(
            None if limit is None else u.Quantity(limit, parameter.unit).value for limit in (lower, upper)
        )
    return model


def si_iv_1403_model(cube, *, profile="gaussian"):
    """
    A starting model for Si IV 140.277 nm, with a start for every spectrum of ``cube``.

    One Gaussian or Lorentzian on a constant background.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        A Si IV 1403 cube cropped to the wavelengths to fit, read with ``memmap=False``. The window
        must leave out the neighboring documented lines, such as O IV 140.116 and 140.481 nm.
    profile : `str`, optional
        The line profile, ``"gaussian"`` or ``"lorentzian"``.

    Returns
    -------
    `~astropy.modeling.CompoundModel`
        The model of `profiles_on_background`, with wavelengths in Å and intensities in the unit of
        ``cube``, and parameters shaped like the spatial axes of ``cube``.

    Notes
    -----
    The line is the run of samples above half maximum around the peak of the mean spectrum,
    over its 10th percentile. Each spectrum's background starts at its own 10th percentile,
    its center at its brightest sample in that run, and its amplitude at that sample's height
    above the background. The starting full width at half maximum is the run's width for every spectrum.

    The center stays within the wavelength window, the width below the window's width, and
    the amplitude nonnegative.

    The rest wavelength is the vacuum value of :cite:t:`depontieu2014`. For several components,
    build the model with `profiles_on_background`; see :ref:`irispy-tutorial-fitting`.
    """
    _, center_name, width_name = _profile(profile)
    check_scaled(cube)
    wavelength = _wavelength(cube)
    _check_window(wavelength, _SI_IV_1403)
    spectra = _spectra(cube)
    background = _background_level(spectra)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.nanmean(spectra, axis=tuple(range(spectra.ndim - 1)))
    mean_peak = np.nanargmax(mean)
    # The first samples either side of the peak below half its height bound the line.
    mean_background = _background_level(mean)
    below = ~(mean - mean_background >= (mean[mean_peak] - mean_background) / 2)
    index = np.arange(mean.size)
    first = index[below & (index < mean_peak)].max(initial=-1)
    last = index[below & (index > mean_peak)].min(initial=mean.size)
    line = (index > first) & (index < last)
    # A fully masked spectrum gets index 0 and a NaN height, so parallel_fit_dask gives it NaN parameters.
    peak = np.argmax(np.where(line & ~np.isnan(spectra), spectra, -np.inf), axis=-1)
    height = np.take_along_axis(spectra, peak[..., np.newaxis], axis=-1)[..., 0] - background
    fwhm = np.full(height.shape, last - first - 1) * np.mean(np.abs(np.diff(wavelength)))
    width = fwhm if profile == "lorentzian" else fwhm / (2 * np.sqrt(2 * np.log(2)))
    return profiles_on_background(
        wavelength[peak][np.newaxis],
        amplitudes=height[np.newaxis] * cube.unit,
        widths=width[np.newaxis],
        background_level=background * cube.unit,
        profile=profile,
        bounds={
            f"{center_name}_1": (wavelength.min(), wavelength.max()),
            f"{width_name}_1": (None, wavelength.max() - wavelength.min()),
            "amplitude_1": (0, None),
        },
    )


def mg_ii_model(cube, *, velocity_range=(-40, 40) * u.km / u.s):
    """
    A starting model for the Mg II k or h line core, with a start for every spectrum of
    ``cube``.

    Two Gaussians, for the blue (k2v or h2v) and red (k2r or h2r) emission peaks, on a constant
    background, as in the k and h components of the Gaussian decomposition of :cite:t:`itn39`.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        One raster of a Mg II window, read with ``memmap=False`` and cropped to the k or the h line
        core: the model has no components for the line wings or for the other documented lines,
        such as the Mg II triplet at 279.88 nm.
    velocity_range : `~astropy.units.Quantity`, optional
        Doppler velocities from the rest wavelength within which to search for the peaks and to keep
        the Gaussians' centers. The default is the search range of :cite:t:`leenaarts2013`.

    Returns
    -------
    `~astropy.modeling.CompoundModel`
        The model of `profiles_on_background`, with wavelengths in Å and intensities in the unit of
        ``cube``, and parameters shaped like the spatial axes of ``cube``. ``mean_1`` starts at the
        blue peak and ``mean_2`` at the red one, but a fit can swap them.

    Notes
    -----
    Where `~irispy.utils.mg_features.calculate_mg_features` finds the peaks of a spectrum
    :cite:p:`pereira2013`, they set its Gaussians' centers and amplitudes; elsewhere these start at
    the medians over the cube. Each width starts at the median distance of its peak from the line
    center (k3 or h3) over the cube, and the background at each spectrum's 10th percentile.

    The centers stay within ``velocity_range`` of the rest wavelength and the amplitudes nonnegative.

    The rest wavelengths are the vacuum values of :cite:t:`depontieu2014`. The Gaussians are a
    proxy for the emission peaks, not a radiative-transfer inversion; :cite:t:`itn39` warns that
    the peaks are hard to recover from them in complex profiles, such as those of flare ribbons.
    """
    check_scaled(cube)
    wavelength = _wavelength(cube)
    covered = [
        (line, rest)
        for line, rest in (("k", _MG_II_K), ("h", _MG_II_H))
        if wavelength.min() <= rest * u.AA <= wavelength.max()
    ]
    if not covered:
        msg = "The cube covers neither the Mg II k nor the h line."
        raise ValueError(msg)
    line, rest = covered[0]
    _check_window(wavelength, rest)
    rest_wavelength = rest * u.AA
    velocity_range = u.Quantity(velocity_range, u.km / u.s)
    features = calculate_mg_features(cube, velocity_range=velocity_range, lines=(line,))
    background = _background_level(_spectra(cube))

    def start(values, name):
        if np.all(np.isnan(values)):
            msg = f"No {name} found in the cube; build a model with profiles_on_background instead."
            raise ValueError(msg)
        return np.where(np.isnan(values), np.nanmedian(values), values)

    def peak_wavelength(feature):
        return rest_wavelength * (1 + features[f"{feature}_velocity"].data * u.km / u.s / _SPEED_OF_LIGHT)

    core = start(peak_wavelength(f"{line}3"), f"{line}3 line center")
    peaks = [f"{line}2v", f"{line}2r"]
    centers = u.Quantity([start(peak_wavelength(peak), f"{peak} peak") for peak in peaks])
    amplitudes = np.clip(
        [start(features[f"{peak}_intensity"].data, f"{peak} peak") - background for peak in peaks], 0, None
    )
    limits = (rest_wavelength * (1 + velocity_range / _SPEED_OF_LIGHT)).to(u.AA)
    # Per-spectrum widths start the fits in more local minima, as a misplaced peak gives a tiny width.
    distance = np.abs(centers - core)
    widths = np.broadcast_to(
        np.median(distance, axis=tuple(range(1, distance.ndim)), keepdims=True), distance.shape, subok=True
    )
    return profiles_on_background(
        centers,
        amplitudes=amplitudes * cube.unit,
        widths=widths,
        background_level=background * cube.unit,
        bounds={
            "mean_1": limits,
            "mean_2": limits,
            "amplitude_1": (0, None),
            "amplitude_2": (0, None),
        },
    )


class FitQualityFlag(_QualityFlag):
    """
    Quality flags for the per-spectrum fits in `maps_from_fit`, most severe first.
    """

    OK = (0, "ok")
    NO_FIT = (1, "no fit: NaN parameters")
    NOT_CONVERGED = (2, "the fitter reports no success")
    AT_BOUND = (3, "a parameter stopped at one of its bounds")
    MASKED_INPUT = (4, "the spectrum has masked samples")


def _fit_property(fit_info, name, fill):
    """
    Return a fit property as an array, using ``fill`` for missing fits.

    `~astropy.modeling.fitting.FitInfoArrayContainer.get_property_as_array` fails where
    `~astropy.modeling.fitting.parallel_fit_dask` left no fit information: a 0 for an all-NaN
    spectrum and `None` for a fit that raised.
    """
    found = [fit_info[index] for index in np.ndindex(fit_info.shape)]
    found = [info.get(name) if isinstance(info, dict) else None for info in found]
    template = next((np.asarray(value) for value in found if value is not None), None)
    if template is None:
        return None
    values = [np.full(template.shape, fill) if value is None else value for value in found]
    return np.reshape(np.asarray(values, dtype=float), fit_info.shape + template.shape)


def maps_from_fit(fitted_model, cube, *, fitter=None, rest_wavelength=None):
    r"""
    Maps of the parameters of a model fitted to every spectrum of a cube.

    Parameters
    ----------
    fitted_model : `~astropy.modeling.Model`
        The model returned by `astropy.modeling.fitting.parallel_fit_dask`, with parameters shaped
        like the spatial axes of ``cube``. Its parameters must have units, as they do after a fit
        with ``data_unit``.
    cube : `~irispy.spectrograph.SpectrogramCube`
        The cube that was fitted.
    fitter : `~astropy.modeling.fitting.Fitter`, optional
        The fitter passed to `~astropy.modeling.fitting.parallel_fit_dask` with ``fit_info=True``.
        Without it the maps have no uncertainties and no fit counts as not converged. That flag
        relies on the ``success`` that `~astropy.modeling.fitting.TRFLSQFitter` and the other
        fitters built on `scipy.optimize.least_squares` report.
    rest_wavelength : `~astropy.units.Quantity`, optional
        The rest wavelength for the Doppler velocities. Defaults to
        `~irispy.utils._spectral.resolve_rest_wavelength`: the one documented line the cube covers
        and otherwise ``cube.meta.rest_wavelength`` (the ``TWAVE`` convention). The resolved value
        and its source are recorded in the metadata of the returned maps.

    Returns
    -------
    `ndcube.NDCollection`
        `~ndcube.NDCube` maps with the spatial WCS of ``cube`` and a spectral
        `~irispy.spectrograph.SpectrogramCube` residual:

        * one per model parameter, keyed by its name, such as ``"mean_1"``;
        * for each Gaussian or Lorentzian component ``i``, ``"velocity_i"``, the Doppler velocity of
          its center in km/s, ``"fwhm_i"``, its full width at half maximum in Å, ``"fwhm_velocity_i"``,
          that width in km/s, and ``"integrated_intensity_i"``, its integral in the unit of ``cube``
          times Å;
        * ``"quality"``, a `FitQualityFlag` for each spectrum, the most severe that applies;
        * ``"residual"``, ``cube`` minus the fitted model, retaining its WCS, uncertainty, mask
          and coordinates.

    Notes
    -----
    When the fit provides covariance, parameter and derived maps include a
    `~astropy.nddata.StdDevUncertainty`. Parameter uncertainties are the square roots of the
    covariance diagonal, NaN where a fit failed and for fixed or tied parameters.

    The derived maps propagate the covariance to first order. The full width at half
    maximum is :math:`2\sqrt{2\ln 2}\,\sigma` for a Gaussian and the ``fwhm`` parameter of a
    Lorentzian, and the integral is :math:`\sqrt{2\pi}\,A\sigma` for a Gaussian and
    :math:`\pi A\,\mathrm{FWHM}/2` for a Lorentzian.
    """
    check_scaled(cube)
    names = fitted_model.param_names
    values = {name: getattr(fitted_model, name).quantity for name in names}
    if unitless := [name for name, value in values.items() if value is None]:
        msg = f"{', '.join(unitless)} have no unit; fit the model with data_unit or start it with units."
        raise ValueError(msg)
    shape = np.broadcast_shapes(*(value.shape for value in values.values()))
    values = {name: np.broadcast_to(value, shape, subok=True) for name, value in values.items()}
    failed = np.any([np.isnan(value.value) for value in values.values()], axis=0)
    # The covariance covers the free parameters only, in the order of the model's.
    free = [name for name in names if not (fitted_model.fixed[name] or fitted_model.tied[name])]
    fit_info = getattr(fitter, "fit_info", None)
    if fitter is not None and not isinstance(fit_info, FitInfoArrayContainer):
        msg = "fitter has no per-spectrum fit information; pass fit_info=True to parallel_fit_dask."
        raise ValueError(msg)
    covariance = None if fit_info is None else _fit_property(fit_info, "param_cov", np.nan)
    wavelength = _wavelength(cube)
    rest_wavelength, rest_source = resolve_rest_wavelength(
        rest_wavelength, meta=cube.meta, wavelength_range=(wavelength.min(), wavelength.max())
    )
    if rest_wavelength is None:
        msg = "No rest wavelength resolves from the metadata or the documented lines; pass rest_wavelength."
        raise ValueError(msg)

    def variance(gradient):
        """
        First-order variance of a function of the free parameters, given its gradient by
        parameter name.
        """
        if covariance is None:
            return None
        if any(name not in free for name in gradient):
            return np.full(shape, np.nan)
        terms = np.zeros(shape)
        for first, first_gradient in gradient.items():
            for second, second_gradient in gradient.items():
                terms += first_gradient * second_gradient * covariance[..., free.index(first), free.index(second)]
        return terms

    def scaled(variance_value, scale):
        """
        The standard deviation of ``scale`` times a quantity of variance
        ``variance_value``.
        """
        return None if variance_value is None else np.abs(scale) * np.sqrt(variance_value)

    one = np.ones(shape)
    maps = [(name, values[name], scaled(variance({name: one}), 1)) for name in names]
    compound = fitted_model.n_submodels > 1
    leaves = [fitted_model[index] for index in range(fitted_model.n_submodels)] if compound else [fitted_model]
    for index, leaf in enumerate(leaves):
        # A tuple, as | composes astropy model classes into a compound model.
        if not isinstance(leaf, (models.Gaussian1D, models.Lorentz1D)):
            continue
        gaussian = isinstance(leaf, models.Gaussian1D)
        suffix = f"_{index}" if compound else ""
        center = ("mean" if gaussian else "x_0") + suffix
        width = ("stddev" if gaussian else "fwhm") + suffix
        amplitude = "amplitude" + suffix
        to_fwhm = 2 * np.sqrt(2 * np.log(2)) if gaussian else 1
        to_area = np.sqrt(2 * np.pi) if gaussian else np.pi / 2
        # Doppler velocity per unit of the center and per Å, and Å per unit of width.
        speed = (_SPEED_OF_LIGHT / rest_wavelength).to_value(u.km / u.s / values[center].unit)
        per_angstrom = (_SPEED_OF_LIGHT / rest_wavelength).to_value(u.km / u.s / u.AA)
        width_scale = values[width].unit.to(u.AA)
        velocity = speed * (values[center].value - u.Quantity(rest_wavelength, values[center].unit).value)
        fwhm = to_fwhm * width_scale * values[width].value
        area = to_area * values[amplitude] * values[width]
        area_error = scaled(
            variance({amplitude: to_area * values[width].value, width: to_area * values[amplitude].value}),
            (values[amplitude].unit * values[width].unit).to(cube.unit * u.AA),
        )
        maps += [
            (f"velocity_{index}", velocity * u.km / u.s, scaled(variance({center: one}), speed)),
            (f"fwhm_{index}", fwhm * u.AA, scaled(variance({width: one}), to_fwhm * width_scale)),
            (
                f"fwhm_velocity_{index}",
                fwhm * per_angstrom * u.km / u.s,
                scaled(variance({width: one}), to_fwhm * width_scale * per_angstrom),
            ),
            (f"integrated_intensity_{index}", area.to(cube.unit * u.AA), area_error),
        ]

    quality = np.full(shape, FitQualityFlag.OK, dtype=np.uint8)
    if cube.mask is not None:
        mask = np.broadcast_to(np.asarray(cube.mask, dtype=bool), cube.data.shape)
        masked = np.any(mask, axis=cube.wavelength_axis)
        quality[masked] = FitQualityFlag.MASKED_INPUT
    for name in free:
        bounds = [bound for bound in fitted_model.bounds[name] if bound is not None and np.isfinite(bound)]
        scale = max([np.nanmax(np.abs(values[name].value), initial=0), *(abs(bound) for bound in bounds)])
        for bound in bounds:
            pinned = np.abs(values[name].value - bound) <= 1e-8 * scale
            quality[pinned] = FitQualityFlag.AT_BOUND
    success = None if fit_info is None else _fit_property(fit_info, "success", 0)
    if success is not None:
        quality[success == 0] = FitQualityFlag.NOT_CONVERGED
    quality[failed] = FitQualityFlag.NO_FIT

    template = make_spatial_template(cube, cube.wavelength_axis)
    cubes = [
        (
            name,
            make_map_cube(
                template,
                value.value,
                value.unit,
                mask_invalid=True,
                uncertainty=None if error is None else StdDevUncertainty(error),
            ),
        )
        for name, value, error in maps
    ]
    cubes.append(("quality", make_map_cube(template, quality, u.dimensionless_unscaled)))
    for _, map_cube in cubes:
        map_cube.meta.update({"rest_wavelength": rest_wavelength, "rest_wavelength_source": rest_source})
    model_values = np.moveaxis(fitted_model(wavelength.reshape((-1,) + (1,) * len(shape))), 0, cube.wavelength_axis)
    residual = np.asarray(cube.data) - model_values.to_value(cube.unit)
    # Coordinates point back to their cube; copy them without copying its data.
    coordinate_memo = {id(cube): None}
    cubes.append(
        (
            "residual",
            SpectrogramCube(
                residual,
                cube.wcs,
                uncertainty=cube.uncertainty,
                unit=cube.unit,
                meta=cube.meta,
                mask=cube.mask,
                extra_coords=deepcopy(cube.extra_coords, coordinate_memo),
                global_coords=deepcopy(cube.global_coords, coordinate_memo),
            ),
        )
    )
    map_axes = tuple(range(len(shape)))
    residual_axes = tuple(axis for axis in range(cube.data.ndim) if axis != cube.wavelength_axis)
    return NDCollection(cubes, aligned_axes=(map_axes,) * (len(cubes) - 1) + (residual_axes,))


class NonThermalQualityFlag(_QualityFlag):
    """
    Quality flags for the non-thermal velocities of `non_thermal_velocity`.
    """

    OK = (0, "ok")
    NO_DATA = (1, "the width is NaN or masked")
    TOO_NARROW = (2, "the width is not above the instrumental and thermal widths")


def non_thermal_velocity(fwhm, wavelength, *, instrumental_fwhm=None, thermal_fwhm=None, temperature=None, ion=None):
    r"""
    The non-thermal velocity of a line from a map of its full width at half maximum.

    The observed width :math:`W` is taken as the instrumental and thermal widths and a
    non-thermal broadening added in quadrature, as in ``eis_width2velocity``
    :cite:p:`warren_eis_width2velocity`, with :math:`v_\mathrm{nt}` and the thermal speed
    :math:`\sqrt{2k_\mathrm{B}T/m}` the speeds at which the profile falls to :math:`1/e`:

    .. math::

        v_\mathrm{nt} = \frac{c}{\lambda\sqrt{4\ln 2}}\sqrt{W^2 - W_\mathrm{inst}^2 - W_\mathrm{th}^2},
        \qquad
        W_\mathrm{th} = \frac{\sqrt{4\ln 2}\,\lambda}{c}\sqrt{\frac{2k_\mathrm{B}T}{m}}.

    Parameters
    ----------
    fwhm : `~irispy.spectrograph.SpectrogramCube`
        A map of the Gaussian full width at half maximum, such as ``"fwhm_1"`` from a Gaussian
        fit processed by `maps_from_fit`.
    wavelength : `~astropy.units.Quantity`
        The rest wavelength :math:`\lambda` of the line.
    instrumental_fwhm : `~astropy.units.Quantity`, optional
        The instrumental width. Defaults to the spectral resolution of the passband of ``wavelength``
        in ``irispy.utils.constants.INSTRUMENTAL_FWHM``, 26 mÅ in the FUV and 53 mÅ in the NUV
        :cite:p:`depontieu2014`.
    thermal_fwhm : `~astropy.units.Quantity`, optional
        The thermal width. Defaults to the width above for the mass of ``ion`` and ``temperature``.
    temperature : `~astropy.units.Quantity`, optional
        The ion temperature :math:`T`. Needed with ``ion`` unless ``thermal_fwhm`` is given.
    ion : `str`, optional
        The ion, such as ``"Si IV"``, for its mass :math:`m`, the standard atomic weight of its element
        :cite:p:`prohaska2022`. Needed with ``temperature`` unless ``thermal_fwhm`` is given.

    Returns
    -------
    `ndcube.NDCollection`
        ``"non_thermal_velocity"``, in km/s with the WCS and mask of ``fwhm`` and its uncertainty
        propagated to first order if ``fwhm`` has one, NaN where the observed width is non-positive
        or not above the others, and ``"quality"``, a `NonThermalQualityFlag` for each pixel.

    Notes
    -----
    Quadrature subtraction assumes Gaussian observed, instrumental and thermal profiles.
    A Lorentzian full width at half maximum from `maps_from_fit` cannot be used here.

    A temperature of maximum abundance from an ionization equilibrium, such as CHIANTI's
    :cite:p:`dere2023`, assumes optically thin lines and does not suit chromospheric lines such as
    Mg II.
    """
    wavelength = u.Quantity(wavelength, u.AA)
    if instrumental_fwhm is None:
        band = next((band for band, (low, high) in PASSBAND_LIMITS.items() if low <= wavelength <= high), None)
        if band is None:
            msg = f"{wavelength} is outside the IRIS passbands; pass instrumental_fwhm."
            raise ValueError(msg)
        instrumental_fwhm = INSTRUMENTAL_FWHM[band]
    if thermal_fwhm is None:
        if ion is None or temperature is None:
            msg = "Pass ion and temperature, for the thermal width, or thermal_fwhm."
            raise ValueError(msg)
        element = ion.split()[0]
        if element not in ATOMIC_MASS:
            msg = f"No atomic mass for {ion!r}; pass thermal_fwhm."
            raise ValueError(msg)
        speed = np.sqrt(2 * constants.k_B * u.Quantity(temperature, u.K) / ATOMIC_MASS[element])
        thermal_fwhm = np.sqrt(4 * np.log(2)) * wavelength * speed / constants.c
    width = u.Quantity(fwhm.data, fwhm.unit).to_value(u.AA)
    radicand = (
        width**2 - u.Quantity(instrumental_fwhm).to_value(u.AA) ** 2 - u.Quantity(thermal_fwhm).to_value(u.AA) ** 2
    )
    scale = (constants.c / (wavelength * np.sqrt(4 * np.log(2)))).to_value(u.km / u.s / u.AA)
    broadened = (width > 0) & (radicand > 0)
    with np.errstate(invalid="ignore"):
        velocity = np.where(broadened, scale * np.sqrt(radicand), np.nan)
    missing = ~np.isfinite(width) | (False if fwhm.mask is None else np.asarray(fwhm.mask, dtype=bool))
    quality = np.where(missing, NonThermalQualityFlag.NO_DATA, NonThermalQualityFlag.OK).astype(np.uint8)
    quality[~missing & ~broadened] = NonThermalQualityFlag.TOO_NARROW
    uncertainty = None
    if (sigma := standard_deviation(fwhm)) is not None:
        width_error = sigma * fwhm.unit.to(u.AA)
        # dv/dW = scale**2 W / v
        with np.errstate(invalid="ignore", divide="ignore"):
            uncertainty = StdDevUncertainty(scale**2 * width * width_error / velocity)
    return NDCollection(
        [
            (
                "non_thermal_velocity",
                make_map_cube(fwhm, velocity, u.km / u.s, mask_invalid=True, uncertainty=uncertainty),
            ),
            ("quality", make_map_cube(fwhm, quality, u.dimensionless_unscaled)),
        ],
        aligned_axes=tuple(range(fwhm.data.ndim)),
    )
