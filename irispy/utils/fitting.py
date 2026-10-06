"""
Starting models for fitting IRIS spectral lines with astropy.
"""

import warnings

import numpy as np

import astropy.units as u
from astropy import constants
from astropy.modeling import models

from irispy.utils._spectral import check_scaled
from irispy.utils.mg_features import calculate_mg_features

__all__ = ["gaussians_on_background", "mg_ii_model", "si_iv_1403_model"]

_PROFILES = {"gaussian": (models.Gaussian1D, "mean", "stddev"), "lorentzian": (models.Lorentz1D, "x_0", "fwhm")}
_BACKGROUNDS = ("constant", "linear")
_SPEED_OF_LIGHT = constants.c.to(u.km / u.s)
# Vacuum rest wavelengths in Å of the IRIS lines documented in the FUV2 and NUV passbands, which the presets fit
# and must not see beside their line: De Pontieu et al. (2014) Table 4, with O IV 1404.806 and S IV from
# Polito et al. (2016), Mg II 2791.599 and the triplet from Pereira et al. (2015), Fe II and Ni I from
# Wülser et al. (2018), and Ni II from IRIS Technical Note 38.
_SI_IV_1403 = 1402.77
_MG_II_K = 2796.352
_MG_II_H = 2803.530
_DOCUMENTED_LINES = (
    ("Fe II", 1392.817),
    ("Ni II", 1393.330),
    ("Si IV", 1393.76),
    ("O IV", 1399.776),
    ("O IV", 1401.157),
    ("Si IV", _SI_IV_1403),
    ("O IV", 1404.806),
    ("S IV", 1404.808),
    ("S IV", 1406.009),
    ("Mg II", 2791.599),
    ("Mg II", _MG_II_K),
    ("Mg II", 2798.754),
    ("Mg II", 2798.823),
    ("Ni I", 2799.47),
    ("Mg II", _MG_II_H),
)


def _profile(profile):
    """
    The astropy model of ``profile`` and the names of its centre and width parameters.
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


def _check_window(wavelength, rest_wavelength):
    """
    Raise if ``wavelength`` covers a documented line other than ``rest_wavelength`` Å.
    """
    low, high = wavelength.min().to_value(u.AA), wavelength.max().to_value(u.AA)
    others = [f"{ion} {line:.3f}" for ion, line in _DOCUMENTED_LINES if low <= line <= high and line != rest_wavelength]
    if others:
        msg = f"The cube also covers {', '.join(others)} Å, which the model leaves out; crop it to the line to fit."
        raise ValueError(msg)


def _spectra(cube):
    """
    The data of ``cube`` with masked samples set to NaN, its spectra along the last
    axis.
    """
    data = np.moveaxis(np.asarray(cube.data, dtype=float), cube.wavelength_axis, -1)
    if cube.mask is not None:
        data = np.where(np.moveaxis(np.asarray(cube.mask, dtype=bool), cube.wavelength_axis, -1), np.nan, data)
    return data


def _background_level(spectra):
    """
    A start for the background of each spectrum: its 10th percentile, as in the gallery
    examples.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)  # Fully masked spectra.
        return np.nanpercentile(spectra, 10, axis=-1)


def gaussians_on_background(
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
        Initial line centres, one per component along the first axis. Further axes give a
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

    One Gaussian or Lorentzian on a constant background. The line is the run of samples above half
    maximum around the peak of the mean spectrum, over its 10th percentile. Each spectrum's
    background starts at its own 10th percentile, its centre at its brightest sample in that run
    and its amplitude at that sample's height above the background; the full width at half maximum
    starts at the run's width for every spectrum. The centre is bounded to the window, the width to
    the window's width and the amplitude to positive values.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        A Si IV 1403 cube cropped to the wavelengths to fit, read with ``memmap=False``. The window
        must leave out the neighbouring documented lines, such as O IV 140.116 and 140.481 nm.
    profile : `str`, optional
        The line profile, ``"gaussian"`` or ``"lorentzian"``.

    Returns
    -------
    `~astropy.modeling.CompoundModel`
        The model of `gaussians_on_background`, with wavelengths in Å and intensities in the unit of
        ``cube``, and parameters shaped like the spatial axes of ``cube``.

    Notes
    -----
    The rest wavelength is the vacuum value of :cite:t:`depontieu2014`. For several components,
    build the model with `gaussians_on_background`; see :ref:`irispy-tutorial-fitting`.
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
    return gaussians_on_background(
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
    Where `~irispy.utils.mg_features.calculate_mg_features` finds the peaks of a spectrum
    :cite:p:`pereira2013`, they set its Gaussians' centres and amplitudes; elsewhere these start at
    the medians over the cube. Each width starts at the median distance of its peak from the line
    centre (k3 or h3) over the cube, and the background at each spectrum's 10th percentile. The
    centres are kept within ``velocity_range`` of the rest wavelength and the amplitudes positive.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        One raster of a Mg II window, read with ``memmap=False`` and cropped to the k or the h line
        core: the model has no components for the line wings or for the other documented lines,
        such as the Mg II triplet at 279.88 nm.
    velocity_range : `~astropy.units.Quantity`, optional
        Doppler velocities from the rest wavelength within which to search for the peaks and to keep
        the Gaussians' centres. The default is the search range of :cite:t:`leenaarts2013`.

    Returns
    -------
    `~astropy.modeling.CompoundModel`
        The model of `gaussians_on_background`, with wavelengths in Å and intensities in the unit of
        ``cube``, and parameters shaped like the spatial axes of ``cube``. ``mean_1`` starts at the
        blue peak and ``mean_2`` at the red one, but a fit can swap them.

    Notes
    -----
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
            msg = f"No {name} found in the cube; build a model with gaussians_on_background instead."
            raise ValueError(msg)
        return np.where(np.isnan(values), np.nanmedian(values), values)

    def peak_wavelength(feature):
        return rest_wavelength * (1 + features[f"{feature}_velocity"].data * u.km / u.s / _SPEED_OF_LIGHT)

    core = start(peak_wavelength(f"{line}3"), f"{line}3 line centre")
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
    return gaussians_on_background(
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
