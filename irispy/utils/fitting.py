"""
Starting models for fitting IRIS spectral lines with astropy.
"""

import warnings

import numpy as np

import astropy.units as u
from astropy import constants
from astropy.modeling import models
from astropy.modeling.fitting import FitInfoArrayContainer
from astropy.modeling.utils import poly_map_domain
from astropy.nddata import StdDevUncertainty

from irispy.spectrograph import RasterCollection
from irispy.utils._spectral import (
    _QualityFlag,
    _to_unit,
    default_rest_wavelength,
    in_memory,
    make_map_cube,
    make_spatial_template,
    standard_deviation,
)
from irispy.utils.constants import ATOMIC_MASS, INSTRUMENTAL_FWHM, PASSBAND_LIMITS
from irispy.utils.lines import _BLEND_WIDTH, _load_lines, get_lines
from irispy.utils.utils import _boolean_mask

__all__ = [
    "FitQualityFlag",
    "line_model",
    "maps_from_fit",
    "non_thermal_velocity",
    "starts_from_model",
]

_SPEED_OF_LIGHT = constants.c.to(u.km / u.s)
# The lines fitted as their blue and red emission peaks, k2v and k2r or h2v and h2r.
_MG_II_PEAKS = {"Mg II k": "Mg II 2796", "Mg II h": "Mg II 2803"}


def _wavelength(cube):
    """
    The wavelengths of ``cube`` in Å.
    """
    return cube.axis_world_coords(cube.wavelength_axis)[0].to(u.AA)


def _rest_wavelength(line):
    """
    The catalog rest wavelength in Å of the documented line named by its ion and a
    wavelength in Å within 1 Å of it, such as ``"Si IV 1403"``.
    """
    ion, _, near = line.rpartition(" ")
    lines = get_lines(main_only=True)
    wavelength = lines["wavelength"][lines["ion"] == ion].to_value(u.AA)
    distance = np.abs(wavelength - float(near)) if near.replace(".", "", 1).isdigit() else np.inf
    close = wavelength[distance <= 1]
    if not close.size or np.ptp(close) > _BLEND_WIDTH.to_value(u.AA):
        msg = (
            f"No single documented line {line!r}; name one by its ion and wavelength in Å, "
            "such as 'Si IV 1403' or 'C II 1335.71'."
        )
        raise ValueError(msg)
    return wavelength[np.argmin(distance)]


def _named_rest_wavelength(name):
    """
    The rest wavelength of the documented line that a component is named after, as
    `line_model` names them, or `None`.
    """
    if name is None:
        return None
    # line_model names the Mg II peaks k2v and k2r, or h2v and h2r.
    line = next((line for line in _MG_II_PEAKS if name in {line[-1] + "2v", line[-1] + "2r"}), name)
    try:
        return _rest_wavelength(_MG_II_PEAKS.get(line, line)) * u.AA
    except ValueError:
        return None


def _spectra(cube):
    """
    The data of ``cube`` with masked and non-finite samples set to NaN, its spectra
    along the last axis.
    """
    data = np.asarray(cube.data, dtype=float)
    kept = np.isfinite(data) & ~_boolean_mask(cube)
    return np.moveaxis(np.where(kept, data, np.nan), cube.wavelength_axis, -1)


def _background_level(spectra):
    """
    A start for the background of each spectrum: its 10th percentile, as in the gallery
    examples.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)  # Fully masked spectra.
        return np.nanpercentile(spectra, 10, axis=-1)


def line_model(lines):
    """
    A starting model for documented IRIS lines: one Gaussian per line on a constant
    background.

    Give it a start in every spectrum of a cube with `starts_from_model`.

    Parameters
    ----------
    lines : `list` of `str`
        Documented lines of the line catalog (`~irispy.utils.lines.get_lines`), each named by its
        ion and its wavelength in Å to within 1 Å, such as ``"Si IV 1403"``, or ``"Mg II k"`` or
        ``"Mg II h"``. A name within 1 Å of separate lines of its ion, such as ``"C II 1335"``,
        needs more digits, such as ``"C II 1335.71"``.

    Returns
    -------
    `~astropy.modeling.CompoundModel`
        A `~astropy.modeling.functional_models.Const1D` background, ``amplitude_0``, plus the
        Gaussians in the order of ``lines``, numbered from 1, with means and standard deviations
        in Å. The components are named ``"background"``, after their lines, such as
        ``"Si IV 1403"``, and ``"k2v"``, ``"k2r"``, ``"h2v"`` and ``"h2r"`` for the Mg II peaks.

    Notes
    -----
    Each Gaussian starts at its line's rest wavelength in the catalog with a standard deviation
    of 0.05 Å. Mg II k and h get two Gaussians each, for the blue and red emission peaks (k2v
    and k2r, or h2v and h2r) as in the Gaussian decomposition of :cite:t:`itn39`. These start
    0.15 Å either side of the rest wavelength, and their means stay within 40 km/s of it, the
    search range of :cite:t:`leenaarts2013`, and their amplitudes stay positive. They are a proxy for the peaks, which
    :cite:t:`itn39` warns are hard to recover in complex profiles, such as those of flare ribbons.

    Pass the wavelengths to the fitter in Å; see :ref:`irispy-tutorial-fitting`.
    """
    model = models.Const1D(0, name="background")
    for line in lines:
        rest = _rest_wavelength(_MG_II_PEAKS.get(line, line)) * u.AA
        if line not in _MG_II_PEAKS:
            model += models.Gaussian1D(1, rest, 0.05 * u.AA, name=line)
            continue
        limits = rest * (1 + [-40, 40] * u.km / u.s / _SPEED_OF_LIGHT)
        for peak, offset in (("2v", -0.15), ("2r", 0.15)):
            model += models.Gaussian1D(
                1,
                rest + offset * u.AA,
                0.05 * u.AA,
                name=line[-1] + peak,
                bounds={"mean": limits, "amplitude": (0, None)},
            )
    return model


def starts_from_model(model, cube):
    """
    A model fitted to one spectrum, with a start for every spectrum of ``cube``.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        A model with one value per parameter, such as one from `line_model` or one fitted to a
        spectrum of ``cube``, and the sum of its components, such as lines on a background.
        Parameters without units are taken in Å and the unit of ``cube``.
    cube : `~irispy.spectrograph.SpectrogramCube`
        The cube to fit, cropped to the wavelengths to fit.

    Returns
    -------
    `~astropy.modeling.Model`
        A copy of ``model`` for `astropy.modeling.fitting.parallel_fit_dask`, with parameters
        shaped like the spatial axes of ``cube`` and its fixed parameters, ties and bounds.

    Notes
    -----
    Each spectrum is started from its own data, as ``scale_guess`` in EISPAC does for EIS fits.
    A `~astropy.modeling.functional_models.Const1D` background starts at the spectrum's 10th
    percentile. A `~astropy.modeling.functional_models.Linear1D` or
    `~astropy.modeling.polynomial.Polynomial1D` one keeps the slope or the higher terms of
    ``model``, and passes through the 10th percentile of the spectrum less those terms at the
    middle of the wavelengths, so that a steep slope does not lower it. A Gaussian or Lorentzian
    line starts at the spectrum's peak above that background, at each wavelength, within two
    widths of the center of ``model``, its center at the peak's wavelength and its amplitude at
    the peak's height, below the background for a line with a negative amplitude. Its width is
    that of ``model``. In a blend, such as the Mg II k2v and k2r peaks, each line searches only
    the samples nearer its own center in ``model`` than any other line's, so the lines start at
    their own peaks. Lines on the same side of the background whose centers in ``model``, a tied
    one where its tie puts it, lie within a sample of each other, such as a narrow core and a
    broad component with one center, search the samples within two widths of any of them and
    share the peak, less the amplitudes of fixed ones, in the ratio of their amplitudes in
    ``model``; where that leaves nothing, they keep the amplitudes of ``model``.

    Fixed and tied parameters and those of other components keep the values of ``model``, and the
    others are clipped to their bounds. Masked and non-finite samples are left out of the starts.
    A line with no samples left within two widths, or none above the background (below it for a
    negative amplitude), keeps the center and amplitude of ``model``, and a spectrum with no
    samples at all gets a NaN background. A line centered outside the wavelengths of ``cube``
    raises a `ValueError`.
    """
    cube = in_memory(cube)
    if any(np.ndim(getattr(model, name).value) for name in model.param_names):
        msg = "model has array parameters; pass one with a value per parameter, such as one fitted to one spectrum."
        raise ValueError(msg)
    compound = model.n_submodels > 1
    nodes = model.traverse_postorder(include_operator=True) if compound else []
    if operators := sorted({node for node in nodes if isinstance(node, str)} - {"+"}):
        msg = f"model joins its components with {', '.join(map(repr, operators))}; pass a sum of backgrounds and lines."
        raise ValueError(msg)
    wavelength = _wavelength(cube).value
    spectra = _spectra(cube)
    # The values of model in Å and the unit of cube, a tied one where its tie puts it: the fitter evaluates the ties
    # without units, in those of the wavelengths and data it is given. Like the fitter, convert only a model with
    # units, as other components, such as a custom model, need not support the conversion.
    if any(getattr(model, name).unit is not None for name in model.param_names):
        untied = model.without_units_for_data(x=1 * u.AA, y=1 * cube.unit)
    else:
        untied = model.copy()
    for name in model.param_names:
        if model.tied[name]:
            setattr(untied, name, model.tied[name](untied))
    fitted = dict(zip(model.param_names, untied.parameters, strict=True))

    starts = {}
    leaves = [model[index] for index in range(model.n_submodels)] if compound else [model]
    names = [
        [name + (f"_{index}" if compound else "") for name in leaf.param_names] for index, leaf in enumerate(leaves)
    ]
    lines = (models.Gaussian1D, models.Lorentz1D)
    centers = [fitted[leaf_names[1]] for leaf, leaf_names in zip(leaves, names, strict=True) if isinstance(leaf, lines)]
    if outside := [center for center in centers if not wavelength.min() <= center <= wavelength.max()]:
        msg = (
            f"model has lines at {', '.join(f'{center:.2f}' for center in outside)} Å, outside the cube's "
            f"{wavelength.min():.2f}-{wavelength.max():.2f} Å; leave them out of the model or crop the cube to them."
        )
        raise ValueError(msg)
    shape = np.zeros_like(wavelength)  # The started background less its value at the middle of the wavelengths.
    constants = {}  # The constant term of each background, and the rest of it at the middle of the wavelengths.
    for leaf, leaf_names in zip(leaves, names, strict=True):
        if isinstance(leaf, models.Const1D):
            constants[leaf_names[0]] = 0
        elif isinstance(leaf, (models.Linear1D, models.Polynomial1D)):
            # The background less its constant term, at each wavelength and, last, at their middle.
            x = np.append(wavelength, wavelength.mean())
            if isinstance(leaf, models.Linear1D):
                slope, constant = leaf_names
                terms = fitted[slope] * x
            else:
                constant, *powers = leaf_names
                # In the polynomial's own variable, for one with a domain.
                x = poly_map_domain(x, leaf.domain, leaf.window)
                terms = sum(
                    (fitted[term] * x**power for power, term in enumerate(powers, 1)),
                    np.zeros_like(x),
                )
            constants[constant] = terms[-1]
            shape += terms[:-1] - terms[-1]
    # The level of each spectrum with the shape of the background removed, which a steep slope would lower.
    background = _background_level(spectra - shape)
    for constant, rest in constants.items():
        starts[constant] = (background - rest) * cube.unit
    level = background[..., np.newaxis] + shape  # The started background at each wavelength.
    line_names = [leaf_names for leaf, leaf_names in zip(leaves, names, strict=True) if isinstance(leaf, lines)]
    amplitudes = [fitted[amplitude] for amplitude, _, _ in line_names]
    signs = [np.sign(amplitude) or 1 for amplitude in amplitudes]  # Negative for an absorption line.
    # Lines on the same side of the background with centers within a sample of each other, such as a narrow core
    # and a broad component of one line, share one peak rather than split the samples at the midpoint of their centers.
    spacing = np.min(np.abs(np.diff(wavelength)), initial=np.inf)
    groups = []
    for index in sorted(range(len(centers)), key=lambda index: (signs[index], centers[index])):
        # Against the group's lowest center, so that a group spans less than a sample.
        first = groups[-1][0] if groups else None
        if first is not None and signs[first] == signs[index] and centers[index] - centers[first] < spacing:
            groups[-1].append(index)
        else:
            groups.append([index])
    for group in groups:
        sign = signs[group[0]]
        distances = [np.abs(wavelength - centers[index]) for index in group]
        within = [d <= 2 * abs(fitted[line_names[index][2]]) for index, d in zip(group, distances, strict=True)]
        # In a blend, each line searches only the samples nearer its own center than any other line's.
        distance = np.min(distances, axis=0)
        near = np.any(within, axis=0) & np.all(
            [distance <= np.abs(wavelength - other) for index, other in enumerate(centers) if index not in group],
            axis=0,
        )
        excess = np.where(near & ~np.isnan(spectra), sign * (spectra - level), -np.inf)
        peak = np.argmax(excess, axis=-1)
        height = np.take_along_axis(excess, peak[..., np.newaxis], axis=-1)[..., 0]
        found = height > 0  # Otherwise no sample lies beyond the background on the side of the line.
        # Fixed amplitudes stay, and the others share what they leave of the peak, so that the starts add up to it.
        free = [index for index in group if not model.fixed[line_names[index][0]]]
        left = height - sum(abs(amplitudes[index]) for index in group if index not in free)
        total = sum(abs(amplitudes[index]) for index in free)
        for index in group:
            amplitude, center, _ = line_names[index]
            starts[center] = np.where(found, wavelength[peak], centers[index]) * u.AA
            if index in free:
                share = abs(amplitudes[index]) / total if total else 1 / len(free)
                starts[amplitude] = np.where(found & (left > 0), sign * left * share, amplitudes[index]) * cube.unit
    values = {}
    for name in model.param_names:
        parameter = getattr(model, name)
        value = parameter.value
        if name in starts and not (model.fixed[name] or model.tied[name]):
            value = starts[name].value if parameter.unit is None else starts[name].to_value(parameter.unit)
            lower, upper = model.bounds[name]
            value = np.clip(value, -np.inf if lower is None else lower, np.inf if upper is None else upper)
        value = np.broadcast_to(value, background.shape)
        values[name] = value if parameter.unit is None else value * parameter.unit
    seeded = model.copy()
    # As parallel_fit_dask itself gives a model one spectrum's parameters, keeping its constraints.
    seeded._reset_parameters(**values)
    return seeded


class FitQualityFlag(_QualityFlag):
    """
    Quality flags for the per-spectrum fits in `maps_from_fit`, most severe first.

    `OK` only means that the fitter converged within the bounds. A fit to noise gets it
    too, so check the signal-to-noise ratio and the reduced chi-square as in
    :ref:`irispy-tutorial-fitting`.
    """

    OK = (0, "ok")
    NO_FIT = (1, "no fit: the fit failed or the spectrum has no usable sample")
    NOT_CONVERGED = (2, "the fitter reports no success")
    AT_BOUND = (3, "a parameter stopped at one of its bounds")
    MASKED_INPUT = (4, "the spectrum has masked samples")


def _fit_property(fit_info, name, fill):
    """
    The fit property ``name`` as an array, ``fill`` where a fit has none, or `None` if
    no fit has it.

    `~astropy.modeling.fitting.FitInfoArrayContainer.get_property_as_array` fails on the 0
    that `~astropy.modeling.fitting.parallel_fit_dask` leaves for an all-NaN spectrum, and gives
    zeros for the `None` of a fit that raised.
    """
    # fit_info[index] raises a KeyError for the entries of a fit_info naming its keys.
    found = list(fit_info._fit_info_array.flat)
    found = [info.get(name) if isinstance(info, dict) else None for info in found]
    template = next((np.asarray(value) for value in found if value is not None), None)
    if template is None:
        return None
    values = [np.full(template.shape, fill) if value is None else value for value in found]
    return np.reshape(np.asarray(values, dtype=float), fit_info.shape + template.shape)


def maps_from_fit(fitted_model, cube, *, fitter, rest_wavelength=None, residual=True):
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
    fitter : `~astropy.modeling.fitting.Fitter`
        The fitter passed to `~astropy.modeling.fitting.parallel_fit_dask`, with ``fit_info=True``
        or ``fit_info`` naming the keys read here: ``"param_cov"`` for the uncertainties, and
        ``"success"`` and ``"active_mask"`` for `FitQualityFlag.NOT_CONVERGED` and
        `FitQualityFlag.AT_BOUND`, which `~astropy.modeling.fitting.TRFLSQFitter` and the other
        fitters built on `scipy.optimize.least_squares` report.
    rest_wavelength : `~astropy.units.Quantity` or `dict`, optional
        The rest wavelength for the Doppler velocities of every Gaussian and Lorentzian component,
        or a `dict` of rest wavelengths by component name, such as ``{"broad": 1402.77 * u.AA}``.
        A component without one gets the rest wavelength of the documented line it is named after,
        as `line_model` names its components, such as ``"Si IV 1403"`` or ``"k2v"``, or else
        ``cube.meta.rest_wavelength``, or the one documented line the cube covers if it does not
        cover ``cube.meta.rest_wavelength``; a blend of lines within 0.1 Å of each other counts as
        its strongest line.
    residual : `bool`, optional
        Whether to return the ``"residual"`` cube. `False` leaves it out, which saves memory and
        time when only the maps are needed. Defaults to `True`.

    Returns
    -------
    `~irispy.spectrograph.RasterCollection`
        Maps with the spatial WCS of ``cube``:

        * one per model parameter, keyed by its name, such as ``"mean_1"``;
        * for each Gaussian or Lorentzian component ``i``, ``"velocity_i"``, the Doppler velocity of
          its center in km/s, ``"fwhm_i"``, its full width at half maximum in Å, ``"fwhm_velocity_i"``,
          that width in km/s, and ``"integrated_intensity_i"``, its integral in the unit of ``cube``
          times Å; ``"velocity_i"`` and ``"fwhm_velocity_i"`` hold the rest wavelength they are
          against in ``meta["fit_rest_wavelength"]``, in nm;
        * ``"quality"``, a `FitQualityFlag` for each spectrum, the most severe that applies;
        * ``"residual"``, unless ``residual`` is `False`, ``cube`` minus the fitted model,
          retaining its WCS, uncertainty, mask and coordinates.

    Raises
    ------
    ValueError
        If every fit of a spectrum with a usable sample failed, as when a model with bounds is
        fitted against wavelengths that are not in Å; see :ref:`irispy-tutorial-fitting`. If a key
        of ``rest_wavelength`` is not the name of a Gaussian or Lorentzian component of the model, or
        if a component without a rest wavelength from ``rest_wavelength`` or its name has its median
        fitted center nearer another documented line than the default rest wavelength above.

    Notes
    -----
    When the fit provides covariance, parameter and derived maps include a
    `~astropy.nddata.StdDevUncertainty`. Parameter uncertainties are the square roots of the
    covariance diagonal, NaN where there is no fit and for fixed or tied parameters.

    The derived maps propagate the covariance to first order; their uncertainties are NaN for
    those that use a fixed or tied parameter: ``"velocity_i"`` uses the center, ``"fwhm_i"`` and
    ``"fwhm_velocity_i"`` the width, and ``"integrated_intensity_i"`` the amplitude and the width.
    The full width at half maximum is :math:`2\sqrt{2\ln 2}\,|\sigma|` for a Gaussian and the
    absolute value of the ``fwhm`` parameter of a Lorentzian, and the integral is
    :math:`\sqrt{2\pi}\,A|\sigma|` for a Gaussian and :math:`\pi A\,|\mathrm{FWHM}|/2` for a Lorentzian.

    A spectrum is `FitQualityFlag.NO_FIT`, and NaN in the parameter and derived maps, where its
    fit failed or where it has no usable sample, one that is finite and not masked.
    A spectrum is `FitQualityFlag.AT_BOUND` where a free parameter is at or beyond one of its
    bounds, to within 1e-8 times the larger of their magnitudes in that spectrum, or where the
    fitter reports the bound as active (the ``active_mask`` of `scipy.optimize.least_squares`).
    """
    names = fitted_model.param_names
    values = {name: getattr(fitted_model, name).quantity for name in names}
    if unitless := [name for name, value in values.items() if value is None]:
        msg = f"{', '.join(unitless)} have no unit; fit the model with data_unit or start it with units."
        raise ValueError(msg)
    fit_info = getattr(fitter, "fit_info", None)
    if not isinstance(fit_info, FitInfoArrayContainer):
        msg = "fitter has no per-spectrum fit information; pass fit_info=True to parallel_fit_dask."
        raise TypeError(msg)
    # Parameter.value is a scalar for one spectrum; Parameter.shape keeps the spatial axes.
    shape = np.broadcast_shapes(*(getattr(fitted_model, name).shape for name in names))
    values = {name: np.broadcast_to(value, shape, subok=True) for name, value in values.items()}
    usable = np.any(np.isfinite(cube.data) & ~_boolean_mask(cube), axis=cube.wavelength_axis)
    # A spectrum without usable samples, given zero weights, "converges" at its start with a zero covariance.
    failed = ~usable | np.any([np.isnan(value.value) for value in values.values()], axis=0)
    # A cube with no usable samples at all, such as a slab of missing steps, still gives its maps.
    if usable.any() and failed.all():
        msg = (
            "Every fit of a spectrum with data failed, giving NaN parameters; see Things to know in the fitting "
            "tutorial, and pass diagnostics='error' and a diagnostics_path to parallel_fit_dask to see the errors."
        )
        raise ValueError(msg)
    values = {name: np.where(failed, np.nan, value.value) * value.unit for name, value in values.items()}
    # The covariance covers the free parameters only, in the order of the model's.
    free = [name for name in names if not (fitted_model.fixed[name] or fitted_model.tied[name])]
    covariance = _fit_property(fit_info, "param_cov", np.nan)
    if covariance is not None:
        covariance[failed] = np.nan
    wavelength = _wavelength(cube)
    compound = fitted_model.n_submodels > 1
    leaves = [fitted_model[index] for index in range(fitted_model.n_submodels)] if compound else [fitted_model]
    # A tuple, as | composes astropy model classes into a compound model.
    lines = {
        index: leaf.name for index, leaf in enumerate(leaves) if isinstance(leaf, (models.Gaussian1D, models.Lorentz1D))
    }
    if isinstance(rest_wavelength, dict):
        named = dict.fromkeys(filter(None, lines.values()))
        if unknown := [name for name in rest_wavelength if name not in named]:
            msg = (
                f"The model has no Gaussian or Lorentzian component named {', '.join(map(repr, unknown))}; "
                f"those are {', '.join(map(repr, named))}."
            )
            raise ValueError(msg)
        given = {name: _to_unit(value, u.AA, f"rest_wavelength[{name!r}]") for name, value in rest_wavelength.items()}
        rests = {index: given.get(name, _named_rest_wavelength(name)) for index, name in lines.items()}
    elif rest_wavelength is None:
        rests = {index: _named_rest_wavelength(name) for index, name in lines.items()}
    else:
        rests = dict.fromkeys(lines, _to_unit(rest_wavelength, u.AA, "rest_wavelength"))
    if unknown := [index for index, rest in rests.items() if rest is None]:
        default = default_rest_wavelength(cube).to(u.AA)
        catalog = get_lines(main_only=True)["wavelength"].to(u.AA)
        # A component nearer another documented line, such as one at Mg II h in the Mg II k window, needs its own.
        elsewhere = []
        for index in unknown:
            center = values[
                ("mean" if isinstance(leaves[index], models.Gaussian1D) else "x_0") + (f"_{index}" if compound else "")
            ]
            center = center[np.isfinite(center)]  # Empty where no spectrum has a fit, as in a slab of missing steps.
            if not center.size:
                continue
            center = np.median(center)
            nearest = catalog[np.argmin(np.abs(catalog - center))]
            # The default of a continuum window, such as 2832, is its center, away from any documented line.
            if abs(nearest - center) < abs(default - center) and abs(nearest - default) > _BLEND_WIDTH:
                elsewhere.append(index)
        if elsewhere:
            labels = ", ".join(f"{index}" + (f" ({lines[index]!r})" if lines[index] else "") for index in elsewhere)
            msg = (
                f"These components are nearer another documented line than {default:.3f}, the rest wavelength of the "
                "cube; name them after their lines, such as 'Si IV 1403', or name them and pass their rest "
                f"wavelengths in rest_wavelength: {labels}."
            )
            raise ValueError(msg)
        rests.update(dict.fromkeys(unknown, default))

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
    velocity_rests = {}  # The rest wavelength of each velocity map.
    for index, rest in rests.items():
        gaussian = isinstance(leaves[index], models.Gaussian1D)
        suffix = f"_{index}" if compound else ""
        center = ("mean" if gaussian else "x_0") + suffix
        width = ("stddev" if gaussian else "fwhm") + suffix
        amplitude = "amplitude" + suffix
        to_fwhm = 2 * np.sqrt(2 * np.log(2)) if gaussian else 1
        to_area = np.sqrt(2 * np.pi) if gaussian else np.pi / 2
        # Doppler velocity per unit of the center and per Å, and Å per unit of width.
        speed = (_SPEED_OF_LIGHT / rest).to_value(u.km / u.s / values[center].unit)
        per_angstrom = (_SPEED_OF_LIGHT / rest).to_value(u.km / u.s / u.AA)
        width_scale = values[width].unit.to(u.AA)
        velocity = speed * (values[center].value - rest.to_value(values[center].unit))
        # Both profiles are even in their widths, which fits without a lower bound can leave negative.
        sign = np.sign(values[width].value)
        fwhm = to_fwhm * width_scale * np.abs(values[width].value)
        area = to_area * values[amplitude] * np.abs(values[width])
        area_error = scaled(
            variance(
                {amplitude: to_area * np.abs(values[width].value), width: to_area * sign * values[amplitude].value}
            ),
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
        velocity_rests[f"velocity_{index}"] = velocity_rests[f"fwhm_velocity_{index}"] = rest

    quality = np.full(shape, FitQualityFlag.OK, dtype=np.uint8)
    quality[np.any(_boolean_mask(cube), axis=cube.wavelength_axis)] = FitQualityFlag.MASKED_INPUT
    for name in free:
        value = values[name].value
        # +1 for the lower bound and -1 for the upper; beyond counts, as LMLSQFitter returns clipped parameters there.
        for bound, side in zip(fitted_model.bounds[name], (1, -1), strict=True):
            if bound is not None and np.isfinite(bound):
                # Per spectrum, so that a runaway fit cannot widen the tolerance for the others.
                tolerance = 1e-8 * np.maximum(np.abs(value), abs(bound))
                quality[side * (value - bound) <= tolerance] = FitQualityFlag.AT_BOUND
    # TRFLSQFitter stops strictly inside a bound, within its own tolerance of it.
    active = _fit_property(fit_info, "active_mask", 0)
    if active is not None:
        quality[np.any(active != 0, axis=-1)] = FitQualityFlag.AT_BOUND
    success = _fit_property(fit_info, "success", 0)
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
    for name, map_cube in cubes:
        if name in velocity_rests:
            map_cube.meta["fit_rest_wavelength"] = velocity_rests[name].to(u.nm)
    cubes.append(("quality", make_map_cube(template, quality, u.dimensionless_unscaled)))
    map_axes = tuple(range(len(shape)))
    if not residual:
        return RasterCollection(cubes, aligned_axes=(map_axes,) * len(cubes))
    model_values = np.moveaxis(fitted_model(wavelength.reshape((-1,) + (1,) * len(shape))), 0, cube.wavelength_axis)
    residual_values = np.asarray(cube.data) - model_values.to_value(cube.unit)
    cubes.append(("residual", make_map_cube(cube, residual_values, cube.unit, uncertainty=cube.uncertainty)))
    residual_axes = tuple(axis for axis in range(cube.data.ndim) if axis != cube.wavelength_axis)
    return RasterCollection(cubes, aligned_axes=(map_axes,) * (len(cubes) - 1) + (residual_axes,))


@u.quantity_input
def non_thermal_velocity(
    fwhm,
    rest_wavelength: u.AA,
    *,
    instrumental_fwhm: u.AA = None,
    thermal_fwhm: u.AA = None,
    temperature: u.K = None,
    ion=None,
):
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
        fit processed by `maps_from_fit`. The ``"width"`` map of
        `~irispy.utils.moments.calculate_moments` is a standard deviation, not a full width at half
        maximum: multiply it by :math:`\sqrt{8 \ln 2} \approx 2.355` first.
    rest_wavelength : `~astropy.units.Quantity`
        The rest wavelength :math:`\lambda` of the line.
    instrumental_fwhm : `~astropy.units.Quantity`, optional
        The instrumental width. Defaults to the spectral resolution of the passband of ``rest_wavelength``
        in `irispy.utils.constants.INSTRUMENTAL_FWHM`, 26 mÅ in the FUV and 53 mÅ in the NUV
        :cite:p:`depontieu2014`.
    thermal_fwhm : `~astropy.units.Quantity`, optional
        The thermal width. Defaults to the width above for the mass of ``ion`` and ``temperature``.
    temperature : `~astropy.units.Quantity`, optional
        The ion temperature :math:`T`. Defaults to the ion's temperature of maximum abundance in the
        line catalog (`~irispy.utils.lines.get_lines`), which neutral ions do not have.
    ion : `str`, optional
        The ion, such as ``"Si IV"``, for its mass :math:`m`, the standard atomic weight of its element
        :cite:p:`prohaska2022`, and its temperature. Needed unless ``thermal_fwhm`` is given.

    Returns
    -------
    `~irispy.spectrograph.SpectrogramCube`
        The non-thermal velocity in km/s, with the WCS of ``fwhm`` and its uncertainty propagated to
        first order if ``fwhm`` has one. It is masked where the width is masked or missing, or not
        above the instrumental and thermal widths. For a result ``v``,
        ``v.mask & ~fwhm.mask & np.isfinite(fwhm.data)`` picks out the pixels where the width is too
        narrow.

    Notes
    -----
    Quadrature subtraction assumes Gaussian observed, instrumental and thermal profiles.
    A Lorentzian full width at half maximum from `maps_from_fit` cannot be used here.

    A temperature of maximum abundance from an ionization equilibrium, such as CHIANTI's
    :cite:p:`dere2023`, assumes optically thin lines and does not suit chromospheric lines such as
    Mg II.
    """
    if instrumental_fwhm is None:
        band = next((band for band, (low, high) in PASSBAND_LIMITS.items() if low <= rest_wavelength <= high), None)
        if band is None:
            msg = f"{rest_wavelength} is outside the IRIS passbands; pass instrumental_fwhm."
            raise ValueError(msg)
        instrumental_fwhm = INSTRUMENTAL_FWHM[band]
    if thermal_fwhm is None:
        if ion is None:
            msg = "Pass ion, for its mass and temperature, or thermal_fwhm."
            raise ValueError(msg)
        element = ion.split()[0]
        if element not in ATOMIC_MASS:
            msg = f"No atomic mass for {ion!r}; pass thermal_fwhm."
            raise ValueError(msg)
        if temperature is None:
            lines = _load_lines()
            lines = lines[lines["ion"] == ion]
            # Neutral ions sit at the lower edge of the ionization equilibrium's temperature grid
            if not len(lines) or lines["ion_stage"][0] == 1 or not np.isfinite(lines["log_t_max"][0]):
                msg = f"The line catalog has no formation temperature for {ion!r}; pass temperature."
                raise ValueError(msg)
            temperature = 10 ** lines["log_t_max"][0] * u.K
        speed = np.sqrt(2 * constants.k_B * temperature / ATOMIC_MASS[element])
        thermal_fwhm = np.sqrt(4 * np.log(2)) * rest_wavelength * speed / constants.c
    width = u.Quantity(fwhm.data, fwhm.unit).to_value(u.AA)
    radicand = width**2 - instrumental_fwhm.to_value(u.AA) ** 2 - thermal_fwhm.to_value(u.AA) ** 2
    scale = (constants.c / (rest_wavelength * np.sqrt(4 * np.log(2)))).to_value(u.km / u.s / u.AA)
    with np.errstate(invalid="ignore"):
        velocity = np.where((width > 0) & (radicand > 0), scale * np.sqrt(radicand), np.nan)
    uncertainty = None
    if (sigma := standard_deviation(fwhm)) is not None:
        width_error = sigma * fwhm.unit.to(u.AA)
        # dv/dW = scale**2 W / v
        with np.errstate(invalid="ignore", divide="ignore"):
            uncertainty = StdDevUncertainty(scale**2 * width * width_error / velocity)
    return make_map_cube(fwhm, velocity, u.km / u.s, mask_invalid=True, uncertainty=uncertainty)
