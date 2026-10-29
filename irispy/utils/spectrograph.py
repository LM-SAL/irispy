"""
This module provides general utility functions called by code in spectrograph.
"""

import numpy as np

import astropy.units as u
from astropy import constants
from astropy.nddata import StdDevUncertainty

from irispy.sji import SJICube
from irispy.spectrograph import SpectrogramCube
from irispy.utils._spectral import _fit_background, check_scaled, make_map_cube, standard_deviation
from irispy.utils.constants import RADIANCE_UNIT, RADIANCE_UNIT_PER_HZ
from irispy.utils.response import get_interpolated_effective_area, get_latest_response
from irispy.utils.utils import _saturated_samples

__all__ = [
    "calculate_dn_to_radiance_factor",
    "radiation_temperature",
    "radiometric_calibration",
    "subtract_background",
]


@u.quantity_input
def radiometric_calibration(
    cube: SpectrogramCube | SJICube,
    *,
    line: u.AA = None,
) -> SpectrogramCube | SJICube:
    """
    Converts a cube from DN to radiance.

    The data are divided by their exposure times and converted with the effective areas at the
    observation time (`~irispy.utils.response.get_latest_response`). For raster files read
    together, that is the time of the first file.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube` | `irispy.sji.SJICube`
        The input cube to be calibrated, in DN.
    line : `astropy.units.Quantity`, optional
        Slit-jaw cubes only: the wavelength of the one line all the counts come from, which gives the
        line intensity in :math:`erg s^{-1} sr^{-1} cm^{-2}`. Defaults to `None`, a continuum flat across
        the passband, which gives :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}`.
        :ref:`irispy-tutorial-calibration` says when each holds.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube` or `irispy.sji.SJICube`
        New cube with the data type and mask of ``cube``, in :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}`,
        or :math:`erg s^{-1} sr^{-1} cm^{-2}` for a slit-jaw cube with ``line``.
        Spectra are NaN, but not masked, where the response has no effective area: outside
        1331-1359 Å and 1388-1408 Å in the FUV and 2782-2835 Å in the NUV, as at the blue end of a
        full-CCD Si IV 1394 window.

    Raises
    ------
    ValueError
        If ``line`` is given for a spectrograph cube, is not one wavelength or lies where the slit-jaw
        passband has no effective area, or if the cube is not from an IRIS slit-jaw passband, such as an
        AIA cube.

    Notes
    -----
    For spectra this does what SolarSoft's `iris_calib_spectrum.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/lmsal/iris2/iris_calib_spectrum.pro>`__
    does, in the same units, but takes the dispersion, plate scale and exposure times from the cube,
    where it takes fixed values and keywords, and gives NaN where it gives infinities.
    The NRL routine `iris_calib.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_calib.pro>`__
    gives the intensity of a whole line, in :math:`erg s^{-1} sr^{-1} cm^{-2}`: to compare with it, sum the
    calibrated spectra over the line and multiply by ``abs(cube.spectral_dispersion)``.
    """
    check_scaled(cube)
    if line is not None and not line.isscalar:
        msg = f"line must be one wavelength, not {line}"
        raise ValueError(msg)
    if isinstance(cube, SJICube):
        return _calibrate_sji(cube, line)
    if line is not None:
        msg = "line is for slit-jaw cubes; spectra are calibrated at each of their wavelengths"
        raise ValueError(msg)
    factor = calculate_dn_to_radiance_factor(
        iris_response=get_latest_response(cube.meta.date_reference),
        wavelength=cube.axis_world_coords(cube.wavelength_axis)[0],
        detector_type=cube.meta.detector_band,
        spectral_dispersion_per_pixel=cube.spectral_dispersion,
        solid_angle=cube.solid_angle,
    )
    return _calibrate(cube, factor, RADIANCE_UNIT)


def _calibrate(cube, factor, unit):
    # Multiplies cube by factor (photons per second to unit) over each exposure time, in the cube's
    # data type. The exposures lie along the first axes: the raster, for several files, then the
    # raster step or time.
    exposure_time = cube.exposure_time.to_value(u.s)
    exposure_time = exposure_time.reshape(exposure_time.shape + (1,) * (cube.data.ndim - exposure_time.ndim))
    factor = ((cube.unit / u.s * factor).to_value(unit) / exposure_time).astype(cube.data.dtype)
    uncertainty = None if cube.uncertainty is None else StdDevUncertainty(cube.uncertainty.array * factor)
    # NDCube arithmetic would deep-copy the meta and, through the coordinates, the whole cube
    return make_map_cube(cube, cube.data * factor, unit, uncertainty=uncertainty)


def _calibrate_sji(cube, line):
    response = get_latest_response(cube.meta.date_reference)
    # NAME_SJI holds the first three digits of each passband, "133" to "283"
    names = [name.decode() for name in response["NAME_SJI"]]
    if cube.meta.spectral_window[:3] not in names:
        msg = f"There is no IRIS slit-jaw response for {cube.meta.spectral_window}"
        raise ValueError(msg)
    band = names.index(cube.meta.spectral_window[:3])
    area, wavelength = response["AREA_SJI"][band], response["LAMBDA"]
    # The intensity times t Ω, per photon counted
    if line is None:
        # DN2PHOT DN = t Ω ∫ A(λ) I_λ λ / (hc) dλ, solved for a constant I_λ
        per_photon = constants.h * constants.c / np.trapezoid(area * wavelength, wavelength)
        unit = RADIANCE_UNIT
    else:
        # The response table is zero or absent where the passband does not reach
        line_area = np.interp(line, wavelength, area, left=0, right=0)
        if not line_area > 0:
            msg = f"The {cube.meta.spectral_window} slit-jaw passband has no effective area at line={line}"
            raise ValueError(msg)
        per_photon = constants.h * constants.c / line / line_area
        unit = RADIANCE_UNIT * u.AA
    solid_angle = cube.meta["CDELT1"] * cube.meta["CDELT2"] * u.arcsec**2
    # The cube's unit, DN_IRIS_SJI_<passband>, holds the passband's photons per DN, those of DN2PHOT_SJI
    return _calibrate(cube, per_photon / u.photon / solid_angle, unit)


def radiation_temperature(
    cube: SpectrogramCube,
) -> SpectrogramCube:
    r"""
    Converts a radiometrically calibrated cube to radiation temperature.

    The radiation (or brightness) temperature is the temperature of the blackbody whose
    Planck function equals the observed specific intensity, :math:`I_\nu = B_\nu(T_\mathrm{rad})`
    (:cite:t:`rybicki1985`, Section 1.5). Inverting the Planck function at each wavelength gives

    .. math::

       T_\mathrm{rad} = \frac{h \nu / k}{\ln \left( 1 + 2 h \nu^3 / (c^2 I_\nu) \right)}.

    :cite:t:`leenaarts2013` and :cite:t:`pereira2013` express Mg II h & k and NUV
    intensities this way.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        Cube in radiance per unit wavelength or frequency, e.g. the output of
        `~irispy.utils.spectrograph.radiometric_calibration`.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube`
        New cube in K, with a `~astropy.nddata.StdDevUncertainty` if ``cube`` has an uncertainty.

    Notes
    -----
    * Do not use the `astropy.units.brightness_temperature` equivalency for this: it is the
      Rayleigh-Jeans limit, which is wrong by orders of magnitude in the UV.
    * The radiation temperature is not the gas temperature. They are equal only for optically thick
      radiation whose source function is the Planck function (:cite:t:`rybicki1985`, Sections 1.4
      and 1.5), which holds only approximately for Mg II h & k :cite:p:`leenaarts2013` and not at
      all for optically thin lines.
    * Samples with zero, negative or NaN radiance are NaN and masked, and +Inf samples are +Inf and
      masked; the input mask is kept.
      Noise just above zero still gives several thousand kelvin.
    * The uncertainty is propagated to first order, so it is unreliable where it is comparable to the radiance.
    """
    check_scaled(cube)
    wavelength = cube.axis_world_coords(cube.wavelength_axis)[0]
    try:
        to_radiance = cube.unit.to(RADIANCE_UNIT_PER_HZ, equivalencies=u.spectral_density(wavelength))
    except u.UnitConversionError:
        msg = (
            f"The cube must be in radiance per unit wavelength or frequency, not {cube.unit}; "
            "convert it with radiometric_calibration first"
        )
        raise ValueError(msg) from None
    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())
    temperature_scale = (constants.h * frequency / constants.k_B).to_value(u.K)
    planck_scale = (2 * constants.h * frequency**3 / constants.c**2 / u.sr).to_value(RADIANCE_UNIT_PER_HZ)
    radiance = np.asarray(cube.data, dtype=float) * to_radiance
    sigma = standard_deviation(cube)
    uncertainty = None
    with np.errstate(divide="ignore", invalid="ignore"):
        temperature = np.where(radiance > 0, temperature_scale / np.log1p(planck_scale / radiance), np.nan)
        if sigma is not None:
            # First-order propagation with dT/dI from differentiating the expression above.
            derivative = temperature**2 * planck_scale / (temperature_scale * radiance * (radiance + planck_scale))
            uncertainty = StdDevUncertainty(derivative * sigma * to_radiance)
    return make_map_cube(cube, temperature, u.K, mask_invalid=True, uncertainty=uncertainty)


def calculate_dn_to_radiance_factor(
    *,
    iris_response,
    wavelength,
    detector_type,
    spectral_dispersion_per_pixel,
    solid_angle,
):
    """
    Calculates the factor that multiplies photons/s to give radiance at the given
    wavelengths.

    ``cube.unit.to(u.photon)`` gives the photons per DN of a Level 2 spectrograph cube.

    Parameters
    ----------
    iris_response: dict
        The IRIS response data loaded from `irispy.utils.response.get_latest_response`.
    wavelength: `astropy.units.Quantity`
        Wavelengths for which the factor is to be calculated.
    detector_type: `str`
        Detector type: 'FUV' or 'NUV'.
    spectral_dispersion_per_pixel: scalar `astropy.units.Quantity`
        Spectral dispersion (wavelength width) of a pixel.
    solid_angle: scalar `astropy.units.Quantity`
        Solid angle corresponding to a pixel.

    Returns
    -------
    `astropy.units.Quantity`
        Factor from photons/s to radiance at each wavelength, NaN outside the
        spectral ranges of the response.
    """
    # Get effective area and interpolate to observed wavelength grid.
    eff_area_interp = get_interpolated_effective_area(
        iris_response,
        detector_type,
        wavelength,
    )
    # Return radiometric converted data assuming input data is in units of photons/s.
    return (
        constants.h
        * constants.c
        / wavelength
        / u.photon
        / spectral_dispersion_per_pixel
        / eff_area_interp
        / solid_angle
    )


def subtract_background(cube, windows, *, degree=1):
    r"""
    Fit a polynomial to line-free wavelength windows of each spectrum and subtract it.

    Only unmasked, finite samples within ``windows`` determine the polynomial.
    The fitted background is then evaluated and subtracted at every wavelength in the spectrum.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        Input cube with a wavelength axis.
    windows : `astropy.units.Quantity`
        A ``(lower, upper)`` wavelength window, or an ``(n, 2)`` array of them, holding no line.
    degree : `int`, optional
        Degree of the polynomial in wavelength, fitted to the unmasked, finite samples in the
        windows by unweighted least squares. Defaults to 1, a straight line.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube`
        ``cube`` minus the fitted background, with the same mask, uncertainty and coordinates, in
        the floating-point type of the data: float32 for Level 2 data.
        Spectra with no more than ``degree`` samples to fit are NaN, except their saturated samples.

    Notes
    -----
    The error of the fitted background is shared by all the samples of a spectrum, so it adds
    up coherently in a sum over wavelength, which a
    `~astropy.nddata.StdDevUncertainty` of independent samples cannot hold. For a constant
    (``degree=0``) fitted to :math:`n` samples of error :math:`\sigma`, it is :math:`\sigma / \sqrt{n}`.

    Saturated samples (see :ref:`irispy-tutorial-saturated-data`) are left out of the fit and stay marked.
    """
    check_scaled(cube)
    background = _fit_background(cube, windows, degree)
    # Their value minus NaN would be NaN
    background[np.isnan(background) & _saturated_samples(cube)] = 0
    values = np.subtract(cube.data, background, out=background)
    return make_map_cube(cube, values, cube.unit, uncertainty=cube.uncertainty)
