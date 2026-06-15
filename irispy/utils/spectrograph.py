"""
This module provides general utility functions called by code in spectrograph.
"""

import numpy as np

import astropy.units as u
from astropy import constants

from irispy.spectrograph import SpectrogramCube
from irispy.utils.constants import RADIANCE_UNIT
from irispy.utils.response import get_interpolated_effective_area, get_latest_response

__all__ = [
    "calculate_dn_to_radiance_factor",
    "convert_photons_per_sec_to_radiance",
    "radiometric_calibration",
    "reshape_1d_wavelength_dimensions_for_broadcast",
]


def _reshape_exposure_time_for_broadcast(cube):
    exposure_time = cube.exposure_time.to(u.s)
    if np.isscalar(exposure_time.value):
        return exposure_time
    exposure_axes = cube._get_axis_coord_index(cube._exposure_time_name, cube._exposure_time_loc)
    item = [np.newaxis] * cube.data.ndim
    for axis in exposure_axes:
        item[axis] = slice(None)
    return exposure_time[tuple(item)]


def _unit_has_inverse_time(unit):
    decomposed = unit.decompose()
    return any(base == u.s and power < 0 for base, power in zip(decomposed.bases, decomposed.powers, strict=True))


def _exposure_time_corrected_cube_for_calibration(cube):
    if _unit_has_inverse_time(cube.unit):
        return cube
    return cube / _reshape_exposure_time_for_broadcast(cube)


def radiometric_calibration(cube: SpectrogramCube) -> SpectrogramCube:
    """
    Performs radiometric calibration on the input cube.

    This takes into consideration also the observation time and uses the latest response.

    The data is also exposure time corrected during the conversion.

    This takes into account the spectral dispersion and solid angle of the pixels
    based on the WCS, which is different from the IDL code that does not take spectral
    dispersion into account. If you want the same results as the IDL code, you can multiply
    the output by the spectral dispersion.

    The spectral dispersion, solid angle, and wavelength axis are delegated to
    `irispy.spectrograph.SpectrogramCube`.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        The input cube to be calibrated.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube`
        New cube in radiance units.

    Raises
    ------
    ValueError
        If the input does not expose a spectral axis, for example a
        fixed-wavelength raster image.

    Notes
    -----
    This is designed to do the same as `iris2/iris_calib_spectrum.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/lmsal/iris2/iris_calib_spectrum.pro>`__ IDL code.

    The calibration output has been confirmed to provide the same results as those provided
    by the SolarSoft IDL routine `IRIS_CALIB <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_calib.pro>`__.
    The major difference being that the output here is accounting for the wavelength, which is why the units
    here are :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}` and not :math:`erg s^{-1} sr^{-1} cm^{-2}`.
    Notice the extra :math:`Å^{-1}` in the units.
    """
    detector_type = cube.meta.detector_band
    spectral_dispersion_per_pixel = cube.spectral_dispersion
    solid_angle = cube.solid_angle
    # Get wavelength for each pixel.
    try:
        wavelength_axis_index = cube.wavelength_axis
    except ValueError as exc:
        msg = "radiometric_calibration requires a spectral axis. Fixed-wavelength raster images are not supported."
        raise ValueError(msg) from exc
    wavelength = cube.axis_world_coords(wavelength_axis_index)[0]
    time_obs = cube.meta.date_reference
    iris_response = get_latest_response(time_obs)
    exp_corrected_cube = _exposure_time_corrected_cube_for_calibration(cube)
    # Convert to radiance units.
    data_quantities = (exp_corrected_cube.data * exp_corrected_cube.unit.to(u.photon / u.s) * (u.photon / u.s),)
    if exp_corrected_cube.uncertainty is not None:
        uncertainty = (
            exp_corrected_cube.uncertainty.array * exp_corrected_cube.unit.to(u.photon / u.s) * (u.photon / u.s)
        )
        data_quantities += (uncertainty,)
    new_data_quantities = convert_photons_per_sec_to_radiance(
        data_quantities=data_quantities,
        iris_response=iris_response,
        wavelength=wavelength,
        detector_type=detector_type,
        spectral_dispersion_per_pixel=spectral_dispersion_per_pixel,
        solid_angle=solid_angle,
    )
    new_data = new_data_quantities[0].value
    new_uncertainty = new_data_quantities[1].value if len(new_data_quantities) > 1 else None
    new_unit = new_data_quantities[0].unit
    return cube.to_nddata(
        data=new_data,
        uncertainty=new_uncertainty,
        unit=new_unit,
        nddata_type=type(cube),
        extra_coords="copy",
        global_coords="copy",
    )


def convert_photons_per_sec_to_radiance(
    *,
    data_quantities,
    iris_response,
    wavelength,
    detector_type,
    spectral_dispersion_per_pixel,
    solid_angle,
):
    """
    Converts data quantities from counts/s to radiance.

    Parameters
    ----------
    data_quantities: iterable of `astropy.units.Quantity`
        Quantities to be converted. Must have units of counts/s or
        radiance equivalent counts, e.g. erg / cm**2 / s / sr / Angstrom.
    iris_response: dict
        The IRIS response data loaded from `irispy.utils.response.get_latest_response`.
    wavelength: `astropy.units.Quantity`
        Wavelength at each element along spectral axis of data quantities.
    detector_type: `str`
        Detector type: 'FUV', 'NUV', or 'SJI'.
    spectral_dispersion_per_pixel: scalar `astropy.units.Quantity`
        Spectral dispersion (wavelength width) of a pixel.
    solid_angle: scalar `astropy.units.Quantity`
        Solid angle corresponding to a pixel.

    Returns
    -------
    `list` of `astropy.units.Quantity`
        Data quantities converted to radiance.

    Notes
    -----
    This is designed to do the same as `nrl/iris_calib.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_calib.pro>`__ IDL code.
    The difference is that this function takes into account the spectral dispersion which the IDL code
    does not.
    To get the same results as the IDL code, you can multiply the output by the spectral dispersion
    or set the keyword to have the value of 1 Angstrom.
    """
    for i, data in enumerate(data_quantities):
        if data.unit != u.photon / u.s:
            msg = (
                f"Invalid unit provided. Unit must be equivalent to {u.photon / u.s}. "
                f"Error found for {i}th element of ``data_quantities`` with unit: {data.unit}"
            )
            raise ValueError(
                msg,
            )
    photons_per_sec_to_radiance_factor = calculate_dn_to_radiance_factor(
        iris_response=iris_response,
        wavelength=wavelength,
        detector_type=detector_type,
        spectral_dispersion_per_pixel=spectral_dispersion_per_pixel,
        solid_angle=solid_angle,
    )
    # Change shape of arrays so they are compatible for broadcasting
    # with data and uncertainty arrays.
    photons_per_sec_to_radiance_factor = reshape_1d_wavelength_dimensions_for_broadcast(
        photons_per_sec_to_radiance_factor,
        data_quantities[0].ndim,
    )
    return [(data * photons_per_sec_to_radiance_factor).to(RADIANCE_UNIT) for data in data_quantities]


def calculate_dn_to_radiance_factor(
    *,
    iris_response,
    wavelength,
    detector_type,
    spectral_dispersion_per_pixel,
    solid_angle,
):
    """
    Calculates multiplicative factor that converts counts/s to radiance for given
    wavelengths.

    Parameters
    ----------
    iris_response: dict
        The IRIS response data loaded from `irispy.utils.response.get_latest_response`.
    wavelength: `astropy.units.Quantity`
        Wavelengths for which counts/s-to-radiance factor is to be calculated
    detector_type: `str`
        Detector type: 'FUV' or 'NUV'.
    spectral_dispersion_per_pixel: scalar `astropy.units.Quantity`
        Spectral dispersion (wavelength width) of a pixel.
    solid_angle: scalar `astropy.units.Quantity`
        Solid angle corresponding to a pixel.

    Returns
    -------
    `astropy.units.Quantity`
        Multiplicative conversion factor from counts/s to radiance units
        for input wavelengths.

    Notes
    -----
    The term "multiplicative" refers to the fact that the conversion factor calculated by the
    `.calculate_dn_to_radiance_factor` function is used to multiply the counts per
    second (cps) data to obtain the radiance data. In other words, the conversion factor is a
    scaling factor that is applied to the cps data to convert it to radiance units.
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


def reshape_1d_wavelength_dimensions_for_broadcast(wavelength, n_data_dim):
    if n_data_dim < 1:
        msg = "IRISSpectrogram dimensions must be at least 1."
        raise ValueError(msg)
    for _ in range(n_data_dim - 1):
        wavelength = wavelength[np.newaxis, ...]
    return wavelength
