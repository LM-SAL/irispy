"""
This module provides general utility functions called by code in spectrograph.
"""

import numpy as np

import astropy.units as u
from astropy import constants

from irispy.spectrograph import SpectrogramCube, SpectrogramCubeSequence
from irispy.utils.constants import RADIANCE_UNIT, SLIT_WIDTH
from irispy.utils.response import get_interpolated_effective_area, get_latest_response

__all__ = [
    "calculate_dn_to_radiance_factor",
    "convert_photons_per_sec_to_radiance",
    "radiometric_calibration",
    "reshape_1d_wavelength_dimensions_for_broadcast",
]


def radiometric_calibration(
    cube: SpectrogramCube | SpectrogramCubeSequence,
) -> SpectrogramCube | SpectrogramCubeSequence:
    """
    Performs radiometric calibration on the input cube or cube sequence.

    This takes into consideration also the observation time and uses the latest response.

    The data is also exposure time corrected during the conversion.

    Notes
    -----
    This is designed to do the same as `iris2/iris_calib_spectrum.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/lmsal/iris2/iris_calib_spectrum.pro>`__ IDL code.

    However, it does not output the same values.
    It is around 5% higher than the IDL code.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube` | `irispy.spectrograph.SpectrogramCubeSequence`
        The input cube to be calibrated.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube` or `irispy.spectrograph.SpectrogramCubeSequence`
        New cube in new units.
    """
    if isinstance(cube, SpectrogramCubeSequence):
        return SpectrogramCubeSequence([radiometric_calibration(c) for c in cube])
    detector_type = cube.meta.detector
    # Get spectral dispersion per pixel.
    spectral_wcs_index = np.where(np.array(cube.wcs.wcs.ctype) == "WAVE")[0][0]
    spectral_dispersion_per_pixel = cube.wcs.wcs.cdelt[spectral_wcs_index] * cube.wcs.wcs.cunit[spectral_wcs_index]
    # Get solid angle from slit width for a pixel.
    lat_wcs_index = ["HPLT" in c for c in cube.wcs.wcs.ctype]
    lat_wcs_index = np.arange(len(cube.wcs.wcs.ctype))[lat_wcs_index][0]
    # The slit width is divided by 2 in the IDL code, unsure why.
    solid_angle = cube.wcs.wcs.cdelt[lat_wcs_index] * cube.wcs.wcs.cunit[lat_wcs_index] * (SLIT_WIDTH / 2)
    # Get wavelength for each pixel.
    wavelength_axis_index = np.where(np.array(cube.wcs.wcs.ctype)[::-1] == "WAVE")[0][0]
    wavelength = cube.axis_world_coords(wavelength_axis_index)[0]
    time_obs = cube.meta.date_reference
    iris_response = get_latest_response(time_obs)
    exp_corrected_cube = cube.apply_exposure_time_correction()
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
    new_cube = SpectrogramCube(
        new_data,
        cube.wcs,
        new_uncertainty,
        new_unit,
        cube.meta,
        mask=cube.mask,
    )
    new_cube._extra_coords = cube.extra_coords
    return new_cube


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
        Quantities to be converted.  Must have units of counts/s or
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
    Calculates multiplicative factor that converts counts/s to radiance for
    given wavelengths.

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
    if n_data_dim == 1:
        pass
    elif n_data_dim == 2:
        wavelength = wavelength[np.newaxis, :]
    elif n_data_dim == 3:
        wavelength = wavelength[np.newaxis, np.newaxis, :]
    else:
        msg = "IRISSpectrogram dimensions must be 2 or 3."
        raise ValueError(msg)
    return wavelength
