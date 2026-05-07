"""
This module provides general utility functions called by code in spectrograph.
"""

import numpy as np

import astropy.units as u
from astropy import constants

from ndcube.wcs.tools import unwrap_wcs_to_fitswcs

from irispy._spectrograph_wcs import _spectrogram_cube_metadata_kwargs_for_copy
from irispy.spectrograph import SpectrogramCube
from irispy.utils.constants import RADIANCE_UNIT, SLIT_WIDTH
from irispy.utils.response import get_interpolated_effective_area, get_latest_response

__all__ = [
    "calculate_dn_to_radiance_factor",
    "convert_photons_per_sec_to_radiance",
    "radiometric_calibration",
    "reshape_1d_wavelength_dimensions_for_broadcast",
]


def _get_calibration_fits_wcs(cube):
    """
    Return the FITS WCS used to derive spectral dispersion and slit solid angle.

    Raster cubes may carry a gWCS on ``cube.wcs`` and a FITS WCS bridge on
    ``cube.basic_wcs``. This helper prefers a direct FITS WCS and falls back to
    unwrapping sliced FITS-WCS adapters when needed.
    """
    for wcs in (cube.wcs, getattr(cube, "basic_wcs", None)):
        if wcs is not None and hasattr(wcs, "wcs"):
            return wcs.wcs
    basic_wcs = getattr(cube, "basic_wcs", None)
    if basic_wcs is None:
        basic_wcs_segments = getattr(cube, "_basic_wcs_segments", None)
        if basic_wcs_segments:
            basic_wcs = basic_wcs_segments[0][2]
    if basic_wcs is not None:
        return unwrap_wcs_to_fitswcs(basic_wcs)[0].wcs
    return unwrap_wcs_to_fitswcs(cube.wcs)[0].wcs


def radiometric_calibration(cube: SpectrogramCube) -> SpectrogramCube:
    """
    Performs radiometric calibration on the input cube.

    This takes into consideration also the observation time and uses the latest response.

    The data is also exposure time corrected during the conversion.

    This takes into account the spectral dispersion and solid angle of the pixels based on the WCS.
    Which is different from the IDL code as does not take spectral dispersion into account.
    If you want the same results as the IDL code, can multiply the output by the spectral dispersion.

    The spectral dispersion and solid angle are calculated using an underlying FITS WCS.
    For FITS-WCS-backed cubes this comes directly from ``cube.wcs``; for raster gWCS-backed cubes
    it falls back to ``cube.basic_wcs``. The wavelength axis and spatial axis are determined
    dynamically from that WCS rather than assuming fixed axis indices.

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
    detector_type = cube.meta.detector
    underlying_wcs = _get_calibration_fits_wcs(cube)
    # Get spectral dispersion per pixel.
    spectral_wcs_index = np.where(np.array(underlying_wcs.ctype) == "WAVE")[0][0]
    spectral_dispersion_per_pixel = underlying_wcs.cdelt[spectral_wcs_index] * underlying_wcs.cunit[spectral_wcs_index]
    # Get solid angle from slit width for a pixel.
    lat_wcs_index = ["HPLT" in c for c in underlying_wcs.ctype]
    lat_wcs_index = np.arange(len(underlying_wcs.ctype))[lat_wcs_index][0]
    solid_angle = underlying_wcs.cdelt[lat_wcs_index] * underlying_wcs.cunit[lat_wcs_index] * SLIT_WIDTH
    # Get wavelength for each pixel.
    try:
        wavelength_axis_index = next(
            axis
            for axis, physical_types in enumerate(cube.array_axis_physical_types)
            if physical_types and "em.wl" in physical_types
        )
    except StopIteration as exc:
        msg = "radiometric_calibration requires a spectral axis. Fixed-wavelength raster images are not supported."
        raise ValueError(msg) from exc
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
    return cube.to_nddata(
        data=new_data,
        uncertainty=new_uncertainty,
        unit=new_unit,
        nddata_type=type(cube),
        extra_coords="copy",
        global_coords="copy",
        **_spectrogram_cube_metadata_kwargs_for_copy(cube),
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
    To get the same results as the IDL code, can multiply the output by the spectral dispersion
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
