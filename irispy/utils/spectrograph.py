"""
This module provides general utility functions called by code in spectrograph.
"""

import numpy as np

import astropy.units as u
from astropy import constants

from ndcube.wcs.tools import unwrap_wcs_to_fitswcs

from irispy.spectrograph import SpectrogramCube, SpectrogramCubeSequence
from irispy.utils.constants import RADIANCE_UNIT, SLIT_WIDTH
from irispy.utils.response import get_interpolated_effective_area, get_latest_response

__all__ = [
    "calculate_dn_to_radiance_factor",
    "convert_photons_per_sec_to_radiance",
    "radiometric_calibration",
    "reshape_1d_wavelength_dimensions_for_broadcast",
]


def _get_wcs_pixel_scales(cube):
    """
    Return ``(spectral_dispersion_per_pixel, slit_pixel_scale)`` from any SpectrogramCube WCS.

    Works with astropy FITS WCS, ndcube SlicedLowLevelWCS, and gWCS objects.

    Returns
    -------
    spectral_dispersion_per_pixel : `~astropy.units.Quantity`
        Wavelength covered per dispersion pixel (e.g. Angstrom/pix).
    slit_pixel_scale : `~astropy.units.Quantity`
        Angular size of one slit pixel (e.g. arcsec/pix).
    """
    if hasattr(cube.wcs, "wcs"):
        wcs = cube.wcs.wcs
        spectral_idx = np.where(np.array(wcs.ctype) == "WAVE")[0][0]
        lat_idx = np.arange(len(wcs.ctype))[["HPLT" in c for c in wcs.ctype]][0]
        return (
            wcs.cdelt[spectral_idx] * wcs.cunit[spectral_idx],
            wcs.cdelt[lat_idx] * wcs.cunit[lat_idx],
        )
    try:
        wcs = unwrap_wcs_to_fitswcs(cube.wcs)[0].wcs
        spectral_idx = np.where(np.array(wcs.ctype) == "WAVE")[0][0]
        lat_idx = np.arange(len(wcs.ctype))[["HPLT" in c for c in wcs.ctype]][0]
        return (
            wcs.cdelt[spectral_idx] * wcs.cunit[spectral_idx],
            wcs.cdelt[lat_idx] * wcs.cunit[lat_idx],
        )
    except TypeError:
        pass

    from dkist.wcs.models import VaryingCelestialTransform  # NOQA: PLC0415

    def _find_vct(model):
        if isinstance(model, VaryingCelestialTransform):
            return model
        for attr in ("left", "right"):
            sub = getattr(model, attr, None)
            if sub is not None:
                found = _find_vct(sub)
                if found is not None:
                    return found
        return None

    underlying = cube.wcs
    for _ in range(10):
        if hasattr(underlying, "forward_transform"):
            break
        if hasattr(underlying, "low_level_wcs"):
            underlying = underlying.low_level_wcs
        elif hasattr(underlying, "_wcs"):
            underlying = underlying._wcs
        else:
            break

    if hasattr(underlying, "forward_transform"):
        vct = _find_vct(underlying.forward_transform)
        if vct is not None:
            slit_scale = abs(vct.cdelt.quantity[0]).to(u.arcsec / u.pix).value * u.arcsec
            spectral = next(
                (
                    model
                    for model in underlying.forward_transform
                    if hasattr(model, "slope")
                    and getattr(model, "slope", None) is not None
                    and hasattr(model.slope, "unit")
                    and model.slope.unit.is_equivalent(u.AA / u.pix)
                ),
                None,
            )
            if spectral is not None:
                disp = abs(spectral.slope.quantity).to(u.AA / u.pix)
                return disp.value * u.AA, slit_scale.to(u.arcsec)

    n_pix = cube.wcs.pixel_n_dim
    p0 = [0.0] * n_pix
    p_d = list(p0)
    p_d[0] = 1.0
    p_s = list(p0)
    p_s[min(1, n_pix - 1)] = 1.0
    w0 = cube.wcs.pixel_to_world(*p0)
    w_d = cube.wcs.pixel_to_world(*p_d)
    w_s = cube.wcs.pixel_to_world(*p_s)

    def _to_list(world):
        return list(world) if isinstance(world, (list, tuple)) else [world]

    w0l, wdl, wsl = _to_list(w0), _to_list(w_d), _to_list(w_s)
    spectral_dispersion = next(
        abs((wc1 - wc0).to(u.AA))
        for wc0, wc1 in zip(w0l, wdl, strict=False)
        if hasattr(wc0, "unit") and wc0.unit.is_equivalent(u.AA)
    )
    sky0, sky1 = next((wc0, wc1) for wc0, wc1 in zip(w0l, wsl, strict=False) if hasattr(wc0, "separation"))
    return spectral_dispersion, sky0.separation(sky1).to(u.arcsec)


def radiometric_calibration(
    cube: SpectrogramCube | SpectrogramCubeSequence,
) -> SpectrogramCube | SpectrogramCubeSequence:
    """
    Performs radiometric calibration on the input cube or cube sequence.

    This takes into consideration also the observation time and uses the latest response.

    The data is also exposure time corrected during the conversion.

    This takes into account the spectral dispersion and solid angle of the pixels based on the WCS.
    Which is different from the IDL code as does not take spectral dispersion into account.
    If you want the same results as the IDL code, can multiply the output by the spectral dispersion.

    The spectral dispersion and solid angle are calculated using the WCS information.
    The wavelength axis and spatial axis should be determined dynamically from the WCS, rather than assuming fixed axis indices.
    For example, the spectral dispersion is calculated as ``cube.wcs.wcs.cdelt[wavelength_axis] * cube.wcs.wcs.cunit[wavelength_axis]``,
    and the solid angle as ``cube.wcs.wcs.cdelt[spatial_axis] * cube.wcs.wcs.cunit[spatial_axis] * SLIT_WIDTH``,
    where ``wavelength_axis`` and ``spatial_axis`` are determined from the WCS.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube` | `irispy.spectrograph.SpectrogramCubeSequence`
        The input cube to be calibrated.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube` or `irispy.spectrograph.SpectrogramCubeSequence`
        New cube in new units.

    Notes
    -----
    This is designed to do the same as `iris2/iris_calib_spectrum.pro <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/lmsal/iris2/iris_calib_spectrum.pro>`__ IDL code.

    The calibration output has been confirmed to provide the same results as those provided
    by the SolarSoft IDL routine `IRIS_CALIB <https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_calib.pro>`__.
    The major difference being that the output here is accounting for the wavelength, which is why the units
    here are :math:`erg s^{-1} sr^{-1} cm^{-2} Å^{-1}` and not :math:`erg s^{-1} sr^{-1} cm^{-2}`.
    Notice the extra :math:`Å^{-1}` in the units.
    """
    if isinstance(cube, SpectrogramCubeSequence):
        return SpectrogramCubeSequence([radiometric_calibration(c) for c in cube])
    detector_type = cube.meta.detector
    spectral_dispersion_per_pixel, slit_pixel_scale = _get_wcs_pixel_scales(cube)
    solid_angle = slit_pixel_scale * SLIT_WIDTH
    # Get wavelength for each pixel.
    wavelength_axis_index = next(
        axis
        for axis, physical_types in enumerate(cube.array_axis_physical_types)
        if physical_types and "em.wl" in physical_types
    )
    wavelength = cube.axis_world_coords(wavelength_axis_index)[0]
    time_obs = cube.meta.date_reference
    iris_response = get_latest_response(time_obs)
    corrected_cube = cube.apply_exposure_time_correction()
    unit_factor = corrected_cube.unit.to(u.photon / u.s)
    # Convert to radiance units.
    data_quantities = (corrected_cube.data * unit_factor * (u.photon / u.s),)
    if corrected_cube.uncertainty is not None:
        data_quantities += (corrected_cube.uncertainty.array * unit_factor * (u.photon / u.s),)
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
    new_cube_kwargs = {
        "data": new_data,
        "uncertainty": new_uncertainty,
        "unit": new_unit,
        "mask": "copy",
        "nddata_type": type(cube),
        "extra_coords": "copy",
        "global_coords": "copy",
        "_basic_wcs": "copy",
    }
    return cube.to_nddata(
        **new_cube_kwargs,
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
