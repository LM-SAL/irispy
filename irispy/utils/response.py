"""
This module provides general utility functions for IRIS Responses.
"""

import numpy as np
import scipy
import scipy.io
from scipy.interpolate import make_interp_spline

import astropy.units as u
from astropy.time import Time
from astropy.units.quantity import Quantity

from sunpy.time import parse_time

from irispy.data import ROOTDIR

__all__ = ["_fit_xput_lite", "get_interpolated_effective_area", "get_latest_response"]


def get_latest_response(
    observation_time=None,
):
    """
    Returns the latest IRIS response structure.

    This is not downloading the latest response file from the IRIS website,'
    but rather reading the latest response file from the local data directory.

    Goal is to replicate the base functionality of the IDL routine
    ``iris_get_response.pro`` in the SSWIDL package.

    There are no plans to support anything but the latest response file.

    This routine does calculate time dependent effective areas using
    as is done in the SSWIDL version of this code.

    As a result of being translated from IDL, it has that IDL code smell and layout.
    The upside to this is that if upstream IDL changes occur, it is easy to adapt this
    in the future compared to if it was Python-ized.

    Parameters
    ----------
    observation_time: `astropy.time.Time`, optional
        Observation times of the datapoints.
        Must be in a parsable time format.
        If not provided, the current time is used.

    Returns
    -------
    `dict`
        Various parameters regarding IRIS response or effective area structure.

        Includes the following keys:

        date_obs : `astropy.time.Time`
        lambda : `astropy.units.Quantity`
        area_sg : `astropy.units.Quantity`
        name_sg : `str`
        dn2phot_sg : `tuple` of length 2
        area_sji : `astropy.units.Quantity`
        name_sji : `str`
        dn2phot_sji : `tuple` of length 4
        comment : `str`
        version : `int`
        version_date : `astropy.time.Time`
    """
    # Avoid circular imports
    from irispy.utils import record_to_dict  # NOQA: PLC0415

    if observation_time is None:
        observation_time = parse_time("now")

    number_of_obs_times = observation_time.size
    if observation_time.size == 1:
        observation_time = [observation_time]
    raw_response_data = scipy.io.readsav(ROOTDIR / "iris_sra_c_20231106.geny", python_dict=True)
    iris_response = record_to_dict(raw_response_data["p0"])
    iris_response = {
        k: val[0] if isinstance(val, np.ndarray) and val.ndim == 1 else val for k, val in iris_response.items()
    }
    # Convert to appropriate types
    iris_response["VERSION"] = int(iris_response["VERSION"])
    iris_response["LAMBDA"] = Quantity(iris_response["LAMBDA"], unit=u.nm)
    iris_response["AREA_SG"] = Quantity(iris_response["AREA_SG"], unit=u.cm**2)
    iris_response["AREA_SJI"] = Quantity(iris_response["AREA_SJI"], unit=u.cm**2)
    iris_response["GEOM_AREA"] = Quantity(iris_response["GEOM_AREA"], unit=u.cm**2)
    iris_response["VERSION"] = int(iris_response["VERSION"])
    iris_response["C_F_TIME"] = Time(iris_response["C_F_TIME"], format="utime")
    iris_response["C_F_LAMBDA"] = Quantity(iris_response["C_F_LAMBDA"], unit="nm")
    iris_response["C_N_TIME"] = Time(iris_response["C_N_TIME"], format="utime")
    iris_response["C_N_LAMBDA"] = Quantity(iris_response["C_N_LAMBDA"], unit="nm")
    iris_response["C_S_TIME"] = Time(iris_response["C_S_TIME"], format="utime")
    iris_response["ELEMENTS"]["DATE"] = parse_time(iris_response["ELEMENTS"]["DATE"].astype(str).tolist())
    iris_response["VERSION_DATE"] = parse_time(iris_response["VERSION_DATE"])
    iris_response["AREA_SG"] = np.zeros(iris_response["AREA_SG"].shape)
    iris_response["AREA_SJI"] = np.zeros(iris_response["AREA_SJI"].shape)
    # Handle multiple observation times by creating a list of responses
    iris_response = [iris_response.copy() for _ in range(number_of_obs_times)]
    # Set DATE_OBS to the observation time.
    for i in range(number_of_obs_times):
        iris_response[i]["DATE_OBS"] = observation_time[i].isot

    # FUV SG effective areas
    lambda_range_fuv = np.array([[133.1, 135.9], [138.8, 140.8]])
    # Rough SG spectral ranges.
    # Setting effective area to 0 outside of these.
    shape_fuv = iris_response[0]["COEFFS_FUV"].shape
    # Time-dependent response for shape_fuv[0] = 3 wavelengths
    iris_fit_fuv = np.zeros((number_of_obs_times, shape_fuv[0]))
    for j in range(shape_fuv[0]):
        iris_fit_fuv[:, j] = _fit_xput_lite(
            observation_time,
            iris_response[0]["C_F_TIME"],
            iris_response[0]["COEFFS_FUV"][j],
        )
    # Interpolate onto lambda grid, separately for each of the two FUV CCD's.
    for j in range(2):
        w_fuv = np.logical_and(
            iris_response[0]["LAMBDA"].value >= lambda_range_fuv[j, 0],
            iris_response[0]["LAMBDA"].value <= lambda_range_fuv[j, 1],
        )
        # For some reason in Python the above for Si IV misses the last point
        # so I add an extra True value in the mask
        if j == 1:
            idx = np.argwhere(w_fuv)[-1] + 1
            w_fuv[idx] = True
        for k in range(number_of_obs_times):
            iris_response[k]["AREA_SG"][0, w_fuv] = make_interp_spline(
                iris_response[k]["C_F_LAMBDA"][j : j + 2],
                iris_fit_fuv[k, j : j + 2],
                k=1,
            )(iris_response[k]["LAMBDA"][w_fuv])

    # The IDL code has a special case here for version 9 and later.
    # Remove wavelength dependence for the Si IV part of the FUV window.
    w_fuv_si_iv = np.logical_and(
        iris_response[0]["LAMBDA"].value >= lambda_range_fuv[1, 0],
        iris_response[0]["LAMBDA"].value <= lambda_range_fuv[1, 1],
    )
    # For some reason in Python the above for Si IV misses the last point
    # so I add an extra True value in the mask
    idx = np.argwhere(w_fuv_si_iv)[-1] + 1
    w_fuv_si_iv[idx] = True
    for k in range(number_of_obs_times):
        mean_area = np.mean(iris_response[k]["AREA_SG"][0, w_fuv_si_iv])
        iris_response[k]["AREA_SG"][0, w_fuv_si_iv] = np.repeat(mean_area, np.sum(w_fuv_si_iv))

    # NUV SG effective areas
    lambda_range_nuv = np.array([278.2, 283.5])
    # Rough SG spectral ranges.
    # Setting effective area to 0 outside of these.
    shape_nuv = iris_response[0]["COEFFS_NUV"].shape
    # Time-dependent response for shape_nuv[0] wavelengths
    iris_fit_nuv = np.zeros((number_of_obs_times, shape_nuv[0]))
    for j in range(shape_nuv[0]):
        iris_fit_nuv[:, j] = _fit_xput_lite(
            observation_time,
            iris_response[0]["C_N_TIME"],
            iris_response[0]["COEFFS_NUV"][j],
        )
    # Interpolate onto lambda grid
    w_nuv = np.where(
        np.bitwise_and(
            iris_response[0]["LAMBDA"].value >= lambda_range_nuv[0],
            iris_response[0]["LAMBDA"].value <= lambda_range_nuv[1],
        ),
    )
    for k in range(number_of_obs_times):
        iris_response[k]["AREA_SG"][1, w_nuv] = make_interp_spline(
            iris_response[k]["C_N_LAMBDA"], iris_fit_nuv[k], k=3, bc_type="natural"
        )(iris_response[k]["LAMBDA"][w_nuv])

    # SJI effective areas
    for nuv in range(2):
        # Calculate baseline SJI area curves
        area_sji = iris_response[0]["GEOM_AREA"]
        for m in range(len(iris_response[0]["INDEX_EL_SJI"][nuv * 2])):
            index_values1 = iris_response[0]["INDEX_EL_SJI"][nuv * 2 : nuv * 2 + 2, m]
            area_sji = area_sji * iris_response[0]["ELEMENTS"][index_values1].trans
        # Apply time dependent profile shape adjustment to FUV SJI
        if nuv == 0:
            # FUV: apply FUV SG "slant", then normalize so that a weighted (2.4:1)
            # sum at C II and Si IV gives constant response
            weight = np.array([2.4, 1])  # Typical solar ratio CII : SiIV
            wavelength = iris_response[0]["C_F_LAMBDA"]
            n_wavelength = len(wavelength)
            wavelength = np.array(
                [
                    wavelength[0].value,
                    ((wavelength[n_wavelength - 2] * 2 + wavelength[n_wavelength - 1]) / 3).value,
                ],
            )  # 2 wavelengths in nm
            # Calculate baseline SG area for scaling purposes
            area_sg = iris_response[0]["GEOM_AREA"]
            for m in range(len(iris_response[0]["INDEX_EL_SG"][nuv])):
                index_values2 = iris_response[0]["INDEX_EL_SG"][nuv, m]
                area_sg = area_sg * iris_response[0]["ELEMENTS"][index_values2].trans
            # SG and SJI areas at wavelength
            interpol_sg = make_interp_spline(
                iris_response[0]["LAMBDA"],
                area_sg,
                k=1,
            )
            area_sg2 = interpol_sg(wavelength)
            area_sj2 = np.zeros((2, 2))
            for m in range(2):
                interpol_sji = make_interp_spline(
                    iris_response[0]["LAMBDA"],
                    area_sji[m],
                    k=1,
                )
                area_sj2[:, m] = interpol_sji(wavelength)
            # Calculate the normalized slant function scal, apply to area_sji
            for k in range(number_of_obs_times):
                # Best-estimate slant, i.e., eff.area @ wavelength / baseline SG @ wavelength
                interpol_sg2 = make_interp_spline(
                    iris_response[k]["LAMBDA"],
                    iris_response[k]["AREA_SG"][0],
                    k=1,
                )
                sca2 = interpol_sg2(wavelength) / area_sg2
                # Normalize slant so that total(wei*asj2*sca2)/total(wei*asj2)=1
                for m in range(2):
                    sca2n = sca2 * np.sum(weight * area_sj2[:, m]) / np.sum(weight * area_sj2[:, m] * sca2)
                    interpol_sca = make_interp_spline(
                        wavelength,
                        sca2n,
                        k=1,
                    )
                    sca1n = interpol_sca(iris_response[k]["LAMBDA"])
                    sca1n = np.clip(sca1n, a_min=0, a_max=None)
                    iris_response[k]["AREA_SJI"][m] = area_sji[m] * sca1n
        else:
            # NUV is not processed
            for k in range(number_of_obs_times):
                iris_response[k]["AREA_SJI"] = [Quantity(x, unit=u.cm**2) for x in iris_response[k]["AREA_SJI"]]
                area_sji = list(area_sji)
                iris_response[k]["AREA_SJI"][2:4] = area_sji
    for j in range(4):
        for k in range(number_of_obs_times):
            # SJI specific time dependency
            iris_fit_sji = _fit_xput_lite(
                observation_time,
                iris_response[k]["C_S_TIME"][j, :],
                iris_response[k]["COEFFS_SJI"][j, :],
            )
            iris_response[k]["AREA_SJI"][j] = iris_response[k]["AREA_SJI"][j] * iris_fit_sji[k]
    # Ensure AREA_SG and AREA_SJI are Quantity arrays
    for response in iris_response:
        if not isinstance(response["AREA_SG"], Quantity):
            response["AREA_SG"] = Quantity(response["AREA_SG"], unit=u.cm**2)
        if not isinstance(response["AREA_SJI"], Quantity):
            response["AREA_SJI"] = Quantity(response["AREA_SJI"], unit=u.cm**2)
    if number_of_obs_times == 1:
        iris_response = iris_response[0]
    return iris_response


def _fit_xput_lite(observation_time, time_cal_coeffs, cal_coeffs):
    """
    To calculate the coefficients of best-fit time function for throughput, for
    which we apply a fit based on ``cal_coeffs``.

    The procedure involved in this function is as follows:

    1. The time difference (in years) is computed from the ``observation_time`` and ``time_cal_coeffs``.
    2. A least-squares fit is performed to determine the best fit for the time-dependent effective areas given the time difference.

    Goal is to replicate the base functionality of the IDL routine
    ``fit_iris_xput.pro`` in the SSWIDL package but without the optional keyword argument.

    Parameters
    ----------
    observation_time: `astropy.time.Time`
        Observation time.
    time_cal_coeffs: `astropy.time.Time`
        Start and end times of intervals of constant ``cal_coeffs[i]``.
        These should be in "utime" format.
    cal_coeffs: `numpy.ndarray`
        Coefficients of best-fit function, with at least two columns.

    Returns
    -------
    `numpy.array`
        Yields the fit used to compute the effective area using the input times ``observation_time``.
    """
    if time_cal_coeffs.shape[1] != 2 or cal_coeffs.shape[1] < 2:
        # Raise ValueError as time coefficient have the wrong format.
        msg = "Incorrect number of elements either in time_cal_coeffs or in cal_coeffs."
        raise ValueError(msg)
    # Some time transformations.
    # Exponent for transition between exp.decay intervals.
    transition_exp = 1.5
    # For loop for carrying out the least-squares fit and computation of fit output.
    fit_out = np.zeros(len(observation_time))
    for i, t in enumerate(observation_time):
        aux_cal_coeffs = np.zeros(2 * time_cal_coeffs.shape[0])
        # Looking for the closest time in the calibration time intervals.
        # Differences are given in years before passing to the next stage.
        t_diff = t - time_cal_coeffs
        t_diff = t_diff.flatten()
        # To convert to an array, quantities need to be dimensionless, hence dividing out the unit.
        t_diff = np.array([x.to(u.year).value for x in t_diff])
        idx = np.where(t_diff < 0)[0]
        if idx.size == 0:
            idx = 1
        else:
            idx = idx[0]
            if idx == 0:
                idx = 1
        # If the observation_time is between the calibration time intervals of a
        # calibration file (idx % !=0) then the aux_coeffs are given by an
        # exponential (coefficient and exponential value).
        # If the observation_time is between the end calibration time interval of
        # a calibration file (cal_file_t) and the beginning calibration
        # time interval of the next calibration file (cal_file_t+1)
        # (idx% 2 == 0) then, the aux_coeffs are given by 4 values
        # corresponding to a partial exponential obtained from
        # cal_file_t and a complementary exponential obtained from the
        # cal_file_t+1
        if idx % 2 != 0:  # I.e., if idx is not even...
            dtt_0 = 1
            exp_0 = np.exp(cal_coeffs[idx // 2, 2] * (t_diff[idx - 1]))
            aux_cal_coeffs[idx - 1 : idx + 1] = np.array([dtt_0, dtt_0 * exp_0])
        else:
            dtt_1 = (t_diff[idx - 1] / (t_diff[idx - 1] - t_diff[idx])) ** transition_exp
            dtt_0 = 1 - dtt_1
            exp_0 = np.exp(cal_coeffs[(idx // 2) - 1, 2] * (t_diff[idx - 2]))
            exp_1 = np.exp(cal_coeffs[idx // 2, 2] * (t_diff[idx]))
            aux_cal_coeffs[idx - 2 : idx + 2] = np.array([dtt_0, dtt_0 * exp_0, dtt_1, dtt_1 * exp_1])
        fit_out[i] = np.matmul(aux_cal_coeffs, cal_coeffs[:, :2].reshape(aux_cal_coeffs.shape[0]))
    return fit_out


def get_interpolated_effective_area(iris_response, detector_type, obs_wavelength):
    """
    To compute the interpolated time-dependent effective area.

    It will generalize to the time of the observation.

    Parameters
    ----------
    iris_response : dict
        The IRIS response data loaded from `irispy.utils.response.get_latest_response`.
    detector_type : `str`
        Detector type: 'FUV' or 'NUV'.
    obs_wavelength : `astropy.units.Quantity`
        The wavelength at which the observation has been taken in Angstroms.

    Returns
    -------
    `numpy.array`
        The effective area(s) determined by interpolation with a spline fit.
    """
    if detector_type.startswith("FUV"):
        detector_type_index = 0
    elif detector_type.startswith("NUV"):
        detector_type_index = 1
    else:
        msg = "Detector type not recognized."
        raise ValueError(msg)
    eff_area = iris_response["AREA_SG"][detector_type_index, :]
    response_wavelength = iris_response["LAMBDA"]
    obs_wavelength = obs_wavelength.to(u.nm)
    # Interpolate the effective areas to cover the wavelengths at which the data is recorded
    tck = make_interp_spline(
        response_wavelength,
        eff_area,
        k=1,
    )
    return tck(obs_wavelength) * u.cm**2
