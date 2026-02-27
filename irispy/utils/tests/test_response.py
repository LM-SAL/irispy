from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as np_test
import pytest
import scipy.io

import astropy.units as u
from astropy.time import Time

from sunpy.time import parse_time

from irispy.data.test import ROOTDIR, get_test_filepath
from irispy.tests.helpers import figure_test
from irispy.utils.response import _fit_xput_lite, get_interpolated_effective_area, get_latest_response


def test_fit_xput_lite_idl():
    fit_xput_idl_inputs = scipy.io.readsav(get_test_filepath("fit_iris_xput_input.sav"), python_dict=True)
    fit_xput_idl_outputs = scipy.io.readsav(get_test_filepath("fit_iris_xput_output.sav"), python_dict=True)

    time_cal_coeffs = fit_xput_idl_inputs["cftime"]
    cal_coeffs = fit_xput_idl_inputs["coeffs_fuv"]
    iris_fit_expected = fit_xput_idl_outputs["rr"]

    test_time = [parse_time("2025-08-05T22:25:04.723", format="utime")]
    time_cal_coeffs_astropy = Time(time_cal_coeffs, format="utime")

    # Test each coefficient set
    for i in range(3):
        iris_fit = _fit_xput_lite(test_time, time_cal_coeffs_astropy, cal_coeffs[i, :, :])
        np_test.assert_almost_equal(
            iris_fit,
            iris_fit_expected[i],
        )


@pytest.fixture
def old_iris_response_data():
    raw_response_data = scipy.io.readsav(Path(ROOTDIR) / "iris_sra_c_20161022.geny", python_dict=True)
    return {name: raw_response_data["p0"][name][0] for name in raw_response_data["p0"].dtype.names}


@pytest.mark.parametrize(
    ("coeff_index", "expected_value"),
    [
        (0, 0.79865301),
        (1, 2.2495413),
        (2, 2.2495413),
    ],
)
def test_fit_iris_xput_lite_old_fuv(old_iris_response_data, coeff_index, expected_value):
    test_time_obs = [parse_time("2013-09-03")]
    time_cal_coeffs = parse_time(old_iris_response_data.get("C_F_TIME"), format="utime")
    cal_coeffs = old_iris_response_data.get("COEFFS_FUV")[coeff_index, :, :]
    result = _fit_xput_lite(test_time_obs, time_cal_coeffs, cal_coeffs)
    expected_array = np.array([expected_value])
    np_test.assert_almost_equal(
        result,
        expected_array,
        decimal=6,
    )


@pytest.mark.parametrize(
    ("coeff_index", "expected_value"),
    [
        (0, 0.23529011),
        (1, 0.25203046),
        (2, 0.25265095),
        (3, 0.253833),
    ],
)
def test_fit_iris_xput_lite_old_nuv(old_iris_response_data, coeff_index, expected_value):
    test_time_obs = [parse_time("2013-09-03")]
    time_cal_coeffs = parse_time(old_iris_response_data.get("C_N_TIME"), format="utime")
    cal_coeffs = old_iris_response_data.get("COEFFS_NUV")[coeff_index, :, :]
    result = _fit_xput_lite(test_time_obs, time_cal_coeffs, cal_coeffs)
    expected_array = np.array([expected_value])
    np_test.assert_almost_equal(
        result,
        expected_array,
        decimal=6,
    )


def test_get_latest_response_to_idl(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    np_test.assert_equal(iris_response["VERSION"], int(idl_response["VERSION"]))
    np_test.assert_equal(iris_response["DATE_OBS"], idl_response["DATE_OBS"].decode())
    np_test.assert_equal(iris_response["LAMBDA"].to_value(), idl_response["LAMBDA"])

    # FUV SG
    np_test.assert_almost_equal(iris_response["AREA_SG"][0].to_value(), idl_response["AREA_SG"][0], decimal=4)
    # NUV SG
    np_test.assert_almost_equal(iris_response["AREA_SG"][1].to_value(), idl_response["AREA_SG"][1], decimal=4)
    # All SJI
    np_test.assert_almost_equal(iris_response["AREA_SJI"][0].to_value(), idl_response["AREA_SJI"][0], decimal=4)
    np_test.assert_almost_equal(iris_response["AREA_SJI"][1].to_value(), idl_response["AREA_SJI"][1], decimal=4)
    np_test.assert_almost_equal(iris_response["AREA_SJI"][2].to_value(), idl_response["AREA_SJI"][2], decimal=4)
    np_test.assert_almost_equal(iris_response["AREA_SJI"][3].to_value(), idl_response["AREA_SJI"][3], decimal=4)


def test_get_latest_response_multiple_inputs():
    times = parse_time(["2025-08-05T22:25:04.723", "2025-08-06T22:25:04.723", "2025-08-07T22:25:04.723"])
    iris_response = get_latest_response(times)
    assert len(iris_response) == 3


def test_get_latest_response_no_observation_time():
    iris_response = get_latest_response()
    assert iris_response is not None


@figure_test
def test_plot_idl_vs_python_fuv_sg(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    fig, ax = plt.subplots()
    aslice = slice(235, 425)
    ax.plot(iris_response["LAMBDA"][aslice], iris_response["AREA_SG"][0, aslice], ".", label="Python", alpha=0.5)
    ax.plot(idl_response["LAMBDA"][aslice], idl_response["AREA_SG"][0, aslice], ".", label="IDL", alpha=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("IRIS Effective Area Comparison - FUV")
    ax.legend(loc="upper left")
    ax.grid()
    return fig


@figure_test
def test_plot_idl_vs_python_nuv_sg(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    fig, ax = plt.subplots()
    aslice = slice(3125, 3275)
    ax.plot(iris_response["LAMBDA"][aslice], iris_response["AREA_SG"][1, aslice], ".", label="Python", alpha=0.5)
    ax.plot(idl_response["LAMBDA"][aslice], idl_response["AREA_SG"][1, aslice], ".", label="IDL", alpha=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("IRIS Effective Area Comparison - NUV")
    ax.legend()
    ax.grid()
    return fig


@figure_test
def test_plot_idl_vs_python_sji_1(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    aslice = slice(175, 525)
    fig, ax = plt.subplots()
    ax.plot(
        iris_response["LAMBDA"][aslice],
        iris_response["AREA_SJI"][0, aslice],
        ".",
        label="Python",
        alpha=0.5,
    )
    ax.plot(
        idl_response["LAMBDA"][aslice],
        idl_response["AREA_SJI"][0, aslice],
        ".",
        label="IDL",
        alpha=0.5,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("IRIS Effective Area Comparison - SJI 1")
    ax.legend()
    ax.grid()
    return fig


@figure_test
def test_plot_idl_vs_python_sji_2(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    aslice = slice(200, 550)
    fig, ax = plt.subplots()
    ax.plot(
        iris_response["LAMBDA"][aslice],
        iris_response["AREA_SJI"][1, aslice],
        ".",
        label="Python",
        alpha=0.5,
    )
    ax.plot(
        idl_response["LAMBDA"][aslice],
        idl_response["AREA_SJI"][1, aslice],
        ".",
        label="IDL",
        alpha=0.5,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("IRIS Effective Area Comparison - SJI 2")
    ax.legend()
    ax.grid()
    return fig


@figure_test
def test_plot_idl_vs_python_sji_3(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    aslice = slice(3110, 3290)
    fig, ax = plt.subplots()
    ax.plot(
        iris_response["LAMBDA"][aslice],
        iris_response["AREA_SJI"][2, aslice],
        ".",
        label="Python",
        alpha=0.5,
    )
    ax.plot(
        idl_response["LAMBDA"][aslice],
        idl_response["AREA_SJI"][2, aslice],
        ".",
        label="IDL",
        alpha=0.5,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("IRIS Effective Area Comparison - SJI 3")
    ax.legend()
    ax.grid()
    return fig


@figure_test
def test_plot_idl_vs_python_sji_4(idl_response):
    iris_response = get_latest_response(parse_time("2025-08-05T22:25:04.723"))

    aslice = slice(3210, 3290)
    fig, ax = plt.subplots()
    ax.plot(
        iris_response["LAMBDA"][aslice],
        iris_response["AREA_SJI"][3, aslice],
        ".",
        label="Python",
        alpha=0.5,
    )
    ax.plot(
        idl_response["LAMBDA"][aslice],
        idl_response["AREA_SJI"][3, aslice],
        ".",
        label="IDL",
        alpha=0.5,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("IRIS Effective Area Comparison - SJI 4")
    ax.legend()
    ax.grid()
    return fig


@figure_test
def test_plot_get_interpolated_effective_area():
    # The idea is that this plot should look the same as the plot for test_plot_idl_vs_python_fuv_sg
    start_obs = parse_time("2025-08-05T22:25:04.723")
    iris_response = get_latest_response(start_obs)
    obs_wavelength = np.linspace(1400.5, 1404.9915000926703, num=692, endpoint=True) * u.Angstrom
    effective_area = get_interpolated_effective_area(
        iris_response,
        detector_type="FUV",
        obs_wavelength=obs_wavelength,
    )
    assert effective_area.shape == obs_wavelength.shape
    assert effective_area.unit.is_equivalent(u.cm**2)
    fig, ax = plt.subplots()
    ax.plot(obs_wavelength.to(u.nm), effective_area)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective Area (cm^2)")
    ax.set_title("Interpolated Effective Area")
    ax.grid()
    return fig
