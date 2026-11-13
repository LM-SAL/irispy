import numpy as np
import pytest
from scipy.io import readsav

import astropy.units as u
from astropy import constants
from astropy.modeling.physical_models import BlackBody
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose

from sunpy.time import parse_time

from irispy.data.test import get_test_filepath
from irispy.io.utils import read_files
from irispy.sji import SJICube
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import DN_UNIT, RADIANCE_UNIT, RADIANCE_UNIT_PER_HZ
from irispy.utils.response import get_latest_response
from irispy.utils.spectrograph import (
    calculate_dn_to_radiance_factor,
    radiation_temperature,
    radiometric_calibration,
    subtract_background,
)


@pytest.mark.parametrize("function", [radiometric_calibration, radiation_temperature])
def test_rejects_unscaled_data(function):
    filename = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    cube = read_files(filename, memmap=True)["Mg II k 2796"]
    with pytest.raises(ValueError, match=r"unscaled.*memmap=False"):
        function(cube)


@pytest.fixture
def c_ii_cube(sns_sg_file):
    return read_files(sns_sg_file, spectral_windows="C II 1336")["C II 1336"][0]


def test_calculate_dn_to_radiance_factor(c_ii_cube):
    idl_wavelength = readsav(get_test_filepath("input_calibration.sav"))["wavelength"] * u.Angstrom
    # IDL factor is in units: erg cm^-2 s^-1 sr^-1 Å^-1
    idl_factor = readsav(get_test_filepath("output_calibration.sav"))["factor"] * RADIANCE_UNIT

    # The slit width is divided by 2 in the IDL code, unsure why.
    factor = calculate_dn_to_radiance_factor(
        iris_response=get_latest_response(parse_time("2025-01-01")),
        wavelength=idl_wavelength,
        detector_type="FUV",
        spectral_dispersion_per_pixel=c_ii_cube.spectral_dispersion,
        solid_angle=c_ii_cube.solid_angle / 2,
    )
    assert len(factor) == len(idl_wavelength)
    # Idl output is here to help check values
    # iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits |
    # % IRIS_CALIB_SPECTRUM: Input wavelength range: 1332.68-1337.50 A
    # % IRIS_CALIB_SPECTRUM: Using FUV response
    # % IRIS_CALIB_SPECTRUM: Selected effective area statistics: (min,mean,max)=(0.106070,0.114238,0.122407)
    # % IRIS_CALIB_SPECTRUM: Spectrum converted by: (min,mean,max)=(28811.731,30982.472,33369.609)
    # Python factor assumes data is in photons / s
    factor_in_idl_units = (factor * 4 * (u.photon / u.s)).to(RADIANCE_UNIT)
    # The bundled test file is a 10-pixel stride of the native data along both the
    # spectral and slit axes, so its CDELTs (dispersion and platescale) are 10x the
    # native values the IDL calibration assumed; the per-DN factor is therefore
    # 1/100 of the IDL one.
    # It is not very accurate, hence a large absolute tolerance is used.
    assert_quantity_allclose(factor_in_idl_units * 100, idl_factor, atol=200 * RADIANCE_UNIT)


def test_radiometric_calibration_of_full_sliced_and_rebinned_cube(c_ii_cube):
    calibrated = radiometric_calibration(c_ii_cube)

    assert isinstance(calibrated, SpectrogramCube)
    assert np.any(calibrated.data != c_ii_cube.data)
    assert calibrated.unit == RADIANCE_UNIT
    assert calibrated.data.shape == c_ii_cube.data.shape

    # Slicing returns a SlicedLowLevelWCS without the FITS ``wcs`` attributes.
    cube_slice = c_ii_cube[10, :, :]
    assert not hasattr(cube_slice.wcs, "wcs")
    calibrated_slice = radiometric_calibration(cube_slice)
    expected_slice = calibrated[10, :, :]
    assert isinstance(calibrated_slice, SpectrogramCube)
    assert calibrated_slice.unit == expected_slice.unit
    assert calibrated_slice.data.shape == cube_slice.data.shape
    np.testing.assert_allclose(calibrated_slice.data, expected_slice.data)
    assert np.array_equal(calibrated_slice.mask, expected_slice.mask)

    # Rebinned values are means, so each gets the calibration factor of its native pixels.
    rebinned = c_ii_cube.rebin((1, 2, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = radiometric_calibration(rebinned).data / rebinned.data
        native_factor = (calibrated.data / c_ii_cube.data)[:, ::2]
    usable = np.isfinite(factor) & np.isfinite(native_factor)
    np.testing.assert_allclose(factor[usable], native_factor[usable], rtol=1e-6)


def test_exposure_time_correction_and_calibration_of_a_combined_cube(raster_sg_files):
    cube = read_files(raster_sg_files, spectral_windows="Si IV 1403", uncertainty=True)["Si IV 1403"]

    corrected = cube.apply_exposure_time_correction()

    expected = cube[1].apply_exposure_time_correction()
    assert corrected.unit == expected.unit
    np.testing.assert_array_equal(corrected[1].data, expected.data)
    np.testing.assert_array_equal(corrected[1].uncertainty.array, expected.uncertainty.array)
    np.testing.assert_allclose(corrected.apply_exposure_time_correction(undo=True).data, cube.data)
    # Each raster is calibrated with its own exposure times
    calibrated = radiometric_calibration(cube)
    assert calibrated.shape == cube.shape
    np.testing.assert_array_equal(calibrated[1].data, radiometric_calibration(cube[1]).data)


def test_calculate_dn_to_radiance_factor_vs_peter_young(c_ii_cube):
    spectral_dispersion_per_pixel = c_ii_cube.spectral_dispersion
    factor = calculate_dn_to_radiance_factor(
        iris_response=get_latest_response(parse_time("2014-09-10")),
        wavelength=[1402.77] * u.Angstrom,
        detector_type="FUV",
        spectral_dispersion_per_pixel=spectral_dispersion_per_pixel,
        solid_angle=c_ii_cube.solid_angle,
    )
    intensity = 1 * factor * 4 * (u.photon / u.s)  # factor assumes data is in photons / s
    """
    I changed the IDL code to print out the entire number.

    IDL> iris_calib, 1, 1402.77, '10-sep-2014'
    % IRIS_CALIB: instrument response version is 009
    Eff. area:       1.0060300 cm^2
    DN:               1.0000000000000000
    Y-binning:        1.0000000000000000
    Exposure time:       1.0000000 s
    Intensity:          43.6270027 erg cm^-2 s^-1 sr^-1
    """
    # Peter Young does not take spectral_dispersion_per_pixel into account, so we change our result to match his.
    idl_unit = u.erg / (u.cm**2 * u.s * u.sr)
    intensity = (intensity * spectral_dispersion_per_pixel).to(idl_unit)
    # The bundled test file is a 10-pixel stride along the slit, so its slit-axis
    # platescale (and hence solid angle) is 10x native, while the IDL run above
    # assumed Y-binning of 1; scale back to compare against the native value.
    # (The spectral dispersion cancels out because it is multiplied back in above.)
    assert_quantity_allclose(intensity * 10, 43.6270027 * idl_unit, rtol=0.0003)


def test_radiometric_calibration_keeps_a_standard_deviation(sns_sg_file):
    cube = read_files(sns_sg_file, uncertainty=True)["C II 1336"][0]
    new_cube = radiometric_calibration(cube)
    assert isinstance(new_cube.uncertainty, StdDevUncertainty)
    assert new_cube.uncertainty.array.shape == cube.data.shape
    assert new_cube.data.dtype == new_cube.uncertainty.array.dtype == np.float32


def test_radiometric_calibration_does_not_share_meta_coordinates_or_mask(sns_sg_file):
    cube = read_files(sns_sg_file)["C II 1336"][0]
    mask = cube.mask.copy()
    new_cube = radiometric_calibration(cube)
    new_cube.meta["note"] = "changed"
    new_cube.extra_coords.add("sequence", 0, np.arange(len(new_cube.data)) * u.one)
    new_cube.mask[...] = ~new_cube.mask
    assert "note" not in cube.meta
    assert "sequence" not in tuple(cube.extra_coords.keys())
    np.testing.assert_array_equal(cube.mask, mask)


WAVELENGTHS = (2796 + np.arange(4)) * u.AA


def make_radiance_cube(radiance, *, unit=RADIANCE_UNIT, uncertainty=None, mask=None):
    equivalencies = u.spectral_density(WAVELENGTHS)
    if uncertainty is not None:
        uncertainty = StdDevUncertainty(uncertainty.to_value(unit, equivalencies=equivalencies))
    return make_test_spectrogram_cube(
        radiance.to_value(unit, equivalencies=equivalencies), WAVELENGTHS, uncertainty=uncertainty, unit=unit, mask=mask
    )


@pytest.mark.parametrize("unit", [RADIANCE_UNIT_PER_HZ, RADIANCE_UNIT])
def test_radiation_temperature_inverts_planck(unit):
    radiance = BlackBody(5000 * u.K)(WAVELENGTHS)[np.newaxis, np.newaxis]
    temperature = radiation_temperature(make_radiance_cube(radiance, unit=unit))
    assert temperature.unit == u.K
    np.testing.assert_allclose(temperature.data, 5000, rtol=1e-6)


def test_radiation_temperature_uncertainty_and_mask():
    radiance = [[[1e-6, 2e-6, 0, -1e-6]]] * RADIANCE_UNIT_PER_HZ
    mask = np.array([[[False, True, False, False]]])
    temperature = radiation_temperature(make_radiance_cube(radiance, uncertainty=1e-8 * radiance.unit, mask=mask))
    # The input mask is kept and samples without a positive radiance are NaN and masked
    assert np.array_equal(temperature.mask, [[[False, True, True, True]]])
    assert np.isfinite(temperature.data[0, 0, :2]).all()
    assert np.isnan(temperature.data[0, 0, 2:]).all()
    # First-order error against a central difference
    step = 1e-12 * RADIANCE_UNIT_PER_HZ
    finite_difference = (
        radiation_temperature(make_radiance_cube(radiance + step)).data[0, 0, 0]
        - radiation_temperature(make_radiance_cube(radiance - step)).data[0, 0, 0]
    ) / 2e-12
    np.testing.assert_allclose(temperature.uncertainty.array[0, 0, 0], finite_difference * 1e-8, rtol=1e-6)


def test_radiation_temperature_on_level_2_cube(sns_sg_file):
    cube = read_files(sns_sg_file, uncertainty=True)["C II 1336"][0]
    with pytest.raises(ValueError, match="radiometric_calibration"):
        radiation_temperature(cube)
    temperature = radiation_temperature(radiometric_calibration(cube))
    assert np.all(temperature.mask[cube.mask])
    assert np.isfinite(temperature.data[~temperature.mask]).all()
    assert (temperature.uncertainty.array[~temperature.mask] > 0).all()


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_subtract_background(degree):
    # A line on a polynomial background, with a masked spike in a window and a spectrum
    # left with too few samples to fit
    wavelengths = np.linspace(1333, 1337, 81) * u.AA
    x = wavelengths.to_value(u.AA) - 1335
    line = 50 * np.exp(-0.5 * ((x - 0.7) / 0.05) ** 2)
    background = np.polynomial.polynomial.polyval(x, [3, 0.5, -0.2][: degree + 1])
    data = np.tile(line + background, (2, 3, 1))
    data[0, 0, 0] = 1e6
    windows = [[1333, 1334], [1336.5, 1337]] * u.AA
    window_mask = (wavelengths <= 1334 * u.AA) | (wavelengths >= 1336.5 * u.AA)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 2, np.flatnonzero(window_mask)[degree:]] = True
    cube = make_test_spectrogram_cube(
        data, wavelengths, uncertainty=StdDevUncertainty(np.full(data.shape, 0.1)), mask=mask
    )
    result = subtract_background(cube, windows, degree=degree)
    expected = np.tile(line, (2, 3, 1))
    expected[0, 0, 0] = 1e6 - background[0]
    expected[1, 2] = np.nan
    np.testing.assert_allclose(result.data, expected, atol=1e-9)
    np.testing.assert_array_equal(result.mask, cube.mask)
    np.testing.assert_array_equal(result.uncertainty.array, cube.uncertainty.array)
    assert result.unit == cube.unit


def test_subtract_background_rounds_the_float64_fit_to_float32(sns_sg_file):
    cube = read_files(sns_sg_file)["Mg II k 2796"][0]
    windows = [[2793.4, 2794.5], [2799.5, 2800.5], [2805, 2806.4]] * u.AA
    as_float64 = SpectrogramCube(cube.data.astype(float), cube.wcs, unit=cube.unit, meta=cube.meta, mask=cube.mask)
    result = subtract_background(cube, windows, degree=3)
    expected = subtract_background(as_float64, windows, degree=3)
    assert result.data.dtype == np.float32
    # Rounded to float32 once in the background and once in the difference
    background = cube.data - expected.data
    # Only the fitted samples, as np.spacing warns about NaN on Windows
    fitted = np.isfinite(expected.data)
    rounding = np.spacing(np.maximum(np.abs(background), np.abs(expected.data))[fitted].astype(np.float32))
    assert np.all(np.abs(result.data - expected.data)[fitted] <= rounding)


def test_subtract_background_integer_mask():
    # An integer mask, which the Mg II feature finder accepts, raised in the background fit
    wavelengths = np.linspace(1333, 1337, 5) * u.AA
    data = np.full((2, 3, 5), 3.0)
    data[0, 0, 0] = 1e6
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[0, 0, 0] = 1
    windows = [[1333, 1334], [1336, 1337]] * u.AA
    result = subtract_background(make_test_spectrogram_cube(data, wavelengths, mask=mask), windows, degree=0)
    boolean = make_test_spectrogram_cube(data, wavelengths, mask=mask.astype(bool))
    expected = subtract_background(boolean, windows, degree=0)
    np.testing.assert_array_equal(result.data, expected.data)
    np.testing.assert_array_equal(result.mask, mask.astype(bool))


@pytest.mark.parametrize(
    ("windows", "match"),
    [
        ([1333, 1334, 1335] * u.AA, r"shape \(2,\) or \(n, 2\)"),
        ([1340, 1341] * u.AA, "No wavelengths between"),
        ([1333, 1334], "windows must be wavelengths"),
    ],
)
def test_subtract_background_rejects_bad_windows(windows, match):
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.linspace(1333, 1337, 5) * u.AA)
    with pytest.raises(ValueError, match=match):
        subtract_background(cube, windows)


def test_subtract_background_rejects_unscaled_data():
    cube = make_test_spectrogram_cube(
        np.ones((1, 1, 5)), np.linspace(1333, 1337, 5) * u.AA, unit=DN_UNIT["FUV_UNSCALED"]
    )
    with pytest.raises(ValueError, match="unscaled"):
        subtract_background(cube, [1333, 1334] * u.AA)


def test_subtract_background_keeps_inf():
    wavelengths = np.linspace(1333, 1337, 81) * u.AA
    data = np.tile(3 + 0.5 * (wavelengths.to_value(u.AA) - 1335), (1, 2, 1))
    data[0, 0, 0] = np.inf  # in a window, so left out of the fit
    data[0, 1, 40] = np.inf  # in the line
    cube = make_test_spectrogram_cube(data, wavelengths)
    corrected = subtract_background(cube, [[1333, 1334], [1336, 1337]] * u.AA)
    assert np.isposinf(corrected.data[0, 0, 0])
    assert np.isposinf(corrected.data[0, 1, 40])
    corrected.data[0, 0, 0] = corrected.data[0, 1, 40] = 0
    np.testing.assert_allclose(corrected.data, 0, atol=1e-9)


def test_radiometric_calibration_of_slit_jaw_images(sns_sji_1330_file):
    cube = read_files(sns_sji_1330_file, uncertainty=True)
    response = get_latest_response(cube.meta.date_reference)
    wavelength, area = response["LAMBDA"], response["AREA_SJI"][0]
    # 12 photons per DN in the FUV slit-jaw passbands
    photon_rate = (
        cube.data
        * 12
        / cube.exposure_time[:, np.newaxis, np.newaxis]
        / (cube.meta["CDELT1"] * cube.meta["CDELT2"] * u.arcsec**2)
    )
    hc = constants.h * constants.c
    line = 1336 * u.AA
    for calibrated, expected in (
        (radiometric_calibration(cube, line=line), photon_rate * hc / line / np.interp(line, wavelength, area)),
        (radiometric_calibration(cube), photon_rate * hc / np.trapezoid(area * wavelength, wavelength)),
    ):
        assert isinstance(calibrated, SJICube)
        assert calibrated.data.dtype == calibrated.uncertainty.array.dtype == np.float32
        assert_quantity_allclose(calibrated.data * calibrated.unit, expected, rtol=1e-6)
        np.testing.assert_allclose(
            calibrated.uncertainty.array * cube.data, calibrated.data * cube.uncertainty.array, rtol=1e-6
        )


@pytest.mark.parametrize("line", [1216 * u.AA, 5000 * u.AA])
def test_radiometric_calibration_rejects_line_outside_slit_jaw_passband(sns_sji_1330_file, line):
    cube = read_files(sns_sji_1330_file)
    with pytest.raises(ValueError, match="1330 slit-jaw passband has no effective area"):
        radiometric_calibration(cube, line=line)


def test_radiometric_calibration_needs_an_iris_slit_jaw_passband(aia_1700_file):
    with pytest.raises(ValueError, match="no IRIS slit-jaw response for 1700"):
        radiometric_calibration(read_files(aia_1700_file))


def test_radiometric_calibration_needs_one_line_wavelength(sns_sji_1330_file):
    with pytest.raises(ValueError, match="line must be one wavelength"):
        radiometric_calibration(read_files(sns_sji_1330_file), line=[1335, 1336] * u.AA)


def test_radiometric_calibration_rejects_line_for_spectra(sns_sg_file):
    cube = read_files(sns_sg_file, spectral_windows="C II 1336")["C II 1336"][0]
    with pytest.raises(ValueError, match="line is for slit-jaw cubes"):
        radiometric_calibration(cube, line=1336 * u.AA)
