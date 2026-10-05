import numpy as np
import pytest
from scipy.io import readsav

import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose

from sunpy.time import parse_time

from irispy.data.test import get_test_filepath
from irispy.io.utils import read_files
from irispy.spectrograph import SpectrogramCube, SpectrogramCubeSequence
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import RADIANCE_UNIT, SLIT_WIDTH
from irispy.utils.response import get_latest_response
from irispy.utils.spectrograph import calculate_dn_to_radiance_factor, radiation_temperature, radiometric_calibration

RADIANCE_PER_HZ = u.erg / u.cm**2 / u.s / u.sr / u.Hz


@pytest.mark.parametrize("function", [radiometric_calibration, radiation_temperature])
@pytest.mark.parametrize("sequence", [False, True])
def test_rejects_unscaled_data(function, sequence):
    filename = get_test_filepath(
        "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
    )
    cubes = read_files(filename, memmap=True)["Mg II k 2796"]
    with pytest.raises(ValueError, match=r"unscaled.*memmap=False"):
        function(cubes if sequence else cubes[0])


@pytest.fixture
def idl_input_rad_cal():
    # Has 'input_spectrum' and 'wavelength' keys
    return readsav(get_test_filepath("input_calibration.sav"))


@pytest.fixture
def idl_output_rad_cal():
    # Has 'outputspectrum' and 'factor' keys
    return readsav(get_test_filepath("output_calibration.sav"))


def test_calculate_dn_to_radiance_factor(sns_sg_file, idl_input_rad_cal, idl_output_rad_cal):
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    idl_wavelength = idl_input_rad_cal["wavelength"] * u.Angstrom
    idl_factor_cgs = idl_output_rad_cal["factor"]

    spectral_dispersion_per_pixel = cube.wcs.wcs.cdelt[0] * cube.wcs.wcs.cunit[0]
    # The slit width is divided by 2 in the IDL code, unsure why.
    solid_angle = cube.wcs.wcs.cdelt[1] * cube.wcs.wcs.cunit[1] * (SLIT_WIDTH / 2)
    iris_response = get_latest_response(parse_time("2025-01-01"))
    factor = calculate_dn_to_radiance_factor(
        iris_response=iris_response,
        wavelength=idl_wavelength,
        detector_type="FUV",
        spectral_dispersion_per_pixel=spectral_dispersion_per_pixel,
        solid_angle=solid_angle,
    )
    assert len(factor) == len(idl_wavelength)
    # Idl output is here to help check values
    # iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits |
    # % IRIS_CALIB_SPECTRUM: Input wavelength range: 1332.68-1337.50 A
    # % IRIS_CALIB_SPECTRUM: Using FUV response
    # % IRIS_CALIB_SPECTRUM: Selected effective area statistics: (min,mean,max)=(0.106070,0.114238,0.122407)
    # % IRIS_CALIB_SPECTRUM: Spectrum converted by: (min,mean,max)=(28811.731,30982.472,33369.609)
    # IDL factor is in units: erg cm^-2 s^-1 sr^-1 Å^-1
    idl_unit = u.erg / (u.cm**2 * u.s * u.sr * u.AA)
    idl_factor = idl_factor_cgs * idl_unit
    # Convert Python factor to IDL units for comparison
    # Python factor assumes data is in photons / s
    factor = factor * 4 * (u.photon / u.s)
    factor_in_idl_units = factor.to(idl_unit)
    # The bundled test file is a 10-pixel stride of the native data along both the
    # spectral and slit axes, so its CDELTs (dispersion and platescale) are 10x the
    # native values the IDL calibration assumed; the per-DN factor is therefore
    # 1/100 of the IDL one.
    factor_in_idl_units = factor_in_idl_units * 100
    # It is not very accurate, hence a large absolute tolerance is used.
    assert_quantity_allclose(factor_in_idl_units, idl_factor, atol=200 * idl_unit)


def test_radiometric_calibration(sns_sg_file):
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    new_cube = radiometric_calibration(cube)
    assert isinstance(new_cube, SpectrogramCube)

    assert np.any(new_cube.data != cube.data)
    assert new_cube.unit == u.erg / (u.cm**2 * u.s * u.sr * u.AA)
    assert new_cube.data.shape == cube.data.shape

    sequence = raster_collection["C II 1336"]
    new_sequence = radiometric_calibration(sequence)
    assert isinstance(new_sequence, SpectrogramCubeSequence)
    assert new_sequence[0].unit == u.erg / (u.cm**2 * u.s * u.sr * u.AA)
    assert new_sequence[0].data.shape == sequence[0].data.shape
    assert np.any(new_sequence[0].data != sequence[0].data)


def test_radiometric_calibration_single_sliced_raster_cube(sns_sg_file):
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    # The slicing operation, returns a slicedWCS which breaks the code
    cube_slice = cube[10, :, :]
    # It also returns a wcs object which does not have the normal lower level wcs attributes
    assert not hasattr(cube_slice.wcs, "wcs")

    calibrated_full_cube = radiometric_calibration(cube)
    calibrated_slice = radiometric_calibration(cube_slice)
    expected_slice = calibrated_full_cube[10, :, :]

    assert isinstance(calibrated_slice, SpectrogramCube)
    assert calibrated_slice.unit == expected_slice.unit
    assert calibrated_slice.data.shape == cube_slice.data.shape
    np.testing.assert_allclose(calibrated_slice.data, expected_slice.data)
    assert np.array_equal(calibrated_slice.mask, expected_slice.mask)


def test_convert_photons_per_sec_to_radiance_vs_peter_young(sns_sg_file):
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]

    solid_angle = cube.wcs.wcs.cdelt[1] * cube.wcs.wcs.cunit[1] * (SLIT_WIDTH)
    spectral_dispersion_per_pixel = cube.wcs.wcs.cdelt[0] * cube.wcs.wcs.cunit[0]
    iris_response = get_latest_response(parse_time("2014-09-10"))
    factor = calculate_dn_to_radiance_factor(
        iris_response=iris_response,
        wavelength=[1402.77] * u.Angstrom,
        detector_type="FUV",
        spectral_dispersion_per_pixel=spectral_dispersion_per_pixel,
        solid_angle=solid_angle,
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
    # Peter Young does not take spectral_dispersion_per_pixel into account
    intensity = intensity * spectral_dispersion_per_pixel
    # so we change our result to match his.
    idl_unit = u.erg / (u.cm**2 * u.s * u.sr)
    intensity = intensity.to(idl_unit)
    # The bundled test file is a 10-pixel stride along the slit, so its slit-axis
    # platescale (and hence solid angle) is 10x native, while the IDL run above
    # assumed Y-binning of 1; scale back to compare against the native value.
    # (The spectral dispersion cancels out because it is multiplied back in above.)
    intensity = intensity * 10
    assert_quantity_allclose(intensity, 43.6270027 * idl_unit, rtol=0.0003)


def test_radiometric_calibration_keeps_a_standard_deviation(sns_sg_file):
    cube = read_files(sns_sg_file, uncertainty=True)["C II 1336"][0]
    new_cube = radiometric_calibration(cube)
    assert isinstance(new_cube.uncertainty, StdDevUncertainty)
    assert new_cube.uncertainty.array.shape == cube.data.shape


def make_radiance_cube(radiance, *, uncertainty=None, mask=None):
    wavelengths = (2796 + np.arange(radiance.shape[-1])) * u.AA
    template = make_test_spectrogram_cube(np.zeros(radiance.shape), wavelengths)
    equivalencies = u.spectral_density(wavelengths)
    radiance = radiance.to(RADIANCE_UNIT, equivalencies=equivalencies)
    if uncertainty is not None:
        uncertainty = StdDevUncertainty(uncertainty.to_value(RADIANCE_UNIT, equivalencies=equivalencies))
    return SpectrogramCube(radiance.value, template.wcs, uncertainty, radiance.unit, template.meta, mask=mask)


@pytest.mark.parametrize("unit", [RADIANCE_PER_HZ, RADIANCE_UNIT])
def test_radiation_temperature_is_planck_not_rayleigh_jeans(unit):
    radiance = np.full((1, 1, 3), 1e-6) * RADIANCE_PER_HZ
    # astropy's brightness_temperature is the Rayleigh-Jeans limit, which fails in the UV
    rayleigh_jeans = radiance[0, 0, 0].to(u.K, equivalencies=u.brightness_temperature(2796 * u.AA))
    assert_quantity_allclose(rayleigh_jeans, 2.831 * u.K, rtol=1e-3)
    cube = make_radiance_cube(radiance)
    if unit == RADIANCE_PER_HZ:
        cube = cube.to(unit, equivalencies=u.spectral_density(cube.axis_world_coords("em.wl")[0]))
    temperature = radiation_temperature(cube)
    assert temperature.unit == u.K
    assert_quantity_allclose(temperature.data[0, 0, 0] * u.K, 5246.6 * u.K, rtol=1e-4)


def test_radiation_temperature_uncertainty_and_mask():
    radiance = [[[1e-6, 2e-6, 0, -1e-6]]] * RADIANCE_PER_HZ
    mask = np.array([[[False, True, False, False]]])
    temperature = radiation_temperature(make_radiance_cube(radiance, uncertainty=1e-8 * radiance.unit, mask=mask))
    # The input mask is kept and samples without a positive radiance are NaN and masked
    assert np.array_equal(temperature.mask, [[[False, True, True, True]]])
    assert np.isfinite(temperature.data[0, 0, :2]).all()
    assert np.isnan(temperature.data[0, 0, 2:]).all()
    # First-order error against a central difference
    step = 1e-12 * RADIANCE_PER_HZ
    finite_difference = (
        radiation_temperature(make_radiance_cube(radiance + step)).data[0, 0, 0]
        - radiation_temperature(make_radiance_cube(radiance - step)).data[0, 0, 0]
    ) / 2e-12
    np.testing.assert_allclose(temperature.uncertainty.array[0, 0, 0], finite_difference * 1e-8, rtol=1e-6)


def test_radiation_temperature_needs_calibrated_data(sns_sg_file):
    cube = read_files(sns_sg_file)["C II 1336"][0]
    with pytest.raises(ValueError, match="radiometric_calibration"):
        radiation_temperature(cube)
    temperature = radiation_temperature(radiometric_calibration(cube))
    assert temperature.unit == u.K
    assert temperature.data.shape == cube.data.shape
    assert np.all(temperature.mask[cube.mask])
