import numpy as np
import pytest
from scipy.io import readsav

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from sunpy.time import parse_time

from irispy.data.test import get_test_filepath
from irispy.io.utils import read_files
from irispy.spectrograph import SpectrogramCube, SpectrogramCubeSequence
from irispy.utils.constants import SLIT_WIDTH
from irispy.utils.response import get_latest_response
from irispy.utils.spectrograph import calculate_dn_to_radiance_factor, radiometric_calibration


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


def test_convert_photons_per_sec_to_radiance_vs_peter_young(sns_sg_file):
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]

    solid_angle = cube.wcs.wcs.cdelt[1] * cube.wcs.wcs.cunit[1] * (SLIT_WIDTH)
    iris_response = get_latest_response(parse_time("2014-09-10"))
    factor = calculate_dn_to_radiance_factor(
        iris_response=iris_response,
        wavelength=[1402.77] * u.Angstrom,
        detector_type="FUV",
        # Peter Young does not take this into account, so its set to 1 Angstrom here
        spectral_dispersion_per_pixel=1 * u.AA,
        solid_angle=solid_angle,
    )
    intensity = 1 * factor * 4 * (u.photon / u.s)  # factor assumes data is in photons / s
    idl_unit = u.erg / (u.cm**2 * u.s * u.sr * u.AA)
    intensity = intensity.to(idl_unit)
    """
    IDL> iris_calib, 1, 1402.77, '10-sep-2014' % IRIS_CALIB: instrument
    response version is 009 Eff.

    area:   1.0060 cm^2
    DN:                1.00
    Y-binning:     1
    Exposure time:   1.00 s
    Intensity:      4.363e+01 erg cm^-2 s^-1 sr^-1
    """
    assert_quantity_allclose(intensity, 43.63 * idl_unit, rtol=0.01)
