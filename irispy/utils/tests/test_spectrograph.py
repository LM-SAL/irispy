import numpy as np
import pytest
from scipy.io import readsav

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose

from sunpy.time import parse_time

from irispy.data.test import get_test_filepath
from irispy.io.utils import read_files
from irispy.spectrograph import SpectrogramCube
from irispy.utils.constants import SLIT_WIDTH
from irispy.utils.response import get_latest_response
from irispy.utils.spectrograph import (
    calculate_dn_to_radiance_factor,
    radiometric_calibration,
    reshape_1d_wavelength_dimensions_for_broadcast,
)

RADIANCE_UNIT = u.erg / (u.cm**2 * u.s * u.sr * u.AA)


@pytest.fixture
def c_ii_cube(sns_sg_file):
    return read_files(sns_sg_file, spectral_windows="C II 1336")["C II 1336"]


def test_calculate_dn_to_radiance_factor(c_ii_cube):
    idl_wavelength = readsav(get_test_filepath("input_calibration.sav"))["wavelength"] * u.Angstrom
    # IDL factor is in units: erg cm^-2 s^-1 sr^-1 Å^-1
    idl_factor = readsav(get_test_filepath("output_calibration.sav"))["factor"] * RADIANCE_UNIT
    fits_wcs = c_ii_cube.fits_wcs.wcs

    # The slit width is divided by 2 in the IDL code, unsure why.
    factor = calculate_dn_to_radiance_factor(
        iris_response=get_latest_response(parse_time("2025-01-01")),
        wavelength=idl_wavelength,
        detector_type="FUV",
        spectral_dispersion_per_pixel=fits_wcs.cdelt[0] * fits_wcs.cunit[0],
        solid_angle=fits_wcs.cdelt[1] * fits_wcs.cunit[1] * (SLIT_WIDTH / 2),
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


def test_radiometric_calibration_preserves_combined_raster_metadata(raster_sg_files):
    cube = read_files(raster_sg_files)["Si IV 1403"]
    calibrated_cube = radiometric_calibration(cube)

    assert len(calibrated_cube.split_rasters()) == len(cube.split_rasters())
    assert calibrated_cube.raster_slice(0).shape == cube.raster_slice(0).shape


def test_exposure_time_correction_on_combined_cube(raster_sg_files):
    cube = read_files(raster_sg_files[:3], spectral_windows="Si IV 1403", uncertainty=True)["Si IV 1403"]

    corrected = cube.apply_exposure_time_correction()

    expected = cube.raster_slice(1).apply_exposure_time_correction()
    assert corrected.unit == expected.unit
    np.testing.assert_array_equal(corrected.raster_slice(1).data, expected.data)
    np.testing.assert_array_equal(corrected.raster_slice(1).uncertainty.array, expected.uncertainty.array)
    np.testing.assert_allclose(corrected.apply_exposure_time_correction(undo=True).data, cube.data)


def test_radiometric_calibration_of_a_cube_already_per_second(c_ii_cube):
    expected = radiometric_calibration(c_ii_cube)
    np.testing.assert_allclose(radiometric_calibration(c_ii_cube.apply_exposure_time_correction()).data, expected.data)


def test_reshape_1d_wavelength_dimensions_for_broadcast():
    assert reshape_1d_wavelength_dimensions_for_broadcast(np.arange(3) * u.AA, 4).shape == (1, 1, 1, 3)
    with pytest.raises(ValueError, match="at least 1"):
        reshape_1d_wavelength_dimensions_for_broadcast(np.arange(3) * u.AA, 0)


def test_radiometric_calibration_rejects_fixed_wavelength_raster_images(c_ii_cube):
    wavelength = SpectralCoord(c_ii_cube.spectral_axis[5])
    image = c_ii_cube.crop([wavelength, None, None, None], [wavelength, None, None, None])

    with pytest.raises(ValueError, match="requires a spectral axis"):
        radiometric_calibration(image)


def test_convert_photons_per_sec_to_radiance_vs_peter_young(c_ii_cube):
    fits_wcs = c_ii_cube.fits_wcs.wcs
    spectral_dispersion_per_pixel = fits_wcs.cdelt[0] * fits_wcs.cunit[0]
    factor = calculate_dn_to_radiance_factor(
        iris_response=get_latest_response(parse_time("2014-09-10")),
        wavelength=[1402.77] * u.Angstrom,
        detector_type="FUV",
        spectral_dispersion_per_pixel=spectral_dispersion_per_pixel,
        solid_angle=fits_wcs.cdelt[1] * fits_wcs.cunit[1] * SLIT_WIDTH,
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
    cube = read_files(sns_sg_file, uncertainty=True)["C II 1336"]
    new_cube = radiometric_calibration(cube)
    assert isinstance(new_cube.uncertainty, StdDevUncertainty)
    assert new_cube.uncertainty.array.shape == cube.data.shape
