"""
Tests for irispy.tests.helpers utilities.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import SpectralCoord

from ndcube import NDCube

from irispy.tests.helpers import make_test_spectrogram_cube


def test_make_test_spectrogram_cube_wcs_and_units():
    """
    Lock in axis ordering assumptions: wavelength is the last axis.
    """
    wavelengths = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    data = np.ones((5, 7, len(wavelengths)))
    cube = make_test_spectrogram_cube(data, wavelengths)
    assert isinstance(cube, NDCube)
    assert cube.data.shape == (5, 7, 100)
    # Wavelength should be the last axis
    wcs = cube.wcs
    assert wcs.naxis == 3
    # Verify axis types in array order: last numpy axis is spectral
    assert cube.array_axis_physical_types[2] == ("em.wl",)
    # Spatial axes should be present on the first two axes
    spatial_types = set(cube.array_axis_physical_types[0])
    assert "custom:pos.helioprojective.lon" in spatial_types
    assert "custom:pos.helioprojective.lat" in spatial_types
    # Verify wavelength values round-trip
    spectral_coords = cube.axis_world_coords(-1)
    assert isinstance(spectral_coords[0], SpectralCoord)
    np.testing.assert_allclose(spectral_coords[0].to_value(u.Angstrom), wavelengths.to_value(u.Angstrom))
