import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import InverseVariance, StdDevUncertainty, UnknownUncertainty

from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils._spectral import check_scaled, in_windows, standard_deviation


def test_check_scaled():
    wavelengths = np.arange(3) * u.nm
    check_scaled(make_test_spectrogram_cube(np.ones((1, 1, 3)), wavelengths))
    integer = make_test_spectrogram_cube(np.ones((1, 1, 3), dtype=np.int16), wavelengths)
    with pytest.raises(ValueError, match="unscaled"):
        check_scaled(integer)
    integer.meta["scaled"] = True  # as the slit-jaw reader gives scaled AIA data
    check_scaled(integer)


@pytest.mark.parametrize("windows", [[[1, 2], [10, 11]], [[10, 11], [1, 2]]])
def test_in_windows_ignores_unavailable_windows(windows):
    wavelengths = np.arange(5) * u.nm
    np.testing.assert_array_equal(in_windows(wavelengths, u.Quantity(windows, u.nm)), [False, True, True, False, False])


def test_in_windows_rejects_empty_union():
    with pytest.raises(ValueError, match="No wavelengths"):
        in_windows(np.arange(5) * u.nm, [[10, 11], [20, 21]] * u.nm)


@pytest.mark.parametrize(
    "uncertainty",
    [
        None,
        StdDevUncertainty(2.0),
        UnknownUncertainty(np.full((1, 1, 3), 2.0)),
        InverseVariance(np.full((1, 1, 3), 0.25)),
        StdDevUncertainty(np.full((1, 1, 3), 2000.0), unit=u.DN / 1000),
    ],
)
def test_standard_deviation(uncertainty):
    cube = make_test_spectrogram_cube(np.ones((1, 1, 3)), np.arange(3) * u.nm, uncertainty=uncertainty)
    sigma = standard_deviation(cube)
    if uncertainty is None:
        assert sigma is None
    else:
        np.testing.assert_allclose(sigma, np.full((1, 1, 3), 2.0))
