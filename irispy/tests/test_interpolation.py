import numpy as np

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from irispy._interpolation import _time_lookup


def test_time_lookup_is_nan_beyond_the_pixel_edges():
    lookup = _time_lookup(np.array([0.0, 1, 4, 5]) * u.s)
    assert_quantity_allclose(lookup([-1, -0.5, 1.5, 3.5, 4] * u.pix), [np.nan, -0.5, 2.5, 5.5, np.nan] * u.s)

    lookup = _time_lookup(np.arange(6.0).reshape(3, 2) * u.s)
    assert_quantity_allclose(
        lookup([-0.5, 2.5, 3, 1] * u.pix, [-0.5, 1.5, 0, -1] * u.pix), [-1.5, 6.5, np.nan, np.nan] * u.s
    )
