import numpy as np

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from irispy._gwcs import _table_lookup


def test_table_lookup_reaches_the_outer_pixel_edges_and_no_further():
    values = [10, 20, 40] * u.s
    edges_and_centers = [-0.5, 0, 0.5, 2, 2.5] * u.pix
    assert_quantity_allclose(_table_lookup(values)(edges_and_centers), [5, 10, 15, 40, 50] * u.s)
    assert_quantity_allclose(
        _table_lookup(values, method="nearest")([-0.5, 0.4, 0.6, 2.5] * u.pix), [10, 10, 20, 40] * u.s
    )
    beyond = [-0.6, 2.6] * u.pix
    assert np.isnan(_table_lookup(values)(beyond)).all()
    # A held table is for world to pixel, where a NaN pixel breaks a crop
    assert_quantity_allclose(_table_lookup(values, hold=True)(beyond), [5, 50] * u.s)
