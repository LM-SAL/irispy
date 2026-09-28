import numpy as np

import astropy.modeling.models as m
import astropy.units as u


def _time_lookup(times, name=None):
    """
    Linear time lookup over pixel indices, extrapolated to the outer pixel edges.

    The table gains nodes at pixel -0.5 and n - 0.5 on each axis, so pixel corners
    have finite times. Pixels further out get NaN, not a plausible but wrong time.
    """
    # Odd reflection extrapolates linearly to pixels -1 and n; averaging each with its
    # neighbour moves those nodes to the pixel edges.
    table = np.pad(times, 1, mode="reflect", reflect_type="odd")
    for axis in range(table.ndim):
        nodes = np.moveaxis(table, axis, 0)
        nodes[0] = (nodes[0] + nodes[1]) / 2
        nodes[-1] = (nodes[-1] + nodes[-2]) / 2
    points = tuple(np.r_[-0.5, np.arange(n), n - 0.5] * u.pix for n in times.shape)
    tabular = (m.Tabular1D, m.Tabular2D)[times.ndim - 1]
    return tabular(points, table, method="linear", bounds_error=False, fill_value=np.nan, name=name)
