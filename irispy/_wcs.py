import numpy as np

from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

from sunpy.coordinates.frames import Helioprojective


def _celestial_frame_from_cube(cube):
    """
    The `~sunpy.coordinates.frames.Helioprojective` frame of this observation, as seen
    by the IRIS observer.

    This does not depend on slicing: it works on combined multi-file cubes and on
    cubes sliced along the scan, step, slit, time, or wavelength axes.
    """
    observer = getattr(cube.meta, "observer", None)
    if observer is not None:
        return Helioprojective(observer=observer, obstime=observer.obstime)

    fits_wcs = cube.fits_wcs
    if fits_wcs is None:
        fits_wcs = cube.meta.get("fits_wcs")
    if fits_wcs is None and cube.meta.get("frame_wcs_headers") is not None:
        # A rebinned SJI cube has no fits_wcs, but binning does not change the frame.
        fits_wcs = WCS(cube.meta["frame_wcs_headers"][0])
    if isinstance(fits_wcs, list):
        fits_wcs = fits_wcs[0] if fits_wcs else None
    if fits_wcs is None or not hasattr(fits_wcs, "celestial"):
        msg = "This cube does not carry the WCS metadata needed to derive a celestial frame."
        raise ValueError(msg)
    return wcs_to_celestial_frame(fits_wcs.celestial)


class _ResolveNegativeIndicesMixin:
    """
    Resolve negative indices before NDCube slicing.

    The sliced WCS keeps negative offsets as they are (astropy#15557), so for example
    ``cube[1:5][-1]`` describes ``cube[0]``, and NDMeta raises on negative slices of
    axis-aware keys.

    TODO: delete this once irispy requires an ndcube that resolves negative indices
    itself (the ndcube ``negative-indices`` branch, which also restores ``meta`` when
    slicing raises).
    """

    def __getitem__(self, item):
        if isinstance(item, tuple) and item.count(Ellipsis) == 1:
            # sanitize_slices counts the Ellipsis itself against the dimensionality.
            at = item.index(Ellipsis)
            item = (*item[:at], *[slice(None)] * (len(self.shape) - len(item) + 1), *item[at + 1 :])
        # sanitize_slices raises for any step other than 1, so dropping the step loses nothing.
        item = tuple(
            slice(*index.indices(length)[:2]) if isinstance(index, slice) else range(length)[index]
            for index, length in zip(sanitize_slices(item, len(self.shape)), self.shape, strict=True)
        )
        if 0 not in self.shape and any(isinstance(index, slice) and index.start >= index.stop for index in item):
            msg = "Slicing would give a length-0 axis, which a WCS cannot describe."
            raise IndexError(msg)
        meta = self.meta
        try:
            return super().__getitem__(item)
        finally:
            # NDCube leaves the unsliced cube without its meta when slicing raises.
            self.meta = meta


# Cell edges, in fractions of a cell
_EDGE_TOLERANCE = 1e-9
# Points times slits per pass of the cell search, about 16 MB per float64 temporary
_SEARCH_SIZE = 2_000_000


class _RasterWCS(WCS):
    """
    The FITS-TAB WCS of a raster, inverted by a search of the step table with numpy.

    wcslib inverts a -TAB coordinate by testing every step cell whose bounding box holds
    the point, which costs milliseconds per point on a sit-and-stare, whose cells
    overlap. For irispy's own layout (WAVE, HPLT-TAB, HPLN-TAB; one table of 2 slit ends
    by N steps; identity PC) this inverts the table exactly and returns what wcslib
    returns: the first step cell that holds the point, and NaN outside the table. Any
    other layout, such as ``.sub()`` or ``.celestial``, is inverted by wcslib.
    Everything else, the forward direction included, is wcslib's.
    """

    def _slit_ends(self):
        """
        Each step's two slit ends, as (step, (latitude, longitude)) arrays in degrees.

        `None` for any layout but irispy's raster table.
        """
        wcs = self.wcs
        tab = wcs.tab
        if (
            list(wcs.ctype) != ["WAVE", "HPLT-TAB", "HPLN-TAB"]
            or len(tab) != 1
            or list(tab[0].map) != [1, 2]
            or tab[0].K[0] != 2
            # One exposure has no cells to search
            or tab[0].K[1] < 2
            or not np.array_equal(wcs.get_pc(), np.eye(3))
        ):
            return None
        coord = tab[0].coord
        return coord[:, 0], coord[:, 1]

    def world_to_pixel_values(self, *world_arrays):
        slit_ends = self._slit_ends()
        if slit_ends is None:
            return super().world_to_pixel_values(*world_arrays)
        a, b = slit_ends
        world = np.broadcast_arrays(*(np.asarray(values, dtype=float) for values in world_arrays))
        wavelength, latitude, longitude = (array.ravel() for array in world)
        step, fraction = _invert_raster_table(a, b, np.stack((latitude, longitude), axis=-1))
        wcs = self.wcs
        # The step axis has no index vector, so table step k is FITS index k + 1 = CRVAL3 + CDELT3 (pixel + 1 - CRPIX3)
        step = (step + 1 - wcs.crval[2]) / wcs.cdelt[2] + wcs.crpix[2] - 1
        # astropy does not expose the slit's index vector, but the slit pixel is affine in the
        # fraction along the slit, so wcslib's forward at two slit pixels of the first step gives it
        first_step = (1 - wcs.crval[2]) / wcs.cdelt[2] + wcs.crpix[2] - 1
        _, *ends = self.pixel_to_world_values([0, 0], [0, 1], [first_step, first_step])
        direction = b[0] - a[0]
        pixel_fraction = (np.stack(ends, axis=-1) - a[0]) @ direction / (direction @ direction)
        slit = (fraction - pixel_fraction[0]) / (pixel_fraction[1] - pixel_fraction[0])
        spectral = (wavelength - wcs.crval[0]) / wcs.cdelt[0] + wcs.crpix[0] - 1
        return tuple(pixel.reshape(world[0].shape) for pixel in (spectral, slit, step))


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _invert_raster_table(a, b, points, window=8):
    """
    The fractional table step and the fraction along the slit of each point.

    Cell k is the bilinear patch between the slits of steps k and k + 1, from slit end
    ``a[k]`` to ``b[k]``. Each point gets the first cell that holds it, and NaN if none
    does.

    Every slit is nearly parallel to the mean slit, so the offset of each step's slit
    along the mean normal tells which side of that slit a point lies on, except within a
    proven tolerance of it. No cell before the first step whose offset comes within that
    tolerance can hold the point. The exact search therefore starts there, found by a
    binary search, and tests ``window`` cells, then every cell for the few points still
    unresolved.
    """
    n_steps = len(a)
    direction = b - a
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=-1) / np.hypot(*direction.T)[:, None]
    mean_normal = normal.mean(axis=0)
    mean_normal /= np.hypot(*mean_normal)
    offset = a @ mean_normal
    height = points @ mean_normal
    # Point x is on the side of step k's slit given by the sign of normal_k . (x - a_k), which is
    # height - offset_k + (normal_k - mean_normal) . (x - a_k), and |x - a_k| <= |x - a_0| + |a_k - a_0|
    reach = np.hypot(*(points - a[0]).T) + np.hypot(*(a - a[0]).T).max()
    tolerance = 2 * np.hypot(*(normal - mean_normal).T).max() * reach + 1e-15
    above = np.searchsorted(np.maximum.accumulate(offset), height - tolerance)
    below = np.searchsorted(-np.minimum.accumulate(offset), -(height + tolerance))
    close = np.where(offset[0] < height - tolerance, above, np.where(offset[0] > height + tolerance, below, 0))
    step = np.full(len(points), np.nan)
    fraction = np.full(len(points), np.nan)
    live = np.flatnonzero(close < n_steps)
    first = np.clip(close[live] - 1, 0, max(n_steps - 1 - window, 0))
    cells = first[:, None] + np.arange(min(window, n_steps - 1))
    step[live], fraction[live] = _search_cells(a, b, points[live], cells)
    left = live[np.isnan(step[live]) & (first + window < n_steps - 1)]
    step[left], fraction[left] = _search_cells(a, b, points[left])
    return step, fraction


def _search_cells(a, b, points, cells=None):
    """
    The first cell that holds each point, of its row of ``cells`` or of all cells.

    Returns its fractional table step and the fraction along the slit, NaN where none
    holds it.
    """
    step = np.full(len(points), np.nan)
    fraction = np.full(len(points), np.nan)
    width = len(a) - 1 if cells is None else cells.shape[1]
    chunk = max(1, _SEARCH_SIZE // (width + 1))
    for start in range(0, len(points), chunk):
        rows = slice(start, start + chunk)
        slits = np.arange(width + 1) if cells is None else np.hstack((cells[rows], cells[rows, -1:] + 1))
        x = points[rows, None, :]
        direction, relative = b[slits] - a[slits], x - a[slits]
        side = _cross(direction, relative)
        # Points on a slit, to rounding, such as on the first and last step's
        lengths = np.hypot(direction[..., 0], direction[..., 1]) * np.hypot(relative[..., 0], relative[..., 1])
        on = np.abs(side) <= 1e-12 * lengths
        below = np.signbit(side)
        candidates = (below[:, :-1] != below[:, 1:]) | on[:, :-1] | on[:, 1:]
        todo = candidates.any(axis=1)
        while todo.any():
            point = np.flatnonzero(todo)
            column = candidates[point].argmax(axis=1)
            cell = column if cells is None else cells[start + point, column]
            f, u = _solve_cell(a[cell], b[cell], a[cell + 1], b[cell + 1], x[point, 0])
            found = (f >= -_EDGE_TOLERANCE) & (f <= 1 + _EDGE_TOLERANCE)
            found &= (u >= -_EDGE_TOLERANCE) & (u <= 1 + _EDGE_TOLERANCE)
            step[start + point[found]] = cell[found] + np.clip(f[found], 0, 1)
            fraction[start + point[found]] = np.clip(u[found], 0, 1)
            candidates[point[~found], column[~found]] = False
            todo[point[found]] = False
            todo[point[~found]] = candidates[point[~found]].any(axis=1)
    return step, fraction


def _solve_cell(a0, b0, a1, b1, x):
    """
    Invert the bilinear patch (1 - f) [(1 - u) a0 + u b0] + f [(1 - u) a1 + u b1] = x.
    """
    # x lies on the slit at f, from A(f) = a0 + f da along D(f) = d0 + f dd,
    # so cross(D(f), x - A(f)) = 0, which is quadratic in f
    d0, dd, da = b0 - a0, (b1 - a1) - (b0 - a0), a1 - a0
    r0 = x - a0
    qa = -_cross(dd, da)
    qb = _cross(dd, r0) - _cross(d0, da)
    qc = _cross(d0, r0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -0.5 * (qb + np.copysign(np.sqrt(np.maximum(qb * qb - 4 * qa * qc, 0)), qb))
        # qc / q is the stable root that tends to the linear solution -qc / qb
        near, far = qc / q, q / qa
        linear = np.abs(qa) <= 1e-12 * np.abs(qb)
        inside = (near >= -_EDGE_TOLERANCE) & (near <= 1 + _EDGE_TOLERANCE)
        f = np.where(linear, -qc / qb, np.where(inside, near, far))
        slit = d0 + f[:, None] * dd
        u = np.einsum("ij,ij->i", x - (a0 + f[:, None] * da), slit) / np.einsum("ij,ij->i", slit, slit)
    return f, u
