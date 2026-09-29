"""
Positions and intensities of the Mg II h and k line centres and emission peaks.

A port of the SolarSoft routines ``iris_get_mg_features_lev2.pro`` and
``iris_get_mg_features.pro``.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import convolve1d

import astropy.units as u
from astropy.constants import c

from irispy.spectrograph import RasterCollection
from irispy.utils._spectral import make_map_cube, make_spatial_template
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED

__all__ = ["calculate_mg_features"]

_REST_WAVELENGTH = {"k": 279.63509493 * u.nm, "h": 280.35297192 * u.nm}  # vacuum
_SAMPLES = 300  # points of the velocity grid the spectra are interpolated to
_SPACING = 10  # grid points within which only the strongest extremum counts
# Line centres further than 3 km/s from this Gaussian (sigma 2 pixels) average along the slit are redone.
_KERNEL = np.exp(-0.5 * (np.arange(-8, 9) / 2) ** 2)
_KERNEL /= _KERNEL.sum()
# Numbers of (minima, maxima) inside the grid for which the middle minimum is the line centre, and for
# which it is the lowest minimum between the two highest maxima.
_MIDDLE_MINIMUM = [(1, 2), (3, 1), (3, 2), (3, 4), (5, 4), (7, 6)]
_BETWEEN_MAXIMA = [(2, 2), (2, 3), (3, 3), (4, 2), (4, 3), (4, 4)]


def calculate_mg_features(cube, *, velocity_range=(-40, 40) * u.km / u.s, lines=("k", "h")):
    """
    Find the line centres and emission peaks of the Mg II h and k lines.

    For each spectrum, the k3 and h3 line centres and the k2v, k2r, h2v and h2r emission peaks
    are found as in Pereira et al. (2013): the spectrum is interpolated to a fine velocity grid,
    its local extrema are classified, and the line centre is refined by a parabola fitted to
    its minimum. Line centres that jump along the slit are redone from their neighbours.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        One raster of a window covering the Mg II lines, read with ``memmap=False``,
        with axes (step, slit, wavelength).
    velocity_range : `~astropy.units.Quantity`, optional
        Doppler velocities, relative to each line's rest wavelength, in which the features are searched for.
    lines : `tuple` of `str`, optional
        The lines to measure, ``"k"`` (279.635 nm) and/or ``"h"`` (280.353 nm, both in vacuum).

    Returns
    -------
    `~irispy.spectrograph.RasterCollection`
        Maps on the (step, slit) plane of the input, keyed ``"{feature}_velocity"`` (km/s) and
        ``"{feature}_intensity"`` (the unit of ``cube``) for the features ``"k2v"``, ``"k3"``,
        ``"k2r"``, ``"h2v"``, ``"h3"`` and ``"h2r"``. Features that are not found are NaN and masked.

    Notes
    -----
    This port finds the same line centres as IDL for almost every spectrum, and peaks within
    a fraction of a km/s. Its deliberate differences are:

    * The peaks are refined with the parabola through the highest point of the velocity grid
      and its neighbours. IDL evaluates a spline on a decreasing grid, which IDL's ``SPLINE``
      does not support; it extrapolates one interval and moves the peaks by about 0.1 km/s.
    * Of two minima within 10 grid points, the deeper one counts; IDL keeps the larger one.
    * Spectra with missing data in the velocity range give no features; IDL interpolates
      through the -200 fill values.
    * When looking for line centres that jump along the slit, IDL zeroes the last one if none
      is missing.
    * The minimum searched for near a guessed line centre is found at the right place when the
      guess is less than 15 grid points from the grid's end.
    * The reversed rasters of V34 observations are not treated specially (IDL descales them
      twice), and the rest wavelength of k is 279.635 nm also when h is not measured (IDL's
      ``/onlyk`` uses 279.644 nm).
    * Each slit is processed at once, which is about 7 times faster than IDL.

    References
    ----------
    * `Pereira et al. (2013), ApJ 778, 143 <https://doi.org/10.1088/0004-637X/778/2/143>`__
    * `iris_get_mg_features.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/uio/utils/iris_get_mg_features.pro>`__
    """
    wavelength = cube.axis_world_coords(cube.wavelength_axis)[0]
    missing = ~np.isfinite(cube.data) | (cube.data == BAD_PIXEL_VALUE_SCALED)
    if cube.mask is not None:
        missing |= np.broadcast_to(cube.mask, cube.data.shape)
    low, high = velocity_range.to_value(u.km / u.s)
    template = make_spatial_template(cube, cube.wavelength_axis)
    maps = []
    for line in lines:
        velocity = ((wavelength - _REST_WAVELENGTH[line]) / _REST_WAVELENGTH[line] * c).to_value(u.km / u.s)
        if not velocity[0] <= low < high <= velocity[-1]:
            msg = f"The spectral window does not cover Mg II {line} from {low} to {high} km/s"
            raise ValueError(msg)
        inside = (velocity >= low - 3) & (velocity <= high + 3)
        grid = np.linspace(velocity[inside][0], velocity[inside][-1], _SAMPLES)
        features = np.full((*cube.data.shape[:2], 3, 2), np.nan)  # blue peak, centre, red peak
        for step, (data, bad) in enumerate(zip(cube.data[..., inside], missing[..., inside], strict=True)):
            valid = ~bad.any(axis=-1)
            if valid.any():
                spectra = _spline(velocity[inside], data[valid].T, grid, tension=0).T
                features[step, valid] = _slit_features(grid, spectra, valid)
        for index, feature in enumerate(("2v", "3", "2r")):
            name = f"{line}{feature}"
            maps.append(
                (f"{name}_velocity", make_map_cube(template, features[..., index, 0], u.km / u.s, mask_invalid=True))
            )
            maps.append(
                (f"{name}_intensity", make_map_cube(template, features[..., index, 1], cube.unit, mask_invalid=True))
            )
    return RasterCollection(maps, aligned_axes=tuple(range(len(template.shape))))


def _slit_features(grid, spectra, valid):
    """
    Blue peak, line centre and red peak, each as (velocity, intensity), of the valid
    spectra of one slit.
    """
    maxima, minima = _extrema(spectra, maxima=True), _extrema(spectra, maxima=False)
    centre = _centres(grid, spectra, maxima, minima, 0.0)
    positions = np.flatnonzero(valid)
    kernel = _KERNEL if valid.size >= _KERNEL.size else np.ones(1)
    for _ in range(2):
        # Redo the line centres that jump from the average along the slit, in which missing ones
        # count as 0 km/s, from the spline through the others.
        slit = np.zeros(valid.size)
        slit[positions] = np.nan_to_num(centre[:, 0])
        good = np.abs(centre[:, 0] - convolve1d(slit, kernel, mode="nearest")[positions]) <= 3
        if np.count_nonzero(good) >= 3:
            guess = _spline(positions[good], centre[good, 0], positions, tension=1)
            redo = ~good
            centre[redo] = _centres(grid, spectra[redo], maxima[redo], minima[redo], guess[redo], forced=True)
    redo = np.isnan(centre[:, 0])
    centre[redo] = _centres(grid, spectra[redo], maxima[redo], minima[redo], 5.0, use_derivative=True)
    blue, red = _peaks(grid, spectra, maxima, centre[:, 0])
    return np.stack([blue, centre, red], axis=1)


def _extrema(spectra, *, maxima):
    """
    Mask of the local extrema of each spectrum, leaving out those within ``_SPACING``
    points of a stronger one (``lclxtrem.pro``).
    """
    strength = spectra if maxima else -spectra
    turn = np.diff(np.sign(np.diff(spectra, axis=-1)), axis=-1)
    candidate = np.zeros(spectra.shape, dtype=bool)
    candidate[:, 1:-1] = turn < 0 if maxima else turn > 0
    flat = ~candidate.any(axis=-1)  # no turning point: the extreme value is the extremum
    candidate[flat, np.argmax(strength[flat], axis=-1)] = True
    # Keep the candidates from the strongest down, unless a kept one is too close.
    order = np.argsort(np.where(candidate, -strength, np.inf), axis=-1)[:, : candidate.sum(axis=-1).max()]
    kept = np.take_along_axis(candidate, order, axis=-1)
    for rank in range(1, order.shape[1]):
        close = np.abs(order[:, :rank] - order[:, rank, np.newaxis]) <= _SPACING
        kept[:, rank] &= ~(close & kept[:, :rank]).any(axis=-1)
    extrema = np.zeros_like(candidate)
    np.put_along_axis(extrema, order, kept, axis=-1)
    return extrema


def _centres(grid, spectra, maxima, minima, guess, *, forced=False, use_derivative=False):
    """
    Line centres (``mg_single`` in ``iris_get_mg_features.pro``), as an (n, 2) array.

    The extrema away from the grid's ends decide where the centre is: next to a guessed
    velocity, from a parabola fitted to the lowest point there, or, for a single blended
    peak, where the spectrum is flattest on its far side (only with ``use_derivative``).
    With ``forced``, the guess is used unless there is a single peak.
    """
    rows = np.arange(len(spectra))
    guess = np.broadcast_to(guess, rows.shape).astype(float)
    half_width = np.full(rows.shape, 15)
    if forced:
        maxima = maxima & (grid > grid[0] + 20) & (grid < grid[-1] - 20)
        blended = np.count_nonzero(maxima, axis=-1) == 1
    else:
        inner = (grid > grid[0] + 10) & (grid < grid[-1] - 10)
        maxima, minima = maxima & inner, minima & inner
        counts = np.stack([np.count_nonzero(minima, axis=-1), np.count_nonzero(maxima, axis=-1)], axis=-1)
        middle = (counts[:, np.newaxis] == _MIDDLE_MINIMUM).all(axis=-1).any(axis=-1)
        between = ~middle & (counts[:, np.newaxis] == _BETWEEN_MAXIMA).all(axis=-1).any(axis=-1)
        pair = ~middle & ~between & (counts == (2, 1)).all(axis=-1)
        index = np.arange(grid.size)
        rank = np.cumsum(minima, axis=-1) - 1
        guess = np.where(middle, grid[np.argmax(minima & (rank == counts[:, :1] // 2), axis=-1)], guess)
        highest = np.sort(np.argpartition(np.where(maxima, spectra, -np.inf), -2, axis=-1)[:, -2:], axis=-1)
        enclosed = minima & (index > highest[:, :1]) & (index < highest[:, 1:])
        found = between & enclosed.any(axis=-1)  # otherwise the guess stays
        guess = np.where(found, grid[np.argmin(np.where(enclosed, spectra, np.inf), axis=-1)], guess)
        lowest = np.argmin(np.where(minima, spectra, np.inf), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            contrast = np.where(minima, spectra, -np.inf).max(axis=-1) / spectra[rows, lowest]
        deep = pair & (contrast > 1.3)
        single = ~middle & ~between & ~pair & (counts[:, 0] == 1) & (counts[:, 1] != 1)
        guess = np.where(deep | single, grid[lowest], guess)
        half_width[middle | deep] = 7
        blended = (pair & ~deep) | (~middle & ~between & ~pair & (counts[:, 1] == 1))
    centre = np.full((rows.size, 2), np.nan)
    fit = ~blended
    centre[fit] = _parabola_minimum(grid, spectra[fit], guess[fit], half_width[fit])
    if use_derivative:
        centre[blended] = _flattest(grid, spectra[blended], np.argmax(maxima[blended], axis=-1))
    return centre


def _parabola_minimum(grid, spectra, guess, half_width):
    """
    Vertex of the parabola fitted to the 7 points around the lowest point near
    ``guess``.
    """
    rows = np.arange(len(spectra))
    spacing = grid[1] - grid[0]
    nearest = np.clip(np.rint((guess - grid[0]) / spacing), 0, grid.size - 1).astype(int)
    index = np.arange(grid.size)
    near = (index >= (nearest - half_width)[:, np.newaxis]) & (index < (nearest + half_width)[:, np.newaxis])
    lowest = np.argmin(np.where(near, spectra, np.inf), axis=-1)
    points = lowest[:, np.newaxis] + np.arange(-3, 4)
    used = (points >= 0) & (points < grid.size)
    points = np.clip(points, 0, grid.size - 1)
    x = np.where(used, grid[points] - grid[lowest, np.newaxis], 0)
    powers = x[..., np.newaxis] ** np.arange(3)
    normal = np.einsum("nki,nkj->nij", powers, powers * used[..., np.newaxis])
    moments = np.einsum("nki,nk->ni", powers * used[..., np.newaxis], spectra[rows[:, np.newaxis], points])
    a = np.linalg.solve(normal, moments[..., np.newaxis])[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = -a[:, 1] / (2 * a[:, 2])
        value = a[:, 0] - a[:, 2] * vertex**2
    close = np.abs(vertex) <= 4 * spacing
    return np.column_stack([np.where(close, grid[lowest] + vertex, np.nan), np.where(close, value, np.nan)])


def _flattest(grid, spectra, peak):
    """
    For a single blended peak: where the spectrum is flattest within 15 km/s of it, on
    its higher side.
    """
    rows = np.arange(len(spectra))
    margin = 15
    reach = int(15 / (grid[1] - grid[0]))
    first, last = np.maximum(peak - reach, 0), np.minimum(peak + reach, grid.size - 1)
    blue = spectra[rows, first] > spectra[rows, last]
    start = np.where(blue, first, peak) + margin
    stop = np.where(blue, peak, last) - margin - 1
    change = np.abs(np.diff(spectra, axis=-1))
    index = np.arange(change.shape[-1])
    window = (index >= start[:, np.newaxis]) & (index <= stop[:, np.newaxis])
    flattest = np.argmin(np.where(window, change, np.inf), axis=-1) + 1
    found = stop >= start
    return np.column_stack([np.where(found, grid[flattest], np.nan), np.where(found, spectra[rows, flattest], np.nan)])


def _peaks(grid, spectra, maxima, centre):
    """
    Blue and red emission peaks (``mg_peaks_single`` in ``iris_get_mg_features.pro``).

    Only the maxima within 50 km/s of the line centre (0 km/s when it is unknown) count,
    and of more than four, the inner four.
    """
    rows = np.arange(len(spectra))
    centre = np.nan_to_num(centre)
    near = maxima & (np.abs(grid - centre[:, np.newaxis]) < 50)
    count = np.count_nonzero(near, axis=-1)
    rank = np.cumsum(near, axis=-1) - 1 - np.where(count > 4, count // 2 - 2, 0)[:, np.newaxis]
    peak = np.stack([np.argmax(near & (rank == slot), axis=-1) for slot in range(4)], axis=-1)
    x, y = grid[peak], spectra[rows[:, np.newaxis], peak]
    count = np.minimum(count, 4)
    blue, red = np.full(rows.shape, -1), np.full(rows.shape, -1)

    def assign(where, blue_slot, red_slot):
        blue[where] = np.broadcast_to(blue_slot, rows.shape)[where]
        red[where] = np.broadcast_to(red_slot, rows.shape)[where]

    one = count == 1
    assign(one & (x[:, 0] > centre), -1, 0)
    assign(one & (x[:, 0] <= centre), 0, -1)
    assign(count == 2, 0, 1)
    three = count == 3
    first_gap = three & (x[:, 0] < centre) & (centre < x[:, 1])
    second_gap = three & ~first_gap & (x[:, 1] < centre) & (centre < x[:, 2])
    weakest = np.argmin(y[:, :3], axis=-1)
    assign(first_gap, 0, 1 + (y[:, 2] > y[:, 1]))
    assign(second_gap, (y[:, 1] > y[:, 0]).astype(int), 2)
    assign(three & ~first_gap & ~second_gap, (weakest == 0).astype(int), 2 - (weakest == 2))
    four = count == 4
    outer = (
        four
        & (x[:, 3] - x[:, 0] < 40)
        & (x[:, 2] - x[:, 1] > 13)
        & (y[:, 0] > 1.06 * y[:, 1])
        & (y[:, 3] > 1.06 * y[:, 2])
    )
    inner = four & ~outer & (((x[:, 1] < centre) & (centre < x[:, 2])) | (centre > x[:, 3]) | (centre < x[:, 0]))
    blueward = four & ~outer & ~inner & (centre < x[:, 1])
    redward = four & ~outer & ~inner & ~blueward & (centre < x[:, 3])
    assign(outer, 0, 3)
    assign(inner, 1, 2)
    assign(blueward, 0, 1 + np.argmax(y[:, 1:], axis=-1))
    assign(redward, np.argmax(y[:, :3], axis=-1), 3)
    return _vertex(grid, spectra, peak, blue), _vertex(grid, spectra, peak, red)


def _vertex(grid, spectra, peak, slot):
    """
    Vertex of the parabola through the chosen grid maximum and its neighbours, as
    (velocity, intensity).
    """
    rows = np.arange(len(spectra))
    index = peak[rows, np.maximum(slot, 0)]
    left, right = np.maximum(index - 1, 0), np.minimum(index + 1, grid.size - 1)
    before, top, after = spectra[rows, left], spectra[rows, index], spectra[rows, right]
    curvature = before - 2 * top + after
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where((left < index) & (index < right) & (curvature < 0), 0.5 * (before - after) / curvature, 0)
    found = slot >= 0
    return np.column_stack(
        [
            np.where(found, grid[index] + shift * (grid[1] - grid[0]), np.nan),
            np.where(found, top - 0.25 * (before - after) * shift, np.nan),
        ]
    )


def _spline(x, y, t, *, tension):
    """
    IDL's ``SPLINE``: the spline under tension (Cline 1974) through ``y`` (along its
    first axis) at ``x``, with end slopes from 3-point formulas, evaluated at ``t``.

    Tension 0 gives a cubic spline.
    """
    sigma = max(tension, 1e-3) * (x.size - 1) / (x[-1] - x[0])
    shape = (-1,) + (1,) * (y.ndim - 1)
    h = np.diff(x)
    slope = np.diff(y, axis=0) / h.reshape(shape)
    sinh = np.sinh(sigma * h)
    off_diagonal = (1 / h - sigma / sinh) / sigma**2
    diagonal = (sigma * np.cosh(sigma * h) / sinh - 1 / h) / sigma**2
    first = -(2 * h[0] + h[1]) / (h[0] + h[1]) / h[0] * y[0] + (h[0] + h[1]) / h[0] / h[1] * y[1]
    first -= h[0] / (h[0] + h[1]) / h[1] * y[2]
    last = h[-1] / (h[-1] + h[-2]) / h[-2] * y[-3] - (h[-1] + h[-2]) / h[-1] / h[-2] * y[-2]
    last += (2 * h[-1] + h[-2]) / (h[-1] + h[-2]) / h[-1] * y[-1]
    bands = np.zeros((3, x.size))
    bands[0, 1:] = bands[2, :-1] = off_diagonal
    bands[1] = np.concatenate([diagonal[:1], diagonal[:-1] + diagonal[1:], diagonal[-1:]])
    moments = solve_banded(
        (1, 1), bands, np.concatenate([slope[:1] - first, np.diff(slope, axis=0), last - slope[-1:]])
    )
    i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, x.size - 2)
    left, right, width = (t - x[i]).reshape(shape), (x[i + 1] - t).reshape(shape), h[i].reshape(shape)
    return (
        (moments[i] * np.sinh(sigma * right) + moments[i + 1] * np.sinh(sigma * left))
        / (sigma**2 * sinh[i].reshape(shape))
        + (y[i] - moments[i] / sigma**2) * right / width
        + (y[i + 1] - moments[i + 1] / sigma**2) * left / width
    )
