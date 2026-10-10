"""
Mg II h and k line feature utilities for IRIS spectrogram cubes.
"""

import warnings

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d

import astropy.units as u

from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils._spectral import check_scaled, make_map_cube, make_spatial_template
from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED

__all__ = ["calculate_mg_features"]

_REST_WAVELENGTH = {"k": 279.63509493 * u.nm, "h": 280.35297192 * u.nm}  # vacuum
# Inner (minima, maxima) counts where the center is the middle minimum, or the lowest between the two highest maxima.
_MIDDLE_MINIMUM = [(1, 2), (3, 1), (3, 2), (3, 4), (5, 4), (7, 6)]
_BETWEEN_MAXIMA = [(2, 2), (2, 3), (3, 3), (4, 2), (4, 3), (4, 4)]


def calculate_mg_features(cube, *, velocity_range=(-40, 40) * u.km / u.s, lines=("k", "h")):
    """
    Find the line centers and emission peaks of the Mg II h and k lines.

    As in :cite:t:`pereira2013`, each spectrum is interpolated to a fine velocity grid, its local
    extrema are classified, and the line center is refined by a parabola fitted to its minimum.
    Line centers that jump along the slit are redone from their neighbors.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        One raster of a Mg II window, read with ``memmap=False``, with axes (step, slit, wavelength).
    velocity_range : `~astropy.units.Quantity`, optional
        Doppler velocities from each line's rest wavelength within which to search (km/s if unitless).
    lines : `tuple` of `str`, optional
        The lines to measure, ``"k"`` (279.635 nm) and/or ``"h"`` (280.353 nm, both in vacuum).

    Returns
    -------
    `~irispy.spectrograph.RasterCollection`
        (step, slit) maps keyed ``"{feature}_velocity"`` (km/s) and ``"{feature}_intensity"`` (the
        unit of ``cube``) for the blue peak, line center and red peak of each measured line:
        ``"k2v"``, ``"k3"``, ``"k2r"`` and ``"h2v"``, ``"h3"``, ``"h2r"``. Features that are not
        found are NaN and masked. ``"{line}_saturated"`` is `True` where an unmasked sample searched
        over (``velocity_range`` widened by 3 km/s) is +Inf, as the readers set the samples clipped at
        the level 2 ceiling; such a line has no features either.

    Notes
    -----
    As in IDL, a line that the window does not cover over ``velocity_range`` is skipped with a
    warning; `ValueError` is raised if the window covers none of ``lines``.

    The line centers match IDL's for almost every spectrum. Deliberate differences from IDL:

    * The peaks are refined by the parabola through their highest grid point and its neighbors.
      IDL evaluates its peak spline at decreasing velocities, which ``SPLINE`` does not support:
      it extrapolates the last interval, moving the peaks by about 0.1 km/s, rarely more than 0.7 km/s.
    * Spectra with missing data (non-finite, fill or masked samples) in the velocity range give no
      features; IDL interpolates through the -200 fill values and uses the line centers it finds
      there in its guesses along the slit.
      Beyond the first and last good line centers, the guess is held at their values instead.
    * The line centers that are redone start from a guess spline along the slit through the others.
      For unevenly spaced knots, as when slit positions are left out, IDL's ``SPLINE`` reuses the
      first interval's diagonal in the last row of its system, so the guesses near and beyond the
      last good slit position can differ.
    * Not reproduced: IDL's ``/onlyk`` uses 279.644 nm for k, its reversed (V34) rasters are
      descaled twice, its jump search along the slit takes the last line center as 0 km/s when
      none is missing, and its minimum near a guessed line center is misplaced when the guess is
      less than 15 grid points (7 in some cases) from the grid's blue end.
    * IDL's ``wave_comp`` wavelength offset has no keyword; shift the cube's wavelength axis instead.

    References
    ----------
    * :cite:t:`pereira2013`
    * :cite:t:`cline1974`
    * `iris_get_mg_features.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/uio/utils/iris_get_mg_features.pro>`__
    """
    if not isinstance(cube, SpectrogramCube):
        msg = f"cube must be a SpectrogramCube, not a {type(cube).__name__}; index it, e.g. raster['Mg II k 2796'][0]"
        raise TypeError(msg)
    if cube.data.ndim != 3 or cube.wavelength_axis != 2:
        msg = "cube must have axes (step, slit, wavelength); slice it with ranges, e.g. cube[:, 300:301]"
        raise ValueError(msg)
    check_scaled(cube)
    velocity_range = u.Quantity(velocity_range, u.km / u.s)
    if velocity_range.shape != (2,) or not velocity_range[0] < velocity_range[1]:
        msg = f"velocity_range must be two increasing velocities, not {velocity_range}"
        raise ValueError(msg)
    if not lines or not set(lines) <= set(_REST_WAVELENGTH) or len(set(lines)) < len(lines):
        msg = f"lines must be 'k', 'h' or both, not {lines!r}"
        raise ValueError(msg)
    low, high = velocity_range.value
    wavelength = cube.axis_world_coords(cube.wavelength_axis)[0]
    if not np.all(np.diff(wavelength) > 0):
        msg = "The wavelengths must increase along the cube, as in level 2 data"
        raise ValueError(msg)
    template = make_spatial_template(cube, cube.wavelength_axis)
    maps, skipped = [], []
    for line in lines:
        velocity = u.Quantity(wavelength).to_value(u.km / u.s, equivalencies=u.doppler_optical(_REST_WAVELENGTH[line]))
        if not velocity[0] <= low < high <= velocity[-1]:
            skipped.append(line)
            continue
        inside = (velocity >= low - 3) & (velocity <= high + 3)
        if np.count_nonzero(inside) < 3:
            msg = f"Too few wavelength points of Mg II {line} between {low} and {high} km/s"
            raise ValueError(msg)
        window = np.asarray(cube.data[..., inside])
        mask = False if cube.mask is None else np.asarray(cube.mask, dtype=bool)
        masked = np.broadcast_to(mask, cube.data.shape)[..., inside]
        missing = ~np.isfinite(window) | np.isin(window, BAD_PIXEL_VALUES_SCALED) | masked
        # Missing, so no features; masked samples are not searched, so they do not saturate either
        saturated = np.any(np.isposinf(window) & ~masked, axis=-1)
        grid = np.linspace(velocity[inside][0], velocity[inside][-1], 300)
        features = np.full((*window.shape[:2], 3, 2), np.nan)  # blue peak, center, red peak
        for step, (data, bad) in enumerate(zip(window, missing, strict=True)):
            valid = ~bad.any(axis=-1)
            if valid.any():
                spectra = _spline(velocity[inside], data[valid].T, grid, tension=0).T
                features[step, valid] = _slit_features(grid, spectra, valid)
        maps += [
            (f"{line}{feature}_{kind}", make_map_cube(template, features[..., index, part], unit, mask_invalid=True))
            for index, feature in enumerate(("2v", "3", "2r"))
            for part, (kind, unit) in enumerate([("velocity", u.km / u.s), ("intensity", cube.unit)])
        ]
        maps.append((f"{line}_saturated", make_map_cube(template, saturated, u.dimensionless_unscaled)))
    msg = f"The spectral window does not cover Mg II {' or '.join(skipped)} from {low} to {high} km/s"
    if not maps:
        raise ValueError(msg)
    if skipped:
        warnings.warn(f"{msg}; skipping it", UserWarning, stacklevel=2)
    return RasterCollection(maps, aligned_axes=(0, 1))


def _slit_features(grid, spectra, valid):
    """
    Blue peak, line center and red peak, as (velocity, intensity), of a slit's spectra.
    """
    maxima, minima = _maxima(spectra), _maxima(-spectra)
    center = _centers(grid, spectra, maxima, minima, 0.0)
    positions = np.flatnonzero(valid)
    blended = np.count_nonzero(maxima & (grid > grid[0] + 20) & (grid < grid[-1] - 20), axis=-1) == 1
    for _ in range(2):
        # Redo centers that jump from the smoothed slit (missing ones count as 0 km/s) from a spline through the
        # rest. As in IDL, slits shorter than the 17-pixel kernel are not smoothed.
        slit = np.zeros(valid.size)
        slit[positions] = np.nan_to_num(center[:, 0])
        average = gaussian_filter1d(slit, 2, mode="nearest") if valid.size >= 17 else slit
        good = np.abs(center[:, 0] - average[positions]) <= 3
        if np.count_nonzero(good) >= 3:
            # Held at the end centers beyond them, where IDL has knots on the fill and the spline would extrapolate
            knots = positions[good]
            guess = _spline(knots, center[good, 0], np.clip(positions, knots[0], knots[-1]), tension=1)
            center[~good] = np.nan  # and stays so where the peak is blended
            redo = ~good & ~blended
            center[redo] = _center_vertex(grid, spectra[redo], guess[redo], 15)
    redo = np.isnan(center[:, 0])
    center[redo] = _centers(grid, spectra[redo], maxima[redo], minima[redo], 5.0, use_derivative=True)
    blue, red = _peaks(grid, spectra, maxima, center[:, 0])
    return np.stack([blue, center, red], axis=1)


def _maxima(spectra):
    """
    Mask of the local maxima of each spectrum (``lclxtrem.pro``).

    A maximum is dropped when it is within 10 grid points of a kept one of larger
    absolute value, so of two close minima, ``_maxima(-spectra)`` keeps the shallower.
    """
    turn = np.diff(np.sign(np.diff(spectra, axis=-1)), axis=-1)
    candidate = np.pad(turn < 0, ((0, 0), (1, 1)))
    flat = ~candidate.any(axis=-1)  # no turning point: the highest value is the maximum
    candidate[flat, np.argmax(spectra[flat], axis=-1)] = True
    order = np.argsort(np.where(candidate, -np.abs(spectra), np.inf), axis=-1)[:, : candidate.sum(axis=-1).max()]
    kept = np.take_along_axis(candidate, order, axis=-1)
    for rank in range(1, order.shape[1]):
        close = np.abs(order[:, :rank] - order[:, rank, np.newaxis]) <= 10
        kept[:, rank] &= ~(close & kept[:, :rank]).any(axis=-1)
    np.put_along_axis(candidate, order, kept, axis=-1)  # order holds every candidate
    return candidate


def _centers(grid, spectra, maxima, minima, guess, *, use_derivative=False):
    """
    Line centers (IDL ``mg_single``), as an (n, 2) array of (velocity, intensity).

    The numbers of extrema away from the grid's ends pick a guess, and the center is the
    vertex of a parabola fitted near it. A single blended peak gives NaN, or with
    ``use_derivative`` the flattest point on its higher side.
    """
    rows = np.arange(len(spectra))
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
    half_width = np.where(middle | deep, 7, 15)
    blended = (pair & ~deep) | (~middle & ~between & ~pair & (counts[:, 1] == 1))
    center = np.full((rows.size, 2), np.nan)
    center[~blended] = _center_vertex(grid, spectra[~blended], guess[~blended], half_width[~blended])
    if use_derivative:
        center[blended] = _flattest(grid, spectra[blended], np.argmax(maxima[blended], axis=-1))
    return center


def _center_vertex(grid, spectra, guess, half_width):
    """
    Vertex of a parabola fitted to 7 points around the lowest point near ``guess``.
    """
    spacing = grid[1] - grid[0]
    nearest = np.clip(np.rint((guess - grid[0]) / spacing), 0, grid.size - 1).astype(int)
    index = np.arange(grid.size)
    near = (index >= (nearest - half_width)[:, np.newaxis]) & (index < (nearest + half_width)[:, np.newaxis])
    lowest = np.argmin(np.where(near, spectra, np.inf), axis=-1)
    points = lowest[:, np.newaxis] + np.arange(-3, 4)
    used = (points >= 0) & (points < grid.size)
    points = np.clip(points, 0, grid.size - 1)
    x = grid[points] - grid[lowest, np.newaxis]
    powers = np.where(used[..., np.newaxis], x[..., np.newaxis] ** np.arange(3), 0)  # leaves out the clipped points
    normal = np.einsum("nki,nkj->nij", powers, powers)
    moments = np.einsum("nki,nk->ni", powers, np.take_along_axis(spectra, points, axis=-1))
    a = np.linalg.solve(normal, moments[..., np.newaxis])[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = -a[:, 1] / (2 * a[:, 2])
        value = a[:, 0] - a[:, 2] * vertex**2
    close = np.abs(vertex) <= 4 * spacing
    return np.where(close[:, np.newaxis], np.column_stack([grid[lowest] + vertex, value]), np.nan)


def _flattest(grid, spectra, peak):
    """
    Flattest point within 15 km/s of a single blended peak, on its higher side.
    """
    rows = np.arange(len(spectra))
    margin = 15  # grid points, as IDL's ``margin``, unlike the reach in km/s
    reach = int(15 / (grid[1] - grid[0]))
    first, last = np.maximum(peak - reach, 0), np.minimum(peak + reach, grid.size - 1)
    blue = spectra[rows, first] > spectra[rows, last]
    start = np.where(blue, first, peak) + margin
    stop = np.where(blue, peak, last) - margin - 1
    change = np.abs(np.diff(spectra, axis=-1))
    index = np.arange(change.shape[-1])
    searched = (index >= start[:, np.newaxis]) & (index <= stop[:, np.newaxis])
    flattest = np.argmin(np.where(searched, change, np.inf), axis=-1) + 1
    found = stop >= start
    return np.where(found[:, np.newaxis], np.column_stack([grid[flattest], spectra[rows, flattest]]), np.nan)


def _peaks(grid, spectra, maxima, center):
    """
    Blue and red emission peaks (IDL ``mg_peaks_single``).

    Only the maxima within 50 km/s of the line center (0 km/s when it is unknown) count,
    and of more than four, the inner four.
    """
    center = np.nan_to_num(center)
    near = maxima & (np.abs(grid - center[:, np.newaxis]) < 50)
    count = np.count_nonzero(near, axis=-1)
    rank = np.cumsum(near, axis=-1) - 1 - np.where(count > 4, count // 2 - 2, 0)[:, np.newaxis]
    peak = np.stack([np.argmax(near & (rank == slot), axis=-1) for slot in range(4)], axis=-1)
    y = np.take_along_axis(spectra, peak, axis=-1)
    (x0, x1, x2, x3), (y0, y1, y2, y3) = grid[peak].T, y.T
    three, four = count == 3, count >= 4
    weakest = np.argmin(y[:, :3], axis=-1)
    outer = (x3 - x0 < 40) & (x2 - x1 > 13) & (y0 > 1.06 * y1) & (y3 > 1.06 * y2)
    inner = ((x1 < center) & (center < x2)) | (center > x3) | (center < x0)
    cases = [  # (condition, blue slot, red slot), of which the first that holds counts
        ((count == 1) & (x0 > center), -1, 0),
        (count == 1, 0, -1),
        (count == 2, 0, 1),
        (three & (x0 < center) & (center < x1), 0, 1 + (y2 > y1)),
        (three & (x1 < center) & (center < x2), y1 > y0, 2),
        (three, weakest == 0, 2 - (weakest == 2)),
        (four & outer, 0, 3),
        (four & inner, 1, 2),
        (four & (center < x1), 0, 1 + np.argmax(y[:, 1:], axis=-1)),
        (four & (center < x3), np.argmax(y[:, :3], axis=-1), 3),
    ]
    condition, blue, red = zip(*cases, strict=True)
    blue, red = np.select(condition, blue, -1), np.select(condition, red, -1)
    return _peak_vertex(grid, spectra, peak, blue), _peak_vertex(grid, spectra, peak, red)


def _peak_vertex(grid, spectra, peak, slot):
    """
    Vertex (velocity, intensity) of the parabola through a peak and its neighbors.
    """
    rows = np.arange(len(spectra))
    index = peak[rows, np.maximum(slot, 0)]
    left, right = np.maximum(index - 1, 0), np.minimum(index + 1, grid.size - 1)
    before, top, after = spectra[rows, left], spectra[rows, index], spectra[rows, right]
    curvature = before - 2 * top + after
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where((left < index) & (index < right) & (curvature < 0), 0.5 * (before - after) / curvature, 0)
    vertex = np.column_stack([grid[index] + shift * (grid[1] - grid[0]), top - 0.25 * (before - after) * shift])
    return np.where((slot >= 0)[:, np.newaxis], vertex, np.nan)


def _spline(x, y, t, *, tension):
    """
    The spline under tension of :cite:t:`cline1974` through ``y`` (along its first axis)
    at ``x``, with end slopes from 3-point formulas, evaluated at ``t``.

    Tension 0 gives a cubic spline. IDL's ``SPLINE`` differs for unevenly spaced ``x``;
    see `calculate_mg_features`.
    """
    sigma = max(tension, 1e-3) * (x.size - 1) / (x[-1] - x[0])
    shape = (-1,) + (1,) * (y.ndim - 1)
    h = np.diff(x)
    slope = np.diff(y, axis=0) / h.reshape(shape)
    sinh = np.sinh(sigma * h)
    off_diagonal = (1 / h - sigma / sinh) / sigma**2
    diagonal = (sigma * np.cosh(sigma * h) / sinh - 1 / h) / sigma**2
    first, last = np.gradient(y, x, axis=0, edge_order=2)[[0, -1]]
    bands = np.zeros((3, x.size))
    bands[0, 1:] = bands[2, :-1] = off_diagonal
    bands[1] = np.concatenate([diagonal[:1], diagonal[:-1] + diagonal[1:], diagonal[-1:]])
    rhs = np.concatenate([slope[:1] - first, np.diff(slope, axis=0), last - slope[-1:]])
    moments = solve_banded((1, 1), bands, rhs)
    i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, x.size - 2)
    left, right, width = (t - x[i]).reshape(shape), (x[i + 1] - t).reshape(shape), h[i].reshape(shape)
    return (
        (moments[i] * np.sinh(sigma * right) + moments[i + 1] * np.sinh(sigma * left))
        / (sigma**2 * sinh[i].reshape(shape))
        + (y[i] - moments[i] / sigma**2) * right / width
        + (y[i + 1] - moments[i + 1] / sigma**2) * left / width
    )
