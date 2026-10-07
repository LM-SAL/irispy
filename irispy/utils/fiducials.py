"""
Find the fiducial marks on the IRIS slit.
"""

import numpy as np
from scipy import ndimage, signal

import astropy.units as u

from irispy.spectrograph import SpectrogramCubeSequence
from irispy.utils._spectral import check_scaled
from irispy.utils.constants import SLIT_WIDTH

__all__ = ["find_fiducials"]

# Width of the running median that gives the level of the slit profile around a mark: well wider than
# a mark, a gap of two pixels (De Pontieu et al. 2014, Section 4.3) blurred by the 0.33-0.4 arcsec resolution.
_ROWS = 21
# Distance between the marks, measured in ten Level 2 observations from 2013 to 2026: 539.1 unsummed rows
# of 0.16635 arcsec, which puts them at rows 238.0 and 777.1 of a window that starts at the first CCD row.
_SEPARATION = 89.7 * u.arcsec


def _cubes(cube_or_sequence):
    return cube_or_sequence.data if isinstance(cube_or_sequence, SpectrogramCubeSequence) else [cube_or_sequence]


def _plate_scale(cube):
    # The solid angle of a pixel is the slit width times the pixel scale along the slit
    return (cube.solid_angle / SLIT_WIDTH).to(u.arcsec) / u.pix


def _candidates(cube_or_sequence, min_depth):
    """
    Dark rows along the slit: their positions, depths and significance.
    """
    sums = counts = 0
    for cube in _cubes(cube_or_sequence):
        check_scaled(cube)
        mask = np.broadcast_to(False if cube.mask is None else np.asarray(cube.mask, dtype=bool), cube.data.shape)
        data = np.ma.masked_array(cube.data, mask=mask | ~np.isfinite(cube.data))
        # Level 2 arrays are (..., slit, wavelength)
        axes = tuple(axis for axis in range(data.ndim) if axis != cube.wavelength_axis - 1)
        sums = sums + data.sum(axis=axes, dtype=float).filled(0)
        counts = counts + data.count(axis=axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts
    valid = np.isfinite(profile)
    baseline = np.full(profile.shape, np.nan)
    baseline[valid] = ndimage.median_filter(profile[valid], size=_ROWS, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(baseline > 0, 1 - profile / baseline, np.nan)
    valid = np.isfinite(depth)
    scatter = np.full(depth.shape, np.nan)
    deviation = np.abs(depth[valid] - ndimage.median_filter(depth[valid], size=_ROWS, mode="nearest"))
    scatter[valid] = 1.4826 * ndimage.median_filter(deviation, size=5 * _ROWS, mode="nearest")
    peaks, _ = signal.find_peaks(np.nan_to_num(depth), height=min_depth, distance=_ROWS)
    peaks = peaks[(peaks > 0) & (peaks < depth.size - 1)]
    left, depths, right, scatter = depth[peaks - 1], depth[peaks], depth[peaks + 1], scatter[peaks]
    # A mark darkens the rows either side, stands out from the noise and blocks no more than all the light
    keep = (np.fmin(left, right) > 0) & (np.fmax(left, right) > 3 * scatter)
    keep &= (depths >= 5 * scatter) & (depths <= 1 + 3 * scatter)
    left, depths, right, scatter = left[keep], depths[keep], right[keep], scatter[keep]
    # The vertex of the parabola through the logarithms is the center of the Gaussian through the rows
    logs = np.log([left, depths, right])
    curvature = logs[0] - 2 * logs[1] + logs[2]
    shift = np.divide(logs[0] - logs[2], 2 * curvature, out=np.zeros_like(curvature), where=curvature < 0)
    positions = peaks[keep] + shift
    with np.errstate(divide="ignore", invalid="ignore"):
        return positions, depths, depths / scatter


def find_fiducials(cube_or_sequence, *, min_depth=0.5):
    """
    Find the fiducial marks along the slit of a spectral window.

    The fiducial marks are two short gaps in the slit, one in each half, that let no light into the
    spectrograph, so they show up as dark rows across the whole spectrum :cite:p:`depontieu2014`.
    The slit profile is the mean of the data over raster steps and wavelengths, and over the cubes
    of a sequence, leaving out masked and non-finite values. It is divided by its running median
    over 21 rows, and the marks are the dips in this ratio. Each dip is placed to a fraction of a
    pixel at the center of the Gaussian through its deepest row and the rows either side, as the
    IRIS team measures the marks (:cite:t:`depontieu2014`, Section 7.6.1).

    Parameters
    ----------
    cube_or_sequence : `~irispy.spectrograph.SpectrogramCube` or `~irispy.spectrograph.SpectrogramCubeSequence`
        One spectral window, read with ``memmap=False``. The slit axis is the axis before the
        wavelength axis, as in the cubes from `irispy.io.read_files`.
    min_depth : `float`, optional
        The smallest fraction of the light a dip must block to count as a mark. In the Level 2
        observations we checked, unsummed or summed by 2 along the slit (``SUMSPAT``), the marks
        block 0.6 to 0.93 of the light and other dark rows in the NUV windows block 0.3 to 0.4, so
        the default of 0.5 separates them. Lower it for data summed by 4 or more, where a mark
        fills less of a pixel.

    Returns
    -------
    positions : `numpy.ndarray`
        The row of each mark along the slit as a fractional array index, in increasing order:
        two, one, or an empty array if no mark is found.
    depths : `numpy.ndarray`
        The fraction of the light each mark blocks at its deepest row, a measure of confidence:
        about 0.9 in unsummed NUV data, less where the PSF and summing spread the mark over more
        rows. It can exceed 1 where the FUV background subtraction takes the mark below zero.

    Notes
    -----
    A dip counts as a mark if

    * it blocks at least ``min_depth`` of the light, and at least 5 times the scatter of the
      ratio around it, the scaled median absolute deviation from its running median over 105 rows;
    * it blocks no more than all the light, within 3 times that scatter;
    * the rows on either side are darker than the profile around them, one of them by more than
      3 times the scatter, since the PSF spreads a mark over more than one row.

    The marks are 89.7″ apart, so when there are several dips the pair at that distance is taken,
    or else the most significant dip.

    The marks are often not visible in the FUV windows, and a window that covers less than about
    half the slit holds at most one of them; see :ref:`irispy-tutorial-data-idiosyncrasies`. Some
    FUV windows show a dark row that is not a mark about 61 rows below the lower mark; when in
    doubt, compare with the NUV windows, where the marks are clear.
    """
    positions, depths, significance = _candidates(cube_or_sequence, min_depth)
    if positions.size < 2:
        return positions, depths
    separation = (_SEPARATION / _plate_scale(_cubes(cube_or_sequence)[0])).to_value(u.pix)
    first, second = np.nonzero(np.abs(positions - positions[:, np.newaxis] - separation) < 2)
    if first.size:
        best = np.argmax(significance[first] + significance[second])
        keep = [first[best], second[best]]
    else:
        keep = [np.argmax(significance)]
    return positions[keep], depths[keep]
