"""
Find the fiducial marks on the IRIS slit.
"""

import numpy as np
from scipy import ndimage, signal

import astropy.units as u

from irispy.utils._spectral import check_scaled
from irispy.utils.constants import SLIT_WIDTH
from irispy.utils.utils import _boolean_mask

__all__ = ["find_fiducials"]

# Width of the running median that gives the level of the slit profile around a mark: well wider than
# a mark, a gap of two pixels (De Pontieu et al. 2014, Section 4.3) blurred by the 0.33-0.4 arcsec resolution.
_ROWS = 21
# Distance between the marks, measured in ten Level 2 observations from 2013 to 2026: 539.1 unsummed rows
# of 0.16635 arcsec, which puts them at rows 238.0 and 777.1 of a window that starts at the first CCD row.
_SEPARATION = 89.7 * u.arcsec
# Those rows, which fit the Level 2 marks better than SolarSoft's nominal 238.5 and 777.5
# (CPX2_1P5 - 1 ∓ fidsep, with the 1-based CPX2_1P5 = 509 in iris_mk_pointdb.pro and fidsep = 269.5 in
# iris_prep_align_image.pro); the Level 2 marks lie within 1.01 rows of them.
_MARK_ROWS = np.array([238.0, 777.1])
# Some FUV windows show a dark row that is not a mark this far below the lower mark, measured in Level 2
# observations from 2014 to 2021; it can stand out from the noise more than the mark
_DARK_ROW = 10.1 * u.arcsec


def _plate_scale(cube):
    # The solid angle of a pixel is the slit width times the pixel scale along the slit
    return (cube.solid_angle / SLIT_WIDTH).to(u.arcsec) / u.pix


def _mark_rows(cube):
    """
    The rows where the Level 2 pipeline puts the marks, or `None` if the header does not
    describe the rows.
    """
    meta = cube.meta
    # The window's first and last CCD rows, in summed rows; a cutout or a slice along the slit keeps them
    start, end = meta[f"TSR{meta._iwin}"], meta[f"TER{meta._iwin}"]
    if cube.data.shape[cube.wavelength_axis - 1] != end - start + 1:
        return None
    # A summed row is centered (summing - 1) / 2 unsummed rows after its first, as in iris_prep_align_image.pro
    summing = meta.spatial_summing_factor
    return (_MARK_ROWS - (summing - 1) / 2) / summing - (start - 1)


def _candidates(cube, min_depth):
    """
    Dark rows along the slit: their positions, depths and significance.
    """
    check_scaled(cube)
    sums = counts = 0
    # A raster at a time, so the masked copies are of one raster, not the whole window
    rasters = cube if cube.data.ndim == 4 else [cube]
    for raster in rasters:
        data = np.ma.masked_array(raster.data, mask=_boolean_mask(raster) | ~np.isfinite(raster.data))
        # Level 2 arrays are (..., slit, wavelength)
        axes = tuple(axis for axis in range(data.ndim) if axis != raster.wavelength_axis - 1)
        sums = sums + data.sum(axis=axes, dtype=float).filled(0)
        counts = counts + data.count(axis=axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts
    valid = np.isfinite(profile)
    # The depth is the dip below the running median of the profile over _ROWS rows
    baseline = np.full(profile.shape, np.nan)
    baseline[valid] = ndimage.median_filter(profile[valid], size=_ROWS, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(baseline > 0, 1 - profile / baseline, np.nan)
    valid = np.isfinite(depth)
    # The scatter of the depth around a row: the scaled median, over 5 * _ROWS rows, of its absolute
    # deviation from its running median over _ROWS rows
    scatter = np.full(depth.shape, np.nan)
    deviation = np.abs(depth[valid] - ndimage.median_filter(depth[valid], size=_ROWS, mode="nearest"))
    scatter[valid] = 1.4826 * ndimage.median_filter(deviation, size=5 * _ROWS, mode="nearest")
    peaks, _ = signal.find_peaks(np.nan_to_num(depth), height=min_depth, distance=_ROWS)
    peaks = peaks[(peaks > 0) & (peaks < depth.size - 1)]
    left, depths, right, scatter = depth[peaks - 1], depth[peaks], depth[peaks + 1], scatter[peaks]
    # A mark darkens the rows either side, one of them by more than 3 times the scatter, stands out
    # from the noise by 5 times the scatter and blocks no more than all the light, within 3 times it
    keep = (np.fmin(left, right) > 0) & (np.fmax(left, right) > 3 * scatter)
    keep &= (depths >= 5 * scatter) & (depths <= 1 + 3 * scatter)
    left, depths, right, scatter = left[keep], depths[keep], right[keep], scatter[keep]
    # The vertex of the parabola through the logarithms is the center of the Gaussian through the
    # deepest row and the rows either side
    logs = np.log([left, depths, right])
    curvature = logs[0] - 2 * logs[1] + logs[2]
    shift = np.divide(logs[0] - logs[2], 2 * curvature, out=np.zeros_like(curvature), where=curvature < 0)
    positions = peaks[keep] + shift
    with np.errstate(divide="ignore", invalid="ignore"):
        return positions, depths, depths / scatter


def find_fiducials(cube, *, min_depth=0.5):
    """
    Find the fiducial marks along the slit of a spectral window.

    The fiducial marks are two short gaps in the slit, one in each half, that let no light into the
    spectrograph, so they show up as dark rows across the whole spectrum :cite:p:`depontieu2014`.
    The marks are the dips in the slit profile, the mean of the data over rasters, raster steps
    and wavelengths, leaving out masked and non-finite values.
    Each mark is placed to a fraction of a pixel, as the IRIS team measures the marks
    (:cite:t:`depontieu2014`, Section 7.6.1).

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
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
    A dip counts as a mark if it blocks at least ``min_depth`` of the light, stands out from the
    noise of the profile around it, and the rows on either side are darker too, since the PSF
    spreads a mark over more than one row.
    The marks are 89.7″ apart, so when there are several dips the pair at that distance is taken.
    Without a pair, the dip within 2 rows of where the Level 2 pipeline puts a mark is taken. In a
    cutout or a cube sliced along the slit, or when no dip is that close, the dip that stands out
    most from the noise is taken instead, except one 10.1″ below another dip: some FUV windows show
    a dark row that is not a mark that far below the lower mark.

    The marks are often not visible in the FUV windows, and a window that covers less than about
    half the slit holds at most one of them; see :ref:`irispy-tutorial-data-idiosyncrasies`. When
    in doubt, compare with the NUV windows, where the marks are clear.
    """
    positions, depths, significance = _candidates(cube, min_depth)
    if positions.size < 2:
        return positions, depths
    plate_scale = _plate_scale(cube)
    separation = (_SEPARATION / plate_scale).to_value(u.pix)
    first, second = np.nonzero(np.abs(positions - positions[:, np.newaxis] - separation) < 2)
    if first.size:
        best = np.argmax(significance[first] + significance[second])
        keep = [first[best], second[best]]
    else:
        # Where the header puts a mark beats significance: noise and the edges of the window make
        # dips that stand out more than a faint mark
        rows = _mark_rows(cube)
        distance = np.inf if rows is None else np.min(np.abs(positions[:, np.newaxis] - rows), axis=1)
        if np.min(distance) < 2:
            keep = [np.argmin(distance)]
        else:
            offset = (_DARK_ROW / plate_scale).to_value(u.pix)
            below = np.any(np.abs(positions - positions[:, np.newaxis] - offset) < 2, axis=1)
            keep = [np.argmax(np.where(below, -np.inf, significance))]
    return positions[keep], depths[keep]
