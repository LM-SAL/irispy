"""
Utilities for measuring the orbital drift of the IRIS spectrograph wavelength scale.
"""

import warnings

import numpy as np
from scipy.optimize import least_squares

import astropy.units as u
from astropy.convolution import Box1DKernel, convolve
from astropy.table import QTable, vstack
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ["calculate_wavelength_drift"]

# Name: rest wavelength (Å), fit range (Å), minimum slit-mean intensity (DN), sign (-1 absorption, 1 emission)
_LINES = {
    "Ni I": (2799.474, (2799.3, 2799.8), 5, -1),
    "Mn I": (2801.902, (2801.6, 2802.4), 5, -1),
    "Fe I": (2805.346, (2805.1, 2805.7), 5, -1),
    "O I": (1355.60, (1355.4, 1355.9), 0.5, 1),
    "Fe II": (1392.82, (1392.6, 1393.1), 0.5, 1),
}
# Drift: the line it is fitted to, and the outlier distance from the median (Å)
_DRIFTS = {"nuv": ("Ni I", 0.08), "fuv": ("O I", 0.05)}
_ORBIT = 5856  # IRIS orbital period in seconds
_SMOOTHING = 300  # running mean width in seconds


def calculate_wavelength_drift(raster):
    """
    Measure the orbital drift of the NUV and FUV wavelength scales.

    For every exposure, the spectrum averaged along the slit is fitted with a Gaussian on a
    constant or linear background around the Ni I 2799.474, Mn I 2801.902 and Fe I 2805.346 Å
    absorption lines (NUV) and the O I 1355.60 and Fe II 1392.82 Å emission lines (FUV). A
    line's shift is its rest wavelength minus the fitted center.

    The NUV and FUV drifts are fitted to the Ni I and O I shifts: shifts more than 0.08 Å (NUV)
    or 0.05 Å (FUV) from the median are dropped, a 5-minute running mean removes oscillations,
    and a sine with the 5856 s orbital period plus a polynomial of order min(whole orbits
    spanned by the surviving measurements, 3) is fitted. Measured coverage shorter than a
    quarter orbit gets a constant drift, the mean of the smoothed shifts, with a warning.

    Parameters
    ----------
    raster : `~irispy.spectrograph.RasterCollection`
        Level 2 spectra. They may be read with ``memmap=True``: only the fit ranges are read,
        and scaled to DN.

    Returns
    -------
    `~astropy.table.QTable`
        One row per exposure, sorted by exposure start ``time``, with its ``raster`` and ``step``
        indices. Each line's column (``"Ni I"`` ... ``"Fe II"``) holds its unsmoothed shift in Å,
        NaN if the line is in no window, too faint, has too few pixels, or its fit is rejected.
        ``"nuv"`` and ``"fuv"`` hold the drifts in Å, NaN if their line is in no window or, with a
        warning, if no more shifts remain than the fit has parameters; adding a drift to its
        detector's wavelengths corrects them.

    Raises
    ------
    ValueError
        If no window contains any of the reference lines.

    Notes
    -----
    This ports `iris_prep_wavecorr_l2.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/iris_prep_wavecorr_l2.pro>`__
    with these differences:

    * The drift is fitted by linear least squares, which needs no starting guess, and exposures
      still without a shift after the running mean are left out. IDL's MPFIT keeps them and then
      silently returns its starting guess, as on OBSID 4000005156, whose IDL drift is
      0.1 + 0.01 sin(2πt/5856 s + 1) Å.
    * IDL fits the sine and polynomial however short the observation.
    * IDL drops only outliers above the median, and its ``MEDIAN`` of an even number of shifts
      is the upper middle value, where `numpy.median` averages the two.
    * The Gaussians are fitted to convergence. IDL's ``CURVEFIT`` stops early: on OBSID
      3824262996 its Ni I shifts differ by 1.3 ± 0.3 mÅ and its NUV drift by up to 1.7 mÅ.
      Fits to faint profiles, and to the weak Fe II line, can differ by tens of mÅ.
    * A fit range of fewer than 5 pixels is extended to 5 from its first pixel. Past the end of
      the window IDL repeats the edge pixel; here the line is fitted to the pixels there are, or
      is NaN if they are fewer than the Gaussian's 4 parameters.
    * Masked pixels count as zero in the slit average, as IDL counts bad pixels.
    * On rasters rolled by 90° or 270°, IDL divides the slit sum by the number of raster steps
      instead of the number of slit pixels.
    * irispy flips rasters that step backwards together with their times; IDL pairs flipped
      data with unflipped times.
    * IDL's ``/nosmooth`` and ``/nosine`` options are not ported.
    """
    windows = {name: _window(raster, rest) for name, (rest, *_) in _LINES.items()}
    used = [key for key in dict.fromkeys(windows.values()) if key is not None]
    if not used:
        msg = f"The raster has no window with any of the reference lines {list(_LINES)}"
        raise ValueError(msg)
    # The cubes of each window used, with their wavelengths
    cubes = {key: [(cube, _wavelengths(cube)) for cube in raster[key].data] for key in used}
    table = vstack(
        [
            QTable(
                {
                    "time": cube.meta["auxiliary times"],
                    "raster": np.full(len(cube.data), index),
                    "step": np.arange(len(cube.data)),
                }
            )
            for index, cube in enumerate(raster[used[0]].data)
        ]
    )
    for name, key in windows.items():
        shifts = np.concatenate([_line_shifts(*pair, *_LINES[name]) for pair in cubes[key]]) if key in cubes else np.nan
        table[name] = shifts * u.AA
    table.sort("time")
    seconds = (table["time"] - table["time"][0]).to_value(u.s)
    for drift, (name, outlier) in _DRIFTS.items():
        fitted = _fit_drift(seconds, table[name].to_value(u.AA), outlier, drift) if windows[name] else np.nan
        table[drift] = fitted * u.AA
    return table


def _wavelengths(cube):
    """
    The wavelengths of ``cube`` in Å.
    """
    return cube.axis_world_coords_values(cube.wavelength_axis)[0].to_value(u.AA)


def _window(raster, wavelength):
    """
    The key of the first window whose wavelengths cover ``wavelength``, or None.
    """
    for key, window in raster.items():
        grid = _wavelengths(window.data[0])
        if grid.min() <= wavelength <= grid.max():
            return key
    return None


def _line_shifts(cube, wavelength, rest, fit_range, min_intensity, sign):
    """
    The shift of one reference line in each exposure of a raster, in Å.
    """
    shifts = np.full(len(cube.data), np.nan)
    bins = np.flatnonzero((wavelength >= fit_range[0]) & (wavelength <= fit_range[1]))
    if 0 < bins.size < 5:  # as IDL, fit at least 5 bins
        bins = np.arange(bins[0], min(bins[0] + 5, wavelength.size))
    if bins.size < 4:  # fewer bins than Gaussian parameters
        return shifts
    # The fit bins in DN, which memmap=True leaves as the FITS integers, averaged along the slit
    # with bad, masked and negative pixels as zero, as in IDL
    data = cube.data[..., bins]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) * cube.meta["BSCALE"] + cube.meta["BZERO"]
    if cube.mask is not None:
        data[np.broadcast_to(cube.mask, cube.data.shape)[..., bins]] = 0
    profiles = np.clip(np.nan_to_num(data, copy=False), 0, None, out=data).mean(axis=1, dtype=float)
    offset = wavelength[bins] - rest
    for step in np.flatnonzero(profiles.mean(axis=1) > min_intensity):
        amplitude, center = _fit_gaussian(offset, profiles[step], background_order=int(bins.size >= 7))
        if amplitude * sign > 0 and abs(center) <= (fit_range[1] - fit_range[0]) / 4:
            shifts[step] = -center
    return shifts


def _fit_gaussian(x, y, background_order):
    """
    The amplitude and center of a Gaussian on a polynomial background fitted to ``y``.
    """
    background = np.polynomial.polynomial.polyfit(x, y, background_order)
    line = y - np.polynomial.polynomial.polyval(x, background)
    peak = np.argmax(np.abs(line))
    fwhm = max(np.count_nonzero(np.abs(line) > np.abs(line[peak]) / 2), 1) * np.ptp(x) / (x.size - 1)

    powers = x ** np.arange(background_order + 1)[:, np.newaxis]

    def residuals(p):
        return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + p[3:] @ powers - y

    def jacobian(p):
        z = (x - p[1]) / p[2]
        gaussian = np.exp(-0.5 * z**2)
        center = p[0] * gaussian * z / p[2]
        return np.column_stack([gaussian, center, center * z, *powers])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        fit = least_squares(residuals, [line[peak], x[peak], fwhm / 2.355, *background], jacobian, method="lm")
    return fit.x[:2] if fit.success else (np.nan, np.nan)


def _fit_drift(seconds, shifts, outlier, name):
    """
    Fit the orbital drift through the shifts of one line, evaluated at every exposure.
    """
    finite = np.isfinite(shifts)
    outliers = np.abs(shifts - (np.median(shifts[finite]) if finite.any() else 0)) > outlier
    smoothed = np.where(outliers, np.nan, shifts)
    # IDL: SMOOTH(shifts, width, /EDGE_TRUNCATE, /NAN), whose width is odd and covers 5 minutes
    width = int(_SMOOTHING / np.gradient(seconds, edge_order=2).mean()) if len(seconds) > 2 else 1
    if width < np.count_nonzero(~outliers):
        with warnings.catch_warnings(action="ignore", category=AstropyUserWarning):  # windows with no shift
            smoothed = convolve(smoothed, Box1DKernel(width | 1), boundary="extend")
    measured = finite & ~outliers
    coverage = np.ptp(seconds[measured]) if measured.any() else 0
    if coverage < _ORBIT / 4:
        design = np.ones((len(seconds), 1))
    else:
        phase = 2 * np.pi * seconds / _ORBIT
        order = min(int(coverage // _ORBIT), 3)
        design = np.column_stack([np.sin(phase), np.cos(phase), *((seconds / _ORBIT) ** np.arange(order + 1)[:, None])])
    if np.count_nonzero(measured) <= design.shape[1]:
        msg = f"Too few shifts to fit the {name.upper()} drift"
        warnings.warn(msg, UserWarning, stacklevel=3)
        return np.full(len(seconds), np.nan)
    if design.shape[1] == 1:
        msg = f"The measured coverage is too short for an orbital fit, so the {name.upper()} drift is constant"
        warnings.warn(msg, UserWarning, stacklevel=3)
    fitted = ~outliers & np.isfinite(smoothed)
    coefficients = np.linalg.lstsq(design[fitted], smoothed[fitted])[0]
    return design @ coefficients
