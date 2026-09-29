"""
Measurement of the orbital drift of the IRIS spectrograph wavelength scale.

A port of the SolarSoft routine ``iris_prep_wavecorr_l2.pro``.
"""

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares

import astropy.units as u
from astropy.table import QTable, vstack

__all__ = ["calculate_wavelength_drift"]

# Reference lines: rest wavelength and fitted range in Å, the minimum mean intensity in DN of
# the spectrum averaged along the slit, and the line's sign (-1 absorption, 1 emission).
_LINES = {
    "Ni I": (2799.474, (2799.3, 2799.8), 5, -1),
    "Mn I": (2801.902, (2801.6, 2802.4), 5, -1),
    "Fe I": (2805.346, (2805.1, 2805.7), 5, -1),
    "O I": (1355.60, (1355.4, 1355.9), 0.5, 1),
    "Fe II": (1392.82, (1392.6, 1393.1), 0.5, 1),
}
# The line each drift curve is fitted to, and the distance from the median, in Å, beyond which a shift is an outlier.
_DRIFTS = {"nuv": ("Ni I", 0.08), "fuv": ("O I", 0.05)}
_ORBIT = 5856  # IRIS orbital period in seconds
_SMOOTHING = 300  # running mean width in seconds


def calculate_wavelength_drift(raster):
    """
    Measure the drift of the NUV and FUV wavelength scales through an observation.

    The orbital motion of IRIS shifts its spectra slightly through each orbit. For every
    exposure, the spectrum averaged along the slit is fitted with a Gaussian on a constant or
    linear background around five reference lines: the Ni I 2799.474, Mn I 2801.902 and
    Fe I 2805.346 Å absorption lines in the NUV and the O I 1355.60 and Fe II 1392.82 Å emission
    lines in the FUV. A line's shift is its rest wavelength minus the fitted centre.

    The NUV and FUV drifts are then fitted through the Ni I and O I shifts: outliers are
    dropped, a 5-minute running mean removes oscillations, and a sine with the orbital period
    plus a polynomial with one order per orbit covered, up to third order, is fitted by least
    squares.

    Parameters
    ----------
    raster : `~irispy.spectrograph.RasterCollection`
        Level 2 spectra read with ``memmap=False``. Only the windows that contain the reference
        lines are used; a line in no window gives no shifts.

    Returns
    -------
    `~astropy.table.QTable`
        One row per exposure, in time order: the exposure start ``time``, the ``raster`` and
        ``step`` it belongs to, one column per reference line with its shift (NaN where the
        line is too faint or the fit is rejected), and the fitted drifts ``"nuv"`` and
        ``"fuv"``. Adding a drift to the wavelengths of its detector corrects them.

    Notes
    -----
    This is a port of ``iris_prep_wavecorr_l2.pro`` with these deliberate differences:

    * The drift is fitted by linear least squares, which is exact because the orbital period
      is fixed. IDL fits it with MPFIT from a fixed starting guess and keeps exposures without
      a shift in the fit; MPFIT then fails silently and returns the starting guess, as it does
      on 4000005156, whose IDL drift is 0.1 + 0.01 sin(2πt/5856 s + 1) Å. Such exposures are
      left out here, and a warning is given if too few shifts remain for a fit.
    * Outliers are shifts more than 0.08 Å (NUV) or 0.05 Å (FUV) from the median on either
      side; IDL only drops those above the median.
    * The Gaussians are fitted to convergence with `scipy.optimize.least_squares`. IDL's
      ``CURVEFIT`` stops earlier: on 3824262996 its Ni I fits leave about four times the
      chi-square, and its Ni I shifts differ by 1.3 ± 0.3 mÅ.
    * irispy flips rasters that step backwards together with their times; IDL pairs flipped
      data with unflipped times.
    * IDL's ``/nosmooth`` and ``/nosine`` options are not ported: the line columns hold the
      unsmoothed shifts.

    References
    ----------
    * `iris_prep_wavecorr_l2.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/iris_prep_wavecorr_l2.pro>`__
    """
    windows = {name: _window(raster, rest) for name, (rest, *_) in _LINES.items()}
    found = [window for window in windows.values() if window is not None]
    if not found:
        msg = f"The raster has no window with any of the reference lines {list(_LINES)}"
        raise ValueError(msg)
    table = vstack(
        [
            QTable(
                {
                    "time": cube.meta["auxiliary times"],
                    "raster": np.full(len(cube.data), index),
                    "step": np.arange(len(cube.data)),
                }
            )
            for index, cube in enumerate(found[0].data)
        ]
    )
    for name, line in _LINES.items():
        window = windows[name]
        shifts = np.concatenate([_line_shifts(cube, *line) for cube in window.data]) if window else np.nan
        table[name] = np.broadcast_to(shifts, len(table)) * u.AA
    table.sort("time")
    seconds = (table["time"] - table["time"][0]).to_value(u.s)
    for drift, (name, outlier) in _DRIFTS.items():
        table[drift] = _fit_drift(seconds, table[name].to_value(u.AA), outlier, drift) * u.AA
    return table


def _window(raster, wavelength):
    """
    The first window of the raster covering ``wavelength``, or None.
    """
    for window in raster.values():
        low, high = window.data[0].meta.spectral_range.to_value(u.AA)
        if low <= wavelength <= high:
            return window
    return None


def _line_shifts(cube, rest, fit_range, min_intensity, sign):
    """
    The shift of one reference line in each exposure of a raster, in Å.
    """
    wavelength = cube.axis_world_coords(cube.wavelength_axis)[0].to_value(u.AA)
    bins = np.flatnonzero((wavelength >= fit_range[0]) & (wavelength <= fit_range[1]))
    if bins.size < 5:  # as IDL, fit at least 5 bins
        bins = np.arange(bins[0], min(bins[0] + 5, wavelength.size))
    # The mean along the slit, with bad pixels and negative values counted as zero, as in IDL
    profiles = np.nan_to_num(np.clip(cube.data[..., bins], 0, None)).mean(axis=1)
    offset = wavelength[bins] - rest
    shifts = np.full(len(profiles), np.nan)
    for step in np.flatnonzero(profiles.mean(axis=1) > min_intensity):
        amplitude, centre = _fit_gaussian(offset, profiles[step], background_order=int(bins.size >= 7))
        if amplitude * sign > 0 and abs(centre) <= (fit_range[1] - fit_range[0]) / 4:
            shifts[step] = -centre
    return shifts


def _fit_gaussian(x, y, background_order):
    """
    The amplitude and centre of a Gaussian on a polynomial background fitted to ``y``.
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
        centre = p[0] * gaussian * z / p[2]
        return np.column_stack([gaussian, centre, centre * z, *powers])

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
        finite = np.isfinite(smoothed)
        with np.errstate(invalid="ignore"):
            smoothed = uniform_filter1d(np.where(finite, smoothed, 0), width | 1, mode="nearest") / uniform_filter1d(
                finite.astype(float), width | 1, mode="nearest"
            )
    phase = 2 * np.pi * seconds / _ORBIT
    order = min(int(np.ptp(seconds) // _ORBIT), 3)
    design = np.column_stack([np.sin(phase), np.cos(phase), *((seconds / _ORBIT) ** np.arange(order + 1)[:, None])])
    fitted = ~outliers & np.isfinite(smoothed)
    if np.count_nonzero(fitted) < design.shape[1]:
        warnings.warn(f"Too few shifts to fit the {name.upper()} drift", UserWarning, stacklevel=3)
        return np.full(len(seconds), np.nan)
    coefficients = np.linalg.lstsq(design[fitted], smoothed[fitted], rcond=None)[0]
    return design @ coefficients
