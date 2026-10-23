"""
Detection of compact bright events, such as UV bursts, in IRIS spectra and slit-jaw
images.
"""

import warnings
from copy import deepcopy

import numpy as np
from scipy import ndimage

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time

from irispy.spectrograph import MosaicCube
from irispy.utils._spectral import _to_unit, check_scaled, check_velocity_range, make_map_cube, make_spatial_template
from irispy.utils.constants import DN_UNIT
from irispy.utils.response import get_interpolated_effective_area, get_latest_response
from irispy.utils.utils import _boolean_mask, _saturated_samples

__all__ = ["find_bright_image_events", "find_bright_spectral_events", "find_si_iv_bursts"]

_SI_IV = 1402.77 * u.AA
# The 500 DN/s default threshold was set on this observation, summed by 2 in wavelength.
_REFERENCE_TIME = Time("2013-10-22T21:00")
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@u.quantity_input
def find_si_iv_bursts(cube, *, threshold=None, velocity_range: u.km / u.s = (-50, 50) * u.km / u.s, median_factor=10):
    """
    Find UV bursts in Si IV 1402.77 Å spectra.

    This is `find_bright_spectral_events` for Si IV 1402.77 Å, with the threshold convention
    and default of ``iris_burst_check.pro``.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        Level 2 spectra of a window covering 1402.77 Å, in DN with axes (step, slit, wavelength),
        or (raster, step, slit, wavelength) for several rasters read together.
    threshold : `~astropy.units.Quantity`, optional
        Burst threshold per wavelength bin for data summed by 2 in wavelength and not along
        the slit; it is scaled by the data's spatial summing and half its spectral summing.
        Defaults to 500 DN/s times the ratio of the 1402.77 Å effective area at the observation
        date to that on 2013-10-22 21:00. ``u.DN`` stands for the data's DN; any other unit
        must convert to ``DN_UNIT["FUV"] / u.s`` from `irispy.utils.constants`.
    velocity_range : `~astropy.units.Quantity`, optional
        ``(lower, upper)`` Doppler velocities from 1402.77 Å of the averaged wavelength bins,
        ends included.
    median_factor : `float` or `None`, optional
        Particle-hit test factor; `None` switches the test off.

    Returns
    -------
    labels : `~irispy.spectrograph.SpectrogramCube`
        As `find_bright_spectral_events`.
    events : `~astropy.table.QTable`
        As `find_bright_spectral_events`; ``events.meta["threshold"]`` is the scaled threshold.

    Notes
    -----
    Port of `iris_burst_check.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/nrl/iris_burst_check.pro>`__.
    It finds the same burst pixels and events as IDL on the reference data, with these deliberate differences:

    * IDL does not scale the threshold by the spectral summing, so the same threshold gives the
      same result only on data summed by 2 in wavelength.
    * The median leaves out missing bins; IDL's includes the -200 fill values.
    * The median of an even number of bins is the mean of the two middle values, not IDL's
      upper one.
    * IDL's ``/no_median`` sets the factor to 1e10, which, unlike ``median_factor=None``, still
      rejects every pixel whose median is not positive.
    * Labels are 32-bit integers numbered by step, then slit position; IDL numbers them by slit
      position, then step, in bytes that wrap after 255.
    * Times are exposure midpoints and coordinates follow the per-step pointing of the WCS;
      IDL reports the shutter-open time, the per-step XCEN in x and a header grid along the slit.

    References
    ----------
    * :cite:t:`young2018`
    """
    meta = cube.meta
    unit = DN_UNIT["FUV"] / u.s
    if threshold is None:
        now, then = (
            get_interpolated_effective_area(get_latest_response(time), "FUV", _SI_IV)
            for time in (meta.date_reference, _REFERENCE_TIME)
        )
        threshold = 500 * (now / then).to_value(u.one) * unit
    threshold = _to_unit(threshold, unit, "threshold") * meta.spatial_summing_factor * meta.spectral_summing_factor / 2
    return find_bright_spectral_events(
        cube, threshold, rest_wavelength=_SI_IV, velocity_range=velocity_range, median_factor=median_factor
    )


@u.quantity_input
def find_bright_spectral_events(
    cube,
    threshold,
    *,
    rest_wavelength: u.AA = None,
    velocity_range: u.km / u.s = (-50, 50) * u.km / u.s,
    median_factor=10,
):
    """
    Find compact bright events in the spectra of one line.

    A pixel (raster step, slit position) is part of an event when the mean intensity of the
    wavelength bins within ``velocity_range`` from ``rest_wavelength`` reaches ``threshold`` and
    stays below ``median_factor`` times their median, which rejects particle hits. Pixels that
    touch, diagonally included, form one event. For sit-and-stare data the step axis is time,
    so events also link in time. Saturated samples (see :ref:`irispy-tutorial-saturated-data`) count
    at their value, as in IDL; other masked samples are left out.

    Parameters
    ----------
    cube : `~irispy.spectrograph.SpectrogramCube`
        Level 2 spectra of one window, in DN with axes (step, slit, wavelength), or
        (raster, step, slit, wavelength) for several rasters read together.
    threshold : `~astropy.units.Quantity`
        Threshold per wavelength bin of the data, applied as given, such as ``80 * u.DN / u.s``.
        ``u.DN`` stands for the data's DN; any other unit must convert to the DN of the
        window's detector per second, ``DN_UNIT[band] / u.s`` from `irispy.utils.constants`.
    rest_wavelength : `~astropy.units.Quantity`, optional
        Rest wavelength of the line. Defaults to that of the window, ``meta.rest_wavelength``.
    velocity_range : `~astropy.units.Quantity`, optional
        ``(lower, upper)`` Doppler velocities from ``rest_wavelength`` of the averaged wavelength
        bins, ends included.
    median_factor : `float` or `None`, optional
        Particle-hit test factor; `None` switches the test off.

    Returns
    -------
    labels : `~irispy.spectrograph.SpectrogramCube`
        Event labels on the (step, slit) plane of each raster: 0 outside events, and 1 to N
        for the N events, numbered on through the rasters.
    events : `~astropy.table.QTable`
        One row per event: ``label``, ``raster`` index, number of pixels ``npix``, and the
        ``step``, ``y``, ``time``, ``coordinate`` and mean ``intensity`` of its brightest pixel.
        The coordinates of several rasters are in the frame of the first.
        ``events.meta["threshold"]`` is the threshold.
    """
    if isinstance(cube, MosaicCube):
        msg = "The burst finders need the exposure time of each raster step, which a mosaic does not have"
        raise TypeError(msg)
    meta = cube.meta
    if rest_wavelength is None:
        rest_wavelength = meta.rest_wavelength
        if rest_wavelength is None:
            msg = f"The {meta.spectral_window} window has no rest wavelength; pass rest_wavelength"
            raise ValueError(msg)
    threshold = _to_unit(threshold, DN_UNIT[meta.detector_band] / u.s, "threshold")
    check_velocity_range(velocity_range)
    check_scaled(cube)
    if cube.data.ndim not in (3, 4) or cube.wavelength_axis != cube.data.ndim - 1:
        msg = "The spectra must have axes ([raster,] step, slit, wavelength); slice with a range, not an index"
        raise ValueError(msg)
    if not DN_UNIT[meta.detector_band].is_equivalent(cube.unit):
        msg = f"The spectra must be in DN, not {cube.unit}; do not correct or calibrate them first"
        raise ValueError(msg)
    wavelength = u.Quantity(cube.axis_world_coords(cube.wavelength_axis)[0])
    velocity = wavelength.to(u.km / u.s, equivalencies=u.doppler_optical(rest_wavelength))
    bins = (velocity >= velocity_range[0]) & (velocity <= velocity_range[1])
    if not bins.any():
        msg = (
            f"The spectral window has no wavelength bins between {velocity_range[0]} and {velocity_range[1]} "
            f"from {rest_wavelength}"
        )
        raise ValueError(msg)
    data = cube.data[..., bins]
    # NaN, the -200 fill and, as in iris_getwindata.pro, any other value below -10
    data = np.ma.masked_where(~(data >= -10), data)
    if cube.mask is not None:
        # Saturated samples are masked, but count at their value, as in IDL
        left_out = _boolean_mask(cube) & ~_saturated_samples(cube)
        data[left_out[..., bins]] = np.ma.masked
    mean = data.mean(axis=-1, dtype=float)
    intensity = (mean / meta["exposure time"].to_value(u.s)[..., np.newaxis]).filled(np.nan)
    burst = intensity >= threshold.to_value(cube.unit / u.s)
    if median_factor is not None:
        burst[burst] = mean[burst] < median_factor * np.ma.median(data[burst], axis=-1)
    # Pixels touch within a raster, not across rasters
    structure = np.zeros((3,) * burst.ndim, dtype=bool)
    structure[(1,) * (burst.ndim - 2)] = _EIGHT_CONNECTED
    labels, count = ndimage.label(burst, structure=structure)
    label, npix, index = _events(intensity, labels, count)
    template = make_spatial_template(cube, cube.wavelength_axis)
    # A raster's gWCS also gives the time and raster step of each pixel
    coordinate = next(world for world in template.wcs.array_index_to_world(*index) if isinstance(world, SkyCoord))
    events = QTable(
        {
            "label": label,
            "raster": index[0] if burst.ndim == 3 else np.zeros(count, dtype=int),
            "npix": npix,
            "step": index[-2],
            "y": index[-1],
            "time": cube.time[index[:-1]],
            "coordinate": coordinate,
            "intensity": intensity[index] * cube.unit / u.s,
        },
        meta={"threshold": threshold},
    )
    return make_map_cube(template, labels, u.one), events


def find_bright_image_events(cube, *, sigma_factor=10, min_pixels=2):
    """
    Find compact bright events in images.

    A pixel is part of an event when it is at least ``sigma_factor`` standard deviations above
    the median of its frame. Pixels that touch within a frame, diagonally included, form one event,
    so an event that lasts several frames is one event in each. To join them in time, label the
    event pixels again: ``scipy.ndimage.label(labels.data > 0, structure=np.ones((3, 3, 3)))``.
    Saturated pixels (see :ref:`irispy-tutorial-saturated-data`) count at their value, as in IDL.

    Parameters
    ----------
    cube : `~irispy.sji.SJICube`
        Slit-jaw images of any band, or IRIS-aligned AIA images, with axes (frame, y, x).
    sigma_factor : `float`, optional
        Threshold, in standard deviations.
    min_pixels : `int`, optional
        Events with fewer pixels are dropped.

    Returns
    -------
    labels : `~irispy.sji.SJICube`
        Event labels with the WCS and extra coordinates of ``cube``: 0 outside events, and 1 to N
        for the N events.
    events : `~astropy.table.QTable`
        One row per event: ``label``, ``frame``, number of pixels ``npix``, the frame's ``threshold``,
        and the ``y``, ``x``, ``time``, ``coordinate`` and ``intensity`` of its brightest pixel.

    Notes
    -----
    The defaults are for UV bursts in 1400 Å slit-jaw images.
    Port of `iris_sji_burst_check.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/nrl/iris_sji_burst_check.pro>`__,
    which fixes the parameters at these defaults. These differences from IDL are deliberate:

    * Edge pixels can belong to events; IDL's ``REGION_GROW`` leaves them out of every region,
      and an above-threshold edge pixel makes the IDL loop never end.
    * Masked pixels other than saturated ones, including both -200 and -199 fill values, are left
      out of the statistics and are never event pixels. IDL excludes only -200, so the frame
      thresholds can differ.
    * The median of an even number of pixels is the mean of the two middle values, not IDL's
      upper one, and the standard deviation is accumulated in double precision.
    * Labels are unique through the cube; IDL numbers the events of each frame from 1, in bytes
      that wrap after 255.

    The 10-sigma default is a quick look: on active-region data it flags events in every frame,
    and :cite:t:`young2018` recommend a threshold chosen for each data set.

    References
    ----------
    * :cite:t:`young2018`
    """
    check_scaled(cube)
    if cube.data.ndim != 3:
        msg = "The slit-jaw cube must have axes (frame, y, x); slice with a range, such as [0:1], not an index"
        raise ValueError(msg)
    # Saturated pixels are masked, but count at their value, as in IDL
    data = cube.data
    if cube.mask is not None:
        data = np.where(_boolean_mask(cube) & ~_saturated_samples(cube), np.nan, data)
    frames = data.reshape(len(data), -1)
    with warnings.catch_warnings():
        # Frames without valid pixels, or with only one
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        threshold = np.nanmedian(frames, axis=1) + sigma_factor * np.nanstd(frames, axis=1, ddof=1, dtype=float)
    within_frame = np.stack([np.zeros_like(_EIGHT_CONNECTED), _EIGHT_CONNECTED, np.zeros_like(_EIGHT_CONNECTED)])
    burst = data >= threshold[:, np.newaxis, np.newaxis]
    labels, _ = ndimage.label(burst, structure=within_frame)
    burst &= (np.bincount(labels.ravel()) >= min_pixels)[labels]
    labels, count = ndimage.label(burst, structure=within_frame)
    label, npix, (frame, y, x) = _events(data, labels, count)
    coordinate, time = cube.wcs.array_index_to_world(frame, y, x)
    events = QTable(
        {
            "label": label,
            "frame": frame,
            "npix": npix,
            "threshold": threshold[frame] * cube.unit,
            "y": y,
            "x": x,
            "time": time,
            "coordinate": coordinate,
            "intensity": data[frame, y, x] * cube.unit,
        }
    )
    # Coordinates point back to their cube; copy them without copying its data.
    extra_coords = deepcopy(cube.extra_coords, {id(cube): None})
    return type(cube)(labels, cube.wcs, unit=u.one, meta=deepcopy(cube.meta), extra_coords=extra_coords), events


def _events(values, labels, count):
    """
    The label, number of pixels and array index of the brightest pixel of each event.
    """
    # Only the labeled pixels: maximum_position makes several copies of whatever it is given
    where = np.flatnonzero(labels)
    event = labels.ravel()[where]
    label = np.arange(1, count + 1)
    peak = ndimage.maximum_position(values.ravel()[where], event, label) if count else []
    npix = np.bincount(event, minlength=count + 1)[1:]
    return label, npix, np.unravel_index(where[np.array(peak, dtype=int).reshape(-1)], labels.shape)
