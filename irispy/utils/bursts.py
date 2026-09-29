"""
Detection of UV bursts in IRIS Si IV spectra and 1400 Å slit-jaw images.
"""

import warnings

import numpy as np
from scipy import ndimage

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time

from irispy.spectrograph import RasterCollection, SpectrogramCubeSequence
from irispy.utils._spectral import check_scaled, make_map_cube, make_spatial_template
from irispy.utils.constants import DN_UNIT
from irispy.utils.response import get_interpolated_effective_area, get_latest_response

__all__ = ["find_si_iv_bursts", "find_sji_bursts"]

_SI_IV = 1402.77 * u.AA
# The 500 DN/s default threshold was set on this observation, summed by 2 in wavelength.
_REFERENCE_TIME = Time("2013-10-22T21:00")
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def find_si_iv_bursts(raster, *, threshold=None, velocity_range=50 * u.km / u.s, median_factor=10):
    """
    Find UV bursts in Si IV 1402.77 Å spectra.

    A pixel (raster step, slit position) is part of a burst when the mean intensity of the
    wavelength bins within ``velocity_range`` of 1402.77 Å reaches the threshold and stays
    below ``median_factor`` times their median, which rejects particle hits. Burst pixels that
    touch, diagonally included, form one event. For sit-and-stare data the step axis is time,
    so events also link in time.

    Parameters
    ----------
    raster : `~irispy.SpectrogramCube`, `~irispy.SpectrogramCubeSequence` or `~irispy.RasterCollection`
        Level 2 spectra in DN with axes (step, slit, wavelength). From a `~irispy.RasterCollection`,
        the first window covering 1402.77 Å is used.
    threshold : `float` or `~astropy.units.Quantity`, optional
        Burst threshold in DN/s per wavelength bin for data summed by 2 in wavelength and not along
        the slit; it is scaled by the data's spatial summing and half its spectral summing.
        Defaults to 500 DN/s times the ratio of the 1402.77 Å effective area at the observation
        date to that on 2013-10-22 21:00. A Quantity in ``u.DN`` per unit time is read as the
        data's DN; any other unit must convert to ``DN_UNIT["FUV"] / u.s`` from `irispy.utils.constants`.
    velocity_range : `~astropy.units.Quantity`, optional
        Half-width of the averaged wavelength range, as a Doppler velocity.
    median_factor : `float` or `None`, optional
        Particle-hit test factor; `None` switches the test off.

    Returns
    -------
    labels : `~irispy.SpectrogramCube` or `~irispy.SpectrogramCubeSequence`
        Event labels on the (step, slit) plane of each raster: 0 outside bursts, and 1 to N
        for the N events, numbered on through the rasters of a sequence.
    events : `~astropy.table.QTable`
        One row per event: ``label``, ``raster`` index, number of pixels ``npix``, and the
        ``step``, ``y``, ``time``, ``coordinate`` and mean ``intensity`` of its brightest pixel.
        The coordinates of a sequence are in the frame of its first raster.
        ``events.meta["threshold"]`` is the scaled threshold.

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
    * `Young et al. (2018), Space Science Reviews 214, 120 <https://doi.org/10.1007/s11214-018-0551-z>`__
    """
    if isinstance(raster, RasterCollection):
        raster = _si_iv_window(raster)
    cubes = raster.data if isinstance(raster, SpectrogramCubeSequence) else [raster]
    meta = cubes[0].meta
    if threshold is None:
        area_ratio = _si_iv_effective_area(meta.date_reference) / _si_iv_effective_area(_REFERENCE_TIME)
        threshold = 500 * area_ratio.to_value(u.one)
    unit = DN_UNIT["FUV"] / u.s
    if isinstance(threshold, u.Quantity):
        threshold = threshold.to(unit, equivalencies=[(u.DN / u.s, unit)])
    threshold = u.Quantity(threshold, unit) * meta.spatial_summing_factor * meta.spectral_summing_factor / 2
    maps, tables, offset = [], [], 0
    for index, cube in enumerate(cubes):
        labels, count, intensity = _label_si_iv(cube, threshold, velocity_range, median_factor)
        label, npix, (step, y) = _events(intensity, labels, count)
        labels[labels > 0] += offset
        template = make_spatial_template(cube, cube.wavelength_axis)
        coordinate = template.wcs.array_index_to_world(step, y)
        if tables:
            # Rasters read separately have their own obstime, which vstack rejects
            coordinate = SkyCoord(tables[0]["coordinate"].frame.realize_frame(coordinate.data))
        table = QTable(
            {
                "label": label + offset,
                "raster": np.full(count, index),
                "npix": npix,
                "step": step,
                "y": y,
                "time": cube.axis_world_coords("time", wcs=cube.extra_coords)[0][step],
                "coordinate": coordinate,
                "intensity": intensity[step, y] * cube.unit / u.s,
            },
            meta={"threshold": threshold},
        )
        maps.append(make_map_cube(template, labels, u.one))
        tables.append(table)
        offset += count
    if not isinstance(raster, SpectrogramCubeSequence):
        return maps[0], tables[0]
    return SpectrogramCubeSequence(maps, meta=raster.meta), vstack(tables)


def find_sji_bursts(sji, *, sigma_factor=10, min_pixels=2):
    """
    Find UV bursts in 1400 Å slit-jaw images.

    A pixel is part of a burst when it is at least ``sigma_factor`` standard deviations above
    the median of its frame. Burst pixels that touch within a frame, diagonally included, form one event.

    Parameters
    ----------
    sji : `~irispy.sji.SJICube`
        A 1400 Å slit-jaw cube with axes (frame, y, x).
    sigma_factor : `float`, optional
        Burst threshold, in standard deviations.
    min_pixels : `int`, optional
        Events with fewer pixels are dropped.

    Returns
    -------
    labels : `~irispy.sji.SJICube`
        Event labels with the WCS and extra coordinates of ``sji``: 0 outside bursts, and 1 to N
        for the N events.
    events : `~astropy.table.QTable`
        One row per event: ``label``, ``frame``, number of pixels ``npix``, the frame's ``threshold``,
        and the ``y``, ``x``, ``time``, ``coordinate`` and ``intensity`` of its brightest pixel.

    Notes
    -----
    Port of `iris_sji_burst_check.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/nrl/iris_sji_burst_check.pro>`__,
    which fixes the parameters at their defaults. It finds the same burst pixels and events on the
    IDL reference data, with these deliberate differences:

    * Edge pixels can belong to events; IDL's ``REGION_GROW`` leaves them out of every region,
      and an above-threshold edge pixel makes the IDL loop never end.
    * Masked pixels, not only the -200 fill, are left out of the statistics and are never burst
      pixels.
    * The median of an even number of pixels is the mean of the two middle values, not IDL's
      upper one, and the standard deviation is accumulated in double precision.
    * Labels are unique through the cube; IDL numbers the events of each frame from 1, in bytes
      that wrap after 255.

    The 10-sigma default is a quick look: on active-region data it flags events in every frame,
    and Young et al. (2018) recommend a threshold chosen for each data set.

    References
    ----------
    * `Young et al. (2018), Space Science Reviews 214, 120 <https://doi.org/10.1007/s11214-018-0551-z>`__
    """
    if sji.meta.spectral_window != "1400":
        msg = f"Bursts are found in 1400 Å slit-jaw images, not {sji.meta.spectral_window}"
        raise ValueError(msg)
    check_scaled(sji)
    if sji.data.ndim != 3:
        msg = "The slit-jaw cube must have axes (frame, y, x); slice with a range, such as [0:1], not an index"
        raise ValueError(msg)
    data = sji.data if sji.mask is None else np.where(sji.mask, np.nan, sji.data)
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
    coordinate, time = sji.wcs.array_index_to_world(frame, y, x)
    events = QTable(
        {
            "label": label,
            "frame": frame,
            "npix": npix,
            "threshold": threshold[frame] * sji.unit,
            "y": y,
            "x": x,
            "time": time,
            "coordinate": coordinate,
            "intensity": data[frame, y, x] * sji.unit,
        }
    )
    return type(sji)(labels, sji.wcs, unit=u.one, meta=sji.meta, extra_coords=sji.extra_coords), events


def _si_iv_window(collection):
    for window in collection.values():
        cube = window.data[0] if isinstance(window, SpectrogramCubeSequence) else window
        low, high = cube.meta.spectral_range
        if low <= _SI_IV <= high:
            return window
    msg = "No spectral window covers Si IV 1402.77 Å"
    raise ValueError(msg)


def _si_iv_effective_area(time):
    return get_interpolated_effective_area(get_latest_response(time), "FUV", _SI_IV)


def _label_si_iv(cube, threshold, velocity_range, median_factor):
    """
    Label the burst pixels of one raster; also return the mean intensity per second.
    """
    check_scaled(cube)
    if cube.data.ndim != 3:
        msg = "The spectra must have axes (step, slit, wavelength); slice with a range, not an index"
        raise ValueError(msg)
    if not DN_UNIT["FUV"].is_equivalent(cube.unit):
        msg = f"The spectra must be in DN, not {cube.unit}; do not correct or calibrate them first"
        raise ValueError(msg)
    wavelength = u.Quantity(cube.axis_world_coords(cube.wavelength_axis)[0])
    bins = np.abs(wavelength.to(u.km / u.s, equivalencies=u.doppler_optical(_SI_IV))) <= velocity_range
    if not bins.any():
        msg = f"The spectral window has no wavelength bins within {velocity_range} of Si IV 1402.77 Å"
        raise ValueError(msg)
    data = cube.data[..., bins]
    # NaN, the -200 fill and, as in iris_getwindata.pro, any other value below -10
    data = np.ma.masked_where(~(data >= -10), data)
    if cube.mask is not None:
        data[np.broadcast_to(cube.mask, cube.data.shape)[..., bins]] = np.ma.masked
    mean = data.mean(axis=-1, dtype=float)
    intensity = (mean / cube.meta["exposure time"].to_value(u.s)[:, np.newaxis]).filled(np.nan)
    burst = intensity >= threshold.to_value(cube.unit / u.s)
    if median_factor is not None:
        burst[burst] = mean[burst] < median_factor * np.ma.median(data[burst], axis=-1)
    labels, count = ndimage.label(burst, structure=_EIGHT_CONNECTED)
    return labels, count, intensity


def _events(values, labels, count):
    """
    The label, number of pixels and array index of the brightest pixel of each event.
    """
    # Only the labelled pixels: maximum_position makes several copies of whatever it is given
    where = np.flatnonzero(labels)
    event = labels.ravel()[where]
    label = np.arange(1, count + 1)
    peak = ndimage.maximum_position(values.ravel()[where], event, label) if count else []
    npix = np.bincount(event, minlength=count + 1)[1:]
    return label, npix, np.unravel_index(where[np.array(peak, dtype=int).reshape(-1)], labels.shape)
