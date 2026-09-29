"""
Detection of UV bursts in IRIS Si IV spectra and 1400 Å slit-jaw images.

Ports of P. Young's SolarSoft routines ``iris_burst_check.pro`` and
``iris_sji_burst_check.pro`` (2017).
"""

import warnings

import numpy as np
from scipy import ndimage

import astropy.units as u
from astropy.constants import c
from astropy.table import QTable, vstack
from astropy.time import Time

from irispy.spectrograph import RasterCollection, SpectrogramCubeSequence
from irispy.utils._spectral import make_map_cube, make_spatial_template
from irispy.utils.response import get_latest_response

__all__ = ["find_si_iv_bursts", "find_sji_bursts"]

_SI_IV = 1402.77 * u.AA
# The default threshold of 500 DN/s was set on this observation, which is summed by 2 in wavelength.
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
    raster : `~irispy.spectrograph.SpectrogramCube`, `~irispy.spectrograph.SpectrogramCubeSequence` or `~irispy.spectrograph.RasterCollection`
        Level 2 spectra read with ``memmap=False``, with axes (step, slit, wavelength).
        From a `~irispy.spectrograph.RasterCollection`, the first window covering 1402.77 Å is used.
    threshold : `float` or `~astropy.units.Quantity`, optional
        Burst threshold in DN/s per wavelength bin, for data summed by 2 in wavelength and not
        along the slit. It is scaled by the spatial summing and by half the spectral summing of the data.
        Defaults to 500 DN/s times the ratio of the 1402.77 Å effective area at the observation
        date to that on 2013-10-22 21:00.
    velocity_range : `~astropy.units.Quantity`, optional
        Half-width of the Doppler velocity range averaged around 1402.77 Å.
    median_factor : `float` or `None`, optional
        A burst pixel's mean must be below this factor times the median of the same bins.
        `None` switches the test off.

    Returns
    -------
    labels : `~irispy.spectrograph.SpectrogramCube` or `~irispy.spectrograph.SpectrogramCubeSequence`
        Event labels on the (step, slit) plane of each raster: 0 outside bursts, and 1 to N
        for the N events, numbered on through the rasters of a sequence.
    events : `~astropy.table.QTable`
        One row per event: its ``label``, the ``raster`` it is in, its number of pixels ``npix``, and
        the ``step``, ``y``, ``time``, ``coordinate`` and mean ``intensity`` of its brightest pixel.
        ``events.meta["threshold"]`` is the threshold applied to the data.

    Notes
    -----
    This is a port of ``iris_burst_check.pro``. It finds the same burst pixels and events on
    the IDL reference data, with these deliberate differences:

    * The threshold is scaled by half the spectral summing. IDL does not scale it, so its
      default, set on data summed by 2, is twice too high for unsummed data and twice too low
      for data summed by 4. Data summed by 2 give the same result.
    * The median leaves out missing bins; IDL's includes the -200 fill values.
    * ``median_factor=None`` skips the median test. IDL's ``/no_median`` raises the factor to
      1e10 instead, which still rejects every pixel whose median is not positive.
    * Labels are 32-bit integers numbered in array order; IDL's are bytes that wrap after 255.
    * Times are exposure midpoints and coordinates follow the per-step pointing of the WCS;
      IDL reports the shutter-open time and a regular grid built from the header.
    * Only the bins within ``velocity_range`` are averaged, and the median is only computed
      for pixels above the threshold, so large sit-and-stare windows are cheap to search.

    References
    ----------
    * `iris_burst_check.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/nrl/iris_burst_check.pro>`__
    * `Young et al. (2018), Space Science Reviews 214, 120 <https://doi.org/10.1007/s11214-018-0551-z>`__
    """
    if isinstance(raster, RasterCollection):
        raster = _si_iv_window(raster)
    cubes = raster.data if isinstance(raster, SpectrogramCubeSequence) else [raster]
    if threshold is None:
        area_ratio = _si_iv_effective_area(cubes[0].meta.date_reference) / _si_iv_effective_area(_REFERENCE_TIME)
        threshold = 500 * area_ratio.to_value(u.one)
    maps, tables, offset = [], [], 0
    for index, cube in enumerate(cubes):
        scaled_threshold = (
            u.Quantity(threshold, cube.unit / u.s)
            * cube.meta.spatial_summing_factor
            * cube.meta.spectral_summing_factor
            / 2
        )
        labels, count, intensity = _label_si_iv(cube, scaled_threshold.value, velocity_range, median_factor)
        labels[labels > 0] += offset
        template = make_spatial_template(cube, cube.wavelength_axis)
        label, npix, (step, y) = _events(intensity, labels, count, offset)
        table = QTable(
            {
                "label": label,
                "raster": np.full(count, index),
                "npix": npix,
                "step": step,
                "y": y,
                "time": cube.axis_world_coords("time", wcs=cube.extra_coords)[0][step],
                "coordinate": template.wcs.array_index_to_world(step, y),
                "intensity": intensity[step, y] * cube.unit / u.s,
            },
            meta={"threshold": scaled_threshold},
        )
        maps.append(make_map_cube(template, labels, u.one))
        tables.append(table)
        offset += count
    if not isinstance(raster, SpectrogramCubeSequence):
        return maps[0], tables[0]
    return SpectrogramCubeSequence(maps, meta=raster.meta), vstack(tables, metadata_conflicts="silent")


def find_sji_bursts(sji, *, sigma_factor=10, min_pixels=2):
    """
    Find UV bursts in 1400 Å slit-jaw images.

    A pixel is part of a burst when it is at least ``sigma_factor`` standard deviations above
    the median of its frame. Burst pixels that touch within a frame, diagonally included, form
    one event, and events with fewer than ``min_pixels`` pixels are dropped.

    Parameters
    ----------
    sji : `~irispy.sji.SJICube`
        A 1400 Å slit-jaw cube read with ``memmap=False``.
    sigma_factor : `float`, optional
        Burst threshold, in standard deviations above the median of each frame.
    min_pixels : `int`, optional
        Smallest event kept.

    Returns
    -------
    labels : `~irispy.sji.SJICube`
        Event labels with the WCS of ``sji``: 0 outside bursts, and 1 to N for the N events.
    events : `~astropy.table.QTable`
        One row per event: its ``label``, ``frame``, number of pixels ``npix``, the frame's
        ``threshold``, and the ``y``, ``x``, ``time``, ``coordinate`` and ``intensity`` of its
        brightest pixel.

    Notes
    -----
    This is a port of ``iris_sji_burst_check.pro``, which fixes ``sigma_factor`` at 10 and
    ``min_pixels`` at 2. It finds the same burst pixels and events on the IDL reference data,
    with these deliberate differences:

    * Pixels on the image edges can belong to events. IDL's ``REGION_GROW`` never grows a
      region from an edge pixel.
    * Masked pixels are left out of the statistics, not only the -200 fill.
    * The median of an even number of pixels is the mean of the two middle values; IDL takes
      the upper one. The standard deviation is accumulated in double precision.
    * Labels are unique through the cube; IDL numbers the events of each frame from 1, in bytes
      that wrap after 255.
    * All frames are processed at once.

    A fixed 10-sigma rule is a quick look: on active-region data it flags events in every frame,
    and Young et al. (2018) recommend thresholds chosen for each data set.

    References
    ----------
    * `iris_sji_burst_check.pro <https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/nrl/iris_sji_burst_check.pro>`__
    * `Young et al. (2018), Space Science Reviews 214, 120 <https://doi.org/10.1007/s11214-018-0551-z>`__
    """
    if sji.meta.spectral_window != "1400":
        msg = f"Bursts are found in 1400 Å slit-jaw images, not {sji.meta.spectral_window}"
        raise ValueError(msg)
    if not sji.meta.get("scaled", True):
        msg = "The slit-jaw data are unscaled; read them with memmap=False"
        raise ValueError(msg)
    data = sji.data if sji.mask is None else np.where(sji.mask, np.nan, sji.data)
    frames = data.reshape(len(data), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames without valid pixels
        threshold = np.nanmedian(frames, axis=1) + sigma_factor * np.nanstd(frames, axis=1, ddof=1, dtype=float)
    within_frame = np.stack([np.zeros_like(_EIGHT_CONNECTED), _EIGHT_CONNECTED, np.zeros_like(_EIGHT_CONNECTED)])
    labels, _ = ndimage.label(data >= threshold[:, np.newaxis, np.newaxis], structure=within_frame)
    kept = np.bincount(labels.ravel()) >= min_pixels
    kept[0] = False
    relabel = np.zeros(kept.size, dtype=labels.dtype)
    relabel[kept] = np.arange(1, kept.sum() + 1)
    labels = relabel[labels]
    count = int(kept.sum())
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
    return type(sji)(labels, sji.wcs, unit=u.one, meta=sji.meta), events


def _si_iv_window(collection):
    for window in collection.values():
        cube = window.data[0] if isinstance(window, SpectrogramCubeSequence) else window
        low, high = cube.meta.spectral_range
        if low <= _SI_IV <= high:
            return window
    msg = "No spectral window covers Si IV 1402.77 Å"
    raise ValueError(msg)


def _si_iv_effective_area(time):
    response = get_latest_response(time)
    closest = np.argmin(np.abs(response["LAMBDA"] - _SI_IV))
    return response["AREA_SG"][0, closest]


def _label_si_iv(cube, threshold, velocity_range, median_factor):
    """
    Label the burst pixels of one raster; also return the mean intensity in DN/s.
    """
    if np.issubdtype(cube.data.dtype, np.integer):
        msg = "The spectrograph data are unscaled; read them with memmap=False"
        raise ValueError(msg)
    wavelength = cube.axis_world_coords(cube.wavelength_axis)[0]
    velocity = ((wavelength - _SI_IV) / _SI_IV * c).to(u.km / u.s)
    bins = np.flatnonzero(np.abs(velocity) <= velocity_range)
    if not bins.size:
        msg = f"The spectral window has no wavelength bins within {velocity_range} of Si IV 1402.77 Å"
        raise ValueError(msg)
    bins = slice(bins[0], bins[-1] + 1)
    data = cube.data[..., bins]
    # The -200 fill, and the values from -199 to -10 that iris_getwindata.pro also treats as missing
    missing = data < -10
    if cube.mask is not None:
        missing |= np.broadcast_to(cube.mask, cube.data.shape)[..., bins]
    exposure = np.broadcast_to(cube.meta["exposure time"].to_value(u.s)[:, np.newaxis], data.shape[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):  # pixels and zero-exposure steps without data
        intensity = np.where(missing, 0, data).sum(axis=-1, dtype=float) / (~missing).sum(axis=-1) / exposure
    burst = intensity >= threshold
    if median_factor is not None:
        median = np.nanmedian(np.where(missing[burst], np.nan, data[burst]), axis=-1) / exposure[burst]
        burst[burst] = intensity[burst] < median_factor * median
    labels, count = ndimage.label(burst, structure=_EIGHT_CONNECTED)
    return labels, count, intensity


def _events(values, labels, count, offset=0):
    """
    The label, number of pixels and array index of the brightest pixel of each event.
    """
    label = np.arange(offset + 1, offset + count + 1)
    peak = np.array(ndimage.maximum_position(values, labels, label), dtype=int).reshape(count, values.ndim)
    npix = np.bincount(labels.ravel(), minlength=offset + count + 1)[offset + 1 :]
    return label, npix, tuple(peak.T)
