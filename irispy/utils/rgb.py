"""
False-color (RGB) rendering of IRIS spectrogram cubes.

Every spatial pixel of a spectrogram cube holds a whole spectrum, which is treated here
as a spectral power distribution and converted into a single sRGB color. The result is
an image where brightness represents the total intensity of the spectral window and hue
represents where in the window that intensity sits, so Doppler shifts and line
asymmetries show up as color.
"""

from importlib import import_module

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.units as u

__all__ = ["calculate_rgb", "plot_rgb"]


def _import_colorsynth():
    try:
        return import_module("colorsynth")
    except ImportError as exc:
        msg = (
            "colorsynth is an optional dependency required for false-color RGB images. "
            "Install it with `pip install colorsynth` or `pip install 'irispy-lmsal[rgb]'`."
        )
        raise ImportError(msg) from exc


def _data_as_float(cube):
    """
    Return the cube data as a float array with masked samples set to NaN.
    """
    data = np.asarray(cube.data, dtype=float)
    if cube.mask is not None:
        data = np.where(np.asarray(cube.mask, dtype=bool), np.nan, data)
    return data


def _wavelength_of(cube, axis_wavelength, ndim):
    """
    Return the wavelength of each spectral bin, shaped to broadcast against the data.
    """
    (wavelength,) = cube.axis_world_coords("em.wl")
    wavelength = u.Quantity(wavelength).to(u.AA)
    if wavelength.ndim != 1:
        msg = f"The wavelength coordinate must be one dimensional, got shape {wavelength.shape}"
        raise ValueError(msg)
    shape = [1] * ndim
    shape[axis_wavelength] = wavelength.size
    return wavelength.reshape(shape)


def _as_value(bound, unit):
    """
    Convert an optional intensity bound into a bare number in the units of the cube.
    """
    if bound is None or not isinstance(bound, u.Quantity):
        return bound
    if unit is None:
        return bound.to_value(u.dimensionless_unscaled)
    return bound.to_value(unit)


def _clipped(norm):
    """
    Keep ``norm`` away from the negative intensities that ``vmin`` maps below zero.

    ``colorsynth`` applies ``norm`` to the intensity after it has been scaled into the
    range [0, 1], so data fainter than ``vmin`` arrives negative and turns ordinary
    choices such as `numpy.sqrt` into NaN.
    """
    if norm is None:
        return None
    return lambda x: norm(np.clip(x, 0, None))


def _midpoints(a, axis):
    """
    Extend an axis of cell centers by one, placing the new samples between the old.
    """
    if a.shape[axis] < 2:
        msg = "Both spatial axes need at least two pixels to draw a false-color image"
        raise ValueError(msg)
    lower = np.take(a, np.arange(a.shape[axis] - 1), axis=axis)
    upper = np.take(a, np.arange(1, a.shape[axis]), axis=axis)
    first = 1.5 * np.take(a, [0], axis=axis) - 0.5 * np.take(a, [1], axis=axis)
    last = 1.5 * np.take(a, [-1], axis=axis) - 0.5 * np.take(a, [-2], axis=axis)
    return np.concatenate([first, (lower + upper) / 2, last], axis=axis)


def _time_grid(cube, shape):
    """
    Return the observation time of every pixel as a matplotlib date number.
    """
    (time,) = cube.axis_world_coords("time", wcs=cube.extra_coords)
    time = matplotlib.dates.date2num(time.to_datetime())
    if time.shape == shape:
        return time
    if time.shape == shape[:1]:
        return np.broadcast_to(time[:, np.newaxis], shape)
    msg = f"The time coordinate has shape {time.shape}, which does not line up with the image shape {shape}"
    raise ValueError(msg)


def _colorbar_axes(ax, fraction, pad):
    """
    Carve a colorbar out of the right hand side of the image axes.

    Splitting the grid cell that ``ax`` already occupies keeps the colorbar under the
    figure layout engine, so its labels are counted when the panels are packed. Stealing
    the space with `mpl_toolkits.axes_grid1` instead leaves the colorbar invisible to
    the engine, and neighbouring panels then overlap it.
    """
    figure = ax.get_figure()
    if pad is None:
        # A layout engine measures the ticks and label itself, so it only needs a
        # visual gap. Without one, the gap is the only room they will get.
        pad = 0.08 if figure.get_layout_engine() is not None else 0.5
    subplotspec = ax.get_subplotspec()
    if subplotspec is None:
        # Axes outside any grid, such as from `matplotlib.figure.Figure.add_axes`.
        pad_inches = pad * ax.get_position().width * figure.get_figwidth()
        cax = make_axes_locatable(ax).append_axes("right", size=f"{fraction * 100}%", pad=pad_inches)
        return cax, False
    gridspec = subplotspec.subgridspec(1, 2, width_ratios=[1 - fraction, fraction], wspace=pad)
    ax.set_subplotspec(gridspec[0])
    return figure.add_subplot(gridspec[1]), True


def _match_heights(ax, cax, aspect, fraction):
    """
    Give the colorbar the same height as the image.

    The grid hands each of them the full height of the cell, but a fixed aspect ratio
    shrinks the image inside it. Pinning both box aspects is what survives the layout
    engine here; an axes locator gets replaced when the engine runs.
    """
    if aspect == "auto":
        return
    scale = 1.0 if aspect == "equal" else float(aspect)
    width = abs(np.subtract(*ax.get_xlim()))
    height = abs(np.subtract(*ax.get_ylim()))
    box_aspect = scale * height / width
    ax.set_box_aspect(box_aspect)
    cax.set_box_aspect(box_aspect * (1 - fraction) / fraction)


def _rest_wavelength_of(cube, rest_wavelength):
    """
    Resolve the rest wavelength of the line, falling back to the window metadata.
    """
    if rest_wavelength is not None:
        return u.Quantity(rest_wavelength)
    try:
        return u.Quantity(cube.meta.rest_wavelength)
    except (AttributeError, TypeError, ValueError):
        return None


def _cell_edges(centers):
    """
    Convert a two dimensional grid of cell centers into the grid of cell corners.

    `matplotlib.pyplot.pcolormesh` can do this itself, but it first checks that the
    centers increase monotonically, which pointing jitter along the raster axis breaks
    even though the interpolation is still the one we want.
    """
    return _midpoints(_midpoints(centers, axis=0), axis=1)


def calculate_rgb(
    cube,
    *,
    wavelength_min=None,
    wavelength_max=None,
    vmin=None,
    vmax=None,
    norm=None,
    percentile=99.9,
    num_intensity=101,
):
    """
    Convert the spectra in a spectrogram cube into a false-color RGB image.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        A three dimensional cube with one spectral axis and two spatial axes.
    wavelength_min : `astropy.units.Quantity`, optional
        Wavelength mapped to the blue end of the human visible range.
        Defaults to the shortest wavelength in the cube.
    wavelength_max : `astropy.units.Quantity`, optional
        Wavelength mapped to the red end of the human visible range.
        Defaults to the longest wavelength in the cube.
    vmin : `float` or `astropy.units.Quantity`, optional
        Intensity mapped to black. Defaults to zero.
    vmax : `float` or `astropy.units.Quantity`, optional
        Intensity mapped to full brightness.
        Defaults to the ``percentile`` percentile of the finite data.
    norm : `callable`, optional
        Function applied to the normalized intensity before it is mapped to a
        color, for example `numpy.sqrt` to brighten the faint structure.
    percentile : `float`, optional
        Percentile of the data used for ``vmax`` when ``vmax`` is not given.
    num_intensity : `int`, optional
        Number of intensity samples in the returned colorbar.

    Returns
    -------
    rgb : `numpy.ndarray`
        Array of shape ``(*spatial_shape, 3)`` with values in the range [0, 1],
        where the spatial axes keep the order they have in the cube.
    colorbar : `tuple`
        The ``(intensity, wavelength, rgb)`` triple describing the two dimensional
        colorbar for this image, ready to hand to `matplotlib.pyplot.pcolormesh`.

    Notes
    -----
    Wavelengths outside ``[wavelength_min, wavelength_max]`` fall outside the
    human visible range and so contribute nothing to the resulting color.
    """
    colorsynth = _import_colorsynth()
    if len(cube.shape) != 3:
        msg = f"A false-color image needs a three dimensional cube, got shape {cube.shape}"
        raise ValueError(msg)
    axis_wavelength = cube.wavelength_axis
    data = _data_as_float(cube)
    wavelength = _wavelength_of(cube, axis_wavelength, data.ndim)
    vmin = _as_value(vmin, cube.unit)
    vmax = _as_value(vmax, cube.unit)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanpercentile(data, percentile)
    rgb, colorbar = colorsynth.rgb_and_colorbar(
        spd=data,
        wavelength=wavelength,
        axis=axis_wavelength,
        spd_min=vmin,
        spd_max=vmax,
        spd_norm=_clipped(norm),
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        num_intensity=num_intensity,
    )
    return np.moveaxis(rgb, axis_wavelength, -1), colorbar


def plot_rgb(
    cube,
    *,
    ax=None,
    cax=None,
    wavelength_min=None,
    wavelength_max=None,
    vmin=None,
    vmax=None,
    norm=None,
    percentile=99.9,
    coordinates="helioprojective",
    aspect=None,
    velocity=True,
    rest_wavelength=None,
    cbar_fraction=0.1,
    cbar_pad=None,
    **kwargs,
):
    """
    Plot a spectrogram cube as a false-color image.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        A three dimensional cube with one spectral axis and two remaining axes.
    ax : `matplotlib.axes.Axes`, optional
        Axes on which to draw the image. A new figure is created if not given.
    cax : `matplotlib.axes.Axes`, optional
        Axes on which to draw the two dimensional colorbar.
        Space is stolen from ``ax`` if not given.
    wavelength_min, wavelength_max : `astropy.units.Quantity`, optional
        Wavelengths mapped to the blue and red ends of the human visible range.
    vmin, vmax : `float` or `astropy.units.Quantity`, optional
        Intensities mapped to black and to full brightness.
    norm : `callable`, optional
        Function applied to the normalized intensity before it is mapped to a color.
    percentile : `float`, optional
        Percentile of the data used for ``vmax`` when ``vmax`` is not given.
    coordinates : ``{"helioprojective", "time"}``, optional
        What to put on the horizontal axis. ``"helioprojective"`` suits a raster,
        whose steps cover ground. ``"time"`` suits a sit-and-stare, whose steps
        revisit the same pointing, so that longitude would show only the drift
        from solar rotation.
    aspect : `str` or `float`, optional
        Aspect ratio of the image axes. Defaults to ``"equal"`` in helioprojective
        coordinates and ``"auto"`` against time. Equal aspect squashes a raster with
        only a handful of steps into a thin strip, so pass ``"auto"`` for those.
    velocity : `bool`, optional
        Whether to label the colorbar with Doppler velocity as well as wavelength.
        Ignored when no rest wavelength is available.
    rest_wavelength : `astropy.units.Quantity`, optional
        Rest wavelength that zero velocity corresponds to.
        Defaults to the ``TWAVE`` value in the window metadata.
    cbar_fraction : `float`, optional
        Fraction of ``ax`` to steal for the colorbar when ``cax`` is not given.
    cbar_pad : `float`, optional
        Space between the image and the colorbar when ``cax`` is not given, as a
        fraction of the mean width of the two. The wavelength ticks and label sit
        in this gap, so too small a value pushes the label onto the image. The
        default leaves room for them itself, unless the figure has a layout
        engine that already accounts for them.
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    `matplotlib.axes.Axes`
        The axes the image was drawn on.

    Notes
    -----
    The velocity axis uses the optical Doppler convention, which is linear in
    wavelength, so it lines up exactly with the wavelength axis beside it.
    """
    if coordinates not in {"helioprojective", "time"}:
        msg = f"`coordinates` must be 'helioprojective' or 'time', got {coordinates!r}"
        raise ValueError(msg)
    rgb, (intensity, wavelength, rgb_colorbar) = calculate_rgb(
        cube,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        percentile=percentile,
    )
    (coords,) = cube.axis_world_coords("custom:pos.helioprojective.lon")
    latitude = coords.Ty.to_value(u.arcsec)
    if coordinates == "helioprojective":
        horizontal = _cell_edges(coords.Tx.to_value(u.arcsec))
        label = "Helioprojective Longitude [arcsec]"
    else:
        horizontal = _cell_edges(_time_grid(cube, latitude.shape))
        label = "Time [UTC]"
    vertical = _cell_edges(latitude)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    managed = False
    if cax is None:
        cax, managed = _colorbar_axes(ax, cbar_fraction, cbar_pad)
    ax.pcolormesh(horizontal, vertical, rgb, shading="flat", **kwargs)
    if aspect is None:
        aspect = "equal" if coordinates == "helioprojective" else "auto"
    ax.set_aspect(aspect)
    if managed:
        _match_heights(ax, cax, aspect, cbar_fraction)
    if coordinates == "time":
        ax.xaxis_date()
    ax.set_xlabel(label)
    ax.set_ylabel("Helioprojective Latitude [arcsec]")
    cax.pcolormesh(intensity, wavelength.to_value(u.AA), rgb_colorbar)
    # Show only the range that carries color; the rest of the window decodes nothing.
    limits = u.Quantity(
        [
            wavelength.min() if wavelength_min is None else wavelength_min,
            wavelength.max() if wavelength_max is None else wavelength_max,
        ]
    )
    cax.set_ylim(*limits.to_value(u.AA))
    cax.ticklabel_format(useOffset=False)
    cax.set_xlabel(f"Intensity [{cube.unit:latex_inline}]" if cube.unit is not None else "Intensity")
    cax.set_ylabel(r"Wavelength [$\mathrm{\AA}$]")
    rest_wavelength = _rest_wavelength_of(cube, rest_wavelength) if velocity else None
    if rest_wavelength is None:
        # Nothing on the right of the colorbar, so keep the wavelength labels there.
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        return ax
    doppler = u.doppler_optical(rest_wavelength)
    cax_velocity = cax.twinx()
    # The twin shares an x-axis with the colorbar, so leaving its box unpinned lets
    # it stretch to the full cell and drag the colorbar back up with it.
    cax_velocity.set_box_aspect(cax.get_box_aspect())
    cax_velocity.set_ylim(*limits.to_value(u.km / u.s, equivalencies=doppler))
    cax_velocity.ticklabel_format(useOffset=False)
    cax_velocity.set_ylabel(r"Doppler Velocity [km s$^{-1}$]")
    return ax
