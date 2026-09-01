"""
False-color (RGB) rendering of IRIS spectrogram cubes.

Each spectrum is treated as a spectral power distribution and converted to one sRGB
color, so brightness is the intensity of the window and hue is where in the window it
sits: Doppler shifts and line asymmetries show up as color.
"""

import warnings

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from irispy.utils.utils import _import_optional
from irispy.visualization import LAT_AXIS_LABEL, LON_AXIS_LABEL

__all__ = ["calculate_rgb", "plot_rgb"]


def _colorbar_axes(ax, fraction, pad):
    """
    Split the grid cell of ``ax`` so the colorbar stays under the layout engine.

    `mpl_toolkits.axes_grid1` hides the colorbar from the engine, so neighbouring panels
    overlap it; `matplotlib.colorbar.make_axes_gridspec` registers it as a colorbar of
    ``ax``, and constrained layout then moves it into a margin.
    """
    figure = ax.get_figure()
    if pad is None:
        # A layout engine measures the tick labels itself; without one the gap must hold them.
        pad = 0.08 if figure.get_layout_engine() is not None else 0.5
    subplotspec = ax.get_subplotspec()
    if subplotspec is None:
        msg = "`ax` is not in a gridspec, so there is no cell to split; pass `cax` explicitly."
        raise ValueError(msg)
    gridspec = subplotspec.subgridspec(1, 2, width_ratios=[1 - fraction, fraction], wspace=pad)
    ax.set_subplotspec(gridspec[0])
    return figure.add_subplot(gridspec[1])


def _match_heights(ax, cax, aspect, fraction):
    """
    Pin both box aspects so the colorbar keeps the image's height.

    An axes locator would do the same but is replaced when the layout engine runs.
    """
    if aspect == "auto":
        return
    scale = 1.0 if aspect == "equal" else float(aspect)
    width = abs(np.subtract(*ax.get_xlim()))
    height = abs(np.subtract(*ax.get_ylim()))
    box_aspect = scale * height / width
    ax.set_box_aspect(box_aspect)
    cax.set_box_aspect(box_aspect * (1 - fraction) / fraction)


def calculate_rgb(
    cube,
    *,
    wavelength_min=None,
    wavelength_max=None,
    vmin=None,
    vmax=None,
    stretch=None,
):
    """
    Convert the spectra in a spectrogram cube into a false-color RGB image.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        A three dimensional cube with one spectral axis.
    wavelength_min, wavelength_max : `astropy.units.Quantity`, optional
        Wavelengths mapped to the blue and red ends of the visible range. Default to
        the ends of the cube's wavelength range; wavelengths outside contribute no color.
    vmin : `float` or `astropy.units.Quantity`, optional
        Intensity mapped to black. Defaults to zero.
    vmax : `float` or `astropy.units.Quantity`, optional
        Intensity mapped to full brightness. Defaults to the 99.9th percentile of the
        finite data.
    stretch : `callable`, optional
        Applied to the intensity once scaled to [0, 1], for example `numpy.sqrt` to
        brighten faint structure.

    Returns
    -------
    rgb : `numpy.ndarray`
        Shape ``(*spatial_shape, 3)``, values in [0, 1], spatial axes in cube order.
    colorbar : `tuple`
        ``(intensity, wavelength, rgb)`` for the two dimensional colorbar, ready for
        `matplotlib.pyplot.pcolormesh`.
    """
    colorsynth = _import_optional("colorsynth", reason="false-color RGB images", extra="rgb")
    if len(cube.shape) != 3:
        msg = f"A false-color image needs a three dimensional cube, got shape {cube.shape}"
        raise ValueError(msg)
    axis_wavelength = cube.wavelength_axis
    # The mask may cover only some axes.
    mask = None if cube.mask is None else np.broadcast_to(np.asarray(cube.mask, dtype=bool), np.shape(cube.data))
    data = np.ma.filled(np.ma.MaskedArray(cube.data, mask=mask, dtype=float), np.nan)
    wavelength = u.Quantity(cube.spectral_axis).to(u.AA)
    if wavelength.ndim != 1:
        msg = f"The wavelength coordinate must be one dimensional, got shape {wavelength.shape}"
        raise ValueError(msg)
    shape = [1] * data.ndim
    shape[axis_wavelength] = wavelength.size
    wavelength = wavelength.reshape(shape)
    if isinstance(vmin, u.Quantity):
        vmin = vmin.to_value(cube.unit or u.one)
    if isinstance(vmax, u.Quantity):
        vmax = vmax.to_value(cube.unit or u.one)
    if vmin is None:
        vmin = 0
    if vmax is None:
        if not np.any(np.isfinite(data)):
            msg = "Every sample is masked or non-finite, so there is no intensity range to map"
            raise ValueError(msg)
        vmax = np.nanpercentile(data, 99.9)
    rgb, colorbar = colorsynth.rgb_and_colorbar(
        spd=data,
        wavelength=wavelength,
        axis=axis_wavelength,
        spd_min=vmin,
        spd_max=vmax,
        # Data below vmin arrives negative, which would make sqrt NaN.
        spd_norm=None if stretch is None else (lambda x: stretch(np.clip(x, 0, None))),
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
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
    stretch=None,
    coordinates="helioprojective",
    aspect=None,
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
        A three dimensional cube with one spectral axis.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on. A new figure is created if not given.
    cax : `matplotlib.axes.Axes`, optional
        Axes for the two dimensional colorbar. If not given, it is split off ``ax``,
        which must then sit in a gridspec cell, as axes from `matplotlib.pyplot.subplots` do.
    wavelength_min, wavelength_max : `astropy.units.Quantity`, optional
        Wavelengths mapped to the blue and red ends of the visible range.
    vmin, vmax : `float` or `astropy.units.Quantity`, optional
        Intensities mapped to black and to full brightness.
    stretch : `callable`, optional
        Applied to the intensity once scaled to [0, 1], for example `numpy.sqrt`.
    coordinates : ``{"helioprojective", "time"}``, optional
        Horizontal axis: longitude for a raster, or time for a sit-and-stare, whose
        longitude would only show solar rotation.
    aspect : `str` or `float`, optional
        Aspect ratio of the image axes. Defaults to ``"equal"`` for helioprojective
        coordinates and ``"auto"`` for time; use ``"auto"`` for a raster with few steps too.
    rest_wavelength : `astropy.units.Quantity` or `False`, optional
        Rest wavelength of the optical Doppler velocity axis drawn right of the colorbar.
        Defaults to ``TWAVE`` from the metadata; `False`, or no ``TWAVE``, draws no
        velocity axis.
    cbar_fraction : `float`, optional
        Fraction of ``ax`` given to the colorbar when ``cax`` is not given.
    cbar_pad : `float`, optional
        Gap between image and colorbar, as a fraction of their mean width, when ``cax``
        is not given. The wavelength tick labels sit in it; the default leaves room for them.
    **kwargs
        Passed to `matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    `matplotlib.axes.Axes`
        The axes the image was drawn on.
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
        stretch=stretch,
    )
    coords = cube.celestial
    latitude = coords.Ty.to_value(u.arcsec)
    if 1 in latitude.shape:
        msg = (
            f"The image is only one pixel wide along an axis (shape {latitude.shape}), so there is no "
            "spacing to size its cells from; use `calculate_rgb` for the colors alone."
        )
        raise ValueError(msg)
    if coordinates == "helioprojective":
        horizontal = coords.Tx.to_value(u.arcsec)
        label = LON_AXIS_LABEL
    else:
        time = matplotlib.dates.date2num(cube.time.to_datetime())
        horizontal = np.broadcast_to(time[:, np.newaxis], latitude.shape)
        label = "Time [UTC]"
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    managed = cax is None
    if managed:
        cax = _colorbar_axes(ax, cbar_fraction, cbar_pad)
    with warnings.catch_warnings():
        # Pointing jitter makes raster coordinates non-monotonic; matplotlib's warning is noise here.
        warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
        ax.pcolormesh(horizontal, latitude, rgb, shading="nearest", **kwargs)
    if aspect is None:
        aspect = "equal" if coordinates == "helioprojective" else "auto"
    ax.set_aspect(aspect)
    if managed:
        _match_heights(ax, cax, aspect, cbar_fraction)
    if coordinates == "time":
        ax.xaxis_date()
    ax.set_xlabel(label)
    ax.set_ylabel(LAT_AXIS_LABEL)
    cax.pcolormesh(intensity, wavelength.to_value(u.AA), rgb_colorbar)
    # Only the mapped range carries color.
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
    if rest_wavelength is False:
        rest_wavelength = None
    elif rest_wavelength is None:
        rest_wavelength = getattr(cube.meta, "rest_wavelength", None)
    if rest_wavelength is None:
        # No velocity axis, so the wavelength labels take the right-hand side.
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        return ax
    doppler = u.doppler_optical(u.Quantity(rest_wavelength))
    cax_velocity = cax.secondary_yaxis(
        "right",
        functions=(
            lambda wl: (wl * u.AA).to_value(u.km / u.s, equivalencies=doppler),
            lambda v: (v * u.km / u.s).to_value(u.AA, equivalencies=doppler),
        ),
    )
    cax_velocity.ticklabel_format(useOffset=False)
    cax_velocity.set_ylabel(r"Doppler Velocity [km s$^{-1}$]")
    return ax
