"""
False-color (RGB) rendering of IRIS spectrogram cubes.

Each spectrum is treated as a spectral power distribution and converted to one sRGB
color. Brightness and hue depend on both intensity and spectral shape, so Doppler shifts
and line asymmetries show up as color.
"""

import warnings

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from irispy.spectrograph import _wavelength_indices
from irispy.utils._spectral import _to_unit, check_scaled, check_velocity_range, default_rest_wavelength
from irispy.utils.utils import _boolean_mask, _import_optional
from irispy.visualization import LAT_AXIS_LABEL, LON_AXIS_LABEL

__all__ = ["calculate_rgb", "plot_rgb"]


def _colorbar_axes(ax, fraction, pad):
    """
    Split the grid cell of ``ax`` so the colorbar stays under the layout engine.

    `mpl_toolkits.axes_grid1` hides the colorbar from the engine, so neighboring panels
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


def _wavelength_limits(wavelength, rest_wavelength, velocity_range):
    """
    The wavelengths ``velocity_range`` from ``rest_wavelength`` maps to the ends of the
    visible range, or the ends of ``wavelength`` if it is `None`.
    """
    if rest_wavelength is None:
        msg = "The cube metadata has no usable TWAVE value. Pass rest_wavelength explicitly."
        raise ValueError(msg)
    wavelength = u.Quantity(wavelength).to(u.AA)
    if velocity_range is None:
        return wavelength.min(), wavelength.max()
    check_velocity_range(velocity_range)
    lower, upper = velocity_range.to(u.AA, equivalencies=u.doppler_optical(rest_wavelength))
    if not _wavelength_indices(wavelength, (lower, upper), allow_empty=True).size:
        msg = (
            f"No wavelength of the window, {wavelength.min():.3f} to {wavelength.max():.3f}, lies within "
            f"velocity_range={velocity_range} of rest_wavelength={rest_wavelength.to(u.AA):.3f}; "
            "pass velocity_range=None to map the whole window"
        )
        raise ValueError(msg)
    return lower, upper


@u.quantity_input
def calculate_rgb(
    cube,
    *,
    velocity_range: u.Quantity[u.km / u.s] | None = (-100, 100) * u.km / u.s,
    rest_wavelength: u.AA = None,
    velocity_norm=None,
    vmin=None,
    vmax=None,
    stretch=None,
):
    """
    Convert the spectra in a spectrogram cube into a false-color RGB image.

    Wavelengths are converted to optical Doppler velocities from the rest wavelength
    and transformed by ``velocity_norm`` before they are mapped to the visible range.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        A three dimensional cube with one spectral axis, such as one raster, ``raster[window][0]``.
    velocity_range : `astropy.units.Quantity` or `None`, optional
        ``(lower, upper)`` Doppler velocities from ``rest_wavelength`` mapped to the blue
        and red ends of the visible range, or `None` to map the whole window. Wavelengths
        outside contribute no color. A range with no wavelength of the window raises a
        `ValueError`. Defaults to ``(-100, 100) * u.km / u.s``.
    rest_wavelength : `astropy.units.Quantity`, optional
        Rest wavelength the Doppler velocities are measured from. Defaults to
        ``cube.meta.rest_wavelength``, or to the one documented line the cube covers
        (`~irispy.utils.lines.get_lines`) if it does not cover ``cube.meta.rest_wavelength``;
        a blend of lines within 0.1 Å of each other counts as its strongest line.
        Required if there is neither.
    velocity_norm : `callable`, optional
        Monotonically increasing transform applied to velocity quantities, arrays and
        scalars alike, returning plain numbers. Defaults to a linear mapping of the
        velocity. For example, ``lambda v: np.arcsinh(v.to_value(u.km / u.s) / 25)``
        makes small shifts stand out while compressing large ones.
    vmin : `astropy.units.Quantity`, optional
        Intensity mapped to black. Defaults to zero. ``u.DN`` stands for the data's DN.
    vmax : `astropy.units.Quantity`, optional
        Intensity mapped to full brightness. Defaults to the 99.9th percentile of the
        finite data. ``u.DN`` stands for the data's DN.
    stretch : `callable`, optional
        Applied to the intensity once scaled to [0, 1], for example `numpy.sqrt` to
        brighten faint structure.

    Returns
    -------
    rgb : `numpy.ndarray`
        Shape ``(*spatial_shape, 3)``, values in [0, 1], spatial axes in cube order.
        NaN for a spectrum with no unmasked, finite sample within ``velocity_range``,
        which `plot_rgb` leaves blank.
    colorbar : `tuple`
        ``(intensity, wavelength, rgb)`` for the two dimensional colorbar, ready for
        `matplotlib.pyplot.pcolormesh`. Each entry represents a spectrum with signal
        in one wavelength bin. Finite-width lines can have different colors at the
        same peak intensity; this is not a calibration of line intensity or velocity.
    """
    check_scaled(cube)
    colorsynth = _import_optional("colorsynth", reason="false-color RGB images", extra="rgb")
    if len(cube.shape) != 3:
        msg = f"A false-color image needs a three dimensional cube, got shape {cube.shape}"
        raise ValueError(msg)
    axis_wavelength = cube.wavelength_axis
    data = np.ma.filled(np.ma.MaskedArray(cube.data, mask=_boolean_mask(cube), dtype=float), np.nan)
    data[np.isinf(data)] = np.nan  # blank like a masked sample
    wavelength = u.Quantity(cube.spectral_axis).to(u.AA)
    if wavelength.ndim != 1:
        msg = f"The wavelength coordinate must be one dimensional, got shape {wavelength.shape}"
        raise ValueError(msg)
    if rest_wavelength is None:
        rest_wavelength = default_rest_wavelength(cube)
    wavelength_min, wavelength_max = _wavelength_limits(wavelength, rest_wavelength, velocity_range)
    mapped = _wavelength_indices(wavelength, (wavelength_min, wavelength_max))
    mapped_slice = [slice(None)] * data.ndim
    mapped_slice[axis_wavelength] = slice(mapped[0], mapped[-1] + 1)
    no_data = ~np.isfinite(data[tuple(mapped_slice)]).any(axis=axis_wavelength)
    doppler = u.doppler_optical(rest_wavelength)

    def wavelength_norm(w):
        # A SpectralCoord will not convert to velocity without its own Doppler settings.
        return velocity_norm(u.Quantity(w).to(u.km / u.s, equivalencies=doppler))

    shape = [1] * data.ndim
    shape[axis_wavelength] = wavelength.size
    wavelength = wavelength.reshape(shape)
    vmin = 0 if vmin is None else _to_unit(vmin, cube.unit, "vmin").value
    if vmax is None:
        if not np.any(np.isfinite(data)):
            msg = "Every sample is masked or non-finite, so there is no intensity range to map"
            raise ValueError(msg)
        vmax = np.nanpercentile(data, 99.9)
    else:
        vmax = _to_unit(vmax, cube.unit, "vmax").value
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
        wavelength_norm=None if velocity_norm is None else wavelength_norm,
    )
    rgb = np.moveaxis(rgb, axis_wavelength, -1)
    rgb[no_data] = np.nan
    return rgb, colorbar


def plot_rgb(
    cube,
    *,
    ax=None,
    cax=None,
    velocity_range=(-100, 100) * u.km / u.s,
    velocity_norm=None,
    vmin=None,
    vmax=None,
    stretch=None,
    coordinates=None,
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
        A three dimensional cube with one spectral axis, such as one raster, ``raster[window][0]``.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on. A new figure is created if not given.
    cax : `matplotlib.axes.Axes`, optional
        Axes for the two dimensional colorbar. If not given, it is split off ``ax``,
        which must then sit in a gridspec cell, as axes from `matplotlib.pyplot.subplots` do.
    velocity_range : `astropy.units.Quantity` or `None`, optional
        Doppler velocities mapped to the blue and red ends of the visible range, or `None`
        to map the whole window, as in `calculate_rgb`.
    velocity_norm : `callable`, optional
        Transform of the Doppler velocity, as in `calculate_rgb`. Defaults to a linear
        mapping of the velocity.
    vmin, vmax : `astropy.units.Quantity`, optional
        Intensities mapped to black and to full brightness, as in `calculate_rgb`.
    stretch : `callable`, optional
        Applied to the intensity once scaled to [0, 1], for example `numpy.sqrt`.
    coordinates : ``{"helioprojective", "time"}`` or `None`, optional
        Horizontal axis: longitude, or time. Defaults to time for a sit-and-stare
        (``NRASTERP`` of 1), whose longitude would only show solar rotation, and to
        longitude otherwise.
    aspect : `str` or `float`, optional
        Aspect ratio of the image axes. Defaults to ``"equal"`` for helioprojective
        coordinates and ``"auto"`` for time; use ``"auto"`` for a raster with few steps too.
    rest_wavelength : `astropy.units.Quantity` or `False`, optional
        Rest wavelength for the color mapping and the optical Doppler velocity axis,
        as in `calculate_rgb`, whose default it shares. `False` hides the velocity axis while
        keeping the default color mapping.
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

    Notes
    -----
    The colorbar shows spectra with signal in a single wavelength bin. The image
    combines signal across the line profile, so its color also depends on line width
    and asymmetry. The colorbar cannot uniquely determine a line's intensity or velocity.
    """
    if coordinates is None:
        coordinates = "time" if (cube.meta or {}).get("NRASTERP") == 1 else "helioprojective"
    if coordinates not in {"helioprojective", "time"}:
        msg = f"`coordinates` must be 'helioprojective' or 'time', got {coordinates!r}"
        raise ValueError(msg)
    mapping_rest = (
        default_rest_wavelength(cube) if rest_wavelength is None or rest_wavelength is False else rest_wavelength
    )
    # calculate_rgb checks the units of the arguments, so it comes before _wavelength_limits
    rgb, (intensity, wavelength, rgb_colorbar) = calculate_rgb(
        cube,
        velocity_range=velocity_range,
        rest_wavelength=mapping_rest,
        velocity_norm=velocity_norm,
        vmin=vmin,
        vmax=vmax,
        stretch=stretch,
    )
    wavelength_min, wavelength_max = _wavelength_limits(cube.spectral_axis, mapping_rest, velocity_range)
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
    if managed and aspect != "auto":
        # Pin both box aspects so the colorbar keeps the image's height; an axes
        # locator would do the same but is replaced when the layout engine runs.
        scale = 1.0 if aspect == "equal" else float(aspect)
        width = abs(np.subtract(*ax.get_xlim()))
        height = abs(np.subtract(*ax.get_ylim()))
        box_aspect = scale * height / width
        ax.set_box_aspect(box_aspect)
        cax.set_box_aspect(box_aspect * (1 - cbar_fraction) / cbar_fraction)
    if coordinates == "time":
        locator = matplotlib.dates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(locator))
    ax.set_xlabel(label)
    ax.set_ylabel(LAT_AXIS_LABEL)
    cax.pcolormesh(intensity, wavelength.to_value(u.AA), rgb_colorbar)
    # Only the mapped range carries color.
    cax.set_ylim(wavelength_min.to_value(u.AA), wavelength_max.to_value(u.AA))
    cax.ticklabel_format(useOffset=False)
    cax.set_xlabel(
        f"Single-bin\nintensity [{cube.unit:latex_inline}]" if cube.unit is not None else "Single-bin\nintensity"
    )
    cax.set_ylabel(r"Wavelength [$\mathrm{\AA}$]")
    if rest_wavelength is False:
        # No velocity axis, so the wavelength labels take the right-hand side.
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        return ax
    doppler = u.doppler_optical(mapping_rest)
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
