import matplotlib as mpl
import numpy as np
from mpl_animators import ArrayAnimatorWCS

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsymmetricPercentileInterval
from astropy.wcs.wcsapi import HighLevelWCSWrapper

import sunpy.visualization.colormaps as cm  # NOQA: F401
from ndcube.visualization.mpl_plotter import MatplotlibPlotter
from ndcube.visualization.plotting_utils import prep_plot_kwargs

from irispy._wcs import _is_sit_and_stare

__all__ = []


LAT_LABELS = {
    "custom:pos.helioprojective.lat",
    "helioprojective latitude",
    "hplt-tan",
    "hplt",
    "lat",
    "latitude",
}
LON_LABELS = {
    "custom:pos.helioprojective.lon",
    "helioprojective longitude",
    "hpln-tan",
    "hpln",
    "lon",
    "longitude",
}
TIME_LABELS = {"seconds from start (s)", "time (utc)", "time"}
SLIDER_SCAN_STEP_LABELS = {"custom:step", "raster_step"}
SLIDER_SCAN_LABELS = {"custom:scan", "raster_scan"}
WAVELENGTH_LABELS = {"wavelength", "wave", "em.wl"}
LON_AXIS_LABEL = "Helioprojective Longitude [arcsec]"
LAT_AXIS_LABEL = "Helioprojective Latitude [arcsec]"
# Coordinates a plot only shows when they are asked for with ``axes_coordinates``.
HIDDEN_BY_DEFAULT = TIME_LABELS | SLIDER_SCAN_STEP_LABELS | SLIDER_SCAN_LABELS
# The pixel axis each celestial coordinate follows, which decides the edge it is labeled on.
PRIMARY_PIXEL_AXES = {"spatial along slit": LAT_LABELS, "raster step": LON_LABELS}


def _shorten_slider_label(label):
    """
    Return a concise label for common IRIS animation sliders.
    """
    lowered_label_parts = {label_part.strip().lower() for label_part in str(label).split(" / ")}
    if lowered_label_parts.intersection(SLIDER_SCAN_STEP_LABELS):
        return "Raster step"
    if lowered_label_parts.intersection(SLIDER_SCAN_LABELS):
        return "Scan number"
    if lowered_label_parts.intersection(WAVELENGTH_LABELS):
        return "Wavelength"
    if lowered_label_parts.intersection(TIME_LABELS):
        return "Time"
    if lowered_label_parts and lowered_label_parts <= LON_LABELS | LAT_LABELS:
        return "Spatial pixel"
    return label


def set_axis_properties(ax, axes_coordinates=None, slices=None):
    """
    Set IRIS axis labels and choose the coordinates shown on the plot edges.

    The raster step and raster scan coordinates, and the time of a raster, are hidden
    unless they are in ``axes_coordinates``. With ``axes_coordinates``, only the
    requested coordinates are shown. Latitude is labeled on the edge of the slit axis
    and longitude on the edge of the raster step axis. When only one of those axes is
    plotted, the other coordinate goes on its opposite edge, so a raster image is laid
    out like its FITS WCS. ``slices`` are the WCSAxes slices (``"x"``, ``"y"`` or an
    index per pixel axis, in WCS order); an animator provides its own.
    """
    # IRISPlotter.plot sets this on the axes or animator it returns
    pixel_axis_names = getattr(ax, "_iris_pixel_axis_names", None)
    if isinstance(ax, ArrayAnimatorWCS):
        slices = ax.slices_wcsaxes
        ax = ax.axes
    physical_types = [(physical_type or "").lower() for physical_type in ax.wcs.world_axis_physical_types]
    # The time of a raster exposure follows its raster step, so a raster shows it only when asked for, while an
    # image along the time of a slit-jaw cube always does
    is_raster = not (SLIDER_SCAN_STEP_LABELS | SLIDER_SCAN_LABELS).isdisjoint(physical_types)
    hidden = HIDDEN_BY_DEFAULT if is_raster else HIDDEN_BY_DEFAULT - TIME_LABELS
    requested = {coord.lower() for coord in axes_coordinates or () if isinstance(coord, str)}
    shown = []
    for axis, physical_type in zip(ax.coords, physical_types, strict=True):
        names = {physical_type, axis.default_label.lower()}
        if names & WAVELENGTH_LABELS:
            axis.set_format_unit(u.nm)
            axis.set_major_formatter("x.xx")
            # A narrow window drawn with equal aspect has no room to label every tick
            axis.set_ticklabel(exclude_overlapping=True)
            axis.set_axislabel("Wavelength [$\\mathrm{nm}$]")
        elif names & LAT_LABELS:
            _set_axis_properties(axis, LAT_AXIS_LABEL, "red")
        elif names & LON_LABELS:
            _set_axis_properties(axis, LON_AXIS_LABEL, "black")
        visible = bool(names & requested) if requested else names.isdisjoint(hidden)
        if visible:
            shown.append((axis, names))
        else:
            _hide_coord(axis)
    if "l" in ax.coords.frame.spine_names:
        _place_celestial_coords(ax, shown, slices, pixel_axis_names)


def _set_axis_properties(axis, label, color):
    """
    Set the axis colors and labels for IRIS SJI and Raster data.

    Parameters
    ----------
    axis : `~astropy.visualization.wcsaxes.core.WCSAxes`
        The axis to set the colors and labels for.
    label : str
        The label to use for the axis.
    color : str
        The color to use for the axis label.
    """
    axis.set_ticklabel(color, fontsize=8)
    axis.set_axislabel(label, color=color, fontsize=8)


def _hide_coord(coord):
    """
    Hide a coordinate with fixed, empty positions.

    A hidden coordinate left in the automatic placement of WCSAxes still competes for an
    edge, and wins it on tick count, so a shown coordinate can end up on an edge where
    it has no ticks.

    TODO: drop the position reset once irispy requires an astropy release with
    astropy/astropy#20499, which skips fully hidden coordinates in that placement.
    """
    coord.set_ticks_visible(False)
    coord.set_ticklabel_visible(False)
    for set_position in (coord.set_ticks_position, coord.set_ticklabel_position, coord.set_axislabel_position):
        set_position("")


def _place_celestial_coords(ax, shown, slices, pixel_axis_names=None):
    if pixel_axis_names is None:
        pixel_axis_names = getattr(ax.wcs, "pixel_axis_names", None)
    pixel_axis_names = list(pixel_axis_names or ())
    slices = list(("x", "y") if slices is None else slices)
    if len(slices) != len(pixel_axis_names):
        return
    plotted = {pixel_axis_names[slices.index(axis)]: edge for axis, edge in (("x", "b"), ("y", "l")) if axis in slices}
    edges = {}
    celestial = []
    for coord, names in shown:
        for pixel_axis, labels in PRIMARY_PIXEL_AXES.items():
            if names & labels:
                celestial.append(coord)
                if pixel_axis in plotted:
                    edges[coord] = plotted[pixel_axis]
    # When both celestial coordinates follow one image axis, they take its two opposite edges, with ticks on their own
    # edge only
    opposite = len(edges) == 1 and len(celestial) == 2
    if opposite:
        ((placed, edge),) = edges.items()
        edges[next(coord for coord in celestial if coord is not placed)] = {"b": "t", "l": "r"}[edge]
    for coord, edge in edges.items():
        _show_coord_on_edge(coord, edge, ticks=edge if opposite else None)


def _show_coord_on_edge(coord, edge, ticks=None):
    coord.set_ticks_visible(True)
    coord.set_ticklabel_visible(True)
    coord.set_ticks_position(ticks or ("bt" if edge in "bt" else "lr"))
    coord.set_ticklabel_position(edge)
    coord.set_axislabel_position(edge)


def _map_aspect(cube, wcs, plot_axes):
    """
    The ``aspect`` that draws a 2D map with its true proportions, ``"auto"`` for an
    image along the raster axis and `None` for other images.

    ``plot_axes`` is in pixel order, as `ndcube` passes it to ``_plot_2D_cube``.
    """
    # The rasters are times, often only a few, so square pixels would draw a sliver
    if "raster scan" in cube.wcs.low_level_wcs.pixel_axis_names:
        return "auto"
    types = set(wcs.world_axis_physical_types)
    # A raster's gWCS also gives each exposure's time and step, which follow the step axis
    spatial = {"custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"} <= types and "em.wl" not in types
    # The step axis of a sit-and-stare is time, along which the slit only follows solar rotation
    if not spatial or _is_sit_and_stare(cube):
        return None
    nx, ny = cube.data.shape[::-1]
    wcs = HighLevelWCSWrapper(wcs)
    scale = []
    # The mean pixel spacing across the middle of the map, which pointing jitter barely changes. Steps
    # without an exposure have no pointing, so measure between the finite ends of each line.
    for pixels in ((np.arange(nx), np.full(nx, ny // 2)), (np.full(ny, nx // 2), np.arange(ny))):
        line = wcs.pixel_to_world(*pixels)
        # A raster's gWCS also gives the time and step of each pixel
        line = line if isinstance(line, SkyCoord) else next(coord for coord in line if isinstance(coord, SkyCoord))
        finite = np.flatnonzero(np.isfinite(line.Tx))
        if finite.size < 2:
            return None
        scale.append(line[finite[0]].separation(line[finite[-1]]) / (finite[-1] - finite[0]))
    return (scale[plot_axes.index("y")] / scale[plot_axes.index("x")]).to_value(u.one)


class Plot2DMixin:
    def update_plot_2d(self, val, im, slider):
        super().update_plot_2d(val, im, slider)
        set_axis_properties(self, getattr(self, "_iris_axes_coordinates", None))


class IRISArrayAnimatorWCS(Plot2DMixin, ArrayAnimatorWCS):
    def _compute_slider_labels_from_wcs(self, slices):
        return [_shorten_slider_label(label) for label in super()._compute_slider_labels_from_wcs(slices)]


def _registered_colormap(name):
    """
    ``name`` when matplotlib knows that colormap, viridis otherwise.
    """
    return name if name in mpl.colormaps else "viridis"


def _make_map(data, header, unit, mask):
    """
    A `sunpy.map.Map` of ``data`` and ``mask`` with a copy of ``header`` that gives
    ``unit`` as its BUNIT.
    """
    from sunpy.map import Map  # NOQA: PLC0415

    # The IRIS DN units, such as DN_IRIS_SJI_1330, are not FITS units; sunpy reads DN
    bases = [u.DN if base.name.startswith("DN_IRIS") else base for base in unit.bases]
    header = header.copy()
    header["bunit"] = u.CompositeUnit(unit.scale, bases, unit.powers).to_string()
    return Map(data, header, mask=None if mask is None else np.asarray(mask))


def _sji_cmap_name(meta):
    """
    The sunpy color table of a slit-jaw or AIA cube's passband or of a Hinode/SOT cube,
    viridis without one.
    """
    # Hinode/SOT cubes have TWAVE1 = 0. sunpy's SOTMap plots with hinodesotintensity, but sunpy makes
    # GenericMaps of the broadband ("SOT") and SP ("SOT-SP") files.
    if meta.get("INSTRUME", "").startswith("SOT"):
        return "hinodesotintensity"
    # A cube built by hand may have no TWAVE1
    twave = meta.get("TWAVE1")
    if not twave:
        return "viridis"
    # sunpy registers the AIA tables as sdoaia<channel>. Its AIA maps build that name from TELESCOP,
    # which is blank in the AIA files supplied with IRIS, and ask for the unregistered aia<channel>.
    prefix = "sdoaia" if meta.get("INSTRUME", "").startswith("AIA") else "irissji"
    return _registered_colormap(f"{prefix}{int(twave)}")


class IRISPlotter(MatplotlibPlotter):
    def _default_cmap_name(self):
        # sunpy registers the IRIS spectrograph color tables as irissjiFUV and irissjiNUV
        return _registered_colormap(f"irissji{getattr(self._ndcube.meta, 'detector_band', '')}")

    def plot(self, axes=None, plot_axes=None, axes_coordinates=None, **kwargs):
        """
        Plot the cube with IRIS defaults.

        For images and animations, if ``cmap`` is falsey, the color table of the data is
        used: the IRIS passband table for slit-jaw images, the sunpy ``sdoaia`` table of
        the channel for AIA cubes, the sunpy ``hinodesotintensity`` table for Hinode/SOT
        cubes and the IRIS FUV or NUV table for spectrograms. For one-dimensional cubes
        ``cmap`` is dropped entirely, even when given. IRIS axis styling is applied to the
        result (see ``set_axis_properties``); all other arguments are passed to the parent
        plotter's ``plot``.

        A 2D map, such as a cut of a scanning raster at one wavelength, a moment map or
        a Hinode/SOT-SP scan, is drawn with its true proportions unless ``aspect`` is
        given. Maps of a sit-and-stare keep square pixels, and an image along the raster axis
        fills the axes. A raster map has its raster steps,
        or the exposures of a sit-and-stare, along x unless ``plot_axes`` is given.

        By default the colors run linearly from the minimum to the maximum of the image,
        or of the first frame of an animation, which the other frames keep, so a few
        bright pixels, such as cosmic-ray hits, can leave little contrast.

        ``clip_interval=(1, 99.5) * u.percent`` sets them to those percentiles of the
        image, or of each frame.
        """
        if plot_axes is None and list(self._ndcube.wcs.low_level_wcs.pixel_axis_names) == [
            "spatial along slit",
            "raster step",
        ]:
            # Solar X along x, as in a map; plot_axes is in array order, (step, slit)
            plot_axes = ["x", "y"]
        if len(self._ndcube.shape) == 1:
            kwargs.pop("cmap", None)
        else:
            kwargs["cmap"] = kwargs.get("cmap") or self._default_cmap_name()
        ax = super().plot(axes=axes, plot_axes=plot_axes, axes_coordinates=axes_coordinates, **kwargs)
        ax._iris_axes_coordinates = axes_coordinates
        # With axes_coordinates, ndcube plots through its combined WCS, whose pixel axis
        # names are unusable (ndcube's CompoundLowLevelWCS raises), so keep the cube's own.
        ax._iris_pixel_axis_names = tuple(self._ndcube.wcs.low_level_wcs.pixel_axis_names)
        slices = prep_plot_kwargs(len(self._ndcube.shape), self._ndcube.wcs, plot_axes, None, None)[0]
        set_axis_properties(ax, axes_coordinates, slices=slices)
        return ax

    def _plot_2D_cube(  # NOQA: N802
        self, wcs, axes=None, plot_axes=None, axes_coordinates=None, axes_units=None, data_unit=None, **kwargs
    ):
        if "aspect" not in kwargs:
            kwargs["aspect"] = _map_aspect(self._ndcube, wcs, plot_axes)
        # ndcube passes clip_interval on to imshow, which rejects it; remove this once the minimum
        # ndcube applies it to 2D cubes, as its animations do.
        if (clip_interval := kwargs.pop("clip_interval", None)) is not None:
            if len(clip_interval) != 2:
                msg = "A range of 2 values must be specified for clip_interval."
                raise ValueError(msg)
            data = self._ndcube.data if data_unit is None else self._ndcube.quantity.to_value(data_unit)
            interval = AsymmetricPercentileInterval(*clip_interval.to_value(u.percent))
            kwargs["vmin"], kwargs["vmax"] = interval.get_limits(np.ma.masked_array(data, self._ndcube.mask))
        return super()._plot_2D_cube(wcs, axes, plot_axes, axes_coordinates, axes_units, data_unit, **kwargs)

    def _animate_cube(
        self,
        wcs,
        plot_axes=None,
        axes_coordinates=None,
        axes_units=None,
        data_unit=None,
        **kwargs,
    ):
        data, wcs, plot_axes, coord_params = self._prep_animate_args(wcs, plot_axes, axes_units, data_unit)
        ax = IRISArrayAnimatorWCS(data, wcs, plot_axes, coord_params=coord_params, **kwargs)
        self._apply_axes_coordinates(ax.axes, axes_coordinates)
        for hidden in self._not_visible_coords(ax.axes, axes_coordinates):
            param = ax.coord_params.get(hidden, {})
            param["ticks"] = False
            ax.coord_params[hidden] = param
        return ax


class SJIPlotter(IRISPlotter):
    def _default_cmap_name(self):
        return _sji_cmap_name(self._ndcube.meta)
