import matplotlib.pyplot as plt
from mpl_animators import ArrayAnimatorWCS

import astropy.units as u

import sunpy.visualization.colormaps as cm  # NOQA: F401
from ndcube.visualization.mpl_plotter import MatplotlibPlotter
from sunpy import log as logger

__all__ = ["IRISArrayAnimatorWCS", "IRISPlotter", "SJIPlotter", "SpectrogramPlotter"]


LAT_LABELS = [
    "custom:pos.helioprojective.lat",
    "helioprojective latitude",
    "hplt-tan",
    "hplt",
    "lat",
    "latitude",
]
LON_LABELS = [
    "custom:pos.helioprojective.lon",
    "helioprojective longitude",
    "hpln-tan",
    "hpln",
    "lon",
    "longitude",
]
TIME_LABEL_PRIORITY = ["seconds from start (s)", "time (utc)", "time"]
SLIDER_SCAN_STEP_LABELS = ["custom:step", "scan_step", "raster_step"]
SLIDER_SCAN_LABELS = ["custom:scan", "raster_scan"]
WAVELENGTH_LABELS = ["wavelength", "wave", "em.wl"]
LON_AXIS_LABEL = "Helioprojective Longitude [arcsec]"
LAT_AXIS_LABEL = "Helioprojective Latitude [arcsec]"
# Coordinates a plot only shows when they are asked for with ``axes_coordinates``.
HIDDEN_BY_DEFAULT = {*TIME_LABEL_PRIORITY, *SLIDER_SCAN_STEP_LABELS, *SLIDER_SCAN_LABELS}
# The pixel axis each celestial coordinate follows, which decides the edge it is labelled on.
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
    if lowered_label_parts.intersection(TIME_LABEL_PRIORITY):
        return "Time"
    if lowered_label_parts and lowered_label_parts.issubset(set(LON_LABELS + LAT_LABELS)):
        return "Spatial pixel"
    return label


def set_axis_properties(ax, axes_coordinates=None, slices=None):
    """
    Set IRIS axis labels and choose the coordinates shown on the plot edges.

    The time, raster step and raster scan coordinates are hidden unless they are in
    ``axes_coordinates``; with ``axes_coordinates``, only the requested coordinates are
    shown. Latitude is labelled on the edge of the slit axis and longitude on the edge
    of the raster step axis, the other one on the opposite edge, so a raster image is
    laid out like its FITS WCS. ``slices`` are the WCSAxes slices (``"x"``, ``"y"`` or
    an index per pixel axis, in WCS order); an animator provides its own.
    """
    pixel_axis_names = getattr(ax, "_iris_pixel_axis_names", None)
    if hasattr(ax, "axes") and not hasattr(ax, "coords"):
        slices = getattr(ax, "slices_wcsaxes", slices)
        ax = ax.axes
    requested = {coord.lower() for coord in axes_coordinates or () if isinstance(coord, str)}
    shown = []
    for axis, physical_type in _iter_coords_with_physical_types(ax):
        default_label = axis.default_label.lower()
        names = {physical_type, default_label}
        if physical_type in WAVELENGTH_LABELS or default_label in WAVELENGTH_LABELS:
            axis.set_format_unit(u.nm)
            axis.set_major_formatter("x.x")
            axis.set_axislabel("Wavelength [$\\mathrm{nm}$]")
        elif default_label == "seconds from start (s)":
            axis.set_axislabel("Seconds from Start [$\\mathrm{s}$]")
        elif physical_type in LAT_LABELS or default_label in LAT_LABELS:
            _set_axis_properties(axis, LAT_AXIS_LABEL, "red")
        elif physical_type in LON_LABELS or default_label in LON_LABELS:
            _set_axis_properties(axis, LON_AXIS_LABEL, "black")
        if names & requested if requested else names.isdisjoint(HIDDEN_BY_DEFAULT):
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


def _iter_coords_with_physical_types(ax):
    physical_types = tuple(getattr(ax.wcs, "world_axis_physical_types", ()) or ())
    for index, coord in enumerate(ax.coords):
        physical_type = physical_types[index].lower() if index < len(physical_types) and physical_types[index] else ""
        yield coord, physical_type


def _hide_coord(coord):
    """
    Hide a coordinate with fixed, empty positions.

    A hidden coordinate left in the automatic placement of WCSAxes still competes for an
    edge, and wins it on tick count, so a shown coordinate can end up on an edge where
    it has no ticks.
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
            if names & set(labels):
                celestial.append(coord)
                if pixel_axis in plotted:
                    edges[coord] = plotted[pixel_axis]
    if len(edges) == 1 and len(celestial) == 2:
        ((placed, edge),) = edges.items()
        edges[next(coord for coord in celestial if coord is not placed)] = {"b": "t", "l": "r"}[edge]
    for coord, edge in edges.items():
        _show_coord_on_edge(coord, edge)


def _show_coord_on_edge(coord, edge):
    coord.set_ticks_visible(True)
    coord.set_ticklabel_visible(True)
    coord.set_ticks_position("bt" if edge in "bt" else "lr")
    coord.set_ticklabel_position(edge)
    coord.set_axislabel_position(edge)


class Plot2DMixin:
    def update_plot_2d(self, val, im, slider):
        super().update_plot_2d(val, im, slider)
        set_axis_properties(self, getattr(self, "_iris_axes_coordinates", None))


class IRISArrayAnimatorWCS(Plot2DMixin, ArrayAnimatorWCS):
    def _compute_slider_labels_from_wcs(self, slices):
        return [_shorten_slider_label(label) for label in super()._compute_slider_labels_from_wcs(slices)]


def _wcs_order_slices(plot_axes, naxis):
    """
    The WCSAxes ``slices`` that ndcube derives from ``plot_axes`` (in array order).
    """
    axes = list(plot_axes) if isinstance(plot_axes, (list, tuple)) else [plot_axes] if plot_axes else [..., "y", "x"]
    if Ellipsis in axes:
        at = axes.index(Ellipsis)
        axes[at : at + 1] = [None] * (naxis - len(axes) + 1)
    return axes[::-1]


class IRISPlotter(MatplotlibPlotter):
    def _default_cmap_name(self):
        return "viridis"

    def plot(self, axes=None, plot_axes=None, axes_coordinates=None, **kwargs):
        """
        Plot the cube with IRIS defaults.

        For images and animations, a falsey ``cmap`` is replaced with a default IRIS
        colormap derived from the cube metadata (falling back to viridis), and
        ``interpolation`` defaults to ``"nearest"``. For one-dimensional cubes ``cmap``
        is dropped entirely, even when given. IRIS axis styling is applied to the result
        (see ``set_axis_properties``); all other arguments are passed to the parent
        plotter's ``plot``.
        """
        if len(self._ndcube.shape) == 1:
            kwargs.pop("cmap", None)
        else:
            if not kwargs.get("cmap"):
                try:
                    kwargs["cmap"] = plt.get_cmap(name=self._default_cmap_name())
                except Exception as e:  # NOQA: BLE001
                    logger.debug(e)
                    kwargs["cmap"] = "viridis"
            kwargs.setdefault("interpolation", "nearest")
        ax = super().plot(axes=axes, plot_axes=plot_axes, axes_coordinates=axes_coordinates, **kwargs)
        ax._iris_axes_coordinates = axes_coordinates
        # With axes_coordinates, ndcube plots through its combined WCS, whose pixel axis
        # names are unusable (ndcube's CompoundLowLevelWCS raises), so keep the cube's own.
        ax._iris_pixel_axis_names = tuple(self._ndcube.wcs.low_level_wcs.pixel_axis_names)
        set_axis_properties(ax, axes_coordinates, slices=_wcs_order_slices(plot_axes, len(self._ndcube.shape)))
        return ax

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


class SpectrogramPlotter(IRISPlotter):
    def plot_rgb(self, **kwargs):
        """
        Plot the cube as a false-color image, coloring each pixel by its spectrum.

        Keyword arguments and the return value are those of `irispy.utils.rgb.plot_rgb`,
        which needs the optional ``colorsynth`` dependency.
        """
        from irispy.utils.rgb import plot_rgb  # NOQA: PLC0415

        return plot_rgb(self._ndcube, **kwargs)


class SJIPlotter(IRISPlotter):
    def _default_cmap_name(self):
        return f"irissji{int(self._ndcube.meta['TWAVE1'])}"
