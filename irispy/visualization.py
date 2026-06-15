from mpl_animators import ArrayAnimatorWCS

import astropy.units as u

import sunpy.visualization.colormaps as cm  # NOQA: F401
from ndcube.visualization.mpl_plotter import MatplotlibPlotter

__all__ = ["IRISArrayAnimatorWCS", "IRISPlotter", "finalize_iris_plot"]


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
TIME_OFFSET_LABELS = ["seconds from start (s)"]
SLIDER_SCAN_STEP_LABELS = ["custom:step", "scan_step", "raster_step"]
SLIDER_SCAN_LABELS = ["custom:scan", "raster_scan"]
WAVELENGTH_LABELS = ["wavelength", "wave", "em.wl"]


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
        return "Scan"
    return label


def set_axis_properties(ax):
    """
    Set IRIS axis labels.
    """
    if hasattr(ax, "axes") and not hasattr(ax, "coords"):
        ax = ax.axes
    for axis, physical_type in _iter_coords_with_physical_types(ax):
        default_label = axis.default_label.lower()
        if physical_type in WAVELENGTH_LABELS or default_label in WAVELENGTH_LABELS:
            axis.set_format_unit(u.nm)
            axis.set_major_formatter("x.x")
            axis.set_axislabel("Wavelength [$\\mathrm{nm}$]")
        elif default_label in TIME_OFFSET_LABELS:
            axis.set_axislabel("Seconds from Start [$\\mathrm{s}$]")
        elif physical_type in LAT_LABELS or default_label in LAT_LABELS:
            _set_axis_properties(axis, "Helioprojective Latitude [arcsec]", "red")
        elif physical_type in LON_LABELS or default_label in LON_LABELS:
            _set_axis_properties(axis, "Helioprojective Longitude [arcsec]", "black")


def finalize_iris_plot(ax, axes_coordinates=None):
    """
    Apply IRIS-specific axis formatting.
    """
    ax._iris_axes_coordinates = axes_coordinates
    set_axis_properties(ax)
    _set_requested_axis_positions(ax, axes_coordinates)
    return ax


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


def _get_coord(ax, labels):
    for coord, physical_type in _iter_coords_with_physical_types(ax):
        if physical_type in labels or coord.default_label.lower() in labels:
            return coord
    return None


def _hide_coord(coord):
    coord.set_ticks_visible(False)
    coord.set_ticklabel_visible(False)
    coord.set_axislabel("")


def _show_coord_on_edge(coord, edge):
    coord.set_ticks_visible(True)
    coord.set_ticklabel_visible(True)
    coord.set_ticks_position(edge)
    coord.set_ticklabel_position(edge)
    coord.set_axislabel_position(edge)


def _set_requested_axis_positions(ax, axes_coordinates):
    requested = {coord.lower() for coord in axes_coordinates or () if isinstance(coord, str)}
    if not requested.intersection(LON_LABELS):
        return

    if hasattr(ax, "axes") and not hasattr(ax, "coords"):
        ax = ax.axes

    for labels in (TIME_LABEL_PRIORITY, SLIDER_SCAN_STEP_LABELS, SLIDER_SCAN_LABELS):
        coord = _get_coord(ax, labels)
        if coord is not None and not requested.intersection(labels):
            _hide_coord(coord)

    lon = _get_coord(ax, LON_LABELS)
    if lon is not None:
        _show_coord_on_edge(lon, "b")

    if requested.intersection(LAT_LABELS):
        lat = _get_coord(ax, LAT_LABELS)
        if lat is not None:
            _show_coord_on_edge(lat, "l")


class Plot2DMixin:
    def update_plot_2d(self, val, im, slider):
        super().update_plot_2d(val, im, slider)
        set_axis_properties(self.axes)
        _set_requested_axis_positions(self.axes, getattr(self, "_iris_axes_coordinates", None))


class IRISArrayAnimatorWCS(Plot2DMixin, ArrayAnimatorWCS):
    def _compute_slider_labels_from_wcs(self, slices):
        return [_shorten_slider_label(label) for label in super()._compute_slider_labels_from_wcs(slices)]


class IRISPlotter(MatplotlibPlotter):
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
