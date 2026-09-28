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


def set_axis_properties(ax, axes_coordinates=None):
    """
    Set IRIS axis labels, and move a requested longitude (and latitude) to the edges.
    """
    if hasattr(ax, "axes") and not hasattr(ax, "coords"):
        ax = ax.axes
    for axis, physical_type in _iter_coords_with_physical_types(ax):
        default_label = axis.default_label.lower()
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

    requested = {coord.lower() for coord in axes_coordinates or () if isinstance(coord, str)}
    if not requested.intersection(LON_LABELS):
        return
    for labels in (TIME_LABEL_PRIORITY, SLIDER_SCAN_STEP_LABELS, SLIDER_SCAN_LABELS):
        coord = _get_coord(ax, labels)
        if coord is not None and not requested.intersection(labels):
            coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
            coord.set_axislabel("")
    lon = _get_coord(ax, LON_LABELS)
    if lon is not None:
        _show_coord_on_edge(lon, "b")
    if requested.intersection(LAT_LABELS):
        lat = _get_coord(ax, LAT_LABELS)
        if lat is not None:
            _show_coord_on_edge(lat, "l")


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


def _show_coord_on_edge(coord, edge):
    coord.set_ticks_visible(True)
    coord.set_ticklabel_visible(True)
    coord.set_ticks_position(edge)
    coord.set_ticklabel_position(edge)
    coord.set_axislabel_position(edge)


class Plot2DMixin:
    def update_plot_2d(self, val, im, slider):
        super().update_plot_2d(val, im, slider)
        set_axis_properties(self.axes, getattr(self, "_iris_axes_coordinates", None))


class IRISArrayAnimatorWCS(Plot2DMixin, ArrayAnimatorWCS):
    def _compute_slider_labels_from_wcs(self, slices):
        return [_shorten_slider_label(label) for label in super()._compute_slider_labels_from_wcs(slices)]


class IRISPlotter(MatplotlibPlotter):
    def _default_cmap_name(self):
        return "viridis"

    def plot(self, *args, **kwargs):
        """
        Plot the cube with IRIS defaults.

        For images and animations, a falsey ``cmap`` is replaced with a default IRIS
        colormap derived from the cube metadata (falling back to viridis), and
        ``interpolation`` defaults to ``"nearest"``. For one-dimensional cubes ``cmap``
        is dropped entirely, even when given. IRIS axis styling is applied to the
        result; all other arguments are passed to the parent plotter's ``plot``.
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
        ax = super().plot(*args, **kwargs)
        ax._iris_axes_coordinates = kwargs.get("axes_coordinates")
        set_axis_properties(ax, ax._iris_axes_coordinates)
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
