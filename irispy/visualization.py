import warnings
from contextlib import contextmanager

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
SCAN_STEP_LABELS = ["custom:step", "scan_step"]
WAVELENGTH_LABELS = ["wavelength", "wave", "em.wl"]


@contextmanager
def _suppress_wcs_nan_tick_formatting_warning():
    """
    Ignore upstream formatter warnings from hidden/NaN WCS animation axes.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*do_format.*",
            category=RuntimeWarning,
        )
        yield


def set_axis_properties(ax, axes_coordinates=None, *, animate=False):
    """
    Set IRIS axis labels and, for raster animations, the visible scan coordinate.
    """
    if hasattr(ax, "axes") and not hasattr(ax, "coords"):
        ax = ax.axes
    for axis in ax.coords:
        if axis.default_label.lower() in WAVELENGTH_LABELS:
            axis.set_format_unit(u.nm)
            axis.set_major_formatter("x.x")
            axis.set_axislabel("Wavelength [$\\mathrm{nm}$]")
        elif axis.default_label.lower() in LAT_LABELS:
            _set_axis_properties(axis, "Helioprojective Latitude [arcsec]", "red")
        elif axis.default_label.lower() in LON_LABELS:
            _set_axis_properties(axis, "Helioprojective Longitude [arcsec]", "black")
    if animate:
        _set_raster_animation_axis_properties(ax, axes_coordinates)


def finalize_iris_plot(ax, axes_coordinates=None):
    """
    Store requested axis-selection state and apply IRIS-specific axis formatting.
    """
    ax._iris_requested_axes_coordinates = axes_coordinates
    set_axis_properties(
        ax,
        axes_coordinates=axes_coordinates,
        animate=hasattr(ax, "slider_axes"),
    )
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


def _set_coord_position(coord, position):
    coord.set_ticks_visible(True)
    coord.set_ticklabel_visible(True)
    coord.set_ticks_position(position)
    coord.set_ticklabel_position(position)
    coord.set_axislabel_position(position)


def _hide_coord(coord):
    coord.set_ticks_visible(False)
    coord.set_ticklabel_visible(False)
    coord.set_ticks_position("#")
    coord.set_ticklabel_position("#")
    coord.set_axislabel_position("#")
    coord.set_axislabel("")


def _select_scan_coord_kind(axes_coordinates):
    if axes_coordinates:
        lowered = {coord.lower() for coord in axes_coordinates if isinstance(coord, str)}
        if lowered.intersection(TIME_LABEL_PRIORITY):
            return "time"
    return "longitude"


def _set_raster_animation_axis_properties(ax, axes_coordinates):
    """
    Place the raster scan coordinate on the frame edge selected by the user.

    Raster image animations can expose either helioprojective longitude or time on the
    scan axis. This helper keeps latitude on the left edge, hides the auxiliary scan-
    step helper, and moves the selected scan coordinate onto the visible frame edge.
    """
    scan_step_coords = [coord for coord in ax.coords if coord.default_label.lower() in SCAN_STEP_LABELS]
    if not scan_step_coords:
        return

    wavelength_coords = [coord for coord in ax.coords if coord.default_label.lower() in WAVELENGTH_LABELS]
    lon_coords = [coord for coord in ax.coords if coord.default_label.lower() in LON_LABELS]
    lat_coords = [coord for coord in ax.coords if coord.default_label.lower() in LAT_LABELS]
    time_coords = [coord for coord in ax.coords if coord.default_label.lower() in TIME_LABEL_PRIORITY]
    if not lon_coords or not lat_coords or not time_coords:
        return
    if wavelength_coords and wavelength_coords[0].get_ticks_position():
        # On 1D spectrum plots, wavelength is on the x-axis; don't reconfigure scan axis.
        return

    selected_scan_kind = _select_scan_coord_kind(axes_coordinates)
    selected_scan_coord = lon_coords[0]
    if selected_scan_kind == "time":
        for preferred_label in TIME_LABEL_PRIORITY:
            for coord in time_coords:
                if coord.default_label.lower() == preferred_label:
                    selected_scan_coord = coord
                    break
            else:
                continue
            break

    for coord in scan_step_coords:
        _hide_coord(coord)
    for coord in time_coords:
        if coord is not selected_scan_coord:
            _hide_coord(coord)
    if selected_scan_kind != "time":
        for coord in lon_coords[1:]:
            _hide_coord(coord)

    if selected_scan_kind == "time":
        _set_coord_position(lon_coords[0], "r")
        _set_coord_position(selected_scan_coord, "b")
    else:
        _set_coord_position(selected_scan_coord, ["b", "r"])
    _set_coord_position(lat_coords[0], "l")


class Plot2DMixin:
    def update_plot(self, val, artist, slider):
        with _suppress_wcs_nan_tick_formatting_warning():
            result = super().update_plot(val, artist, slider)
        if self.plot_dimensionality == 2:
            set_axis_properties(
                self.axes,
                axes_coordinates=getattr(self, "_iris_requested_axes_coordinates", None),
                animate=True,
            )
        return result


class IRISArrayAnimatorWCS(Plot2DMixin, ArrayAnimatorWCS):
    pass


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
        with _suppress_wcs_nan_tick_formatting_warning():
            ax = IRISArrayAnimatorWCS(data, wcs, plot_axes, coord_params=coord_params, **kwargs)
            self._apply_axes_coordinates(ax.axes, axes_coordinates)
            for hidden in self._not_visible_coords(ax.axes, axes_coordinates):
                param = ax.coord_params.get(hidden, {})
                param["ticks"] = False
                ax.coord_params[hidden] = param
        return ax
