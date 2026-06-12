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
SLIDER_SCAN_STEP_LABELS = [*SCAN_STEP_LABELS, "raster_step"]
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
    for axis, physical_type in _iter_coords_with_physical_types(ax):
        default_label = axis.default_label.lower()
        if physical_type in WAVELENGTH_LABELS or default_label in WAVELENGTH_LABELS:
            axis.set_format_unit(u.nm)
            axis.set_major_formatter("x.x")
            axis.set_axislabel("Wavelength [$\\mathrm{nm}$]")
        elif physical_type in LAT_LABELS or default_label in LAT_LABELS:
            _set_axis_properties(axis, "Helioprojective Latitude [arcsec]", "red")
        elif physical_type in LON_LABELS or default_label in LON_LABELS:
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


def _iter_coords_with_physical_types(ax):
    physical_types = tuple(getattr(ax.wcs, "world_axis_physical_types", ()) or ())
    for index, coord in enumerate(ax.coords):
        physical_type = physical_types[index].lower() if index < len(physical_types) and physical_types[index] else ""
        yield coord, physical_type


def _coords_matching(ax, labels):
    labels = {label.lower() for label in labels}
    return list(
        dict.fromkeys(
            coord
            for coord, physical_type in _iter_coords_with_physical_types(ax)
            if physical_type in labels or coord.default_label.lower() in labels
        )
    )


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
    step_coords = _coords_matching(ax, SLIDER_SCAN_STEP_LABELS)
    scan_coords = _coords_matching(ax, SLIDER_SCAN_LABELS)
    if not step_coords and not scan_coords:
        return

    wavelength_coords = _coords_matching(ax, WAVELENGTH_LABELS)
    lon_coords = _coords_matching(ax, LON_LABELS)
    lat_coords = _coords_matching(ax, LAT_LABELS)
    time_coords = _coords_matching(ax, TIME_LABEL_PRIORITY)
    if not lon_coords or not lat_coords or not time_coords:
        return
    if wavelength_coords and wavelength_coords[0].get_ticks_position():
        # On 1D spectrum plots, wavelength is on the x-axis; don't reconfigure scan axis.
        return

    selected_scan_kind = _select_scan_coord_kind(axes_coordinates)
    selected_scan_coord = lon_coords[0]
    if selected_scan_kind == "time":
        selected_scan_coord = next(
            (
                coord
                for preferred_label in TIME_LABEL_PRIORITY
                for coord in time_coords
                if coord.default_label.lower() == preferred_label
            ),
            selected_scan_coord,
        )

    for coord in [*step_coords, *scan_coords]:
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
    def _slider_axis_changes_visible_wcs(self, ax_ind):
        try:
            wcs_pixel_axis = self.wcs.pixel_n_dim - ax_ind - 1
            axis_correlation_matrix = self.wcs.axis_correlation_matrix
        except AttributeError:
            return True

        visible_world_axes = [
            index
            for index, coord in enumerate(self.axes.coords)
            if any(position != "#" for position in coord.get_ticks_position())
        ]
        if not visible_world_axes:
            return True
        return any(axis_correlation_matrix[index, wcs_pixel_axis] for index in visible_world_axes)

    def _update_plot_2d_data_only(self, val, artist, slider):
        artist.set_array(self.data_transposed)
        if self.clip_interval is not None:
            vmin, vmax = self._get_2d_plot_limits()
            artist.set_clim(vmin, vmax)
        slider.cval = val

    def update_plot(self, val, artist, slider):
        if self.plot_dimensionality != 2:
            with _suppress_wcs_nan_tick_formatting_warning():
                return super().update_plot(val, artist, slider)

        ind = int(val)
        if ind == int(slider.cval):
            return None
        ax_ind = self.slider_axes[slider.slider_ind]
        self.frame_slice[ax_ind] = ind
        self.slices_wcsaxes[self.wcs.pixel_n_dim - ax_ind - 1] = ind
        reset_wcs = self._slider_axis_changes_visible_wcs(ax_ind)

        with _suppress_wcs_nan_tick_formatting_warning():
            if reset_wcs:
                self.update_plot_2d(val, artist, slider)
            else:
                self._update_plot_2d_data_only(val, artist, slider)

        if reset_wcs:
            self._apply_coord_params(self.axes)
            set_axis_properties(
                self.axes,
                axes_coordinates=getattr(self, "_iris_requested_axes_coordinates", None),
                animate=True,
            )
        return super(ArrayAnimatorWCS, self).update_plot(val, artist, slider)


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
        with _suppress_wcs_nan_tick_formatting_warning():
            ax = IRISArrayAnimatorWCS(data, wcs, plot_axes, coord_params=coord_params, **kwargs)
            self._apply_axes_coordinates(ax.axes, axes_coordinates)
            for hidden in self._not_visible_coords(ax.axes, axes_coordinates):
                param = ax.coord_params.get(hidden, {})
                param["ticks"] = False
                ax.coord_params[hidden] = param
        return ax
