import matplotlib.pyplot as plt
from mpl_animators import ArrayAnimatorWCS

import astropy.units as u

import sunpy.visualization.colormaps as cm  # NOQA: F401
from ndcube.visualization.mpl_plotter import MatplotlibPlotter
from ndcube.visualization.mpl_sequence_plotter import MatplotlibSequencePlotter, SequenceAnimator
from sunpy import log as logger

__all__ = ["IRISArrayAnimatorWCS", "IRISPlotter", "IRISSequencePlotter", "SJIPlotter"]


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
TIME_LABELS = ["time (utc)", "time"]
WAVELENGTH_LABELS = ["wavelength", "wave", "em.wl"]
SCAN_LABELS = set(LON_LABELS + LAT_LABELS)


def _shorten_slider_label(label):
    label_parts = {label_part.strip().lower() for label_part in str(label).split(" / ")}
    if label_parts.intersection(WAVELENGTH_LABELS):
        return "Wavelength"
    if label_parts.intersection(TIME_LABELS):
        return "Time"
    if label_parts and label_parts.issubset(SCAN_LABELS):
        return "Scan"
    return label


def set_axis_properties(ax):
    """
    Set the axis colors and labels for IRIS SJI and Raster data.
    """
    if isinstance(ax, ArrayAnimatorWCS):
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


class Plot2DMixin:
    def update_plot_2d(self, val, im, slider):
        super().update_plot_2d(val, im, slider)
        set_axis_properties(self.axes)


class IRISArrayAnimatorWCS(Plot2DMixin, ArrayAnimatorWCS):
    def _compute_slider_labels_from_wcs(self, slices):
        return [_shorten_slider_label(label) for label in super()._compute_slider_labels_from_wcs(slices)]


class IRISSequenceAnimator(Plot2DMixin, SequenceAnimator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("slider_labels", ["Raster step", "Scan number"])
        super().__init__(*args, **kwargs)


class _IRISPlotMixin:
    """
    Shared IRIS plot behavior for cube and sequence plotters.
    """

    def _default_cmap_name(self):
        return f"irissji{int(self._ndcube.meta.detector[:3])}"

    def plot(self, *args, **kwargs):
        """
        Plot the cube with IRIS defaults.

        If ``cmap`` is falsey, it is replaced with a default IRIS colormap derived from
        the cube metadata (falling back to viridis). For one-dimensional cubes ``cmap``
        is dropped entirely, even when given. IRIS axis styling is applied to the
        result; all other arguments are passed to the parent plotter's ``plot``.
        """
        cmap = kwargs.get("cmap")
        if not cmap:
            try:
                cmap = plt.get_cmap(name=self._default_cmap_name())
            except Exception as e:  # NOQA: BLE001
                logger.debug(e)
                cmap = "viridis"
        kwargs["cmap"] = cmap
        if len(self._ndcube.shape) == 1:
            kwargs.pop("cmap")
        ax = super().plot(*args, **kwargs)
        set_axis_properties(ax)
        return ax


class IRISPlotter(_IRISPlotMixin, MatplotlibPlotter):
    def _animate_cube(
        self,
        wcs,
        plot_axes=None,
        axes_coordinates=None,
        axes_units=None,
        data_unit=None,
        **kwargs,
    ):
        # This is a copy of the super method, but with the replacement
        # of the ArrayAnimatorWCS with IRISArrayAnimatorWCS.
        data, wcs, plot_axes, coord_params = self._prep_animate_args(wcs, plot_axes, axes_units, data_unit)
        ax = IRISArrayAnimatorWCS(data, wcs, plot_axes, coord_params=coord_params, **kwargs)
        # We need to modify the visible axes after the axes object has been created.
        # This call affects only the initial draw
        self._apply_axes_coordinates(ax.axes, axes_coordinates)
        # This changes the parameters for future iterations
        for hidden in self._not_visible_coords(ax.axes, axes_coordinates):
            param = ax.coord_params.get(hidden, {})
            param["ticks"] = False
            ax.coord_params[hidden] = param
        return ax


class SJIPlotter(IRISPlotter):
    def _default_cmap_name(self):
        return f"irissji{int(self._ndcube.meta['TWAVE1'])}"


class IRISSequencePlotter(_IRISPlotMixin, MatplotlibSequencePlotter):
    def animate(self, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        """
        Animate the `~ndcube.NDCubeSequence` with the sequence axis as a slider.

        Keyword arguments are passed to
        `ndcube.visualization.mpl_plotter.MatplotlibPlotter.plot` and therefore only
        apply to cube axes, not the sequence axis.
        See that method's docstring for definition of keyword arguments.

        Parameters
        ----------
        sequence_axis_coords: `str` optional
            The name of the coordinate in `~ndcube.NDCubeSequence.sequence_axis_coords`
            to be used as the slider pixel values.
            If None, array indices will be used.

        sequence_axis_unit: `astropy.units.Unit` or `str`, optional
            The unit in which the sequence_axis_coordinates should be displayed.
            If None, the default unit will be used.
        """
        return IRISSequenceAnimator(
            self._ndcube, sequence_axis_coords=sequence_axis_coords, sequence_axis_unit=sequence_axis_unit, **kwargs
        )
