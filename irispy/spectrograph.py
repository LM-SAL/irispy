import textwrap

import matplotlib.pyplot as plt

from ndcube import NDCollection
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube

from irispy._spectrograph_wcs import (
    _SPECTROGRAM_CUBE_METADATA_DEFAULTS,
    _SPECTROGRAM_CUBE_METADATA_KWARGS,
    _SpectrogramCubeWCSMixin,
)
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.visualization import IRISPlotter, finalize_iris_plot

__all__ = ["RasterCollection", "SpectrogramCube"]


class SpectrogramCube(_SpectrogramCubeWCSMixin, SpecCube):
    """
    Class representing spectrogram data described by a single WCS.

    A raster window is exposed as one cube, whether it comes from a single file,
    a combined multi-file raster, or a sit-and-stare observation.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.
    wcs: `astropy.wcs.WCS` or `gwcs.WCS`
        The WCS object containing the axes' information
    unit : `astropy.units.Unit` or `str`
        Unit for the dataset. Strings that can be converted to a Unit are allowed.
    meta : `dict` object
        Additional meta information about the dataset. Must contain at least the
        following keys:
        - detector type: str, (FUV1, FUV2 or NUV)
        - OBSID: int
        - spectral window: str
    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn't mandatory. If the uncertainty
        has no such attribute the uncertainty is stored as UnknownUncertainty.
        Defaults to None.
    mask : any type, optional
        Mask for the dataset. Masks should follow the numpy convention
        that valid data points are marked by False and invalid ones with True.
        Defaults to None.
    copy : `bool`, optional
        Indicates whether to save the arguments as copy. True copies every attribute
        before saving it while False tries to save every parameter as reference.
        Note however that it is not always possible to save the input as reference.
        Default is False.
    """

    def __init__(self, data, wcs, uncertainty, unit, meta, *, mask=None, copy=False, **kwargs) -> None:
        for attr in _SPECTROGRAM_CUBE_METADATA_KWARGS:
            setattr(self, attr, kwargs.pop(attr, _SPECTROGRAM_CUBE_METADATA_DEFAULTS.get(attr)))
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    @property
    def time(self):
        time = super().time
        if time.format == "jd":
            time.format = "isot"
        return time

    def __getitem__(self, item):
        normalized_item = self._normalize_basic_wcs_item(item)
        item_for_super = normalized_item if normalized_item is not None else item
        sliced_self = super().__getitem__(item_for_super)
        if isinstance(sliced_self, SpectrogramCube):
            sliced_self._basic_wcs = self._slice_basic_wcs(item_for_super)
            self._slice_raster_metadata(item_for_super, sliced_self)
        return sliced_self

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def _time_bounds(self):
        if self.global_coords and "time" in self.global_coords:
            return self.global_coords["time"].min().isot, self.global_coords["time"].max().isot
        if self.extra_coords:
            try:
                extra_coord_time = self.axis_world_coords("time", wcs=self.extra_coords)
                if extra_coord_time:
                    return extra_coord_time[0].min().isot, extra_coord_time[0].max().isot
            except ValueError as e:
                logger.debug(f"Unable to determine time bounds for SpectrogramCube string representation: {e}")
        try:
            return self.time.min().isot, self.time.max().isot
        except (ValueError, AttributeError) as e:
            logger.debug(f"Unable to determine time bounds for SpectrogramCube string representation: {e}")
            return "Unknown", "Unknown"

    def __str__(self) -> str:
        instance_start, instance_end = self._time_bounds()
        return textwrap.dedent(
            f"""
            SpectrogramCube
            ---------------
            Obs ID:             {self.meta.get("OBSID")}
            Obs Description:    {self.meta.get("OBS_DESC")}
            Obs Date:           {instance_start} -- {instance_end}
            Data shape:         {self.shape}
            Axis Types:         {self.array_axis_physical_types}
            Roll:               {self.meta.get("SAT_ROT")}
            """,
        )

    def plot(self, *args, slider_labels=None, **kwargs):
        """
        Plot the spectrogram cube.

        Parameters
        ----------
        slider_labels : sequence of str, optional
            Labels to use for animation sliders, in the order the sliders are shown.
        **kwargs
            Additional keyword arguments are passed to the ndcube plotting machinery.
        """
        cmap = kwargs.get("cmap")
        if not cmap:
            detector = getattr(self.meta, "detector", None)
            if detector and len(str(detector)) >= 3:
                try:
                    cmap = plt.get_cmap(name=f"irissji{int(str(detector)[:3])}")
                except (ValueError, KeyError):
                    cmap = "viridis"
            else:
                cmap = "viridis"
        kwargs["cmap"] = cmap
        if len(self.shape) == 1:
            kwargs.pop("cmap")
        if slider_labels is not None:
            kwargs["slider_labels"] = slider_labels
        return finalize_iris_plot(IRISPlotter(ndcube=self).plot(*args, **kwargs), kwargs.get("axes_coordinates"))

    @property
    def basic_wcs(self):
        return self._basic_wcs

    @property
    def raster_boundaries(self):
        if self._separate_raster_axis:
            return tuple(slice(index, index + 1) for index in range(self.shape[0]))
        if self._raster_boundaries is None:
            return ()
        return tuple(slice(start, stop) for start, stop in self._raster_boundaries)

    def raster_slice(self, index):
        """
        Return the subcube corresponding to one original raster.
        """
        if self._separate_raster_axis:
            return self[index]
        boundaries = self.raster_boundaries
        if not boundaries:
            if index == 0:
                return self
            msg = "Raster index out of range."
            raise IndexError(msg)
        return self[boundaries[index]]

    def split_rasters(self):
        """
        Split the cube into per-raster subcubes.
        """
        if self._separate_raster_axis:
            return tuple(self[i] for i in range(self.shape[0]))
        boundaries = self.raster_boundaries
        if not boundaries:
            return (self,)
        return tuple(self[raster_slice] for raster_slice in boundaries)

    def remove_cosmic_rays(
        self,
        *,
        method="rsliding",
        sigma: float | None = None,
        max_iters: int | None = None,
        method_kwargs=None,
    ):
        """
        Return a cleaned copy of the cube with cosmic rays removed.

        This is a convenience wrapper around `irispy.utils.cosmic_rays.remove_cosmic_rays`.

        Parameters
        ----------
        method : ``{"rsliding", "astroscrappy"}``, optional
            Backend used to detect and clean cosmic rays.
        sigma : `float`, optional
            Shared clipping threshold override for the selected backend.
        max_iters : `int`, optional
            Shared iteration-count override for the selected backend.
        method_kwargs : `dict`, optional
            Additional keyword arguments passed to the selected backend.

        Returns
        -------
        `irispy.spectrograph.SpectrogramCube`
            Cleaned cube with the same metadata and coordinates as the original.
        """
        return remove_cosmic_rays(
            self,
            method=method,
            sigma=sigma,
            max_iters=max_iters,
            method_kwargs=method_kwargs,
        )


class RasterCollection(NDCollection):
    """
    Subclass of NDCollection for raster spectral windows keyed by window name.

    Each value is a `SpectrogramCube`.
    """

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
            Raster Collection
            -----------------
            Spectral Windows (cube keys): {tuple(self.keys())}
            Number of Cubes: {len(self)}
            Aligned dimensions: {self.aligned_dimensions}
            Aligned physical types: {self.aligned_axis_physical_types}
            """,
        )
