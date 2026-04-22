import textwrap
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np

from ndcube import NDCollection
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube
from sunraster import SpectrogramSequence as SpecSeq

from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.visualization import IRISPlotter, IRISSequencePlotter, finalize_iris_plot

__all__ = ["RasterCollection", "SpectrogramCube", "SpectrogramCubeSequence"]


def _normalize_tuple_index(item, ndim):
    """
    Normalize a tuple index to explicit per-axis entries.

    Returns
    -------
    list or None
        A normalized list of length ``ndim`` when normalization is valid.
        Returns ``None`` when the tuple contains more than one ellipsis.
    """
    normalized_item = []
    ellipsis_seen = False
    for subitem in item:
        if subitem is Ellipsis:
            if ellipsis_seen:
                return None
            ellipsis_seen = True
            missing_dims = ndim - (len(item) - 1)
            normalized_item.extend([slice(None)] * missing_dims)
        else:
            normalized_item.append(subitem)
    if len(normalized_item) < ndim:
        normalized_item.extend([slice(None)] * (ndim - len(normalized_item)))
    return normalized_item


class SpectrogramCube(SpecCube):
    """
    Class representing spectrogram data described by a single WCS.

    Idea is that this class holds one complete raster scan or a sit and stare.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.
    wcs: `astropy.wcs.WCS`
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
        self._basic_wcs = kwargs.pop("_basic_wcs", None)
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    @property
    def time(self):
        time = super().time
        if time.format == "jd":
            time = time.copy()
            time.format = "isot"
        return time

    def _slice_basic_wcs(self, item):
        if self._basic_wcs is None:
            return None
        if isinstance(item, tuple):
            item = _normalize_tuple_index(item, self.data.ndim)
            if item is None or not all(isinstance(subitem, (Integral, slice)) for subitem in item):
                return None
            item = tuple(item)
        elif not isinstance(item, (Integral, slice)) and item is not Ellipsis:
            return None
        try:
            return self._basic_wcs.slice(item, numpy_order=True)
        except (IndexError, NotImplementedError, TypeError, ValueError) as e:
            logger.debug(f"Unable to slice SpectrogramCube basic_wcs with item {item!r}: {e}")
            return None

    def __getitem__(self, item):
        sliced_self = super().__getitem__(item)
        sliced_self._basic_wcs = self._slice_basic_wcs(item)
        return sliced_self

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def __str__(self) -> str:
        instance_start = None
        instance_end = None
        if self.global_coords and "time" in self.global_coords:
            instance_start = self.global_coords["time"].min().isot
            instance_end = self.global_coords["time"].max().isot
        elif self.extra_coords:
            try:
                extra_coord_time = self.axis_world_coords("time", wcs=self.extra_coords)
            except ValueError as e:
                logger.debug(f"Unable to determine time bounds for SpectrogramCube string representation: {e}")
                extra_coord_time = None
            if extra_coord_time:
                instance_start = extra_coord_time[0].min().isot
                instance_end = extra_coord_time[0].max().isot
        if instance_start is None or instance_end is None:
            try:
                instance_start = self.time.min().isot
                instance_end = self.time.max().isot
            except ValueError as e:
                logger.debug(f"Unable to determine time bounds for SpectrogramCube string representation: {e}")
                instance_start = "Unknown"
                instance_end = "Unknown"
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

    def plot(self, *args, **kwargs):
        cmap = kwargs.get("cmap")
        if not cmap:
            try:
                cmap = plt.get_cmap(name=f"irissji{int(self.meta.detector[:3])}")
            except Exception as e:  # NOQA: BLE001
                logger.debug(e)
                cmap = "viridis"
        kwargs["cmap"] = cmap
        if len(self.shape) == 1:
            kwargs.pop("cmap")
        return finalize_iris_plot(IRISPlotter(ndcube=self).plot(*args, **kwargs), kwargs.get("axes_coordinates"))

    @property
    def basic_wcs(self):
        return self._basic_wcs

    def spectrum_at(self, target, *, clip=True):
        """
        Return the spectrum at the raster pixel nearest a sky coordinate.

        Parameters
        ----------
        target : `astropy.coordinates.SkyCoord`
            Sky coordinate to sample.
        clip : `bool`, optional
            If `True`, off-raster targets are clipped to the nearest edge pixel.
            If `False`, out-of-bounds targets raise `ValueError`.

        Returns
        -------
        `irispy.spectrograph.SpectrogramCube`
            One-dimensional spectrum extracted from the nearest raster pixel.
        """
        if self.data.ndim != 3 or self.basic_wcs is None:
            msg = "spectrum_at requires a 3D raster cube with a basic_wcs bridge."
            raise ValueError(msg)

        step_index, slit_index = self.basic_wcs.celestial.world_to_array_index(target)
        if clip:
            step_index = int(np.clip(step_index, 0, self.shape[0] - 1))
            slit_index = int(np.clip(slit_index, 0, self.shape[1] - 1))
        else:
            step_index = int(step_index)
            slit_index = int(slit_index)
            if not (0 <= step_index < self.shape[0] and 0 <= slit_index < self.shape[1]):
                msg = "Target is outside the raster bounds."
                raise ValueError(msg)

        start = self.wcs.array_index_to_world(step_index, slit_index, 0)
        stop = self.wcs.array_index_to_world(step_index, slit_index, self.shape[-1] - 1)
        return self.crop(start, stop)

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


class SpectrogramCubeSequence(SpecSeq):
    """
    Class representing spectrogram data described by a collection of separate WCSes.

    So each individual `SpectrogramCube` within represents a single complete raster scan.
    The sequence contains multiple such cubes till the end of the observation.

    Parameters
    ----------
    data_list: `list`
        List of `SpectrogramCube` objects from the same spectral window and OBS ID.
    meta: `dict` or header object, optional
        Metadata associated with the sequence.
    common_axis: `int`, optional
        The axis of the NDCubes corresponding to time.
    """

    def __init__(self, data_list, meta=None, common_axis=0, **kwargs) -> None:
        if data_list and len(np.unique([cube.meta["OBSID"] for cube in data_list])) != 1:
            msg = "Constituent SpectrogramCube objects must have same value of 'OBSID' in its meta."
            raise ValueError(msg)
        super().__init__(data_list, meta=meta, common_axis=common_axis, **kwargs)

    def __str__(self) -> str:
        return super().__str__()

    def plot(self, *args, **kwargs):
        cmap = kwargs.get("cmap")
        if not cmap:
            try:
                cmap = plt.get_cmap(name=f"irissji{int(self.meta.detector[:3])}")
            except Exception as e:  # NOQA: BLE001
                logger.debug(e)
                cmap = "viridis"
        kwargs["cmap"] = cmap
        if len(self.shape) == 1:
            kwargs.pop("cmap")
        return finalize_iris_plot(
            IRISSequencePlotter(ndcube=self).plot(*args, **kwargs),
            kwargs.get("axes_coordinates"),
        )

    def remove_cosmic_rays(
        self,
        *,
        method="rsliding",
        sigma: float | None = None,
        max_iters: int | None = None,
        method_kwargs=None,
    ):
        """
        Return a cleaned copy of each cube in the sequence.

        This is a convenience wrapper around `irispy.utils.cosmic_rays.remove_cosmic_rays`.
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
    Subclass of NDCollection for holding a collection of `.SpectrogramCube` or
    `.SpectrogramCubeSequence` with keys being the spectral windows.
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
