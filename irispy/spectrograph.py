import textwrap

import matplotlib.pyplot as plt
import numpy as np

from ndcube import NDCollection
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube
from sunraster import SpectrogramSequence as SpecSeq

from irispy.visualization import IRISPlotter, IRISSequencePlotter, set_axis_properties

__all__ = ["RasterCollection", "SpectrogramCube", "SpectrogramCubeSequence"]


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
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        return SpectrogramCube(
            result.data,
            result.wcs,
            result.uncertainty,
            result.unit,
            result.meta,
            mask=result.mask,
        )

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def __str__(self) -> str:
        instance_start = None
        instance_end = None
        if self.global_coords and "time" in self.global_coords:
            instance_start = self.global_coords["time"].min().isot
            instance_end = self.global_coords["time"].max().isot
        elif self.extra_coords and self.axis_world_coords("time", wcs=self.extra_coords):
            instance_start = self.axis_world_coords("time", wcs=self.extra_coords)[0].min().isot
            instance_end = self.axis_world_coords("time", wcs=self.extra_coords)[0].max().isot
        return textwrap.dedent(
            f"""
            SpectrogramCube
            ---------------
            OBS ID:             {self.meta.get("OBSID")}
            OBS Description:    {self.meta.get("OBS_DESC")}
            OBS period:         {self.meta.get("STARTOBS")} -- {self.meta.get("ENDOBS")}
            Spectrogram period: {instance_start} -- {instance_end}
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
        ax = IRISPlotter(ndcube=self).plot(*args, **kwargs)
        set_axis_properties(ax)
        return ax


class SpectrogramCubeSequence(SpecSeq):
    """
    Class representing spectrogram data described by a collection of separate
    WCSes.

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
        # Check that all spectrograms are from same spectral window and OBS ID.
        if len(np.unique([cube.meta["OBSID"] for cube in data_list])) != 1:
            msg = "Constituent SpectrogramCube objects must have same value of 'OBSID' in its meta."
            raise ValueError(msg)
        super().__init__(data_list, meta=meta, common_axis=common_axis, **kwargs)

    def __str__(self) -> str:
        # Overload it get the class name in the string
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
        ax = IRISSequencePlotter(ndcube=self).plot(*args, **kwargs)
        set_axis_properties(ax)
        return ax


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
