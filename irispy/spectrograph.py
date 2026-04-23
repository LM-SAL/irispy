import textwrap
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

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

    A raster window is exposed as one cube, whether it comes from a single file,
    a combined multi-file raster, or a sit-and-stare observation.

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
        self._basic_wcs_segments = kwargs.pop("_basic_wcs_segments", None)
        self._raster_boundaries = kwargs.pop("_raster_boundaries", None)
        self._memmap = kwargs.pop("_memmap", False)
        self._memmap_slice = kwargs.pop("_memmap_slice", None)
        self._raster_memmap_segments = kwargs.pop("_raster_memmap_segments", None)
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    @property
    def time(self):
        time = super().time
        if time.format == "jd":
            time = time.copy()
            time.format = "isot"
        return time

    def _normalize_basic_wcs_item(self, item):
        if isinstance(item, tuple):
            item = _normalize_tuple_index(item, self.data.ndim)
            if item is None or not all(isinstance(subitem, (Integral, slice)) for subitem in item):
                return None
            return tuple(item)
        if isinstance(item, (Integral, slice)):
            return (item, *([slice(None)] * (self.data.ndim - 1)))
        if item is Ellipsis:
            return tuple([slice(None)] * self.data.ndim)
        return None

    def _slice_single_basic_wcs(self, normalized_item):
        if self._basic_wcs is None:
            return None
        try:
            return self._basic_wcs.slice(normalized_item, numpy_order=True)
        except (IndexError, NotImplementedError, TypeError, ValueError) as e:
            logger.debug(f"Unable to slice SpectrogramCube basic_wcs with item {normalized_item!r}: {e}")
            return None

    def _slice_segment_basic_wcs_index(self, scan_index, normalized_item):
        for segment_start, segment_stop, segment_wcs in self._basic_wcs_segments:
            if segment_start <= scan_index < segment_stop:
                relative_item = (scan_index - segment_start, *normalized_item[1:])
                try:
                    return segment_wcs.slice(relative_item, numpy_order=True)
                except (IndexError, NotImplementedError, TypeError, ValueError) as e:
                    logger.debug(
                        "Unable to slice SpectrogramCube segment basic_wcs with "
                        f"item {relative_item!r}: {e}"
                    )
                    return None
        return None

    def _slice_segment_basic_wcs_slice(self, scan_item, normalized_item):
        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None
        for segment_start, segment_stop, segment_wcs in self._basic_wcs_segments:
            if segment_start <= scan_start and scan_stop <= segment_stop:
                relative_item = (slice(scan_start - segment_start, scan_stop - segment_start), *normalized_item[1:])
                try:
                    return segment_wcs.slice(relative_item, numpy_order=True)
                except (IndexError, NotImplementedError, TypeError, ValueError) as e:
                    logger.debug(
                        "Unable to slice SpectrogramCube segment basic_wcs with "
                        f"item {relative_item!r}: {e}"
                    )
                    return None
        return None

    def _slice_segment_basic_wcs(self, normalized_item):
        if not self._basic_wcs_segments:
            return None
        scan_item = normalized_item[0]
        if isinstance(scan_item, Integral):
            scan_index = scan_item if scan_item >= 0 else self.shape[0] + scan_item
            return self._slice_segment_basic_wcs_index(scan_index, normalized_item)
        return self._slice_segment_basic_wcs_slice(scan_item, normalized_item)

    def _slice_basic_wcs(self, item):
        normalized_item = self._normalize_basic_wcs_item(item)
        if normalized_item is None:
            return None
        return self._slice_single_basic_wcs(normalized_item) or self._slice_segment_basic_wcs(normalized_item)

    def _slice_basic_wcs_segments_for_slice(self, scan_item):
        if not self._basic_wcs_segments:
            return None
        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None
        sliced_segments = []
        for segment_start, segment_stop, segment_wcs in self._basic_wcs_segments:
            overlap_start = max(segment_start, scan_start)
            overlap_stop = min(segment_stop, scan_stop)
            if overlap_start >= overlap_stop:
                continue
            relative_item = (slice(overlap_start - segment_start, overlap_stop - segment_start), slice(None), slice(None))
            try:
                overlap_wcs = segment_wcs.slice(relative_item, numpy_order=True)
            except (IndexError, NotImplementedError, TypeError, ValueError) as e:
                logger.debug(
                    "Unable to slice SpectrogramCube segment basic_wcs with "
                    f"item {relative_item!r}: {e}"
                )
                return None
            sliced_segments.append((overlap_start - scan_start, overlap_stop - scan_start, overlap_wcs))
        return sliced_segments or None

    def _slice_raster_boundaries_for_slice(self, scan_item):
        if not self._raster_boundaries:
            return None
        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None
        boundaries = []
        for boundary_start, boundary_stop in self._raster_boundaries:
            overlap_start = max(boundary_start, scan_start)
            overlap_stop = min(boundary_stop, scan_stop)
            if overlap_start < overlap_stop:
                boundaries.append((overlap_start - scan_start, overlap_stop - scan_start))
        return boundaries or None

    def _slice_raster_memmap_segments_for_slice(self, scan_item):
        if not self._raster_memmap_segments:
            return None
        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None
        segments = []
        for segment_start, segment_stop, path, ext, flip, memmap_slice in self._raster_memmap_segments:
            overlap_start = max(segment_start, scan_start)
            overlap_stop = min(segment_stop, scan_stop)
            if overlap_start >= overlap_stop:
                continue
            base_start = 0 if memmap_slice is None or memmap_slice.start is None else memmap_slice.start
            local_start = base_start + (overlap_start - segment_start)
            local_stop = base_start + (overlap_stop - segment_start)
            segments.append((overlap_start - scan_start, overlap_stop - scan_start, path, ext, flip, slice(local_start, local_stop)))
        return segments or None

    def _slice_raster_metadata(self, item, sliced_self):
        normalized_item = self._normalize_basic_wcs_item(item)
        if normalized_item is None:
            sliced_self._basic_wcs_segments = None
            sliced_self._raster_boundaries = None
            sliced_self._raster_memmap_segments = None
            return

        scan_item = normalized_item[0]
        for attr in ("_raster_wcs_header", "_raster_observer"):
            if hasattr(self, attr):
                setattr(sliced_self, attr, getattr(self, attr))
        sliced_self._memmap = self._memmap

        if isinstance(scan_item, Integral):
            sliced_self._basic_wcs_segments = None
            sliced_self._raster_boundaries = None
            sliced_self._raster_memmap_segments = None
            return

        for attr in ("_raster_pc_table", "_raster_crval_table"):
            value = getattr(self, attr, None)
            if value is not None:
                setattr(sliced_self, attr, value[scan_item])
        sliced_self._basic_wcs_segments = self._slice_basic_wcs_segments_for_slice(scan_item)
        sliced_self._raster_boundaries = self._slice_raster_boundaries_for_slice(scan_item)
        sliced_self._raster_memmap_segments = self._slice_raster_memmap_segments_for_slice(scan_item)
        if sliced_self._raster_memmap_segments and len(sliced_self._raster_memmap_segments) == 1:
            _, _, path, ext, flip, memmap_slice = sliced_self._raster_memmap_segments[0]
            sliced_self._memmap_path = path
            sliced_self._memmap_ext = ext
            sliced_self._flip = flip
            sliced_self._memmap_slice = memmap_slice
        else:
            for attr in ("_memmap_path", "_memmap_ext", "_flip", "_memmap_slice"):
                if hasattr(sliced_self, attr):
                    delattr(sliced_self, attr)

    def __getitem__(self, item):
        sliced_self = super().__getitem__(item)
        sliced_self._basic_wcs = self._slice_basic_wcs(item)
        self._slice_raster_metadata(item, sliced_self)
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

    @property
    def raster_boundaries(self):
        if self._raster_boundaries is None:
            return ()
        return tuple(slice(start, stop) for start, stop in self._raster_boundaries)

    def raster_slice(self, index):
        """
        Return the subcube corresponding to one original raster.
        """
        boundaries = self.raster_boundaries
        if not boundaries:
            if index == 0:
                return self
            msg = "Raster index out of range."
            raise IndexError(msg)
        return self[boundaries[index]]

    def split_rasters(self):
        """
        Split the cube back into per-raster subcubes.
        """
        boundaries = self.raster_boundaries
        if not boundaries:
            return (self,)
        return tuple(self[raster_slice] for raster_slice in boundaries)

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
        if self.data.ndim != 3:
            msg = "spectrum_at requires a 3D raster cube."
            raise ValueError(msg)
        if self.basic_wcs is not None:
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
        else:
            sky_grid = self.axis_world_coords("custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat")[0]
            target_in_frame = target.transform_to(sky_grid)
            if not clip:
                lon = target_in_frame.Tx.to(u.arcsec)
                lat = target_in_frame.Ty.to(u.arcsec)
                if (
                    lon < np.nanmin(sky_grid.Tx)
                    or lon > np.nanmax(sky_grid.Tx)
                    or lat < np.nanmin(sky_grid.Ty)
                    or lat > np.nanmax(sky_grid.Ty)
                ):
                    msg = "Target is outside the raster bounds."
                    raise ValueError(msg)
            separation = sky_grid.separation(target_in_frame)
            step_index, slit_index = np.unravel_index(np.nanargmin(separation.to_value(u.arcsec)), separation.shape)

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
    Temporary helper for per-raster views of one spectral window.

    Each element is one complete raster scan. New raster reads should prefer the
    combined `SpectrogramCube` model and use `split_rasters()` only when explicit
    per-raster access is needed.

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

    def as_cube(self):
        """
        Combine a raster sequence into one cube along the scan axis.

        This supports both eager and memmapped raster sequences.
        """
        if any(getattr(cube, "_memmap", False) for cube in self):
            from irispy.io.spectrograph import _combine_raster_sequence_lazy  # NOQA: PLC0415

            return _combine_raster_sequence_lazy(self)
        from irispy.io.spectrograph import _combine_raster_sequence  # NOQA: PLC0415

        return _combine_raster_sequence(self)

    def plot(self, *args, **kwargs):
        axes_coordinates = kwargs.get("axes_coordinates")
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
        kwargs.pop("axes_coordinates", None)
        return finalize_iris_plot(IRISSequencePlotter(ndcube=self).plot(*args, **kwargs), axes_coordinates)

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
    Subclass of NDCollection for raster spectral windows keyed by window name.

    Each value is normally a `SpectrogramCube`. `SpectrogramCubeSequence` remains
    possible for transitional or opt-in per-raster workflows.
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
