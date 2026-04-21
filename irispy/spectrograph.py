import textwrap
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from ndcube import NDCollection
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube
from sunraster import SpectrogramSequence as SpecSeq

from irispy.utils.cosmic_rays import remove_cosmic_rays
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
        self._basic_wcs = kwargs.pop("_basic_wcs", None)
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    @property
    def exposure_time(self):
        try:
            return super().exposure_time
        except ValueError:
            # After scalar slicing, ndcube promotes axis-specific extra_coords to
            # global_coords. Check there first before falling back to FITS EXPTIME.
            gc = self.global_coords
            if gc:
                from sunraster.spectrogram import SUPPORTED_EXPOSURE_NAMES  # NOQA: PLC0415

                for name in SUPPORTED_EXPOSURE_NAMES:
                    if name in gc:
                        return gc[name]
            exptime = self.meta.get("EXPTIME")
            if exptime is not None:
                import astropy.units as u  # NOQA: PLC0415

                return exptime * u.s
            raise

    @property
    def time(self):
        time = super().time
        if time.format == "jd":
            time = time.copy()
            time.format = "isot"
        return time

    def __getitem__(self, item):
        result = super().__getitem__(item)
        extra_coords = result.extra_coords if not result.extra_coords.is_empty else None
        new_sc = SpectrogramCube(
            result.data,
            result.wcs,
            result.uncertainty,
            result.unit,
            result.meta,
            mask=result.mask,
            extra_coords=extra_coords,
            _basic_wcs=self._basic_wcs,
        )
        # ndcube promotes scalar-indexed extra_coords to global_coords.
        # Copy them explicitly so they are bound to the new cube, not result.
        for name, coord in result.global_coords.items():
            physical_type = result.global_coords.physical_types[name]
            if isinstance(physical_type, tuple):
                physical_type = physical_type[0]
            new_sc.global_coords.add(name, physical_type, coord)
        return new_sc

    def _exposure_time_array_axis(self):
        exposure_time_name = self._exposure_time_name
        extra_coord_names = tuple(self.extra_coords.keys())
        if exposure_time_name is None and "exposure time" in extra_coord_names:
            exposure_time_name = "exposure time"
        if exposure_time_name is not None and exposure_time_name in extra_coord_names:
            mapping = self.extra_coords[exposure_time_name].mapping
            if len(mapping) == 1:
                return self.data.ndim - 1 - int(mapping[0])
        if self._exposure_time_name is not None and self._exposure_time_loc is not None:
            exposure_axis = self._get_axis_coord_index(self._exposure_time_name, self._exposure_time_loc)
            if isinstance(exposure_axis, tuple):
                exposure_axis = exposure_axis[0]
            return int(exposure_axis)
        matching_axes = [
            axis for axis, size in enumerate(self.data.shape) if size == np.asarray(self.exposure_time).shape[0]
        ]
        if len(matching_axes) == 1:
            return matching_axes[0]
        msg = "Unable to determine exposure time array axis."
        raise ValueError(msg)

    def apply_exposure_time_correction(self, undo=False, force=False):  # noqa: FBT002
        try:
            return super().apply_exposure_time_correction(undo=undo, force=force)
        except (TypeError, ValueError):
            from sunraster.spectrogram import (  # NOQA: PLC0415
                _calculate_exposure_time_correction,
                _uncalculate_exposure_time_correction,
            )

            exposure_time_s = self.exposure_time.to(u.s).value
            if not np.isscalar(exposure_time_s):
                exposure_axis = self._exposure_time_array_axis()
                item = [np.newaxis] * self.data.ndim
                item[exposure_axis] = slice(None)
                exposure_time_s = exposure_time_s[tuple(item)]
            if undo is True:
                new_data, new_uncertainty, new_unit = _uncalculate_exposure_time_correction(
                    self.data,
                    self.uncertainty,
                    self.unit,
                    exposure_time_s,
                    force=force,
                )
            else:
                new_data, new_uncertainty, new_unit = _calculate_exposure_time_correction(
                    self.data,
                    self.uncertainty,
                    self.unit,
                    exposure_time_s,
                    force=force,
                )
            new_cube = deepcopy(self)
            new_cube._data = new_data
            new_cube._uncertainty = new_uncertainty
            new_cube._extra_coords = self.extra_coords
            new_cube._unit = new_unit
            return new_cube

    @staticmethod
    def _slice_or_index(lower, upper, keepdims):
        if lower == upper and not keepdims:
            return lower
        return slice(lower, upper + 1)

    def _nearest_spatial_indices(self, coordinate):
        sample_world = self.wcs.pixel_to_world(*([0] * self.wcs.pixel_n_dim))
        sample_world = sample_world if isinstance(sample_world, (list, tuple)) else [sample_world]
        frame = next(world.frame for world in sample_world if hasattr(world, "frame"))
        coord = coordinate.transform_to(frame)
        lon = self.axis_world_coords_values("custom:pos.helioprojective.lon")[0].to_value(u.arcsec)
        lat = self.axis_world_coords_values("custom:pos.helioprojective.lat")[0].to_value(u.arcsec)
        distance = (lon - coord.Tx.to_value(u.arcsec)) ** 2 + (lat - coord.Ty.to_value(u.arcsec)) ** 2
        return np.unravel_index(np.nanargmin(distance), distance.shape)

    def _manual_crop_item(self, points, keepdims):
        if self.data.ndim != 3 or len(points) != 2:
            return None
        if any(not isinstance(point, (tuple, list)) or len(point) != 2 for point in points):
            return None

        spectral_points = [point[0] for point in points if point[0] is not None]
        spatial_points = [point[1] for point in points if point[1] is not None]
        if not spectral_points and not spatial_points:
            return None

        item = [slice(None)] * self.data.ndim
        if spectral_points:
            spectral_axis = self.spectral_axis.to_value(self.spectral_axis.unit)
            spectral_indices = [
                int(np.nanargmin(np.abs(spectral_axis - spectral_point.to_value(self.spectral_axis.unit))))
                for spectral_point in spectral_points
            ]
            item[2] = self._slice_or_index(min(spectral_indices), max(spectral_indices), keepdims)
        if spatial_points:
            spatial_indices = [self._nearest_spatial_indices(spatial_point) for spatial_point in spatial_points]
            raster_indices = [idx[0] for idx in spatial_indices]
            slit_indices = [idx[1] for idx in spatial_indices]
            item[0] = self._slice_or_index(min(raster_indices), max(raster_indices), keepdims)
            item[1] = self._slice_or_index(min(slit_indices), max(slit_indices), keepdims)
        return tuple(item)

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
            except ValueError:
                extra_coord_time = None
            if extra_coord_time:
                instance_start = extra_coord_time[0].min().isot
                instance_end = extra_coord_time[0].max().isot
        if instance_start is None or instance_end is None:
            try:
                instance_start = self.time.min().isot
                instance_end = self.time.max().isot
            except ValueError:
                pass
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
        ax = IRISPlotter(ndcube=self).plot(*args, **kwargs)
        set_axis_properties(ax)
        return ax

    @property
    def basic_wcs(self):
        """
        Return a standard astropy WCS, when one is available alongside the gWCS.
        """
        return self._basic_wcs

    def crop(self, *points, wcs=None, keepdims=False):
        manual_item = None
        if wcs is None:
            # Keep supporting the pre-gWCS raster shorthand where crop points
            # are just (spectral, sky) pairs with omitted time/step entries.
            manual_item = self._manual_crop_item(points, keepdims)
        if manual_item is not None:
            return self[manual_item]
        return super().crop(*points, wcs=wcs, keepdims=keepdims)

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
