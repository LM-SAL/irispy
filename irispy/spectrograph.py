import textwrap

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame

from ndcube import NDCollection
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube

from irispy._spectrograph_wcs import _SpectrogramCubeWCSMixin
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.visualization import IRISPlotter, finalize_iris_plot

__all__ = ["RasterCollection", "SpectrogramCube"]

RASTER_SEGMENT_STEP_SEARCH_RADIUS = 2
RASTER_SEGMENT_SLIT_SEARCH_RADIUS = 16


class SpectrogramCube(_SpectrogramCubeWCSMixin, SpecCube):
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
        self._raster_wcs_header = kwargs.pop("_raster_wcs_header", None)
        self._raster_pc_table = kwargs.pop("_raster_pc_table", None)
        self._raster_crval_table = kwargs.pop("_raster_crval_table", None)
        self._raster_observer = kwargs.pop("_raster_observer", None)
        self._memmap_path = kwargs.pop("_memmap_path", None)
        self._memmap_ext = kwargs.pop("_memmap_ext", None)
        self._flip = kwargs.pop("_flip", False)
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    @property
    def time(self):
        time = super().time
        # The gWCS TemporalFrame may return a Time object in JD format.
        # Switch to ISOT for a more readable representation.
        if time.format == "jd":
            time = time.copy()
            time.format = "isot"
        return time

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
        Split the cube into per-raster subcubes.
        """
        boundaries = self.raster_boundaries
        if not boundaries:
            return (self,)
        return tuple(self[raster_slice] for raster_slice in boundaries)

    def _search_segment_grid(self, segment_cube, target, step_indices, slit_indices):
        """Evaluate the gWCS over a grid and return the pixel nearest to *target*."""
        step_grid, slit_grid = np.meshgrid(step_indices, slit_indices, indexing="ij")
        sky = segment_cube.wcs.array_index_to_world(step_grid, slit_grid, np.zeros_like(step_grid))[1]
        separation = sky.separation(target.transform_to(sky.frame)).to_value(u.arcsec)
        local = np.unravel_index(np.nanargmin(separation), separation.shape)
        return float(separation[local]), int(step_grid[local]), int(slit_grid[local])

    def _nearest_raster_segment(self, target, *, clip):
        if not self._raster_boundaries:
            return None

        best_match = None
        for segment_index, raster_slice in enumerate(self.raster_boundaries):
            segment_cube = self[raster_slice]
            if segment_cube.basic_wcs is not None:
                guess_target = target.transform_to(wcs_to_celestial_frame(segment_cube.basic_wcs.celestial))
                step_guess, slit_guess = segment_cube.basic_wcs.celestial.world_to_array_index(guess_target)
                guess_valid = (
                    np.isfinite(step_guess)
                    and np.isfinite(slit_guess)
                    and 0 <= step_guess < segment_cube.shape[0]
                    and 0 <= slit_guess < segment_cube.shape[1]
                )
                if guess_valid:
                    step_guess = int(np.rint(np.clip(step_guess, 0, segment_cube.shape[0] - 1)))
                    slit_guess = int(np.rint(np.clip(slit_guess, 0, segment_cube.shape[1] - 1)))
                    step_indices = np.arange(
                        max(0, step_guess - RASTER_SEGMENT_STEP_SEARCH_RADIUS),
                        min(segment_cube.shape[0], step_guess + RASTER_SEGMENT_STEP_SEARCH_RADIUS + 1),
                        dtype=int,
                    )
                    slit_indices = np.arange(
                        max(0, slit_guess - RASTER_SEGMENT_SLIT_SEARCH_RADIUS),
                        min(segment_cube.shape[1], slit_guess + RASTER_SEGMENT_SLIT_SEARCH_RADIUS + 1),
                        dtype=int,
                    )
                    score, step, slit = self._search_segment_grid(segment_cube, target, step_indices, slit_indices)
                    if best_match is None or score < best_match[0]:
                        best_match = (score, segment_index, step, slit)
                    continue
                if not clip:
                    continue

            step_indices = np.arange(segment_cube.shape[0], dtype=int)
            slit_indices = np.arange(segment_cube.shape[1], dtype=int)
            score, step, slit = self._search_segment_grid(segment_cube, target, step_indices, slit_indices)
            if best_match is None or score < best_match[0]:
                best_match = (score, segment_index, step, slit)

        if best_match is None:
            if clip:
                return None
            msg = "Target is outside the raster bounds."
            raise ValueError(msg)
        return best_match

    def _crop_spectrum_at_pixel(self, cube, step_index, slit_index):
        """Extract a 1D spectrum from *cube* at the given pixel."""
        start = cube.wcs.array_index_to_world(step_index, slit_index, 0)
        stop = cube.wcs.array_index_to_world(step_index, slit_index, cube.shape[-1] - 1)
        return cube.crop(start, stop)

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
            target_in_frame = target.transform_to(wcs_to_celestial_frame(self.basic_wcs.celestial))
            step_index, slit_index = self.basic_wcs.celestial.world_to_array_index(target_in_frame)
            if clip:
                step_index = int(np.rint(np.clip(step_index, 0, self.shape[0] - 1)))
                slit_index = int(np.rint(np.clip(slit_index, 0, self.shape[1] - 1)))
            else:
                step_index = int(np.rint(step_index))
                slit_index = int(np.rint(slit_index))
                if not (0 <= step_index < self.shape[0] and 0 <= slit_index < self.shape[1]):
                    nearest_segment = self._nearest_raster_segment(target, clip=clip)
                    if nearest_segment is not None:
                        _, nearest_segment_index, step_index, slit_index = nearest_segment
                        return self._crop_spectrum_at_pixel(self.raster_slice(nearest_segment_index), step_index, slit_index)
                    msg = "Target is outside the raster bounds."
                    raise ValueError(msg)
            return self._crop_spectrum_at_pixel(self, step_index, slit_index)

        nearest_segment = self._nearest_raster_segment(target, clip=clip)
        if nearest_segment is not None:
            _, nearest_segment_index, step_index, slit_index = nearest_segment
            return self._crop_spectrum_at_pixel(self.raster_slice(nearest_segment_index), step_index, slit_index)

        sky_grid = self.axis_world_coords("custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat")[0]
        target_in_frame = target.transform_to(sky_grid.frame)
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
        return self._crop_spectrum_at_pixel(self, step_index, slit_index)

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
