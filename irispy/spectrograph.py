import textwrap
from copy import deepcopy
from numbers import Integral

import numpy as np

import astropy.units as u
import gwcs
from astropy.time import Time
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube import NDCollection
from ndcube.visualization import PlotterDescriptor
from ndcube.wcs.wrappers import ResampledLowLevelWCS
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube
from sunraster.spectrogram import _calculate_exposure_time_correction, _uncalculate_exposure_time_correction

from irispy._spectrograph_wcs import _raster_crop_bounds
from irispy._wcs import _celestial_frame_from_cube, _ResolveNegativeIndicesMixin
from irispy.utils.constants import SLIT_WIDTH
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.visualization import SpectrogramPlotter

__all__ = ["RasterCollection", "SpectrogramCube"]


class SpectrogramCube(_ResolveNegativeIndicesMixin, SpecCube):
    """
    Class representing spectrogram data described by a single WCS.

    A raster window is exposed as one cube, whether it comes from a single file,
    a combined multi-file raster, or a sit-and-stare observation.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.
    wcs: `astropy.wcs.WCS` or ``gwcs.WCS``
        The WCS object containing the axes' information
    unit : `astropy.units.Unit` or `str`, optional
        Unit for the dataset. Strings that can be converted to a Unit are allowed.
        Defaults to None, which takes the unit of ``data`` if it is a `~astropy.units.Quantity`.
    meta : `dict` object, optional
        Additional meta information about the dataset, such as the
        `~irispy.meta.SGMeta` of the observation the data comes from.
        Defaults to None.
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

    plotter = PlotterDescriptor(default_type=SpectrogramPlotter)

    def __init__(self, data, wcs, uncertainty=None, unit=None, meta=None, *, mask=None, copy=False, **kwargs) -> None:
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def _time_bounds(self):
        if "time" in self.global_coords:
            return self.global_coords["time"].min().isot, self.global_coords["time"].max().isot
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

    def plot(self, *args, **kwargs):
        return self.plotter.plot(*args, **kwargs)

    celestial_frame = property(_celestial_frame_from_cube)
    _get_crop_bounds = _raster_crop_bounds

    def axis_world_coords(self, *axes, **kwargs):
        # TODO: remove once ndcube returns C-ordered world coordinates or astropy's Time string
        # formats respect memory order. ndcube transposes N-D coordinates (Fortran order), and
        # astropy's isot/iso/datetime64 then emit them out of order, scrambling a 4D cube's times.
        # astropy main emits them in order since astropy/astropy#19942, which is not released yet.
        return tuple(
            coord.copy() if isinstance(coord, Time) else coord for coord in super().axis_world_coords(*axes, **kwargs)
        )

    @property
    def fits_wcs(self):
        """
        The plain FITS WCS built from the window header, or `None` when no single FITS
        WCS describes this cube (for example a combined multi-file cube).
        """
        root, low = self.meta.get("fits_wcs") if hasattr(self.meta, "get") else None, self.wcs.low_level_wcs
        item = [slice(None)] * low.pixel_n_dim
        if isinstance(low, SlicedLowLevelWCS):
            low, item = low._wcs, list(low._slices_array)
        if root is None or not isinstance(low, gwcs.WCS):
            return None
        if isinstance(root, list):
            if not isinstance(item[0], Integral):
                return None
            root, item = root[item[0]], item[1:]
        return root if all(i == slice(None) for i in item) else root.slice(tuple(item), numpy_order=True)

    @property
    def _separate_raster_axis(self):
        return "raster_scan" in self.wcs.low_level_wcs.world_axis_names

    def raster_slice(self, index):
        """
        Return the subcube corresponding to one original raster.
        """
        if not isinstance(index, Integral):
            msg = "Raster index must be an integer."
            raise TypeError(msg)

        n_rasters = self.shape[0] if self._separate_raster_axis else 1
        if not -n_rasters <= index < n_rasters:
            msg = "Raster index out of range."
            raise IndexError(msg)

        return self[index] if self._separate_raster_axis else self

    def split_rasters(self):
        """
        Split the cube into per-raster subcubes.
        """
        return tuple(self[i] for i in range(self.shape[0])) if self._separate_raster_axis else (self,)

    def apply_exposure_time_correction(self, undo=False, force=False):  # NOQA: FBT002 (sunraster signature)
        exposure_time = self.exposure_time.to_value(u.s)
        if np.ndim(exposure_time) < 2:
            return super().apply_exposure_time_correction(undo=undo, force=force)
        # sunraster expects one exposure axis; a combined cube has two (scan and step).
        axes = self._get_axis_coord_index(self._exposure_time_name, self._exposure_time_loc)
        item = tuple(slice(None) if axis in axes else np.newaxis for axis in range(self.data.ndim))
        correct = _uncalculate_exposure_time_correction if undo else _calculate_exposure_time_correction
        new_cube = deepcopy(self)
        new_cube._data, new_cube._uncertainty, new_cube._unit = correct(
            self.data, self.uncertainty, self.unit, exposure_time[item], force=force
        )
        return new_cube

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

    @property
    def _fits_wcsprm(self):
        """
        Raw FITS WCS keywords (``Wcsprm``) of the native pixels of this cube.

        Slicing and rebinning keep the native pixel scales, so these come from the
        unsliced WCS: ``self.wcs`` for a cube built on a FITS WCS, otherwise the
        per-file ``fits_wcs`` in ``meta``.
        """
        root = self.wcs.low_level_wcs
        while isinstance(root, (SlicedLowLevelWCS, ResampledLowLevelWCS)):
            root = root._wcs
        fits_wcs = root if hasattr(root, "wcs") else self.meta.get("fits_wcs") if hasattr(self.meta, "get") else None
        fits_wcs = fits_wcs[0] if isinstance(fits_wcs, list) else fits_wcs
        if fits_wcs is None:
            msg = "This cube has no FITS WCS to take its pixel scales from."
            raise ValueError(msg)
        return fits_wcs.wcs

    @property
    def spectral_dispersion(self):
        """
        Spectral dispersion per pixel along the wavelength axis.
        """
        wcs = self._fits_wcsprm
        mask = np.array([ctype == "WAVE" for ctype in wcs.ctype])
        if not mask.any():
            msg = "Cannot determine spectral axis (no WAVE ctype in WCS) for spectral_dispersion"
            raise ValueError(msg)
        idx = np.argmax(mask)
        return wcs.cdelt[idx] * u.Unit(wcs.cunit[idx])

    @property
    def solid_angle(self):
        """
        Solid angle per spatial pixel (slit width x spatial pixel scale).
        """
        wcs = self._fits_wcsprm
        mask = np.array(["HPLT" in ctype for ctype in wcs.ctype])
        if not mask.any():
            msg = "Cannot determine latitude axis (no HPLT ctype in WCS) for solid_angle computation"
            raise ValueError(msg)
        lat_idx = np.argmax(mask)
        return wcs.cdelt[lat_idx] * u.Unit(wcs.cunit[lat_idx]) * SLIT_WIDTH

    @property
    def wavelength_axis(self):
        """
        Index of the spectral (wavelength) axis.
        """
        try:
            return next(
                axis
                for axis, physical_types in enumerate(self.array_axis_physical_types)
                if physical_types and "em.wl" in physical_types
            )
        except StopIteration:
            msg = "Could not identify a spectral wavelength axis on the cube"
            raise ValueError(msg) from None


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

    @property
    def aligned_axis_physical_types(self):
        # NDCollection builds each tuple from a set, whose order changes from run to run.
        # TODO: remove once irispy requires an ndcube with sunpy/ndcube#983.
        types = super().aligned_axis_physical_types
        return None if types is None else [tuple(sorted(axis_types)) for axis_types in types]
