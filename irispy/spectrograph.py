import textwrap
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from ndcube import NDCollection
from ndcube.wcs.tools import unwrap_wcs_to_fitswcs
from sunpy import log as logger
from sunraster import SpectrogramCube as SpecCube

from irispy._spectrograph_wcs import (
    _SPECTROGRAM_CUBE_METADATA_DEFAULTS,
    _SPECTROGRAM_CUBE_METADATA_KWARGS,
    _SpectrogramCubeWCSMixin,
)
from irispy._wcs import _celestial_frame_from_cube
from irispy.utils.constants import SLIT_WIDTH
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

    def _new_instance(self, **kwargs):
        for attr in _SPECTROGRAM_CUBE_METADATA_KWARGS:
            kwargs.setdefault(attr, getattr(self, attr, _SPECTROGRAM_CUBE_METADATA_DEFAULTS.get(attr)))
        return super()._new_instance(**kwargs)

    def to_nddata(self, *args, nddata_type=None, **kwargs):
        if nddata_type is None:
            return super().to_nddata(*args, **kwargs)
        try:
            copies_metadata = issubclass(nddata_type, SpectrogramCube)
        except TypeError:
            copies_metadata = False
        if copies_metadata:
            for attr in _SPECTROGRAM_CUBE_METADATA_KWARGS:
                if hasattr(self, attr):
                    kwargs.setdefault(attr, "copy")
        return super().to_nddata(*args, nddata_type=nddata_type, **kwargs)

    @property
    def time(self):
        time = super().time
        if time.format == "jd":
            time.format = "isot"
        return time

    def __getitem__(self, item):
        normalized_item = self._normalize_fits_wcs_item(item)
        item_for_super = normalized_item if normalized_item is not None else item
        sliced_self = super().__getitem__(item_for_super)
        if isinstance(sliced_self, SpectrogramCube):
            sliced_self._fits_wcs = self._slice_fits_wcs(item_for_super)
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

    def plot(self, *args, **kwargs):
        """
        Plot the spectrogram cube.

        Parameters
        ----------
        **kwargs
            Passed to the ndcube plotting machinery, e.g. ``slider_labels``
            to override animation slider labels.
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
        return finalize_iris_plot(IRISPlotter(ndcube=self).plot(*args, **kwargs), kwargs.get("axes_coordinates"))

    celestial_frame = property(_celestial_frame_from_cube)

    @property
    def fits_wcs(self):
        """
        The plain FITS WCS built from the window header, or `None` when no single FITS
        WCS describes this cube (for example a combined multi-file cube).
        """
        return self._fits_wcs

    def raster_slice(self, index):
        """
        Return the subcube corresponding to one original raster.
        """
        if not isinstance(index, Integral):
            msg = "Raster index must be an integer."
            raise TypeError(msg)

        if self._separate_raster_axis:
            n_rasters = self.shape[0]
        elif self._raster_boundaries is None:
            n_rasters = 1
        else:
            n_rasters = len(self._raster_boundaries)

        if index < 0:
            index += n_rasters
        if index < 0 or index >= n_rasters:
            msg = "Raster index out of range."
            raise IndexError(msg)

        if self._separate_raster_axis:
            return self[index]
        if self._raster_boundaries is None:
            return self
        return self[slice(*self._raster_boundaries[index])]

    def split_rasters(self):
        """
        Split the cube into per-raster subcubes.
        """
        if self._separate_raster_axis:
            return tuple(self[i] for i in range(self.shape[0]))
        if self._raster_boundaries is None:
            return (self,)
        return tuple(self[slice(start, stop)] for start, stop in self._raster_boundaries)

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
        Raw FITS WCS keywords (``Wcsprm``) backing this cube.

        Raster cubes may carry a gWCS on ``self.wcs`` and a FITS WCS bridge on
        ``self.fits_wcs``. This prefers a direct FITS WCS and falls back to unwrapping
        sliced FITS-WCS adapters when needed.
        """
        for wcs in (self.wcs, self.fits_wcs):
            if wcs is not None and hasattr(wcs, "wcs"):
                return wcs.wcs
        fits_wcs = self.fits_wcs
        if fits_wcs is None and self._fits_wcs_segments:
            fits_wcs = self._fits_wcs_segments[0][2]
        if fits_wcs is not None:
            return unwrap_wcs_to_fitswcs(fits_wcs)[0].wcs
        return unwrap_wcs_to_fitswcs(self.wcs)[0].wcs

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
