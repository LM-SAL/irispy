import textwrap
import warnings
from numbers import Integral
from functools import cached_property

import numpy as np

import gwcs
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube.visualization import PlotterDescriptor
from sunpy.util import MetaDict
from sunpy.util.exceptions import SunpyMetadataWarning
from sunraster import SpectrogramCube

from irispy._wcs import _celestial_frame_from_cube, _ResolveNegativeIndicesMixin
from irispy.utils import calculate_dust_mask
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.utils.dust import remove_dust as _remove_dust
from irispy.visualization import SJIPlotter

__all__ = ["AIACube", "SJICube"]


class SJICube(_ResolveNegativeIndicesMixin, SpectrogramCube):
    """
    Class representing SJI Image described by a single WCS.

    Parameters
    ----------
    data : `numpy.ndarray`
        The array holding the actual data in this object.
    wcs : `astropy.wcs.WCS`
        The WCS object containing the axes information
    unit : `astropy.units.Unit` or `str`
        Unit for the dataset.
        Strings that can be converted to a Unit are allowed.
    meta : `dict` object
        Additional meta information about the dataset.
    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn't mandatory. If the
        uncertainty has no such attribute the uncertainty is stored as
        UnknownUncertainty.
        Defaults to None.
    mask : any type, optional
        Mask for the dataset. Masks should follow the numpy convention
        that valid data points are marked by False and invalid ones with True.
        Defaults to None.
    copy : `bool`, optional
        Indicates whether to save the arguments as copy. True copies every
        attribute before saving it while False tries to save every parameter
        as reference. Note however that it is not always possible to save the
        input as reference.
        Default is False.
    """

    plotter = PlotterDescriptor(default_type=SJIPlotter)

    def __init__(
        self,
        data,
        wcs,
        *,
        uncertainty=None,
        unit=None,
        meta=None,
        mask=None,
        copy=False,
        **kwargs,
    ) -> None:
        self.dust_masked = False
        super().__init__(
            data,
            wcs,
            uncertainty=uncertainty,
            mask=mask,
            meta=meta,
            unit=unit,
            copy=copy,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def __str__(self) -> str:
        if self.wcs.world_n_dim == 2:
            instance_start = self.global_coords.get("Time (UTC)")
            instance_end = None
        else:
            instance_start = self.wcs.pixel_to_world(0, 0, 0)[-1]
            instance_end = self.wcs.pixel_to_world(0, 0, self.data.shape[0] - 1)[-1]
        return textwrap.dedent(
            f"""
            SJICube
            -------
            Observatory:           {self.meta.get("TELESCOP", "IRIS")}
            Instrument:            {self.meta.get("INSTRUME")}
            Bandpass:              {self.meta.get("TWAVE1")}
            Obs Date:              {instance_start} -- {instance_end}
            Total Frames in Obs:   {self.meta.get("NBFRAMES")}
            Obs ID:                {self.meta.get("OBSID")}
            Obs Description:       {self.meta.get("OBS_DESC")}
            Axis Types:            {self.array_axis_physical_types}
            Roll:                  {self.meta.get("SAT_ROT")}
            Cube dimensions:       {self.shape}
            """,
        )

    def apply_dust_mask(self, *, undo=False):
        """
        Applies or undoes an update of the mask with the dust particles positions.

        Rewrite self.mask with/without the dust positions.

        Parameters
        ----------
        undo: `bool`
            If False, dust particles positions mask will be applied.
            If True, dust particles positions mask will be removed.
            Default=False
        """
        dust_mask = calculate_dust_mask(self.data)
        if undo:
            if self.mask is not None:
                self.mask = self.mask & ~dust_mask
            self.dust_masked = False
        else:
            if self.mask is None:
                self.mask = np.zeros(self.shape, dtype=bool)
            self.mask = self.mask | dust_mask
            self.dust_masked = True

    remove_cosmic_rays = remove_cosmic_rays
    remove_dust = _remove_dust

    celestial_frame = property(_celestial_frame_from_cube)

    @cached_property
    def fits_wcs(self):
        """
        Returns a standard WCS instead of gWCS.
        """
        headers = self.meta.get("frame_wcs_headers") if self.meta is not None else None
        low = self.wcs.low_level_wcs
        item = [slice(None)] * low.pixel_n_dim
        if isinstance(low, SlicedLowLevelWCS):
            low, item = low._wcs, list(low._slices_array)
        # Any other low-level WCS (e.g. a rebinned one) no longer matches the frame headers.
        if headers is None or not isinstance(low, gwcs.WCS):
            return None
        frames = headers[item[0]]
        spatial = tuple(item[1:])
        if isinstance(frames, MetaDict):
            return WCS(frames).slice(spatial, numpy_order=True)
        return [WCS(h).slice(spatial, numpy_order=True) for h in frames]

    def to_maps(self, index: int | list[int] | None = None):
        """
        Return SunPy Maps for the requested frame(s).

        Parameters
        ----------
        index : int, list, optional
            The index of the SJI steps you want.
            By default None which will return the entire cube as a map sequence.

        Returns
        -------
        `sunpy.map.Map` or `sunpy.map.MapSequence`
            A single Map if index is an int, otherwise a MapSequence.
        """
        from sunpy.map import Map  # NOQA: PLC0415

        if self.fits_wcs is None:
            msg = "This cube has no FITS WCS (for example, it was rebinned), so it cannot be converted to maps."
            raise ValueError(msg)
        if isinstance(index, Integral):
            idx_list = [index]
        elif index is None:
            idx_list = range(self.data.shape[0])
        else:
            idx_list = index

        # We can shortcut if the Cube has been reduced to a 2D slice
        if self.wcs.world_n_dim == 2:
            # TODO: Missing metadata
            return Map(self.data, self.fits_wcs)
        # pixel_to_world does not wrap negative indices the way the data and fits_wcs lists do.
        idx_list = [range(self.data.shape[0])[i] for i in idx_list]
        data_wcs = ((self.data[i], self.fits_wcs[i]) for i in idx_list)
        times_iso = (self.wcs.pixel_to_world(0, 0, i)[-1].utc.isot for i in idx_list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SunpyMetadataWarning)
            maps = Map(data_wcs, sequence=True)
        for m, t in zip(maps, times_iso, strict=True):
            m.meta["DATE-OBS"] = t
            m.meta["INSTRUME"] = self.meta.get("INSTRUME", "SJI")
            m.meta["TELESCOP"] = self.meta.get("TELESCOP", "IRIS")
            m.meta["EXPTIME"] = self.meta.get("EXPTIME", 0.0)
            m.meta["TWAVE1"] = self.meta.get("TWAVE1")
            # sunpy reads the wavelength (shown in the map name and plot title) from these.
            m.meta["WAVELNTH"] = self.meta.get("TWAVE1")
            m.meta["WAVEUNIT"] = "Angstrom"
            m.plot_settings["cmap"] = f"irissji{int(self.meta['TWAVE1'])}"
        return maps[0] if isinstance(index, Integral) else maps


class AIACube(SJICube):
    """
    Subclass of the SJICube.

    It is the same outside of the name.
    """

    def __str__(self) -> str:
        return super().__str__().replace("SJICube", "AIACube")
