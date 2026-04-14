import textwrap
import warnings
from numbers import Integral

import matplotlib.pyplot as plt

from astropy.wcs import WCS

from sunpy import log as logger
from sunpy.map import Map
from sunpy.util import MetaDict
from sunpy.util.exceptions import SunpyMetadataWarning
from sunraster import SpectrogramCube

from irispy.utils import calculate_dust_mask
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.utils.dust import remove_dust as _remove_dust
from irispy.visualization import IRISPlotter, set_axis_properties

__all__ = ["AIACube", "SJICube"]


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


class SJICube(SpectrogramCube):
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
    scaled : `bool`, optional
        Indicates if the data has been scaled.
    """

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
        scaled=None,
        **kwargs,
    ) -> None:
        self.scaled = scaled
        self.dust_masked = False
        self._basic_wcs = kwargs.pop("_basic_wcs", None)
        if self._basic_wcs is not None and not isinstance(self._basic_wcs, list):
            self._basic_wcs = [self._basic_wcs]
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

    def _get_basic_wcs_slice_item(self, item):
        basic_wcs_item = None
        if self._basic_wcs is not None and self.data.ndim == 3:
            if isinstance(item, (Integral, slice)):
                basic_wcs_item = item
            elif item is Ellipsis:
                basic_wcs_item = slice(None)
            elif isinstance(item, tuple):
                normalized_item = _normalize_tuple_index(item, self.data.ndim)
                if (
                    normalized_item is not None
                    and normalized_item
                    and isinstance(normalized_item[0], (Integral, slice))
                ):
                    basic_wcs_item = normalized_item[0]
        return basic_wcs_item

    def __getitem__(self, item):
        sliced_self = super().__getitem__(item)
        sliced_self.scaled = self.scaled
        basic_wcs_item = self._get_basic_wcs_slice_item(item)
        if basic_wcs_item is not None:
            sliced_self._basic_wcs = self._basic_wcs[basic_wcs_item]
        return sliced_self

    def plot(self, *args, **kwargs):
        cmap = kwargs.get("cmap")
        if not cmap:
            try:
                cmap = plt.get_cmap(name=f"irissji{int(self.meta['TWAVE1'])}")
            except Exception as e:  # NOQA: BLE001
                logger.debug(e)
                cmap = "viridis"
        kwargs["cmap"] = cmap
        ax = IRISPlotter(ndcube=self).plot(*args, **kwargs)
        set_axis_properties(ax)
        return ax

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
            # If undo kwarg IS set, unmask dust pixels.
            self.mask[dust_mask] = False
            self.dust_masked = False
        else:
            # If undo kwarg is NOT set, mask dust pixels.
            self.mask[dust_mask] = True
            self.dust_masked = True

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
        `irispy.sji.SJICube`
            Cleaned cube with the same metadata and coordinates as the original.
        """
        return remove_cosmic_rays(
            self,
            method=method,
            sigma=sigma,
            max_iters=max_iters,
            method_kwargs=method_kwargs,
        )

    def remove_dust(
        self,
        *,
        dust_mask=None,
        temporal_window=2,
        exposure_normalize=True,
        fallback="spatial",
        spatial_box=5,
    ):
        """
        Return a new cube with dust-darkened pixels repaired.

        This is a convenience wrapper around `irispy.utils.dust.remove_dust`.

        Parameters
        ----------
        dust_mask : `numpy.ndarray`, optional
            Boolean mask marking pixels to repair. If omitted, a mask is derived
            from data values.
        temporal_window : `int`, optional
            Number of neighboring frames on either side to use for temporal
            replacement.
        exposure_normalize : `bool`, optional
            If `True`, normalize temporal candidate pixels by exposure time when
            metadata are available.
        fallback : {``"spatial"``, None}, optional
            Fallback behavior when temporal replacement is unavailable.
        spatial_box : `int`, optional
            Size of the local median filter used by the spatial fallback.

        Returns
        -------
        `irispy.sji.SJICube`
            Cleaned cube with dust-darkened pixels repaired.
        """
        return _remove_dust(
            self,
            dust_mask=dust_mask,
            temporal_window=temporal_window,
            exposure_normalize=exposure_normalize,
            fallback=fallback,
            spatial_box=spatial_box,
        )

    @property
    def basic_wcs(self):
        """
        Returns a standard WCS instead of gWCS.
        """
        if self._basic_wcs is None:
            return None
        if isinstance(self._basic_wcs, MetaDict):
            return WCS(self._basic_wcs)
        return [WCS(wcs_header) for wcs_header in self._basic_wcs]

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
        if isinstance(index, int):
            idx_list = [index]
        elif index is None:
            idx_list = range(self.data.shape[0])
        else:
            idx_list = index

        # We can shortcut if the Cube has been reduced to a 2D slice
        if self.wcs.world_n_dim == 2:
            # TODO: Missing metadata
            return Map(self.data, self.basic_wcs)
        data_wcs = ((self.data[i], self.basic_wcs[i]) for i in idx_list)
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
            m.plot_settings["cmap"] = f"irissji{int(self.meta['TWAVE1'])}"
        return maps[0] if isinstance(index, int) else maps


class AIACube(SJICube):
    """
    Subclass of the SJICube.

    It is the same outside of the name.
    """

    def __str__(self) -> str:
        return super().__str__().replace("SJICube", "AIACube")
