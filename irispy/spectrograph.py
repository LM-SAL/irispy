import textwrap

import numpy as np

import astropy.units as u

from ndcube import NDCollection
from ndcube.visualization import PlotterDescriptor
from ndcube.wcs.tools import unwrap_wcs_to_fitswcs
from sunraster import SpectrogramCube as SpecCube
from sunraster import SpectrogramSequence as SpecSeq

from irispy._wcs import _ResolveNegativeIndicesMixin
from irispy.utils.constants import SLIT_WIDTH
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.visualization import IRISSequencePlotter, SpectrogramPlotter

__all__ = ["MosaicCube", "RasterCollection", "SpectrogramCube", "SpectrogramCubeSequence"]


class SpectrogramCube(_ResolveNegativeIndicesMixin, SpecCube):
    """
    Class representing spectrogram data described by a single WCS.

    Idea is that this class holds one complete raster scan or a sit and stare.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.
    wcs: `astropy.wcs.WCS`
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
            Obs ID:             {self.meta.get("OBSID")}
            Obs Description:    {self.meta.get("OBS_DESC")}
            Obs Date:           {instance_start} -- {instance_end}
            Data shape:         {self.shape}
            Axis Types:         {self.array_axis_physical_types}
            Roll:               {self.meta.get("SAT_ROT")}
            """,
        )

    remove_cosmic_rays = remove_cosmic_rays

    @property
    def _fits_wcs(self):
        """
        Underlying FITS WCS object, unwrapped if necessary.
        """
        if hasattr(self.wcs, "wcs"):
            return self.wcs.wcs
        return unwrap_wcs_to_fitswcs(self.wcs)[0].wcs

    @property
    def spectral_dispersion(self):
        """
        Spectral dispersion per pixel along the wavelength axis.
        """
        wcs = self._fits_wcs
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
        wcs = self._fits_wcs
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


def _wavelength_indices(wavelengths, wavelength_range, *, allow_empty=False):
    """
    Indices of the ``wavelengths`` between two wavelengths, inclusive.

    The bounds are widened by 1e-6 Å, so that a wavelength typed from the grid is kept
    despite the rounding of the WCS unit conversion. With ``allow_empty=True``, a range
    without samples returns an empty array.
    """
    wavelength_range = u.Quantity(wavelength_range, u.AA)
    if wavelength_range.shape != (2,):
        msg = f"Expected two wavelengths, got {wavelength_range}"
        raise ValueError(msg)
    low, high = wavelength_range + [-1e-6, 1e-6] * u.AA
    indices = np.flatnonzero((wavelengths >= low) & (wavelengths <= high))
    if not indices.size and not allow_empty:
        msg = f"No wavelengths between {wavelength_range[0]} and {wavelength_range[1]}"
        raise ValueError(msg)
    return indices


class MosaicCube(SpectrogramCube):
    """
    An IRIS full-disk mosaic of one spectral window, read by `irispy.io.read_mosaic`.

    The array axes are (wavelength, solar Y, solar X), the reverse of a Level 2 raster,
    so the raster analysis helpers reject it; use `to_map` and
    `~irispy.utils.moments.calculate_moments`.
    """

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
            MosaicCube
            ----------
            Obs ID:          {self.meta.get("OBSID")}
            Rest wavelength: {self.meta.get("LAMREF")} Angstrom
            Obs Date:        {self.meta.get("DATE_OBS")} -- {self.meta.get("DATE_END")}
            Data shape:      {self.shape}
            Axis Types:      {self.array_axis_physical_types}
            """,
        )

    def to_map(self, wavelength):
        """
        Return a `sunpy.map.Map` at one wavelength or averaged over a range.

        The color map is that of the nearest slit-jaw passband. Masked values are left
        out of the average.

        Parameters
        ----------
        wavelength : `astropy.units.Quantity`
            One wavelength, for the nearest mosaic wavelength, or two, for the mean
            between them (inclusive).

        Returns
        -------
        `sunpy.map.GenericMap`
        """
        from sunpy.map import Map  # NOQA: PLC0415

        wavelength = u.Quantity(wavelength, u.AA)
        axis = self.wavelength_axis
        wavelengths = u.Quantity(self.axis_world_coords(axis)[0], u.AA)
        if wavelength.isscalar:
            indices = [np.argmin(np.abs(wavelengths - wavelength))]
        else:
            indices = _wavelength_indices(wavelengths, wavelength)
        data = np.take(self.data, indices, axis=axis)
        if self.mask is not None:
            data = np.where(np.take(self.mask, indices, axis=axis), np.nan, data)
        # A masked mean leaves positions with no unmasked value NaN, without a warning.
        data = np.ma.masked_invalid(data).mean(axis=axis).filled(np.nan)
        header = unwrap_wcs_to_fitswcs(self.wcs)[0].celestial.to_header()
        mean_wavelength = wavelengths[indices].mean()
        header.update(
            {
                "TELESCOP": "IRIS",
                "OBSRVTRY": "IRIS",
                "INSTRUME": "SPEC",  # the Level 2 spectrograph value
                "EXPTIME": self.meta.get("EXPTIME"),
                "DATE-END": self.meta.get("DATE_END"),
                "WAVELNTH": mean_wavelength.to_value(u.AA),
                "WAVEUNIT": "Angstrom",
                "BUNIT": "DN",
            }
        )
        sunpy_map = Map(data, header)
        passband = min((1330, 1400, 2796, 2832), key=lambda band: abs(band - mean_wavelength.to_value(u.AA)))
        sunpy_map.plot_settings["cmap"] = f"irissji{passband}"
        return sunpy_map


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

    plotter = PlotterDescriptor(default_type=IRISSequencePlotter)

    def __init__(self, data_list, meta=None, common_axis=0, **kwargs) -> None:
        # Check that all spectrograms are from same spectral window and OBS ID.
        if len(np.unique([cube.meta["OBSID"] for cube in data_list])) != 1:
            msg = "Constituent SpectrogramCube objects must have same value of 'OBSID' in its meta."
            raise ValueError(msg)
        super().__init__(data_list, meta=meta, common_axis=common_axis, **kwargs)


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

    @property
    def aligned_axis_physical_types(self):
        # NDCollection builds each tuple from a set, whose order changes from run to run.
        types = super().aligned_axis_physical_types
        return None if types is None else [tuple(sorted(axis_types)) for axis_types in types]
