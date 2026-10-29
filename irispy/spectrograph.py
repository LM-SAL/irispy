import textwrap
from copy import deepcopy
from numbers import Integral

import numpy as np

import astropy.units as u
import gwcs
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube import NDCollection
from ndcube.visualization import PlotterDescriptor
from ndcube.wcs.tools import unwrap_wcs_to_fitswcs
from ndcube.wcs.wrappers import ResampledLowLevelWCS
from sunraster import SpectrogramCube as SpecCube
from sunraster.spectrogram import _calculate_exposure_time_correction, _uncalculate_exposure_time_correction

from irispy._wcs import _celestial_frame_from_cube, _ResolveNegativeIndicesMixin, _wcs_array_axis_physical_types
from irispy.utils.constants import SLIT_WIDTH
from irispy.visualization import IRISPlotter, _make_map

__all__ = ["MosaicCube", "RasterCollection", "SpectrogramCube"]


class SpectrogramCube(_ResolveNegativeIndicesMixin, SpecCube):
    """
    Class representing spectrogram data described by a single WCS.

    A raster window is exposed as one cube, whether it comes from a single file,
    a combined multi-file raster, or a sit-and-stare observation.

    Parameters
    ----------
    data : `numpy.ndarray`
        The array holding the actual data in this object.
    wcs : `astropy.wcs.WCS` or ``gwcs.WCS``
        The WCS object containing the axes' information
    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn't mandatory. If the uncertainty
        has no such attribute the uncertainty is stored as UnknownUncertainty.
        Defaults to None.
    unit : `astropy.units.Unit` or `str`, optional
        Unit for the dataset. Strings that can be converted to a Unit are allowed.
        Defaults to None, which takes the unit of ``data`` if it is a `~astropy.units.Quantity`.
    meta : `dict` object, optional
        Additional meta information about the dataset, such as the
        `~irispy.meta.SGMeta` of the observation the data comes from.
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

    Examples
    --------
    The array axes are raster, raster step or time, slit position and wavelength; the
    window of one raster file has one raster:

    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import read_files
    >>> filename = get_test_filepath(
    ...     "sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits"
    ... )
    >>> cube = read_files(filename, spectral_windows="Si IV 1403")["Si IV 1403"]
    >>> cube.shape, cube.unit, cube.wavelength_axis
    ((1, 187, 40, 29), Unit("DN_IRIS_FUV"), 3)
    >>> cube.meta.spectral_window, cube.meta.rest_wavelength
    ('Si IV 1403', <Quantity 140.277 nm>)
    >>> cube.spectral_dispersion.to("pm")
    <Quantity 25.44 pm>
    >>> cube[0, :, 10:20, 5:10].shape
    (187, 10, 5)
    """

    plotter = PlotterDescriptor(default_type=IRISPlotter)

    def __init__(self, data, wcs, *, uncertainty=None, unit=None, meta=None, mask=None, copy=False, **kwargs) -> None:
        super().__init__(data, wcs, unit=unit, uncertainty=uncertainty, mask=mask, meta=meta, copy=copy, **kwargs)

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def __str__(self) -> str:
        try:
            times = self.time
            obs_date = f"{times.min().isot} -- {times.max().isot}"
        except ValueError:  # a cube built on a WCS with no time, such as a FITS WCS
            obs_date = None
        return textwrap.dedent(
            f"""
            SpectrogramCube
            ---------------
            Obs ID:             {self.meta.get("OBSID")}
            Obs Description:    {self.meta.get("OBS_DESC")}
            Obs Date:           {obs_date}
            Data shape:         {self.shape}
            Axis Types:         {_wcs_array_axis_physical_types(self)}
            Roll:               {self.meta.get("SAT_ROT")}
            """,
        )

    celestial_frame = property(_celestial_frame_from_cube)

    @property
    def fits_wcs(self):
        """
        The FITS WCS of the cube, following its slicing.

        For the cube of a raster window, a list with the FITS ``-TAB`` WCS of each
        raster, with the pointing of each exposure, as `irispy.sji.SJICube.fits_wcs`
        gives one per frame; for one raster, ``cube[k]``, its WCS. A mosaic's own WCS.
        `None` when the cube has none, as after rebinning.
        """
        root, low = self.meta.get("fits_wcs"), self.wcs.low_level_wcs
        item = [slice(None)] * low.pixel_n_dim
        if isinstance(low, SlicedLowLevelWCS):
            low, item = low._wcs, list(low._slices_array)
        if isinstance(low, WCS):
            root = low
        elif root is None or not isinstance(low, gwcs.WCS):
            return None
        if isinstance(root, list):
            if not isinstance(item[0], Integral):
                return [_sliced(wcs, item[1:]) for wcs in root[item[0]]]
            root, item = root[item[0]], item[1:]
        return _sliced(root, item)

    def apply_exposure_time_correction(self, undo=False, force=False):  # NOQA: FBT002 (sunraster signature)
        exposure_time = self.exposure_time.to_value(u.s)
        if np.ndim(exposure_time) < 2:
            return super().apply_exposure_time_correction(undo=undo, force=force)
        # sunraster expects one exposure axis; a combined cube has two (scan and step).
        # TODO: drop this and sunraster's private helpers once its apply_exposure_time_correction takes several
        # exposure axes, or is replaced by plain arithmetic as sunpy/sunraster#284 proposes.
        axes = self._get_axis_coord_index(self._exposure_time_name, self._exposure_time_loc)
        item = tuple(slice(None) if axis in axes else np.newaxis for axis in range(self.data.ndim))
        correct = _uncalculate_exposure_time_correction if undo else _calculate_exposure_time_correction
        new_cube = deepcopy(self)
        new_cube._data, new_cube._uncertainty, new_cube._unit = correct(
            self.data, self.uncertainty, self.unit, exposure_time[item], force=force
        )
        return new_cube

    @property
    def time(self):
        """
        The time of each exposure, an `~astropy.time.Time`; a single time for one
        exposure.
        """
        if "time" not in self.wcs.world_axis_physical_types:
            # Indexing one exposure drops the time from the WCS into the global coordinates
            dropped = [coordinate for coordinate in self.global_coords.values() if isinstance(coordinate, Time)]
            if dropped:
                return dropped[0]
        return super().time

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
        fits_wcs = root if hasattr(root, "wcs") else self.meta.get("fits_wcs")
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
                for axis, physical_types in enumerate(_wcs_array_axis_physical_types(self))
                if physical_types and "em.wl" in physical_types
            )
        except StopIteration:
            msg = "Could not identify a spectral wavelength axis on the cube"
            raise ValueError(msg) from None


def _sliced(wcs, item):
    return wcs if all(index == slice(None) for index in item) else wcs.slice(tuple(item), numpy_order=True)


def _wavelength_indices(wavelengths, wavelength_range, *, allow_empty=False):
    """
    Indices of the ``wavelengths`` between two wavelengths, in either order, inclusive.

    The bounds are widened by 1e-6 Å, so that a wavelength typed from the grid is kept
    despite the rounding of the WCS unit conversion. With ``allow_empty=True``, a range
    without samples returns an empty array.
    """
    wavelength_range = u.Quantity(wavelength_range)
    if wavelength_range.shape != (2,):
        msg = f"Expected two wavelengths, got {wavelength_range}"
        raise ValueError(msg)
    low, high = np.sort(wavelength_range) + [-1e-6, 1e-6] * u.AA
    indices = np.flatnonzero((wavelengths >= low) & (wavelengths <= high))
    if not indices.size and not allow_empty:
        msg = f"No wavelengths between {wavelength_range[0]} and {wavelength_range[1]}"
        raise ValueError(msg)
    return indices


class MosaicCube(SpectrogramCube):
    """
    An IRIS full-disk mosaic of one spectral window, read by `irispy.io.read_mosaic`.

    The array axes are solar X, solar Y and wavelength, as in a Level 2 raster.
    `to_map`, `~irispy.utils.moments.calculate_moments`, `~irispy.utils.moments.average_window`,
    `~irispy.utils.red_blue.calculate_red_blue_asymmetry`, `~irispy.utils.rgb.calculate_rgb`,
    `~irispy.utils.rgb.plot_rgb`, `~irispy.utils.mg_features.calculate_mg_features` and
    `~irispy.utils.spectrograph.subtract_background` take a mosaic. The burst finders and
    `~irispy.utils.spectrograph.radiometric_calibration` do not: they need the exposure time
    of each raster step, which a mosaic does not have.
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

    @u.quantity_input
    def to_map(self, wavelength: u.AA):
        """
        Return a `sunpy.map.Map` at one wavelength or averaged over a range.

        As in `~irispy.utils.moments.average_window`, masked values are left out of the mean
        and positions with a saturated sample (see :ref:`irispy-tutorial-saturated-data`) are NaN.
        The color map is that of the nearest
        slit-jaw passband.

        Parameters
        ----------
        wavelength : `astropy.units.Quantity`
            One wavelength, for the nearest mosaic wavelength, or two, for the mean
            between them (inclusive).

        Returns
        -------
        `sunpy.map.GenericMap`
            In the cube's unit, with DN for the IRIS data numbers, and masked where it is NaN.

        Raises
        ------
        ValueError
            If one wavelength is more than half a wavelength step outside the mosaic,
            or two have no mosaic wavelength between them.
        """
        from irispy.utils.moments import average_window  # NOQA: PLC0415

        wavelengths = u.Quantity(self.axis_world_coords(self.wavelength_axis)[0], u.AA)
        if wavelength.isscalar:
            distance = np.abs(wavelengths - wavelength)
            if distance.min() > np.abs(self.spectral_dispersion) / 2:
                msg = (
                    f"{wavelength} is more than half a wavelength step outside the mosaic wavelengths, "
                    f"{wavelengths[0]:.3f} to {wavelengths[-1]:.3f}"
                )
                raise ValueError(msg)
            nearest = wavelengths[np.argmin(distance)]
            wavelength = u.Quantity([nearest, nearest])
        mean_wavelength = wavelengths[_wavelength_indices(wavelengths, wavelength)].mean()
        window = average_window(self, wavelength)
        header = unwrap_wcs_to_fitswcs(self.wcs)[0].celestial.swapaxes(0, 1).to_header()
        header.update(
            {
                "TELESCOP": "IRIS",
                "OBSRVTRY": "IRIS",
                "INSTRUME": "SPEC",  # the Level 2 spectrograph value
                "EXPTIME": self.meta.get("EXPTIME"),
                "DATE-END": self.meta.get("DATE_END"),
                "WAVELNTH": mean_wavelength.to_value(u.AA),
                "WAVEUNIT": "Angstrom",
            }
        )
        # A map is (solar Y, solar X)
        sunpy_map = _make_map(window.data.T, header, window.unit, window.mask.T)
        passband = min((1330, 1400, 2796, 2832), key=lambda band: abs(band - mean_wavelength.to_value(u.AA)))
        sunpy_map.plot_settings["cmap"] = f"irissji{passband}"
        return sunpy_map


class RasterCollection(NDCollection):
    """
    A `~ndcube.NDCollection` of `SpectrogramCube` objects with aligned raster axes.

    `irispy.io.read_files` and `irispy.io.read_spectrograph_lvl2` key it by spectral
    window, and the analysis functions that return several maps, such as
    `irispy.utils.mg_features.calculate_mg_features`, by map name.
    """

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
            Raster Collection
            -----------------
            Keys: {tuple(self.keys())}
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
