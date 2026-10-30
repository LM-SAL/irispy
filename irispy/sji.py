import textwrap
from numbers import Integral
from functools import cached_property

import gwcs
from astropy.wcs import WCS

from ndcube.visualization import PlotterDescriptor
from sunpy.util import MetaDict
from sunraster import SpectrogramCube

from irispy._wcs import (
    _celestial_frame_from_cube,
    _fits_wcs_from_cube,
    _ResolveNegativeIndicesMixin,
    _root_and_item,
    _wcs_array_axis_physical_types,
)
from irispy.utils._spectral import check_scaled
from irispy.visualization import SJIPlotter, _make_map, _sji_cmap_name

__all__ = ["AIACube", "SJICube", "SOTCube"]


def _slice_header(header, spatial):
    """
    The FITS header of one frame after the spatial slice ``spatial`` (numpy order) of
    its cube, in the file's own units.

    ``preserve_units`` keeps the header's arcsec keywords; without it ``to_header``
    rewrites the celestial keywords in degrees.
    """
    # ndcube spells a full axis slice(0, n, None)
    shape = (header["naxis2"], header["naxis1"])
    if all(isinstance(axis, slice) and axis.indices(n) == (0, n, 1) for axis, n in zip(spatial, shape, strict=True)):
        return header
    sliced = WCS(header, preserve_units=True).slice(spatial, numpy_order=True)
    header = header.copy()
    header.update(sliced.to_header())
    header["naxis1"], header["naxis2"] = sliced.pixel_shape
    return header


class SJICube(_ResolveNegativeIndicesMixin, SpectrogramCube):
    """
    Class representing SJI Image described by a single WCS.

    Parameters
    ----------
    data : `numpy.ndarray`
        The array holding the actual data in this object.
    wcs : `astropy.wcs.WCS`
        The WCS object containing the axes information
    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn't mandatory. If the
        uncertainty has no such attribute the uncertainty is stored as
        UnknownUncertainty.
        Defaults to None.
    unit : `astropy.units.Unit` or `str`
        Unit for the dataset.
        Strings that can be converted to a Unit are allowed.
    meta : `dict` object
        Additional meta information about the dataset.
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

    Examples
    --------
    The array axes are time, solar Y and solar X; the exposure times are an extra coordinate:

    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import read_files
    >>> cube = read_files(
    ...     get_test_filepath(
    ...         "sns/iris_l2_20210905_001833_3620258102_SJI_1330_t000_test.fits"
    ...     )
    ... )
    >>> cube.shape, cube.unit
    ((52, 40, 37), Unit("DN_IRIS_SJI_1330"))
    >>> cube.meta.spectral_window, cube.meta.observing_mode_id
    ('1330', 3620258102)
    >>> cube[0, 10:20, 10:20].shape
    (10, 10)
    >>> cube.exposure_time[:2]
    <Quantity [8.00004959, 8.00007725] s>
    """

    plotter = PlotterDescriptor(default_type=SJIPlotter)

    def __init__(self, data, wcs, *, uncertainty=None, unit=None, meta=None, mask=None, copy=False, **kwargs) -> None:
        super().__init__(data, wcs, uncertainty=uncertainty, unit=unit, meta=meta, mask=mask, copy=copy, **kwargs)

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def __str__(self) -> str:
        if self.wcs.world_n_dim == 2:
            instance_start = self.global_coords.get("Time (UTC)")
            instance_end = None
        else:
            instance_start, instance_end = self.time[[0, -1]]
        return textwrap.dedent(
            f"""
            SJICube
            -------
            Observatory:           {self.meta.get("TELESCOP", "IRIS")}
            Instrument:            {self.meta.get("INSTRUME")}
            Bandpass:              {self.meta.spectral_window}
            Obs Date:              {instance_start} -- {instance_end}
            Obs ID:                {self.meta.get("OBSID")}
            Obs Description:       {self.meta.get("OBS_DESC")}
            Axis Types:            {_wcs_array_axis_physical_types(self)}
            Roll:                  {self.meta.get("SAT_ROT")}
            Cube dimensions:       {self.shape}
            """,
        )

    @property
    def time(self):
        """
        The time of each frame, an `~astropy.time.Time`; a single time for one frame.
        """
        # Indexing one frame moves its time from the WCS to the global coordinates, where sunraster does not look
        if "Time (UTC)" in self.global_coords:
            return self.global_coords["Time (UTC)"]
        return super().time

    @property
    def exposure_time(self):
        """
        The exposure time of each frame; a single value for one frame.
        """
        # Indexing one frame moves the exposure time from the extra coordinates to the global
        # coordinates, where sunraster does not look
        if "exposure time" in self.global_coords:
            return self.global_coords["exposure time"]
        return super().exposure_time

    def _get_axis_coord_index(self, axis_name, coord_loc):
        # The exposure time of a slit-jaw cube is an extra coordinate, whose axis sunraster
        # gives as a pixel axis int, where apply_exposure_time_correction unpacks a tuple
        # of array axes, as for a coordinate in the meta.
        index = super()._get_axis_coord_index(axis_name, coord_loc)
        if coord_loc == "extra_coords":
            return (self.data.ndim - 1 - index,)
        return index

    celestial_frame = property(_celestial_frame_from_cube)
    fits_wcs = cached_property(_fits_wcs_from_cube)

    @cached_property
    def _frame_headers(self):
        """
        The FITS header of each frame of this cube, following its slicing: one header
        for a single frame, a list for a time axis, `None` when the cube has no frame
        headers or an integer spatial index left a 1-D image, which no header describes.
        """
        headers = self.meta.get("frame_wcs_headers")
        root, item = _root_and_item(self)
        # A rebinned cube no longer matches the frame headers
        if headers is None or not isinstance(root, gwcs.WCS) or not all(isinstance(axis, slice) for axis in item[1:]):
            return None
        frames, spatial = headers[item[0]], tuple(item[1:])
        if isinstance(frames, MetaDict):
            return _slice_header(frames, spatial)
        return [_slice_header(h, spatial) for h in frames]

    def _fits_wcs_roots(self, item):
        # The file's header of each frame as a WCS, built for the frames asked for only
        headers = self.meta.get("frame_wcs_headers")
        if headers is None:
            return None
        frames = headers[item]
        return WCS(frames) if isinstance(frames, MetaDict) else [WCS(h) for h in frames]

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
            Each map is built from its frame's FITS header, so sunpy picks the map class
            from the instrument (`~sunpy.map.sources.SJIMap` for slit-jaw images), and the
            map has the cube's unit, with DN for the IRIS data numbers such as ``DN_IRIS_SJI_1330``,
            the frame's mask and the frame's own exposure time. Slit-jaw and AIA maps
            plot with the color table of their passband, such as ``irissji1330`` or ``sdoaia304``,
            and Hinode/SOT maps with ``hinodesotintensity``.

        Examples
        --------
        >>> from irispy.data.test import get_test_filepath
        >>> from irispy.io import read_files
        >>> cube = read_files(
        ...     get_test_filepath(
        ...         "sns/iris_l2_20210905_001833_3620258102_SJI_1330_t000_test.fits"
        ...     )
        ... )
        >>> sji_map = cube.to_maps(0)
        >>> sji_map.date.isot, sji_map.wavelength, sji_map.data.shape
        ('2021-09-05T00:18:37.740', <Quantity 1330. Angstrom>, (40, 37))
        >>> len(cube.to_maps([0, 1]))
        2
        """
        from sunpy.map import MapSequence  # NOQA: PLC0415

        # The maps give the IRIS DN units as DN, which the raw integers of a memmap read are not
        check_scaled(self)
        if self._frame_headers is None:
            msg = (
                "This cube has no FITS WCS header for its frames (for example, it was rebinned or indexed along a "
                "spatial axis), so it cannot be converted to maps."
            )
            raise ValueError(msg)
        if isinstance(index, Integral):
            idx_list = [index]
        elif index is None:
            idx_list = range(self.data.shape[0])
        else:
            idx_list = index

        is_2d = self.wcs.world_n_dim == 2
        if is_2d:
            frames = [(self.data, self._frame_headers, self.mask)]
        else:
            # The data and header lists wrap negative indices; resolve them once for both.
            idx_list = [range(self.data.shape[0])[i] for i in idx_list]
            frames = [
                (self.data[i], self._frame_headers[i], None if self.mask is None else self.mask[i]) for i in idx_list
            ]
        # Keep the requested order; a MapSequence sorts by date by default
        maps = MapSequence([_make_map(data, header, self.unit, mask) for data, header, mask in frames], sortby=None)
        cmap = _sji_cmap_name(self.meta)
        for m in maps:
            m.plot_settings["cmap"] = cmap
        return maps[0] if is_2d or isinstance(index, Integral) else maps


class AIACube(SJICube):
    """
    Subclass of the SJICube.

    It is the same outside of the name.
    """

    def __str__(self) -> str:
        return super().__str__().replace("SJICube", "AIACube")


class SOTCube(SJICube):
    """
    Hinode/SOT image and magnetic data supplied with IRIS observations :cite:p:`itn32`.
    """

    def __str__(self) -> str:
        return super().__str__().replace("SJICube", "SOTCube")
