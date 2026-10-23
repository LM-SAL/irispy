from copy import copy, deepcopy
from types import SimpleNamespace
from pathlib import Path
from contextlib import nullcontext

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sunpy import log as logger
from sunpy.coordinates import get_earth
from sunpy.coordinates.frames import Helioprojective
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy._spectrograph_wcs import _sanitize_raster_wcs_tables
from irispy.io._level2 import _check_level2
from irispy.io._mask import _check_scaled, _memmap_fill_mask
from irispy.io._raster_combine import _combine_raster_cubes
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED, DN_UNIT, READOUT_NOISE, SATURATION_LIMIT

__all__ = ["read_spectrograph_lvl2"]


def _nuv_t_obs_from_source_filenames(source_data, exposure_times, auxiliary_times, *, filename):
    """
    Return NUV exposure midpoints (T_OBS).

    The midpoints are parsed from the level 1 source filenames, which encode the
    exposure midpoint. Rows with an exposure time of 0 s have no usable source filename,
    so they fall back to their planned ``auxiliary_times``.
    """
    valid_rows = exposure_times > 0 * u.s
    t_obs = auxiliary_times.copy()
    if np.any(valid_rows):
        if source_data is None or len(source_data) != len(exposure_times):
            found = 0 if source_data is None else len(source_data)
            msg = f"Expected {len(exposure_times)} NUV source filename rows, found {found}"
            raise ValueError(msg)
        try:
            timestamps = [
                Path(source_filename).name[4:21] for source_filename in source_data["NUVfilename"][valid_rows]
            ]
            t_obs[valid_rows] = Time.strptime(timestamps, "%Y%m%d_%H%M%S%f")
        except (KeyError, TypeError, ValueError) as error:
            msg = "Invalid timestamp in NUV source filenames"
            raise ValueError(msg) from error
    missing_rows = np.flatnonzero(~valid_rows)
    if missing_rows.size:
        logger.warning(
            f"EXPTIMEN is 0 s at row(s) {missing_rows.tolist()} in {filename}; frames retained with auxiliary times."
        )
    return t_obs


def _auxiliary_times(hdulist):
    """
    Return the time of each exposure of a raster file: ``STARTOBS`` plus the auxiliary ``TIME``.
    """
    auxiliary = hdulist[-2]
    return Time(hdulist[0].header["STARTOBS"]) + TimeDelta(auxiliary.data[:, auxiliary.header["TIME"]], format="sec")


def _exposure_pointing(auxiliary_data, auxiliary_header):
    """
    Return the ``XCENIX`` and ``YCENIX`` of each exposure, NaN for an exposure missing
    on both detectors, which has them as 0.
    """
    x, y = auxiliary_data[:, auxiliary_header["XCENIX"]], auxiliary_data[:, auxiliary_header["YCENIX"]]
    unknown = (x == 0) & (y == 0)
    return np.where(unknown, np.nan, x), np.where(unknown, np.nan, y)


def _create_tabular_wcs(header, auxiliary_hdu, *, date_obs, flip=False):
    """
    Create a FITS-TAB WCS from the per-step pointing in the auxiliary table.
    """
    header = copy(header)
    auxiliary_data = auxiliary_hdu.data[::-1] if flip else auxiliary_hdu.data
    # FITS-TAB does not convert table units, and celestial WCS values must be degrees.
    arcsec_to_deg = u.arcsec.to(u.deg)
    spatial_pixels = np.array([1, header["NAXIS2"]], dtype=float)
    spatial_offsets = spatial_pixels - header["CRPIX2"]
    longitude_scale = header["CDELT3"] or header["CDELT2"]

    x, y = _exposure_pointing(auxiliary_data, auxiliary_hdu.header)
    longitude = x[:, None] + longitude_scale * (
        auxiliary_data[:, auxiliary_hdu.header["PC3_2IX"], None] * spatial_offsets
    )
    latitude = y[:, None] + header["CDELT2"] * (
        auxiliary_data[:, auxiliary_hdu.header["PC2_2IX"], None] * spatial_offsets
    )
    coordinates = np.stack((latitude, longitude), axis=-1) * arcsec_to_deg

    spatial_index = (header["CRVAL2"] + header["CDELT2"] * (spatial_pixels - header["CRPIX2"])) * arcsec_to_deg
    table_data = np.array(
        [(coordinates, spatial_index)],
        dtype=[
            ("COORDS", float, coordinates.shape),
            ("SPATIAL", float, spatial_index.shape),
        ],
    )
    table = fits.BinTableHDU(table_data, name="WCS-TABLE")
    table.header["TUNIT1"] = "deg"
    table.header["TUNIT2"] = "deg"

    header["CTYPE2"] = "HPLT-TAB"
    header["CTYPE3"] = "HPLN-TAB"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "deg"
    header["DATE-OBS"] = date_obs
    header["MJD-OBS"] = Time(date_obs).mjd
    header["CRVAL2"] *= arcsec_to_deg
    header["CDELT2"] *= arcsec_to_deg
    header["CRPIX3"] = 1
    header["CRVAL3"] = 1
    header["CDELT3"] = 1
    for row in range(1, 4):
        for column in range(1, 4):
            header[f"PC{row}_{column}"] = float(row == column)
    for axis in (2, 3):
        header[f"PS{axis}_0"] = table.name
        header[f"PS{axis}_1"] = "COORDS"
        header[f"PV{axis}_3"] = axis - 1
    # The step axis has no index vector: FITS-TAB then indexes COORDS directly by the
    # 1-based step. wcslib searches an index vector linearly, so an explicit 1..N index
    # made every lookup cost O(step).
    header["PS2_2"] = "SPATIAL"

    return WCS(header, fits.HDUList([fits.PrimaryHDU(), table]))


def _with_spectral_axis(wcs, header):
    """
    Return a copy of the FITS-TAB ``wcs`` of a window of the same file, with the
    spectral axis of ``header``.
    """
    wcs = deepcopy(wcs)
    # In the header's unit: wcslib converts them to SI, as it did those of the header the WCS was built from
    wcs.wcs.cunit[0] = header["CUNIT1"]
    wcs.wcs.crval[0], wcs.wcs.cdelt[0], wcs.wcs.crpix[0] = header["CRVAL1"], header["CDELT1"], header["CRPIX1"]
    wcs.wcs.set()
    wcs.pixel_shape = (header["NAXIS1"], *wcs.pixel_shape[1:])
    return wcs


def read_spectrograph_lvl2(
    filenames,
    *,
    spectral_windows=None,
    uncertainty=False,
    memmap=False,
    revert_v34=False,
):
    """
    Reads either a SINGLE IRIS level 2 spectrograph FITs or a list of them.

    Does not handle tar files.

    Parameters
    ----------
    filenames: `str`, `pathlib.Path`, `astropy.io.fits.HDUList` or a `list` of them
        Files to be read, or files already opened, which are left open. They must all be
        associated with the same OBS number. They are read in order of their ``DATE_OBS``,
        whatever their order here. Open files with ``do_not_scale_image_data=True`` when using
        ``memmap=True``, and without it otherwise. Open gzipped files with ``decompress_in_memory=True``.
    spectral_windows: iterable of `str` or `str`
        Spectral windows to extract from files. Default=None, implies, extract all
        spectral windows.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        integer data stay unscaled, in ``DN_IRIS_FUV_UNSCALED`` or ``DN_IRIS_NUV_UNSCALED``,
        not the usual data numbers (DN).
        The ``-200`` DN fill value and the ``-199`` DN floor that lower samples are clipped to (the
        raw integers ``-32768`` and ``-32764`` with ``memmap=True``) keep their value and are marked
        in the mask in either mode; with ``memmap=True`` it is a lazy Dask array, computed only for
        the slices that are used.
        With ``memmap=False``, the samples at the level 2 ceiling,
        `irispy.utils.constants.SATURATION_LIMIT`, keep their value and are marked in the mask
        and, if there are any, in ``meta["saturated"]`` (see :ref:`irispy-tutorial-saturated-data`).
    revert_v34 : `bool`, optional.
        Will undo the flipping of the raster step axis made to V34 observations
        (data, mask, uncertainty, WCS, times and per-step metadata).
        Defaults to `False`.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`

    Examples
    --------
    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import read_spectrograph_lvl2
    >>> filename = get_test_filepath(
    ...     "sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits"
    ... )
    >>> raster = read_spectrograph_lvl2(
    ...     filename, spectral_windows=["Si IV 1403", "Mg II k 2796"]
    ... )
    >>> len(raster), raster["Si IV 1403"].shape, raster["Mg II k 2796"].shape
    (2, (1, 187, 40, 29), (1, 187, 40, 52))
    >>> raster["Si IV 1403"].unit, raster["Mg II k 2796"].unit
    (Unit("DN_IRIS_FUV"), Unit("DN_IRIS_NUV"))
    """
    if isinstance(filenames, (str, bytes, Path, fits.HDUList)):
        filenames = [filenames]
    filenames = list(filenames)
    if not filenames:
        msg = "filenames is empty: give at least one raster file"
        raise ValueError(msg)
    if len(filenames) > 1:
        # The sequence follows time whatever the order given, such as that of an unsorted glob.
        date_obs = []
        for filename in filenames:
            try:
                header = filename[0].header if isinstance(filename, fits.HDUList) else fits.getheader(filename)
                date_obs.append(header["DATE_OBS"])
            except (OSError, ValueError, TypeError):
                # A directory or a file that is not FITS: raises the level 2 error, chained to this one
                _check_level2(filename, None, "raster")
            if not date_obs[-1]:  # a file with no data
                with nullcontext(filename) if isinstance(filename, fits.HDUList) else fits.open(filename) as hdulist:
                    date_obs[-1] = _auxiliary_times(hdulist)[0].isot
        date_obs = Time(date_obs)
        order = date_obs.argsort()
        filenames = [filenames[i] for i in order]
        # Each file's WCS has its own observer, at its own DATE_OBS as SGMeta.observer: one ephemeris call for all
        observers = get_earth(date_obs[order])
    # The files are combined into one cube per window, with a raster axis, whose gWCS is built once from all of them
    compute_uncertainty = uncertainty and not memmap
    for file_index, filename in enumerate(filenames):
        if isinstance(filename, fits.HDUList):
            context = nullcontext(filename)
        else:
            try:
                # The extensions are read out of order, which rewinds a gzip stream: inflate it once instead.
                context = fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap, decompress_in_memory=True)
            except (OSError, ValueError, TypeError):
                # A directory or a file that is not FITS: raises the level 2 error, chained to this one
                _check_level2(filename, None, "raster")
        with context as hdulist:
            # An HDU list made in memory has no filename, so name what was given
            given = hdulist.filename() or f"The data given ({type(filename).__name__})"
            _check_level2(given, hdulist[0].header, "raster")
            # Files made before SolarSoft's sxaddpar saved non-finite values as strings (June 2018) write a NaN
            # as an unquoted NAN, such as STEPT_DV, which astropy cannot parse: it becomes the string "NAN",
            # as IDL's sxpar reads it
            hdulist[0].verify("silentfix")
            if file_index == 0:
                # After a discussion with the IRIS team, it was decided that instead of the
                # OBSID, we will use STEPS_AV less than -0.01 to identify V34 observations.
                v34 = hdulist[0].header["STEPS_AV"] < -0.01
                flip = v34 and not revert_v34
                # V34 rasters are flipped along the step axis: everything indexed by step must follow.
                steps = slice(None, None, -1) if flip else slice(None)
                windows_in_obs = np.array(
                    [hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)],
                )
                # If spectral_window is not set then get every window.
                # Else take the appropriate windows
                if spectral_windows is not None and not isinstance(spectral_windows, str):
                    # A NumPy array has no truth value
                    spectral_windows = list(spectral_windows)
                if not spectral_windows:
                    spectral_windows_req = windows_in_obs
                    window_fits_indices = range(1, len(hdulist) - 2)
                else:
                    spectral_windows_req = [spectral_windows] if isinstance(spectral_windows, str) else spectral_windows
                    spectral_windows_req = np.asarray(list(dict.fromkeys(spectral_windows_req)), dtype="U")
                    window_is_in_obs = np.asarray([window in windows_in_obs for window in spectral_windows_req])
                    if not all(window_is_in_obs):
                        missing_windows = spectral_windows_req[~window_is_in_obs]
                        msg = (
                            f"Spectral windows {missing_windows.tolist()} not in {given}, "
                            f"which has {windows_in_obs.tolist()}"
                        )
                        raise ValueError(msg)
                    # Indices must follow the order of the requested windows, not the file order,
                    # since they are zipped with ``spectral_windows_req`` below.
                    window_fits_indices = [
                        int(np.nonzero(windows_in_obs == window)[0][0]) + 1 for window in spectral_windows_req
                    ]
                spectral_windows_req = spectral_windows_req.tolist()  # the keys are str, not numpy.str_
                data_dict = {window_name: [] for window_name in spectral_windows_req}
            aux = hdulist[-2]
            file_times = _auxiliary_times(hdulist)
            aux_times = file_times[steps]
            # The archive writes a file with no data with DATE_OBS and DATE_END empty and its pointing and roll as 0
            no_data = {}
            if not hdulist[0].header["DATE_OBS"]:
                no_data = {
                    "DATE_OBS": (file_times[0].isot, "STARTOBS plus the auxiliary TIME of the first exposure"),
                    "DATE_END": (file_times[-1].isot, "STARTOBS plus the auxiliary TIME of the last exposure"),
                    **dict.fromkeys(("XCEN", "YCEN", "SAT_ROT"), (np.nan, "Unknown: the file has no data")),
                }
                logger.warning(f"{given} has no data: its exposures are masked and its pointing and roll are NaN.")
            if len(filenames) == 1:
                observer = get_earth(Time(hdulist[0].header["DATE_OBS"] or file_times[0]))
            else:
                observer = observers[file_index]
            if file_index == 0:
                # The pointing of a multi-file read is in the frame of its first file, as its gWCS
                first_observer = observer
            aux_data = aux.data[steps]
            fov_center = SkyCoord(
                *_exposure_pointing(aux_data, aux.header),
                unit=u.arcsec,
                frame=Helioprojective,
                observer=first_observer,
                obstime=first_observer.obstime,
            )
            obs_vrix = aux_data[:, aux.header["OBS_VRIX"]] * u.m / u.s
            ophaseix = aux_data[:, aux.header["OPHASEIX"]]
            exposure_times_fuv = aux_data[:, aux.header["EXPTIMEF"]] * u.s
            exposure_times_nuv = aux_data[:, aux.header["EXPTIMEN"]] * u.s
            pc_indices = [aux.header[key] for key in ("PC2_2IX", "PC2_3IX", "PC3_2IX", "PC3_3IX")]
            pc = aux_data[:, pc_indices].reshape(-1, 2, 2) * u.pix
            if flip:
                # Reversing the step axis reverses its PC column.
                pc[:, :, 1] *= -1
            # Every exposure, sit-and-stare included, is pointed at its own FOV center.
            crval = np.column_stack((aux_data[:, aux.header["XCENIX"]], aux_data[:, aux.header["YCENIX"]])) * u.arcsec
            # Exposure midpoints (T_OBS); NUV ones come from the source filenames.
            t_obs_nuv = None
            file_wcs = wcs_tables = None
            # The meta of each window copies one dict of the primary header: walking its cards is slow
            primary = dict(hdulist[0].header.items())
            for i, window_name in enumerate(spectral_windows_req):
                ext = window_fits_indices[i]
                window_header = hdulist[ext].header
                data = hdulist[ext].data[steps]
                meta = SGMeta(primary, window_name, data_shape=data.shape)
                # As if made from the header itself, which NDMeta keeps as the meta's original_meta
                meta._original_meta = meta._fits_header = hdulist[0].header
                for key, (value, comment) in no_data.items():
                    meta.add(key, value, comment, overwrite=True)
                meta.add("auxiliary times", aux_times, None, 0)
                if "FUV" in meta.detector:
                    exposure_times = exposure_times_fuv
                    dn_unit = DN_UNIT["FUV"]
                    readout_noise = READOUT_NOISE["FUV"]
                    times = aux_times + exposure_times_fuv / 2
                else:
                    exposure_times = exposure_times_nuv
                    dn_unit = DN_UNIT["NUV"]
                    readout_noise = READOUT_NOISE["NUV"]
                    if no_data:
                        # Its own warning covers the missing NUV exposures
                        t_obs_nuv = aux_times
                    elif t_obs_nuv is None:
                        # The source filename table is in file order, so undo (and redo) any flip. The archive
                        # writes the NUV source filenames of a file without NUV exposures as A0 columns,
                        # which astropy cannot read.
                        t_obs_nuv = _nuv_t_obs_from_source_filenames(
                            hdulist[-1].data if np.any(exposure_times_nuv > 0 * u.s) else None,
                            exposure_times_nuv[steps],
                            aux_times[steps],
                            filename=given,
                        )[steps]
                    times = t_obs_nuv
                meta.add("exposure time", exposure_times, None, 0)
                meta.add("exposure FOV center", fov_center, None, 0)
                meta.add("observer radial velocity", obs_vrix, None, 0)
                meta.add("orbital phase", ophaseix, None, 0)
                # The windows of a raster file differ only in NAXIS1, CRVAL1, CDELT1 and CRPIX1
                # (3374 files of 20 observations, 2013-2026, checked), so their pointing is the file's
                if wcs_tables is None:
                    wcs_tables = _sanitize_raster_wcs_tables(pc, crval, pc_only=window_header["CDELT3"] == 0)
                # Building the FITS-TAB WCS is most of the read
                if file_wcs is None:
                    file_wcs = _create_tabular_wcs(window_header, aux, date_obs=observer.obstime.utc.isot, flip=flip)
                    _set_wcs_aux_obs_coord(file_wcs, observer)
                fits_wcs = _with_spectral_axis(file_wcs, window_header)
                if memmap:  # the data stay the FITS integers, which these scale to DN
                    meta["BSCALE"], meta["BZERO"] = window_header.get("BSCALE", 1), window_header.get("BZERO", 0)
                    # The cube of several files makes its own
                    mask = None if len(filenames) > 1 else _memmap_fill_mask(data, window_header)
                    if np.issubdtype(data.dtype, np.integer):  # the FITS integers, not DN
                        dn_unit = DN_UNIT[f"{meta.detector_band}_UNSCALED"]
                else:
                    _check_scaled(window_header)
                    # Level 2 clips at the ceiling, where iris_prep's +Inf saturation flags were lost
                    saturated = data >= SATURATION_LIMIT.value
                    mask = np.isin(data, BAD_PIXEL_VALUES_SCALED) | saturated
                    if saturated.any():
                        meta.add("saturated", saturated, "Samples at the level 2 ceiling, also masked", (0, 1, 2))
                out_uncertainty = (
                    StdDevUncertainty(calculate_uncertainty(data, readout_noise, dn_unit))
                    if compute_uncertainty
                    else None
                )
                meta.add("fits_wcs", fits_wcs)
                # Only the combined cube is made: a cube for each file would be left for the cyclic garbage
                # collector, which then takes a second or more over the cubes of a long observation
                read = SimpleNamespace(
                    # The cube of several files reads them on demand: a memmap would hold its file open
                    data=(filename, ext, flip)
                    if memmap and len(filenames) > 1 and not isinstance(filename, fits.HDUList)
                    else data,
                    mask=mask,
                    uncertainty=out_uncertainty,
                    unit=dn_unit,
                    meta=meta,
                    time=times,
                    shape=data.shape,
                    dtype=data.dtype,
                )
                data_dict[window_name].append((read, window_header, *wcs_tables))
    window_data_pairs = [
        (window_name, _combine_raster_cubes(*zip(*reads, strict=True), memmap=memmap))
        for window_name, reads in data_dict.items()
    ]
    return RasterCollection(window_data_pairs, aligned_axes=(0, 1, 2))
