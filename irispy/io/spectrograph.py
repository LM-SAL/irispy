from copy import copy, deepcopy
from types import SimpleNamespace
from pathlib import Path
from contextlib import nullcontext

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sunpy import log as logger
from sunpy.coordinates import get_earth
from sunpy.coordinates.frames import Helioprojective
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy._spectrograph_wcs import _sanitize_raster_wcs_tables
from irispy.io._level2 import _check_level2
from irispy.io._mask import _check_scaled, _memmap_fill_mask
from irispy.io._raster_combine import _add_raster, _combine_raster_cubes, _sky_time_models
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


def _create_tabular_wcs(header, auxiliary_data, auxiliary_header, *, date_obs):
    """
    Create a FITS-TAB WCS from the per-step pointing in the auxiliary table, given in
    step order.
    """
    header = copy(header)
    # FITS-TAB does not convert table units, and celestial WCS values must be degrees.
    arcsec_to_deg = u.arcsec.to(u.deg)
    spatial_pixels = np.array([1, header["NAXIS2"]], dtype=float)
    spatial_offsets = spatial_pixels - header["CRPIX2"]
    longitude_scale = header["CDELT3"] or header["CDELT2"]

    x, y = _exposure_pointing(auxiliary_data, auxiliary_header)
    longitude = x[:, None] + longitude_scale * (auxiliary_data[:, auxiliary_header["PC3_2IX"], None] * spatial_offsets)
    latitude = y[:, None] + header["CDELT2"] * (auxiliary_data[:, auxiliary_header["PC2_2IX"], None] * spatial_offsets)
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


def _open_raster(filename, *, memmap):
    """
    Open a raster file, or pass an open HDU list through.
    """
    if isinstance(filename, fits.HDUList):
        return nullcontext(filename)
    try:
        # The extensions are read out of order, which rewinds a gzip stream: inflate it once instead.
        return fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap, decompress_in_memory=True)
    except (OSError, ValueError, TypeError):
        # A directory or a file that is not FITS: raises the level 2 error, chained to this one
        _check_level2(filename, None, "raster")
        raise


def _date_obs(filename):
    """
    The ``DATE_OBS`` of a raster file, or the time of its first exposure for a file with
    no data, which the archive writes with an empty ``DATE_OBS``.
    """
    try:
        header = filename[0].header if isinstance(filename, fits.HDUList) else fits.getheader(filename)
    except (OSError, ValueError, TypeError):
        # A directory or a file that is not FITS: raises the level 2 error, chained to this one
        _check_level2(filename, None, "raster")
        raise
    if header["DATE_OBS"]:
        return header["DATE_OBS"]
    with _open_raster(filename, memmap=False) as hdulist:
        return _auxiliary_times(hdulist)[0].isot


def _select_windows(hdulist, given, spectral_windows, *, revert_v34):
    """
    What to read from every file of an observation, from its first file: the names,
    extensions and detector bands of the windows, in the order requested, and the order
    of the steps.
    """
    header = hdulist[0].header
    in_obs = [header[f"TDESC{i}"] for i in range(1, header["NWIN"] + 1)]
    if spectral_windows is None:
        names = in_obs
    else:
        requested = (
            [spectral_windows] if isinstance(spectral_windows, str) else [str(name) for name in spectral_windows]
        )
        # In the order requested, each once; an empty request is every window
        names = list(dict.fromkeys(requested)) or in_obs
    missing = [name for name in names if name not in in_obs]
    if missing:
        msg = f"Spectral windows {missing} not in {given}, which has {in_obs}"
        raise ValueError(msg)
    exts = [in_obs.index(name) + 1 for name in names]
    # After a discussion with the IRIS team, it was decided that instead of the
    # OBSID, we will use STEPS_AV less than -0.01 to identify V34 observations.
    flip = header["STEPS_AV"] < -0.01 and not revert_v34
    return SimpleNamespace(
        names=names,
        exts=exts,
        bands={"NUV" if header[f"TDET{ext}"].startswith("NUV") else "FUV" for ext in exts},
        flip=flip,
        # V34 rasters are flipped along the step axis: everything indexed by step must follow.
        steps=slice(None, None, -1) if flip else slice(None),
    )


def _read_raster_file(hdulist, given, selection, observer, frame_observer, *, filename, memmap, lazy, uncertainty):
    """
    Read one raster file.

    Returns a record with what the gWCS of the combined cubes needs from the file: its
    pointing tables, slit reference pixel, plate scales, exposure times by detector band
    and observer, and in ``windows`` one record per selected window with its header,
    unit, meta, FITS WCS, shape and dtype, and either the ``source`` a ``lazy`` read
    opens on demand or its ``arrays``: data, mask, uncertainty and saturated samples.
    ``frame_observer`` is the observer of the first file, whose frame the pointing is
    in, as the gWCS is.
    """
    steps, flip = selection.steps, selection.flip
    # The meta of each window copies one dict of the primary header: walking its cards is slow
    try:
        primary = dict(hdulist[0].header.items())
    except fits.VerifyError:
        # Files made before SolarSoft's sxaddpar saved non-finite values as strings (June 2018) write a NaN
        # as an unquoted NAN, such as STEPT_DV, which astropy cannot parse: it becomes the string "NAN",
        # as IDL's sxpar reads it. Fixing every header would take 0.4-1 ms a file.
        hdulist[0].verify("silentfix")
        primary = dict(hdulist[0].header.items())
    file_times = _auxiliary_times(hdulist)
    # The archive writes a file with no data with DATE_OBS and DATE_END empty and its pointing and roll as 0
    no_data = {}
    if not hdulist[0].header["DATE_OBS"]:
        no_data = {
            "DATE_OBS": (file_times[0].isot, "STARTOBS plus the auxiliary TIME of the first exposure"),
            "DATE_END": (file_times[-1].isot, "STARTOBS plus the auxiliary TIME of the last exposure"),
            **dict.fromkeys(("XCEN", "YCEN", "SAT_ROT"), (np.nan, "Unknown: the file has no data")),
        }
        logger.warning(f"{given} has no data: its exposures are masked and its pointing and roll are NaN.")
    aux = hdulist[-2]
    # In native byte order, as the per-exposure meta made from it is: FITS stores it big-endian
    aux_data = aux.data[steps].astype(aux.data.dtype.newbyteorder("="))

    def column(key):
        return aux_data[:, aux.header[key]]

    fov_center = SkyCoord(
        *_exposure_pointing(aux_data, aux.header),
        unit=u.arcsec,
        frame=Helioprojective,
        observer=frame_observer,
        obstime=frame_observer.obstime,
    )
    exposure_times = {"FUV": column("EXPTIMEF") * u.s, "NUV": column("EXPTIMEN") * u.s}
    observer_velocity, orbital_phase = column("OBS_VRIX") * u.m / u.s, column("OPHASEIX")
    pc = (
        aux_data[:, [aux.header[key] for key in ("PC2_2IX", "PC2_3IX", "PC3_2IX", "PC3_3IX")]].reshape(-1, 2, 2) * u.pix
    )
    if flip:
        # Reversing the step axis reverses its PC column.
        pc[:, :, 1] *= -1
    # Every exposure, sit-and-stare included, is pointed at its own FOV center.
    crval = np.column_stack((column("XCENIX"), column("YCENIX"))) * u.arcsec
    pc, crval = _sanitize_raster_wcs_tables(pc, crval)
    # Exposure midpoints (T_OBS) of each band read; NUV ones come from the source filenames
    aux_times = file_times[steps]
    times = {}
    if "FUV" in selection.bands:
        times["FUV"] = aux_times + exposure_times["FUV"] / 2
    if "NUV" in selection.bands:
        if no_data:
            times["NUV"] = aux_times  # its own warning covers the missing NUV exposures
        else:
            # The source filename table is in file order. The archive writes the NUV source filenames of a file
            # without NUV exposures as A0 columns, which astropy cannot read.
            nuv_exposure_times = aux.data[:, aux.header["EXPTIMEN"]] * u.s
            times["NUV"] = _nuv_t_obs_from_source_filenames(
                hdulist[-1].data if np.any(nuv_exposure_times > 0 * u.s) else None,
                nuv_exposure_times,
                file_times,
                filename=given,
            )[steps]
    # The windows of a raster file differ only in NAXIS1, CRVAL1, CDELT1 and CRPIX1 (3374 files of 20 observations,
    # 2013-2026, checked), so one FITS-TAB WCS serves them all. Building it is most of the read.
    first_header = hdulist[selection.exts[0]].header
    file_wcs = _create_tabular_wcs(first_header, aux_data, aux.header, date_obs=observer.obstime.utc.isot)
    _set_wcs_aux_obs_coord(file_wcs, observer)
    windows = []
    for name, ext in zip(selection.names, selection.exts, strict=True):
        header = hdulist[ext].header
        data = hdulist[ext].data[steps]
        meta = SGMeta(primary, name, data_shape=data.shape)
        # As if made from the header itself, which NDMeta keeps as the meta's original_meta
        meta._original_meta = meta._fits_header = hdulist[0].header
        for key, (value, comment) in no_data.items():
            meta.add(key, value, comment, overwrite=True)
        band = meta.detector_band
        meta.add("auxiliary times", aux_times, None, 0)
        meta.add("exposure time", exposure_times[band], None, 0)
        meta.add("exposure FOV center", fov_center, None, 0)
        meta.add("observer radial velocity", observer_velocity, None, 0)
        meta.add("orbital phase", orbital_phase, None, 0)
        window = SimpleNamespace(
            header=header,
            unit=DN_UNIT[band],
            meta=meta,
            fits_wcs=_with_spectral_axis(file_wcs, header),
            shape=data.shape,
            dtype=data.dtype,
        )
        if memmap:  # the data stay the FITS integers, which these scale to DN
            meta["BSCALE"], meta["BZERO"] = header.get("BSCALE", 1), header.get("BZERO", 0)
            if np.issubdtype(data.dtype, np.integer):  # the FITS integers, not DN
                window.unit = DN_UNIT[f"{band}_UNSCALED"]
        if lazy:
            # The cube reads each file on demand; a memmap would keep every file open
            window.source = data if isinstance(filename, fits.HDUList) else (filename, ext, flip)
        elif memmap:
            window.arrays = {"data": data, "mask": _memmap_fill_mask(data, header)}
        else:
            _check_scaled(header)
            # Level 2 clips at the ceiling, where iris_prep's +Inf saturation flags were lost
            saturated = data >= SATURATION_LIMIT.value
            window.arrays = {
                "data": data,
                "mask": np.isin(data, BAD_PIXEL_VALUES_SCALED) | saturated,
                "saturated": saturated if saturated.any() else None,
                # memmap data are unscaled integers, so the photon noise would be wrong
                "uncertainty": calculate_uncertainty(data, READOUT_NOISE[band], window.unit) if uncertainty else None,
            }
        windows.append(window)
    return SimpleNamespace(
        pc=pc,
        crval=crval,
        crpix2=first_header["CRPIX2"],
        cdelt2=first_header["CDELT2"],
        cdelt3=first_header["CDELT3"],
        times=times,
        observer=observer,
        windows=windows,
    )


def read_spectrograph_lvl2(
    filenames,
    *,
    spectral_windows=None,
    uncertainty=False,
    memmap=False,
    revert_v34=False,
):
    """
    Read IRIS Level 2 raster files into one cube per spectral window.

    Each cube has axes (raster, step, slit, wavelength), with one raster per file in time
    order. ``cube[k]`` is raster ``k``, and a single file, such as a sit-and-stare, gives
    one raster. Use `irispy.io.read_files` for tar files.

    Samples with the ``-200`` DN fill value, or clipped to the ``-199`` DN floor, keep their
    value and are marked in the mask. With ``memmap=True`` these are the raw integers
    ``-32768`` and ``-32764``.

    Parameters
    ----------
    filenames : `str`, `pathlib.Path`, `astropy.io.fits.HDUList` or a `list` of them
        Files to read, or open HDU lists, which are left open. They must all come from one
        observation, with the same ``OBSID`` and ``STARTOBS``, and are read in order of
        ``DATE_OBS``, whatever the order given. Open HDU lists with
        ``do_not_scale_image_data=True`` for ``memmap=True``, and without it otherwise.
        Open gzipped files with ``decompress_in_memory=True``.
    spectral_windows : `str` or iterable of `str`, optional
        The spectral windows to read. Defaults to `None`, every window of the files.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        integer data stay unscaled, in ``DN_IRIS_FUV_UNSCALED`` or ``DN_IRIS_NUV_UNSCALED``,
        not the usual data numbers (DN). The mask is a lazy Dask array, computed only for the
        slices that are used. The data of several files read together are a lazy Dask array too.
        With ``memmap=False``, the samples at the level 2 ceiling,
        `irispy.utils.constants.SATURATION_LIMIT`, keep their value and are marked in the mask
        and, if there are any, in ``meta["saturated"]`` (see :ref:`irispy-tutorial-saturated-data`).
    revert_v34 : `bool`, optional
        Will undo the flipping of the raster step axis made to V34 observations
        (data, mask, uncertainty, WCS, times and per-step metadata).
        Defaults to `False`.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        The cubes, keyed by spectral window.

    Raises
    ------
    ValueError
        If a file does not have one of ``spectral_windows``.

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
        # The rasters follow time whatever the order given, such as that of an unsorted glob.
        date_obs = Time([_date_obs(filename) for filename in filenames])
        order = date_obs.argsort()
        filenames = [filenames[i] for i in order]
        # Each file's WCS uses the observer at its own DATE_OBS, as SGMeta.observer does; one ephemeris call gets all
        observers = get_earth(date_obs[order])
    # Several files read with memmap stay on disk and are read a file at a time
    lazy = memmap and len(filenames) > 1
    files = []
    for index, filename in enumerate(filenames):
        with _open_raster(filename, memmap=memmap) as hdulist:
            # An HDU list made in memory has no filename, so name what was given
            given = hdulist.filename() or f"The data given ({type(filename).__name__})"
            _check_level2(given, hdulist[0].header, "raster")
            if index == 0:
                selection = _select_windows(hdulist, given, spectral_windows, revert_v34=revert_v34)
                rasters = {name: {} for name in selection.names}
            if len(filenames) == 1:
                observer = get_earth(Time(hdulist[0].header["DATE_OBS"] or _auxiliary_times(hdulist)[0]))
            else:
                observer = observers[index]
            # Only the combined cubes are built. A cube for each file would be left to the cyclic garbage
            # collector, which then takes a second or more for a long observation
            file = _read_raster_file(
                hdulist,
                given,
                selection,
                observer,
                # The pointing of every file is in the frame of the first, as the gWCS is
                files[0].observer if files else observer,
                filename=filename,
                memmap=memmap,
                lazy=lazy,
                uncertainty=uncertainty and not memmap,
            )
            files.append(file)
            if not lazy:
                for name, window in zip(selection.names, file.windows, strict=True):
                    _add_raster(rasters[name], index, len(filenames), **window.arrays)
    non_spectral = _sky_time_models(files)
    cubes = []
    for i, name in enumerate(selection.names):
        reads = [file.windows[i] for file in files]
        band_model = non_spectral[reads[0].meta.detector_band]
        cubes.append((name, _combine_raster_cubes(reads, rasters[name], band_model, lazy=lazy)))
    return RasterCollection(cubes, aligned_axes=(0, 1, 2))
