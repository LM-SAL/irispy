import os
import sys
import gzip
import json
import hashlib
import tarfile
from pathlib import Path

from filelock import FileLock

from astropy.config.paths import get_cache_dir
from astropy.io import fits
from astropy.table import Table

from ndcube import NDCollection
from sunpy import log

from irispy.io._level2 import _check_level2
from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2

__all__ = ["fits_info", "read_files"]


def _primary_header(file):
    """
    Read the primary header, and nothing else, of a FITS file that may be gzipped.

    `astropy.io.fits.getheader` seeks past the primary data, which inflates a gzipped
    slit-jaw cube in full just to classify it.
    """
    opener = gzip.open if file.suffix == ".gz" else open
    with opener(file, "rb") as fileobj:
        return fits.Header.fromfile(fileobj)


def _get_simple_metadata(header):
    """
    Get simple metadata from a primary FITS header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        The primary header to inspect.

    Returns
    -------
    `tuple`
        A tuple containing the instrument name and description.
    """
    instrume = header.get("INSTRUME", "")
    describe = header.get("TDESC1", "")
    if instrume == "SOT-SP":
        # SOT-SP maps share a band name; BTYPE distinguishes the measured quantities.
        describe = f"{describe} {header.get('BTYPE', '')}".strip()
    return instrume, describe


def _extract_tarfile(filenames, extract_dir=None):
    """
    Extracts tar files, each into its own folder under ``extract_dir``.

    The folder is named after the tar file with a suffix from its absolute path, so two
    archives with the same name in different folders never share one.

    A complete earlier extraction of the same tar file (same size and modification
    time) is reused while all of its files still exist with their full sizes.
    Concurrent callers wait for extraction to finish before reusing its files.

    Parameters
    ----------
    filenames : `list of str`
        The filenames of the tar files to extract.
    extract_dir : `str` or `pathlib.Path`, optional
        The folder to extract into. Defaults to the ``irispy`` cache folder,
        ``astropy.config.paths.get_cache_dir("irispy")``, so the archive's own folder
        stays untouched and can be read-only.
    """
    root = Path(get_cache_dir("irispy") if extract_dir is None else extract_dir)
    expanded_files = []
    for fname in filenames:
        filename = Path(fname)
        if tarfile.is_tarfile(filename):
            stem = filename.with_suffix("").with_suffix("").name  # removes .tar.gz or .tar
            digest = hashlib.sha1(os.fsencode(filename.resolve()), usedforsecurity=False).hexdigest()[:8]
            target = root / f"{stem}-{digest}"
            target.mkdir(parents=True, exist_ok=True)
            # The cache check must share the lock with extraction: a waiting caller can then reuse the files.
            with FileLock(target / ".irispy-extracted.lock"):
                # Written only after a complete extraction: the tar file it came from and the
                # extracted files with their sizes, so a truncated file is extracted again.
                marker = target / ".irispy-extracted.json"
                source = [filename.stat().st_size, filename.stat().st_mtime_ns]
                files = {}
                if marker.is_file():
                    try:
                        extracted = json.loads(marker.read_text())
                    except ValueError:
                        extracted = {}
                    if extracted.get("source") == source and isinstance(extracted.get("files"), dict):
                        files = extracted["files"]
                if not files or not all(
                    (target / name).is_file() and (target / name).stat().st_size == size for name, size in files.items()
                ):
                    marker.unlink(missing_ok=True)
                    with tarfile.open(filename, "r") as tar:
                        tar.extractall(target, filter="data")
                        # The data filter extracts a member named "/a.fits" to "a.fits": the map must match it,
                        # or the returned path would point outside the folder.
                        files = {
                            member.name.lstrip("/" + os.sep): member.size
                            for member in tar.getmembers()
                            if member.isfile()
                        }
                    marker.write_text(json.dumps({"source": source, "files": files}))
            expanded_files.extend(target / name for name in files)
        else:
            expanded_files.append(filename)
    return expanded_files


def _get_spec_group_key(file, header):
    """
    Group spectrograph FITS files that belong to the same observation.

    Parameters
    ----------
    file : `pathlib.Path`
        The FITS file.
    header : `astropy.io.fits.Header`
        Its primary header.

    Returns
    -------
    `tuple`
        Key built from stable observation-identifying header metadata.
        Falls back to a per-file key if the observation-grouping metadata is missing.
    """
    obsid = header.get("OBSID")
    startobs = header.get("STARTOBS")
    if obsid is None or not startobs:
        return file, None
    return obsid, startobs


def _get_spec_return_key(group_key, describe, returns):
    key = f"{describe}"
    if key not in returns:
        return key
    obsid, startobs = group_key
    if startobs is None:
        return f"{describe} ({Path(obsid).stem})"
    key = f"{describe} ({obsid})"
    if key not in returns:
        return key
    return f"{describe} ({obsid}, {startobs})"


def fits_info(filename: str) -> None:
    """
    Prints what each extension of a level 2 raster, slit-jaw, AIA or SOT file holds.

    Parameters
    ----------
    filename : str
        Filename to load.

    Examples
    --------
    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import fits_info
    >>> fits_info(
    ...     get_test_filepath(
    ...         "sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits"
    ...     )
    ... )
    Filename: ...iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits
    Observation: Medium sit-and-stare 0.3x60 1s  C II   Si IV   Mg II h/k   Mg II w s
    OBS ID: 3620258102
    No. Name ... Description
    --- ------- ... ...
      0 PRIMARY ... Primary Header (no data)
      1 ... C II 1336 (1333 - 1337 AA)
    ...
      8 ... Mg II k 2796 (2793 - 2807 AA)
      9 ... Auxiliary data
     10 ... Auxiliary data
    """

    def get_description(window):
        # Windows 1 to NWIN of the primary header are the data HDUs; the HDUs after them hold auxiliary data.
        if window == 0:
            return "Primary Header (no data)"
        if window > header["NWIN"]:
            return "Auxiliary data"
        description = header[f"TDESC{window}"].replace("_", " ")
        # AIA and SOT files give no wavelength range.
        if f"TWMIN{window}" in header:
            description += f" ({header[f'TWMIN{window}']:.0f} - {header[f'TWMAX{window}']:.0f} AA)"
        return description

    table = Table(
        names=("No.", "Name", "Ver", "Type", "Cards", "Dimensions", "Format", "Description"),
        dtype=("S8", "S32", "S8", "S16", "S8", "S40", "S16", "U200"),
    )
    with fits.open(filename) as hdulist:
        hdu_info = hdulist.info(output=False)
        header = hdulist[0].header
        results = [
            f"Filename: {Path(filename).absolute()}",
            f"Observation: {header['OBS_DESC']}",
            f"OBS ID: {header['OBSID']}",
        ]
        for idx in range(len(hdulist)):
            description = get_description(idx if "SPEC" in header["INSTRUME"] else idx + 1)
            hdu_info[idx] = (*hdu_info[idx][:-1], description)
            table.add_row(list(map(str, hdu_info[idx])))

    sys.stdout.write("\n".join(results) + "\n")
    table.pprint(max_width=-1)
    sys.stdout.flush()


def read_files(
    filenames,
    *,
    spectral_windows=None,
    uncertainty=False,
    memmap=False,
    allow_errors=False,
    revert_v34=False,
    extract_dir=None,
):
    """
    Read raster, SJI, AIA, or Hinode/SOT data supplied with IRIS observations.

    The goal is be able to download an entire IRIS observation and read it
    in one go, without having to worry about the type of file.

    Parameters
    ----------
    filenames : `list` of `str` or `pathlib.Path`, `str`, `pathlib.Path`
        Filename(s) or folder(s) to load. For a folder, every ``iris_l2_*``, ``aia_l2_*``, ``sot_l2_*``
        and ``sotsp_l2_*`` ``.fits``, ``.fits.gz`` and ``.tar.gz`` file directly in it is read. The folder
        should hold either the tarballs of an observation downloaded from the IRIS archive or the files
        extracted from them, not both: with both, each slit-jaw, AIA and SOT cube comes back twice and
        each raster scan appears twice in its raster.
        A file given more than once, by any path or symlink, is read once.
        Tar files are extracted first. Full-disk mosaics are not read here, use `irispy.io.read_mosaic`.
    spectral_windows : iterable of `str` or `str`, optional
        Spectral windows to extract from the raster files; slit-jaw, AIA and SOT files have none.
        Defaults to `None`, which extracts all spectral windows.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), FITS data are opened using Astropy/NumPy
        memory mapping rather than being fully read into memory at once. This
        can keep memory usage low when working with many files, since array
        data are accessed from disk on demand. In this mode FITS image scaling
        is disabled, so IRIS data are the raw integers of the file, in the
        ``DN_IRIS_*_UNSCALED`` units, which the analysis functions refuse; AIA and
        Hinode/SOT data are stored unscaled and keep their units. If ``memmap=True``,
        the uncertainty is never computed.
    allow_errors : `bool`, optional
        Will continue loading the files if one fails to load.
        Defaults to `False`.
    revert_v34 : `bool`, optional
        Will undo the flipping of the raster step axis made to V34 observations.
        Passed to `irispy.io.read_spectrograph_lvl2` only; slit-jaw, AIA and SOT files are not affected.
        Defaults to `False`.
    extract_dir : `str` or `pathlib.Path`, optional
        The folder that tar files are extracted into, each in a subfolder named after it and its path.
        Defaults to the ``irispy`` cache folder, ``astropy.config.paths.get_cache_dir("irispy")``
        (``~/.cache/irispy`` unless ``XDG_CACHE_HOME`` is set), so the archive's own folder
        can be read-only. An archive extracted there before is not extracted again. The extracted
        files stay until you delete them; see :ref:`irispy-known-issues-extracted-archives`.

    Returns
    -------
    `irispy.sji.SJICube`, `irispy.sji.AIACube`, `irispy.sji.SOTCube`, `irispy.spectrograph.RasterCollection` or `ndcube.NDCollection`
        What the files hold, if they hold one product: an `~irispy.sji.SJICube` (or AIA or SOT cube) for a slit-jaw file,
        a `~irispy.spectrograph.RasterCollection` keyed by spectral window for the raster files of one observation.
        Several products come back in an `~ndcube.NDCollection` keyed by band or measured quantity, with the
        filename added to repeated names; rasters are keyed by the file's first spectral window, such as
        ``"C II 1336"``, with the OBSID, then also STARTOBS, added to repeated names.

    Raises
    ------
    ValueError
        If ``filenames`` holds a full-disk mosaic, or no supported IRIS file could be loaded.
    Exception
        Whatever reading a file raises, unless ``allow_errors=True``, which logs a warning and skips the file.

    Examples
    --------
    One slit-jaw file gives an `~irispy.sji.SJICube`; one raster file gives a
    `~irispy.spectrograph.RasterCollection` keyed by spectral window:

    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import read_files
    >>> sji = read_files(
    ...     get_test_filepath(
    ...         "sns/iris_l2_20210905_001833_3620258102_SJI_1400_t000_test.fits"
    ...     )
    ... )
    >>> sji.shape
    (62, 40, 37)
    >>> filename = get_test_filepath(
    ...     "sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits"
    ... )
    >>> raster = read_files(filename)
    >>> len(raster), raster["Si IV 1403"].shape
    (8, (1, 187, 40, 29))

    The same on the sample data, reading one spectral window of the rasters:

    >>> import irispy.data.sample as sample_data  # doctest: +REMOTE_DATA
    >>> type(read_files(sample_data.SJI_1330)).__name__  # doctest: +REMOTE_DATA
    'SJICube'
    >>> window = "Mg II k 2796"
    >>> raster = read_files(
    ...     sample_data.RASTER_FITS, spectral_windows=window
    ... )  # doctest: +REMOTE_DATA
    >>> list(raster.keys())  # doctest: +REMOTE_DATA
    ['Mg II k 2796']
    """
    given = [filenames] if isinstance(filenames, (str, bytes, Path)) else list(filenames)
    paths = []
    for f in map(Path, given):
        if f.is_dir():
            # The IRIS tarballs and files, and the AIA and SOT files extracted from the tarballs.
            # An archive download folder also holds iris_l2_<obs>_movies.tar, the quick-look movies.
            paths.extend(
                p
                for p in f.iterdir()
                if p.name.startswith(("iris_l2_", "aia_l2_", "sot_l2_", "sotsp_l2_"))
                and p.name.endswith((".fits", ".fits.gz", ".tar.gz"))
            )
        else:
            paths.append(f)
    filenames = sorted({p.resolve() for p in paths})
    if spectral_windows is not None and not isinstance(spectral_windows, str):
        # Each observation is read with them, and a generator would be used up by the first
        spectral_windows = list(spectral_windows)
    returns = {}
    spec_groups = {}
    for filename in filenames:
        if "IRISMosaic_" in filename.name:
            msg = f"{filename} is a full-disk mosaic; read it with irispy.io.read_mosaic"
            raise ValueError(msg)
        sji_tarfile = filename.name.endswith(("SDO.tar.gz", "SOTFG.tar.gz", "SOTSP.tar.gz"))
        raster_tarfile = filename.name.endswith("_raster.tar.gz")
        try:
            # The members of a tarball are classified like loose files, so the rasters of several
            # tarballs are grouped by observation instead of overwriting each other by first window.
            files = (
                sorted(_extract_tarfile([filename], extract_dir=extract_dir))
                if sji_tarfile or raster_tarfile
                else [filename]
            )
        except Exception as e:
            if allow_errors:
                log.warning(f"File {filename} failed to load with {e}")
                continue
            raise
        # One member of a tarball failing to load does not stop the others from loading.
        for file in files:
            try:
                try:
                    header = _primary_header(file) if file.name.endswith((".fits", ".fits.gz")) else None
                except (OSError, EOFError, ValueError, TypeError):
                    # A file that is not FITS: raises the level 2 error, chained to this one
                    _check_level2(file, None)
                product = _check_level2(file, header)
                _, describe = _get_simple_metadata(header)
                log.debug(f"Processing file: {file} as {product}")
                if product == "SJI":
                    # SOT-SP archives may contain multiple observations of the same quantity.
                    key = describe if describe not in returns else f"{describe} ({file.stem})"
                    suffix = 2
                    while key in returns:
                        key = f"{describe} ({file.stem}, {suffix})"
                        suffix += 1
                    returns[key] = read_sji_lvl2(file, memmap=memmap, uncertainty=uncertainty)
                else:
                    # Keep the first file's description, so the group needs no second header read.
                    group_key = _get_spec_group_key(file, header)
                    spec_groups.setdefault(group_key, (describe, []))[1].append(file)
            except Exception as e:
                if allow_errors:
                    log.warning(f"File {file} failed to load with {e}")
                    continue
                raise
    for group_key, (describe, file_group) in spec_groups.items():
        try:
            key = _get_spec_return_key(group_key, describe, returns)
            returns[key] = read_spectrograph_lvl2(
                file_group,
                spectral_windows=spectral_windows,
                memmap=memmap,
                uncertainty=uncertainty,
                revert_v34=revert_v34,
            )
        except Exception as e:
            if allow_errors:
                log.warning(f"File group {file_group} failed to load with {e}")
                continue
            raise
    if not returns:
        msg = f"No supported IRIS files were loaded from {', '.join(map(str, given)) or '[]'}."
        raise ValueError(msg)
    return NDCollection(returns.items()) if len(returns) > 1 else next(iter(returns.values()))
