import sys
import json
import tarfile
from pathlib import Path
from contextlib import nullcontext

from filelock import FileLock

from astropy.io import fits
from astropy.table import Table

from ndcube import NDCollection
from sunpy import log

from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2

__all__ = ["fits_info", "read_files"]


def _get_simple_metadata(file):
    """
    Get simple metadata from a FITS file.

    Parameters
    ----------
    file : `pathlib.Path` or `astropy.io.fits.HDUList`
        The FITS file or open HDU list to inspect.

    Returns
    -------
    `tuple`
        A tuple containing the instrument name and description.
    """
    if isinstance(file, fits.HDUList):
        header = file[0].header
    else:
        if not file.name.endswith((".fits", ".fits.gz")):
            return "", ""
        header = fits.getheader(file)
    instrume = header.get("INSTRUME", "")
    describe = header.get("TDESC1", "")
    return instrume, describe


def _extract_tarfile(filenames):
    """
    Extracts a tar file to the same location as the tar file.

    A complete earlier extraction of the same tar file (same size and modification
    time) is reused while all of its files still exist.
    Concurrent callers wait for extraction to finish before reusing its files.

    Parameters
    ----------
    filenames : `list of str`
        The filenames of the tar files to extract.
    """
    expanded_files = []
    for fname in filenames:
        filename = Path(fname)
        if tarfile.is_tarfile(filename):
            extract_dir = filename.with_suffix("").with_suffix("")  # removes .tar.gz or .tar
            extract_dir.mkdir(parents=True, exist_ok=True)
            # The cache check must share the lock with extraction: a waiting caller can then reuse the files.
            with FileLock(extract_dir / ".irispy-extracted.lock"):
                # Written only after a complete extraction: the tar file it came from and the extracted files.
                marker = extract_dir / ".irispy-extracted.json"
                source = [filename.stat().st_size, filename.stat().st_mtime_ns]
                names = []
                if marker.is_file():
                    try:
                        extracted = json.loads(marker.read_text())
                    except ValueError:
                        extracted = {}
                    if extracted.get("source") == source:
                        names = extracted.get("files", [])
                if not names or not all((extract_dir / name).is_file() for name in names):
                    marker.unlink(missing_ok=True)
                    with tarfile.open(filename, "r") as tar:
                        tar.extractall(extract_dir, filter="data")
                        names = [member.name for member in tar.getmembers() if member.isfile()]
                    marker.write_text(json.dumps({"source": source, "files": names}))
            expanded_files.extend(extract_dir / name for name in names)
        else:
            expanded_files.append(filename)
    return expanded_files


def _get_spec_group_key(file):
    """
    Group spectrograph FITS files that belong to the same observation.

    Parameters
    ----------
    file : `pathlib.Path`
        The FITS file to inspect.

    Returns
    -------
    `tuple`
        Key built from stable observation-identifying header metadata.
        Falls back to a per-file key if the header cannot be read or if
        the observation-grouping metadata is missing.
    """
    try:
        header = fits.getheader(file)
    except OSError:
        return file, None
    obsid = header.get("OBSID")
    startobs = header.get("STARTOBS")
    if obsid is None or not startobs:
        return file, None
    return obsid, startobs


def _observation_identity(file, *, header=None, source_path=None):
    """
    Observation identity for the one-observation scope: ``(OBSID, STARTOBS)`` or a file-
    path fallback.

    Pass ``header`` when it is already loaded, to avoid re-reading (e.g. re-
    decompressing) the file.
    """
    if header is None:
        obsid, startobs = _get_spec_group_key(file)
    else:
        obsid, startobs = header.get("OBSID"), header.get("STARTOBS")
    if obsid is None or not startobs:
        return ("file", Path(file if source_path is None else source_path).resolve())
    return ("obs", obsid, startobs)


class _ReadFilesInputError(ValueError):
    """
    ``read_files`` rejected its input (multiple observations, unnameable product).

    Always raised, even under ``allow_errors``: this is a bad request, not a skippable
    load failure.
    """


def _reject_multiple_observations(observations):
    if len(observations) > 1:
        msg = (
            f"read_files reads one observation at a time; got {len(observations)}: "
            f"{sorted(map(str, observations))}. Read each observation separately."
        )
        raise _ReadFilesInputError(msg)


def fits_info(filename: str) -> None:
    """
    Prints information about the extension of a raster or SJI level 2 data file.

    Parameters
    ----------
    filename : str
        Filename to load.
    """

    def get_description(idx, idx_mod):
        # The last two extensions are reserved for auxiliary data.
        auxiliary_extension_indices = [num_extensions - 2, num_extensions - 1]
        if idx == 0 and idx_mod == 0:
            return "Primary Header (no data)"
        if idx not in auxiliary_extension_indices:
            text_modifier = (
                f" ({header[f'TDET{idx + idx_mod}'][:3]})" if "SPEC" in header[f"TDET{idx + idx_mod}"] else ""
            )
            return f"{header[f'TDESC{idx + idx_mod}'].replace('_', ' ')} ({header[f'TWMIN{idx + idx_mod}']:.0f} - {header[f'TWMAX{idx + idx_mod}']:.0f} AA{text_modifier})"
        return "Auxiliary data"

    results = [
        f"Filename: {Path(filename).absolute()}",
        f"Observation: {fits.getval(filename, 'OBS_DESC')}",
        f"OBS ID: {fits.getval(filename, 'OBSID')}",
    ]
    table = Table(
        names=("No.", "Name", "Ver", "Type", "Cards", "Dimensions", "Format", "Description"),
        dtype=("S8", "S32", "S8", "S16", "S8", "S40", "S16", "U200"),
    )
    with fits.open(filename) as hdulist:
        hdu_info = hdulist.info(output=False)
        header = hdulist[0].header
        num_extensions = len(hdulist)
        for idx in range(num_extensions):
            description = get_description(idx, 0 if "SPEC" in header["INSTRUME"] else 1)
            hdu_info[idx] = (*hdu_info[idx][:-1], description)
            table.add_row(list(map(str, hdu_info[idx])))

    sys.stdout.write("\n".join(results) + "\n")
    table.pprint()
    sys.stdout.flush()


def read_files(
    filenames, *, spectral_windows=None, uncertainty=False, memmap=False, raw=False, allow_errors=False, **kwargs
):
    """
    A wrapper function to read any number of raster, SJI or IRIS-aligned AIA data files.

    The goal is be able to download an entire IRIS observation and read it
    in one go, without having to worry about the type of file.

    Parameters
    ----------
    filename : `list` of `str`, `str`, `pathlib.Path`
        Filename(s) to load.
    spectral_windows: iterable of `str` or `str`
        Spectral windows to extract from files. Default=None, implies, extract all
        spectral windows.
    uncertainty : `bool`, optional
        If `True` (not the default), compute the uncertainty for the data. This requires scaled
        values and is not available for aligned AIA data.
    memmap : `bool`, optional
        If `True`, request FITS memory mapping where supported. Scaling may require materializing
        the data in memory; this option does not change whether returned values are raw or scaled.
    raw : `bool`, optional
        If `True`, return unscaled raw FITS values and retain ``BSCALE`` and ``BZERO`` in metadata.
        Raw values have count units and are rejected by scaled-data analysis functions. Raw data
        cannot be combined with ``uncertainty=True``. Defaults to `False` (scaled values).
    allow_errors : `bool`, optional
        Will continue loading the files if one fails to load.
        Defaults to `False`.
    kwargs : `dict`, optional
        Additional keyword arguments to pass to the reader functions.

    Returns
    -------
    `ndcube.NDCollection`
        Always returned, including for a single input. Entries are flat cubes or cube sequences.
        Keys are the bare product names: a spectral window such as ``"Si IV 1403"`` for spectrograph
        window sequences, an SJI channel such as ``"SJI_2832"`` for slit-jaw cubes, and an AIA
        channel such as ``"304_THIN"`` for aligned AIA cubes. Select a product directly, e.g.
        ``result["Si IV 1403"]``. When the same product name appears in more than one file (e.g. an
        original and a deconvolved SJI channel), the first keeps the bare name and later entries are
        disambiguated with their source filename (``"SJI_2832:…_deconvolved.fits.gz"``). Observation identity (``OBSID``, ``STARTOBS``) and the
        processing ``STATUS`` are in each cube's ``meta``, not the key. Use
        `~irispy.io.read_spectrograph_lvl2` for a `~irispy.spectrograph.RasterCollection` keyed by bare
        window names.

    Notes
    -----
    Reads a single observation: all inputs must share one ``OBSID``/``STARTOBS`` (or, for files without
    those headers such as AIA archives, one source path). If inputs span more than one observation, a
    `ValueError` is raised; read each observation separately. Exact duplicate input paths are read once;
    different files sharing a product name are both retained under filename-qualified keys. Unsupported
    files are skipped with a warning, and a `ValueError` is raised if no supported product loads (with
    ``allow_errors=True`` load failures are skipped with a warning instead). Compressed FITS files and
    the supported SDO/raster tar archives are handled by this function.
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = sorted({Path(f).resolve() for f in filenames})
    collected = []
    only_spectrograph = True
    spec_groups = {}
    spec_sources = {}
    observations = set()
    for filename in filenames:
        if filename.name.startswith("IRISMosaic_"):
            msg = f"{filename} is a full-disk mosaic; read it with irispy.io.read_mosaic"
            raise ValueError(msg)
        try:
            sdo_tarfile = bool(filename.name.endswith("SDO.tar.gz"))
            raster_tarfile = bool(filename.name.endswith("_raster.tar.gz"))
            context = (
                fits.open(filename, memmap=memmap and raw, do_not_scale_image_data=raw, decompress_in_memory=True)
                if filename.name.endswith((".fits", ".fits.gz"))
                else nullcontext()
            )
            with context as hdulist:
                instrume, describe = _get_simple_metadata(hdulist if hdulist is not None else filename)
                log.debug(f"Processing file: {filename} with instrume: {instrume}")
                if sdo_tarfile or instrume in ["IRIS", "SJI"] or instrume.startswith("AIA"):
                    file = _extract_tarfile([filename]) if sdo_tarfile else [filename]
                    for f in file:
                        sji_context = (
                            fits.open(f, memmap=memmap and raw, do_not_scale_image_data=raw, decompress_in_memory=True)
                            if sdo_tarfile
                            else nullcontext(hdulist)
                        )
                        with sji_context as sji_hdulist:
                            _, describe = _get_simple_metadata(sji_hdulist)
                            product_name = describe or sji_hdulist[0].header.get("TDET") or "SJI"
                            source_path = filename if sdo_tarfile else f
                            observations.add(
                                _observation_identity(f, header=sji_hdulist[0].header, source_path=source_path)
                            )
                            _reject_multiple_observations(observations)
                            only_spectrograph = False
                            collected.append(
                                (
                                    product_name,
                                    f.name,
                                    read_sji_lvl2(
                                        sji_hdulist, memmap=memmap, raw=raw, uncertainty=uncertainty, **kwargs
                                    ),
                                )
                            )
                elif raster_tarfile:
                    file = _extract_tarfile([filename])
                    group_key = _get_spec_group_key(file[0])
                    if group_key[1] is None:
                        group_key = (filename, None)
                    spec_groups.setdefault(group_key, []).extend(file)
                    spec_sources.setdefault(group_key, filename)
                elif instrume == "SPEC":
                    group_key = _get_spec_group_key(filename)
                    spec_groups.setdefault(group_key, []).append(filename)
                    spec_sources.setdefault(group_key, filename)
                else:
                    log.warning(f"File {filename} has unrecognized INSTRUME={instrume!r} and was not loaded")
        except _ReadFilesInputError:
            raise
        except Exception as e:
            if allow_errors:
                log.warning(f"File {filename} failed to load with {e}")
                continue
            raise
    for group_key, file_group in spec_groups.items():
        try:
            collection = read_spectrograph_lvl2(
                file_group, spectral_windows=spectral_windows, memmap=memmap, raw=raw, uncertainty=uncertainty, **kwargs
            )
            observations.add(_observation_identity(file_group[0], source_path=spec_sources[group_key]))
            _reject_multiple_observations(observations)
            source_name = Path(spec_sources[group_key]).name
            collected.extend((window, source_name, sequence) for window, sequence in collection.items())
        except _ReadFilesInputError:
            raise
        except Exception as e:
            if allow_errors:
                log.warning(f"File group {file_group} failed to load with {e}")
                continue
            raise
    if not collected:
        msg = f"No supported IRIS files were loaded from {filenames}."
        raise ValueError(msg)
    groups = {}
    for product_name, source_name, value in collected:
        groups.setdefault(product_name, []).append((source_name, value))
    keyed = []
    for product_name, entries in groups.items():
        # The first entry (input order) keeps the bare name; later ones (e.g. a deconvolved
        # variant) are filename-qualified so the primary product stays clean and stable.
        keyed.append((product_name, entries[0][1]))
        keyed.extend((f"{product_name}:{source_name}", value) for source_name, value in entries[1:])
    keys = [key for key, _ in keyed]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        msg_0 = f"Cannot uniquely name products: {duplicates}"
        raise _ReadFilesInputError(msg_0)
    returns = dict(keyed)
    if only_spectrograph:
        return NDCollection(returns.items(), aligned_axes=(0, 1, 2))
    return NDCollection(returns.items())
