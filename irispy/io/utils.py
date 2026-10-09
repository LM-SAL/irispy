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
    if instrume == "SOT-SP":
        # SOT-SP maps share a band name; BTYPE distinguishes the measured quantities.
        describe = f"{describe} {header.get('BTYPE', '')}".strip()
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


def _get_spec_return_key(file_group, describe, returns):
    key = f"{describe}"
    if key not in returns:
        return key
    group_key = _get_spec_group_key(file_group[0])
    obsid, startobs = group_key
    if startobs is None:
        return f"{describe} ({Path(obsid).stem})"
    key = f"{describe} ({obsid})"
    if key not in returns:
        return key
    return f"{describe} ({obsid}, {startobs})"


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


def read_files(filenames, *, spectral_windows=None, uncertainty=False, memmap=False, allow_errors=False, **kwargs):
    """
    Read raster, SJI, AIA, or Hinode/SOT data supplied with IRIS observations.

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
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), FITS data are opened using Astropy/NumPy
        memory mapping rather than being fully read into memory at once. This
        can keep memory usage low when working with many files, since array
        data are accessed from disk on demand. In this mode FITS image scaling
        is disabled, so the returned data are unscaled/raw FITS values rather
        than automatically scaled physical values. If ``memmap=True``, the
        uncertainty is never computed.
    allow_errors : `bool`, optional
        Will continue loading the files if one fails to load.
        Defaults to `False`.
    kwargs : `dict`, optional
        Additional keyword arguments to pass to the reader functions.

    Returns
    -------
    `ndcube.NDCollection`
        Cubes keyed by band or measured quantity. Repeated names include the filename.
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = sorted(filenames)
    filenames = [Path(f) for f in filenames]
    returns = {}
    spec_groups = {}
    for filename in filenames:
        if filename.name.startswith("IRISMosaic_"):
            msg = f"{filename} is a full-disk mosaic; read it with irispy.io.read_mosaic"
            raise ValueError(msg)
        try:
            sji_tarfile = filename.name.endswith(("SDO.tar.gz", "SOTFG.tar.gz", "SOTSP.tar.gz"))
            raster_tarfile = bool(filename.name.endswith("_raster.tar.gz"))
            context = (
                fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap, decompress_in_memory=True)
                if filename.name.endswith((".fits", ".fits.gz"))
                else nullcontext()
            )
            with context as hdulist:
                instrume, describe = _get_simple_metadata(hdulist if hdulist is not None else filename)
                log.debug(f"Processing file: {filename} with instrume: {instrume}")
                if sji_tarfile or instrume in ["IRIS", "SJI"] or instrume.startswith(("AIA", "SOT")):
                    file = _extract_tarfile([filename]) if sji_tarfile else [filename]
                    for f in sorted(file):
                        sji_context = (
                            fits.open(f, memmap=memmap, do_not_scale_image_data=memmap, decompress_in_memory=True)
                            if sji_tarfile
                            else nullcontext(hdulist)
                        )
                        with sji_context as sji_hdulist:
                            instrume, describe = _get_simple_metadata(sji_hdulist)
                            # SOT-SP archives may contain multiple observations of the same quantity.
                            key = describe if describe not in returns else f"{describe} ({f.stem})"
                            returns[key] = read_sji_lvl2(sji_hdulist, memmap=memmap, uncertainty=uncertainty, **kwargs)
                elif raster_tarfile:
                    file = _extract_tarfile([filename]) if raster_tarfile else [filename]
                    instrume, describe = _get_simple_metadata(file[0])
                    returns[f"{describe}"] = read_spectrograph_lvl2(
                        file, spectral_windows=spectral_windows, memmap=memmap, uncertainty=uncertainty, **kwargs
                    )
                elif instrume == "SPEC":
                    group_key = _get_spec_group_key(filename)
                    spec_groups.setdefault(group_key, []).append(filename)
                else:
                    log.warning(f"File {filename} has unrecognized INSTRUME={instrume!r} and was not loaded")
        except Exception as e:
            if allow_errors:
                log.warning(f"File {filename} failed to load with {e}")
                continue
            raise
    for file_group in spec_groups.values():
        try:
            instrume, describe = _get_simple_metadata(file_group[0])
            key = _get_spec_return_key(file_group, describe, returns)
            returns[key] = read_spectrograph_lvl2(
                file_group, spectral_windows=spectral_windows, memmap=memmap, uncertainty=uncertainty, **kwargs
            )
        except Exception as e:
            if allow_errors:
                log.warning(f"File group {file_group} failed to load with {e}")
                continue
            raise
    if not returns:
        msg = f"No supported IRIS files were loaded from {filenames}."
        raise ValueError(msg)
    return NDCollection(returns.items()) if len(returns) > 1 else next(iter(returns.values()))
