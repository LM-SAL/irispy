import sys
import tarfile
from pathlib import Path

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
    file : `pathlib.Path`
        The FITS file to read.

    Returns
    -------
    `tuple`
        A tuple containing the instrument name and description.
    """
    if not file.name.endswith(".fits") and not file.name.endswith(".fits.gz"):
        return "", ""
    header = fits.getheader(file)
    instrume = header.get("INSTRUME", "")
    describe = header.get("TDESC1", "")
    return instrume, describe


def _extract_tarfile(filenames):
    """
    Extracts a tar file to the same location as the tar file.

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
            with tarfile.open(filename, "r") as tar:
                tar.extractall(extract_dir, filter="data")
                expanded_files.extend([extract_dir / member.name for member in tar.getmembers() if member.isfile()])
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
        With keys being the value of TDESC1, the values being the cube.
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = sorted(filenames)
    filenames = [Path(f) for f in filenames]
    returns = {}
    spec_groups = {}
    for filename in filenames:
        try:
            sdo_tarfile = bool(filename.name.endswith("SDO.tar.gz"))
            raster_tarfile = bool(filename.name.endswith("_raster.tar.gz"))
            instrume, describe = _get_simple_metadata(filename)
            log.debug(f"Processing file: {filename} with instrume: {instrume}")
            if sdo_tarfile or instrume in ["IRIS", "SJI"] or instrume.startswith("AIA"):
                file = _extract_tarfile([filename]) if sdo_tarfile else [filename]
                for f in file:
                    instrume, describe = _get_simple_metadata(f)
                    returns[f"{describe}"] = read_sji_lvl2(f, memmap=memmap, uncertainty=uncertainty, **kwargs)
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
                log.warning(f"INSTRUME: {instrume} was not recognized and not loaded")
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
