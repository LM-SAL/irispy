"""
Shrink real IRIS level 2 FITS files into small test files.

The outputs keep the structure of real files, and the test suite relies on
these invariants:

- Exposures (the time axis) are decimated by index selection; the data, the
    auxiliary table, the source-filename table and the NEXP keyword all
    describe the same selected exposures.
- Spatial and spectral axes are decimated with a stride, keeping real pixel
    values only (bad-pixel and saturation sentinels survive exactly), and
    CDELT/CRPIX are updated so world coordinates still span the original
    observation.
- The auxiliary table stays second-to-last and the level 1 source-filename
    table last, as in real files.
- Real spectrograph files understate TFIELDS in the source-filename table
    header (7 declared for 9 TTYPEn cards); this is repaired before the table
    is first parsed, since astropy caches the parse.
"""

import argparse
from pathlib import Path

import numpy as np

from astropy.io import fits

STRIDE = 10


def _output_path(file):
    return file.with_name(file.name.replace(".fits", "_test.fits", 1))


def _decimate_wcs(header, axis, stride):
    # Every stride-th pixel is kept starting from the first, so the pixel
    # scale grows by the stride and the (1-based) reference pixel moves
    if f"CDELT{axis}" in header:
        header[f"CDELT{axis}"] = header[f"CDELT{axis}"] * stride
    if f"CRPIX{axis}" in header:
        header[f"CRPIX{axis}"] = (header[f"CRPIX{axis}"] - 1) / stride + 1


def _time_indices(hdus, sg, sns):
    """
    The exposures to keep, or `None` to keep all of them (raster steps).
    """
    if sg and not sns:
        return None
    if sg:
        count = len(hdus[-2].data)
        valid = np.arange(count)
    else:
        # SJI: keep only exposures that have a level 1 source file
        source_filenames = hdus[-1].data["SJIfilename"]
        valid = np.flatnonzero(np.char.str_len(np.char.strip(source_filenames.astype(str))))
        count = len(hdus[0].data)
    target = max(2, round(count / STRIDE))
    return valid[np.rint(np.linspace(0, len(valid) - 1, target)).astype(int)]


def compress(files: list) -> None:
    from tqdm import tqdm  # NOQA: PLC0415

    for file in tqdm(files):
        with fits.open(file, memmap=False) as hdus:
            header = hdus[0].header
            if "INSTRUME" not in header or "NRASTERP" not in header:
                tqdm.write(f"Skipping {file}: not an IRIS level 2 file")
                continue
            # The layout this script relies on: the auxiliary per-exposure
            # table second-to-last, the source-filename table last
            if not isinstance(hdus[-1], (fits.TableHDU, fits.BinTableHDU)) or "PZTX" not in hdus[-2].header:
                tqdm.write(f"Skipping {file}: unexpected HDU layout")
                continue

            # Repair understated TFIELDS before the table is first parsed -
            # astropy caches the parse
            table_header = hdus[-1].header
            n_columns = sum(1 for key in table_header if key.startswith("TTYPE"))
            if table_header["TFIELDS"] != n_columns:
                table_header["TFIELDS"] = n_columns

            hdus.verify("fix")

            sg = "SPEC" in header["INSTRUME"]
            sns = header["NRASTERP"] == 1
            time_indices = _time_indices(hdus, sg, sns)

            for hdu in hdus:
                if isinstance(hdu, (fits.TableHDU, fits.BinTableHDU)) or "PZTX" in hdu.header:
                    # Per-exposure rows: the source-filename and auxiliary tables
                    if time_indices is not None:
                        hdu.data = hdu.data[time_indices]
                elif hdu.data is None:
                    continue
                elif hdu.data.ndim == 1:
                    # Resizing 1D extensions can cause issues, so drop the data
                    hdu.data = None
                elif hdu.data.ndim == 2:
                    hdu.data = hdu.data[::STRIDE]
                    _decimate_wcs(hdu.header, 2, STRIDE)
                elif hdu.data.ndim == 3:
                    if time_indices is not None:
                        hdu.data = hdu.data[time_indices]
                    hdu.data = hdu.data[:, ::STRIDE, ::STRIDE]
                    _decimate_wcs(hdu.header, 1, STRIDE)
                    _decimate_wcs(hdu.header, 2, STRIDE)
                else:
                    msg = "HDU with more than 3 dimensions not supported"
                    raise ValueError(msg)

            if time_indices is not None:
                header["NEXP"] = len(time_indices)
            hdus.writeto(_output_path(file), overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Shrink IRIS level 2 FITS files into *_test.fits copies")
    parser.add_argument("folder", type=Path, help="Folder of .fits/.fits.gz files to shrink")
    args = parser.parse_args()
    files = sorted(
        file
        for pattern in ("*.fits", "*.fits.gz")
        for file in args.folder.glob(pattern)
        if "_test" not in file.name
    )
    if not files:
        parser.error(f"no .fits or .fits.gz files found in {args.folder}")
    compress(files)


if __name__ == "__main__":
    main()
