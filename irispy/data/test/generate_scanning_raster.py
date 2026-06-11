"""
Generate the synthetic scanning-raster fixture in ``scanning_raster/``.

Derives four tiny raster files (one complete 4-step scan each) from the
bundled 20140329 raster file, so an offline test can exercise the
``read_files`` tar path and the multi-scan combine contract that otherwise
only the remote-data test covers.

Run from the repository root::

    python irispy/data/test/generate_scanning_raster.py
"""

from pathlib import Path

import numpy as np

from astropy.io import fits

N_SCANS = 4
N_STEPS = 4
N_SLIT = 24
N_WAVE = 16
SCAN_CADENCE_S = 60.0
SYNTHETIC_OBSID = 4000000001
WINDOWS = ("Si IV 1403", "Mg II k 2796")

SOURCE = (
    Path(__file__).parent
    / "raster/iris_l2_20140329_140938_3860258481_raster/iris_l2_20140329_140938_3860258481_raster_t000_r00000.fits"
)
OUTPUT_DIR = Path(__file__).parent / "scanning_raster"


def _window_indices(primary_header):
    windows = [primary_header[f"TDESC{i}"] for i in range(1, primary_header["NWIN"] + 1)]
    return [windows.index(window) + 1 for window in WINDOWS]


def _build_primary_header(source_header, scan):
    header = source_header.copy()
    original_indices = _window_indices(source_header)
    n_windows = source_header["NWIN"]
    for per_window_key in ("TDESC", "TDET", "TWMIN", "TWMAX", "TWAVE"):
        values = [source_header.get(f"{per_window_key}{index}") for index in original_indices]
        for index in range(1, n_windows + 1):
            header.remove(f"{per_window_key}{index}", ignore_missing=True)
        for new_index, value in enumerate(values, start=1):
            if value is not None:
                header[f"{per_window_key}{new_index}"] = value
    header["NWIN"] = len(WINDOWS)
    header["OBSID"] = SYNTHETIC_OBSID
    header["RASRPT"] = scan + 1
    header["RASNRPT"] = N_SCANS
    return header


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    with fits.open(SOURCE) as hdulist:
        window_ext_indices = _window_indices(hdulist[0].header)
        for scan in range(N_SCANS):
            primary = fits.PrimaryHDU(header=_build_primary_header(hdulist[0].header, scan))
            windows = []
            for ext in window_ext_indices:
                data = np.ascontiguousarray(hdulist[ext].data[:N_STEPS, :N_SLIT, :N_WAVE])
                windows.append(fits.ImageHDU(data=data, header=hdulist[ext].header.copy()))
            aux_header = hdulist[-2].header.copy()
            aux_data = hdulist[-2].data[:N_STEPS].copy()
            # Each synthetic scan repeats the same pointing, offset in time.
            aux_data[:, aux_header["TIME"]] += scan * SCAN_CADENCE_S
            aux = fits.ImageHDU(data=aux_data, header=aux_header)
            # The reader never touches the final extension; it only needs to
            # exist so the spectral windows sit at indices 1..NWIN.
            last = fits.BinTableHDU.from_columns(
                [fits.Column(name="FILENAME", format="8A", array=np.array(["synthetic"]))]
            )
            out = OUTPUT_DIR / (
                f"iris_l2_20140329_140938_{SYNTHETIC_OBSID}_raster_t000_r{scan:05d}.fits"
            )
            fits.HDUList([primary, *windows, aux, last]).writeto(out, overwrite=True)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
