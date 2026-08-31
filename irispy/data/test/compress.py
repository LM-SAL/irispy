"""
Short script I used to create the test FITS files in this folder.

WARNING: This overrides the original files.
"""

import numpy as np

if __name__ == "__main__":

    def compress(files: list) -> None:
        from scipy.ndimage import zoom  # NOQA: PLC0415
        from tqdm import tqdm  # NOQA: PLC0415

        from astropy.io import fits  # NOQA: PLC0415

        for file in tqdm(files):
            hdus = fits.open(file)
            sg = "SPEC" in hdus[0].header["INSTRUME"]
            sns = hdus[0].header["NRASTERP"] == 1
            time_indices = None
            if not sg:
                source_filenames = hdus[-1].data["SJIfilename"]
                valid_indices = np.flatnonzero(np.char.str_len(np.char.strip(source_filenames.astype(str))))
                target_length = round(len(hdus[0].data) * 0.1)
                time_indices = valid_indices[np.rint(np.linspace(0, len(valid_indices) - 1, target_length)).astype(int)]
            for hdu in hdus:
                aux = "PZTX" in hdu.header
                hdu.verify("fix")
                if isinstance(hdu, fits.hdu.table.TableHDU):
                    if not sg:
                        hdu.data = hdu.data[time_indices]
                    elif sns:
                        if "TTYPE9" in hdu.header:
                            hdu.header["TFIELDS"] = 9
                        target_length = len(hdus[-2].data)
                        indices = np.rint(np.linspace(0, len(hdu.data) - 1, target_length)).astype(int)
                        hdu.data = hdu.data[indices]
                        # Keep the truncated file self-consistent - readers
                        # validate the table length against the exposures.
                        hdus[0].header["NEXP"] = target_length
                    continue
                if hdu.data is None:
                    continue
                if aux:
                    if not sg:
                        hdu.data = hdu.data[time_indices]
                    # Only resize spectrograph sit-and-stare AUX data.
                    elif sns:
                        factor = (0.1, 1)
                        hdu.data = zoom(hdu.data, factor)
                elif hdu.data.ndim == 1:
                    # Can't pop out the array, resizing can cause issues
                    # So I remove the data and move on.
                    hdu.data = None
                    continue
                elif hdu.data.ndim == 2:
                    factor = (0.1, 1)
                    hdu.data = zoom(hdu.data, factor)
                    hdu.header["NAXIS1"] = hdu.data.shape[1]
                    hdu.header["NAXIS2"] = hdu.data.shape[0]
                    hdu.header["CRPIX1"] = hdu.header["CRPIX1"] * factor[1]
                    hdu.header["CRPIX2"] = hdu.header["CRPIX2"] * factor[0]
                elif hdu.data.ndim == 3:
                    if not sg:
                        hdu.data = hdu.data[time_indices]
                    factor = (1, 0.1, 0.1) if not sg or not sns else (0.1, 0.1, 0.1)
                    hdu.data = zoom(hdu.data, factor)
                    hdu.header["NAXIS1"] = hdu.data.shape[2]
                    hdu.header["NAXIS2"] = hdu.data.shape[1]
                    hdu.header["NAXIS3"] = hdu.data.shape[0]
                    hdu.header["CRPIX1"] = hdu.header["CRPIX1"] * factor[2]
                    hdu.header["CRPIX2"] = hdu.header["CRPIX2"] * factor[1]
                    hdu.header["CRPIX3"] = hdu.header["CRPIX3"] * factor[0]
                else:
                    msg = "HDU with more than 3 dimensions not supported"
                    raise ValueError(msg)
                hdu = fits.CompImageHDU(hdu.data, hdu.header)  # NOQA: PLW2901
            hdus.writeto(f"{file}", overwrite=True)

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compress FITS files")
    parser.add_argument("files", nargs="+", help="Folder location of FITS files to compress")
    args = parser.parse_args()
    files = list(Path(args.files[0]).glob("*.fits"))
    compress(files)
