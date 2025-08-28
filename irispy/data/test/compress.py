"""
Short script I used to create the test FITS files in this folder.

WARNING: This overrides the original files.
"""


def compress(files: list) -> None:
    from scipy.ndimage import zoom  # NOQA: PLC0415

    from astropy.io import fits  # NOQA: PLC0415

    for file in files:
        hdus = fits.open(file)
        sg = "SPEC" in hdus[0].header["INSTRUME"]
        sns = hdus[0].header["NRASTERP"] == 1
        for hdu in hdus:
            aux = hdu.header.get("XTENSION", "") == "IMAGE"
            hdu.verify("fix")
            if isinstance(hdu, fits.hdu.table.TableHDU) or hdu.data is None:
                continue
            # Can't pop out the array, resizing can cause issues
            # So I remove the data and move on.
            if hdu.header.get("NAXIS1") is None:
                continue
            if hdu.data.ndim == 1:
                hdu.data = None
                continue
            if hdu.data.ndim == 2:
                factor = (0.1, 1)
            elif hdu.data.ndim == 3:
                factor = (1, 0.1, 0.1) if sg and not sns else (0.1, 0.1, 0.1)
            hdu.data = zoom(hdu.data, factor, order=0)
            if hdu.data.ndim == 2 and not aux:
                hdu.header["NAXIS1"] = hdu.data.shape[1]
                hdu.header["NAXIS2"] = hdu.data.shape[0]
                hdu.header["CRPIX1"] = hdu.header["CRPIX1"] * factor[1]
                hdu.header["CRPIX2"] = hdu.header["CRPIX2"] * factor[0]
            elif hdu.data.ndim == 3 and not aux:
                hdu.header["NAXIS1"] = hdu.data.shape[2]
                hdu.header["NAXIS2"] = hdu.data.shape[1]
                hdu.header["NAXIS3"] = hdu.data.shape[0]
                hdu.header["CRPIX1"] = hdu.header["CRPIX1"] * factor[2]
                hdu.header["CRPIX2"] = hdu.header["CRPIX2"] * factor[1]
                hdu.header["CRPIX3"] = hdu.header["CRPIX3"] * factor[0]
        hdus.writeto(f"{file}", overwrite=True)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compress FITS files")
    parser.add_argument("files", nargs="+", help="Folder location of FITS files to compress")
    args = parser.parse_args()
    files = list(Path(args.files[0]).glob("*.fits"))
    compress(files)
