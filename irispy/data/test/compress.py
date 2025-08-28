"""
Short script I used to create the test FITS files in this folder.

WARNING: This overrides the original files.
"""


def compress(files: list) -> None:
    from scipy.ndimage import zoom  # NOQA: PLC0415
    from tqdm import tqdm  # NOQA: PLC0415

    from astropy.io import fits  # NOQA: PLC0415

    for file in tqdm(files):
        hdus = fits.open(file)
        sg = "SPEC" in hdus[0].header["INSTRUME"]
        sns = hdus[0].header["NRASTERP"] == 1
        for hdu in hdus:
            aux = "PZTX" in hdu.header
            hdu.verify("fix")
            if isinstance(hdu, fits.hdu.table.TableHDU) or hdu.data is None:
                continue
            if aux:
                # Only resize the sit and stare data as its 3D
                # The raster image has steps and I want to keep them
                if sns:
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
                factor = (1, 0.1, 0.1) if sg and not sns else (0.1, 0.1, 0.1)
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


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compress FITS files")
    parser.add_argument("files", nargs="+", help="Folder location of FITS files to compress")
    args = parser.parse_args()
    files = list(Path(args.files[0]).glob("*.fits"))
    compress(files)
