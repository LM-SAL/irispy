r"""
Cut real IRIS level 2 FITS files down to the exposures, windows and pixels a test needs.

Unlike ``compress.py``, which keeps every tenth pixel, this keeps index ranges, so the
kept values and their neighbours are exactly those of the whole file:

- The data values are unchanged, raw integers with their BSCALE/BZERO.
- The auxiliary table and the level 1 filename table keep the rows of the kept
    exposures, and NEXP (and NRASTERP for a raster) counts them.
- CRPIXn follow each cut. Evenly spaced exposures also update CDELT3 (and STEPS_AV/STEPT_AV
    for a raster) as ``compress.py`` does; otherwise CDELT3/CRPIX3 describe the whole file.
- Spectrograph windows are renumbered 1 to NWIN in the order given, with TWMINn/TWMAXn
    for the kept wavelengths. A window can be kept more than once, under a new name and
    rest wavelength. Its other keywords, such as TSR/TER/TSC/TEC and the statistics,
    still describe the whole window.
- The understated TFIELDS of the filename table is repaired, as in ``compress.py``.

Examples, run from this directory::

    python cut.py iris_l2_20130902_163935_4000255147_SJI_1400_t000.fits \
        bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits --exposures 0,69,227
    python cut.py iris_l2_20130902_182935_4000005156_raster_t000_r00000.fits \
        bursts/iris_l2_20130902_182935_4000005156_raster_t000_r00000_si_iv_test.fits \
        --rows 385:576 --window "Si IV 1403" 262:305
"""

import re
import argparse
from contextlib import ExitStack

import numpy as np

from astropy.io import fits


def _range(text):
    start, stop = (int(value) for value in text.split(":"))
    return slice(start, stop)


def _exposures(text, count):
    if ":" in text:
        return np.arange(count)[slice(*(int(value) if value else None for value in text.split(":")))]
    return np.array([int(value) for value in text.split(",")])


def _window_cards(header):
    """
    The cards of each window, from the block of numbered keywords that follows NWIN.
    """
    cards = header.cards
    start = stop = header.index("NWIN") + 1
    windows = {}
    while stop < len(cards) and (match := re.fullmatch(r"(.*?)(\d+)", cards[stop].keyword)):
        windows.setdefault(int(match[2]), []).append((match[1], cards[stop]))
        stop += 1
    return start, stop, windows


def cut(source, output, *, exposures=None, rows=None, columns=None, windows=None):
    with ExitStack() as files:
        hdus = files.enter_context(fits.open(source, memmap=True, do_not_scale_image_data=True))
        # Repair understated TFIELDS before the table is first parsed - astropy caches the parse
        table_header = hdus[-1].header
        table_header["TFIELDS"] = sum(1 for key in table_header if key.startswith("TTYPE"))
        primary = hdus[0].header
        spectrograph = "SPEC" in primary["INSTRUME"]
        images = list(hdus[1:-2]) if spectrograph else [hdus[0]]
        if windows:
            start, stop, cards = _window_cards(primary)
            names = [primary[f"TDESC{number}"] for number in range(1, primary["NWIN"] + 1)]
            numbers = [names.index(name) + 1 for name, _, _ in windows]
            # The HDUs are cut in place, which keeps the raw data and their scaling. A window kept
            # twice is read again, as a copied HDU would lose the scaling.
            kept = [
                files.enter_context(fits.open(source, memmap=True, do_not_scale_image_data=True))[number]
                if number in numbers[:index]
                else images[number - 1]
                for index, number in enumerate(numbers)
            ]
            block = []
            for new, (image, number, (_, pixels, rename)) in enumerate(
                zip(kept, numbers, windows, strict=True), start=1
            ):
                values = {}
                if pixels is not None:
                    header = image.header
                    wavelength = (
                        header["CRVAL1"]
                        + (np.array([pixels.start, pixels.stop - 1]) + 1 - header["CRPIX1"]) * header["CDELT1"]
                    )
                    values = {"TWMIN": float(wavelength[0]), "TWMAX": float(wavelength[1])}
                    header["CRPIX1"] -= pixels.start
                    image.data = image.data[..., pixels]
                if rename is not None:
                    values |= {"TDESC": rename[0], "TWAVE": rename[1]}
                block += [
                    (f"{prefix}{new}", values.get(prefix, card.value), card.comment) for prefix, card in cards[number]
                ]
            for index in reversed(range(start, stop)):
                del primary[index]
            for index, card in enumerate(block, start=start):
                primary.insert(index, card)
            primary["NWIN"] = len(windows)
            hdus = fits.HDUList([hdus[0], *kept, *hdus[-2:]])
            images = kept
        if exposures is not None:
            step = np.diff(exposures)
            even = len(exposures) == 1 or (step[0] > 0 and np.all(step == step[0]))
            stride = int(step[0]) if len(exposures) > 1 else 1
            for image in images:
                image.data = image.data[exposures]
                if even:
                    image.header["CDELT3"] *= stride
                    image.header["CRPIX3"] = (image.header["CRPIX3"] - 1 - exposures[0]) / stride + 1
            for table in hdus[-2:]:
                table.data = table.data[exposures]
            primary = hdus[0].header
            primary["NEXP"] = len(exposures)
            if spectrograph and primary["NRASTERP"] > 1:
                primary["NRASTERP"] = len(exposures)
                if even:
                    primary["STEPS_AV"] *= stride
                    primary["STEPT_AV"] *= stride
        for axis, pixels in ((2, rows), (1, columns)):
            if pixels is not None:
                for image in images:
                    image.data = image.data[(slice(None),) * (3 - axis) + (pixels,)]
                    image.header[f"CRPIX{axis}"] -= pixels.start
        hdus.writeto(output, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Cut an IRIS level 2 FITS file down to index ranges")
    parser.add_argument("source", help="The whole level 2 file")
    parser.add_argument("output", help="The file to write")
    parser.add_argument("--exposures", help="Exposures (raster steps or slit-jaw frames) to keep: 0,69,227 or 0:400:57")
    parser.add_argument("--rows", type=_range, help="Slit positions, or slit-jaw rows, to keep: 385:576")
    parser.add_argument("--columns", type=_range, help="Slit-jaw columns to keep: 100:250")
    parser.add_argument(
        "--window",
        nargs="+",
        action="append",
        metavar="ARG",
        help="A window to keep, in output order: NAME [PIXELS [NEW_NAME NEW_WAVELENGTH]], such as "
        '"Mg II k 2796" 364:398 "Mg II h 2803" 2803.53',
    )
    args = parser.parse_args()
    windows = [
        (name, _range(rest[0]) if rest else None, (rest[1], float(rest[2])) if len(rest) == 3 else None)
        for name, *rest in args.window or []
    ]
    with fits.open(args.source, memmap=True) as hdus:
        count = hdus[0].header["NEXP"]
    exposures = None if args.exposures is None else _exposures(args.exposures, count)
    cut(args.source, args.output, exposures=exposures, rows=args.rows, columns=args.columns, windows=windows)


if __name__ == "__main__":
    main()
