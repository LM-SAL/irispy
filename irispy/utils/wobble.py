import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patheffects

from astropy.io import fits
from astropy.time import TimeDelta
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS

from sunpy.time import parse_time
from sunpy.visualization.colormaps.color_tables import iris_sji_color_table

__all__ = ["generate_wobble_movie"]


def generate_wobble_movie(
    files: list | str | Path,
    *,
    outdir: str | Path = "./",
    trim: bool = False,
    timestamp: bool = True,
    wobble_cadence: int = 180,
    ffmpeg_path: str | Path | None = None,
    **kwargs,
) -> None:
    """
    Creates a wobble movie from a list of files.

    ..note:

        This requires FFMPEG to be installed and discoverable.
        If FFMPEG is not found, you can pass it as an argument called ``ffmpeg_path``.

    Parameters
    ----------
    files : Union[list, str, pathlib.Path]
        Files to create a wobble movie from.
        If a string or Path is passed, it will encapsulated in a list.
    outdir : Union[str, pathlib.Path], optional
        Location to save the movie(s).
        Defaults to the current working directory.
    trim : `bool`, optional
        Movie is trimmed to include only area that has data in all frames, by default False
    timestamp : `bool`, optional
        If `True`, will add a timestamp to the wobble movie.
        Optional, defaults to `True`.
    wobble_cadence : `int`, optional
        Sets the cadence of the wobble movie in seconds.
        Optional, defaults to 180 seconds.
    ffmpeg_path : Union[str, pathlib.Path], optional
        Path to FFMPEG executable, by default `None`.
        In theory you will not need to do this but matplotlib might not be able to find the ffmpeg exe.
    **kwargs : `dict`, optional
        Keyword arguments to passed to `matplotlib.animation.FuncAnimation`.

    Returns
    -------
    `list`
        A list of the movies created.

    Notes
    -----
    This is designed to be used on IRIS Level 2 SJI data.

    2832 is considered the best wavelength to use for wobble movies.

    Timestamps take the main header cadence and add that to the "DATEOBS".
    They do not use the information in the AUX array.
    """
    # Avoid circular imports
    from irispy.utils import image_clipping  # NOQA: PLC0415

    if ffmpeg_path:
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    if isinstance(files, str | Path):
        files = [files]
    filenames = []
    for a_file in files:
        data, header = fits.getdata(a_file, header=True)
        wcs = WCS(header)
        numframes = header["NAXIS3"]
        date = header["DATE_OBS"].split(".")[0]
        # Calculate index to downsample in time to accentuate the wobble
        cadence = header["CDELT3"]
        cadence_sample = max(1, np.floor(wobble_cadence / cadence))
        if timestamp:
            timestamps = [
                parse_time(header["STARTOBS"]) + TimeDelta(cadence, format="sec") * i for i in range(numframes)
            ]
        else:
            timestamps = [parse_time(header["STARTOBS"])]
        # Trim down to only that part of the movie that contains data in all frames
        if trim:
            # TODO: improve this, it trims a bit but not fully
            dmin = np.min(data, axis=0)
            dmask = dmin > -200
            dmx = np.sum(dmask, axis=1)
            dmy = np.sum(dmask, axis=0)
            (subx,) = np.where(dmx > (np.max(dmx) * 0.8))
            (suby,) = np.where(dmy > (np.max(dmy) * 0.8))
            data = data[:, suby[0] : suby[-1], subx[0] : subx[-1]]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=wcs.dropaxis(-1))
        colormap = iris_sji_color_table(str(int(header["TWAVE1"])))
        vmin, vmax = image_clipping(data)
        image = ax.imshow(
            data[0],
            origin="lower",
            cmap=colormap,
            norm=ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()),
        )
        ax.set_xlabel("Solar X")
        ax.set_ylabel("Solar Y")
        if timestamp:
            title = ax.text(
                0.5,
                0.95,
                str(timestamps[0]),
                color="w",
                transform=ax.transAxes,
                ha="center",
                path_effects=[patheffects.withStroke(linewidth=3, foreground="black")],
            )
        else:
            title = ax.text(0.5, 0.95, "")

        def update(i):
            image.set_array(data[i])  # NOQA: B023
            if timestamp:
                title.set_text(str(timestamps[i]))  # NOQA: B023
            return image, title  # NOQA: B023

        anim = animation.FuncAnimation(
            fig,
            func=update,
            frames=range(0, numframes, int(cadence_sample)),
            blit=True,
            repeat=False,
            **kwargs,
        )
        clean_filename = re.sub(r"[^\w\-_\. ]", "_", f"{header['TDESC1']}_{date}_wobble.mp4")
        filename = Path(outdir) / Path(clean_filename)
        writervideo = animation.FFMpegWriter(fps=12)
        anim.save(filename, writer=writervideo)
        filenames.append(filename)
    return filenames
