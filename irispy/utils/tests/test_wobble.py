import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation

from astropy.io import fits

from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED
from irispy.utils.wobble import generate_wobble_movie


@pytest.mark.parametrize(("trim", "fill"), [(False, -200), (True, -200), (True, -199)])
def test_generate_wobble_movie(sns_sji_2832_file, tmp_path, trim, fill):
    header = fits.getheader(sns_sji_2832_file)
    header["CUNIT3"] = "s"
    header["CDELT3"] = 180
    data = np.full((3, 8, 8), fill, dtype=float)
    data[:, 1:-1, 1:-1] = np.arange(108).reshape(3, 6, 6) + 1
    filename = tmp_path / "sji.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(filename)

    movies = generate_wobble_movie(filename, outdir=tmp_path, trim=trim)

    assert len(movies) == 1
    movie = movies[0]
    assert movie.parent == tmp_path
    assert movie.stat().st_size > 0
    ffmpeg = shutil.which(animation.FFMpegWriter.bin_path())
    decoded = subprocess.run(  # noqa: S603 - decode a movie created in this test's temporary directory.
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(movie),
            "-vf",
            "scale=1:1",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    # Each decoded frame is one grayscale pixel, so one byte represents one frame.
    assert len(decoded.stdout) == len(data)
    fig = plt.gcf()
    rendered = fig.axes[0].images[0].get_array()
    if trim:
        assert rendered.shape[0] < data.shape[1]
        assert rendered.shape[1] < data.shape[2]
        assert np.all(rendered > max(BAD_PIXEL_VALUES_SCALED))
    else:
        np.testing.assert_array_equal(rendered, data[-1])
    plt.close(fig)
