import dask.array as da
import numpy as np
import pytest

from astropy.io import fits

from irispy.io._mask import _MappedArray
from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2


@pytest.mark.parametrize("reader", ["sji", "raster"])
def test_memmap_mask_reads_only_the_requested_frame(reader, request, monkeypatch):
    filename = request.getfixturevalue("bursts_sji_1400_file" if reader == "sji" else "sns_sg_file")
    reads = []
    getitem = _MappedArray.__getitem__

    def record_read(self, item):
        reads.append(item)
        return getitem(self, item)

    monkeypatch.setattr(_MappedArray, "__getitem__", record_read)
    cube = (
        read_sji_lvl2(filename, memmap=True)
        if reader == "sji"
        else read_spectrograph_lvl2(filename, spectral_windows="C II 1336", memmap=True)["C II 1336"][0]
    )
    assert isinstance(cube.mask, da.Array)
    assert cube.mask.dtype == bool
    assert cube.mask.shape == cube.data.shape
    assert cube.mask.chunks == ((1,) * cube.shape[0], *[(size,) for size in cube.shape[1:]])
    assert reads == []

    with fits.open(filename, do_not_scale_image_data=True) as hdulist:
        header = hdulist[0 if reader == "sji" else 1].header
    fill = (np.array([-200, -199]) - header.get("BZERO", 0)) / header.get("BSCALE", 1)
    np.testing.assert_array_equal(cube.mask[:1].compute(), np.isin(cube.data[:1], fill))
    assert len(reads) == 1
    assert reads[0][0] == slice(0, 1)
