import dask.array as da
import numpy as np
import pytest

from irispy.io._mask import _Mapped
from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2


@pytest.mark.parametrize("reader", ["sji", "raster", "sot"])
def test_memmap_mask_reads_only_the_requested_frame(reader, request, monkeypatch):
    filename = request.getfixturevalue(
        {"sji": "bursts_sji_1400_file", "raster": "sns_sg_file", "sot": "sot_fg_file"}[reader]
    )
    reads = []
    getitem = _Mapped.__getitem__

    def record_read(self, item):
        reads.append(item)
        return getitem(self, item)

    monkeypatch.setattr(_Mapped, "__getitem__", record_read)

    def read(memmap):
        if reader != "raster":
            return read_sji_lvl2(filename, memmap=memmap)
        return read_spectrograph_lvl2(filename, spectral_windows="C II 1336", memmap=memmap)["C II 1336"]

    cube = read(memmap=True)
    # The first frame, of the first raster
    first = np.s_[0, :1] if reader == "raster" else np.s_[:1]
    assert isinstance(cube.mask, da.Array)
    assert cube.mask.dtype == bool
    assert cube.mask.shape == cube.data.shape
    assert reads == []

    np.testing.assert_array_equal(cube.mask[first].compute(), read(memmap=False).mask[first])
    assert len(reads) == 1
    assert reads[0][-3] == slice(0, 1)

    # Every frame at one position is one read, not one for each frame
    reads.clear()
    cube.mask[..., 0, 0].compute()
    assert len(reads) == 1
