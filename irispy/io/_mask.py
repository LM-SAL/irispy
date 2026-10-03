"""
Fill masks for raw, memory-mapped Level 2 data.
"""

import numpy as np

from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED


class _MappedArray:
    def __init__(self, data):
        self.data = data
        self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim

    def __getitem__(self, item):
        return self.data[item]


def _fill_mask_values(data, raw_m200, raw_m199):
    return (data == raw_m200) | (data == raw_m199)


def _memmap_fill_mask(data, header):
    # Dask copies NumPy arrays at graph construction. Hide the mapped array behind
    # a slice-only wrapper, and skip tokenizing its contents or probing a slice.
    import dask.array as da  # NOQA: PLC0415

    mapped = da.from_array(
        _MappedArray(data),
        chunks=(1, *data.shape[1:]),
        name=False,
        meta=np.empty((0,) * data.ndim, dtype=data.dtype),
    )
    raw_fill = (np.asarray(BAD_PIXEL_VALUES_SCALED) - header.get("BZERO", 0)) / header.get("BSCALE", 1)
    # Integral raw codes compare directly as integers, without converting the data.
    raw_m200, raw_m199 = [int(value) if value.is_integer() else value for value in raw_fill]
    return mapped.map_blocks(
        _fill_mask_values, raw_m200, raw_m199, dtype=bool, meta=np.empty((0,) * data.ndim, dtype=bool)
    )
