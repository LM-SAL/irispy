"""
Fill masks and the scaling check for raw, memory-mapped Level 2 data.
"""

import dask.array as da
import numpy as np

from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED


class _MappedMask:
    # The mask of a mapped array: ``mask(values, *args)`` of each slice of it that is read
    def __init__(self, data, mask, *args):
        self.data, self.mask, self.args = data, mask, args
        self.shape, self.dtype, self.ndim = data.shape, np.dtype(bool), data.ndim

    def __getitem__(self, item):
        return self.mask(self.data[item], *self.args)


def _fill_mask_values(data, raw_m200, raw_m199):
    mask = (data == raw_m200) | (data == raw_m199)
    return mask | np.isnan(data) if data.dtype.kind == "f" else mask


def _memmap_fill_mask(data, header, *, nan_only=False):
    if nan_only:
        mask, args = np.isnan, ()
    else:
        raw_fill = (np.asarray(BAD_PIXEL_VALUES_SCALED) - header.get("BZERO", 0)) / header.get("BSCALE", 1)
        # Integral raw codes compare directly as integers, without converting the data.
        raw_m200, raw_m199 = [int(value) if value.is_integer() else value for value in raw_fill]
        mask, args = _fill_mask_values, (raw_m200, raw_m199)
    meta = np.empty((0,) * data.ndim, dtype=bool)
    # A multi-file memmap raster is already a Dask array that reads its file on demand.
    if isinstance(data, da.Array):
        return data.map_blocks(mask, *args, dtype=bool, meta=meta)
    # Dask copies NumPy arrays at graph construction. Hide the mapped array behind a slice-only
    # wrapper, which makes the mask in the same layer, and skip tokenizing its contents or probing a slice.
    # Chunks of up to 16 MiB of exposures. A chunk for each exposure makes a task for each, 1.5 s for the 9495
    # exposures of a sit-and-stare, and a slice after an integer index, as in cube[0][i], reads whole chunks.
    frames = max(1, min(data.shape[-3], 16 * 2**20 // (int(np.prod(data.shape[-2:])) * data.dtype.itemsize)))
    # The axes before the last three are those of the rasters
    chunks = (*(1,) * (data.ndim - 3), frames, *data.shape[-2:])
    return da.from_array(_MappedMask(data, mask, *args), chunks=chunks, name=False, meta=meta)


def _is_scaled(header):
    # astropy drops BSCALE and BZERO once it scales the data, but keeps an identity scaling, as in the AIA and SOT cubes
    return header.get("BSCALE", 1) == 1 and header.get("BZERO", 0) == 0


def _check_scaled(header):
    if not _is_scaled(header):
        msg = "An HDU list opened with do_not_scale_image_data=True needs memmap=True"
        raise ValueError(msg)
