"""
The DN, the fill mask and the saturated samples of raw, memory-mapped Level 2 data, each
made from the values of the file when a slice of it is used.
"""

import dask.array as da
import numpy as np

from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, BAD_PIXEL_VALUES_SCALED, SATURATION_LIMIT

# The raw bytes of the exposures in one chunk of a lazy array, at most. Of 1 to 256 MiB on five observations, 16 MiB
# is the largest that adds no memory to a read of a whole window: every Dask worker holds a chunk at once. Smaller
# chunks make more tasks and open a file of several more often; larger ones slow cube[k][i], which reads whole chunks.
_CHUNK_BYTES = 16 * 2**20


class _Mapped:
    # The raw data of a file seen through ``function``: ``function(values, *args)`` of each slice of it that is read
    def __init__(self, data, function, args, dtype):
        self.data, self.function, self.args = data, function, args
        self.shape, self.dtype, self.ndim = data.shape, np.dtype(dtype), data.ndim

    def __getitem__(self, item):
        return self.function(self.data[item], *self.args)


def _when_read(data, function, *args, dtype):
    """
    ``function(values, *args)`` of the raw ``data``, the mapped array of a file or the
    ``_WindowFiles`` of several, as a Dask array that reads the values when a slice of
    it is used.
    """
    # A chunk for each exposure makes a task for each, 1.5 s for the 9495 exposures of a sit-and-stare, and a slice
    # reads the whole chunks it touches
    frames = max(1, min(data.shape[-3], _CHUNK_BYTES // (int(np.prod(data.shape[-2:])) * data.dtype.itemsize)))
    # Dask copies NumPy arrays at graph construction. Hide the data behind a slice-only wrapper, which applies the
    # function in the layer that reads them, and skip tokenizing their contents or probing a slice.
    return da.from_array(
        _Mapped(data, function, args, dtype),
        # The axes before the last three are those of the rasters, one file each
        chunks=(*(1,) * (data.ndim - 3), frames, *data.shape[-2:]),
        name=False,
        meta=np.empty((0,) * data.ndim, dtype=dtype),
    )


def _raw(header, value):
    # The raw value of the file that scales to ``value``, an integer when it is one, to compare integers as they are
    raw = (value - header.get("BZERO", 0)) / header.get("BSCALE", 1)
    return int(raw) if float(raw).is_integer() else raw


def _fill_mask_values(data, raw_m200, raw_m199, raw_ceiling=None):
    mask = (data == raw_m200) | (data == raw_m199)
    if data.dtype.kind == "f":
        mask |= np.isnan(data)
    return mask if raw_ceiling is None else mask | (data >= raw_ceiling)


def _memmap_fill_mask(data, header, *, nan_only=False, saturated=False):
    """
    The lazy mask of the raw ``data``: the fill values, or only the NaN of a Hinode/SOT
    cube, and with ``saturated`` the samples at the level 2 ceiling.
    """
    if nan_only:
        return _when_read(data, np.isnan, dtype=bool)
    fills = [_raw(header, value) for value in BAD_PIXEL_VALUES_SCALED]
    ceiling = [_raw(header, SATURATION_LIMIT.value)] if saturated else []
    return _when_read(data, _fill_mask_values, *fills, *ceiling, dtype=bool)


def _memmap_saturated(data, header):
    """
    The samples of the raw ``data`` at the level 2 ceiling, lazily.
    """
    return _when_read(data, np.greater_equal, _raw(header, SATURATION_LIMIT.value), dtype=bool)


def _dn(raw, bscale, bzero, fills):
    # As astropy scales the 16-bit integers of an IRIS file: their DN, in steps of 0.25 DN, are exact in 32-bit floats
    if bscale == 1 and bzero == 0:
        data = np.array(raw)
    else:
        data = np.array(raw, dtype=np.float32)
        np.multiply(data, bscale, out=data)
        data += bzero
    if fills:
        # The fill values of a slit-jaw cube are NaN, or -200 where the data are integers
        data[_fill_mask_values(raw, *fills)] = np.nan if data.dtype.kind == "f" else BAD_PIXEL_VALUE_SCALED
    return data


def _memmap_dn(data, header, *, replace_fills=False):
    """
    The raw ``data`` in DN, lazily, with the values of a ``memmap=False`` read.

    ``BSCALE`` and ``BZERO`` are applied as astropy applies them, and with
    ``replace_fills`` the fill values of a slit-jaw cube are replaced.
    """
    bscale, bzero = header.get("BSCALE", 1), header.get("BZERO", 0)
    fills = tuple(_raw(header, value) for value in BAD_PIXEL_VALUES_SCALED) if replace_fills else ()
    dtype = data.dtype if bscale == 1 and bzero == 0 else np.float32
    return _when_read(data, _dn, bscale, bzero, fills, dtype=dtype)


def _check_scaled(header):
    # astropy drops BSCALE and BZERO once it scales the data, but keeps an identity scaling, as in the AIA and SOT cubes
    if header.get("BSCALE", 1) != 1 or header.get("BZERO", 0) != 0:
        msg = "An HDU list opened with do_not_scale_image_data=True needs memmap=True"
        raise ValueError(msg)
