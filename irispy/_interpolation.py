import numpy as np


def _pad_axis0(value, target_length):
    """
    Pad axis 0 to ``target_length`` by repeating the last entry.

    Works on anything supporting numpy fancy indexing along axis 0, including
    `~numpy.ndarray`, `~astropy.units.Quantity`, `~astropy.time.Time` and
    `~astropy.coordinates.SkyCoord`.
    """
    if value.shape[0] == target_length:
        return value
    return value[np.minimum(np.arange(target_length), value.shape[0] - 1)]


def _interpolate_bad_axis0_rows(values, bad_rows):
    """
    Replace bad axis-0 rows by linear interpolation from good rows.
    """
    bad_rows = np.asarray(bad_rows, dtype=bool)
    if not bad_rows.any():
        return values

    good_indices = np.nonzero(~bad_rows)[0]
    if good_indices.size == 0:
        msg = "Cannot interpolate when every row is bad."
        raise ValueError(msg)

    bad_indices = np.nonzero(bad_rows)[0]
    unit = getattr(values, "unit", None)
    raw_values = values.to_value(unit) if unit is not None else np.asarray(values)
    flat_values = raw_values.reshape(raw_values.shape[0], -1)
    for column in range(flat_values.shape[1]):
        flat_values[bad_indices, column] = np.interp(
            bad_indices,
            good_indices,
            flat_values[good_indices, column],
        )
    if unit is not None:
        values[bad_indices] = raw_values[bad_indices] * unit
    return values
