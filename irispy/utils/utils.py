"""
This module provides general utility functions.
"""

from importlib import import_module

import dask.array as da
import numpy as np

import astropy.units as u

__all__ = ["calculate_uncertainty"]


def _import_optional(module_name: str, *, reason: str, extra: str):
    """
    Import an optional dependency, raising a helpful error when it is missing.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        msg = (
            f"{module_name} is an optional dependency required for {reason}. "
            f"Install it with `pip install {module_name}` or "
            f"`pip install 'irispy-lmsal[{extra}]'`."
        )
        raise ImportError(msg) from exc


def _boolean_mask(cube):
    """
    The mask of ``cube`` as a read-only boolean array of its data's shape, `False`
    everywhere without one.

    A mask may be a scalar, cover only the trailing axes or be an integer array.
    """
    mask = False if cube.mask is None else cube.mask
    # A Dask mask stays one, for in_chunks
    mask = mask.astype(bool) if isinstance(mask, da.Array) else np.asarray(mask, dtype=bool)
    return np.broadcast_to(mask, cube.data.shape)


def _saturated_samples(cube):
    """
    `True` for the saturated samples of ``cube``: those its mask and
    ``cube.meta["saturated"]`` mark.

    The readers mask the samples at the level 2 ceiling,
    ``irispy.utils.constants.SATURATION_LIMIT``, keeping their value, and mark them in
    ``meta["saturated"]``, which follows the cube through slicing, exposure time
    correction, calibration and `~irispy.utils.spectrograph.subtract_background`, but
    not rebinning.
    """
    saturated = cube.meta.get("saturated")
    # Rebinning leaves the mark at the old shape
    if saturated is None or cube.mask is None or np.shape(saturated) != cube.data.shape:
        # Not an array of the size of the cube
        return np.broadcast_to(np.False_, cube.data.shape)
    mask = _boolean_mask(cube)
    # Lazy with the mask only: the marks of a cube read into memory stay a Dask array in its metadata
    return mask & (saturated.astype(bool) if isinstance(mask, da.Array) else np.asarray(saturated, dtype=bool))


@u.quantity_input
def calculate_uncertainty(data: np.ndarray, readout_noise: u.photon, unit: u.UnitBase) -> np.ndarray:
    """
    Calculates the uncertainty of a given data array.

    Parameters
    ----------
    data : numpy.ndarray
        The data array.
    readout_noise : astropy.units.Quantity
        The readout noise, needs to be a unit that is convertible to photon.
    unit : astropy.units.Unit
        The unit of ``data``, and of the result.

    Returns
    -------
    numpy.ndarray
        The standard deviation of each value (photon and readout noise), in ``unit``
        but with the unit removed.
    """
    # Negative DN (dark-subtracted noise or the -200 fill) have no photon noise, only readout noise
    photons = np.clip((data * unit).to(u.photon).value, 0, None)
    # A Python float keeps the data's dtype; a numpy scalar would promote float32 data to float64
    readout_photons = float(readout_noise.to_value(u.photon))
    return u.Quantity(np.sqrt(photons + readout_photons**2), unit=u.photon).to(unit).value
