"""
Query the packaged IRIS spectral line database.
"""

from functools import cache

import numpy as np

import astropy.units as u
from astropy.table import QTable

from irispy.data import ROOTDIR

__all__ = ["get_lines"]

_REGIONS = ("quiet_sun", "active_region", "flare")
_ABUNDANCES = ("coronal", "photospheric")
_CATEGORIES = ("flare", "coronal", "transition_region", "chromospheric", "cool_metal")
_PASSBANDS = ("FUV1", "FUV2", "NUV")


@cache
def _load_lines():
    return QTable.read(ROOTDIR / "iris_lines.ecsv", format="ascii.ecsv")


def get_lines(
    wavelength_range=None, *, region=None, abundance=None, main_only=False, categories=None, include_unranked=True
):
    """
    Return lines in the IRIS passbands from the packaged line database.

    Parameters
    ----------
    wavelength_range : `~astropy.units.Quantity`, optional
        Two endpoints in any spectral unit, in either order; both are included.
        Wavelengths are in vacuum.
    region : `str`, optional
        Rank lines by their reference-model strength in ``'quiet_sun'``,
        ``'active_region'``, or ``'flare'``. Results are grouped as FUV1, FUV2,
        then NUV, with strongest first and unranked lines last within each
        passband. Requires ``abundance``. By default, lines are sorted by wavelength.
    abundance : `str`, optional
        ``'coronal'`` or ``'photospheric'`` abundances for ranking. Requires
        ``region``.
    main_only : `bool`, optional
        Return a curated selection of lines documented in the IRIS literature; their
        ``reference`` column names the citing paper in the documentation's
        bibliography.
    categories : `str` or iterable of `str`, optional
        Keep only ``'flare'``, ``'coronal'``, ``'transition_region'``,
        ``'chromospheric'``, or ``'cool_metal'`` lines. These are catalog labels,
        not measurements of the formation height in an observation.
    include_unranked : `bool`, optional
        If `False`, drop lines with no predicted strength in the ranking model,
        or in every model when ``region`` is not given.

    Returns
    -------
    `~astropy.table.QTable`
        A copy of the selected lines. The predicted strengths
        (``intensity_<region>_<abundance>``) are normalized to the strongest
        predicted line in each passband, so compare them only within one passband and
        model.

    Notes
    -----
    Strengths are normalized, integrated optically thin intensities for fixed
    reference atmospheres. Neutral and singly ionized lines have no prediction.
    Density, opacity, and ionization effects can invalidate the remaining
    predictions; see the line database documentation before comparing observations.
    """
    if region is not None and region not in _REGIONS:
        msg = f"region must be one of {_REGIONS}."
        raise ValueError(msg)
    if abundance is not None and abundance not in _ABUNDANCES:
        msg = f"abundance must be one of {_ABUNDANCES}."
        raise ValueError(msg)
    if (region is None) != (abundance is None):
        msg = "Supply region and abundance together to rank lines."
        raise ValueError(msg)

    table = _load_lines()
    selected = np.ones(len(table), dtype=bool)
    if wavelength_range is not None:
        endpoints = u.Quantity(wavelength_range)
        if endpoints.shape != (2,) or not np.all(np.isfinite(endpoints.value)) or np.any(endpoints.value <= 0):
            msg = "wavelength_range must contain two finite, positive spectral endpoints."
            raise ValueError(msg)
        # Widen by a few ulps so endpoints survive spectral-unit round trips.
        low, high = np.sort(endpoints.to_value(u.angstrom, equivalencies=u.spectral())) * (
            1 + 4 * np.finfo(float).eps * np.array([-1, 1])
        )
        wavelength = table["wavelength"].to_value(u.angstrom)
        selected &= (wavelength >= low) & (wavelength <= high)
    if main_only:
        selected &= table["main"]
    if categories is not None:
        categories = [categories] if isinstance(categories, str) else list(categories)
        if any(category not in _CATEGORIES for category in categories):
            msg = f"categories must be drawn from {_CATEGORIES}."
            raise ValueError(msg)
        selected &= np.isin(table["category"], categories)

    intensity_column = f"intensity_{region}_{abundance}" if region is not None else None
    if not include_unranked:
        columns = [intensity_column] if intensity_column else [c for c in table.colnames if c.startswith("intensity_")]
        selected &= np.any([np.isfinite(table[column]) for column in columns], axis=0)

    # The table is stored in wavelength order; the stable lexsort keeps that order for ties.
    # The copy keeps the cached table's metadata out of the caller's hands.
    result = table[selected].copy()
    if intensity_column:
        intensity = result[intensity_column]
        passband = np.array([_PASSBANDS.index(band) for band in result["passband"]])
        order = np.lexsort((result["wavelength"].value, np.where(np.isfinite(intensity), -intensity, np.inf), passband))
        return result[order]
    return result
