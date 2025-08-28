"""
This module provides constants used elsewhere.
"""

import astropy.units as u

__all__ = [
    "BAD_PIXEL_VALUE_SCALED",
    "BAD_PIXEL_VALUE_UNSCALED",
    "DN_UNIT",
    "RADIANCE_UNIT",
    "READOUT_NOISE",
    "SLIT_WIDTH",
    "SPECTRAL_BAND",
]

# The following value is only appropriate for byte scaled images
BAD_PIXEL_VALUE_SCALED = -200
# The following value is only appropriate for unscaled images
BAD_PIXEL_VALUE_UNSCALED = -32768
# Define some properties of IRIS detectors.
# Source: IRIS instrument paper (https://link.springer.com/article/10.1007/s11207-014-0485-y)
DETECTOR_GAIN = {"NUV": 18.0, "FUV": 6.0, "SJI": 18.0}
DETECTOR_YIELD = {"NUV": 1.0, "FUV": 1.5, "SJI": 1.0}
DN_UNIT = {
    "NUV": u.def_unit("DN_IRIS_NUV", DETECTOR_GAIN["NUV"] / DETECTOR_YIELD["NUV"] * u.photon),
    "FUV": u.def_unit("DN_IRIS_FUV", DETECTOR_GAIN["FUV"] / DETECTOR_YIELD["FUV"] * u.photon),
    "SJI": u.def_unit("DN_IRIS_SJI", DETECTOR_GAIN["SJI"] / DETECTOR_YIELD["SJI"] * u.photon),
    "SJI_UNSCALED": u.def_unit("DN_IRIS_SJI_UNSCALED", u.ct),
}
READOUT_NOISE = {
    "NUV": 1.2 * DN_UNIT["NUV"],
    "FUV": 3.1 * DN_UNIT["FUV"],
    "SJI": 1.2 * DN_UNIT["SJI"],
}
RADIANCE_UNIT = u.erg / u.cm**2 / u.s / u.steradian / u.Angstrom
SLIT_WIDTH = 0.33 * u.arcsec
SPECTRAL_BAND = {
    "1343": "FUV",
    "1400": "FUV",
    "2786": "NUV",
    "2814": "NUV",
    "2826": "NUV",
    "2830": "NUV",
    "2832": "NUV",
    "C II 1336": "FUV",
    "Cl I 1352": "FUV",
    "Fe XII 1349": "FUV",
    "Mg II k 2796": "NUV",
    "O I 1356": "FUV",
    "Si IV 1394": "FUV",
    "Si IV 1403": "FUV",
}
