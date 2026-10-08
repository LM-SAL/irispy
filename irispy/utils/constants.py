"""
This module provides constants used elsewhere.
"""

import astropy.units as u

__all__ = [
    "ATOMIC_MASS",
    "BAD_PIXEL_VALUES_SCALED",
    "BAD_PIXEL_VALUE_SCALED",
    "BAD_PIXEL_VALUE_UNSCALED",
    "DN_UNIT",
    "INSTRUMENTAL_FWHM",
    "PASSBAND_LIMITS",
    "RADIANCE_UNIT",
    "RADIANCE_UNIT_PER_HZ",
    "READOUT_NOISE",
    "SATURATION_LIMIT",
    "SLIT_WIDTH",
    "SPECTRAL_BAND",
]

# The following value is only appropriate for byte scaled images
BAD_PIXEL_VALUE_SCALED = -200
# Both Level 2 fill values represent missing samples; -199 is an IRIS SolarSoft convention.
BAD_PIXEL_VALUES_SCALED = (BAD_PIXEL_VALUE_SCALED, -199)
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
# Level 2 data are int16 with BSCALE 0.25 and BZERO 7992. iris_prep sets saturated samples to +Inf, which int16
# cannot hold, so they and any brighter sample are clipped to the top code 32760: 0.25 * 32760 + 7992 = 16182 DN.
SATURATION_LIMIT = 16182 * u.DN
RADIANCE_UNIT = u.erg / u.cm**2 / u.s / u.steradian / u.Angstrom
RADIANCE_UNIT_PER_HZ = u.erg / u.cm**2 / u.s / u.steradian / u.Hz
SLIT_WIDTH = 0.33 * u.arcsec
# Spectral resolution (FWHM) of the spectrograph by passband: De Pontieu et al. (2014) Table 1.
# Their Sect. 7.3.3 measures 25.85 mA (FUV1) and 50.54 mA (NUV) on solar lines, upper bounds that include the
# solar width; the 31.8 mA often used for FUV2 is a pre-launch laboratory value (Tian et al. 2014, Text S5).
INSTRUMENTAL_FWHM = {"FUV1": 26 * u.mAA, "FUV2": 26 * u.mAA, "NUV": 53 * u.mAA}
# Vacuum wavelength limits of the spectrograph passbands: De Pontieu et al. (2014) Table 2.
PASSBAND_LIMITS = {"FUV1": [1331.7, 1358.4] * u.AA, "FUV2": [1389.0, 1407.0] * u.AA, "NUV": [2782.7, 2835.1] * u.AA}
# Abridged standard atomic weights of the elements of the IRIS lines: Prohaska et al. (2022), via ciaaw.org.
ATOMIC_MASS = {
    "H": 1.0080 * u.u,
    "He": 4.0026 * u.u,
    "C": 12.011 * u.u,
    "N": 14.007 * u.u,
    "O": 15.999 * u.u,
    "Mg": 24.305 * u.u,
    "Si": 28.085 * u.u,
    "S": 32.06 * u.u,
    "Cl": 35.45 * u.u,
    "Ca": 40.078 * u.u,
    "Mn": 54.938 * u.u,
    "Fe": 55.845 * u.u,
    "Ni": 58.693 * u.u,
}
SPECTRAL_BAND = {
    "1330": "FUV",
    "1336": "FUV",
    "1343": "FUV",
    "1349": "FUV",
    "1352": "FUV",
    "1356": "FUV",
    "1394": "FUV",
    "1400": "FUV",
    "1403": "FUV",
    "2786": "NUV",
    "2796": "NUV",
    "2814": "NUV",
    "2826": "NUV",
    "2830": "NUV",
    "2831": "NUV",
    "2832": "NUV",
    "C II 1336": "FUV",
    "Cl I 1352": "FUV",
    "Fe XII 1349": "FUV",
    "Mg II k 2796": "NUV",
    "O I 1356": "FUV",
    "Si IV 1394": "FUV",
    "Si IV 1403": "FUV",
}
