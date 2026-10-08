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
# The level 2 clipping ceiling, not a detector saturation level. Level 2 data are int16 with BSCALE 0.25 and BZERO
# 7992, and the level 2 writer clips every sample to -199 to 16182 DN, the top code 32760: 0.25 * 32760 + 7992 = 16182
# (irisl12_savesjidata.pro lines 545-552, irisl12_savespectraldata.pro lines 935-945). Samples at it include every
# sample that iris_prep flagged as saturated, and any calibrated sample above the ceiling.
#
# Open questions for the IRIS team (SolarSoft as mirrored on 2026-10-07, level 2 writer L12-2019-08-08):
#
# - Is losing the saturation flags in the warping intended? iris_prep sets raw level 1 samples at or above sat_thresh
#   to +Inf (iris_prep.pro lines 594-600). Since 2015-06-01 ("Do not turn NaNs into Infs!", infterpolate.pro line 28,
#   inf_poly_2d.pro line 15), the warping marks Infs with NaN in a mask and restores them only where the interpolated
#   mask is infinite, which NaN never is (infterpolate.pro line 66, inf_poly_2d.pro line 51). The flagged samples
#   leave it near 2e4 DN (iris_prep_geowave_correct.pro lines 110-123 and 142-161), so the writer clips them to
#   16182 DN like any bright sample instead of writing its saturation code 32764, 16183 DN (irisl12_savesjidata.pro
#   line 552, irisl12_changelog.rtf lines 581-582 and 620).
# - Are the level 2 NSATPIX and TSATPXn known to be empty? They count the +Inf samples the writer receives
#   (irisl12_savesjidata.pro lines 628 and 687, irisl12_savespectraldata.pro lines 1030-1031, 1091 and 1130), so
#   they are 0 whenever the flags are lost, as in every level 2 file checked.
# - Why is iris_prep's threshold 16000 DN (sat_thresh = 1.6e4, iris_prep.pro lines 67 and 595), below the 16383 DN
#   of the 14-bit camera?
#
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/iris_prep.pro
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/infterpolate.pro
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/inf_poly_2d.pro
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/iris_prep_geowave_correct.pro
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/uio/level1to2/irisl12_savesjidata.pro
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/uio/level1to2/irisl12_savespectraldata.pro
# https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/uio/level1to2/irisl12_changelog.rtf
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
