"""
Constants of the IRIS instrument and its level 2 data.
"""

import astropy.units as u

__all__ = [
    "BAD_PIXEL_VALUE_SCALED",
    "DN_UNIT",
    "INSTRUMENTAL_FWHM",
    "RADIANCE_UNIT",
    "READOUT_NOISE",
    "SATURATION_LIMIT",
    "SLIT_WIDTH",
    "SPECTRAL_BAND",
]

#: The level 2 fill value of samples with no data, in DN; the readers also treat -199 DN, the floor that lower
#: samples are clipped to, as missing.
BAD_PIXEL_VALUE_SCALED = -200
# Both Level 2 fill values represent missing samples; -199 is an IRIS SolarSoft convention.
BAD_PIXEL_VALUES_SCALED = (BAD_PIXEL_VALUE_SCALED, -199)
# Camera gain (electrons per DN), yield (electrons per photon) and read noise (DN) by spectrograph detector and
# slit-jaw passband: De Pontieu et al. (2014) Sect. 7.1. The photons per DN, 4 (FUV) and 18 (NUV) for the spectrograph
# and 12 (1330, 1400) and 18 (2796, 2832) for the slit-jaw imager, are those of Wülser et al. (2018) Table 1 and of
# the response files (DN2PHOT_SG and DN2PHOT_SJI).
DETECTOR_GAIN = {"NUV": 18.0, "FUV": 6.0, "SJI_1330": 18.0, "SJI_1400": 18.0, "SJI_2796": 18.0, "SJI_2832": 18.0}
DETECTOR_YIELD = {"NUV": 1.0, "FUV": 1.5, "SJI_1330": 1.5, "SJI_1400": 1.5, "SJI_2796": 1.0, "SJI_2832": 1.0}
#: The DN unit of each spectrograph detector and slit-jaw passband, which converts to photons.
DN_UNIT = {
    band: u.def_unit(f"DN_IRIS_{band}", DETECTOR_GAIN[band] / DETECTOR_YIELD[band] * u.photon) for band in DETECTOR_GAIN
}
#: The readout noise of each spectrograph detector and slit-jaw passband, in its DN unit.
READOUT_NOISE = {band: (3.1 if band == "FUV" else 1.2) * DN_UNIT[band] for band in DETECTOR_GAIN}
# The level 2 clipping ceiling, not a detector saturation level. Level 2 data are int16 with BSCALE 0.25 and BZERO
# 7992, clipped to -199 to 16182 DN, the top code 32760 (irisl12_savesjidata.pro lines 545-552,
# irisl12_savespectraldata.pro lines 935-945). Saturated and merely bright samples both end here, and level 2 cannot
# tell them apart.
#
# Open questions for the IRIS team (SolarSoft as mirrored on 2026-10-07, level 2 writer L12-2019-08-08):
#
# - Is losing the saturation flags in the warping intended? iris_prep sets raw samples at or above sat_thresh to +Inf
#   (iris_prep.pro lines 594-600), but since 2015-06-01 ("Do not turn NaNs into Infs!", infterpolate.pro line 28,
#   inf_poly_2d.pro line 15) the warping marks Infs with NaN in a mask and restores them only where the interpolated
#   mask is infinite, which NaN never is (infterpolate.pro line 66, inf_poly_2d.pro line 51). They leave it near 2e4 DN
#   (iris_prep_geowave_correct.pro lines 110-123 and 142-161) and are clipped to 16182 DN like any bright sample; the
#   writer's saturation code 32764, 16183 DN, is never written (irisl12_savesjidata.pro line 552, irisl12_changelog.rtf
#   lines 581-582 and 620).
# - Are the level 2 NSATPIX and TSATPXn known to be empty? They count the +Inf samples the writer receives
#   (irisl12_savesjidata.pro lines 628 and 687, irisl12_savespectraldata.pro lines 1030-1031, 1091 and 1130), so they
#   are 0 in every level 2 file checked.
# - Why is iris_prep's sat_thresh 16000 DN (iris_prep.pro lines 67 and 595), below the 14-bit camera's 16383 DN?
#   Saturated level 1 frames top out at 16358 DN, and their NSATPIX counts samples from about 15063 DN.
#
# Files: https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/lmsal/calibration/ (iris_prep.pro, infterpolate.pro,
# inf_poly_2d.pro, iris_prep_geowave_correct.pro) and https://sohoftp.nascom.nasa.gov/solarsoft/iris/idl/uio/level1to2/
# (irisl12_savesjidata.pro, irisl12_savespectraldata.pro, irisl12_changelog.rtf).
#: The level 2 ceiling, to which every sample above it, saturated or merely bright, is clipped.
SATURATION_LIMIT = 16182 * u.DN
#: The unit of radiometrically calibrated spectra, and of slit-jaw images calibrated without ``line``.
RADIANCE_UNIT = u.erg / u.cm**2 / u.s / u.steradian / u.Angstrom
RADIANCE_UNIT_PER_HZ = u.erg / u.cm**2 / u.s / u.steradian / u.Hz
# De Pontieu et al. (2014) Table 1.
#: The width of the spectrograph slit.
SLIT_WIDTH = 0.33 * u.arcsec
# Spectral resolution (FWHM) of the spectrograph by passband: De Pontieu et al. (2014) Table 1.
# Their Sect. 7.3.3 measures 25.85 mA (FUV1) and 50.54 mA (NUV) on solar lines, upper bounds that include the
# solar width; the 31.8 mA often used for FUV2 is a pre-launch laboratory value (Tian et al. 2014, Text S5).
#: The spectral resolution (FWHM) of each spectrograph passband.
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
#: The detector, ``"FUV"`` or ``"NUV"``, of each spectral window name.
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
