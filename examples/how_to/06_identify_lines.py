"""
==============================
Identify lines in IRIS spectra
==============================

In this example, we will use the irispy line database to identify lines in an IRIS observation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

import astropy.units as u

from irispy.io import read_files
from irispy.utils.lines import get_lines

###############################################################################
# `We start with a raster of active region 12268 <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20150130_055150_3893010094_2015-01-30T05%3A51%3A502015-01-30T05%3A51%3A50.xml>`__,
# taken during the decay of an M1.7 flare. Its three spectral windows cover the full
# FUV1, FUV2 and NUV passbands with 15 s exposures. To keep the download small, we
# use a cutout of 16 raster steps that covers a flare ribbon and the hot post-flare loops.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20150130_055150_3893010094_cutout_raster.fits.gz",
    known_hash="603aa2a5dbe0cf9738e3628451dd24361da05b8eeecd42962ac1253db05888eb",
)
raster = read_files(raster_filename)
print(raster.keys())

###############################################################################
# `~irispy.utils.lines.get_lines` returns the lines within a wavelength range.
# Given a ``region`` and an ``abundance``, it sorts them by their predicted strength.
# These are the five strongest lines predicted for a flare in the C II window.

c_ii = raster["C II 1336"][0]
(wavelength,) = c_ii.axis_world_coords("wl")
lines = get_lines(wavelength[[0, -1]], region="flare", abundance="coronal")
lines["intensity_flare_coronal"].info.format = ".3f"
print(lines["ion", "wavelength", "wavelength_source", "intensity_flare_coronal"][:5])

###############################################################################
# To compare the predictions with the data, we take the 99th percentile of each
# window over the cutout at every wavelength. This picks out the bright ribbon
# and loops and ignores cosmic-ray spikes, which hit far fewer than 1% of the pixels.
#
# IRIS Level 2 data and the database both use vacuum wavelengths, so we mark the lines
# predicted above 1 % of each passband's strongest line, and the documented IRIS lines,
# in each window directly.

fig, axes = plt.subplots(3, 1, figsize=(10, 11), layout="constrained")
for ax, window in zip(axes, raster.keys(), strict=True):
    cube = raster[window][0]
    (wavelength,) = cube.axis_world_coords("wl")
    spectrum = np.nanpercentile(np.where(cube.mask, np.nan, cube.data), 99, axis=(0, 1))
    ax.plot(wavelength.to_value(u.AA), spectrum, color="black", linewidth=0.8)
    lines = get_lines(wavelength[[0, -1]])
    lines = lines[(lines["intensity_flare_coronal"] >= 0.01) | lines["main"]]
    # Labels alternate between the top and bottom edges so that neighbouring lines do not overlap.
    for i, line in enumerate(lines):
        position = line["wavelength"].to_value(u.AA)
        ax.axvline(position, color="tab:blue", linestyle="--", linewidth=0.8)
        y, va = (0.98, "top") if i % 2 else (0.02, "bottom")
        ax.text(
            position, y, line["ion"], transform=ax.get_xaxis_transform(), rotation=90, ha="right", va=va, fontsize=8
        )
    ax.set(title=window, yscale="log", ylabel=f"Intensity [{cube.unit}]", ymargin=0.25)
axes[-1].set_xlabel("Vacuum wavelength [Å]")

plt.show()

###############################################################################
# The predictions identify the bright emission lines, including the broad Fe XXI
# 1354.08 Å line from the hot loops. The flare model predicts O I to be weak, because
# its temperature distribution starts above where O I forms, so O I, Cl I and C I are
# labelled here only as documented lines: they have no predictions because CHIANTI
# has no Cl I data and its C I model has no transitions in this range.
#
# The predictions also fail in the other direction. The only NUV prediction away from
# Mg II is Al II 2817 Å, at 6 % of Mg II k in the flare model, and nothing is there:
# an optically thin estimate for a singly ionized ion, set by the coolest DEM bin,
# cannot be compared with the optically thick Mg II lines.
#
# Two bright lines near 1357 Å are still unlabelled. Lines without a prediction are in
# the database too, so a narrow range around each peak lists the candidates. The
# feature at 1386.7 Å is below the documented FUV2 passband, which starts at 1389 Å,
# so the database does not cover it.

for peak in [1357.14, 1357.66]:
    candidates = get_lines([peak - 0.02, peak + 0.02] * u.AA)
    print(f"{peak} Å:", ", ".join(f"{line['ion']} {line['wavelength'].value:.3f}" for line in candidates))

###############################################################################
# Apart from C I, the candidates are laboratory lines of heavy elements such as
# tungsten and iridium, or iron-group lines with laboratory intensities a thousand
# times weaker than those of C I. So these peaks are C I 1357.13 and 1357.66 Å, two
# more lines of the C I multiplet that the documented 1354.28 and 1355.84 Å lines belong to.
#
# Most of the NUV features away from Mg II are photospheric absorption lines. The
# database lists their candidates in the ``photospheric`` category, but without
# predicted strengths, and there are many in every Angstrom; identifying them needs
# a solar atlas.

photospheric = get_lines([2812, 2818] * u.AA, categories="photospheric")
print(f"{len(photospheric)} photospheric candidates between 2812 and 2818 Å")
