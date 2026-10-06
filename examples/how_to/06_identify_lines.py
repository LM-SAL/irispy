"""
==============================
Identify lines in IRIS spectra
==============================

Find candidate transitions in an IRIS flare raster using the spectral line database.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from matplotlib.lines import Line2D

import astropy.units as u

from irispy.io import read_files
from irispy.utils.lines import get_lines

###############################################################################
# This `raster of active region 12268 <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20150130_055150_3893010094_2015-01-30T05%3A51%3A502015-01-30T05%3A51%3A50.xml>`__
# was taken during the decay of an M1.7 flare, with 15 s exposures covering FUV1,
# FUV2, and NUV. The 16-step cutout covers a flare ribbon and hot post-flare loops.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20150130_055150_3893010094_cutout_raster.fits.gz",
    known_hash="603aa2a5dbe0cf9738e3628451dd24361da05b8eeecd42962ac1253db05888eb",
)
raster = read_files(raster_filename)

###############################################################################
# `~irispy.utils.lines.get_lines` ranks candidates in the C II window using the
# flare reference atmosphere and coronal abundances. C II itself is unranked
# because its optically thick formation lies outside the model's scope.

c_ii = raster["C II 1336"][0]
(wavelength,) = c_ii.axis_world_coords("wl")
lines = get_lines(wavelength[[0, -1]], region="flare", abundance="coronal")
lines["intensity_flare_coronal"].info.format = ".3f"
print(lines["ion", "wavelength", "wavelength_source", "intensity_flare_coronal"][:5])

###############################################################################
# The 99th percentile at each wavelength highlights the ribbon and loops while
# reducing sensitivity to isolated cosmic-ray spikes. Different wavelengths can
# select different pixels, so the composite cannot validate model intensity ratios.
#
# Both Level 2 data and the catalog use vacuum wavelengths. Each panel marks the
# curated lines and candidates at least 1 % of the passband's strongest prediction.
# Scores normalize only eligible predictions; a high-scoring line can still be
# much fainter than an unranked Mg II line.

fig, axes = plt.subplots(3, 1, figsize=(10, 11), layout="constrained")
for ax, window in zip(axes, raster.keys(), strict=True):
    cube = raster[window][0]
    (wavelength,) = cube.axis_world_coords("wl")
    data = np.where(cube.mask | ~np.isfinite(cube.data), np.nan, cube.data)
    valid = np.isfinite(data).any(axis=(0, 1))
    spectrum = np.full(len(wavelength), np.nan)
    spectrum[valid] = np.nanpercentile(data[..., valid], 99, axis=(0, 1))
    ax.plot(wavelength.to_value(u.AA), spectrum, color="black", linewidth=0.8)
    lines = get_lines(wavelength[[0, -1]])
    lines = lines[(lines["intensity_flare_coronal"] >= 0.01) | lines["main"]]
    for i, line in enumerate(lines):
        position = line["wavelength"].to_value(u.AA)
        color = "tab:blue" if line["main"] else "0.5"
        ax.axvline(position, color=color, linestyle="--", linewidth=0.8)
        y, va = (0.98, "top") if i % 2 else (0.02, "bottom")
        ax.text(
            position,
            y,
            line["ion"],
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va=va,
            fontsize=8,
            color=color,
        )
    ax.set(title=window, yscale="log", ylabel=f"Intensity [{cube.unit}]", ymargin=0.25)
axes[-1].set_xlabel("Vacuum wavelength [Å]")
fig.legend(
    handles=[
        Line2D([], [], color="tab:blue", linestyle="--", label="Curated IRIS line"),
        Line2D([], [], color="0.5", linestyle="--", label="Reference-model candidate"),
    ],
    loc="outside upper center",
    ncols=2,
)

plt.show()

###############################################################################
# Fe XXI 1354.08 Å is the broad line from the hot loops. C II, O I, Cl I, C I, and
# Mg II are documented lines without predictions.
# C II and Mg II need radiative transfer; O I needs recombination and charge exchange.
#
# Strengths assume a fixed DEM and pressure; ``region="flare"`` does not fit this
# raster's atmosphere. Si IV can become optically thick in flares, affecting its
# doublet ratio.
#
# Search around the two unlabelled peaks near 1357 Å, including unranked candidates.
# The feature at 1386.7 Å falls below the catalog's FUV2 limit of 1389 Å.

for peak in [1357.14, 1357.66]:
    candidates = get_lines([peak - 0.02, peak + 0.02] * u.AA)
    print(f"{peak} Å:", ", ".join(f"{line['ion']} {line['wavelength'].value:.3f}" for line in candidates))

###############################################################################
# `Peter Young's IRIS line list <https://pyoung.org/iris/iris_line_list.pdf>`__, Table 1,
# identifies C I lines at 1357.134 and 1357.659 Å, consistent with these features.
# Laboratory intensities depend on the excitation source and have no common
# scale across elements and ionization stages.
#
# Most NUV features away from Mg II are photospheric absorption lines, but cool
# metals can also emit in the chromosphere during flares. The ``cool_metal`` label
# does not assign a formation height; identification requires the spectrum and a solar atlas.

cool_metals = get_lines([2812, 2818] * u.AA, categories="cool_metal")
print(f"{len(cool_metals)} cool-metal candidates between 2812 and 2818 Å")
