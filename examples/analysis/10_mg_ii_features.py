"""
===================================
Map the Mg II h and k line features
===================================

The Mg II h and k lines each have two emission peaks, such as k2v and k2r, on either side of
a central reversal, such as k3. They form at different heights in the chromosphere, so their
positions and intensities trace its velocities and how these change with height.

In this example, we will measure them and attempt to reproduce Figure 3 of
:cite:t:`long2024`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from scipy.stats import gaussian_kde

import astropy.units as u

from irispy.io import read_files
from irispy.spectrograph import SpectrogramCube
from irispy.utils.mg_features import calculate_mg_features

###############################################################################
# We will start by getting some data from the IRIS archive.
#
# Using the url: https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20200402_224709_3610108077_2020-04-02T22%3A47%3A092020-04-02T22%3A47%3A09.xml
#
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2020/04/02/20200402_224709_3610108077/iris_l2_20200402_224709_3610108077_raster.tar.gz>`__.
# To keep the download small, we use a cutout.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20200402_224709_3610108077_cutout_2796_raster.fits.gz",
    known_hash="e1c0f4cfaa82a782046b22684b8c657013751abcf57252eeab1f6ced8bd26cb5",
)
mg_ii = read_files(raster_filename)["Mg II k 2796"][0]

###############################################################################
# `~irispy.utils.mg_features.calculate_mg_features` finds the velocity and
# intensity of the blue peak, line centre and red peak of both lines, within
# 40 km/s of their rest wavelengths. Each is a map with the spatial WCS of the raster.

features = calculate_mg_features(mg_ii)
features.keys()

###############################################################################
# The figure shows the separation of the k2 peaks, the k3 velocity, the
# difference of the k3 and h3 velocities and the asymmetry of the k2 peaks.

velocity = {name: features[f"{name}_velocity"].data for name in ("k2v", "k3", "k2r", "h3")}
k2v, k2r = features["k2v_intensity"].data, features["k2r_intensity"].data
maps = {
    r"$\Delta v_{k2}$ (km/s)": (velocity["k2r"] - velocity["k2v"], "cubehelix_r", 0, 50),
    r"$v_{k3}$ (km/s)": (velocity["k3"], "coolwarm", -20, 20),
    "k-h separation (km/s)": (velocity["k3"] - velocity["h3"], "coolwarm", -2, 2),
    "k asymmetry": ((k2v - k2r) / (k2v + k2r), "coolwarm", -0.5, 0.5),
}

###############################################################################
# The paper compares three regions: the leading and following polarities of the active region
# and an area of emerging flux. We try to bound the same regions as in the paper.

coordinates = features["k3_velocity"].axis_world_coords()[0]
solar_x, solar_y = coordinates.Tx.to_value(u.arcsec), coordinates.Ty.to_value(u.arcsec)
regions = {"Leading polarity": (-344, 526), "Following polarity": (-423, 548), "Emerging flux": (-415, 520)}
inside = {name: (np.abs(solar_x - x) <= 5) & (np.abs(solar_y - y) <= 5) for name, (x, y) in regions.items()}

###############################################################################
# The top row shows the maps with the regions outlined, and the bottom row the
# distribution of each quantity within each region. As in the paper, the k2
# peaks are furthest apart in the emerging flux and closest in the leading
# polarity, and the k3 velocities, k-h separations and asymmetries
# are mostly small and positive.

wcs = features["k3_velocity"].wcs
fig = plt.figure(figsize=(14, 7.5), layout="constrained")
for column, (label, (values, cmap, vmin, vmax)) in enumerate(maps.items()):
    ax = fig.add_subplot(2, 4, column + 1, projection=wcs)
    SpectrogramCube(values, wcs).plot(axes=ax, plot_axes=["x", "y"], cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(ax.images[0], ax=ax, location="top", label=label)
    # The first world axis is latitude, along the slit (y), and the second is longitude, along the raster (x).
    for coord, side, name in ((ax.coords[0], "l", "Solar Y" if column == 0 else " "), (ax.coords[1], "b", "Solar X")):
        coord.set_ticklabel(exclude_overlapping=True, fontsize=8, color="black")
        coord.set_ticks_position(side)
        coord.set_ticklabel_position(side)
        coord.set_axislabel(name, color="black")
        coord.set_axislabel_position(side)
    ax_kde = fig.add_subplot(2, 4, column + 5)
    grid = np.linspace(vmin, vmax, 200)
    for color, (name, region) in zip(("C0", "C1", "C2"), inside.items(), strict=True):
        # The maps are plotted with the raster steps along x, so the masks are transposed to match.
        ax.contour(region.T, levels=[0.5], colors=color, linewidths=2)
        sample = values[region]
        density = gaussian_kde(sample[np.isfinite(sample)])(grid)
        ax_kde.fill_between(grid, density, color=color, alpha=0.5, label=name)
    ax_kde.set_xlim(vmin, vmax)
    ax_kde.set_ylim(bottom=0)
    ax_kde.set_xlabel(label)
    ax_kde.set_ylabel("Density" if column == 0 else None)
fig.legend(*ax_kde.get_legend_handles_labels(), loc="outside lower center", ncols=3)
fig.suptitle(f"IRIS Mg II features, {mg_ii.meta.date_start.isot[:19]}")

plt.show()
