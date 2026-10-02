"""
=============================
Build an AIA synthetic raster
=============================

In this example we build an AIA 1700 Å synthetic raster, sampling the AIA
image closest in time to each raster step along the IRIS slit. To correct the IRIS
pointing, we align the SJI 2832 Å images to AIA and apply the same correction to
the spectrograph slit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from scipy.ndimage import map_coordinates
from sunkit_image.coalignment import coalign_map

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsymmetricPercentileInterval

import sunpy.map

from irispy.io import read_files

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20170305_164021_3620106076_2017-03-05T16%3A40%3A212017-03-05T16%3A40%3A21.xml>`__.
# The full observation is available as a `Level 2 raster tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2017/03/05/20170305_164021_3620106076/iris_l2_20170305_164021_3620106076_raster.tar.gz>`__
# and an `SDO tarball <https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2017/03/05/20170305_164021_3620106076/iris_l2_20170305_164021_3620106076_SDO.tar.gz>`__.
#
# To keep the download small, we use cutouts of the 2832 Å raster window and the
# SJI 2832 Å and AIA 1700 Å cubes, covering raster steps 96 to 223 (counting from zero).
#
# The IRIS team provides these AIA cubes on the IRIS field of view, so unlike in
# :ref:`sphx_glr_generated_gallery_coalign_01_coalign_iris_aia.py`, they do not
# need an AIA pointing update.

raster_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20170305_164021_3620106076_cutout_2832_raster.fits.gz",
    known_hash="945d4a1178ecc024925558f7bd04a3e3ab2dc3eea4133464088b4bbc6672c3cc",
)
sji_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20170305_164021_3620106076_cutout_SJI_2832.fits.gz",
    known_hash="4802b42f2389a14cc1692a9c92f49d12fe0e588ed2be44d018cf387a2292b5e2",
)
aia_filename = pooch.retrieve(
    "https://github.com/LM-SAL/irispy-data/releases/download/v1/aia_l2_20170305_164021_3620106076_cutout_1700.fits.gz",
    known_hash="f55401575c6967653e2cab60f79be399d5ee5386023e2316b5fc3d3ae6cc71e8",
)

raster = read_files(raster_filename, spectral_windows="2832")["2832"][0]
sji_cube = read_files(sji_filename)
aia_cube = read_files(aia_filename)

###############################################################################
# We take the raster at the wavelength closest to the reference wavelength of the
# window, transpose it so the slit runs vertically, and set missing pixels to NaN.
#
# All three datasets give exposure midpoints, which we use to find the SJI and AIA
# exposures closest to each raster step.

wavelength_index = np.abs(raster.spectral_axis - raster.meta.rest_wavelength).argmin()
continuum = raster[:, :, wavelength_index]
iris_image = np.where(continuum.mask, np.nan, continuum.data).T

(sji_times,) = sji_cube.axis_world_coords("time")
(aia_times,) = aia_cube.axis_world_coords("time")
sji_indices = [np.abs(sji_times - time).argmin() for time in raster.time]
aia_indices = [np.abs(aia_times - time).argmin() for time in raster.time]
print(f"Largest SJI time difference: {np.max(np.abs(sji_times[sji_indices] - raster.time)).to_value(u.s):.1f} s")
print(f"Largest AIA time difference: {np.max(np.abs(aia_times[aia_indices] - raster.time)).to_value(u.s):.1f} s")

###############################################################################
# To align an SJI image to AIA, we crop it to its valid region and fill any
# remaining missing pixels with the median, as template matching needs finite
# values. We reproject AIA onto the cropped SJI grid, padded by 20 arcsec so the
# search has room to move. The pointing correction is how far ``coalign_map``
# moves the reference coordinate.


def align_sji(sji_map, aia_map):
    """
    Return the (Solar-X, Solar-Y) pointing correction that aligns ``sji_map`` to ``aia_map``.
    """
    valid = np.isfinite(sji_map.data)
    rows = np.flatnonzero(valid.any(axis=1))
    columns = np.flatnonzero(valid.any(axis=0))
    template = sji_map.submap(
        [columns[0] + 2, rows[0] + 2] * u.pix,
        top_right=[columns[-1] - 2, rows[-1] - 2] * u.pix,
    )
    template = sunpy.map.Map(np.nan_to_num(template.data, nan=np.nanmedian(template.data)), template.meta)
    padding = np.ceil((20 * u.arcsec / u.Quantity(template.scale)).to_value(u.pix)).astype(int)
    reference_header = template.meta.copy()
    reference_header["crpix1"] += padding[0]
    reference_header["crpix2"] += padding[1]
    reference_header["naxis1"] += 2 * padding[0]
    reference_header["naxis2"] += 2 * padding[1]
    aligned = coalign_map(template, aia_map.reproject_to(reference_header), method="match_template")
    return u.Quantity(
        [
            aligned.reference_coordinate.Tx - template.reference_coordinate.Tx,
            aligned.reference_coordinate.Ty - template.reference_coordinate.Ty,
        ]
    )


###############################################################################
# Now we build the synthetic raster. At each step, we take the coordinates of
# every slit pixel from the raster WCS, which includes the per-step pointing, add
# the pointing correction, and sample AIA there with bilinear interpolation.
# The raster WCS has a single obstime, so we give the slit coordinates the frame of
# the matching SJI exposure. Each SJI/AIA pair is only aligned once.

slit_pixels = np.arange(raster.shape[1])
aia_image = np.full(iris_image.shape, np.nan)
shifts = np.empty((raster.shape[0], 2))
alignments = {}

for step, (sji_index, aia_index) in enumerate(zip(sji_indices, aia_indices, strict=True)):
    if (sji_index, aia_index) not in alignments:
        sji_map = sji_cube.to_maps(sji_index)
        sji_map = sunpy.map.Map(np.where(sji_cube.mask[sji_index], np.nan, sji_map.data), sji_map.meta)
        aia_map = aia_cube.to_maps(aia_index)
        aia_map = sunpy.map.Map(np.where(aia_cube.mask[aia_index], np.nan, aia_map.data), aia_map.meta)
        alignments[sji_index, aia_index] = (sji_map.coordinate_frame, aia_map, align_sji(sji_map, aia_map))
    sji_frame, aia_map, shift = alignments[sji_index, aia_index]
    shifts[step] = shift.to_value(u.arcsec)

    _, slit = raster.wcs.pixel_to_world(0, slit_pixels, step)
    slit = SkyCoord(slit.Tx + shift[0], slit.Ty + shift[1], frame=sji_frame)
    aia_x, aia_y = aia_map.world_to_pixel(slit)
    aia_image[:, step] = map_coordinates(aia_map.data, [aia_y.value, aia_x.value], order=1, cval=np.nan)

print(f"Aligned {len(alignments)} exposure pairs for {raster.shape[0]} raster steps.")

###############################################################################
# We can now compare the two rasters, each normalized for display. Note that each
# column was observed at a different time.

normalize = AsymmetricPercentileInterval(1, 99.5)
frames = [normalize(aia_image), normalize(iris_image)]
titles = ["AIA 1700 Å synthetic raster", "IRIS 2832 Å continuum raster"]
fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharex=True, sharey=True, layout="constrained")
for ax, image, title in zip(axes, frames, titles, strict=True):
    ax.imshow(image, origin="lower", interpolation="none", cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Raster step in cutout")
axes[0].set_ylabel("Slit pixel")

###############################################################################
# The pointing correction only changes when the SJI/AIA pair does; a large jump
# suggests a bad match.

fig, ax = plt.subplots(layout="constrained")
ax.plot(shifts[:, 0], label="Solar-X")
ax.plot(shifts[:, 1], label="Solar-Y")
ax.set(xlabel="Raster step in cutout", ylabel="Pointing correction (arcsec)", title="SJI pointing corrections")
ax.legend()

###############################################################################
# We can also draw the bright AIA 1700 Å features, at half of their display range,
# as contours on the IRIS raster. They should outline its brighter patches.

fig, ax = plt.subplots(figsize=(4, 7), layout="constrained")
ax.imshow(frames[1], origin="lower", interpolation="none", cmap="gray", vmin=0, vmax=1)
ax.contour(frames[0], levels=[0.5], colors="red", linewidths=1)
ax.set(xlabel="Raster step in cutout", ylabel="Slit pixel", title="IRIS 2832 Å with AIA 1700 Å contours")

###############################################################################
# Finally, we look for a residual offset by correlating the two rasters at integer
# shifts, ignoring missing pixels. A positive shift moves the IRIS raster right or
# up. The two passbands form at different heights, so the correlation stays well
# below 1, and a peak near zero only shows that little residual translation is left.


def cross_correlate(reference, image, max_shift=10):
    """
    Return the Pearson correlation between ``reference`` and ``image`` for integer shifts of
    ``image``.
    """
    ny, nx = reference.shape
    lags = np.arange(-max_shift, max_shift + 1)
    correlation = np.empty((lags.size, lags.size))
    for j, dy in enumerate(lags):
        for i, dx in enumerate(lags):
            ref = reference[max(dy, 0) : ny + min(dy, 0), max(dx, 0) : nx + min(dx, 0)]
            moved = image[max(-dy, 0) : ny + min(-dy, 0), max(-dx, 0) : nx + min(-dx, 0)]
            valid = np.isfinite(ref) & np.isfinite(moved)
            correlation[j, i] = np.corrcoef(ref[valid], moved[valid])[0, 1]
    return lags, correlation


lags, correlation = cross_correlate(aia_image, iris_image)
peak_y, peak_x = np.unravel_index(correlation.argmax(), correlation.shape)
print(f"Peak r = {correlation[peak_y, peak_x]:.3f} at dx = {lags[peak_x]} steps, dy = {lags[peak_y]} slit pixels")

fig, ax = plt.subplots(layout="constrained")
im = ax.imshow(correlation, origin="lower", extent=[lags[0] - 0.5, lags[-1] + 0.5] * 2)
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.plot(lags[peak_x], lags[peak_y], "r+", markersize=12)
ax.set(xlabel="IRIS x shift (raster steps)", ylabel="IRIS y shift (slit pixels)")

plt.show()

# sphinx_gallery_thumbnail_number = 1
