"""
============================
Co-align IRIS SJI to SDO/AIA
============================

In this example we will show how to co-align an IRIS dataset to SDO/AIA.

The IRIS instrument team at LMSAL provides AIA data cubes which are co-aligned to the IRIS FOV for
each observation via the `IRIS data search page <https://iris.lmsal.com/search/>`__.

Therefore this example is more of a showcase of functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch
from sunkit_image.coalignment import coalign_map

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import sunpy.map
from aiapy.calibrate import update_pointing
from aiapy.calibrate.utils import get_pointing_table
from sunpy.net import Fido
from sunpy.net import attrs as a

from irispy.io import read_sji_lvl2

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20250710_121126_3893010094_2025-07-10T12%3A11%3A262025-07-10T12%3A11%3A26.xml>`__.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2025/07/10/20250710_121126_3893010094/iris_l2_20250710_121126_3893010094_SJI_2832_t000_deconvolved.fits.gz",
    known_hash="0875acc65711969a93ce67474b6236bc98ce5a2bc49901ccad0f70ccf1478033",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

sji_2832 = read_sji_lvl2(sji_filename)

###############################################################################
# We will want to align the data to AIA.
# First we pick one frame of the observation and its time.

(time_sji,) = sji_2832.axis_world_coords("time")
sji_index = 8
sji_time = Time(time_sji[sji_index])
# We need a sunpy map, as the co-alignment works on sunpy maps only for now.
sji_map = sji_2832.to_maps(sji_index)

###############################################################################
# We will download the closest AIA 170 nm image from the Virtual Solar Observatory (VSO).
# Once we have acquired it, we will need to use `aiapy` to "prep" this image.

search_results = Fido.search(
    a.Time(sji_time, sji_time + TimeDelta(1 * u.minute), near=sji_time),
    a.Instrument.aia,
    a.Wavelength(1700 * u.AA),
)
files = Fido.fetch(search_results, site="NSO")
aia_map = sunpy.map.Map(files[0])
pointing_table = get_pointing_table(
    source="JSOC",
    time_range=(sji_time - TimeDelta(5 * 60 * u.minute), sji_time + TimeDelta(1 * u.minute)),
)
aia_map = update_pointing(aia_map, pointing_table=pointing_table)

# Crop the AIA FOV to the IRIS FOV plus a margin at least as large as the expected shift,
# otherwise you will contend with edge effects.
aia_crop = aia_map.submap(
    bottom_left=SkyCoord(
        sji_map.bottom_left_coord.Tx - 50 * u.arcsec,
        sji_map.bottom_left_coord.Ty - 50 * u.arcsec,
        frame="helioprojective",
        observer=sji_map.bottom_left_coord.observer,
    ),
    top_right=SkyCoord(
        sji_map.top_right_coord.Tx + 50 * u.arcsec,
        sji_map.top_right_coord.Ty + 50 * u.arcsec,
        frame="helioprojective",
        observer=sji_map.top_right_coord.observer,
    ),
)

###############################################################################
# The pointing information is not accurate to the pixel, so the IRIS and AIA images are
# slightly misaligned. We improve this by cross-correlating IRIS with AIA.
# ``sunkit-image`` works on sunpy maps, so we use the SJI map rather than the cube. Both
# images need the same plate scale, so we first resample AIA to the IRIS pixel size, then
# co-align with the ``match_template`` method; see
# `~sunkit_image.coalignment.match_template.match_template_coalign` for the details.

nx = (aia_crop.scale.axis1 * aia_crop.dimensions.x) / sji_map.scale.axis1.to(u.arcsec / u.pix)
ny = (aia_crop.scale.axis2 * aia_crop.dimensions.y) / sji_map.scale.axis2.to(u.arcsec / u.pix)
aia_upsampled = aia_crop.resample(u.Quantity([nx, ny]))

# We need to prepare the SJI data by removing NaNs.
sji_map_corrected_data = sji_map.data.copy()
nan_mask = ~np.isfinite(sji_map_corrected_data)
if np.any(nan_mask):
    sji_map_corrected_data[nan_mask] = 0
sji_map_corrected = sunpy.map.Map(sji_map_corrected_data, sji_map.meta)

coaligned_sji_map = coalign_map(sji_map_corrected, aia_upsampled, method="match_template")
# The co-alignment only updates the pointing in the metadata, so we plot the original
# data (with its NaNs, which show as blank) with the new metadata.
coaligned_sji_map = sunpy.map.Map(sji_map.data, coaligned_sji_map.meta)

###############################################################################
# Finally, we can plot the results of the co-alignment.

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection=sji_map.wcs)
sji_map.plot(axes=ax1)
aia_upsampled.draw_contours(axes=ax1, levels=[500] * u.DN, colors=["red"], linewidths=2)
ax1.set_title("IRIS SJI with AIA contours")

ax2 = fig.add_subplot(122, projection=coaligned_sji_map.wcs, sharex=ax1, sharey=ax1)
coaligned_sji_map.plot(axes=ax2)
aia_upsampled.draw_contours(axes=ax2, levels=[500] * u.DN, colors=["red"], linewidths=2)
ax2.set_title("Co-aligned IRIS SJI with AIA contours")
ax2.coords[1].set_ticks_visible(False)
ax2.coords[1].set_ticklabel_visible(False)

xlims_world = [-570, -490] * u.arcsec
ylims_world = [-210, -140] * u.arcsec
world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=coaligned_sji_map.coordinate_frame)
pixel_coords_x, pixel_coords_y = coaligned_sji_map.wcs.world_to_pixel(world_coords)
ax2.set_xlim(pixel_coords_x)
ax2.set_ylim(pixel_coords_y)

fig.tight_layout()

plt.show()
