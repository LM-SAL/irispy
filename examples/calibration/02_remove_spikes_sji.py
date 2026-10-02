"""
=====================================
Remove Cosmic Rays from IRIS SJI data
=====================================

This example illustrates how to remove cosmic ray hits from IRIS SJI data.

We will use the ``astroscrappy`` backend, which has to be installed separately with ``pip`` or ``conda``
and is the better choice for imaging data. See the
`astroscrappy documentation <https://astroscrappy.readthedocs.io/en/latest/>`__ for how it works and its parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooch

from astropy.visualization import quantity_support

from irispy.io import read_files

quantity_support()

###############################################################################
# `We start with getting data from the IRIS data archive <https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20260206_210853_3460104433_2026-02-06T21%3A08%3A532026-02-06T21%3A08%3A53.xml>`__.
#
# This dataset was taken during a South Atlantic Anomaly (SAA) passage, so it has many
# cosmic ray hits: a worst case, good for testing the algorithm but not ideal for science.
#
# In this case, we will use ``pooch`` to keep this example self-contained
# but you can download the data manually using your browser as well.
#
# You will need to update the path to the data in the next section if you do that.

sji_filename = pooch.retrieve(
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/02/06/20260206_210853_3460104433/iris_l2_20260206_210853_3460104433_SJI_2832_t000.fits.gz",
    known_hash="d5088c6a0753ea9ce7b525865ba2edf13a637097ee995985b511833897c88ca6",
)

###############################################################################
# We will now open the data using a helper function which is designed to read
# all files from a single observation.

sji_2832 = read_files(sji_filename)
sji_frame = sji_2832[5]
del sji_2832

###############################################################################
# Now we use `~irispy.SJICube.remove_cosmic_rays` with the ``astroscrappy`` backend, a
# general-purpose algorithm widely used for imaging data; it is not the default. As with
# ``rsliding``, its defaults are not tuned for IRIS, so read its documentation and experiment.
# On spectra, it does not remove the small negative dips that often flank positive spikes,
# nor purely negative spikes.

# These settings were found to work well for SJI data (thanks to Juraj).
method_kwargs = {"sigclip": 3, "objlim": 5, "readnoise": 3.1, "satlevel": np.inf, "cleantype": "medmask"}
sji_astroscrappy = sji_frame.remove_cosmic_rays(method="astroscrappy", method_kwargs=method_kwargs)

###############################################################################
# Be cautious: an aggressive setting also removes real features.

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6), subplot_kw={"projection": sji_frame.wcs}, sharex=True, sharey=True, layout="constrained"
)

sji_frame.plot(axes=axes[0], aspect="auto", vmin=0, vmax=500, origin="lower")
axes[0].set_title("Original")
sji_astroscrappy.plot(axes=axes[1], aspect="auto", vmin=0, vmax=500, origin="lower")
axes[1].set_title("astroscrappy")

plt.show()
