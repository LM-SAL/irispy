.. _irispy-tutorial-calibration:

********************************
Calibration of IRIS Observations
********************************

Wavelength Calibration
======================

The wavelength calibration is automatically performed to the best of current knowledge.
This is accurate to only a few pixels, and should be manually checked.
There are several photospheric spectral lines that can be used for accurate wavelength calibration, most notably the Ni I 279.9474 nm line in the NUV and the O I 135.560 nm line in the FUV.

.. note::

   The IRIS automatic wavelength calibration is based on these lines averaged over the *entire slit*.
   The slit may cover regions of significant line-of-sight flows, such as flux emergence in active regions, and thus these photospheric lines may not necessarily be at their nominal rest wavelengths as assumed by the automatic calibration.
   In such cases, the user is advised, again, to perform manual wavelength calibration by avoiding such regions under the slit coverage.

A detailed discussion of the wavelength calibration steps for IRIS and how to use them on data can be found in `IRIS Technical Note 20 <https://www.lmsal.com/iris_science/doc?cmd=dcur&proj_num=IS0203&file_type=pdf>`__.

Radiometric Calibration
=======================

The IRIS data are given in Data Number units (DN).
To convert these to a flux in physical units (e.g., erg s\ :sup:`-1` sr\ :sup:`-1` cm\ :sup:`-2` Å\ :sup:`-1`) one must perform radiometric calibration.
The latest calibration data are included in ``irispy`` and can be read:

.. code-block:: python

   >>> from sunpy.time import parse_time

   >>> from irispy.utils.response import get_interpolated_effective_area, get_latest_response

   >>> response = get_latest_response(observation_time=parse_time("2020-01-01T00:00:00"))

where ``observation_time`` is an astropy Time object with the time of the observations (compatible with `sunpy.time.parse_time`).
The output is a dictionary with the following keys:

.. code-block:: python

   >>> response.keys()
   dict_keys(['DATE_OBS', 'LAMBDA', 'AREA_SG', 'NAME_SG', 'DN2PHOT_SG', 'AREA_SJI', 'NAME_SJI', 'DN2PHOT_SJI', 'COEFFS_FUV', 'C_F_TIME', 'C_F_LAMBDA', 'COEFFS_NUV', 'C_N_TIME', 'C_N_LAMBDA', 'COEFFS_SJI', 'C_S_TIME', 'GEOM_AREA', 'ELEMENTS', 'INDEX_EL_SG', 'INDEX_EL_SJI', 'COMMENT', 'VERSION', 'VERSION_DATE'])

Here ``AREA_SG`` and ``AREA_SJI`` are the effective areas (in cm\ :sup:`-2`) as a function of wavelength (``LAMBDA``) respectively for the spectrograph and slit-jaw camera.
The ``DN2PHOT_*`` tags give the conversion from DN counts to photons.

.. warning::

   ``get_latest_response`` will only apply the most up to date calibration.
   It is not possible to specify a particular version of the calibration data, it is only possible to specify the time of the observation, and the routine will return the appropriate calibration for that time.

To convert the spectral units from DN to flux one must do the following conversion:

.. math::

   \mathrm{Flux}(\mathrm{erg}\: \mathrm{s}^{-1}\: \mathrm{cm}^{-2} \text{Å}^{-1}\: \mathrm{sr}^{-1}) = \mathrm{Flux}(\mathrm{DN}) \frac{E_\lambda \cdot \mathrm{DN2PHOT\_SG}}{A_\mathrm{eff} \cdot \mathrm{Pix}_{xy} \cdot \mathrm{Pix}_{\lambda} \cdot t_\mathrm{exp} \cdot W_\mathrm{slit}},

where :math:`E_\lambda \equiv h \cdot c / \lambda` is the photon energy (in erg), :math:`\mathrm{DN2PHOT\_SG}` is the number of photons per DN (get from ``iris_get_response``), :math:`A_\mathrm{eff}` is the effective area (in cm\ :sup:`-2`), :math:`\mathrm{Pix}_{xy}` is the size of the spatial pixels in radians (e.g. multiply the spatial binning factor by :math:`\pi/(180\cdot 3600 \cdot 6)`), :math:`\mathrm{Pix}_{\lambda}` is the size of the spectral pixels in Å, :math:`t_\mathrm{exp}` is the exposure time in seconds and :math:`W_\mathrm{slit}` is the slit width in radians (:math:`W_\mathrm{slit}\equiv \pi/(180\cdot 3600 \cdot 3)`).

A detailed discussion of the radiometric calibration steps for IRIS and how to use them on data can be found in `IRIS Technical Note 24 <https://www.lmsal.com/iris_science/doc?cmd=dcur&proj_num=IS0123&file_type=pdf>`__.

.. note::

   The exposure time :math:`t_\mathrm{exp}` can be different for each exposure in the same sequence, when Automatic Exposure Control (AEC) is switched on.
   This is the default for most active region observations, although different exposures are only used in extreme cases when very bright phenomena can lead to CCD saturation (e.g., a flare).
   The level 2 FITS headers have only one value for the exposure time (the value without AEC).
   The sequence-dependent exposure times are available in the auxiliary metadata in the FITS files (see :ref:`irispy-tutorial-lev2`), with table index given by ``EXPTIMEF``, ``EXPTIMEN``, and ``EXPTIME`` for FUV, NUV, and slit-jaw, respectively.

``irispy`` has a routine to perform the conversion from DN to physical units and is described in this example :ref:`sphx_glr_generated_gallery_basic_radiometric_calibration.py`.
