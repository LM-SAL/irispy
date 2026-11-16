.. _irispy-tutorial-calibration:

********************************
Calibration of IRIS Observations
********************************

Wavelength Calibration
======================

The wavelength calibration is automatically performed to the best of current knowledge.
This is accurate to only a few pixels, and should be manually checked.
There are several photospheric spectral lines that can be used for accurate wavelength calibration, most notably the Ni I 279.9474 nm line in the NUV and the O I 135.560 nm line in the FUV.
These and all Level 2 wavelengths are in vacuum.

.. note::

   The IRIS automatic wavelength calibration is based on these lines averaged over the *entire slit*.
   The slit may cover regions of significant line-of-sight flows, such as flux emergence in active regions, and thus these photospheric lines may not necessarily be at their nominal rest wavelengths as assumed by the automatic calibration.
   In such cases, the user is advised, again, to perform manual wavelength calibration by avoiding such regions under the slit coverage.

A detailed discussion of the wavelength calibration steps for IRIS and how to use them on data can be found in IRIS Technical Note 20 :cite:p:`itn20`.
`~irispy.utils.wavelength_drift.calculate_wavelength_drift` measures how the NUV and FUV wavelength scales drift during an observation.
The :ref:`Mg II Dopplergram example <sphx_glr_generated_gallery_analysis_02_mg_ii_dopplergrams.py>` measures this drift and corrects a raster with it.

Radiometric Calibration
=======================

The IRIS data are given in Data Number units (DN).
To convert these to a flux in physical units (e.g., erg s\ :sup:`-1` sr\ :sup:`-1` cm\ :sup:`-2` Å\ :sup:`-1`) one must perform radiometric calibration.
The latest calibration data are included in ``irispy`` and can be read:

.. code-block:: python

   >>> from irispy.utils.response import get_latest_response

   >>> response = get_latest_response(observation_time="2020-01-01T00:00:00")

where ``observation_time`` is one observation time, in any format `sunpy.time.parse_time` accepts.
The output is a dictionary with the following keys:

.. code-block:: python

   >>> response.keys()
   dict_keys(['DATE_OBS', 'LAMBDA', 'AREA_SG', 'NAME_SG', 'DN2PHOT_SG', 'AREA_SJI', 'NAME_SJI', 'DN2PHOT_SJI', 'COEFFS_FUV', 'C_F_TIME', 'C_F_LAMBDA', 'COEFFS_NUV', 'C_N_TIME', 'C_N_LAMBDA', 'COEFFS_SJI', 'C_S_TIME', 'GEOM_AREA', 'ELEMENTS', 'INDEX_EL_SG', 'INDEX_EL_SJI', 'COMMENT', 'VERSION', 'VERSION_DATE'])

Here ``AREA_SG`` and ``AREA_SJI`` are the effective areas (in cm\ :sup:`2`) as a function of wavelength (``LAMBDA``) respectively for the spectrograph and slit-jaw camera.
The ``DN2PHOT_*`` tags give the conversion from DN counts to photons.

.. warning::

   ``get_latest_response`` will only apply the most up to date calibration.
   It is not possible to specify a particular version of the calibration data, it is only possible to specify the time of the observation, and the routine will return the appropriate calibration for that time.

To convert the spectral units from DN to flux one must do the following conversion:

.. math::

   \mathrm{Flux}(\mathrm{erg}\: \mathrm{s}^{-1}\: \mathrm{cm}^{-2} \text{Å}^{-1}\: \mathrm{sr}^{-1}) = \mathrm{Flux}(\mathrm{DN}) \frac{E_\lambda \cdot \mathrm{DN2PHOT\_SG}}{A_\mathrm{eff} \cdot \mathrm{Pix}_{xy} \cdot \mathrm{Pix}_{\lambda} \cdot t_\mathrm{exp} \cdot W_\mathrm{slit}},

where :math:`E_\lambda \equiv h \cdot c / \lambda` is the photon energy (in erg), :math:`\mathrm{DN2PHOT\_SG}` is the number of photons per DN (``response["DN2PHOT_SG"]`` from `~irispy.utils.response.get_latest_response`), :math:`A_\mathrm{eff}` is the effective area (in cm\ :sup:`2`), :math:`\mathrm{Pix}_{xy}` is the size of the spatial pixels in radians (e.g. multiply the spatial binning factor by :math:`\pi/(180\cdot 3600 \cdot 6)`), :math:`\mathrm{Pix}_{\lambda}` is the size of the spectral pixels in Å, :math:`t_\mathrm{exp}` is the exposure time in seconds and :math:`W_\mathrm{slit}` is the slit width in radians (:math:`W_\mathrm{slit}\equiv \pi/(180\cdot 3600 \cdot 3)`).

A detailed discussion of the radiometric calibration steps for IRIS and how to use them on data can be found in IRIS Technical Note 24 :cite:p:`itn24`.

.. note::

   The exposure time :math:`t_\mathrm{exp}` can be different for each exposure in the same sequence, when Automatic Exposure Control (AEC) is switched on.
   This is the default for most active region observations, although different exposures are only used in extreme cases when very bright phenomena can lead to CCD saturation (e.g., a flare).
   The level 2 FITS headers have only one value for the exposure time (the value without AEC).
   The sequence-dependent exposure times are available in the auxiliary metadata in the FITS files (see :ref:`irispy-tutorial-lev2`), with table index given by ``EXPTIMEF``, ``EXPTIMEN``, and ``EXPTIME`` for FUV, NUV, and slit-jaw, respectively.

The routine that converts DN to physical units is described in this example :ref:`sphx_glr_generated_gallery_calibration_03_radiometric_calibration.py`; IRIS Technical Note 26 :cite:p:`itn26` describes the calibration process it follows.

Other units
-----------

`~irispy.utils.spectrograph.radiometric_calibration` gives radiance per unit wavelength.
``cube.to`` converts the data and the uncertainty to another unit.
A conversion to radiance per unit frequency, or in photons, depends on the wavelength, so it also needs `astropy.units.spectral_density` of the cube's wavelengths, as in :ref:`sphx_glr_generated_gallery_calibration_03_radiometric_calibration.py`:

.. code-block:: python

   >>> import numpy as np
   >>> import astropy.units as u

   >>> import irispy.data.sample as sample_data  # doctest: +REMOTE_DATA
   >>> from irispy.io import read_files
   >>> from irispy.utils.spectrograph import radiometric_calibration

   >>> raster = read_files(sample_data.RASTER_FITS, spectral_windows="Mg II k 2796", uncertainty=True)  # doctest: +REMOTE_DATA
   >>> cube = raster["Mg II k 2796"][0]  # doctest: +REMOTE_DATA
   >>> calibrated = radiometric_calibration(cube)  # doctest: +REMOTE_DATA
   >>> wavelength = calibrated.axis_world_coords("em.wl")[0]  # doctest: +REMOTE_DATA
   >>> per_hz = calibrated.to(u.erg / (u.s * u.sr * u.cm**2 * u.Hz), equivalencies=u.spectral_density(wavelength))  # doctest: +REMOTE_DATA
   >>> per_hz.unit, per_hz.uncertainty.unit  # doctest: +REMOTE_DATA
   (Unit("erg / (Hz s sr cm2)"), Unit("erg / (Hz s sr cm2)"))

From radiance to DN
-------------------

To predict the counts of a spectrum, such as one from a model atmosphere, as IDL's ``iris_erg2dn`` does, divide its radiance by `~irispy.utils.spectrograph.calculate_dn_to_radiance_factor`, which gives photons per second, and convert those to DN per second with ``.to(cube.unit / u.s)``, as the DN units are defined in photons (18 per DN in the NUV, 4 in the FUV).
Multiplied by the exposure times, the DN per second of the calibrated cube give back its DN, and are NaN where the response has no effective area:

.. code-block:: python

   >>> from irispy.utils.response import get_latest_response
   >>> from irispy.utils.spectrograph import calculate_dn_to_radiance_factor

   >>> factor = calculate_dn_to_radiance_factor(
   ...     iris_response=get_latest_response(cube.meta.date_reference),
   ...     wavelength=wavelength,
   ...     detector_type=cube.meta.detector_band,
   ...     spectral_dispersion_per_pixel=cube.spectral_dispersion,
   ...     solid_angle=cube.solid_angle,
   ... )  # doctest: +REMOTE_DATA
   >>> dn_per_second = (calibrated.data * calibrated.unit / factor).to(cube.unit / u.s)  # doctest: +REMOTE_DATA
   >>> dn = dn_per_second * cube.exposure_time[:, np.newaxis, np.newaxis]  # doctest: +REMOTE_DATA
   >>> dn.unit, bool(np.nanmax(np.abs(dn.value - cube.data)) < 1e-3)  # doctest: +REMOTE_DATA
   (Unit("DN_IRIS_NUV"), True)

Slit-jaw images
---------------

`~irispy.utils.spectrograph.radiometric_calibration` also calibrates a `~irispy.sji.SJICube`.
A slit-jaw pixel counts the photons of a whole passband, so you choose what spectrum to assume within it:

* ``line=wavelength`` assumes that all counts come from one line at that wavelength and gives its intensity in erg s\ :sup:`-1` sr\ :sup:`-1` cm\ :sup:`-2`,

  .. math::

     I = \mathrm{DN} \frac{\mathrm{DN2PHOT\_SJI} \cdot h c / \lambda}{A_\mathrm{eff}(\lambda) \cdot \Omega \cdot t_\mathrm{exp}}.

* ``line=None``, the default, assumes a continuum flat across the passband and gives erg s\ :sup:`-1` sr\ :sup:`-1` cm\ :sup:`-2` Å\ :sup:`-1`,

  .. math::

     I_\lambda = \mathrm{DN} \frac{\mathrm{DN2PHOT\_SJI} \cdot h c}{\Omega \cdot t_\mathrm{exp} \int A_\mathrm{eff}(\lambda) \, \lambda \, d\lambda}.

Here :math:`A_\mathrm{eff}` is the passband's effective area (``response["AREA_SJI"]``) at the observation date, :math:`\Omega` is the solid angle of a pixel, ``CDELT1`` times ``CDELT2``, and :math:`t_\mathrm{exp}` is the exposure time of each frame.

.. code-block:: python

   >>> import astropy.units as u

   >>> import irispy.data.sample as sample_data  # doctest: +REMOTE_DATA
   >>> from irispy.io import read_files
   >>> from irispy.utils.lines import get_lines
   >>> from irispy.utils.spectrograph import radiometric_calibration

   >>> sji = read_files(sample_data.SJI_2796)  # doctest: +REMOTE_DATA
   >>> mg_ii_k = get_lines([2796, 2797] * u.AA, main_only=True)["wavelength"][0]
   >>> radiometric_calibration(sji, line=mg_ii_k).unit  # doctest: +REMOTE_DATA
   Unit("erg / (s sr cm2)")

Neither assumption holds exactly; the 2796 passband, for one, also takes in the wings of Mg II k.
The FUV passbands overlap, so a single-line intensity from a FUV image is approximate: in late 2021 the 1330 passband responds at 33% and 48% of its peak at Si IV 1394 and 1403 Å.
`~irispy.utils.response.get_latest_response` gives the response of each passband at any date.

Radiation Temperature
=====================

`irispy.utils.spectrograph.radiation_temperature` converts a radiometrically calibrated cube to radiation temperature by inverting the Planck function.
Its docstring gives the formula and the caveats.
A cube still in DN raises an error, so calibrate it first:

.. code-block:: python

   >>> import numpy as np

   >>> import irispy.data.sample as sample_data  # doctest: +REMOTE_DATA
   >>> from irispy.io import read_files
   >>> from irispy.utils.spectrograph import radiation_temperature, radiometric_calibration

   >>> raster = read_files(sample_data.RASTER_FITS, spectral_windows="Mg II k 2796")  # doctest: +REMOTE_DATA
   >>> calibrated = radiometric_calibration(raster["Mg II k 2796"][0])  # doctest: +REMOTE_DATA
   >>> temperature = radiation_temperature(calibrated)  # doctest: +REMOTE_DATA
   >>> temperature.unit  # doctest: +REMOTE_DATA
   Unit("K")
   >>> int(np.median(temperature.data[~temperature.mask]))  # doctest: +REMOTE_DATA
   5003
