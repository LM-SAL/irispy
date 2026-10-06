.. _irispy-line-database:

***************************
IRIS spectral line database
***************************

``irispy`` provides an offline catalog of atomic lines in the three IRIS spectrograph passbands, with approximate strengths computed from fixed reference atmospheres.
Query the strongest FUV2 predictions for the quiet Sun with coronal abundances using `irispy.utils.lines.get_lines`::

    >>> import astropy.units as u
    >>> from irispy.utils.lines import get_lines
    >>> fuv2 = get_lines([1389, 1407] * u.angstrom, region="quiet_sun", abundance="coronal")
    >>> for line in fuv2[:5]:
    ...     print(f"{line['ion']:6} {line['wavelength'].value:8.2f} {line['intensity_quiet_sun_coronal']:5.2f}  {line['wavelength_source']}")
    Si IV   1393.76  1.00  observed
    Si IV   1402.77  0.50  observed
    O IV    1401.16  0.29  ritz
    O IV    1404.81  0.08  chianti
    O IV    1399.78  0.07  chianti

Wavelength ranges are inclusive and accept any spectral unit; catalog wavelengths are in vacuum.
Supplying both ``region`` (``quiet_sun``, ``active_region``, or ``flare``) and ``abundance`` (``coronal`` or ``photospheric``) ranks lines strongest first within each passband, ordered FUV1, FUV2, NUV.
Unranked lines follow the predictions unless excluded with ``include_unranked=False``.
Without a ranking model, results are sorted by wavelength.
``main_only=True`` selects :ref:`documented IRIS lines <documented-lines>`; ``categories`` filters by :ref:`catalog category <line-categories>`.

The :ref:`gallery example <sphx_glr_generated_gallery_how_to_06_identify_lines.py>` marks candidate lines on an observation.

Passbands
---------

Table 2 of :cite:t:`depontieu2014` defines the passbands: FUV1 1331.7--1358.4, FUV2 1389.0--1407.0, and NUV 2782.7--2835.1 Angstrom.
:cite:t:`wulser2018` give the same FUV ranges, and the full-detector windows of Level 2 files from 2015 and 2026 span 1331.68--1358.28, 1380.66--1406.70, and 2783.23--2835.10 Angstrom.
The FUV2 readout extends below 1389 Angstrom; the catalog follows the published passband limits without margins.

What the table contains
-----------------------

The catalog includes all `NIST Atomic Spectra Database <https://physics.nist.gov/PhysRefData/ASD/lines_form.html>`__ lines within the passbands, across all elements and ionization stages, including forbidden transitions.

* ``wavelength`` selects the NIST observed or Ritz wavelength with the smaller available uncertainty.
  Observed values take precedence when neither uncertainty is known; a sole available wavelength is used directly.
  Both values and uncertainties are retained, with the selection recorded in ``wavelength_source``.
  NIST explains that `Ritz wavelengths are usually more accurate in the vacuum ultraviolet <https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html>`__.
* ``lower`` and ``upper`` give the level configurations, terms, and J values, with energies in ``lower_energy`` and ``upper_energy``.
* ``nist_intensity`` records laboratory intensities from arc, spark, or other excitation sources, with no common solar intensity scale.
* ``intensity_<region>_<abundance>`` contains the six normalized reference-model strengths described below.
* ``log_t_max`` is the temperature of peak ionization fraction in the default equilibrium model, not the measured formation temperature of a line.
* ``reference`` identifies the publication for a documented line; ``main`` flags membership in that selection.
* ``wavelength_is_theoretical`` flags CHIANTI wavelengths derived from theoretical energy levels.

CHIANTI bound-bound transitions with available level and equilibrium ionization data are included regardless of strength or prediction availability.
Transitions unmatched to NIST have ``wavelength_source="chianti"``.
Theoretical wavelengths can be less accurate: see `CHIANTI's wavelength conventions <https://db.chiantidatabase.org/o/o_4.html>`__.
Documented lines absent from both catalogs use their published wavelength and ``wavelength_source="literature"``.

.. _documented-lines:

Documented IRIS lines
---------------------

``main_only=True`` selects the following published IRIS identifications.
This selection is not a complete solar atlas; the full catalog also includes lines such as the Si II and Fe II blends near Fe XXI :cite:p:`young2015`.

======== ========== ===========================
Ion      Wavelength Reference
======== ========== ===========================
C II     1334.53    :cite:t:`depontieu2014`
C II     1335.66    :cite:t:`rathore2015`
C II     1335.71    :cite:t:`depontieu2014`
Fe XII   1349.40    :cite:t:`depontieu2014`
Cl I     1351.66    :cite:t:`itn38`
Fe XXI   1354.08    :cite:t:`depontieu2014`
C I      1354.29    :cite:t:`young2015`
Mn XVIII 1355.01    :cite:t:`depontieu2021`
O I      1355.60    :cite:t:`depontieu2014`
C I      1355.85    :cite:t:`lin2017`
Fe II    1392.82    :cite:t:`wulser2018`
Ni II    1393.33    :cite:t:`itn38`
Si IV    1393.76    :cite:t:`depontieu2014`
O IV     1399.78    :cite:t:`depontieu2014`
O IV     1401.16    :cite:t:`depontieu2014`
Si IV    1402.77    :cite:t:`depontieu2014`
O IV     1404.81    :cite:t:`polito2016`
S IV     1404.81    :cite:t:`polito2016`
S IV     1406.01    :cite:t:`polito2016`
Mg II    2791.60    :cite:t:`pereira2015`
Mg II    2796.35    :cite:t:`depontieu2014`
Mg II    2798.75    :cite:t:`pereira2015`
Mg II    2798.82    :cite:t:`pereira2015`
Ni I     2799.47    :cite:t:`wulser2018`
Mg II    2803.53    :cite:t:`depontieu2014`
======== ========== ===========================

O IV 1399.78, O IV 1404.81, and Mn XVIII are absent from NIST and carry CHIANTI's wavelengths; Fe II 1392.82 and Ni II 1393.33 are in neither catalog and carry their references' wavelengths.

How the strengths are predicted
-------------------------------

The integrated optically thin intensity is

.. math::

   I = \frac{1}{4\pi}\int G(T, n_e)\,\mathrm{DEM}(T)\,\mathrm{d}T,

where the contribution function :math:`G` combines the upper-level population, radiative decay rate, photon energy, ionization fraction, and elemental abundance, and :math:`\mathrm{DEM}(T)=n_e n_H\,\mathrm{d}h/\mathrm{d}T`.
:math:`G` is computed with `fiasco <https://fiasco.readthedocs.io>`__ from CHIANTI 11.0.2 atomic data, including proton collisions and two-ion level-population models where the data exist.
Strengths are normalized to the strongest prediction in each passband and reference model.

* **Temperature structure:** CHIANTI's standard DEMs; quiet Sun and active region from `Vernazza and Reeves (1978) <https://doi.org/10.1086/190539>`__, and flare from the 1973 August 9 flare of `Dere and Cook (1979) <https://doi.org/10.1086/157013>`__.
* **Density:** fixed electron-pressure parameters, :math:`n_e T = 3\times10^{15}` K cm\ :sup:`-3` for quiet Sun and active region and :math:`10^{16}` K cm\ :sup:`-3` for flare.
* **Ionization:** fiasco's default ``chianti`` temperature-dependent equilibrium ionization fractions.
  Density affects excitation; ion fractions depend only on temperature.
  CHIANTI 11's advanced density-dependent ionization and charge-transfer models, which can change strong UV radiances by factors of two to five, are not used :cite:p:`dufresne2024`.
* **Abundances:** photospheric from `Asplund, Amarsi, and Grevesse (2021) <https://doi.org/10.1051/0004-6361/202140445>`__, and coronal, the same set with low first-ionization-potential (FIP) elements such as Mg, Si, and Fe enhanced by 0.5 dex.

Matching requires the same ion, wavenumbers within 5 cm\ :sup:`-1` (or three times NIST's uncertainty), identical J values, and energies within 5 cm\ :sup:`-1` for both levels.
NIST lines without level data, such as Ca II 1341.89, are matched within 0.02 Angstrom (or three times NIST's uncertainty).
Ambiguous cases are not merged.
When NIST lists several radiative channels of one transition, catalog matching selects the channel with the largest A-value.

Assumptions and limitations
---------------------------

Region names select fixed reference atmospheres rather than conditions inferred from the observation.
Rankings compare integrated energy emission within one passband and model, without predicting peak brightness, detector counts, or identification probabilities.
Candidates require confirmation from the observed spectrum and solar line identifications.
NaN strengths mark excluded ions, ions lacking CHIANTI data, and NIST lines with no CHIANTI counterpart.

* **Cool ions:** neutral and singly ionized lines are retained without predictions.
  Mg II h and k and C II require optically thick radiative-transfer calculations :cite:p:`leenaarts2013,rathore_carlsson2015`, and O I 1355.6, though optically thin, is set by recombination cascades and charge exchange with hydrogen :cite:p:`lin2015`.
* **Opacity and ionization equilibrium:** predictions assume optically thin emission and equilibrium ionization.
  Si IV can become optically thick during flares :cite:p:`kerr2019`, and dynamic plasma can depart from equilibrium.
* **Density:** at :math:`10^5` K the flare model uses :math:`n_e=10^{11}` cm\ :sup:`-3`, while IRIS flare ribbons have shown densities near :math:`10^{13}` cm\ :sup:`-3`, which change the O IV and S IV intensities and ratios :cite:p:`polito2016`.
* **Temperature structure:** the DEMs are 1970s averages, derived with older coronal abundances and reused for both abundance sets; they are least certain at low temperatures, where the active-region DEM rises by more than two orders of magnitude at 10\ :sup:`4` K.
  The abundance alternatives test sensitivity; they are not independently derived atmospheres or an accuracy bound.
* **Omitted processes and features:** absorption, photoexcitation, non-equilibrium ionization, sunspots, and molecular lines such as H\ :sub:`2` and CO.

.. _line-categories:

Categories
----------

Categories apply to ions, using published identifications for chromospheric diagnostics and equilibrium ion-fraction peaks for other ions.
They do not specify an observed formation height or whether a line appears in emission or absorption.
The temperature boundaries follow :cite:t:`tian2017,depontieu2021`:

* ``flare``: :math:`T_\mathrm{max} \geq 10` MK, such as Fe XXI.
* ``coronal``: :math:`0.8 \leq T_\mathrm{max} < 10` MK, such as Fe XII, the coronal line of :cite:t:`testa2016`, and Mn XVIII.
* ``transition_region``: :math:`0.02 \leq T_\mathrm{max} < 0.8` MK, such as Si IV, O IV, and S IV.
* ``chromospheric``: Mg II, C II, O I, C I, and Cl I, whose IRIS diagnostic formation is documented in :cite:t:`leenaarts2013,rathore_carlsson2015,lin2015,lin2017,itn38` rather than given by their ion-fraction peaks.
* ``cool_metal``: other neutral or singly ionized metals from lithium to zinc, such as Si II, Fe I, and Fe II.
  These include photospheric absorption lines and chromospheric emission blends, including those near Fe XXI in flares :cite:p:`young2015`.

Other lines, including laboratory lines of heavy elements, have a blank category.

Regenerating the table
----------------------

Run ``tools/make_line_database.py`` from a development checkout with the ``density`` extra, fiasco 0.8.2 or later, and a full CHIANTI database configured using `fiasco's instructions <https://fiasco.readthedocs.io/en/latest/quick_start.html>`__::

    python tools/make_line_database.py

A run takes a few hours: it downloads the current NIST data, recomputes the strengths, and overwrites ``iris_lines.ecsv``.
Raw NIST responses are not archived, so later runs change the catalog as NIST updates its data.
``get_lines().meta`` records the software and `NIST ASD <https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml>`__ versions, each passband's query URL, parameters, and ``query_date``, and the DEM and abundance files.
