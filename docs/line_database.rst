IRIS spectral line database
===========================

``irispy`` ships an offline table of atomic lines in the three IRIS spectrograph passbands, with reference-model strengths for eligible ions.
Use it to find candidate transitions near a wavelength; a candidate is not a confirmed solar line identification.
Use `irispy.utils.lines.get_lines`::

    >>> import astropy.units as u
    >>> from irispy.utils.lines import get_lines
    >>> documented = get_lines(main_only=True)
    >>> len(documented)
    25
    >>> lines = get_lines([1402, 1404] * u.angstrom,
    ...                   region="quiet_sun", abundance="photospheric")
    >>> str(lines["ion"][0])
    'Si IV'

Ranges accept any spectral unit and include both endpoints; all wavelengths are in vacuum.
Supplying ``region`` (``quiet_sun``, ``active_region``, or ``flare``) and ``abundance`` (``coronal`` or ``photospheric``) groups results as FUV1, FUV2, then NUV, sorting strongest first within each passband; otherwise lines are sorted by wavelength.
Lines without a prediction come last within each passband, and ``include_unranked=False`` removes them.
``main_only=True`` selects a :ref:`curated set of documented IRIS lines <documented-lines>`, and ``categories`` filters by :ref:`catalog category <line-categories>`.

Lines in each passband
----------------------

The five strongest predicted FUV2 lines for the quiet Sun with coronal abundances are::

    >>> fuv2 = get_lines([1389, 1407] * u.angstrom, region="quiet_sun", abundance="coronal")
    >>> for line in fuv2[:5]:
    ...     print(f"{line['ion']:6} {line['wavelength'].value:8.2f} {line['intensity_quiet_sun_coronal']:5.2f}  {line['wavelength_source']}")
    Si IV   1393.76  1.00  observed
    Si IV   1402.77  0.50  observed
    O IV    1401.16  0.29  ritz
    O IV    1404.81  0.08  chianti
    O IV    1399.78  0.07  chianti

The figure shows every line predicted to reach 10\ :sup:`-3` of its passband's strongest line, using coronal abundances.
Grey ticks along the bottom mark lines without a prediction.
:ref:`sphx_glr_generated_gallery_how_to_06_identify_lines.py` marks these lines on an IRIS observation.

.. plot::
    :show-source-link: False

    import matplotlib.pyplot as plt
    import numpy as np

    import astropy.units as u

    from irispy.utils.lines import get_lines

    table = get_lines()
    models = {"quiet_sun": ("Quiet Sun", "o"), "active_region": ("Active region", "s"), "flare": ("Flare", "^")}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
    for ax, (band, (low, high)) in zip(axes, table.meta["passband_limits_angstrom"].items()):
        lines = table[table["passband"] == band]
        wavelength = lines["wavelength"].to_value(u.AA)
        strength = np.array([lines[f"intensity_{region}_coronal"] for region in models])
        ranked = np.isfinite(strength).any(axis=0)
        ax.vlines(wavelength[~ranked], 1e-3, 1.5e-3, color="0.75", linewidth=0.5)
        ax.vlines(wavelength[ranked], 1e-3, np.nanmax(strength[:, ranked], axis=0), color="0.5", linewidth=0.8)
        for values, (label, marker) in zip(strength, models.values()):
            ax.plot(wavelength, values, marker, markerfacecolor="none", linestyle="none", label=label)
        labelled = []
        for line in lines[lines["main"]]:
            position = line["wavelength"].to_value(u.AA)
            if all(abs(position - other) > 0.5 for other in labelled):
                ax.annotate(f"{line['ion']} {position:.1f}", (position, 1.5), rotation=90, ha="center", fontsize=8)
                labelled.append(position)
        ax.set(xlim=(low, high), ylim=(1e-3, 30), yscale="log", title=band, ylabel="Relative strength")
    axes[-1].set_xlabel("Vacuum wavelength [Å]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3)
    plt.show()

Passbands
---------

The passbands are those of Table 2 of :cite:t:`depontieu2014`, without margins: FUV1 1331.7--1358.4, FUV2 1389.0--1407.0, and NUV 2782.7--2835.1 Angstrom.
:cite:t:`wulser2018` give the same FUV ranges, and the full-detector windows of Level 2 files from 2015 and 2026 span 1331.68--1358.28, 1380.66--1406.70, and 2783.23--2835.10 Angstrom.
The FUV2 detector is therefore read out below 1389 Angstrom, but no documented passband includes that region, so the database stops at 1389.0 Angstrom.

What the table contains
-----------------------

Every line in the `NIST Atomic Spectra Database <https://physics.nist.gov/PhysRefData/ASD/lines_form.html>`__ within the passbands is included, for all elements and ionization stages, including forbidden lines.

* ``wavelength`` selects the NIST observed or Ritz wavelength with the smaller available uncertainty.
  When neither uncertainty is known, the observed value is preferred; when only one wavelength exists, it is used.
  Both values and their uncertainties are retained, and ``wavelength_source`` identifies the selection.
  NIST explains that `Ritz wavelengths are usually more accurate in the vacuum ultraviolet <https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html>`__.
  This selection is not a recommendation for a Doppler reference: check the original measurement, blends, and the instrument calibration conventions :cite:p:`wulser2018`.
* ``lower`` and ``upper`` give the level configurations, terms, and J values, with energies in ``lower_energy`` and ``upper_energy``.
* ``nist_intensity`` is NIST's laboratory intensity, from arc, spark, or other laboratory sources; it says nothing about solar strengths.
* ``intensity_<region>_<abundance>`` are the six normalized, integrated reference-model intensities described below.
  Each column is normalized to 1 for the strongest predicted line in each passband, so values compare lines within one passband and one model only.
* ``log_t_max`` is the temperature of peak ionization fraction in the default equilibrium model, not the measured formation temperature of a line.
  ``category`` is a coarse catalog label independent of prediction availability.
* ``reference`` names the source for a line in the curated selection, and ``main`` is ``True`` for those lines.
* ``wavelength_is_theoretical`` flags CHIANTI wavelengths derived from theoretical energy levels.
  CHIANTI wavelengths can also come from observed energy levels; ``wavelength_source="chianti"`` alone does not distinguish them.

CHIANTI bound-bound transitions in the passbands with available level and equilibrium ionization data are included, whether or not they have a prediction.
Unmatched transitions have ``wavelength_source="chianti"``; catalog inclusion does not depend on a strength threshold.
Theoretical wavelengths can be less accurate: see `CHIANTI's wavelength conventions <https://db.chiantidatabase.org/o/o_4.html>`__.
Documented lines in neither catalog are added with the wavelength from their reference and ``wavelength_source`` ``literature``.

.. _documented-lines:

Documented IRIS lines
---------------------

This is a curated selection of lines identified in the IRIS literature, not a complete solar atlas.
Each carries its reference in the ``reference`` column, and ``main_only=True`` selects them.
Known lines outside this selection remain available in the full catalog, including Si II and Fe II blends near Fe XXI :cite:p:`young2015`.

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

Table 4 of :cite:t:`depontieu2014` lists the lines that set the thermal coverage of the spectrograph.
:cite:t:`depontieu2021` add the Mg II triplet, the O IV and S IV multiplets, and the hot Mn XVIII line that can blend with Fe XXI.
:cite:t:`itn38` add Cl I 1351.66 and the Ni II 1393.33 blend of Si IV, and :cite:t:`wulser2018` name the Fe II 1392.82 and Ni I 2799.47 wavelength calibration lines.
The two C I lines are the blend with Fe XXI of :cite:t:`young2015` and the chromospheric diagnostic of :cite:t:`lin2017`.
O IV 1399.78, O IV 1404.81, and Mn XVIII are absent from NIST and carry CHIANTI's wavelengths; Fe II 1392.82 and Ni II 1393.33 are in neither catalog and carry their references' wavelengths.

How the strengths are predicted
-------------------------------

Each finite strength is obtained by normalizing an integrated optically thin line intensity for a fixed reference atmosphere,

.. math::

   I = \frac{1}{4\pi}\int G(T, n_e)\,\mathrm{DEM}(T)\,\mathrm{d}T,

where the contribution function :math:`G` combines the upper-level population, radiative decay rate, photon energy, ionization fraction, and elemental abundance, and :math:`\mathrm{DEM}(T)=n_e n_H\,\mathrm{d}h/\mathrm{d}T`.
:math:`G` is computed with `fiasco <https://fiasco.readthedocs.io>`__ from CHIANTI 11.0.2 atomic data, including proton collisions and two-ion level-population models where the data exist.

* **Temperature structure:** CHIANTI's standard DEMs; quiet Sun and active region from `Vernazza and Reeves (1978) <https://doi.org/10.1086/190539>`__, and flare from the 1973 August 9 flare of `Dere and Cook (1979) <https://doi.org/10.1086/157013>`__.
* **Density:** fixed electron-pressure parameters, :math:`n_e T = 3\times10^{15}` K cm\ :sup:`-3` for quiet Sun and active region and :math:`10^{16}` K cm\ :sup:`-3` for flare.
  These are model inputs, not densities inferred from an observation.
* **Ionization:** fiasco's default ``chianti`` temperature-dependent equilibrium ionization fractions.
  Density changes the excitation calculation, but the ion fractions do not vary with density here.
  CHIANTI 11's advanced density-dependent ionization and charge-transfer models are not enabled by this calculation :cite:p:`dufresne2024`.
* **Abundances:** photospheric from `Asplund, Amarsi, and Grevesse (2021) <https://doi.org/10.1051/0004-6361/202140445>`__, and coronal, the same set with low first-ionization-potential (FIP) elements such as Mg, Si, and Fe enhanced by 0.5 dex.
  The coronal set therefore strengthens low-FIP lines relative to those of C, N, O, and the noble gases.

A CHIANTI prediction is attached to a NIST line only when both describe the same transition: same ion, wavenumbers within 5 cm\ :sup:`-1` (or three times NIST's uncertainty), and the same J values and energies within 5 cm\ :sup:`-1` for both levels.
NIST lists a few lines without level data, such as Ca II 1341.89; those take the prediction within 0.02 Angstrom (or three times NIST's uncertainty).
Ambiguous cases are not merged.
When NIST lists several radiative channels of one transition, catalog matching selects the channel with the largest A-value.
Neutral and singly ionized transitions, including Mg II's E1 and weak M2 entries, remain without predictions.

Assumptions and limitations
---------------------------

The region names select reference models; they do not fit an observation or establish its physical conditions.
The strengths compare integrated energy emission within one passband and model.
They do not predict peak brightness, IRIS detector counts, or the probability of a line identification.

* **Excluded cool ions.** Neutral and singly ionized lines are retained without predictions.
  Mg II h and k and C II require optically thick radiative-transfer calculations :cite:p:`leenaarts2013,rathore_carlsson2015`.
  O I 1355.6 is optically thin, but its emission is dominated by recombination cascades and its ion balance by charge exchange with hydrogen :cite:p:`lin2015`.
  Excluding these ions is a conservative scope restriction; it does not imply that every transition of them is optically thick or weak.
* **Remaining predictions are conditional.** Optically thin calculations are useful for suitable transition-region and coronal lines, but a finite value is not a validity guarantee.
  Si IV can become optically thick during flares :cite:p:`kerr2019`.
  The equilibrium ionization approximation also needs checking in dynamic plasma.
* **Density sensitivity.** At :math:`10^5` K the flare model uses :math:`n_e=10^{11}` cm\ :sup:`-3`.
  IRIS flare diagnostics have inferred ribbon densities near :math:`10^{13}` cm\ :sup:`-3`, which change the O IV and S IV intensities and ratios :cite:p:`polito2016`.
  CHIANTI's advanced ionization treatment can also change strong UV radiances by factors of two to five :cite:p:`dufresne2024`.
* **Generic temperature structure.** The DEMs are 1970s averages, derived with older coronal abundances and used unchanged for both abundance sets; real quiet Sun, plage, and flares vary widely.
  The alternative abundance sets are sensitivity choices, not independently derived atmospheres or an accuracy bound.
* **Cool plasma is poorly constrained.** The DEMs are least certain at their coolest temperatures, and the active-region DEM rises by more than two orders of magnitude at 10\ :sup:`4` K.
  Removing cool-ion predictions prevents them from setting the normalization, but does not validate the remaining DEM or line ratios.
* **No prediction does not mean weak.** Ions without CHIANTI data, and NIST transitions that cannot be matched, have NaN strengths; so do Cl I and C I, whose lines are bright in IRIS spectra.
* **Not modeled:** absorption, photoexcitation, non-equilibrium ionization, sunspots, and molecular lines such as H\ :sub:`2` and CO.

.. _line-categories:

Categories
----------

Categories are catalog labels, not formation-height measurements, and all lines of an ion share one category.
They do not establish whether a transition appears in emission or absorption in a particular observation.
The established chromospheric diagnostics below are identified from the literature, independent of their model temperature or prediction availability.
For other ions the boundaries use the equilibrium ion-fraction peak, following the temperature ranges of :cite:t:`tian2017` and :cite:t:`depontieu2021`:

* ``flare``: :math:`T_\mathrm{max} \geq 10` MK, such as Fe XXI.
* ``coronal``: :math:`0.8 \leq T_\mathrm{max} < 10` MK, such as Fe XII, the coronal line of :cite:t:`testa2016`, and Mn XVIII.
* ``transition_region``: :math:`0.02 \leq T_\mathrm{max} < 0.8` MK, such as Si IV, O IV, and S IV.
* ``chromospheric``: Mg II, C II, O I, C I, and Cl I, whose IRIS diagnostic formation is documented in :cite:t:`leenaarts2013,rathore_carlsson2015,lin2015,lin2017,itn38`.
  C II's coronal-equilibrium ion fraction peaks near 25 000 K, but its optically thick cores can form near 10 000 K :cite:p:`rathore_carlsson2015`; the ion-fraction peak alone is not a formation diagnostic.
* ``cool_metal``: other neutral or singly ionized metals from lithium to zinc, such as Si II, Fe I, and Fe II.
  These include photospheric absorption lines and chromospheric emission blends, including those near Fe XXI in flares :cite:p:`young2015`.

Other lines, including laboratory lines of heavy elements, have a blank category.

Regenerating the table
----------------------

``tools/make_line_database.py`` updates the table in a development checkout; it is not part of the installed package.
It needs the ``density`` extra, fiasco 0.8.2 or later, and a full CHIANTI database installed following `fiasco's instructions <https://fiasco.readthedocs.io/en/latest/quick_start.html>`__::

    python tools/make_line_database.py

The script prints the software and database versions, downloads current NIST data, recomputes the strengths, and overwrites ``iris_lines.ecsv``; it takes a few hours.
Review the update with ``git diff``.
The table's metadata records the software versions, NIST queries, category boundaries, and the DEM and abundance files used.
