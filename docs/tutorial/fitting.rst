.. _irispy-tutorial-fitting:

**********************
Fitting spectral lines
**********************

``irispy`` does not wrap line fitting, because astropy already does it well.
What ``irispy`` adds, in `irispy.utils.fitting`, are starting models for IRIS lines:

* `~irispy.utils.fitting.gaussians_on_background`: any number of Gaussian or Lorentzian lines on a constant or linear background, from starts that you choose.
* `~irispy.utils.fitting.si_iv_1403_model`: one line for Si IV 140.277 nm, started for every spectrum from the data.
* `~irispy.utils.fitting.mg_ii_model`: two Gaussians for the emission peaks of the Mg II k or h line core, started from `~irispy.utils.mg_features.calculate_mg_features`.

Every starting parameter can be an array over the spatial axes of a cube, so each spectrum gets its own start.

Fitting a cube
==============

We use a cutout of an active region observation with the Si IV 1403 window, as in the gallery examples, and a small part of its field of view:

.. code-block:: python

    >>> import numpy as np
    >>> import pooch
    >>> import astropy.units as u
    >>> from astropy.coordinates import SpectralCoord
    >>> from astropy.modeling.fitting import TRFLSQFitter, parallel_fit_dask
    >>> from irispy.io import read_files
    >>> from irispy.utils.fitting import si_iv_1403_model
    >>> filename = pooch.retrieve(
    ...     "https://github.com/LM-SAL/irispy-data/releases/download/v1/iris_l2_20180102_153155_3610108077_cutout_raster.tar.gz",
    ...     known_hash="ff80e6a7900d4d5e1716a6415db25d40b6058f3523184b549c7e0d9928c0b68b",
    ... )  # doctest: +REMOTE_DATA
    >>> raster = read_files(filename, spectral_windows="Si IV 1403", uncertainty=True)  # doctest: +REMOTE_DATA
    >>> cube = raster["Si IV 1403"][0][100:140, 200:260]  # doctest: +REMOTE_DATA

The starting models describe one line and no other, so the cube has to be cropped to that line.
The Si IV 1403 window also holds O IV and S IV lines, and `~irispy.utils.fitting.si_iv_1403_model` raises an error if the cube covers any of them.
Here we keep the wavelengths halfway to the neighbouring O IV lines at 140.116 and 140.481 nm:

.. code-block:: python

    >>> si_iv = 1402.77 * u.AA
    >>> window = [(1401.157 * u.AA + si_iv) / 2, (si_iv + 1404.806 * u.AA) / 2]
    >>> cube = cube.crop([SpectralCoord(window[0]), None], [SpectralCoord(window[1]), None])  # doctest: +REMOTE_DATA
    >>> model = si_iv_1403_model(cube)  # doctest: +REMOTE_DATA
    >>> model.param_names  # doctest: +REMOTE_DATA
    ('amplitude_0', 'amplitude_1', 'mean_1', 'stddev_1')
    >>> model.mean_1.shape  # doctest: +REMOTE_DATA
    (40, 60)

The background is component 0 and the line component 1.
Now we fit every spectrum:

.. code-block:: python

    >>> wavelength = cube.axis_world_coords("em.wl")[0].to(u.AA)  # doctest: +REMOTE_DATA
    >>> good = np.isfinite(cube.data) & ~cube.mask  # doctest: +REMOTE_DATA
    >>> fitter = TRFLSQFitter()
    >>> fitted = parallel_fit_dask(
    ...     model=model,
    ...     fitter=fitter,
    ...     data=np.where(good, cube.data, 0),
    ...     data_unit=cube.unit,
    ...     weights=np.where(good, 1 / cube.uncertainty.array, 0),
    ...     world=(wavelength,),
    ...     fitting_axes=2,
    ...     fit_info=True,
    ...     scheduler="single-threaded",
    ... )  # doctest: +REMOTE_DATA
    >>> covariance = fitter.fit_info.get_property_as_array("param_cov")  # doctest: +REMOTE_DATA
    >>> errors = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))  # doctest: +REMOTE_DATA
    >>> fitted.mean_1.quantity.shape, errors.shape  # doctest: +REMOTE_DATA
    ((40, 60), (40, 60, 4))

``errors[..., i]`` is the 1σ uncertainty of the parameter ``fitted.param_names[i]``.
The other properties of each fit, such as ``status``, ``success`` and ``nfev``, come out of ``fitter.fit_info`` in the same way.
The default ``scheduler`` uses several processes; ``"single-threaded"`` keeps this example simple.
A ``dask.distributed.Client`` can be passed instead.

Things to know
==============

* **Pass the wavelengths in Å.**
  The starting models are in Å, and astropy converts a model's parameters to the unit of the wavelengths it is given but not their bounds.
  Its least-squares fitters also do not rescale the parameters, so a fit in metres, the unit of the cube's WCS, barely moves the line centres and widths.
  That is why the wavelengths are passed as ``world`` rather than the cube itself.
* **Masked and missing samples.**
  The fitters do not accept NaN, so replace bad samples with any number and give them zero weight, as above.
  Spectra that are entirely NaN, and fits that raise an error, give NaN parameters and a zero covariance.
* **Uncertainties.**
  With weights of 1/σ, ``param_cov`` is the covariance of the parameters given those uncertainties, and the errors are only as good as the uncertainties of the cube.
  Without weights, astropy scales it by the variance of the residuals, as `scipy.optimize.curve_fit` does with ``absolute_sigma=False``.
* **Bounds.**
  `~astropy.modeling.fitting.TRFLSQFitter` keeps the parameters within their bounds while it fits;
  `~astropy.modeling.fitting.LMLSQFitter`, which is faster, clips them to the bounds at each step.
* **Faint lines.**
  When a line is not much brighter than the noise, its fit is poorly constrained whatever its start.
  Average neighbouring spectra first, for example with ``cube.rebin((2, 2, 1))``.
* **Two Gaussians can swap.**
  In `~irispy.utils.fitting.mg_ii_model`, ``mean_1`` starts at the blue peak and ``mean_2`` at the red one, but both may move anywhere within the velocity range.
  In the gallery example about one fit in a thousand ends with them the other way round, so sort them by wavelength before making maps.
* **Several components.**
  `~irispy.utils.fitting.si_iv_1403_model` offers one component because the layout of several depends on what is observed:
  a narrow and a broad Gaussian with one centroid in active-region loops :cite:p:`dudik2017`, a static and a redshifted component in flare ribbons :cite:p:`yu2020`, and components tens of km/s from the line centre in UV bursts :cite:p:`peter2014,young2018`.
  Build such a model with `~irispy.utils.fitting.gaussians_on_background` and starts that suit your data.

The gallery examples :ref:`sphx_glr_generated_gallery_analysis_01_spectral_fitting.py` and :ref:`sphx_glr_generated_gallery_analysis_07_mg_ii_two_gaussian_fitting.py` turn the fitted parameters into maps.
