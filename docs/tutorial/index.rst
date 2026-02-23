.. _irispy-tutorial-index:

***********************
The ``irispy`` tutorial
***********************

Welcome to the tutorial for ``irispy``.
The goal of this tutorial is to provide a gentle introduction to using ``irispy`` to analyze IRIS data.
``irispy`` is a instrument team led, free and open-source Python toolkit for analyzing IRIS data.

**Who this is for**

This tutorial assumes you know how to run code in Python and that you are familiar with the sunpy ecosystem.
Therefore, we do recommend that you first read over :ref:`sunpy-tutorial-index` to become familiar with the broader Python ecosystem around solar physics.
The sunpy tutorial also covers more low level topics such as units and general World Coordinate Systems (WCS) which will not be covered here in the same detail.
But we do not assume you have any prior experience with IRIS data or the specific tools in ``irispy``.

Since ``irispy`` depends heavily on other packages in the scientific Python ecosystem, including ``sunpy``, ``ndcube``, ``astropy``, ``numpy``, and ``matplotlib``.
This tutorial explains the basics of the packages needed to use ``irispy``, but is not a comprehensive tutorial for these packages and references their own introductory tutorials where relevant.

**How to use this tutorial**

This tutorial is designed to be read from start to finish.
You can either read it without running any code, or copy and paste the code snippets as you read.
Each chapter of the tutorial provides a self-contained set of codes.

.. toctree::
   :maxdepth: 2

   installation
   acquiring_data
   level_2
   calibration
   data_notes
