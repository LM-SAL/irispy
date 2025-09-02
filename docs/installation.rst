.. _irispy_install:

************
Installation
************

.. warning::

   Please be aware that the package name on pypi and conda-forge is ``irispy-lmsal`` to avoid name clashes with other packages.
   However, the package is imported as ``irispy`` and is referred to as ``irispy`` in the documentation.

Release version
===============

``irispy`` is part of the wider ecosystem of scientific Python packages for solar physics and therefore, a working installation is more about installing the scientific Python ecosystem.

To install the Miniforge Python distribution `download the executable for your Operating System <https://github.com/conda-forge/miniforge#miniforge3>`__.

The reason we choose Miniforge over Anaconda is mainly due to size, as Anaconda comes with a full install of packages you probably do not need, and this way you have more direct control over what has been installed into your Python virtual environment.

Using Miniforge
---------------

To install, launch a system command prompt or the 'Miniforge Prompt' (on Windows).

.. note::

    We strongly recommend using a `conda virtual environment. <https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307>`__

Now to install ``irispy`` within the default conda virtual environment:

.. code-block:: console

    $ conda install irispy-lmsal

This will install ``irispy``.

Updating
--------
You can update to the latest version by running:

.. code-block:: console

    $ conda update irispy-lmsal

.. _irispy_dev_install:

Development version
===================

This section outlines how to install the development version of ``irispy``.

Stable Dependencies Install
---------------------------

Create a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^
The first step is to create a conda environment (let's call it ``irispy-dev``) in which to install the development version of ``irispy``.
This will allow you to keep your root environment clean of development packages.
From the command line, type:

.. code-block:: console

    $ conda create -n irispy-dev pip

This creates the ``irispy-dev`` conda environment with just ``pip``.
Next, you must activate that environment, i.e., switch into it.

.. code-block:: console

    $ conda activate irispy-dev

Clone ``irispy`` repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we need to clone the `irispy repository`_ from `GitHub`_ into a directory called ``irispy-git``.
From the directory in which you want ``irispy-git`` to reside.
If you want to develop ``irispy``, you will want to fork the repository on GitHub and use that URL in the clone step above.

.. code-block:: console

    $ git clone <personal fork URL> irispy-git
    $ cd irispy-git
    $ git remote add upstream git@github.com:LM-SAL/irispy.git

Install ``irispy``
^^^^^^^^^^^^^^^^^^

Finally, we can install the development version.

.. code-block:: console

    $ cd irispy-git
    $ pip install -e ".[dev]"

You are now ready to develop ``irispy``.

Notice we install no dependencies or use ``conda`` to install this.
The reason for this is that it is simply easier to use ``pip`` to set up development packages.

At times you might need to get the updated changes; to do so:

.. code-block:: console

    $ git remote update -p

From here, you will need to decide if you need to merge changes or rebase changes when you need to contribute the changes back.

.. _irispy repository: https://github.com/LM-SAL/irispy
.. _GitHub: https://github.com/
