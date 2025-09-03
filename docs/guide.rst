.. _irispy_guide:

*******************
An ``irispy`` guide
*******************

This guide is intended to help the solar community to start working with level 2 data using Python.
It is especially intended for those with limited Python experience who want to begin using the language to analyze data.

This guide will cover:

- reading level 2 files
- loading data from any of the spectral windows

In the future more features will arrive:

- a GUI to interact with the data
- a way to save and load regions of interest
- analysis tools
- Density Diagnostics
- Dopplergrams
- PSF and filter functions
- reading of L3 data

Currently, ``irispy`` does not provide a way to download the data from the `IRIS archive <https://iris.lmsal.com/data.html>`__.
We recommend browsing the catalogue using your web browser.

This guide uses some "sample data" from the IRIS archive that can be accessed:

.. code-block:: python

  >>> import irispy.data.sample as sample_data  # doctest: +REMOTE_DATA

Once the level 2 data has downloaded, the next step is to read, extract, and inspect them.

.. code-block:: python

  >>> from irispy.io import read_files  # doctest: +REMOTE_DATA

The sample data is from this `observation
<https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20211001_060925_3683602040_2021-10-01T06%3A09%3A252021-10-01T06%3A09%3A25.xml>`__.

It is a small observation of an active region (NOAA 12880) that contains one sunspot.
Let us retrieve the header of the raster file and display the description of the observation:

.. code-block:: python

    >>> raster = read_files(sample_data.RASTER_TAR)  # doctest: +REMOTE_DATA

.. note::
    By default, this will load the data into memory.
    You can pass ``memmap=True`` to avoid this; the data array will be a `numpy.memmap` instead.
    In this case, the data are not loaded into system memory, but written to a temporary file.

.. code-block:: python

    >>> raster  # doctest: +REMOTE_DATA
    <irispy.spectrograph.RasterCollection object at ...>
    <BLANKLINE>
    Raster Collection
    -----------------
    Spectral Windows (cube keys): (np.str_('C II 1336'), np.str_('Si IV 1394'), np.str_('Mg II k 2796'))
    Number of Cubes: 3
    Aligned dimensions: [5 16 548]
    Aligned physical types: [('meta.obs.sequence',), ...]
    <BLANKLINE>

We get some basic information about the raster file from this: what spectral windows were observed, the size of the cube, and the wavelength keys.

Let us check the metadata of this collection, this is stored as a ``meta`` attribute:

.. code-block:: python

    >>> raster["C II 1336"][0].meta  # doctest: +REMOTE_DATA
    <irispy.meta.SGMeta object at ...>
    <BLANKLINE>
    SGMeta
    ------
    Observatory:     IRIS
    Instrument:      SPEC
    Detector:        FUV1
    Spectral Window: C II 1336
    Spectral Range:  [1331.70275015 1358.28579039] Angstrom
    Spectral Band:   FUV
    Dimensions:      [ 16 548 513]
    Date:            2021-10-01T06:09:25.090
    OBS ID:          3683602040
    OBS Description: Very large sparse 16-step raster 15x175 16s   Deep x 0.5 Spatial x 2
    <BLANKLINE>

Note this is not on the main object but each individual element, in this case the spectral window.
While SJI files contain just one spectral window per file, raster files have several spectral windows per file.

If we want to check the primary header of the raster, we can do the following:

.. code-block:: python

    >>> raster["C II 1336"][0].meta.fits_header  # doctest: +REMOTE_DATA
    SIMPLE  =                    T / Written by IDL:  Mon Nov 15 09:21:38 2021
    BITPIX  =                   16 / Number of bits per data pixel
    NAXIS   =                    0 / Number of data axes
    EXTEND  =                    T / FITS data may contain extensions
    DATE    = '2021-11-15'         / Creation UTC (CCCC-MM-DD) date of FITS header
    COMMENT FITS (Flexible Image Transport System) format is defined in 'Astronomy
    COMMENT and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H
    TELESCOP= 'IRIS    '           /
    INSTRUME= 'SPEC    '           /
    ...

As the SJI level 2 data are simpler than the raster files, since they have only one spectral window per file, we will start with this data instead of the raster file above.

We use the following command to read and load the data from a SJI level 2 file:

.. code-block:: python

    >>> iris_sji = read_files(sample_data.SJI_1330)  # doctest: +REMOTE_DATA
    >>> iris_sji  # doctest: +REMOTE_DATA
    <irispy.sji.SJICube object at ...>
    <BLANKLINE>
    SJICube
    -------
    Observatory:           IRIS
    Instrument:            SJI
    Bandpass:              1330.0
    Obs. Start:            2021-10-01T06:09:24.920
    Obs. End:              2021-10-01T06:11:44.461
    Instance Start:        2021-10-01T06:09:25.020
    Instance End:          2021-10-01T06:11:37.580
    Total Frames in Obs.:  None
    IRIS Obs. id:          3683602040
    IRIS Obs. Description: Very large sparse 16-step raster 15x175 16s   Deep x 0.5 Spatial x 2
    Axis Types:            [('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'time', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM', 'custom:CUSTOM'), ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'), ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat')]
    Roll:                  0.000464606
    Cube dimensions:       (20, 548, 555)
    <BLANKLINE>

Metadata
========

Here we highlight some of the more important metadata that is available.

We can use it to find out what kind of data this is:

.. code-block:: python

    >>> iris_sji.meta["OBS_DESC"]  # doctest: +REMOTE_DATA
    'Very large sparse 16-step raster 15x175 16s   Deep x 0.5 Spatial x 2'

When the observation started:

.. code-block:: python

    >>> iris_sji.meta['STARTOBS']   # doctest: +REMOTE_DATA
    '2021-10-01T06:09:24.920'

It is possible it might be in ``"DATE_OBS"`` instead.

The exposure times:

.. code-block:: python

    >>> iris_sji.exposure_time   # doctest: +REMOTE_DATA
    <Quantity [0.50031197, 0.50025398, 0.50023699, 0.50024003, 0.50023901,
               0.50028503, 0.50024903, 0.500269  , 0.50026202, 0.500247  ,
               0.50029403, 0.50021601, 0.50028402, 0.50023901, 0.50024903,
               0.50025803, 0.500283  , 0.50029802, 0.50029498, 0.50027299] s>

In most cases, the exposure times are fixed for all scans in a raster.
However, when automatic exposure compensation (AEC) is enabled and there is a very energetic event (e.g., a flare), IRIS will automatically use a lower exposure time to prevent detector saturation.

If the exposure time varies, you can get the time-dependent exposure times in seconds from the auxiliary metadata, second to last HDU in the file with the keys ``"EXPTIMEF"`` and ``"EXPTIMEN"``.

To get arrays of timestamps, or exposure times or "xcenix", that information will be in the ``extra_coords`` attribute.

.. code-block:: python

    >>> iris_sji.extra_coords  # doctest: +REMOTE_DATA
    <ndcube.extra_coords.extra_coords.ExtraCoords object at ...>
    ExtraCoords(exposure time (0) None: QuantityTableCoordinate ['exposure time'] [None]:
    <Quantity [0.50031197, 0.50025398, 0.50023699, 0.50024003, 0.50023901,
               0.50028503, 0.50024903, 0.500269  , 0.50026202, 0.500247  ,
               0.50029403, 0.50021601, 0.50028402, 0.50023901, 0.50024903,
               0.50025803, 0.500283  , 0.50029802, 0.50029498, 0.50027299] s>,
                obs_vrix (0) None: QuantityTableCoordinate ['obs_vrix'] [None]:
    <Quantity [-253.13569641, -242.44810486, -231.77319336, -221.11309814,
               -210.41799927, -199.78419495, -189.16329956, -178.50950623,
               -167.91630554, -157.33630371, -146.72239685, -136.17030334,
               -125.63009644, -115.05719757, -104.5714035 ,  -94.14320374,
                -83.69550323,  -73.3214035 ,  -62.97399902,  -52.65399933] m / s>,
                ophaseix (0) None: QuantityTableCoordinate ['ophaseix'] [None]:
    <Quantity [0.77429509, 0.77548558, 0.77667391, 0.77786386, 0.77905941,
               0.78024989, 0.78144038, 0.78263599, 0.78382647, 0.78501666,
               0.78621155, 0.78740203, 0.78859252, 0.78978807, 0.79097688,
               0.79216683, 0.79336196, 0.79455239, 0.79574287, 0.79693335] arcsec>,
                pztx (0) None: QuantityTableCoordinate ['pztx'] [None]:
    <Quantity [-7.97803831e+00, -3.98715830e+00,  3.72256944e-03,
                3.99460268e+00, -7.97803831e+00, -3.98715830e+00,
                3.72256944e-03,  3.99460268e+00, -7.97803831e+00,
               -3.98715830e+00,  3.72256944e-03,  3.99460268e+00,
               -7.97803831e+00, -3.98715830e+00,  3.72256944e-03,
                3.99460268e+00, -7.97803831e+00, -3.98715830e+00,
                3.72256944e-03,  3.99460268e+00] arcsec>,
                pzty (0) None: QuantityTableCoordinate ['pzty'] [None]:
    <Quantity [0.6446346 , 0.66160059, 0.67856681, 0.69553316, 0.6446346 ,
               0.66160059, 0.67856681, 0.69553316, 0.6446346 , 0.66160059,
               0.67856681, 0.69553316, 0.6446346 , 0.66160059, 0.67856681,
               0.69553316, 0.6446346 , 0.66160059, 0.67856681, 0.69553316] arcsec>,
                slit x position (0) None: QuantityTableCoordinate ['slit x position'] [None]:
    <Quantity [258.75      , 270.74543085, 282.74086427, 294.73629541,
               258.75      , 270.74543085, 282.74086427, 294.73629541,
               258.75      , 270.74543085, 282.74086427, 294.73629541,
               258.75      , 270.74543085, 282.74086427, 294.73629541,
               258.75      , 270.74543085, 282.74086427, 294.73629541] arcsec>,
                slit y position (0) None: QuantityTableCoordinate ['slit y position'] [None]:
    <Quantity [254.75, 254.75, 254.75, 254.75, 254.75, 254.75, 254.75, 254.75,
               254.75, 254.75, 254.75, 254.75, 254.75, 254.75, 254.75, 254.75,
               254.75, 254.75, 254.75, 254.75] arcsec>,
                xcenix (0) None: QuantityTableCoordinate ['xcenix'] [None]:
    <Quantity [-321.64163621, -321.64154081, -321.64054553, -321.63951873,
               -321.5924215 , -321.59850309, -321.60135777, -321.56819773,
               -321.55565282, -321.55661478, -321.51550993, -321.5241685 ,
               -321.4984636 , -321.49132346, -321.47172876, -321.48122647,
               -321.46051587, -321.41851219, -321.42161527, -321.42543197] arcsec>,
                ycenix (0) None: QuantityTableCoordinate ['ycenix'] [None]:
    <Quantity [390.41458808, 390.43178122, 390.44696156, 390.46218927,
               390.40669468, 390.41598631, 390.42799954, 390.43424635,
               390.38567211, 390.39919174, 390.41787952, 390.43324879,
               390.40355692, 390.4319302 , 390.43515948, 390.44981385,
               390.41605352, 390.43774154, 390.45774336, 390.47763699] arcsec>)

Understanding a level 2 FITS file
=================================

The structure of the level 2 FITS data file is as follows:

Level 2 FITS files are multi-extension FITS files.
An "extension" refers to a part of the file containing self-consistent information, which may be a header or its corresponding data.
The first extension is called ``primary`` and its ``extension number`` is 0.

The extensions in a level 2 SJI FITS file has the following numbers:

   - ``0``: header and data corresponding to the spectral images observed by the SJI.
   - ``1``: header and auxiliary 31 values from each exposure taken by the SJI in the spectral band of the file.
     It is an array of float values with dimensions :math:`no. images \times 31`.
   - ``2``: header and extra data from each exposure taken by the SJI in the spectral band of the file.
     It is a record array containing 5 string fields for each exposure.
     The values of each field can be accessed as the key in a dictionary or as an attribute.
     See example in the last code block of this section.

A level 2 raster FITS file has the following extensions:

   -  ``0``: main header with the main information of the observation.
      This header has information about all the spectral windows contained in the file and other relevant and
      general information.
      This extension DOES NOT have spectral data associated with the file.
   -  ``1`` to ``N``: header and data for the N spectral windows contained in the file.
   -  ``N+1``: header and auxiliary 47 values from each exposure considered in the file.
      It is an array of float values with dimensions :math:`no. acquisitions \times 47`.
   -  ``N+2``: header and extra information data from each exposure considered in the file.
      It is a record array containing 9 string fields for each exposure. The values of
      each field can be accessed as the key in a dictionary or as an attribute.
      See example in the last code block of this section.

The function `astropy.io.fits.info` shows the information of the extensions contained in the level 2 file.
For a SJI file:

.. code-block:: python

   >>> from astropy.io import fits   # doctest: +REMOTE_DATA
   >>> fits.info(sample_data.SJI_1330)   # doctest: +REMOTE_DATA
    Filename: ...iris_l2_20211001_060925_3683602040_SJI_1330_t000.fits.gz
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     162   (555, 548, 20)   int16 (rescales to float32)
      1                1 ImageHDU        38   (31, 20)   float64
      2                1 TableHDU        33   20R x 5C   [A10, A10, A4, A66, A63]

and for the raster file:

.. code-block:: python

    >>> fits.info(sample_data.RASTER_FITS)  # doctest: +REMOTE_DATA
    Filename: ...iris_l2_20211001_060925_3683602040_raster_t000_r00000.fits
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     215   ()
      1                1 ImageHDU        33   (513, 548, 16)   int16 (rescales to float32)
      2                1 ImageHDU        33   (512, 548, 16)   int16 (rescales to float32)
      3                1 ImageHDU        33   (1018, 548, 16)   int16 (rescales to float32)
      4                1 ImageHDU        54   (47, 16)   float64
      5                1 TableHDU        53   16R x 7C   [A10, A10, A4, A10, A4, A66, A66]

If you would like a bit more information, we have a similar function within ``irispy``:

.. code-block:: python

    >>> from irispy.io import fits_info  # doctest: +REMOTE_DATA
    >>> fits_info(sample_data.SJI_1330)  # doctest: +REMOTE_DATA
    Filename: ...iris_l2_20211001_060925_3683602040_SJI_1330_t000.fits.gz
    Observation: Very large sparse 16-step raster 15x175 16s   Deep x 0.5 Spatial x 2
    OBS ID: 3683602040
    No.   Name  Ver    Type    Cards   Dimensions              Format                  Description
    --- ------- --- ---------- ----- -------------- --------------------------- -------------------------
      0 PRIMARY   1 PrimaryHDU   162 (555, 548, 20) int16 (rescales to float32) SJI 1330 (1310 - 1350 AA)
      1           1   ImageHDU    38       (31, 20)                     float64            Auxiliary data
      2           1   TableHDU    33       20R x 5C    [A10, A10, A4, A66, A63]            Auxiliary data

.. code-block:: python

    >>> fits_info(sample_data.RASTER_FITS)  # doctest: +REMOTE_DATA
    Filename: ...iris_l2_20211001_060925_3683602040_raster_t000_r00000.fits
    Observation: Very large sparse 16-step raster 15x175 16s   Deep x 0.5 Spatial x 2
    OBS ID: 3683602040
    No.   Name  Ver    Type    Cards    Dimensions                 Format                       Description
    --- ------- --- ---------- ----- --------------- --------------------------------- -----------------------------
      0 PRIMARY   1 PrimaryHDU   215              ()                                        Primary Header (no data)
      1           1   ImageHDU    33  (513, 548, 16)       int16 (rescales to float32)    C II 1336 (1332 - 1358 AA)
      2           1   ImageHDU    33  (512, 548, 16)       int16 (rescales to float32)   Si IV 1394 (1381 - 1407 AA)
      3           1   ImageHDU    33 (1018, 548, 16)       int16 (rescales to float32) Mg II k 2796 (2783 - 2835 AA)
      4           1   ImageHDU    54        (47, 16)                           float64                Auxiliary data
      5           1   TableHDU    53        16R x 7C [A10, A10, A4, A10, A4, A66, A66]                Auxiliary data


If we now want to recover the main header of any file:

.. code-block:: python

    # The main header of a SJI file
    >>> fits.getheader(sample_data.SJI_1330)  # doctest: +REMOTE_DATA
    SIMPLE  =                    T / Written by IDL:  Mon Nov 15 09:26:15 2021
    BITPIX  =                   16 / Number of bits per data pixel
    NAXIS   =                    3 / Number of data axes
    NAXIS1  =                  555 /
    NAXIS2  =                  548 /
    NAXIS3  =                   20 /
    EXTEND  =                    T / FITS data may contain extensions
    DATE    = '2021-11-15'         / Creation UTC (CCCC-MM-DD) date of FITS header
    COMMENT FITS (Flexible Image Transport System) format is defined in 'Astronomy
    COMMENT and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H
    TELESCOP= 'IRIS    '           /
    INSTRUME= 'SJI     '           /
    ...

    # The main header of a raster file
    >>> fits.getheader(sample_data.RASTER_FITS)  # doctest: +REMOTE_DATA
    SIMPLE  =                    T / Written by IDL:  Sat Aug 20 21:58:03 2022
    BITPIX  =                   16 / Number of bits per data pixel
    NAXIS   =                    0 / Number of data axes
    EXTEND  =                    T / FITS data may contain extensions
    DATE    = '2022-08-21'         / Creation UTC (CCCC-MM-DD) date of FITS header
    ...

    # The individual header corresponding to Si IV 1403 in the raster
    >>> fits.getheader(sample_data.RASTER_FITS, 2)  # doctest: +REMOTE_DATA
    XTENSION= 'IMAGE   '           / IMAGE extension
    BITPIX  =                   16 / Number of bits per data pixel
    NAXIS   =                    3 / Number of data axes
    NAXIS1  =                  512 /
    NAXIS2  =                  548 /
    NAXIS3  =                   16 /
    ...

The same can be done with the data using `astropy.io.fits.getdata`.

As the number of spectral windows in a raster file may vary from one observation to another, a good way to access the last two extensions of the level 2 file is to use a negative index:

.. code-block:: python

    # The header corresponding to the extra information extension
    >>> fits.getheader(sample_data.RASTER_FITS, -1)  # doctest: +REMOTE_DATA
    XTENSION= 'TABLE   '           / ASCII table extension
    BITPIX  =                    8 / 8 bit bytes
    NAXIS   =                    2 / 2-dimensional ASCII table
    NAXIS1  =                  296 / Number of positions along axis 1
    NAXIS2  =                   16 / Number of positions along axis 2
    ...
    >>> # The data for the extra information extension
    >>> data = fits.getdata(sample_data.RASTER_FITS, -1)  # doctest: +REMOTE_DATA
    >>> # The names of the records
    >>> data.dtype.names # doctest: +REMOTE_DATA
    ('FRMID', 'FUVFDBID', 'FUVCRSID', 'NUVFDBID', 'NUVCRSID', 'FUVfilename', 'NUVfilename')

We can access the values of the variables stored in the data corresponding to the extra information extension as either an attribute or a key:

.. code-block:: python

    # An example is the record: "FUVfilename"
    >>> data.FUVfilename # doctest: +REMOTE_DATA
    chararray(['/irisa/data/level1/2021/10/01/H0600/iris20211001_06092534_fuv.fits',
              '/irisa/data/level1/2021/10/01/H0600/iris20211001_06092706_fuv.fits',
              ...
              '/irisa/data/level1/2021/10/01/H0600/iris20211001_06094981_fuv.fits',
              '/irisa/data/level1/2021/10/01/H0600/iris20211001_06095140_fuv.fits'],
              dtype='<U66')

`More information on the level 2 data can be found in ITN 26. <https://iris.lmsal.com/itn26/iris_level2.html>`__
