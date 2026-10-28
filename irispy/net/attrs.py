"""
Search attributes for `irispy.net.IRISClient`, available as ``a.iris`` once `irispy.net`
is imported.

They narrow a search with ``a.Instrument("IRIS")`` or ``a.Provider("LMSAL")``.
"""

import astropy.units as u

from sunpy.net.attr import Range, SimpleAttr

__all__ = ["Description", "ObsID", "Product", "Target", "XCen", "YCen"]


class ObsID(SimpleAttr):
    """
    The OBS ID of the observations, such as ``3683602040``.

    Parameters
    ----------
    value : `int` or `str`
    """

    def __init__(self, value):
        super().__init__(int(value))


class Description(SimpleAttr):
    """
    Text in the title, goal or science objectives of the observations, ignoring case.

    For example ``"Hinode"`` finds the observations made with Hinode and ``"sit-and-stare"`` the
    sit-and-stare observations.

    Parameters
    ----------
    value : `str`
    """


class Target(SimpleAttr):
    """
    The target of the observations as the archive labels it.

    For example ``"AR"`` (active region), ``"QS"`` (quiet Sun), ``"CH"`` (coronal hole) or
    ``"FI"`` (filament); the ``Target`` column of a search shows the labels.

    Parameters
    ----------
    value : `str`
    """

    def __init__(self, value):
        # The HCR matches the target case-sensitively: target=ar finds nothing.
        super().__init__(value.upper())


class XCen(Range):
    """
    The solar x of the observations' pointing, from ``xmin`` to ``xmax``.

    Parameters
    ----------
    xmin, xmax : `~astropy.units.Quantity`
        Angles from disk center, positive to the west.
    """

    @u.quantity_input
    def __init__(self, xmin: u.arcsec, xmax: u.arcsec):
        super().__init__(xmin, xmax)

    def collides(self, other):
        return isinstance(other, self.__class__)


class YCen(Range):
    """
    The solar y of the observations' pointing, from ``ymin`` to ``ymax``.

    Parameters
    ----------
    ymin, ymax : `~astropy.units.Quantity`
        Angles from disk center, positive to the north.
    """

    @u.quantity_input
    def __init__(self, ymin: u.arcsec, ymax: u.arcsec):
        super().__init__(ymin, ymax)

    def collides(self, other):
        return isinstance(other, self.__class__)


class Product(SimpleAttr):
    """
    The files to find in place of the raster tarball and slit-jaw files.

    One of ``"raster"``, ``"sji"``, ``"deconvolved"`` (deconvolved slit-jaw files), ``"sdo"``
    (co-aligned AIA cubes) or ``"sot"`` (Hinode/SOT tarballs); combine products with ``|``,
    such as ``Product("raster") | Product("sdo")``.
    `irispy.io.read_files` on a folder that holds both the slit-jaw and the deconvolved files
    gives two cubes for each band.

    Parameters
    ----------
    value : `str`
    """

    def __init__(self, value):
        super().__init__(value.lower())
