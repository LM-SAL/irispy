"""
This package contains all of irispy's test data.
"""

from pathlib import Path

from astropy.utils.data import get_pkg_data_filename

import irispy

__all__ = [
    "ROOTDIR",
    "get_test_filepath",
]

ROOTDIR = Path(irispy.__file__).parent / "data" / "test"


def get_test_filepath(filename, package="irispy.data.test", **kwargs):
    """
    Return the full path to a test file in the ``data/test`` directory.

    Parameters
    ----------
    filename : `str`
        The name of the file inside the ``data/test`` directory.
    package : `str`, optional
        The package in which to look for the file. Defaults to "irispy.data.test".

    Returns
    -------
    filepath : `str`
        The full path to the file.

    Notes
    -----
    This is a wrapper around `astropy.utils.data.get_pkg_data_filename` which
    sets the ``package`` kwarg to be 'sunpy.data.test`.
    """
    if isinstance(filename, Path):
        # NOTE: get_pkg_data_filename does not accept Path objects
        filename = filename.as_posix()
    return get_pkg_data_filename(filename, package=package, **kwargs)
