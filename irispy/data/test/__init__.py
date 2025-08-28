"""
This package contains all of irispy's test data.
"""

import os
import re
import fnmatch
from pathlib import Path

from astropy.utils.data import get_pkg_data_filename

import irispy

__all__ = [
    "FILE_LIST",
    "ROOTDIR",
    "get_test_data_filenames",
    "get_test_filepath",
]

ROOTDIR = Path(irispy.__file__).parent / "data" / "test"
FILE_LIST = ROOTDIR.glob("*.[!p]*")


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


def get_test_data_filenames():
    """
    Return a list of all test files in ``data/test`` directory.

    This ignores any ``py``, ``pyc`` and ``__*__`` files in these directories.

    Returns
    -------
    `list`
        The name of all test files in ``data/test`` directory.
    """
    get_test_data_filenames_list = []
    excludes = ["*.pyc", "*" + os.path.sep + "__*__", "*.py"]
    excludes = r"|".join([fnmatch.translate(x) for x in excludes]) or r"$."
    for root, _, root_files in os.walk(ROOTDIR):
        files = [Path(root) / f for f in root_files]
        files = [f for f in files if not re.match(excludes, str(f))]
        get_test_data_filenames_list.extend(files)
    return get_test_data_filenames_list
