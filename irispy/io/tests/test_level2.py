import os
import re
import shutil
from pathlib import Path

import pytest

from astropy.io import fits

from irispy.io import read_files, read_sji_lvl2
from irispy.io._level2 import _classify
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.tests.test_mosaic import HEADER as MOSAIC_HEADER

# The cards of a level 3 file made by iris_make_fits_level3: INSTRUME stays "IRIS", the level is 3.
LEVEL3_CARDS = {"INSTRUME": "IRIS", "DATA_LEV": 3.0, "LVL_NUM": 3.0, "OBSID": "3630104144"}


@pytest.fixture
def level3_file(tmp_path):
    filename = tmp_path / "iris_l3_20150917_073915_3630104144_t000_HI6563_im.fits"
    fits.PrimaryHDU(header=fits.Header(LEVEL3_CARDS)).writeto(filename)
    return filename


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ({"INSTRUME": "SJI", "DATA_LEV": 2.0, "LVL_NUM": 2.0}, ("SJI", 2.0)),
        ({"INSTRUME": "AIA_3", "DATA_LEV": 2.0, "LVL_NUM": 2.0}, ("SJI", 2.0)),
        ({"INSTRUME": "SOT-SP", "LVL_NUM": 2.0}, ("SJI", 2.0)),
        ({"INSTRUME": "SPEC", "DATA_LEV": 2.0, "LVL_NUM": 2.0}, ("raster", 2.0)),
        (LEVEL3_CARDS, ("SJI", 3.0)),
        (MOSAIC_HEADER, (None, None)),
        ({"TELESCOP": "SDO/AIA", "INSTRUME": "AIA_3", "LVL_NUM": 1.0}, ("SJI", 1.0)),
        ({}, (None, None)),
    ],
    ids=["sji", "aia", "sot", "raster", "level3", "mosaic", "aia-level1", "empty"],
)
def test_classify(cards, expected):
    assert _classify(fits.Header(cards)) == expected


@pytest.mark.parametrize("kind", ["text file", "level 3 file"])
def test_read_files_rejects_what_is_not_level_2(tmp_path, level3_file, kind):
    path = {"text file": tmp_path / "notes.txt", "level 3 file": level3_file}[kind]
    if kind == "text file":
        path.write_text("not a FITS file")

    with pytest.raises(ValueError, match=re.escape(f"{path} is not an IRIS level 2 file")):
        read_files(path)
    with pytest.raises(ValueError, match="No supported IRIS files were loaded"):
        read_files(path, allow_errors=True)


@pytest.mark.parametrize(
    ("reader", "product", "wrong_fixture"),
    [(read_sji_lvl2, "SJI", "sns_sg_file"), (read_spectrograph_lvl2, "raster", "sns_sji_1330_file")],
    ids=["sji-reader", "spectrograph-reader"],
)
def test_readers_reject_other_products_and_levels(request, level3_file, reader, product, wrong_fixture):
    wrong_file = request.getfixturevalue(wrong_fixture)
    for path in (wrong_file, level3_file):
        message = re.escape(f"{path} is not an IRIS level 2 {product} file")
        with pytest.raises(ValueError, match=message):
            reader(path)
        with fits.open(path) as hdulist, pytest.raises(ValueError, match=message):
            reader(hdulist)


@pytest.mark.parametrize(
    ("reader", "product", "level2_fixture"),
    [(read_sji_lvl2, "SJI", "sns_sji_1400_file"), (read_spectrograph_lvl2, "raster", "sns_sg_file")],
    ids=["sji-reader", "spectrograph-reader"],
)
@pytest.mark.parametrize("kind", ["bytes", "HDUList"])
def test_readers_name_the_data_they_reject_without_a_filename(
    request, level3_file, reader, product, level2_fixture, kind
):
    # Bytes are refused even when they hold a level 2 file
    if kind == "bytes":
        data = Path(request.getfixturevalue(level2_fixture)).read_bytes()
    else:
        data = fits.HDUList.fromstring(level3_file.read_bytes())

    with pytest.raises(ValueError, match=re.escape(f"The data given ({kind}) is not an IRIS level 2 {product} file")):
        reader(data)


def test_read_spectrograph_lvl2_refuses_an_empty_list():
    with pytest.raises(ValueError, match="filenames is empty"):
        read_spectrograph_lvl2([])


def test_read_spectrograph_lvl2_names_a_wrong_file_among_several(sns_sg_file, level3_file):
    with pytest.raises(ValueError, match=re.escape(f"{level3_file} is not an IRIS level 2 raster file")):
        read_spectrograph_lvl2([sns_sg_file, level3_file])


def test_read_files_names_bytes_in_its_type_error(sns_sg_file):
    with pytest.raises(TypeError, match="not 'bytes'"):
        read_files(Path(sns_sg_file).read_bytes())


# read_files reads a folder, so only the single-file readers reject one
@pytest.mark.parametrize(
    ("reader", "product", "kind"),
    [
        (read_files, "file", "text file"),
        (read_sji_lvl2, "SJI file", "directory"),
        (read_sji_lvl2, "SJI file", "text file"),
        (read_spectrograph_lvl2, "raster file", "directory"),
        (read_spectrograph_lvl2, "raster file", "text file"),
    ],
    ids=[
        "read-files-text file",
        "sji-reader-directory",
        "sji-reader-text file",
        "spectrograph-reader-directory",
        "spectrograph-reader-text file",
    ],
)
def test_readers_reject_what_is_not_fits_despite_its_suffix(tmp_path, sns_sg_file, reader, product, kind):
    # Named like FITS, so it is opened before it is classified
    path = tmp_path / "x.fits"
    if kind == "directory":
        path.mkdir()
    else:
        path.write_text("not a FITS file")
    message = re.escape(f"{path} is not an IRIS level 2 {product}")

    with pytest.raises(ValueError, match=message) as excinfo:
        reader(path)
    # The error from opening it is chained
    assert isinstance(excinfo.value.__context__, (OSError, ValueError))
    if reader is read_spectrograph_lvl2:
        # Several files are opened first, to order them by DATE_OBS
        with pytest.raises(ValueError, match=message):
            reader([sns_sg_file, path])


@pytest.mark.parametrize("reader", [read_files, read_sji_lvl2, read_spectrograph_lvl2])
@pytest.mark.parametrize(
    ("kind", "error"),
    [("missing file", FileNotFoundError), ("missing folder", FileNotFoundError), ("unreadable file", PermissionError)],
)
def test_readers_raise_the_os_error_for_a_path_they_cannot_read(tmp_path, sns_sg_file, reader, kind, error):
    path = tmp_path / ("obs" if kind == "missing folder" else Path(sns_sg_file).name)
    if kind == "unreadable file":
        if os.name == "nt" or os.geteuid() == 0:
            pytest.skip("Windows and root read a file without read permission")
        shutil.copy(sns_sg_file, path)
        path.chmod(0)

    with pytest.raises(error):
        reader(path)
    if reader is read_spectrograph_lvl2:
        # Several files are opened first, to order them by DATE_OBS
        with pytest.raises(error):
            reader([sns_sg_file, path])


def test_read_spectrograph_lvl2_names_the_wrong_file_in_a_list(sns_sg_file, sns_sji_1330_file):
    with pytest.raises(ValueError, match=re.escape(f"{sns_sji_1330_file} is not an IRIS level 2 raster file")):
        read_spectrograph_lvl2([sns_sg_file, sns_sji_1330_file])
