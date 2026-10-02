import io
import os
import gzip
import tarfile
from pathlib import Path

import numpy as np
import pytest

from astropy.io import fits

from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.io.utils import _extract_tarfile, _get_spec_group_key, fits_info, read_files


@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize("reader", [read_files, read_sji_lvl2], ids=["read-files", "sji-reader"])
def test_decompresses_sji_once(tmp_path, monkeypatch, memmap, reader):
    source = get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits")
    filename = tmp_path / "sji.fits.gz"
    with Path(source).open("rb") as original, gzip.open(filename, "wb") as compressed:
        compressed.write(original.read())
    expected = read_files(source, memmap=memmap)
    opens = []
    real_open = gzip.GzipFile.__init__

    def tracked_open(self, *args, **kwargs):
        opens.append(1)
        return real_open(self, *args, **kwargs)

    def unexpected_rewind(_self):
        pytest.fail("The gzip stream was rewound and decompressed again")

    monkeypatch.setattr(gzip.GzipFile, "__init__", tracked_open)
    monkeypatch.setattr(gzip._GzipReader, "_rewind", unexpected_rewind)

    cube = reader(filename, memmap=memmap)

    assert opens == [1]
    np.testing.assert_array_equal(cube.data, expected.data)
    np.testing.assert_array_equal(cube.mask, expected.mask)


@pytest.mark.parametrize(
    "file_fixture", ["sns_sg_file", "sns_sji_1330_file", "sns_sji_1400_file", "sns_sji_2796_file", "sns_sji_2832_file"]
)
def test_fits_info(capsys, request, file_fixture):
    filename = request.getfixturevalue(file_fixture)
    fits_info(filename)
    assert filename in capsys.readouterr().out


def test_read_files_with_mix(sns_sg_file, sns_sji_1330_file):
    returns = read_files([sns_sg_file, sns_sji_1330_file])
    assert len(returns) == 2


@pytest.mark.parametrize(
    "file_fixture", ["sns_sg_file", "sns_sji_1330_file", "sns_sji_1400_file", "sns_sji_2796_file", "sns_sji_2832_file"]
)
@pytest.mark.parametrize("as_list", [False, True])
def test_read_files_single_file(request, file_fixture, as_list):
    filename = request.getfixturevalue(file_fixture)
    assert read_files([filename] if as_list else filename)


def test_read_files_raster_file_list(raster_sg_files):
    returns = read_files(raster_sg_files)
    assert set(returns.keys()) == {
        "C II 1336",
        "1343",
        "Fe XII 1349",
        "O I 1356",
        "Si IV 1403",
        "2832",
        "2826",
        "2814",
        "Mg II k 2796",
    }
    assert len(returns["Si IV 1403"]) == len(raster_sg_files)


def test_get_spec_group_key_falls_back_when_metadata_is_missing(monkeypatch, tmp_path):
    filename = tmp_path / "missing-grouping-metadata.fits"
    filename.touch()

    monkeypatch.setattr("irispy.io.utils.fits.getheader", lambda _: {"OBSID": None, "STARTOBS": ""})

    assert _get_spec_group_key(filename) == (filename, None)


def test_get_spec_group_key_falls_back_when_header_read_fails(monkeypatch, tmp_path):
    filename = tmp_path / "unreadable-header.fits"
    filename.touch()

    def raise_oserror(_):
        msg = "cannot read header"
        raise OSError(msg)

    monkeypatch.setattr("irispy.io.utils.fits.getheader", raise_oserror)

    assert _get_spec_group_key(filename) == (filename, None)


def test_read_files_raster_file_list_multiple_observations_use_unique_keys(raster_sg_file, sns_sg_file):
    filenames = sorted([raster_sg_file, sns_sg_file])
    first_describe = fits.getheader(filenames[0]).get("TDESC1")
    second_obsid = fits.getheader(filenames[1]).get("OBSID")

    returns = read_files(filenames)

    assert len(returns) == 2
    assert set(returns.keys()) == {first_describe, f"{first_describe} ({second_obsid})"}


def test_read_files_grouped_spectrograph_honors_allow_errors(
    monkeypatch,
    raster_sg_files,
    sns_sji_1330_file,
    sns_sji_1400_file,
):
    expected = read_files([sns_sji_1330_file, sns_sji_1400_file])

    def raise_value_error(*_args, **_kwargs):
        msg = "grouped spectrograph load failed"
        raise ValueError(msg)

    monkeypatch.setattr("irispy.io.utils.read_spectrograph_lvl2", raise_value_error)

    returns = read_files([sns_sji_1330_file, sns_sji_1400_file, *raster_sg_files], allow_errors=True)

    assert list(returns.keys()) == list(expected.keys())


def test_read_files_sji_more_than_one(sns_sji_1330_file, sns_sji_1400_file):
    returns = read_files([sns_sji_1330_file, sns_sji_1400_file])
    assert len(returns) == 2


def test_read_files_raises_when_no_files_are_supported(tmp_path):
    filename = tmp_path / "not-a-fits.txt"
    filename.write_text("not a supported IRIS file")

    with pytest.raises(ValueError, match="No supported IRIS files were loaded"):
        read_files(filename)


@pytest.mark.remote_data
def test_read_files_raster_scanning(remote_raster_scanning_tar):
    returns = read_files(remote_raster_scanning_tar)
    assert len(returns) == 8  # spectral windows
    np.testing.assert_array_equal(
        returns["C II 1336"].shape, (29, 4, 388, 186)
    )  # 29 time steps, 4 steps, 388 spatial pixels, 186 spectral pixels
    np.testing.assert_array_equal(returns.aligned_dimensions, [29, 4, 388])


def test_extract_tarfile_reuses_complete_extraction(tmp_path):
    tar_path = tmp_path / "obs_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in ("a.fits", "b.fits"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    extract_dir = tmp_path / "obs_raster"
    first = _extract_tarfile([tar_path])
    assert first == [extract_dir / "a.fits", extract_dir / "b.fits"]
    # A second call must not extract again, or it would restore "abc".
    (extract_dir / "a.fits").write_bytes(b"new")
    assert _extract_tarfile([tar_path]) == first
    assert (extract_dir / "a.fits").read_bytes() == b"new"
    # A missing file means the extraction is incomplete, so it is extracted again.
    (extract_dir / "b.fits").unlink()
    assert _extract_tarfile([tar_path]) == first
    assert (extract_dir / "a.fits").read_bytes() == b"abc"
    # A replaced tar file is extracted again, even if its modification time is older.
    old_mtime_ns = tar_path.stat().st_mtime_ns - 10**9
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("a.fits")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"abcd"))
    os.utime(tar_path, ns=(old_mtime_ns, old_mtime_ns))
    assert _extract_tarfile([tar_path]) == [extract_dir / "a.fits"]
    assert (extract_dir / "a.fits").read_bytes() == b"abcd"
