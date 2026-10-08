import io
import os
import gzip
import tarfile
import threading
from pathlib import Path

import numpy as np
import pytest

from astropy.io import fits

from ndcube import NDCollection

from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.io.utils import _extract_tarfile, _get_spec_group_key, _observation_identity, fits_info, read_files


@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize("reader", [read_files, read_sji_lvl2], ids=["read-files", "sji-reader"])
def test_decompresses_sji_once(tmp_path, monkeypatch, memmap, reader):
    source = get_test_filepath("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits")
    filename = tmp_path / "sji.fits.gz"
    with Path(source).open("rb") as original, gzip.open(filename, "wb") as compressed:
        compressed.write(original.read())
    expected = read_files(source, memmap=memmap)
    expected = next(iter(expected.values()))
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
    if reader is read_files:
        cube = next(iter(cube.values()))

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
    assert isinstance(returns, NDCollection)
    assert len(returns) == fits.getheader(sns_sg_file)["NWIN"] + 1


@pytest.mark.parametrize("file_fixture", ["sns_sg_file", "sns_sji_1330_file"])
def test_read_files_rejects_raw_uncertainty(request, file_fixture):
    filename = request.getfixturevalue(file_fixture)
    with pytest.raises(ValueError, match=r"uncertainty=True.*raw FITS values"):
        read_files(filename, raw=True, uncertainty=True)


@pytest.mark.parametrize(
    "file_fixture", ["sns_sg_file", "sns_sji_1330_file", "sns_sji_1400_file", "sns_sji_2796_file", "sns_sji_2832_file"]
)
@pytest.mark.parametrize("as_list", [False, True])
def test_read_files_single_file(request, file_fixture, as_list):
    filename = request.getfixturevalue(file_fixture)
    result = read_files([filename] if as_list else filename)
    assert isinstance(result, NDCollection)
    assert len(result) == fits.getheader(filename).get("NWIN", 1)


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


def test_observation_identity_uses_source_path_when_observation_metadata_is_missing(tmp_path):
    extracted_file = tmp_path / "extracted" / "product.fits"
    source_path = tmp_path / "observation_raster.tar.gz"

    identity = _observation_identity(extracted_file, source_path=source_path)

    assert identity == ("file", source_path.resolve())


def test_observation_identity_uses_obsid_startobs(monkeypatch, tmp_path):
    filename = tmp_path / "product.fits"
    filename.touch()
    monkeypatch.setattr("irispy.io.utils.fits.getheader", lambda _: {"OBSID": "3620258102", "STARTOBS": "2021-09-05"})

    assert _observation_identity(filename) == ("obs", "3620258102", "2021-09-05")


def test_read_files_rejects_multiple_observations(raster_sg_file, sns_sg_file):
    filenames = sorted([raster_sg_file, sns_sg_file])
    with pytest.raises(ValueError, match="one observation at a time"):
        read_files(filenames)


def test_read_files_sji_keys_do_not_depend_on_unrelated_inputs(sns_sji_1330_file, sns_sg_file):
    single = read_files(sns_sji_1330_file)
    mixed = read_files([sns_sji_1330_file, sns_sg_file])

    assert set(single.keys()).issubset(mixed.keys())


def test_read_files_retains_repeated_products(tmp_path, sns_sji_1330_file):
    duplicate_product = tmp_path / "second_sji_segment.fits"
    duplicate_product.write_bytes(Path(sns_sji_1330_file).read_bytes())

    returns = read_files([sns_sji_1330_file, duplicate_product])

    assert len(returns) == 2
    assert "SJI_1330" in returns  # the primary keeps the bare name
    assert any(key.startswith("SJI_1330:") for key in returns)  # the variant is filename-qualified


def test_read_files_deduplicates_exact_paths(sns_sji_1330_file):
    returns = read_files([sns_sji_1330_file, sns_sji_1330_file])

    assert len(returns) == 1


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
    c_ii = returns["C II 1336"]
    np.testing.assert_array_equal(
        c_ii.shape, (29, 4, 388, 186)
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


def test_extract_tarfile_serializes_concurrent_callers(tmp_path, monkeypatch):
    tar_path = tmp_path / "obs_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("a.fits")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    extracting = threading.Semaphore(0)
    release = threading.Event()
    original_extractall = tarfile.TarFile.extractall

    def paused_extractall(self, *args, **kwargs):
        extracting.release()
        assert release.wait(10), "Extraction was not released"
        original_extractall(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", paused_extractall)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_extract_tarfile([tar_path]))) for _ in range(2)]
    try:
        threads[0].start()
        assert extracting.acquire(timeout=10), "First caller did not start extracting"
        threads[1].start()
        # The second caller must wait until the first has finished extracting.
        assert not extracting.acquire(timeout=1), "Both callers entered extraction"
    finally:
        release.set()
        for thread in threads:
            thread.join(10)

    # The second caller reused the finished extraction instead of extracting again.
    assert not extracting.acquire(blocking=False)
    assert results == [[tmp_path / "obs_raster" / "a.fits"]] * 2
    assert (tmp_path / "obs_raster" / "a.fits").read_bytes() == b"abc"
