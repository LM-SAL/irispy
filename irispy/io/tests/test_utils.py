import io
import os
import re
import gzip
import shutil
import hashlib
import tarfile
import threading
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

from astropy.config.paths import temporary_cache_dir_path
from astropy.io import fits
from astropy.io.fits import file as fits_file

from ndcube import NDCollection

from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.utils import _extract_tarfile, _get_spec_group_key, fits_info, read_files
from irispy.sji import SOTCube


def _extract_target(root, tar_path):
    # The folder a tar file is extracted into: its name and the first 8 hex digits of its path's SHA-1
    digest = hashlib.sha1(os.fsencode(Path(tar_path).resolve()), usedforsecurity=False).hexdigest()[:8]
    return Path(root) / f"{Path(tar_path).with_suffix('').with_suffix('').name}-{digest}"


def _read_opened_raster(filename, memmap):
    # As a caller who opened the file as the docstring says, and keeps it open
    with fits.open(filename, do_not_scale_image_data=memmap, decompress_in_memory=True) as hdulist:
        raster = read_spectrograph_lvl2(hdulist, memmap=memmap)
        assert not hdulist.fileinfo(0)["file"].closed
    return raster


@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize(
    ("source", "reader"),
    [
        ("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits", read_files),
        ("bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits", read_sji_lvl2),
        (
            "raster/iris_l2_20140329_140938_3860258481_raster/iris_l2_20140329_140938_3860258481_raster_t000_r00000_test.fits",
            read_files,
        ),
        (
            "raster/iris_l2_20140329_140938_3860258481_raster/iris_l2_20140329_140938_3860258481_raster_t000_r00000_test.fits",
            read_spectrograph_lvl2,
        ),
        # Scaled integers, as most rasters are
        ("v34/iris_l2_20260206_210853_3460104433_raster_t000_r00000_test.fits", _read_opened_raster),
    ],
    ids=["sji-read-files", "sji-reader", "raster-read-files", "raster-reader", "raster-opened"],
)
def test_decompresses_file_once(tmp_path, monkeypatch, memmap, source, reader):
    source = get_test_filepath(source)
    filename = tmp_path / "compressed.fits.gz"
    with Path(source).open("rb") as original, gzip.open(filename, "wb") as compressed:
        compressed.write(original.read())

    def first_cube(result):
        return result["C II 1336"][0] if isinstance(result, NDCollection) else result

    expected = first_cube(read_files(source, memmap=memmap))
    inflated = []
    real_read = gzip._GzipReader.read

    def tracked_read(self, size=-1):
        chunk = real_read(self, size)
        inflated.append(len(chunk))
        return chunk

    def unexpected_rewind(_self):
        pytest.fail("The gzip stream was rewound and decompressed again")

    monkeypatch.setattr(gzip._GzipReader, "read", tracked_read)
    monkeypatch.setattr(gzip._GzipReader, "_rewind", unexpected_rewind)

    cube = first_cube(reader(filename, memmap=memmap))

    # Inflated once for the data; read_files reads the primary header first, not the whole file.
    assert sum(inflated) < 2 * Path(source).stat().st_size
    np.testing.assert_array_equal(cube.data, expected.data)
    np.testing.assert_array_equal(cube.mask, expected.mask)
    assert cube.unit == expected.unit


@pytest.mark.parametrize(
    "file_fixture", ["sns_sg_file", "sns_sji_1330_file", "sns_sji_1400_file", "sns_sji_2796_file", "sns_sji_2832_file"]
)
def test_fits_info(capsys, request, monkeypatch, file_fixture):
    filename = request.getfixturevalue(file_fixture)
    opens = []
    real_init = fits_file._File.__init__

    def tracked_init(self, *args, **kwargs):
        opens.append(1)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(fits_file._File, "__init__", tracked_init)

    fits_info(filename)

    assert filename in capsys.readouterr().out
    assert opens == [1]


@pytest.mark.parametrize(
    ("file_fixture", "description"),
    [("aia_1700_file", "1700"), ("sot_fg_file", "G band 4305"), ("sot_sp_file", "6302A")],
)
def test_fits_info_labels_the_image_of_aia_and_sot_files(capsys, request, file_fixture, description):
    fits_info(request.getfixturevalue(file_fixture))

    image, auxiliary = capsys.readouterr().out.splitlines()[-2:]
    assert image.endswith(f" {description}")
    assert auxiliary.endswith(" Auxiliary data")


def test_fits_info_does_not_truncate_to_the_terminal_width(capsys, monkeypatch, sns_sji_1330_file):
    monkeypatch.setenv("COLUMNS", "40")

    fits_info(sns_sji_1330_file)

    assert "..." not in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"revert_v34": True}, {"revert_v34": True, "allow_errors": True}],
    ids=["defaults", "revert-v34", "revert-v34-allow-errors"],
)
def test_read_files_with_mix(tmp_path, sns_sg_file, sns_sji_1330_file, kwargs):
    # No test raster is a V34 observation (STEPS_AV < -0.01), so mark a copy as one.
    sg_file = tmp_path / "v34_raster.fits"
    with fits.open(sns_sg_file, memmap=False) as hdulist:
        hdulist[0].header["STEPS_AV"] = -1.0
        hdulist.writeto(sg_file)

    returns = read_files([sg_file, sns_sji_1330_file], spectral_windows="C II 1336", **kwargs)

    assert set(returns.keys()) == {"raster", "SJI_1330"}
    # revert_v34 goes to the raster reader only; the slit-jaw reader does not take it.
    revert_v34 = kwargs.get("revert_v34", False)
    cube = returns["raster"]["C II 1336"][0]
    reference = read_spectrograph_lvl2(sg_file, spectral_windows="C II 1336", revert_v34=revert_v34)["C II 1336"][0]
    other = read_spectrograph_lvl2(sg_file, spectral_windows="C II 1336", revert_v34=not revert_v34)["C II 1336"][0]

    np.testing.assert_array_equal(cube.data, reference.data)
    np.testing.assert_array_equal(cube.time.jd, reference.time.jd)
    assert not np.array_equal(cube.data, other.data)
    np.testing.assert_array_equal(cube.data, other.data[::-1])
    np.testing.assert_array_equal(cube.time.jd, other.time.jd[::-1])


@pytest.mark.parametrize(
    "file_fixture", ["sns_sg_file", "sns_sji_1330_file", "sns_sji_1400_file", "sns_sji_2796_file", "sns_sji_2832_file"]
)
@pytest.mark.parametrize("as_list", [False, True])
def test_read_files_single_file(request, file_fixture, as_list):
    filename = request.getfixturevalue(file_fixture)
    assert read_files([filename] if as_list else filename)


def test_read_files_accepts_mixed_types_and_reads_duplicates_once(
    tmp_path, monkeypatch, sns_sji_1330_file, sns_sji_1400_file
):
    # The same file as a str, a Path, a relative path and a symlink is read once.
    link = tmp_path / "link.fits"
    link.symlink_to(sns_sji_1330_file)
    monkeypatch.chdir(Path(sns_sji_1330_file).parent)
    filenames = [Path(sns_sji_1330_file), sns_sji_1330_file, Path(sns_sji_1330_file).name, link, sns_sji_1400_file]

    returns = read_files(filenames)

    assert set(returns.keys()) == {"SJI_1330", "SJI_1400"}


def test_read_files_folder(tmp_path, sns_sg_file, sns_sji_1330_file, sns_sji_1400_file):
    folder = tmp_path / "obs"
    folder.mkdir()
    with tarfile.open(folder / "iris_l2_20210905_001833_3620258102_raster.tar.gz", "w:gz", compresslevel=1) as tar:
        tar.add(sns_sg_file, arcname=Path(sns_sg_file).name)
    for filename in (sns_sji_1330_file, sns_sji_1400_file):
        (folder / Path(filename).name).symlink_to(filename)
    # An archive download folder also holds the quick-look movies
    (folder / "iris_l2_20210905_001833_3620258102_movies.tar").touch()

    returns = read_files(folder, extract_dir=tmp_path / "extracted")

    assert set(returns.keys()) == {"raster", "SJI_1330", "SJI_1400"}


def test_read_files_folder_of_aia_and_sot_files(tmp_path, aia_1700_file, sot_fg_file, sot_mg_file, sot_sp_file):
    # The files extracted from the SDO, SOTFG and SOTSP tarballs
    for filename in (aia_1700_file, sot_fg_file, sot_mg_file, sot_sp_file):
        (tmp_path / Path(filename).name).symlink_to(filename)

    returns = read_files(tmp_path)

    assert set(returns.keys()) == {"1700", "G band 4305", "TF Na I 5896", "6302A B_LOS"}


def test_read_files_names_a_folder_with_no_iris_files(tmp_path):
    (tmp_path / "notes.fits").touch()

    with pytest.raises(ValueError, match=re.escape(f"No supported IRIS files were loaded from {tmp_path}.")):
        read_files(tmp_path)


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
    si_iv = returns["Si IV 1403"]
    assert si_iv.shape == (len(raster_sg_files), 8, 109, 29)


def test_get_spec_group_key_falls_back_when_metadata_is_missing():
    filename = Path("missing-grouping-metadata.fits")

    assert _get_spec_group_key(filename, {"OBSID": None, "STARTOBS": ""}) == (filename, None)


@pytest.mark.parametrize("source", ["files", "tarballs"])
def test_read_files_raster_file_list_multiple_observations_use_unique_keys(
    tmp_path, raster_sg_file, sns_sg_file, source
):
    filenames = sorted([raster_sg_file, sns_sg_file])
    second_obsid = fits.getheader(filenames[1]).get("OBSID")
    if source == "tarballs":
        tarballs = [tmp_path / f"{i}_raster.tar.gz" for i in range(len(filenames))]
        for tar_path, filename in zip(tarballs, filenames, strict=True):
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(filename, arcname=Path(filename).name)
        filenames = tarballs

    returns = read_files(filenames)

    assert len(returns) == 2
    assert set(returns.keys()) == {"raster", f"raster ({second_obsid})"}


def test_read_files_opens_each_spectrograph_file_at_most_twice(monkeypatch, raster_sg_files, sns_sg_file):
    # Two observations, so the second needs its OBSID in the key.
    filenames = [*raster_sg_files, sns_sg_file]
    opens = []
    real_init = fits_file._File.__init__

    def tracked_init(self, fileobj=None, *args, **kwargs):
        opens.append(Path(fileobj).name)
        return real_init(self, fileobj, *args, **kwargs)

    monkeypatch.setattr(fits_file._File, "__init__", tracked_init)

    assert len(read_files(filenames)) == 2
    # The header that sorts the files into observations is read without opening a FITS file; the
    # reader opens each file once, and once more in an observation of several files to order them.
    expected = {Path(filename).name: 2 for filename in raster_sg_files}
    expected[Path(sns_sg_file).name] = 1
    assert Counter(opens) == expected


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


def test_read_files_skips_a_bad_tar_member_with_allow_errors(tmp_path, raster_sg_files):
    tar_path = tmp_path / "obs_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for filename in raster_sg_files:
            tar.add(filename, arcname=Path(filename).name)
        info = tarfile.TarInfo("corrupt.fits")  # sorts before the rasters
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))

    with pytest.raises(ValueError, match=r"corrupt\.fits is not an IRIS level 2 file"):
        read_files(tar_path)
    returns = read_files(tar_path, allow_errors=True)

    # The other members still load
    assert returns["Si IV 1403"].shape[0] == len(raster_sg_files)


def test_read_files_sji_more_than_one(sns_sji_1330_file, sns_sji_1400_file):
    returns = read_files([sns_sji_1330_file, sns_sji_1400_file])
    assert len(returns) == 2


def test_read_files_sot(tmp_path, sot_fg_file, sot_mg_file, sot_sp_file):
    # A SOT-SP tar file can hold more than one map of a quantity
    tar_path = tmp_path / "iris_l2_20160108_191211_3680100932_SOTSP.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(sot_sp_file, arcname="sotsp_a.fits")
        tar.add(sot_sp_file, arcname="sotsp_b.fits")

    returns = read_files([sot_fg_file, sot_mg_file, str(tar_path)], extract_dir=tmp_path / "extracted")

    assert set(returns.keys()) == {"G band 4305", "TF Na I 5896", "6302A B_LOS", "6302A B_LOS (sotsp_b)"}
    assert all(isinstance(cube, SOTCube) for cube in returns.values())
    assert (_extract_target(tmp_path / "extracted", tar_path) / "sotsp_b.fits").is_file()


@pytest.mark.parametrize("source", ["files", "archive"], ids=["same-stem-files", "same-stem-members"])
def test_read_files_sot_repeated_names(tmp_path, sot_sp_file, source):
    # Four files with the same stem in different directories
    if source == "archive":
        tar_path = tmp_path / "obs_SOTSP.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for i in range(4):
                tar.add(sot_sp_file, arcname=f"{i}/sotsp.fits")
        filenames = [tar_path]
    else:
        filenames = []
        for i in range(4):
            (tmp_path / str(i)).mkdir()
            filenames.append(shutil.copy(sot_sp_file, tmp_path / str(i) / "sotsp.fits"))

    returns = read_files(filenames)

    assert len(returns) == 4


def test_read_files_raster_tarball_matches_the_loose_files(tmp_path, raster_sg_files):
    tar_path = tmp_path / "iris_l2_20140329_140938_3860258481_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for filename in raster_sg_files:
            tar.add(filename, arcname=Path(filename).name)
    expected = read_files(raster_sg_files)

    returns = read_files(tar_path)

    assert list(returns.keys()) == list(expected.keys())
    np.testing.assert_array_equal(returns.aligned_dimensions, expected.aligned_dimensions)
    for key in expected:
        cube, reference = returns[key], expected[key]
        assert cube.shape[0] == reference.shape[0] == len(raster_sg_files)
        np.testing.assert_array_equal(cube.data, reference.data)
        np.testing.assert_array_equal(cube.mask, reference.mask)
        assert cube.meta["DATE_OBS"] == reference.meta["DATE_OBS"]
        assert (cube.time == reference.time).all()


@pytest.mark.remote_data
def test_read_files_raster_scanning(remote_raster_scanning_tar):
    returns = read_files(remote_raster_scanning_tar)
    assert len(returns) == 8  # spectral windows
    np.testing.assert_array_equal(
        returns["C II 1336"].shape, (29, 4, 388, 186)
    )  # 29 rasters, 4 scan steps, 388 spatial pixels, 186 spectral pixels
    np.testing.assert_array_equal(returns.aligned_dimensions, [29, 4, 388])


def test_extract_tarfile_reuses_complete_extraction(tmp_path):
    tar_path = tmp_path / "obs_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in ("a.fits", "b.fits"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    extract_dir = _extract_target(tmp_path, tar_path)
    first = _extract_tarfile([tar_path], extract_dir=tmp_path)
    assert first == [extract_dir / "a.fits", extract_dir / "b.fits"]
    # A second call must not extract again, or it would restore "abc".
    (extract_dir / "a.fits").write_bytes(b"new")
    assert _extract_tarfile([tar_path], extract_dir=tmp_path) == first
    assert (extract_dir / "a.fits").read_bytes() == b"new"
    # A missing file means the extraction is incomplete, so it is extracted again.
    (extract_dir / "b.fits").unlink()
    assert _extract_tarfile([tar_path], extract_dir=tmp_path) == first
    assert (extract_dir / "a.fits").read_bytes() == b"abc"
    # So does a truncated file.
    (extract_dir / "b.fits").write_bytes(b"ab")
    assert _extract_tarfile([tar_path], extract_dir=tmp_path) == first
    assert (extract_dir / "b.fits").read_bytes() == b"abc"
    # A replaced tar file is extracted again, even if its modification time is older.
    old_mtime_ns = tar_path.stat().st_mtime_ns - 10**9
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("a.fits")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"abcd"))
    os.utime(tar_path, ns=(old_mtime_ns, old_mtime_ns))
    assert _extract_tarfile([tar_path], extract_dir=tmp_path) == [extract_dir / "a.fits"]
    assert (extract_dir / "a.fits").read_bytes() == b"abcd"


def test_extract_tarfile_strips_the_leading_slash_of_a_member_name(tmp_path):
    tar_path = tmp_path / "obs_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("/a.fits")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    extracted = _extract_target(tmp_path, tar_path) / "a.fits"

    files = _extract_tarfile([tar_path], extract_dir=tmp_path)

    # extractall(filter="data") strips the slash, so the file is in the folder, not at /a.fits
    assert files == [extracted]
    assert extracted.read_bytes() == b"abc"
    # and the extraction is found complete and reused
    extracted.write_bytes(b"new")
    assert _extract_tarfile([tar_path], extract_dir=tmp_path) == [extracted]
    assert extracted.read_bytes() == b"new"


def test_extract_tarfile_keeps_same_named_archives_apart(tmp_path, raster_sg_file, sns_sg_file):
    # The same archive name in two folders, such as two downloads of one observation, with the same member name
    tarballs = []
    for folder, filename in zip(("first", "second"), (raster_sg_file, sns_sg_file), strict=True):
        (tmp_path / folder).mkdir()
        tarballs.append(tmp_path / folder / "obs_raster.tar.gz")
        with tarfile.open(tarballs[-1], "w:gz") as tar:
            tar.add(filename, arcname="raster.fits")
    expected = {fits.getheader(raster_sg_file)["OBSID"], fits.getheader(sns_sg_file)["OBSID"]}

    extracted = _extract_tarfile(tarballs, extract_dir=tmp_path / "extracted")
    returns = read_files(tarballs, extract_dir=tmp_path / "extracted")

    assert [file.parent for file in extracted] == [_extract_target(tmp_path / "extracted", tar) for tar in tarballs]
    assert len(returns) == 2
    assert {raster["C II 1336"][0].meta["OBSID"] for raster in returns.values()} == expected


def test_extract_tarfile_defaults_to_the_irispy_cache(tmp_path):
    # The archive's folder can be read-only, such as a download folder on a shared mount.
    archive_dir = tmp_path / "downloads"
    archive_dir.mkdir()
    tar_path = archive_dir / "obs_raster.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("a.fits")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    archive_dir.chmod(0o555)
    try:
        with temporary_cache_dir_path(tmp_path, namespace="irispy") as cache:
            files = _extract_tarfile([tar_path])
            assert files == [_extract_target(cache, tar_path) / "a.fits"]
            assert files[0].read_bytes() == b"abc"
    finally:
        archive_dir.chmod(0o755)
    assert list(archive_dir.iterdir()) == [tar_path]


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
    threads = [
        threading.Thread(target=lambda: results.append(_extract_tarfile([tar_path], extract_dir=tmp_path)))
        for _ in range(2)
    ]
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
    assert results == [[_extract_target(tmp_path, tar_path) / "a.fits"]] * 2
    assert (_extract_target(tmp_path, tar_path) / "a.fits").read_bytes() == b"abc"


def test_read_files_reads_the_windows_of_a_generator_from_every_observation(sns_sg_file, raster_sg_file):
    returns = read_files([sns_sg_file, raster_sg_file], spectral_windows=(w for w in ["Mg II k 2796"]))
    assert [list(raster.keys()) for raster in returns.values()] == [["Mg II k 2796"], ["Mg II k 2796"]]
