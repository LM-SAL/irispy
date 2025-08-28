import numpy as np
import pytest

from irispy.io.utils import fitsinfo, read_files


def test_fitsinfo(capsys, sns_sg_file, sns_sji_1330_file, sns_sji_1400_file, sns_sji_2796_file, sns_sji_2832_file):
    fitsinfo(sns_sg_file)
    captured = capsys.readouterr()
    assert sns_sg_file in captured.out

    fitsinfo(sns_sji_1330_file)
    captured = capsys.readouterr()
    assert sns_sji_1330_file in captured.out

    fitsinfo(sns_sji_1400_file)
    captured = capsys.readouterr()
    assert sns_sji_1400_file in captured.out

    fitsinfo(sns_sji_2796_file)
    captured = capsys.readouterr()
    assert sns_sji_2796_file in captured.out

    fitsinfo(sns_sji_2832_file)
    captured = capsys.readouterr()
    assert sns_sji_2832_file in captured.out


def test_read_files_with_mix(sns_sg_file, sns_sji_1330_file):
    returns = read_files([sns_sg_file, sns_sji_1330_file])
    assert len(returns) == 2


def test_read_files_raster(sns_sg_file):
    # Simple test to ensure it does not error
    assert read_files(sns_sg_file)
    assert read_files([sns_sg_file])


def test_read_files_sji(sns_sji_1330_file, sns_sji_1400_file, sns_sji_2796_file, sns_sji_2832_file):
    # Simple test to ensure it does not error
    assert read_files(sns_sji_1330_file)
    assert read_files(sns_sji_1400_file)
    assert read_files(sns_sji_2796_file)
    assert read_files(sns_sji_2832_file)
    assert read_files([sns_sji_2832_file])


def test_read_files_sji_more_than_one(sns_sji_1330_file, sns_sji_1400_file):
    returns = read_files([sns_sji_1330_file, sns_sji_1400_file])
    assert len(returns) == 2


@pytest.mark.remote_data
def test_read_files_raster_scanning(remote_raster_scanning_tar):
    returns = read_files(remote_raster_scanning_tar)
    assert len(returns) == 8  # spectral windows
    np.testing.assert_array_equal(
        returns["C II 1336"].shape, (29, 4, 388, 186)
    )  # 29 time steps, 4 steps, 388 spatial pixels, 186 spectral pixels
    np.testing.assert_array_equal(returns.aligned_dimensions, [29, 4, 388])
