import pytest

from irispy.utils.wobble import generate_wobble_movie


@pytest.mark.filterwarnings("ignore:invalid value encountered in do_format")
def test_generate_wobble_movie(fake_long_sns_obs, tmp_path):
    movies = generate_wobble_movie(fake_long_sns_obs, outdir=tmp_path)
    assert movies != []
    movies = generate_wobble_movie(fake_long_sns_obs, outdir=tmp_path, trim=True)
    assert movies != []
