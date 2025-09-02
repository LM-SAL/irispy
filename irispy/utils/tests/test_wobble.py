from irispy.utils.wobble import wobble_movie


def test_wobble_movie(fake_long_sns_obs, tmp_path):
    movies = wobble_movie(fake_long_sns_obs, outdir=tmp_path)
    assert movies != []
    movies = wobble_movie(fake_long_sns_obs, outdir=tmp_path, trim=True)
    assert movies != []
