from irispy.utils.wobble import generate_wobble_movie


def test_generate_wobble_movie(fake_long_sns_obs, tmp_path):
    movies = generate_wobble_movie(fake_long_sns_obs, outdir=tmp_path)
    assert movies != []
    movies = generate_wobble_movie(fake_long_sns_obs, outdir=tmp_path, trim=True)
    assert movies != []
