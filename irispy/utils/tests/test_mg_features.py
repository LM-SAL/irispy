import copy

import dask.array as da
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.io import readsav

import astropy.units as u

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.spectrograph import SpectrogramCube
from irispy.utils.constants import SATURATION_LIMIT
from irispy.utils.mg_features import _center_vertex, _maxima, _peak_vertex, _peaks, _spline, calculate_mg_features

TEST_FILE = "mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits"
WINDOWS = {"k": "Mg II k 2796", "h": "Mg II h 2803"}
# (step, slit) of the line centers IDL finds from another guess along the slit.
# Its guess spline also goes through the centers it finds in the -200 fill
# (h slit rows 0-7 and 729-770), which changes the spline's tension everywhere
# and its values near row 729. For these spectra the lowest point within 15 grid points
# of the guess is at the edge of that range, so a guess rounded to another grid point
# gives another center (0.019 and 0.157 km/s away). From IDL's guesses the port gives IDL's centers.
# Row 8 is the first good row after the fill, where the port holds its
# guess at the first good center.
OTHER_GUESS = {"k": [], "h": [[0, 112], [2, 8], [2, 727]]}


@pytest.fixture(scope="module")
def raster():
    # Steps 13, 32 and 57 of the file, with its Mg II window split into one window per line.
    return read_spectrograph_lvl2(get_test_filepath(TEST_FILE))


@pytest.fixture(scope="module")
def idl():
    return readsav(get_test_filepath("mg_features/iris_get_mg_features_lev2_4000005156_r00000.sav"))


@pytest.mark.parametrize(("index", "line"), list(enumerate(WINDOWS)))
def test_matches_idl(raster, idl, index, line):
    cube = raster[WINDOWS[line]][0]
    features = calculate_mg_features(cube, lines=(line,))
    missing = (cube.data == -200).any(axis=-1)  # IDL interpolates through the fill
    # Velocity (km/s) and relative intensity tolerances. The line centers differ by at most 1.1e-4 km/s
    # and 1.0e-6. IDL's extrapolated peak spline puts its peaks within 2 grid points (0.27 km/s each) of the
    # highest one and the port's are within half a point, so they differ by less than 0.69 km/s (0.682
    # seen), and by up to 0.049 in intensity.
    for feature, key, velocity, intensity in [
        ("3", "lc", 1e-3, 1e-5),
        ("2v", "bp", 0.7, 0.05),
        ("2r", "rp", 0.7, 0.05),
    ]:
        ours = np.stack([features[f"{line}{feature}_{kind}"].data for kind in ("velocity", "intensity")], axis=-1)
        theirs = idl[key][..., index]
        found = np.isfinite(theirs) & ~missing[..., np.newaxis]
        np.testing.assert_array_equal(np.isfinite(ours), found)
        close = np.isclose(ours[..., 0], theirs[..., 0], rtol=0, atol=velocity)
        close &= np.isclose(ours[..., 1], theirs[..., 1], rtol=intensity, atol=0)
        off = np.argwhere(found[..., 0] & ~close).tolist()
        if key == "lc":
            assert off == OTHER_GUESS[line]
        else:  # the peaks are found around the line center
            assert all(pixel in OTHER_GUESS[line] for pixel in off)


def test_output(raster):
    features = calculate_mg_features(raster["Mg II k 2796"][0], lines=("k",))
    velocity = features["k3_velocity"]
    assert velocity.unit == u.km / u.s
    assert velocity.data.shape == (3, 771)
    np.testing.assert_array_equal(velocity.mask, np.isnan(velocity.data))
    assert features["k3_intensity"].unit == raster["Mg II k 2796"][0].unit
    assert set(features.aligned_axes.values()) == {(0, 1)}


@pytest.fixture(scope="module")
def both_lines(raster):
    # The two windows put back at their place in the file's Mg II window, with zeros between them.
    k, h = raster["Mg II k 2796"][0], raster["Mg II h 2803"][0]
    offset = int(k.wcs.wcs.crpix[0] - h.wcs.wcs.crpix[0])
    data = np.zeros((*k.data.shape[:2], offset + h.data.shape[-1]), dtype=k.data.dtype)
    mask = np.zeros(data.shape, dtype=bool)
    data[..., : k.data.shape[-1]], data[..., offset:] = k.data, h.data
    mask[..., : k.data.shape[-1]], mask[..., offset:] = k.mask, h.mask
    return SpectrogramCube(data, k.wcs, unit=k.unit, mask=mask)


def test_both_lines(raster, both_lines):
    features = calculate_mg_features(both_lines)
    assert list(features.keys()) == [
        f"{line}{feature}_{kind}"
        for line in "kh"
        for feature in ("2v", "3", "2r")
        for kind in ("velocity", "intensity")
    ]
    for line, window in WINDOWS.items():
        for key, value in calculate_mg_features(raster[window][0], lines=(line,)).items():
            np.testing.assert_array_equal(features[key].data, value.data)


def test_saturation_limit(both_lines):
    cube = copy.deepcopy(both_lines)
    cube.data[1, 300, 10] = SATURATION_LIMIT.value  # at -17.0 km/s from k
    cube.data[1, 400, 0] = SATURATION_LIMIT.value  # at -44.3 km/s, not searched
    plain = calculate_mg_features(cube)
    assert np.isfinite(plain["k3_velocity"].data[1, [300, 400]]).all()
    features = calculate_mg_features(cube, saturation_limit=SATURATION_LIMIT)
    assert list(zip(*np.nonzero(features["k_saturated"].data), strict=True)) == [(1, 300)]
    assert not features["h_saturated"].data.any()
    for key, value in features.items():
        if key.endswith("_saturated"):
            continue
        expected = plain[key].data.copy()
        if key.startswith("k"):
            expected[1, 300] = np.nan
        np.testing.assert_array_equal(value.data, expected, err_msg=key)
        np.testing.assert_array_equal(value.mask, np.isnan(expected), err_msg=key)


def test_saturation_limit_ignores_masked_samples(both_lines):
    cube = copy.deepcopy(both_lines)
    cube.data[1, 300, 10] = SATURATION_LIMIT.value
    cube.mask[1, 300, 10] = True
    cube.data[1, 301, 10] = np.inf  # unmasked, so saturated like in calculate_moments, though not finite
    saturated = calculate_mg_features(cube, lines=("k",), saturation_limit=SATURATION_LIMIT)["k_saturated"].data
    assert list(zip(*np.nonzero(saturated), strict=True)) == [(1, 301)]


def test_saturation_limit_per_second(raster):
    # The same rate is 16182 DN in the 4 s step but not in the 1 s ones
    cube = raster["Mg II k 2796"][0].apply_exposure_time_correction()
    cube.meta.add("exposure time", [1, 4, 1] * u.s, None, 0, overwrite=True)
    cube.data[:, 300, 10] = SATURATION_LIMIT.value / 4
    expected = calculate_mg_features(cube, lines=("k",))["k3_velocity"].data
    assert np.isfinite(expected[:, 300]).all()
    expected[1, 300] = np.nan
    for limit in (SATURATION_LIMIT, SATURATION_LIMIT.value):
        velocity = calculate_mg_features(cube, lines=("k",), saturation_limit=limit)["k3_velocity"].data
        np.testing.assert_array_equal(velocity, expected)


def test_dask_data(raster):
    cube = raster["Mg II k 2796"][0]
    lazy = SpectrogramCube(da.from_array(cube.data, chunks=(1, 771, 34)), cube.wcs, unit=cube.unit, mask=cube.mask)
    expected = calculate_mg_features(cube, lines=("k",))
    for key, value in calculate_mg_features(lazy, lines=("k",)).items():
        np.testing.assert_array_equal(value.data, expected[key].data)


def test_velocity_range(raster):
    cube = raster["Mg II k 2796"][0]
    features = calculate_mg_features(cube, lines=("k",), velocity_range=(-30, 30))
    assert np.nanmax(np.abs(calculate_mg_features(cube, lines=("k",))["k2r_velocity"].data)) > 33
    for key in ("k2v_velocity", "k3_velocity", "k2r_velocity"):
        assert np.nanmax(np.abs(features[key].data)) <= 33  # the grid reaches 3 km/s beyond the range


def test_skips_uncovered_line(raster):
    with pytest.warns(UserWarning, match="does not cover Mg II h"):
        features = calculate_mg_features(raster["Mg II k 2796"][0])
    assert {key[0] for key in features} == {"k"}
    with pytest.raises(ValueError, match="does not cover Mg II h"):
        calculate_mg_features(raster["Mg II k 2796"][0], lines=("h",))


@pytest.mark.parametrize("fill", [-200, -199])
def test_missing_data(raster, fill):
    cube = copy.deepcopy(raster["Mg II k 2796"][0])
    cube.data[1, 100, 10] = np.nan
    cube.mask[1, 200, 10] = True  # a finite pixel
    cube.data[1, 300, 10] = fill
    velocity = calculate_mg_features(cube, lines=("k",))["k3_velocity"].data
    assert np.isnan(velocity[1, [100, 200, 300]]).all()
    assert np.isfinite(velocity[1, [99, 199, 299]]).all()


def test_good_centers_far_from_slit_end(raster):
    # Extrapolated far beyond the good line centers, the spline through them would overflow.
    cube = copy.deepcopy(raster["Mg II k 2796"][0])
    cube.data[:, 20:] = np.linspace(50, 100, cube.data.shape[-1])  # featureless
    cube.mask[:, 20:] = False
    velocity = calculate_mg_features(cube, lines=("k",))["k3_velocity"].data  # warnings are errors
    assert np.isfinite(velocity[:, :20]).any()
    assert np.isnan(velocity[:, 20:]).all()


def test_short_slit(raster):
    # As in IDL, a slit shorter than the 17-pixel kernel is not smoothed, so no line center jumps.
    cube = copy.deepcopy(raster["Mg II k 2796"][0][:, 100:116])
    cube.data[:, 8, 2:] = cube.data[:, 8, :-2]  # its line center moves 5.5 km/s redward
    velocity = calculate_mg_features(cube, lines=("k",))["k3_velocity"].data
    alone = calculate_mg_features(cube[:, 8:9], lines=("k",))["k3_velocity"].data
    np.testing.assert_array_equal(velocity[:, 8:9], alone)


def test_rejects_unscaled_data():
    raster = read_spectrograph_lvl2(get_test_filepath(TEST_FILE), memmap=True)
    with pytest.raises(ValueError, match="unscaled"):
        calculate_mg_features(raster["Mg II k 2796"][0], lines=("k",))


def test_rejects_other_input(raster):
    with pytest.raises(TypeError, match="index it"):
        calculate_mg_features(raster["Mg II k 2796"])
    with pytest.raises(ValueError, match="slice it with ranges"):
        calculate_mg_features(raster["Mg II k 2796"][0][:, 300])
    cube = raster["Mg II k 2796"][0]
    wcs = cube.wcs.deepcopy()
    wcs.wcs.cdelt[0] *= -1
    with pytest.raises(ValueError, match="wavelengths must increase"):
        calculate_mg_features(SpectrogramCube(cube.data[..., ::-1], wcs, unit=cube.unit), lines=("k",))


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"velocity_range": (40, -40)}, "two increasing velocities"),
        ({"velocity_range": 40 * u.km / u.s}, "two increasing velocities"),
        ({"velocity_range": (-0.1, 0.1)}, "Too few wavelength points of Mg II k"),
        ({"lines": ()}, "lines must be"),
        ({"lines": ("K",)}, "lines must be"),
        ({"lines": ("k", "k")}, "lines must be"),
    ],
)
def test_rejects_bad_arguments(raster, keywords, match):
    with pytest.raises(ValueError, match=match):
        calculate_mg_features(raster["Mg II k 2796"][0], **{"lines": ("k",), **keywords})


def test_maxima():
    # As in lclxtrem.pro, a maximum is dropped only near a kept one of larger absolute value.
    x = np.arange(40)
    bumps = 3 * np.exp(-((x - 5) ** 2) / 2), 2 * np.exp(-((x - 13) ** 2) / 2), np.exp(-((x - 21) ** 2) / 2)
    np.testing.assert_array_equal(np.flatnonzero(_maxima(sum(bumps)[np.newaxis])), [5, 21])
    # With no turning point, the highest value counts.
    np.testing.assert_array_equal(np.flatnonzero(_maxima(x[np.newaxis] * 1.0)), [39])


def test_parabola_vertices():
    grid = np.linspace(-10, 10, 201)
    # Vertices between grid points, the second next to the grid's start, where only the 5 points on the
    # grid are fitted. The asymmetric term makes the fit depend on which points are used.
    vertices = np.array([0.237, -9.863])
    offset = grid - vertices[:, np.newaxis]
    spectra = 2 + 3 * offset**2 + offset**3 * np.exp(-(offset**2))
    minimum = _center_vertex(grid, spectra, vertices, np.array([15, 15]))
    lowest = np.argmin(spectra, axis=-1)
    for row, points in enumerate([np.arange(lowest[0] - 3, lowest[0] + 4), np.arange(lowest[1] + 4)]):
        a2, a1, a0 = np.polyfit(grid[points] - grid[lowest[row]], spectra[row, points], 2)
        vertex = -a1 / (2 * a2)
        np.testing.assert_allclose(minimum[row], [grid[lowest[row]] + vertex, a0 - a2 * vertex**2], atol=1e-10)
    spectra = np.tile(5 - 2 * (grid - 0.237) ** 2, (2, 1))
    maximum = _peak_vertex(grid, spectra, np.full((2, 4), np.argmax(spectra[0])), np.array([0, -1]))
    np.testing.assert_allclose(maximum, [[0.237, 5], [np.nan, np.nan]], atol=1e-10)


def test_peaks():
    grid = np.linspace(-40, 40, 161)
    spectra = np.zeros((3, grid.size))
    maxima = np.zeros(spectra.shape, dtype=bool)
    # Three maxima with the center, 15 km/s, outside both gaps, of which the weakest is dropped; one
    # maximum redward of the center, 0 km/s; and none.
    for row, velocities, heights in [(0, [-20, -10, 10], [3, 1, 2]), (1, [10], [1])]:
        index = np.searchsorted(grid, velocities)
        spectra[row, index], maxima[row, index] = heights, True
    blue, red = _peaks(grid, spectra, maxima, np.array([15.0, 0.0, 0.0]))
    np.testing.assert_array_equal(blue, [[-20, 3], [np.nan, np.nan], [np.nan, np.nan]])
    np.testing.assert_array_equal(red, [[10, 2], [10, 1], [np.nan, np.nan]])


@pytest.fixture(scope="module")
def idl_spline():
    return readsav(get_test_filepath("mg_features/idl_spline.sav"))


def test_spline_matches_idl(idl_spline):
    for tension in (0, 1):
        spline = _spline(idl_spline["x"], idl_spline["y"], idl_spline["t"], tension=tension)
        np.testing.assert_allclose(spline, idl_spline[f"tension_{tension}"], atol=1e-8)


def test_spline_uneven_knots(idl_spline):
    # At tension 0 the spline is the cubic spline with the slopes of the parabolas through the three
    # knots at each end. IDL's SPLINE uses the first interval's diagonal in the last row of its system,
    # which is right only for even knots, and here gives values up to 2.3 away.
    x, y, t = (idl_spline[f"uneven_{name}"] for name in "xyt")
    spline = _spline(x, y, t, tension=0)
    np.testing.assert_allclose(spline, CubicSpline(x, y, bc_type=((1, 2), (1, 7 / 6)))(t), atol=1e-6)
    assert np.abs(spline - idl_spline["uneven_tension_0"]).max() > 2
    assert np.abs(_spline(x, y, t, tension=1) - idl_spline["uneven_tension_1"]).max() > 1
