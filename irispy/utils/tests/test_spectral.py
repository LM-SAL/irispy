import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import InverseVariance, StdDevUncertainty, UnknownUncertainty
from astropy.time import Time

from irispy.io import read_files
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils._spectral import check_scaled, in_windows, make_map_cube, standard_deviation
from irispy.utils.bursts import find_si_iv_bursts
from irispy.utils.constants import DN_UNIT
from irispy.utils.mg_features import calculate_mg_features
from irispy.utils.moments import calculate_moments
from irispy.utils.red_blue import calculate_red_blue_asymmetry
from irispy.utils.spectrograph import radiometric_calibration


def test_check_scaled():
    wavelengths = np.arange(3) * u.nm
    # Integers in DN, as AIA cubes are stored
    check_scaled(make_test_spectrogram_cube(np.ones((1, 1, 3), dtype=np.int16), wavelengths))
    for band in ("SJI", "FUV", "NUV"):
        for unit in (DN_UNIT[f"{band}_UNSCALED"], DN_UNIT[f"{band}_UNSCALED"] / u.s):
            with pytest.raises(ValueError, match="unscaled"):
                check_scaled(make_test_spectrogram_cube(np.ones((1, 1, 3)), wavelengths, unit=unit))
        # Converting the raw integers to counts would lose the unit that marks them
        assert not DN_UNIT[f"{band}_UNSCALED"].is_equivalent(u.ct)


@pytest.mark.parametrize("windows", [[[1, 2], [10, 11]], [[10, 11], [1, 2]]])
def test_in_windows_ignores_unavailable_windows(windows):
    wavelengths = np.arange(5) * u.nm
    np.testing.assert_array_equal(in_windows(wavelengths, u.Quantity(windows, u.nm)), [False, True, True, False, False])


def test_in_windows_rejects_empty_union():
    with pytest.raises(ValueError, match="No wavelengths"):
        in_windows(np.arange(5) * u.nm, [[10, 11], [20, 21]] * u.nm)


@pytest.mark.parametrize(
    "uncertainty",
    [
        None,
        StdDevUncertainty(2.0),
        UnknownUncertainty(np.full((1, 1, 3), 2.0)),
        InverseVariance(np.full((1, 1, 3), 0.25)),
        StdDevUncertainty(np.full((1, 1, 3), 2000.0), unit=u.DN / 1000),
    ],
)
def test_standard_deviation(uncertainty):
    cube = make_test_spectrogram_cube(np.ones((1, 1, 3)), np.arange(3) * u.nm, uncertainty=uncertainty)
    sigma = standard_deviation(cube)
    if uncertainty is None:
        assert sigma is None
    else:
        np.testing.assert_allclose(sigma, np.full((1, 1, 3), 2.0))


def test_make_map_cube_preserves_independent_coordinates():
    cube = make_test_spectrogram_cube(np.ones((3, 4, 2)), [1402.77, 1402.80] * u.AA)
    times = Time("2026-10-07") + np.arange(3) * u.s
    cube.extra_coords.add("time", 0, times, physical_types="time")
    template = cube[..., 0]
    template.global_coords.add("reference_time", "time", times[0])
    values = np.ones(template.shape)
    maps = [make_map_cube(template, values, u.km / u.s) for _ in range(2)]
    for output in maps:
        assert output.wcs is template.wcs
        assert output.global_coords["reference_time"] == times[0]
        output_times = output.axis_world_coords("time", wcs=output.extra_coords)[0]
        np.testing.assert_allclose(output_times.unix, times.unix, rtol=0, atol=1e-6)
        sliced = output[1:, :]
        sliced_times = sliced.axis_world_coords("time", wcs=sliced.extra_coords)[0]
        np.testing.assert_allclose(sliced_times.unix, times[1:].unix, rtol=0, atol=1e-6)

    maps[0].extra_coords.add("exposure", 0, np.ones(3) * u.s, physical_types="time.duration")
    maps[0].global_coords.remove("reference_time")
    for unchanged in (template, maps[1]):
        assert unchanged.extra_coords.keys() == ("time",)
        assert unchanged.global_coords["reference_time"] == times[0]


@pytest.mark.parametrize(
    ("window", "analysis"),
    [
        ("Si IV 1403", lambda cube: calculate_moments(cube, velocity_range=(-50, 50) * u.km / u.s)),
        ("Si IV 1403", lambda cube: calculate_red_blue_asymmetry(cube, velocity_range=(25, 75) * u.km / u.s)),
        ("Mg II k 2796", lambda cube: calculate_mg_features(cube, lines=("k",))),
        ("Si IV 1403", lambda cube: {"labels": find_si_iv_bursts(cube, threshold=80 * u.DN / u.s)[0]}),
        ("Si IV 1403", lambda cube: {"calibrated": radiometric_calibration(cube)}),
    ],
    ids=["moments", "red_blue", "mg_features", "bursts", "calibration"],
)
def test_maps_of_a_window_keep_its_raster_axis(sns_sg_file, window, analysis):
    cube = read_files(sns_sg_file, spectral_windows=window)[window]
    maps, raster_maps = analysis(cube), analysis(cube[0])
    for key, value in raster_maps.items():
        assert maps[key].data.shape == (1, *value.data.shape)
        np.testing.assert_array_equal(maps[key].data[0], value.data)
