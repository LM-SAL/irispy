import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import InverseVariance, StdDevUncertainty, UnknownUncertainty
from astropy.time import Time

from irispy.data.test import get_test_filepath
from irispy.io import read_files
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils._spectral import in_memory, in_windows, make_map_cube, standard_deviation
from irispy.utils.bursts import find_bright_image_events, find_si_iv_bursts
from irispy.utils.cosmic_rays import remove_cosmic_rays
from irispy.utils.dust import calculate_dust_mask, remove_dust
from irispy.utils.fiducials import find_fiducials
from irispy.utils.fitting import line_model, starts_from_model
from irispy.utils.mg_features import calculate_mg_features
from irispy.utils.moments import average_window, calculate_moments
from irispy.utils.red_blue import calculate_red_blue_asymmetry
from irispy.utils.rgb import calculate_rgb
from irispy.utils.spectrograph import radiometric_calibration, subtract_background

SJI = "bursts/iris_l2_20130902_163935_4000255147_SJI_1400_t000_test.fits"
SI_IV = "bursts/iris_l2_20130902_182935_4000005156_raster_t000_r00000_si_iv_test.fits"
MG_II = "mg_features/iris_l2_20130902_182935_4000005156_raster_t000_r00000_mg_features_test.fits"


def test_in_memory(sns_sg_file):
    eager, lazy = (
        read_files(sns_sg_file, memmap=memmap, uncertainty=True)["Si IV 1403"][:, 2:] for memmap in (False, True)
    )
    assert in_memory(eager) is eager
    loaded = in_memory(lazy)
    for array, expected in (
        (loaded.data, eager.data),
        (loaded.mask, eager.mask),
        (loaded.uncertainty.array, eager.uncertainty.array),
    ):
        assert isinstance(array, np.ndarray)
        np.testing.assert_array_equal(array, expected)
    assert loaded.unit == eager.unit


@pytest.mark.parametrize(
    ("filename", "analysis"),
    [
        (SJI, lambda cube: remove_dust(cube).data),
        (SJI, calculate_dust_mask),
        (SJI, lambda cube: pytest.importorskip("rsliding") and remove_cosmic_rays(cube).data),
        (SJI, lambda cube: find_bright_image_events(cube)[0].data),
        (SI_IV, lambda raster: find_si_iv_bursts(raster["Si IV 1403"])[0].data),
        (SI_IV, lambda raster: calculate_moments(raster["Si IV 1403"], velocity_range=None)["centroid"].data),
        (SI_IV, lambda raster: calculate_red_blue_asymmetry(raster["Si IV 1403"])["red_blue_asymmetry"].data),
        (SI_IV, lambda raster: pytest.importorskip("colorsynth") and calculate_rgb(raster["Si IV 1403"][0])[0]),
        (SI_IV, lambda raster: average_window(raster["Si IV 1403"], [1402.7, 1402.9] * u.AA).data),
        (SI_IV, lambda raster: subtract_background(raster["Si IV 1403"], [1402.5, 1402.6] * u.AA).data),
        (SI_IV, lambda raster: radiometric_calibration(raster["Si IV 1403"]).data),
        (MG_II, lambda raster: find_fiducials(raster["Mg II k 2796"])[0]),
        (MG_II, lambda raster: calculate_mg_features(raster["Mg II k 2796"], lines=("k",))["k3_velocity"].data),
        (MG_II, lambda raster: starts_from_model(line_model(["Mg II k"]), raster["Mg II k 2796"][0]).parameters),
        # Stays lazy
        (SI_IV, lambda raster: raster["Si IV 1403"].apply_exposure_time_correction().data),
    ],
    ids=[
        "remove_dust",
        "calculate_dust_mask",
        "remove_cosmic_rays",
        "find_bright_image_events",
        "find_si_iv_bursts",
        "calculate_moments",
        "calculate_red_blue_asymmetry",
        "calculate_rgb",
        "average_window",
        "subtract_background",
        "radiometric_calibration",
        "find_fiducials",
        "calculate_mg_features",
        "starts_from_model",
        "apply_exposure_time_correction",
    ],
)
def test_memmap_read_gives_the_result_of_a_read_into_memory(filename, analysis):
    # Each function reads the lazy arrays of a memmap=True cube into memory first, unless it leaves them lazy
    lazy, eager = (analysis(read_files(get_test_filepath(filename), memmap=memmap)) for memmap in (True, False))
    np.testing.assert_array_equal(lazy, eager)


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
        ("Si IV 1403", lambda cube: {"calibrated": radiometric_calibration(cube)}),
    ],
    ids=["moments", "red_blue", "calibration"],  # Mg II features and the burst finder have their own
)
def test_maps_of_a_window_keep_its_raster_axis(sns_sg_file, window, analysis):
    cube = read_files(sns_sg_file, spectral_windows=window)[window]
    maps, raster_maps = analysis(cube), analysis(cube[0])
    for key, value in raster_maps.items():
        assert maps[key].data.shape == (1, *value.data.shape)
        np.testing.assert_array_equal(maps[key].data[0], value.data)
