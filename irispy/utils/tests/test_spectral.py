import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import InverseVariance, StdDevUncertainty, UnknownUncertainty, VarianceUncertainty
from astropy.time import Time

from ndcube import NDCube

from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils._spectral import check_scaled, in_windows, make_map_cube, make_spatial_template, standard_deviation


def test_check_scaled():
    wavelengths = np.arange(3) * u.nm
    check_scaled(make_test_spectrogram_cube(np.ones((1, 1, 3)), wavelengths))
    integer = make_test_spectrogram_cube(np.ones((1, 1, 3), dtype=np.int16), wavelengths)
    check_scaled(integer)
    integer.meta["scaled"] = False
    with pytest.raises(ValueError, match="unscaled"):
        check_scaled(integer)
    integer.meta["scaled"] = True  # as the slit-jaw reader gives scaled AIA data
    check_scaled(integer)


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
        InverseVariance(np.full((1, 1, 3), 2.5e-7), unit=(u.DN / 1000) ** -2),
        VarianceUncertainty(np.full((1, 1, 3), 4_000_000), unit=(u.DN / 1000) ** 2),
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


def test_standard_deviation_rejects_unknown_uncertainty():
    cube = make_test_spectrogram_cube(
        np.ones((1, 1, 3)), np.arange(3) * u.nm, uncertainty=UnknownUncertainty(np.full((1, 1, 3), 2.0))
    )
    with pytest.raises(TypeError, match="UnknownUncertainty has no defined interpretation"):
        standard_deviation(cube)


def test_make_map_cube_preserves_independent_coordinates():
    cube = make_test_spectrogram_cube(np.ones((3, 4, 2)), [1402.77, 1402.80] * u.AA)
    times = Time("2026-10-07") + np.arange(3) * u.s
    cube.extra_coords.add("time", 0, times, physical_types="time")
    cube.global_coords.add("reference_time", "time", times[0])
    template = make_spatial_template(cube, cube.wavelength_axis)
    assert isinstance(template, NDCube)
    assert not isinstance(template, SpectrogramCube)
    values = np.ones(template.shape)
    maps = [make_map_cube(template, values, u.km / u.s) for _ in range(2)]
    for output in maps:
        assert isinstance(output, NDCube)
        assert not isinstance(output, SpectrogramCube)
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
