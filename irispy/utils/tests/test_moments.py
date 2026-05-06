import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.modeling.models import Gaussian1D
from astropy.tests.helper import assert_quantity_allclose

from irispy.io.utils import read_files
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.tests.helpers import figure_test, make_test_spectrogram_cube
from irispy.utils.moments import calculate_moments


def test_calculate_moments_basic(sns_sg_file):
    """
    Test that calculate_moments runs on real data and returns correct shapes and units.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"]
    rest_wvl = 1332.9 * u.Angstrom
    moments = calculate_moments(cube, rest_wavelength=rest_wvl, wings=0.1 * u.Angstrom)
    assert isinstance(moments, RasterCollection)
    assert set(moments.keys()) == {"intensity", "centroid", "width", "velocity", "velocity_width"}
    intensity = moments["intensity"]
    centroid = moments["centroid"]
    width = moments["width"]
    velocity = moments["velocity"]
    velocity_width = moments["velocity_width"]
    # Check shapes: cube is (nt, ny, nwl), so moments should be (nt, ny)
    assert intensity.shape == cube.shape[:-1]
    assert centroid.shape == cube.shape[:-1]
    assert width.shape == cube.shape[:-1]
    assert velocity.shape == cube.shape[:-1]
    assert velocity_width.shape == cube.shape[:-1]
    assert isinstance(intensity, SpectrogramCube)
    assert isinstance(centroid, SpectrogramCube)
    assert isinstance(width, SpectrogramCube)
    assert isinstance(velocity, SpectrogramCube)
    assert isinstance(velocity_width, SpectrogramCube)
    assert intensity.unit == cube.unit
    assert centroid.unit == u.nm
    assert width.unit == u.nm
    assert velocity.unit == u.km / u.s
    assert velocity_width.unit == u.km / u.s
    finite_mask = np.isfinite(centroid.data)
    assert np.all(
        (centroid.data[finite_mask] * centroid.unit >= rest_wvl - 0.1 * u.Angstrom)
        & (centroid.data[finite_mask] * centroid.unit <= rest_wvl + 0.1 * u.Angstrom)
    )
    assert np.all(width.data[finite_mask] >= 0)
    assert np.all(intensity.data >= 0)


def test_calculate_moments_sliced_cube(sns_sg_file):
    """
    Test that calculate_moments works on a sliced cube and with default arguments.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"]
    cube_slice = cube[10, :, :]
    moments = calculate_moments(cube_slice)
    assert set(moments.keys()) == {"intensity", "centroid", "width"}
    assert moments["intensity"].shape == cube_slice.shape[:-1]
    assert moments["centroid"].shape == cube_slice.shape[:-1]
    assert moments["width"].shape == cube_slice.shape[:-1]


def test_calculate_moments_asymmetric_wings(sns_sg_file):
    """
    Test that calculate_moments works with asymmetric wings.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"]
    rest_wvl = 1332.9 * u.Angstrom
    moments = calculate_moments(cube, rest_wavelength=rest_wvl, wings=(0.05, 0.15) * u.Angstrom)
    assert set(moments.keys()) == {"intensity", "centroid", "width", "velocity", "velocity_width"}
    assert moments["intensity"].shape == cube.shape[:-1]
    centroid = moments["centroid"]
    finite_mask = np.isfinite(centroid.data)
    assert np.all(
        (centroid.data[finite_mask] * centroid.unit >= rest_wvl - 0.05 * u.Angstrom)
        & (centroid.data[finite_mask] * centroid.unit <= rest_wvl + 0.15 * u.Angstrom)
    )


def test_calculate_moments_asymmetric_wings_tuple_quantity():
    """
    Test that asymmetric wings can be given as a tuple of Quantity objects.
    """
    wvls = np.linspace(1.0, 5.0, 5) * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, len(wvls))), wvls)
    moments = calculate_moments(cube, rest_wavelength=3 * u.nm, wings=(1.1 * u.nm, 0.1 * u.nm))
    assert_quantity_allclose(moments["intensity"].data[0, 0] * moments["intensity"].unit, 2 * u.DN)
    assert_quantity_allclose(moments["centroid"].data[0, 0] * moments["centroid"].unit, 2.5 * u.nm)


def test_calculate_moments_asymmetric_wings_rejects_bare_tuple():
    wvls = np.linspace(1.0, 5.0, 5) * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, len(wvls))), wvls)

    with pytest.raises(TypeError, match=r"wings tuple elements must be astropy\.units\.Quantity"):
        calculate_moments(cube, rest_wavelength=3 * u.nm, wings=(1.1, 0.1))


def test_calculate_moments_wings_without_rest_wavelength(sns_sg_file):
    """
    Test that calculate_moments raises an error when wings is given without
    rest_wavelength.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"]
    with pytest.raises(ValueError, match="rest_wavelength must be provided"):
        calculate_moments(cube, wings=1.0 * u.Angstrom)


def test_calculate_moments_requires_wavelength_axis():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm)
    cube.wcs.wcs.ctype[0] = "TIME"
    cube.wcs.wcs.cunit[0] = "s"
    cube.wcs.wcs.set()

    with pytest.raises(ValueError, match="Could not identify a spectral wavelength axis"):
        calculate_moments(cube)


def test_calculate_moments_ignores_negative_nonfinite_and_masked_values():
    wavelengths = np.arange(4) * u.nm
    clean_cube = make_test_spectrogram_cube(np.array([[[1.0, 0.0, 0.0, 0.0]]]), wavelengths)

    dirty_cube = make_test_spectrogram_cube(np.array([[[1.0, -5.0, np.nan, 10.0]]]), wavelengths)
    dirty_cube.mask = np.array([[[False, False, False, True]]])

    clean_moments = calculate_moments(clean_cube)
    dirty_moments = calculate_moments(dirty_cube)

    assert clean_moments.keys() == dirty_moments.keys()
    for key in clean_moments:
        assert clean_moments[key].unit == dirty_moments[key].unit
        np.testing.assert_allclose(dirty_moments[key].data, clean_moments[key].data, equal_nan=True)


def test_calculate_moments_known_gaussian():
    """
    Test calculate_moments against a known Gaussian profile.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum = gauss(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, wings=1.0 * u.Angstrom)
    intensity = moments["intensity"]
    centroid = moments["centroid"]
    width = moments["width"]
    velocity = moments["velocity"]
    velocity_width = moments["velocity_width"]
    # Intensity is the per-pixel sum (default integrated=False)
    expected_intensity = np.sum(gauss(wvls.value))
    assert_quantity_allclose(intensity.data[0, 0] * intensity.unit, expected_intensity * u.DN, rtol=0.01)
    # Centroid should be close to the Gaussian mean
    assert_quantity_allclose(centroid.data[0, 0] * centroid.unit, 140.277 * u.nm, atol=0.001 * u.nm)
    # Width should be close to the Gaussian stddev
    assert_quantity_allclose(width.data[0, 0] * width.unit, 0.005 * u.nm, rtol=0.05)
    # Velocity should be near zero because the Gaussian mean matches rest_wavelength
    assert_quantity_allclose(velocity.data[0, 0] * velocity.unit, 0 * u.km / u.s, atol=1 * u.km / u.s)
    # Velocity width: stddev/lambda0 * c
    expected_velocity_width = (0.05 * u.Angstrom / (1402.77 * u.Angstrom) * constants.c).to(u.km / u.s)
    assert_quantity_allclose(velocity_width.data[0, 0] * velocity_width.unit, expected_velocity_width, rtol=0.05)


@figure_test
def test_calculate_moments_known_gaussian_figure():
    """
    Visual regression test for moment extraction on known Gaussian profiles.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    rest_wvl = 1402.77 * u.Angstrom
    # Per-row properties: (amplitude, stddev in Å, Doppler shift in km/s)
    row_props = [
        (5.0, 0.03, -20.0),  # row 0: faint, narrow, blue
        (10.0, 0.05, 0.0),  # row 1: medium, medium, rest
        (15.0, 0.07, 20.0),  # row 2: bright, wide, red
    ]
    data = np.zeros((3, 3, len(wvls)))
    for row, (amp, std, vel) in enumerate(row_props):
        offset = (rest_wvl * vel * u.km / u.s / constants.c).to(u.Angstrom).value
        mean_wvl = rest_wvl.value + offset
        gauss = Gaussian1D(amplitude=amp, mean=mean_wvl, stddev=std)
        spectrum = gauss(wvls.value)
        data[row, :, :] = spectrum
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube, rest_wavelength=rest_wvl, wings=0.5 * u.Angstrom)
    intensity = moments["intensity"]
    velocity = moments["velocity"]
    width = moments["width"]
    centroid = moments["centroid"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Panel 1: all three Gaussian profiles overlaid
    ax = axes[0, 0]
    row_colors = ["C0", "k", "C3"]
    row_labels = ["Row 0 (amp=5, std=0.03, v=-20)", "Row 1 (amp=10, std=0.05, v=0)", "Row 2 (amp=15, std=0.07, v=+20)"]
    for row in range(3):
        ax.step(wvls.value, data[row, 1, :], where="mid", color=row_colors[row], alpha=0.5, label=row_labels[row])
        ax.plot(wvls.value, data[row, 1, :], "o", color=row_colors[row], markersize=3, alpha=0.7)
        c = (centroid.data[row, 1] * centroid.unit).to(u.Angstrom).value
        ax.axvline(c, color=row_colors[row], linestyle="--", alpha=0.7)
    ax.set_xlabel("Wavelength [AA]")
    ax.set_ylabel("Intensity [DN]")
    ax.set_title("Input spectra: three different Gaussians")
    ax.legend(loc="upper right", fontsize=7)
    # Panel 2: intensity map (three bands: faint / medium / bright)
    ax = axes[0, 1]
    im = ax.imshow(intensity.data, origin="lower", cmap="viridis")
    ax.set_title(f"Intensity [{intensity.unit}]")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    plt.colorbar(im, ax=ax)
    # Panel 3: velocity map (three bands: blue / white / red)
    ax = axes[1, 0]
    im = ax.imshow(velocity.data, origin="lower", cmap="coolwarm", vmin=-25, vmax=25)
    ax.set_title(f"Velocity [{velocity.unit}]")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    for row in range(3):
        actual_v = velocity.data[row, 1]
        ax.text(1.0, row, f"{actual_v:.1f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax)
    # Panel 4: width map (three bands: narrow / medium / wide)
    ax = axes[1, 1]
    im = ax.imshow(width.data, origin="lower", cmap="plasma")
    ax.set_title(f"Width [{width.unit}]")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def test_calculate_moments_negative_values_zeroed():
    """
    Test that negative data values are treated as zero during moment calculation.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    spectrum = np.ones_like(wvls.value) * 10.0
    spectrum[10:20] = -5.0
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube)
    intensity = moments["intensity"]
    centroid = moments["centroid"]
    # Intensity should equal sum of positive values only
    expected_intensity = np.sum(np.where(spectrum < 0, 0, spectrum))
    assert_quantity_allclose(intensity.data[0, 0] * intensity.unit, expected_intensity * u.DN, rtol=1e-10)
    # Centroid should be the same as if negatives were zeroed manually
    clean_spectrum = np.where(spectrum < 0, 0, spectrum)
    expected_centroid = np.sum(clean_spectrum * wvls.value) / np.sum(clean_spectrum)
    assert_quantity_allclose(
        centroid.data[0, 0] * centroid.unit, expected_centroid * u.Angstrom, atol=0.01 * u.Angstrom
    )


def test_calculate_moments_nan_values_zeroed():
    """
    Test that NaN data values are treated as zero during moment calculation.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    spectrum = np.ones_like(wvls.value) * 10.0
    spectrum[10:20] = np.nan
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube)
    intensity = moments["intensity"]
    centroid = moments["centroid"]
    width = moments["width"]
    # Intensity should equal sum of finite values only
    expected_intensity = np.sum(np.where(np.isfinite(spectrum), spectrum, 0))
    assert_quantity_allclose(intensity.data[0, 0] * intensity.unit, expected_intensity * u.DN, rtol=1e-10)
    assert np.isfinite(centroid.data[0, 0])
    assert np.isfinite(width.data[0, 0])


def test_calculate_moments_masked_values_zeroed():
    """
    Test that masked spectral bins do not contribute to moments.
    """
    wvls = np.array([1.0, 2.0, 3.0]) * u.nm
    data = np.array([[[1.0, 1.0, 1000.0]]])
    cube = make_test_spectrogram_cube(data, wvls)
    cube.mask = np.array([[[False, False, True]]])
    moments = calculate_moments(cube)
    assert_quantity_allclose(moments["intensity"].data[0, 0] * moments["intensity"].unit, 2 * u.DN)
    assert not moments["intensity"].mask[0, 0]
    assert_quantity_allclose(moments["centroid"].data[0, 0] * moments["centroid"].unit, 1.5 * u.nm)
    assert_quantity_allclose(moments["width"].data[0, 0] * moments["width"].unit, 0.5 * u.nm)


def test_calculate_moments_zero_intensity():
    """
    Test that a completely zero spectrum returns NaN centroid and width.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    spectrum = np.zeros_like(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube)
    intensity = moments["intensity"]
    centroid = moments["centroid"]
    width = moments["width"]
    assert intensity.data[0, 0] == 0
    assert np.isnan(centroid.data[0, 0])
    assert np.isnan(width.data[0, 0])


def test_calculate_moments_below_min_intensity_masks_all_products():
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    data = gauss(wvls.value).reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)

    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, min_intensity=1e6 * u.DN)

    for key in ("intensity", "centroid", "width", "velocity", "velocity_width"):
        assert np.isnan(moments[key].data[0, 0])
        assert moments[key].mask[0, 0]


def test_calculate_moments_vectorized_spatial():
    """
    Test that calculate_moments correctly handles different spectra per spatial pixel.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    # Pixel (0, 0): Gaussian centered at 1402.77
    gauss_0 = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    # Pixel (0, 1): Gaussian centered at 1402.80 (slightly redshifted)
    gauss_1 = Gaussian1D(amplitude=10.0, mean=1402.80, stddev=0.05)
    spectrum_0 = gauss_0(wvls.value)
    spectrum_1 = gauss_1(wvls.value)
    data = np.stack([spectrum_0, spectrum_1]).reshape(1, 2, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, wings=0.5 * u.Angstrom)
    centroid = moments["centroid"]
    velocity = moments["velocity"]
    # Pixel (0, 0) should be near rest wavelength
    assert_quantity_allclose(centroid.data[0, 0] * centroid.unit, 140.277 * u.nm, atol=0.001 * u.nm)
    assert_quantity_allclose(velocity.data[0, 0] * velocity.unit, 0 * u.km / u.s, atol=1 * u.km / u.s)
    # Pixel (0, 1) should be slightly redshifted
    expected_centroid_1 = 140.280 * u.nm
    expected_velocity_1 = ((140.280 * u.nm - 140.277 * u.nm) / (140.277 * u.nm) * constants.c).to(u.km / u.s)
    assert_quantity_allclose(centroid.data[0, 1] * centroid.unit, expected_centroid_1, atol=0.001 * u.nm)
    assert_quantity_allclose(velocity.data[0, 1] * velocity.unit, expected_velocity_1, atol=1 * u.km / u.s)


def test_calculate_moments_wings_excludes_outside():
    """
    Test that pixels outside the wings window are excluded from the calculation.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    # Create a spectrum with two peaks: one at 1402.77 (the target) and one at 1403.3 (outside wings)
    gauss_target = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    gauss_outside = Gaussian1D(amplitude=20.0, mean=1403.3, stddev=0.05)
    spectrum = gauss_target(wvls.value) + gauss_outside(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    # With wings=0.1 nm = 1 A, only the target peak should be included
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, wings=0.1 * u.Angstrom)
    centroid = moments["centroid"]
    # If the outside peak were included, centroid would be pulled to ~1403.0
    # With wings excluding it, centroid should stay near 1402.77
    assert_quantity_allclose(centroid.data[0, 0] * centroid.unit, 140.277 * u.nm, atol=0.001 * u.nm)


def test_calculate_moments_wings_empty_window():
    """
    Test that an empty wings window raises ValueError.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    spectrum = np.ones_like(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    # wings range is completely outside the spectral coverage
    with pytest.raises(ValueError, match="No wavelength points found within the specified wings"):
        calculate_moments(cube, rest_wavelength=1500.0 * u.Angstrom, wings=1.0 * u.Angstrom)


def test_calculate_moments_saturation_limit():
    """
    Test that saturation_limit masks saturated pixels.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum = gauss(wvls.value)
    # Artificially saturate the peak
    spectrum[np.argmax(spectrum)] = 1e5
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube, saturation_limit=1e4)
    assert np.isnan(moments["intensity"].data[0, 0])
    assert np.isnan(moments["centroid"].data[0, 0])
    assert np.isnan(moments["width"].data[0, 0])


def test_calculate_moments_integrated():
    """
    Test that integrated=True returns intensity in DN·nm.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum = gauss(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, wings=1.0 * u.Angstrom, integrated=True)
    intensity = moments["intensity"]
    centroid = moments["centroid"]
    width = moments["width"]
    assert intensity.unit == u.DN * u.nm
    # Intensity value should be the analytic integral
    expected_intensity = np.sqrt(2 * np.pi) * 10.0 * 0.005
    assert_quantity_allclose(intensity.data[0, 0] * intensity.unit, expected_intensity * u.DN * u.nm, rtol=0.01)
    # Centroid and width should be unchanged from the non-integrated case
    assert_quantity_allclose(centroid.data[0, 0] * centroid.unit, 140.277 * u.nm, atol=0.001 * u.nm)
    assert_quantity_allclose(width.data[0, 0] * width.unit, 0.005 * u.nm, rtol=0.05)


def test_calculate_moments_min_intensity_mixed_pixels():
    """
    Test that min_intensity only masks pixels below the threshold.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    # Create a 2x2 spatial grid with different amplitudes
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum_high = gauss(wvls.value)  # amplitude 10
    gauss_low = Gaussian1D(amplitude=1.0, mean=1402.77, stddev=0.05)
    spectrum_low = gauss_low(wvls.value)  # amplitude 1
    data = np.zeros((2, 2, len(wvls)))
    data[0, 0] = spectrum_high
    data[0, 1] = spectrum_low
    data[1, 0] = spectrum_low
    data[1, 1] = spectrum_high
    cube = make_test_spectrogram_cube(data, wvls)
    # Threshold between the two amplitudes: low pixels masked, high pixels kept
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, min_intensity=50 * u.DN)
    # High pixels should be valid
    assert np.isfinite(moments["intensity"].data[0, 0])
    assert np.isfinite(moments["intensity"].data[1, 1])
    # Low pixels should be NaN
    assert np.isnan(moments["intensity"].data[0, 1])
    assert np.isnan(moments["intensity"].data[1, 0])
    assert np.isnan(moments["centroid"].data[0, 1])
    assert np.isnan(moments["centroid"].data[1, 0])
    assert np.isnan(moments["width"].data[0, 1])
    assert np.isnan(moments["width"].data[1, 0])


def test_calculate_moments_min_intensity_at_threshold():
    """
    Test that pixels exactly at min_intensity are NOT masked.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum = gauss(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    intensity_value = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom)["intensity"].data[0, 0]
    # Set threshold exactly equal to the intensity — should be kept
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, min_intensity=intensity_value)
    assert np.isfinite(moments["intensity"].data[0, 0])
    assert np.isfinite(moments["centroid"].data[0, 0])
    assert np.isfinite(moments["width"].data[0, 0])


def test_calculate_moments_preserves_time_without_spectral_global_coord(sns_sg_file):
    """
    Test that moment maps keep scan times without adding a fixed wavelength coordinate.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"]
    moments = calculate_moments(cube)
    intensity = moments["intensity"]
    assert "time" in tuple(intensity.extra_coords.keys())
    assert "em.wl" not in tuple(intensity.global_coords.keys())
    np.testing.assert_array_equal(
        intensity.axis_world_coords("time", wcs=intensity.extra_coords)[0].isot,
        cube.axis_world_coords("time", wcs=cube.extra_coords)[0].isot,
    )
