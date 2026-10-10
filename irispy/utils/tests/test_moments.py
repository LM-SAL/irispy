import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy import constants
from astropy.modeling.models import Gaussian1D
from astropy.nddata import StdDevUncertainty, VarianceUncertainty
from astropy.tests.helper import assert_quantity_allclose

from irispy.io.utils import read_files
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.tests.helpers import figure_test, make_test_spectrogram_cube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import DN_UNIT, READOUT_NOISE
from irispy.utils.moments import average_window, calculate_moments
from irispy.utils.spectrograph import subtract_background

SI_IV_WAVELENGTHS = np.linspace(1401, 1405, 161) * u.AA
SI_IV_WINDOWS = [[1401, 1401.8], [1404, 1405]] * u.AA
SI_IV_MOMENTS = {"rest_wavelength": 1402.77 * u.AA, "velocity_range": (-107, 107) * u.km / u.s}  # 0.5 Å


def si_iv_on_background(degree):
    x = SI_IV_WAVELENGTHS.to_value(u.AA)
    background = np.polynomial.polynomial.polyval(x - 1403, [30, 4][: degree + 1])
    return np.tile(Gaussian1D(amplitude=500, mean=1402.8, stddev=0.08)(x) + background, (2, 3, 1))


def test_calculate_moments_basic(sns_sg_file):
    """
    Test that calculate_moments runs on real data and returns correct shapes and units.
    """
    raster_collection = read_files(sns_sg_file, uncertainty=True)
    cube = raster_collection["C II 1336"][0]
    # TWAVE1: the C II line, which the window brackets. The bundled test data is a
    # 10-pixel stride of the native data (~0.26 A/pixel), so the velocity range must span
    # several of those coarse pixels.
    rest_wvl = 1335.71 * u.Angstrom
    moments = calculate_moments(cube, rest_wavelength=rest_wvl, velocity_range=(-225, 225) * u.km / u.s)
    assert isinstance(moments, RasterCollection)
    assert set(moments.keys()) == {"intensity", "centroid", "width", "velocity", "velocity_width", "saturated"}
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
        (centroid.data[finite_mask] * centroid.unit >= rest_wvl - 1.0 * u.Angstrom)
        & (centroid.data[finite_mask] * centroid.unit <= rest_wvl + 1.0 * u.Angstrom)
    )
    assert np.all(width.data[finite_mask] >= 0)
    assert np.all(intensity.data >= 0)
    # Every map has a non-negative error, NaN where the value is not finite
    for key, moment in moments.items():
        if key == "saturated":
            continue
        error = moment.uncertainty.array
        assert np.isfinite(error).any()
        assert np.all(error[np.isfinite(error)] >= 0)
        assert np.isnan(error[~np.isfinite(moment.data)]).all()


def test_calculate_moments_sliced_cube(sns_sg_file):
    """
    Test that calculate_moments works on a sliced cube and with default arguments.

    With real IRIS data, rest_wavelength is auto-detected from cube metadata, so
    velocity outputs are included.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    cube_slice = cube[10, :, :]
    moments = calculate_moments(cube_slice)
    assert "intensity" in moments
    assert "centroid" in moments
    assert "width" in moments
    assert moments["intensity"].shape == cube_slice.shape[:-1]
    assert moments["centroid"].shape == cube_slice.shape[:-1]
    assert moments["width"].shape == cube_slice.shape[:-1]


@pytest.mark.parametrize(
    "velocity_range",
    [(-1.1 / 3 * constants.c, 0.1 / 3 * constants.c), np.array([-1.1, 0.1]) / 3 * constants.c, (-110000, 10000)],
)
def test_calculate_moments_asymmetric_velocity_range(velocity_range):
    """
    An asymmetric range, as a tuple of Quantities, a Quantity, or a bare pair in km/s:

    1.9 to 3.1 nm.
    """
    wvls = np.linspace(1.0, 5.0, 5) * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, len(wvls))), wvls)
    moments = calculate_moments(cube, rest_wavelength=3 * u.nm, velocity_range=velocity_range)
    assert_quantity_allclose(moments["intensity"].data[0, 0] * moments["intensity"].unit, 2 * u.DN)
    assert_quantity_allclose(moments["centroid"].data[0, 0] * moments["centroid"].unit, 2.5 * u.nm)


@pytest.mark.parametrize("velocity_range", [50 * u.km / u.s, (50, -50) * u.km / u.s])
def test_calculate_moments_velocity_range_must_be_increasing(velocity_range):
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.linspace(1.0, 5.0, 5) * u.nm)
    with pytest.raises(ValueError, match="two increasing velocities"):
        calculate_moments(cube, rest_wavelength=3 * u.nm, velocity_range=velocity_range)


def test_calculate_moments_rejects_unscaled_data():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 3), dtype=np.int16), [500.0, 501.0, 502.0] * u.nm)
    with pytest.raises(ValueError, match="unscaled"):
        calculate_moments(cube)


def test_calculate_moments_velocity_range_without_rest_wavelength(sns_sg_file):
    """
    Test that calculate_moments auto-detects rest_wavelength from cube metadata when
    velocity_range is given without explicit rest_wavelength.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    moments = calculate_moments(cube, velocity_range=(-1100, 1100) * u.km / u.s)
    assert "velocity" in moments


def test_calculate_moments_requires_wavelength_axis():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm)
    cube.wcs.wcs.ctype[0] = "TIME"
    cube.wcs.wcs.cunit[0] = "s"
    cube.wcs.wcs.set()

    with pytest.raises(ValueError, match="Could not identify a spectral wavelength axis"):
        calculate_moments(cube)


def test_calculate_moments_velocity_range_no_meta_no_rest_wavelength():
    """
    velocity_range with no rest_wavelength and no detectable meta should raise.
    """
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), np.arange(5) * u.nm)
    # Simulate meta without rest_wavelength attribute
    del cube.meta["TWAVE1"]
    with pytest.raises((ValueError, AttributeError), match="rest_wavelength must be provided"):
        calculate_moments(cube, velocity_range=(-100, 100) * u.km / u.s)


def test_calculate_moments_ignores_negative_nonfinite_and_masked_values():
    wavelengths = (500 + np.arange(5)) * u.nm
    clean_cube = make_test_spectrogram_cube(
        [[[1.0, 2.0, 0.0, 0.0, 0.0]]], wavelengths, uncertainty=StdDevUncertainty([[[0.1, 0.2, 0.0, 0.0, 0.0]]])
    )

    dirty_cube = make_test_spectrogram_cube(
        [[[1.0, 2.0, -5.0, np.nan, 10.0]]], wavelengths, uncertainty=StdDevUncertainty([[[0.1, 0.2, 0.3, 0.4, 0.5]]])
    )
    dirty_cube.mask = np.array([[[False, False, False, False, True]]])

    clean_moments = calculate_moments(clean_cube, rest_wavelength=501 * u.nm)
    dirty_moments = calculate_moments(dirty_cube, rest_wavelength=501 * u.nm)

    assert clean_moments.keys() == dirty_moments.keys()
    for key in clean_moments:
        assert clean_moments[key].unit == dirty_moments[key].unit
        assert not dirty_moments[key].mask[0, 0]
        np.testing.assert_allclose(dirty_moments[key].data, clean_moments[key].data, equal_nan=True)
        if key == "saturated":
            continue
        error = clean_moments[key].uncertainty.array
        assert np.isfinite(error).all()
        np.testing.assert_allclose(dirty_moments[key].uncertainty.array, error)


def test_calculate_moments_known_gaussian():
    """
    Test calculate_moments against a known Gaussian profile.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum = gauss(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, velocity_range=(-214, 214) * u.km / u.s)
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
    moments = calculate_moments(cube, rest_wavelength=rest_wvl, velocity_range=(-107, 107) * u.km / u.s)
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


def test_calculate_moments_min_intensity():
    """
    Test that min_intensity masks every map of the pixels below it, and only those.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    spectrum = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)(wvls.value)
    # Pixels above, at and below the threshold
    cube = make_test_spectrogram_cube(np.stack([2 * spectrum, spectrum, 0.5 * spectrum]).reshape(1, 3, -1), wvls)
    threshold = calculate_moments(cube)["intensity"].data[0, 1] * u.DN
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, min_intensity=threshold)
    for key in ("intensity", "centroid", "width", "velocity", "velocity_width"):
        np.testing.assert_array_equal(moments[key].mask[0], [False, False, True], err_msg=key)
        assert np.isfinite(moments[key].data[0, :2]).all(), key
        assert np.isnan(moments[key].data[0, 2]), key


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
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, velocity_range=(-107, 107) * u.km / u.s)
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


def test_calculate_moments_velocity_range_excludes_outside():
    """
    Test that samples outside velocity_range are excluded from the calculation.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    # Create a spectrum with two peaks: one at 1402.77 (the target) and one at 1403.3 (outside the range)
    gauss_target = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    gauss_outside = Gaussian1D(amplitude=20.0, mean=1403.3, stddev=0.05)
    spectrum = gauss_target(wvls.value) + gauss_outside(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    # Within 21 km/s, 0.1 A, only the target peak should be included
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.Angstrom, velocity_range=(-21, 21) * u.km / u.s)
    centroid = moments["centroid"]
    # If the outside peak were included, centroid would be pulled to ~1403.0
    # With the range excluding it, centroid should stay near 1402.77
    assert_quantity_allclose(centroid.data[0, 0] * centroid.unit, 140.277 * u.nm, atol=0.001 * u.nm)


def test_calculate_moments_velocity_range_empty_window():
    """
    Test that an empty velocity_range raises ValueError.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    spectrum = np.ones_like(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    # The range is completely outside the spectral coverage
    with pytest.raises(ValueError, match="No wavelengths between"):
        calculate_moments(cube, rest_wavelength=1500.0 * u.Angstrom, velocity_range=(-200, 200) * u.km / u.s)


def test_calculate_moments_saturated():
    wvls = (500 + np.arange(5)) * u.nm
    # Pixels with +Inf, as the readers set clipped samples, and with -Inf and NaN, which are not saturated
    data = np.tile([1.0, 2.0, 16181.75, 2.0, 1.0], (4, 1))
    data[1, 2], data[2, 0], data[3, 0], data[3, 4] = np.inf, np.inf, -np.inf, np.nan
    cube = make_test_spectrogram_cube(data[np.newaxis], wvls)
    moments = calculate_moments(cube, rest_wavelength=502 * u.nm)
    np.testing.assert_array_equal(moments["saturated"].data, [[False, True, True, False]])
    for key, moment in moments.items():
        if key == "saturated":
            assert moment.mask is None or not moment.mask.any()
            continue
        assert np.isnan(moment.data[0, 1:3]).all(), key
        assert np.isfinite(moment.data[0, [0, 3]]).all(), key
        np.testing.assert_array_equal(moment.mask, [[False, True, True, False]], err_msg=key)
    # Only the samples within velocity_range count: 700 km/s is 1.17 nm, short of the +Inf at 500 nm
    inside = calculate_moments(cube, rest_wavelength=502 * u.nm, velocity_range=(-700, 700) * u.km / u.s)
    np.testing.assert_array_equal(inside["saturated"].data, [[False, True, False, False]])


def test_calculate_moments_saturated_ignores_masked_samples():
    data = np.array([[[10.0, 500.0, 1000.0, 500.0, np.inf]]])
    mask = np.zeros(data.shape, dtype=bool)
    mask[..., -1] = True
    cube = make_test_spectrogram_cube(data, (500 + np.arange(5)) * u.nm, mask=mask)
    moments = calculate_moments(cube)
    assert not moments["saturated"].data.any()
    assert moments["intensity"].data[0, 0] == 2010


def test_calculate_moments_saturated_survives_exposure_time_correction(sns_sg_file):
    cube = read_files(sns_sg_file)["C II 1336"][0]
    cube.data[0, 3, 5] = np.inf
    for unit_cube in (cube, cube.apply_exposure_time_correction()):
        saturated = calculate_moments(unit_cube)["saturated"].data
        assert saturated[0, 3]
        assert saturated.sum() == 1


def test_calculate_moments_saturated_survives_subtract_background():
    data = si_iv_on_background(1)
    data[1, 2, 80] = np.inf  # 1403.0 Å, within velocity_range
    data[0, 1, 1:] = np.inf  # one finite sample in the windows, too few to fit a straight line
    cube = make_test_spectrogram_cube(data, SI_IV_WAVELENGTHS)
    corrected = subtract_background(cube, SI_IV_WINDOWS)
    assert np.isnan(corrected.data[0, 1, 0])
    assert np.isposinf(corrected.data[0, 1, 1:]).all()
    moments = calculate_moments(corrected, **SI_IV_MOMENTS)
    saturated = np.zeros((2, 3), dtype=bool)
    saturated[0, 1] = saturated[1, 2] = True
    np.testing.assert_array_equal(moments["saturated"].data, saturated)
    assert not moments["saturated"].mask[0, 1]
    assert np.isnan(moments["intensity"].data[1, 2])
    assert np.isfinite(moments["intensity"].data[0, 0])


def test_calculate_moments_constant_background_preserves_uncertainties():
    # A few ulps above 2.3 would drop the zero residuals from uncertainty propagation.
    wavelengths = (500 + np.arange(35)) * u.nm
    signal = np.zeros((2, 3, wavelengths.size))
    signal[..., 31:34] = [1, 2, 1]
    cube = make_test_spectrogram_cube(signal + 2.3, wavelengths, uncertainty=StdDevUncertainty(np.ones(signal.shape)))
    corrected = subtract_background(cube, [500, 530] * u.nm, degree=0)
    np.testing.assert_array_equal(corrected.data[..., :31], 0)
    expected = calculate_moments(
        make_test_spectrogram_cube(signal, wavelengths, uncertainty=cube.uncertainty), rest_wavelength=532 * u.nm
    )
    moments = calculate_moments(corrected, rest_wavelength=532 * u.nm)
    for name, moment in moments.items():
        np.testing.assert_allclose(moment.data, expected[name].data, rtol=1e-13, err_msg=name)
        if name != "saturated":
            np.testing.assert_allclose(
                moment.uncertainty.array, expected[name].uncertainty.array, rtol=1e-13, err_msg=name
            )


def test_calculate_moments_masks_pixels_without_samples():
    # A pixel masked within velocity_range, and one whose background had too few samples to fit, so is NaN
    mask = np.zeros((2, 3, SI_IV_WAVELENGTHS.size), dtype=bool)
    mask[0, 1, (SI_IV_WAVELENGTHS >= 1402.2 * u.AA) & (SI_IV_WAVELENGTHS <= 1403.4 * u.AA)] = True
    in_windows = (SI_IV_WAVELENGTHS <= 1401.8 * u.AA) | (SI_IV_WAVELENGTHS >= 1404 * u.AA)
    mask[1, 2, np.flatnonzero(in_windows)[1:]] = True  # one sample left for a straight line
    cube = make_test_spectrogram_cube(si_iv_on_background(1), SI_IV_WAVELENGTHS, mask=mask)
    moments = calculate_moments(subtract_background(cube, SI_IV_WINDOWS), **SI_IV_MOMENTS)
    for name, moment in moments.items():
        assert moment.mask[0, 1], name
        assert moment.mask[1, 2], name
        assert not moment.mask[0, 0], name
        assert np.isfinite(moment.data[0, 0]), name
    assert moments["intensity"].data[0, 1] == 0


def test_calculate_moments_integrated():
    """
    Test that integrated=True returns intensity in DN·nm.
    """
    wvls = np.linspace(1402.0, 1403.5, 100) * u.Angstrom
    gauss = Gaussian1D(amplitude=10.0, mean=1402.77, stddev=0.05)
    spectrum = gauss(wvls.value)
    data = spectrum.reshape(1, 1, -1)
    cube = make_test_spectrogram_cube(data, wvls)
    moments = calculate_moments(
        cube, rest_wavelength=1402.77 * u.Angstrom, velocity_range=(-214, 214) * u.km / u.s, integrated=True
    )
    intensity = moments["intensity"]
    assert intensity.unit == u.DN * u.nm
    # Intensity value should be the analytic integral
    expected_intensity = np.sqrt(2 * np.pi) * 10.0 * 0.005
    assert_quantity_allclose(intensity.data[0, 0] * intensity.unit, expected_intensity * u.DN * u.nm, rtol=0.01)


def test_calculate_moments_preserves_time_without_spectral_global_coord(sns_sg_file):
    """
    Test that moment maps keep scan times without adding a fixed wavelength coordinate.
    """
    raster_collection = read_files(sns_sg_file)
    cube = raster_collection["C II 1336"][0]
    moments = calculate_moments(cube)
    intensity = moments["intensity"]
    assert "time" in tuple(intensity.extra_coords.keys())
    assert "em.wl" not in tuple(intensity.global_coords.keys())
    np.testing.assert_array_equal(
        intensity.axis_world_coords("time", wcs=intensity.extra_coords)[0].isot,
        cube.axis_world_coords("time", wcs=cube.extra_coords)[0].isot,
    )


def test_calculate_moments_uncertainty_by_hand():
    # Intensity 7, centroid 501 3/7 nm, offsets (-10, -3, 4)/7 nm, variance 26/49 nm^2
    cube = make_test_spectrogram_cube(
        [[[1.0, 2.0, 4.0]]], [500.0, 501.0, 502.0] * u.nm, uncertainty=StdDevUncertainty([[[0.1, 0.2, 0.3]]])
    )
    moments = calculate_moments(cube, rest_wavelength=500.5 * u.nm)
    centroid_error = np.sqrt(2.8) / 49
    width_error = np.sqrt(75.32) / 98 / np.sqrt(26)
    speed = constants.c.to_value(u.km / u.s) / 500.5
    expected = {
        "intensity": np.sqrt(0.14),
        "centroid": centroid_error,
        "width": width_error,
        "velocity": centroid_error * speed,
        "velocity_width": width_error * speed,
    }
    for key, error in expected.items():
        assert isinstance(moments[key].uncertainty, StdDevUncertainty)
        np.testing.assert_allclose(moments[key].uncertainty.array, error, rtol=1e-10)


@pytest.mark.parametrize(("background", "peak", "low", "high"), [(50, 1000, 0.95, 1.05), (0, 100, 1.0, 1.35)])
def test_calculate_moments_uncertainty_monte_carlo(background, peak, low, high):
    # A Si IV line with FUV photon and readout noise, 4000 times; with no background,
    # zeroing the negative samples makes the errors conservative
    rng = np.random.default_rng(0)
    wavelengths = np.arange(1402.0, 1403.5, 0.02596) * u.AA
    truth = background + peak * np.exp(-0.5 * ((wavelengths.value - 1402.77) / 0.05) ** 2)
    photons_per_dn = DN_UNIT["FUV"].to(u.photon)
    readout = READOUT_NOISE["FUV"].to_value(DN_UNIT["FUV"])
    data = rng.poisson(np.broadcast_to(truth * photons_per_dn, (1, 4000, truth.size))) / photons_per_dn
    data += rng.normal(0, readout, data.shape)
    sigma = calculate_uncertainty(data, READOUT_NOISE["FUV"], DN_UNIT["FUV"])
    cube = make_test_spectrogram_cube(data, wavelengths, uncertainty=StdDevUncertainty(sigma))
    moments = calculate_moments(cube, rest_wavelength=1402.77 * u.AA, velocity_range=(-64, 64) * u.km / u.s)
    for key in moments.keys() - {"saturated"}:
        ratio = np.median(moments[key].uncertainty.array) / np.std(moments[key].data)
        assert low < ratio < high, key


def test_calculate_moments_uncertainty_types_and_invalid_pixels():
    wavelengths = [500.0, 501.0, 502.0] * u.nm
    data = [[[1.0, 2.0, 1.0], [0.1, 0.2, 0.1]]]
    sigma = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
    standard = calculate_moments(
        make_test_spectrogram_cube(data, wavelengths, uncertainty=StdDevUncertainty(sigma)), min_intensity=1
    )
    # test_standard_deviation covers the other types
    variance = calculate_moments(
        make_test_spectrogram_cube(data, wavelengths, uncertainty=VarianceUncertainty(sigma**2)), min_intensity=1
    )
    for key, moment in standard.items():
        if key == "saturated":
            continue
        np.testing.assert_allclose(variance[key].uncertainty.array, moment.uncertainty.array)
        assert np.isfinite(moment.uncertainty.array[0, 0])
        assert np.isnan(moment.uncertainty.array[0, 1])  # below min_intensity
    for moment in calculate_moments(make_test_spectrogram_cube(np.ones((1, 1, 3)), wavelengths)).values():
        assert moment.uncertainty is None


def test_calculate_moments_uncertainty_integrated():
    # integrated=True scales only the intensity and its error, by the step, also on a decreasing axis
    wavelengths = [500.0, 500.5, 501.0] * u.nm
    ascending = make_test_spectrogram_cube(
        [[[1.0, 2.0, 4.0]]], wavelengths, uncertainty=StdDevUncertainty([[[0.1, 0.2, 0.3]]])
    )
    descending = make_test_spectrogram_cube(
        [[[4.0, 2.0, 1.0]]], wavelengths[::-1], uncertainty=StdDevUncertainty([[[0.3, 0.2, 0.1]]])
    )
    summed = calculate_moments(ascending, rest_wavelength=500.5 * u.nm)
    for cube in (ascending, descending):
        integrated = calculate_moments(cube, rest_wavelength=500.5 * u.nm, integrated=True)
        for key, moment in summed.items():
            scale = 0.5 if key == "intensity" else 1
            np.testing.assert_allclose(integrated[key].data, scale * moment.data, err_msg=key)
            if key != "saturated":
                np.testing.assert_allclose(
                    integrated[key].uncertainty.array, scale * moment.uncertainty.array, err_msg=key
                )


def test_calculate_moments_uncertainty_nan_where_undefined():
    # One sample left, none left, and a width of 0 from exact zeros, which rounds to about 1e-14 nm
    data = [[[33.9, -0.9, -2.0], [-1.0, -2.0, np.nan], [0.0, 33.9, 0.0]]]
    cube = make_test_spectrogram_cube(
        data, [500.0, 501.0, 502.0] * u.nm, uncertainty=StdDevUncertainty(np.full((1, 3, 3), 0.1))
    )
    moments = calculate_moments(cube, rest_wavelength=501 * u.nm)
    centroid_error = np.sqrt(0.02) / 33.9
    expected = {
        "intensity": [0.1, np.nan, np.sqrt(0.03)],
        "centroid": [np.nan, np.nan, centroid_error],
        "velocity": [np.nan, np.nan, centroid_error * constants.c.to_value(u.km / u.s) / 501],
        "width": [np.nan, np.nan, np.nan],
        "velocity_width": [np.nan, np.nan, np.nan],
    }
    for key, error in expected.items():
        np.testing.assert_allclose(moments[key].uncertainty.array[0], error, err_msg=key)


def test_calculate_moments_scalar_uncertainty_with_velocity_range():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 5)), (500 + np.arange(5)) * u.nm)
    cube.uncertainty = StdDevUncertainty(3.0)
    moments = calculate_moments(cube, rest_wavelength=502 * u.nm, velocity_range=(-896, 896) * u.km / u.s)  # 1.5 nm
    np.testing.assert_allclose(moments["intensity"].uncertainty.array, 3 * np.sqrt(3))


def test_calculate_moments_zero_rest_wavelength_warns():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 3)), [500.0, 501.0, 502.0] * u.nm)
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        calculate_moments(cube, rest_wavelength=0 * u.nm)


@pytest.mark.parametrize(
    ("method", "values", "errors"),
    [("mean", [3, 4, np.nan], [np.sqrt(0.13) / 2, 0.3, np.nan]), ("sum", [6, 4, np.nan], [np.sqrt(0.13), 0.3, np.nan])],
)
def test_average_window_by_hand(method, values, errors):
    # The window holds the samples at 501 and 502 nm; NaN and masked samples are left out
    mask = np.zeros((1, 3, 4), dtype=bool)
    mask[0, 2, 1:3] = True
    cube = make_test_spectrogram_cube(
        [[[1.0, 2.0, 4.0, 8.0], [1.0, np.nan, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]]],
        [500.0, 501.0, 502.0, 503.0] * u.nm,
        uncertainty=StdDevUncertainty(np.broadcast_to([0.1, 0.2, 0.3, 0.4], (1, 3, 4))),
        mask=mask,
    )
    window = average_window(cube, [5005, 5025] * u.AA, method=method)
    assert window.unit == cube.unit
    assert isinstance(window.uncertainty, StdDevUncertainty)
    np.testing.assert_allclose(window.data[0], values)
    np.testing.assert_allclose(window.uncertainty.array[0], errors)
    np.testing.assert_array_equal(window.mask[0], [False, False, True])


def test_average_window_saturated_is_nan_and_masked():
    cube = make_test_spectrogram_cube(
        [[[1.0, np.inf, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]]], [500.0, 501.0, 502.0, 503.0] * u.nm
    )
    window = average_window(cube, [5005, 5025] * u.AA)
    assert np.isnan(window.data[0, 0])
    assert window.mask[0, 0]
    assert window.data[0, 1] == 3
    assert not window.mask[0, 1]


def test_average_window_without_uncertainty():
    cube = make_test_spectrogram_cube(np.ones((1, 1, 3)), [500.0, 501.0, 502.0] * u.nm)
    assert average_window(cube, [501, 502] * u.nm).uncertainty is None


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[[True, False, False]]]),
        np.array([True, False, False]),
        np.broadcast_to([1, 0, 0], (2, 2, 3)).astype(np.uint8),
    ],
)
def test_average_window_mask_representations(mask):
    data = np.broadcast_to([1.0, 2.0, 4.0], (2, 2, 3)).copy()
    cube = make_test_spectrogram_cube(data, [500, 501, 502] * u.nm, mask=mask)
    original_mask = mask.copy()
    original_data = data.copy()
    window = average_window(cube, [500, 502] * u.nm)
    np.testing.assert_allclose(window.data, np.full((2, 2), 3.0))
    assert not window.mask.any()
    assert cube.mask is mask
    np.testing.assert_array_equal(cube.mask, original_mask)
    np.testing.assert_array_equal(cube.data, original_data)


def test_average_window_real_data(sns_sg_file):
    cube = read_files(sns_sg_file, uncertainty=True)["C II 1336"][0]
    window = average_window(cube, [1335.2, 1336.2] * u.AA)
    assert window.shape == cube.shape[:-1]
    assert window.wcs.world_axis_physical_types == cube.wcs.world_axis_physical_types[1:]
    assert "time" in tuple(window.extra_coords.keys())
    assert np.isfinite(window.data[~window.mask]).all()
    assert np.isfinite(window.uncertainty.array[~window.mask]).all()


@pytest.mark.parametrize(
    ("wavelength_range", "kwargs", "match"),
    [
        ([501, 502] * u.nm, {"method": "median"}, "method must be"),
        ([501.2, 501.8] * u.nm, {}, "No wavelengths between"),
        ([501] * u.nm, {}, "two wavelengths"),
    ],
)
def test_average_window_rejects_bad_input(wavelength_range, kwargs, match):
    cube = make_test_spectrogram_cube(np.ones((1, 1, 3)), [500.0, 501.0, 502.0] * u.nm)
    with pytest.raises(ValueError, match=match):
        average_window(cube, wavelength_range, **kwargs)
