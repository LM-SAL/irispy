import numpy as np
import pooch
import pytest

import astropy.units as u
from astropy.table import QTable

from irispy.data.test import get_test_filepath
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.spectrograph import RasterCollection, SpectrogramCubeSequence
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.wavelength_drift import _LINES, _fit_drift, _line_shifts, _wavelengths, calculate_wavelength_drift

IRISPY_DATA = "https://github.com/LM-SAL/irispy-data/releases/download/v1"
LINES = ["Ni I", "Mn I", "Fe I", "O I", "Fe II"]
CROP = get_test_filepath(
    "wavelength_drift/iris_l2_20140708_114109_3824262996_raster_t000_r00000_wavelength_drift_test.fits"
)
CROP_STEPS = list(range(0, 400, 57))
SHORT = "too short for an orbital fit"


def idl_reference(obsid):
    return QTable.read(get_test_filepath(f"wavelength_drift/iris_prep_wavecorr_l2_{obsid}_r00000.ecsv"))


def assert_shifts_match_idl(table, idl):
    """
    IDL's CURVEFIT stops early, which moves its centres on these files by up to 2 mÅ.
    """
    assert all(np.abs((table["time"] - idl["time"]).to_value(u.s)) < 1e-6)
    for name in LINES:
        ours, theirs = table[name].to_value(u.AA), idl[name].to_value(u.AA)
        difference = np.abs(ours - theirs)[np.isfinite(ours) & np.isfinite(theirs)]
        if name == "Fe II":
            # Fits to the weak Fe II line can land on spikes, and are kept or rejected near the range limit.
            assert np.mean(np.isnan(ours) == np.isnan(theirs)) >= 0.95
            assert np.mean(difference < 1e-3) >= 0.75
        else:
            np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
            assert difference.max() < (1e-3 if name == "O I" else 2.5e-3)


def seconds(table):
    return (table["time"] - table["time"][0]).to_value(u.s)


def synthetic_o_i_shifts(size, sign=1, pixel=0.15, slope=0, scale=1):
    """
    The O I shifts of a line 0.02 Å redward of rest and 0.8 pixels wide, on ``size``
    pixels of ``pixel`` Å from 1355.05 Å and a background sloping by ``slope`` DN/Å, all
    times ``scale``.
    """
    wavelengths = 1355.05 + pixel * np.arange(size)
    x = wavelengths - 1355.62
    profile = scale * (100 + slope * x + sign * 50 * np.exp(-0.5 * (x / (0.8 * pixel)) ** 2))
    cube = make_test_spectrogram_cube(np.broadcast_to(profile, (2, 3, size)).copy(), wavelengths * u.AA)
    return _line_shifts(cube, _wavelengths(cube), *_LINES["O I"])


def test_shifts_match_idl():
    table = calculate_wavelength_drift(read_spectrograph_lvl2(CROP))
    assert table.colnames == ["time", "raster", "step", *LINES, "nuv", "fuv"]
    assert list(table["step"]) == list(range(len(CROP_STEPS)))
    idl = idl_reference("3824262996")[CROP_STEPS]
    assert_shifts_match_idl(table, idl)
    # Each drift is fitted to its own line, and follows IDL's fit through all 400 steps
    for drift, line, outlier, atol in [("nuv", "Ni I", 0.08, 4e-3), ("fuv", "O I", 0.05, 2e-3)]:
        fitted = _fit_drift(seconds(table), table[line].to_value(u.AA), outlier, drift)
        np.testing.assert_array_equal(table[drift].to_value(u.AA), fitted)
        np.testing.assert_allclose(table[drift].to_value(u.AA), idl[drift].to_value(u.AA), atol=atol)


def test_rasters_out_of_time_order():
    raster = read_spectrograph_lvl2(CROP)
    table = calculate_wavelength_drift(raster)
    later_first = RasterCollection(
        [(key, SpectrogramCubeSequence([window.data[0][3:], window.data[0][:3]])) for key, window in raster.items()],
        aligned_axes=(0, 1, 2),
    )
    swapped = calculate_wavelength_drift(later_first)
    assert list(swapped["raster"]) == [1, 1, 1, 0, 0, 0, 0, 0]
    assert list(swapped["step"]) == [0, 1, 2, 0, 1, 2, 3, 4]
    assert all(swapped["time"] == table["time"])
    for name in [*LINES, "nuv", "fuv"]:
        np.testing.assert_array_equal(swapped[name], table[name])


def test_drift_fit_matches_idl():
    idl = idl_reference("3824262996")
    for drift, line, outlier in [("nuv", "Ni I", 0.08), ("fuv", "O I", 0.05)]:
        fitted = _fit_drift(seconds(idl), idl[line].to_value(u.AA), outlier, drift)
        np.testing.assert_allclose(fitted, idl[drift].to_value(u.AA), atol=1e-8)


def test_drift_fit_leaves_out_missing_shifts():
    # IDL keeps the exposure without a Ni I shift in its fit, so MPFIT returns its starting guess:
    # the IDL "nuv" column is 0.1 + 0.01 sin(2πt/5856 s + 1) Å, about 0.1 Å from the shifts.
    idl = idl_reference("4000005156")
    shifts = idl["Ni I"].to_value(u.AA)
    assert np.isnan(shifts).any()  # fixture check
    with pytest.warns(UserWarning, match=f"{SHORT}, so the NUV drift is constant"):
        fitted = _fit_drift(seconds(idl), shifts, 0.08, "nuv")
    np.testing.assert_allclose(fitted, np.nanmean(shifts))


def test_drift_fit_of_a_short_observation():
    # The O I shifts of 3602506433 r00004, which span 36 s: a sine through the three that are not
    # outliers reached -3.3 Å
    times = np.arange(8) * 5.0
    shifts = np.array([np.nan, np.nan, np.nan, 1.5, np.nan, -109.4, -22.5, -115.2]) * 1e-3
    with pytest.warns(UserWarning, match=f"{SHORT}, so the FUV drift is constant"):
        fitted = _fit_drift(times, shifts, 0.05, "fuv")
    np.testing.assert_allclose(fitted, np.mean(shifts[5:]))


def test_drift_fit_just_under_a_quarter_orbit():
    times = np.arange(8) * 209.0  # 1463 s
    with pytest.warns(UserWarning, match=f"{SHORT}, so the FUV drift is constant"):
        _fit_drift(times, 0.01 * np.sin(times / 900), 0.05, "fuv")


@pytest.mark.parametrize("last_shift", [np.nan, 0.5])
def test_drift_fit_of_short_measured_coverage(last_shift):
    idl = idl_reference("3824262996")
    times = seconds(idl)
    shifts = idl["Ni I"].to_value(u.AA).copy()
    shifts[8:] = np.nan  # only 221 s measured in a raster lasting over three hours
    shifts[-1] = last_shift  # a distant outlier must not extend the coverage either
    with pytest.warns(UserWarning, match=f"{SHORT}, so the NUV drift is constant"):
        fitted = _fit_drift(times, shifts, 0.08, "nuv")
    np.testing.assert_allclose(fitted, fitted[0])
    assert shifts[:8].min() <= fitted[0] <= shifts[:8].max()


def test_drift_fit_coverage_excludes_smoothing():
    times = np.arange(200) * 30.0
    shifts = np.full(len(times), np.nan)
    shifts[:48] = 0.01 * np.sin(times[:48] / 900)  # 1410 s; smoothing extends this past a quarter orbit
    with pytest.warns(UserWarning, match=f"{SHORT}, so the NUV drift is constant"):
        fitted = _fit_drift(times, shifts, 0.08, "nuv")
    np.testing.assert_allclose(fitted, fitted[0])


def test_drift_fit_polynomial_order_uses_measured_coverage():
    times = np.arange(100) * 300.0
    orbits = times / 5856
    shifts = np.full(len(times), np.nan)
    measured = slice(20, 60)  # less than two measured orbits in a five-orbit observation
    shifts[measured] = 0.01 * np.sin(2 * np.pi * orbits[measured] + 0.3) + 5e-4 * orbits[measured] ** 2
    fitted = _fit_drift(times, shifts, 0.08, "nuv")
    expected = _fit_drift(times[measured], shifts[measured], 0.08, "nuv")
    np.testing.assert_allclose(fitted[measured], expected, atol=1e-12)


@pytest.mark.parametrize(("measured", "fits"), [(3, False), (4, True)])
def test_drift_fit_needs_more_shifts_than_parameters(measured, fits):
    times = np.arange(8) * 210.0  # 1470 s, just over a quarter orbit: 3 parameters, a sine and a constant
    shifts = np.full(len(times), np.nan)
    bins = np.linspace(0, len(times) - 1, measured, dtype=int)
    shifts[bins] = 0.01 * np.sin(times[bins] / 900)
    if fits:
        assert np.isfinite(_fit_drift(times, shifts, 0.05, "fuv")).all()
    else:
        with pytest.warns(UserWarning, match="Too few shifts to fit the FUV drift"):
            assert np.isnan(_fit_drift(times, shifts, 0.05, "fuv")).all()


def test_drift_fit_counts_only_measured_shifts():
    # The running mean spreads 4 shifts over 20 exposures, but the orbital fit has 4 parameters
    times = np.arange(130) * 60.0
    shifts = np.full(len(times), np.nan)
    shifts[[10, 40, 70, 110]] = 0.01
    with pytest.warns(UserWarning, match="Too few shifts to fit the NUV drift"):
        assert np.isnan(_fit_drift(times, shifts, 0.08, "nuv")).all()


@pytest.mark.parametrize("measured", [0, 1])
def test_constant_drift_needs_more_than_one_shift(measured):
    times = np.arange(200) * 30.0
    shifts = np.full(len(times), np.nan)
    shifts[:measured] = 0.01
    with pytest.warns(UserWarning, match="Too few shifts to fit the NUV drift"):
        assert np.isnan(_fit_drift(times, shifts, 0.08, "nuv")).all()


def test_drift_fit_drops_outliers_on_both_sides():
    times = np.arange(200) * 30.0
    drift = 0.01 * np.sin(2 * np.pi * times / 5856 + 0.3)
    shifts = drift.copy()
    shifts[[50, 150]] = np.median(drift) + np.array([0.1, -0.1])
    np.testing.assert_allclose(_fit_drift(times, shifts, 0.08, "nuv"), _fit_drift(times, drift, 0.08, "nuv"), atol=1e-5)


def test_drift_fit_polynomial_order_is_at_most_3():
    # 4.5 orbits at a cadence of 300 s, where the running mean is one exposure wide
    times = np.arange(88) * 300.0
    orbits = times / 5856
    shifts = 0.01 * np.sin(2 * np.pi * orbits + 0.3) + 1e-4 * orbits**3
    np.testing.assert_allclose(_fit_drift(times, shifts, 0.08, "nuv"), shifts, atol=1e-10)
    shifts += 2e-5 * orbits**4
    assert not np.allclose(_fit_drift(times, shifts, 0.08, "nuv"), shifts, atol=1e-5)


def test_drift_fit_across_a_gap():
    times = np.arange(40) * 60.0
    shifts = 0.01 * np.sin(times / 180)
    shifts[10:25] = np.nan  # longer than the 5-minute smoothing window
    assert np.isfinite(_fit_drift(times, shifts, 0.05, "nuv")).all()


@pytest.mark.parametrize(
    ("size", "sign", "expected"),
    [
        (12, 1, -0.02),  # 3 pixels in the fit range, padded to 5
        (7, 1, -0.02),  # padded to the 4 the window has, one per parameter
        (6, 1, np.nan),  # 3 pixels, fewer than the parameters
        (3, 1, np.nan),  # no pixel in the fit range
        (12, -1, np.nan),  # an absorption line where O I is in emission
    ],
)
def test_line_shifts_of_a_synthetic_line(size, sign, expected):
    np.testing.assert_allclose(synthetic_o_i_shifts(size, sign), expected, atol=1e-9)


def test_line_shifts_of_a_faint_line():
    # Its slit mean is below the 0.5 DN threshold of O I
    assert np.isnan(synthetic_o_i_shifts(12, scale=1e-3)).all()


def test_line_shifts_on_a_sloping_background():
    # 10 pixels in the fit range, so the background is linear
    np.testing.assert_allclose(synthetic_o_i_shifts(30, pixel=0.05, slope=20), -0.02, atol=1e-9)


def test_masked_pixels_count_as_zero():
    cube = read_spectrograph_lvl2(CROP, spectral_windows="Mg II k 2796")["Mg II k 2796"].data[0]
    wavelength = _wavelengths(cube)
    cube.data[:, :300] = 0
    zeroed = _line_shifts(cube, wavelength, *_LINES["Ni I"])
    cube.data[:, :300] = 5000
    cube.mask[:, :300] = True
    np.testing.assert_array_equal(_line_shifts(cube, wavelength, *_LINES["Ni I"]), zeroed)


def test_window_chosen_by_its_wavelengths():
    raster = read_spectrograph_lvl2(CROP)
    cut = raster["Mg II k 2796"].data[0][..., :5]
    assert cut.meta.spectral_range[1] > _LINES["Ni I"][0] * u.AA  # fixture check: the header still covers Ni I
    first_cut = RasterCollection([("cut", SpectrogramCubeSequence([cut])), *raster.items()], aligned_axes=(0, 1, 2))
    np.testing.assert_array_equal(
        calculate_wavelength_drift(first_cut)["Ni I"], calculate_wavelength_drift(raster)["Ni I"]
    )


def test_memmap_data_give_the_same_table():
    # memmap=True leaves the FITS integers, which are scaled to DN for the fit ranges only
    table = calculate_wavelength_drift(read_spectrograph_lvl2(CROP))
    memmap = calculate_wavelength_drift(read_spectrograph_lvl2(CROP, memmap=True))
    for name in [*LINES, "nuv", "fuv"]:
        np.testing.assert_array_equal(memmap[name], table[name])


def test_missing_lines():
    nuv_only = read_spectrograph_lvl2(CROP, spectral_windows="Mg II k 2796")
    table = calculate_wavelength_drift(nuv_only)  # no warning: the FUV lines are in no window
    assert np.isnan(table["O I"]).all()
    assert np.isnan(table["fuv"]).all()
    assert np.isfinite(table["nuv"]).all()
    sns = get_test_filepath("sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000_test.fits")
    with pytest.raises(ValueError, match="no window with any of the reference lines"):
        calculate_wavelength_drift(read_spectrograph_lvl2(sns, spectral_windows="C II 1336"))
    # Its O I window ends 3 pixels after the start of the fit range
    with pytest.warns(UserWarning, match="Too few shifts"):
        table = calculate_wavelength_drift(read_spectrograph_lvl2(sns))
    assert np.isnan(table["O I"]).all()


@pytest.mark.remote_data
@pytest.mark.filterwarnings(f"ignore:.*{SHORT}")  # it spans 184 s
def test_whole_file_matches_idl():
    # Every step of the windows with the reference lines, cut to their fit ranges (see overview.txt)
    filename = pooch.retrieve(
        f"{IRISPY_DATA}/iris_l2_20130902_182935_4000005156_raster_t000_r00000_wavelength_drift.fits.gz",
        known_hash="c9f1166b18acae244c1ad9d015d04b2390e81adf8dd00edf5c012b0a5d726141",
    )
    assert_shifts_match_idl(calculate_wavelength_drift(read_spectrograph_lvl2(filename)), idl_reference("4000005156"))
