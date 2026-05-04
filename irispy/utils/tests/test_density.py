import sys
import types

import numpy as np
import pytest

import astropy.units as u

from irispy.utils.density import density_diagnostic, map_ratio_to_quantity


def _install_fake_fiasco(monkeypatch, ratio):
    calls = {}

    def line_ratio(ion, numerator, denominator, density_grid, **kwargs):
        calls["ion"] = ion
        calls["numerator"] = numerator
        calls["denominator"] = denominator
        calls["density_grid"] = density_grid
        calls["line_ratio_kwargs"] = kwargs
        return u.Quantity(ratio, u.dimensionless_unscaled)

    fake_fiasco = types.SimpleNamespace(line_ratio=line_ratio)
    monkeypatch.setitem(sys.modules, "fiasco", fake_fiasco)
    return calls


def test_map_ratio_to_quantity():
    density = [1e8, 1e9, 1e10] * u.cm**-3
    theoretical_ratio = [0.2, 0.5, 0.8] * u.dimensionless_unscaled

    mapped_density = map_ratio_to_quantity([0.2, 0.65, 1.0], density, theoretical_ratio)

    assert mapped_density.unit == density.unit
    assert mapped_density.shape == (3,)
    assert u.allclose(mapped_density[:2], [1e8, 5.5e9] * u.cm**-3)
    assert np.isnan(mapped_density[-1].value)


def test_map_ratio_to_quantity_decreasing_curve():
    density = [1e8, 1e9, 1e10] * u.cm**-3
    theoretical_ratio = [0.8, 0.5, 0.2] * u.dimensionless_unscaled

    assert u.allclose(map_ratio_to_quantity(0.35, density, theoretical_ratio), 5.5e9 * u.cm**-3)


def test_map_ratio_to_quantity_rejects_non_monotonic_curve():
    density = [1e8, 1e9, 1e10] * u.cm**-3
    theoretical_ratio = [0.2, 0.8, 0.5] * u.dimensionless_unscaled

    with pytest.raises(ValueError, match="must be monotonic"):
        map_ratio_to_quantity(0.35, density, theoretical_ratio)


def test_density_diagnostic(monkeypatch):
    _install_fake_fiasco(monkeypatch, [0.5, 1.0])
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5] * u.K,
        formation_temperature=2e5 * u.K,
    )

    result = density_diagnostic(
        [1.0, 1.5] * u.ct,
        [2.0, 2.0] * u.ct,
        [1e10, 1e11] * u.cm**-3,
        ion=fake_ion,
        numerator=1399.78 * u.angstrom,
        denominator=1401.16 * u.angstrom,
        intensity_numerator_uncertainty=[0.1, 0.1] * u.ct,
        intensity_denominator_uncertainty=[0.2, 0.2] * u.ct,
    )

    np.testing.assert_allclose(result["ratio"].value, [0.5, 0.75])
    assert result["ratio_uncertainty"] is not None
    assert result["density_lower"] is not None
    assert result["density_upper"] is not None
    assert u.allclose(result["density"], [1e10, 5.5e10] * u.cm**-3)


def test_density_diagnostic_builds_theoretical_ratio_with_fiasco(monkeypatch):
    calls = _install_fake_fiasco(
        monkeypatch,
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
        ],
    )
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5, 3e5] * u.K,
        formation_temperature=2e5 * u.K,
    )
    density_grid = [1, 2, 3] * u.cm**-3

    result = density_diagnostic(
        [0.4],
        [1.0],
        density_grid,
        ion=fake_ion,
        numerator=1399.78 * u.angstrom,
        denominator=1401.16 * u.angstrom,
        line_ratio_kwargs={"use_two_ion_model": False},
    )

    assert calls["ion"] is fake_ion
    assert u.allclose(calls["density_grid"], density_grid)
    assert calls["line_ratio_kwargs"] == {"use_two_ion_model": False}
    np.testing.assert_allclose(result["theoretical_ratio"].value, [0.2, 0.4, 0.6])
    assert u.allclose(result["density"], [2] * u.cm**-3)


def test_density_diagnostic_requires_ion():
    with pytest.raises(ValueError, match="ion, numerator, and denominator are required"):
        density_diagnostic(
            [4, 8] * u.ct,
            [2, 4] * u.ct,
            [1e10, 1e11] * u.cm**-3,
            ion=None,
            numerator=1399.78 * u.angstrom,
            denominator=1401.16 * u.angstrom,
        )


def test_density_diagnostic_selects_monotonic_branch(monkeypatch):
    _install_fake_fiasco(monkeypatch, [0.1, 0.2, 0.3, 0.25, 0.2])
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5] * u.K,
        formation_temperature=2e5 * u.K,
    )

    result = density_diagnostic(
        [2.4, 2.8],
        [10.0, 10.0],
        density_grid=[1, 2, 3, 4, 5] * u.cm**-3,
        ion=fake_ion,
        numerator=1399.78 * u.angstrom,
        denominator=1401.16 * u.angstrom,
    )

    np.testing.assert_allclose(result["theoretical_ratio"].value, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result["density_grid"].value, [1, 2, 3])


def test_density_diagnostic_zero_denominator(monkeypatch):
    _install_fake_fiasco(monkeypatch, [0.5, 1.0])
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5] * u.K,
        formation_temperature=2e5 * u.K,
    )

    result = density_diagnostic(
        [1.0, 1.5] * u.ct,
        [2.0, 0.0] * u.ct,
        [1e10, 1e11] * u.cm**-3,
        ion=fake_ion,
        numerator=1399.78 * u.angstrom,
        denominator=1401.16 * u.angstrom,
    )

    np.testing.assert_allclose(result["ratio"].value, [0.5, np.nan])
    assert result["ratio_uncertainty"] is None
    assert result["density_lower"] is None
    assert result["density_upper"] is None


def test_density_diagnostic_partial_uncertainty_raises():
    with pytest.raises(ValueError, match="Both numerator and denominator uncertainties"):
        density_diagnostic(
            [4, 8] * u.ct,
            [2, 4] * u.ct,
            [1e10, 1e11] * u.cm**-3,
            ion=types.SimpleNamespace(),
            numerator=1399.78 * u.angstrom,
            denominator=1401.16 * u.angstrom,
            intensity_numerator_uncertainty=[0.1, 0.1] * u.ct,
        )


def test_density_diagnostic_explicit_temperature(monkeypatch):
    """
    Regression: explicit scalar temperature must not be clobbered by
    formation_temperature.
    """
    _install_fake_fiasco(
        monkeypatch,
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
        ],
    )
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5, 3e5] * u.K,
        formation_temperature=1e5 * u.K,
    )
    density_grid = [1, 2, 3] * u.cm**-3

    result = density_diagnostic(
        [0.4],
        [1.0],
        density_grid,
        ion=fake_ion,
        numerator=1399.78 * u.angstrom,
        denominator=1401.16 * u.angstrom,
        temperature=2e5 * u.K,
    )

    # If temperature=2e5 K were ignored, the fallback formation temperature
    # would use the first row and return density=1.
    assert u.allclose(result["density"], [2] * u.cm**-3)


def test_density_diagnostic_rejects_vector_temperature(monkeypatch):
    _install_fake_fiasco(
        monkeypatch,
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
        ],
    )
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5, 3e5] * u.K,
        formation_temperature=1e5 * u.K,
    )

    with pytest.raises(ValueError, match="temperature must be scalar"):
        density_diagnostic(
            [0.4],
            [1.0],
            [1, 2, 3] * u.cm**-3,
            ion=fake_ion,
            numerator=1399.78 * u.angstrom,
            denominator=1401.16 * u.angstrom,
            temperature=[1e5, 2e5] * u.K,
        )


def test_density_diagnostic_uncertainty_shape_mismatch(monkeypatch):
    _install_fake_fiasco(monkeypatch, [0.5, 1.0])
    fake_ion = types.SimpleNamespace(
        temperature=[1e5, 2e5] * u.K,
        formation_temperature=2e5 * u.K,
    )

    with pytest.raises(ValueError, match="Uncertainty shapes"):
        density_diagnostic(
            [1.0, 1.5] * u.ct,
            [2.0, 2.0] * u.ct,
            [1e10, 1e11] * u.cm**-3,
            ion=fake_ion,
            numerator=1399.78 * u.angstrom,
            denominator=1401.16 * u.angstrom,
            intensity_numerator_uncertainty=[[0.1], [0.1]] * u.ct,
            intensity_denominator_uncertainty=[[0.2], [0.2]] * u.ct,
        )
