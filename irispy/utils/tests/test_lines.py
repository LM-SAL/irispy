import io
import csv
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import astropy.units as u

from irispy.utils import lines
from irispy.utils.lines import get_lines

GENERATOR = Path(__file__).resolve().parents[3] / "tools" / "make_line_database.py"

DOCUMENTED = [
    ("C II", 1334.5323, "depontieu2014"),
    ("C II", 1335.6628, "rathore2015"),
    ("C II", 1335.7079, "depontieu2014"),
    ("Fe XII", 1349.40, "depontieu2014"),
    ("Cl I", 1351.657, "itn38"),
    ("Fe XXI", 1354.08, "depontieu2014"),
    ("C I", 1354.284, "young2015"),
    ("Mn XVIII", 1355.014, "depontieu2021"),
    ("O I", 1355.5977, "depontieu2014"),
    ("C I", 1355.843, "lin2017"),
    ("Fe II", 1392.817, "wulser2018"),
    ("Ni II", 1393.330, "itn38"),
    ("Si IV", 1393.76, "depontieu2014"),
    ("O IV", 1399.776, "depontieu2014"),
    ("O IV", 1401.157, "depontieu2014"),
    ("Si IV", 1402.77, "depontieu2014"),
    ("O IV", 1404.806, "polito2016"),
    ("S IV", 1404.808, "polito2016"),
    ("S IV", 1406.009, "polito2016"),
    ("Mg II", 2791.599, "pereira2015"),
    ("Mg II", 2796.352, "depontieu2014"),
    ("Mg II", 2798.754, "pereira2015"),
    ("Mg II", 2798.823, "pereira2015"),
    ("Ni I", 2799.47, "wulser2018"),
    ("Mg II", 2803.530, "depontieu2014"),
]
REGIONS = ["quiet_sun", "active_region", "flare"]
ABUNDANCES = ["coronal", "photospheric"]
INTENSITY_COLUMNS = [f"intensity_{region}_{abundance}" for region in REGIONS for abundance in ABUNDANCES]


def test_documented_lines():
    table = get_lines(main_only=True)
    assert len(table) == len(DOCUMENTED)
    for ion, wavelength, reference in DOCUMENTED:
        (line,) = table[(table["ion"] == ion) & (np.abs(table["wavelength"].value - wavelength) <= 0.01)]
        assert line["reference"] == reference
    # Lines from neither catalog carry the reference's wavelength and no prediction.
    assert set(table["wavelength_source"][table["ion"] == "Ni II"]) == {"literature"}
    full = get_lines()
    np.testing.assert_array_equal(full["main"], full["reference"].filled("") != "")


def test_mg_ii_channels_are_unranked():
    # NIST lists Mg II 2798.75 as a strong E1 and a weak M2 channel between the same levels.
    mg_ii = get_lines([2798.75, 2798.76] * u.angstrom)
    mg_ii = mg_ii[mg_ii["ion"] == "Mg II"]
    e1, m2 = mg_ii[np.argsort(mg_ii["nist_transition_probability"].value)[::-1]]
    assert e1["main"]
    assert m2["transition_type"] == "M2"
    assert np.isnan(e1["intensity_quiet_sun_coronal"])
    assert np.isnan(m2["intensity_quiet_sun_coronal"])


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("abundance", ABUNDANCES)
def test_si_iv_strongest_in_fuv2(region, abundance):
    strongest = get_lines([1388.5, 1407.5] * u.angstrom, region=region, abundance=abundance)[0]
    assert strongest["ion"] == "Si IV"
    assert strongest["wavelength"].to_value(u.angstrom) == pytest.approx(1393.76, abs=0.001)


def test_normalized_per_passband():
    table = get_lines()
    for passband in ("FUV1", "FUV2", "NUV"):
        for column in INTENSITY_COLUMNS:
            assert np.nanmax(table[table["passband"] == passband][column]) == pytest.approx(1)


def test_neutral_and_singly_ionized_lines_unranked():
    table = get_lines()
    excluded = table[table["ion_stage"] <= 2]
    assert {"Mg II", "C II", "O I", "Fe II", "Al II"} <= set(excluded["ion"])
    assert np.all(np.isnan([excluded[column] for column in INTENSITY_COLUMNS]))


def test_categories():
    table = get_lines()
    categories = table["category"].filled("")
    # Categories describe the ion, so Mg II's unranked M2 entries are chromospheric too.
    assert set(categories[table["ion"] == "Mg II"]) == {"chromospheric"}
    assert set(categories[table["ion"] == "Si IV"]) == {"transition_region"}
    assert set(categories[table["ion"] == "Fe XII"]) == {"coronal"}
    assert set(categories[table["ion"] == "Fe XXI"]) == {"flare"}
    # These metals may appear in photospheric absorption or chromospheric emission.
    cool_metals = table[categories == "cool_metal"]
    assert {"Si II", "Fe I", "Fe II"} <= set(cool_metals["ion"])
    assert not set(cool_metals["element"]) & {"W", "Pt", "Th", "Pu"}
    assert np.all(np.isnan([cool_metals[column] for column in INTENSITY_COLUMNS]))
    assert set(get_lines(categories="cool_metal")["category"]) == {"cool_metal"}
    chromospheric = get_lines(categories="chromospheric", main_only=True)
    assert {"Mg II", "C II", "C I", "O I", "Cl I"} <= set(chromospheric["ion"])
    assert np.isnan([chromospheric[column] for column in INTENSITY_COLUMNS]).all()
    assert set(get_lines(categories=["flare", "transition_region"], main_only=True)["category"]) == {
        "flare",
        "transition_region",
    }
    assert len(get_lines(categories=[])) == 0


@pytest.mark.parametrize("unit", [u.angstrom, u.nm, u.Hz, u.eV])
def test_spectral_range(unit):
    endpoints = ([1402, 1404] * u.angstrom).to(unit, equivalencies=u.spectral())
    expected = get_lines([1402, 1404] * u.angstrom)
    np.testing.assert_array_equal(get_lines(endpoints)["wavelength"], expected["wavelength"])
    np.testing.assert_array_equal(get_lines(endpoints[::-1])["wavelength"], expected["wavelength"])


def test_range_includes_endpoints():
    wavelength = get_lines(main_only=True)["wavelength"][0]
    result = get_lines([wavelength, wavelength])
    assert len(result) >= 1
    assert np.all(result["wavelength"] == wavelength)
    assert len(get_lines([1, 2] * u.angstrom)) == 0


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("abundance", ABUNDANCES)
def test_ranking(region, abundance):
    table = get_lines(region=region, abundance=abundance)
    intensity = table[f"intensity_{region}_{abundance}"]
    band_order = np.array([("FUV1", "FUV2", "NUV").index(band) for band in table["passband"]])
    assert np.all(np.diff(band_order) >= 0)
    # Strengths are comparable only within one passband, including the placement of unranked lines.
    for band in ("FUV1", "FUV2", "NUV"):
        values = intensity[table["passband"] == band]
        ranked = np.isfinite(values)
        assert np.all(np.diff(values[ranked]) <= 0)
        assert np.all(ranked[: ranked.sum()])
        assert not np.any(ranked[ranked.sum() :])
    ranked_only = get_lines(region=region, abundance=abundance, include_unranked=False)
    assert len(ranked_only) == np.isfinite(intensity).sum()


def test_include_unranked_without_region():
    table = get_lines(include_unranked=False)
    assert not np.any(np.all(np.isnan([table[column] for column in INTENSITY_COLUMNS]), axis=0))


def test_default_order():
    table = get_lines()
    assert np.all(np.diff(table["wavelength"].value) >= 0)
    # Mg II's E1 and M2 entries share wavelength, ion, and levels; ties keep the packaged order.
    packaged = lines._load_lines()
    for column in ("ion", "lower", "upper", "transition_type", "wavelength_source"):
        np.testing.assert_array_equal(np.asarray(table[column]), np.asarray(packaged[column]))


def test_results_are_independent():
    changed = get_lines()
    changed["wavelength"][0] = 1 * u.angstrom
    changed.meta["passband_limits_angstrom"]["FUV1"][0] = 0
    fresh = get_lines()
    assert fresh["wavelength"][0] != 1 * u.angstrom
    assert fresh.meta["passband_limits_angstrom"]["FUV1"][0] != 0


def test_wavelength_provenance():
    table = get_lines()
    assert table["wavelength_is_theoretical"].dtype.kind == "b"
    assert table["wavelength_is_theoretical"].unit is None
    (c_i,) = table[(table["ion"] == "C I") & (abs(table["wavelength"].value - 1354.28888) < 1e-5)]
    assert c_i["wavelength_source"] == "ritz"
    assert c_i["wavelength_uncertainty"] == c_i["ritz_wavelength_uncertainty"]
    assert c_i["wavelength_uncertainty"] < c_i["observed_wavelength_uncertainty"]
    chianti = table[table["wavelength_source"] == "chianti"]
    assert chianti["wavelength_is_theoretical"].any()
    assert (~chianti["wavelength_is_theoretical"]).any()
    assert not table["wavelength_is_theoretical"][table["wavelength_source"] != "chianti"].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"region": "sunspot", "abundance": "coronal"},
        {"region": "quiet_sun"},
        {"abundance": "coronal"},
        {"abundance": "feldman"},
        {"categories": "unknown"},
        {"wavelength_range": [1, 2, 3] * u.angstrom},
        {"wavelength_range": [0, 1] * u.angstrom},
        {"wavelength_range": [np.nan, 1] * u.angstrom},
        {"wavelength_range": [np.inf, 1] * u.eV},
        {"wavelength_range": 1400 * u.angstrom},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError, match=r"region|abundance|categories|wavelength_range"):
        get_lines(**kwargs)


def test_non_spectral_units():
    with pytest.raises(u.UnitConversionError):
        get_lines([1, 2] * u.s)
    with pytest.raises(u.UnitConversionError):
        get_lines([1402, 1404])


@pytest.mark.skipif(not GENERATOR.exists(), reason="The generator is only in development checkouts.")
def test_merge_lines(monkeypatch):
    pytest.importorskip("plasmapy")
    spec = importlib.util.spec_from_file_location("make_line_database", GENERATOR)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(
        tool,
        "DOCUMENTED",
        [
            ("C II", 1335.7079, "depontieu2014"),
            ("Mn XVIII", 1355.014, "depontieu2021"),
            ("Fe II", 1392.817, "wulser2018"),
            ("O IV", 1401.157, "depontieu2014"),
            ("Mg II", 2798.754, "pereira2015"),
        ],
    )
    levels = {"lower": "a", "upper": "b", "lower_energy": 0.0, "upper_energy": 7e4, "lower_j": 0.5, "upper_j": 1.5}

    def nist(ion, wavelength, a=1e8, kind="", uncertainty=0.001, **overrides):
        element, stage = ion.split()
        return {
            "ion": ion,
            "element": element,
            "ion_stage": tool.roman.from_roman(stage),
            "wavelength": wavelength,
            "wavelength_source": "observed",
            "wavelength_uncertainty": uncertainty,
            "transition_type": kind,
            "nist_transition_probability": a,
            "passband": tool._passband(wavelength),
            "log_t_max": np.nan,
            "category": "",
            "main": False,
            "reference": "",
            **levels,
            **dict.fromkeys(tool.INTENSITY_COLUMNS, np.nan),
            **overrides,
        }

    def predicted(ion, wavelength, strength, **overrides):
        element, stage = ion.split()
        return {
            "ion": ion,
            "element": element,
            "ion_stage": tool.roman.from_roman(stage),
            "wavelength": wavelength,
            "passband": tool._passband(wavelength),
            **levels,
            **dict.fromkeys(tool.INTENSITY_COLUMNS, strength),
            **overrides,
        }

    no_levels = dict.fromkeys(("lower_energy", "upper_energy", "lower_j", "upper_j"), np.nan)
    rows = [
        nist("C II", 1335.7079),
        nist("Ca II", 1341.89, uncertainty=np.nan, lower="", upper="", **no_levels),
        nist("Fe XII", 1349.4),
        nist("Fe XXI", 1354.08),
        nist("Si IV", 1393.76),
        nist("O IV", 1401.15, upper="c", upper_energy=71300.0),
        nist("O IV", 1401.157),
        nist("Mg II", 2798.754, a=4.7e8),
        nist("Mg II", 2798.754, a=1e-2, kind="M2"),
        nist("Fe I", 2805.3465),
    ]
    predictions = [
        predicted("C II", 1335.708, 10.0),
        predicted("Ca II", 1341.9, 1.0),
        predicted("O IV", 1343.5, 1e-6),
        predicted("Mn XVIII", 1355.014, 1e-6),
        predicted("O IV", 1401.157, 2.0),
        predicted("Mg II", 2798.753, 5.0),
    ]
    formation = {
        "C II": 4.4,
        "Ca II": 4.05,
        "Mn XVIII": 6.9,
        "O IV": 5.15,
        "Si IV": 4.9,
        "Fe XII": 6.15,
        "Fe XXI": 7.05,
        "Mg II": 4.15,
        "Fe II": 4.15,
    }
    merged = {
        (row["ion"], row["wavelength"], row["transition_type"]): row
        for row in tool.merge_lines(rows, predictions, formation)
    }

    c_ii = merged["C II", 1335.7079, ""]
    assert c_ii["intensity_quiet_sun_coronal"] == 1
    assert c_ii["reference"] == "depontieu2014"
    assert c_ii["main"]
    # A NIST line without level data takes the prediction instead of a CHIANTI-only duplicate.
    (ca_ii,) = [row for key, row in merged.items() if key[0] == "Ca II"]
    assert ca_ii["wavelength_source"] == "observed"
    assert ca_ii["intensity_quiet_sun_coronal"] == pytest.approx(0.1)
    # Documented lines take the closest catalog line; channels at one wavelength go by A-value.
    assert merged["O IV", 1401.157, ""]["main"]
    assert not merged["O IV", 1401.15, ""]["main"]
    assert np.isfinite(merged["Mg II", 2798.754, ""]["intensity_quiet_sun_coronal"])
    assert merged["Mg II", 2798.754, ""]["reference"] == "pereira2015"
    assert np.isnan(merged["Mg II", 2798.754, "M2"]["intensity_quiet_sun_coronal"])
    # Catalog inclusion is independent of strength; undocumented catalog gaps are filled from the reference.
    assert ("O IV", 1343.5, "") in merged
    assert merged["Mn XVIII", 1355.014, ""]["wavelength_source"] == "chianti"
    fe_ii = merged["Fe II", 1392.817, ""]
    assert fe_ii["wavelength_source"] == "literature"
    assert fe_ii["reference"] == "wulser2018"
    assert np.isnan([fe_ii[column] for column in tool.INTENSITY_COLUMNS]).all()
    categories = {key[0]: row["category"] for key, row in merged.items()}
    assert categories == {
        "C II": "chromospheric",
        "Ca II": "cool_metal",
        "Fe XII": "coronal",
        "Fe XXI": "flare",
        "Si IV": "transition_region",
        "O IV": "transition_region",
        "Mg II": "chromospheric",
        "Mn XVIII": "coronal",
        "Fe II": "cool_metal",
        "Fe I": "cool_metal",
    }


@pytest.mark.skipif(not GENERATOR.exists(), reason="The generator is only in development checkouts.")
def test_nist_wavelength_selection(monkeypatch):
    pytest.importorskip("plasmapy")
    spec = importlib.util.spec_from_file_location("make_line_database", GENERATOR)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    fields = [
        "element",
        "sp_num",
        "obs_wl_vac(A)",
        "ritz_wl_vac(A)",
        "unc_obs_wl",
        "unc_ritz_wl",
        "Type",
        "conf_i",
        "term_i",
        "J_i",
        "conf_k",
        "term_k",
        "J_k",
        "Ei(cm-1)",
        "Ek(cm-1)",
        "intens",
        "Aki(s^-1)",
        "line_ref",
    ]
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=fields)
    writer.writeheader()
    for element, stage, observed, ritz, obs_unc, ritz_unc in [
        ("C", 1, "1354.284", "1354.28888", "0.003", "0.00006"),
        ("Si", 4, "1393.76", "1393.7546", "0.001", "0.003"),
        ("Si", 4, "", "1402.7697", "", ""),
        ("Mg", 2, "2796.352", "2796.352", "", ""),
    ]:
        writer.writerow(dict(zip(fields[:6], (element, stage, observed, ritz, obs_unc, ritz_unc), strict=True)))
    monkeypatch.setattr(tool, "urlopen", lambda *_args, **_kwargs: io.BytesIO(content.getvalue().encode()))
    rows, _ = tool.download_nist()
    assert [(row["wavelength"], row["wavelength_source"]) for row in rows] == [
        (1354.28888, "ritz"),
        (1393.76, "observed"),
        (1402.7697, "ritz"),
        (2796.352, "observed"),
    ]
    assert rows[0]["wavelength_uncertainty"] == 0.00006
    assert rows[0]["observed_wavelength"] == 1354.284
    assert not any(row["wavelength_is_theoretical"] for row in rows)


@pytest.mark.skipif(not GENERATOR.exists(), reason="The generator is only in development checkouts.")
@pytest.mark.parametrize("ion_stage", [2, 3])
def test_unpredicted_ion_retains_catalog_transition(monkeypatch, ion_stage):
    pytest.importorskip("plasmapy")
    fiasco = pytest.importorskip("fiasco")
    spec = importlib.util.spec_from_file_location("make_line_database", GENERATOR)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ion_name = f"Mg {tool.roman.to_roman(ion_stage)}"
    raw_name = f"mg_{ion_stage}"
    ion = SimpleNamespace(
        ion_name_roman=ion_name,
        atomic_symbol="Mg",
        atomic_number=12,
        ionization_stage=ion_stage,
        n_levels=2,
        ionization_fraction=np.ones(1),
        formation_temperature=1e4 * u.K,
        transitions=SimpleNamespace(
            is_bound_bound=np.array([True]),
            is_observed=np.array([False]),
            wavelength=np.array([2796.35]) * u.AA,
            lower_level=np.array([1]),
            upper_level=np.array([3]),
        ),
        levels=SimpleNamespace(
            level=np.array([1, 3]),
            energy=np.array([0, 35761]) / u.cm,
            total_angular_momentum=np.array([0.5, 1.5]) * u.dimensionless_unscaled,
            label=np.array(["ground", "upper"]),
        ),
    )
    # Excluded ions must not solve populations; missing excitation data must also retain the catalog row.
    if ion_stage == 3:

        def unavailable_contribution(*_args, **_kwargs):
            msg = "No excitation data."
            raise fiasco.util.exceptions.MissingDatasetException(msg)

        ion.contribution_function = unavailable_contribution
    monkeypatch.setattr(fiasco, "list_ions", lambda **_kwargs: [raw_name])
    monkeypatch.setattr(fiasco, "Ion", lambda *_args, **_kwargs: ion)
    model = {"temperature_bin_center": np.array([1e4]) * u.K}
    monkeypatch.setattr(fiasco.io, "Parser", lambda *_args: SimpleNamespace(parse=lambda: model))
    rows, formation, unranked = tool.predict_lines()
    (row,) = rows
    assert row["ion"] == ion_name
    assert row["upper"] == "upper"  # Non-contiguous level identifiers still resolve correctly.
    assert row["wavelength_is_theoretical"]
    assert np.isnan([row[column] for column in INTENSITY_COLUMNS]).all()
    assert formation == {ion_name: 4.0}
    assert raw_name in unranked
