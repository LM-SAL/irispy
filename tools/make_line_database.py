"""
Update irispy/data/iris_lines.ecsv.

Needs ``pip install -e '.[density]'`` and a full CHIANTI database in fiasco's configured
location (https://fiasco.readthedocs.io/en/latest/quick_start.html). Run it, then review
the changes with ``git diff``. docs/line_database.rst explains the science.

python tools/make_line_database.py
"""

import io
import csv
import sys
import hashlib
from pathlib import Path
from datetime import UTC, datetime
from fractions import Fraction
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from plasmapy.utils import roman

import astropy
import astropy.units as u
from astropy.table import QTable

DATA_DIR = Path(__file__).resolve().parents[1] / "irispy" / "data"
NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
# Vacuum wavelengths in Angstrom from De Pontieu et al. (2014) Table 2, without margins. Level 2
# full-detector windows from 2015 and 2026 span 1331.68-1358.28, 1380.66-1406.70 and
# 2783.23-2835.10 A: the FUV2 readout below 1389.0 A is outside the documented passband.
PASSBANDS = {"FUV1": [1331.7, 1358.4], "FUV2": [1389.0, 1407.0], "NUV": [2782.7, 2835.1]}
PRESSURES = {"quiet_sun": 3e15, "active_region": 3e15, "flare": 1e16}  # n_e T in K cm^-3
ABUNDANCES = {"coronal": "sun_coronal_2021_chianti", "photospheric": "sun_photospheric_2021_asplund"}
# IRIS lines documented in the literature: ion, vacuum wavelength, and the key of the reference
# in docs/references.bib. Wavelengths are the NIST values where NIST lists the line, else
# CHIANTI's, else the reference's; a line in neither catalog is added from the reference.
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
# Formation-temperature boundaries of the categories, in log10 K. Tian (2017) defines the
# transition region as 0.02-0.8 MK; the Fe XXI flare line forms at 10 MK and above
# (De Pontieu et al. 2021).
CATEGORY_BOUNDARIES = [("flare", 7.0), ("coronal", 5.9), ("transition_region", 4.3)]
INTENSITY_COLUMNS = [f"intensity_{region}_{abundance}" for region in PRESSURES for abundance in ABUNDANCES]
LEVELS = ("lower", "upper", "lower_energy", "upper_energy", "lower_j", "upper_j")
# CHIANTI-only lines are kept if they reach this fraction of their passband's strongest line.
THRESHOLD = 1e-3
# Catalog identity tolerance for level energies and transition wavenumbers, in cm^-1.
# Accommodates Fe XII's 2 cm^-1 catalog difference; not a measurement uncertainty.
ENERGY_TOLERANCE = 5.0
# Neutral or singly ionized metals that can appear as photospheric lines.
PHOTOSPHERIC_METALS = {"Li", "Be", "Na", "Mg", "Al", "Si", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
# Vacuum wavelengths in Angstrom for all spectra, with levels, A-values, and references.
NIST_PARAMETERS = {
    "spectra": "",
    "unit": 0,
    "format": 2,
    "line_out": 0,
    "output": 0,
    "page_size": 50000,
    "show_obs_wl": 1,
    "show_calc_wl": 1,
    "unc_out": 1,
    "order_out": 0,
    "show_av": 3,
    "allowed_out": 1,
    "forbid_out": 1,
    "conf_out": "on",
    "term_out": "on",
    "J_out": "on",
    "enrg_out": "on",
    "intens_out": "on",
    "bibrefs": 1,
    "remove_js": "on",
}
CHIANTI_ONLY = {
    "wavelength_source": "chianti",
    "wavelength_uncertainty": np.nan,
    "observed_wavelength": np.nan,
    "observed_wavelength_uncertainty": np.nan,
    "ritz_wavelength": np.nan,
    "ritz_wavelength_uncertainty": np.nan,
    "transition_type": "",
    "nist_intensity": "",
    "nist_transition_probability": np.nan,
    "nist_line_reference": "",
    "main": False,
    "reference": "",
    "category": "",
}
LITERATURE_ONLY = {
    **CHIANTI_ONLY,
    "wavelength_source": "literature",
    **dict.fromkeys(("lower", "upper"), ""),
    **dict.fromkeys(("lower_energy", "upper_energy", "lower_j", "upper_j", "log_t_max"), np.nan),
    **dict.fromkeys(INTENSITY_COLUMNS, np.nan),
}


def _text(value):
    # NIST CSV wraps text as Excel string expressions. Do not evaluate them.
    value = value.strip()
    return value[2:-1] if value.startswith('="') and value.endswith('"') else value


def _number(value):
    value = _text(value).strip("[]?*+ ")
    return float(value) if value else np.nan


def _level_number(value):
    # J values and energies that are ambiguous or qualified ("1/2,3/2", "[123]", "123+x") become NaN.
    try:
        return float(Fraction(_text(value)))
    except (ValueError, ZeroDivisionError):
        return np.nan


def _passband(wavelength):
    for band, (low, high) in PASSBANDS.items():
        if low <= wavelength <= high:
            return band
    return None


def download_nist():
    """
    Download every NIST line in each passband; return the rows and the queries made.
    """
    rows, queries = [], {}
    for band, (low, high) in PASSBANDS.items():
        parameters = dict(NIST_PARAMETERS, low_w=low, upp_w=high)
        url = f"{NIST_URL}?{urlencode(parameters)}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (irispy line database generator)"})  # noqa: S310
        with urlopen(request, timeout=120) as response:  # noqa: S310
            content = response.read().decode("utf-8-sig")
        queries[band] = {"url": url, "parameters": parameters, "query_date": datetime.now(UTC).isoformat()}
        for record in csv.DictReader(io.StringIO(content)):
            observed = _number(record["obs_wl_vac(A)"])
            ritz = _number(record["ritz_wl_vac(A)"])
            source = "observed" if np.isfinite(observed) else "ritz"
            wavelength = observed if source == "observed" else ritz
            if not low <= wavelength <= high:
                continue
            element = _text(record["element"])
            stage = int(record["sp_num"])
            rows.append(
                {
                    "ion": f"{element} {roman.to_roman(stage)}",
                    "element": element,
                    "ion_stage": stage,
                    "wavelength": wavelength,
                    "wavelength_source": source,
                    "wavelength_uncertainty": _number(record["unc_obs_wl" if source == "observed" else "unc_ritz_wl"]),
                    "observed_wavelength": observed,
                    "observed_wavelength_uncertainty": _number(record["unc_obs_wl"]),
                    "ritz_wavelength": ritz,
                    "ritz_wavelength_uncertainty": _number(record["unc_ritz_wl"]),
                    "transition_type": _text(record["Type"]),
                    "lower": " ".join(_text(record[key]) for key in ("conf_i", "term_i", "J_i")).strip(),
                    "upper": " ".join(_text(record[key]) for key in ("conf_k", "term_k", "J_k")).strip(),
                    "lower_energy": _level_number(record["Ei(cm-1)"]),
                    "upper_energy": _level_number(record["Ek(cm-1)"]),
                    "lower_j": _level_number(record["J_i"]),
                    "upper_j": _level_number(record["J_k"]),
                    "nist_intensity": _text(record["intens"]),
                    "nist_transition_probability": _number(record["Aki(s^-1)"]),
                    "nist_line_reference": _text(record["line_ref"]),
                    "log_t_max": np.nan,
                    "passband": band,
                    "category": "",
                    "main": False,
                    "reference": "",
                    **dict.fromkeys(INTENSITY_COLUMNS, np.nan),
                }
            )
    return rows, queries


def predict_lines():
    """
    Predict optically thin intensities of IRIS-band transitions for each DEM and
    abundance set.
    """
    import fiasco  # noqa: PLC0415
    from fiasco.util.exceptions import MissingDatasetException  # noqa: PLC0415

    dem = {region: fiasco.io.Parser(f"{region}.dem").parse() for region in PRESSURES}
    formation_grid = 10 ** np.arange(3.0, 8.55, 0.01) * u.K
    predictions, formation, unranked = [], {}, {}
    for name in fiasco.list_ions(sort=True):
        ion = fiasco.Ion(name, formation_grid)
        try:
            formation[ion.ion_name_roman] = float(np.log10(ion.formation_temperature.to_value(u.K)))
            transitions = ion.transitions
            levels = ion.levels
        except MissingDatasetException as exc:
            unranked[name] = str(exc)
            continue
        bound = transitions.is_bound_bound
        wavelengths = transitions.wavelength[bound].to_value(u.angstrom)
        indices = np.flatnonzero([_passband(wavelength) is not None for wavelength in wavelengths])
        if not len(indices):
            continue
        # Neutral and singly ionized iron-peak metals (Sc-Zn) are reference data:
        # their optically thin predictions are dominated by the coolest DEM bin.
        if ion.ionization_stage <= 2 and 21 <= ion.atomic_number <= 30:
            unranked[name] = "Neutral and singly ionized iron-peak metals are intentionally unranked."
            continue
        print(f"Computing {name}: {len(indices)} transitions, {ion.n_levels} levels", flush=True)  # noqa: T201
        intensity = {column: np.zeros(len(indices)) for column in INTENSITY_COLUMNS}
        try:
            for region, model in dem.items():
                temperature = model["temperature_bin_center"]
                fraction = fiasco.Ion(name, temperature).ionization_fraction
                if not np.all(np.isfinite(fraction)):
                    msg = f"Non-finite ionization fractions for {name}, {region}."
                    raise ValueError(msg)
                # Bins without the ion contribute nothing; skip their population solves.
                keep = fraction > 0
                if not keep.any():
                    continue
                density = PRESSURES[region] * u.K / u.cm**3 / temperature[keep]
                for abundance, dataset in ABUNDANCES.items():
                    # The abundance set also sets fiasco's proton/electron ratio for proton collisions.
                    region_ion = fiasco.Ion(name, temperature[keep], abundance=dataset)
                    contribution = region_ion.contribution_function(density, couple_density_to_temperature=True)
                    # I = sum(G * DEM * dT) / 4 pi; CHIANTI DEMs and fiasco's G both use n_e*n_H.
                    integrated = np.sum(contribution[:, 0, indices] * model["em"][keep, None], axis=0) / (
                        4 * np.pi * u.sr
                    )
                    values = integrated.to_value(u.erg / u.cm**2 / u.s / u.sr)
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        msg = f"Invalid predicted intensities for {name}, {region}, {abundance}."
                        raise ValueError(msg)
                    intensity[f"intensity_{region}_{abundance}"] = values
        except MissingDatasetException as exc:
            unranked[name] = str(exc)
            continue
        # CHIANTI level numbers are not always contiguous (e.g. S II), so look them up.
        level_index = {int(level): index for index, level in enumerate(levels.level)}
        energy = levels.energy.to_value(u.cm**-1, equivalencies=u.spectral())
        j = levels.total_angular_momentum.value
        for position, index in enumerate(indices):
            lower = level_index[int(transitions.lower_level[bound][index])]
            upper = level_index[int(transitions.upper_level[bound][index])]
            predictions.append(
                {
                    "ion": ion.ion_name_roman,
                    "element": ion.atomic_symbol,
                    "ion_stage": ion.ionization_stage,
                    "wavelength": float(wavelengths[index]),
                    "passband": _passband(wavelengths[index]),
                    "lower": str(levels.label[lower]),
                    "upper": str(levels.label[upper]),
                    "lower_energy": float(energy[lower]),
                    "upper_energy": float(energy[upper]),
                    "lower_j": float(j[lower]),
                    "upper_j": float(j[upper]),
                    **{column: float(values[position]) for column, values in intensity.items()},
                }
            )
    return predictions, formation, unranked


def _identity(row):
    # NIST lists some lines without level data (e.g. Ca II 1341.89); their wavelength identifies them.
    return tuple(row[key] for key in LEVELS) if np.isfinite(row["lower_energy"]) else row["wavelength"]


def _same_transition(nist, model):
    if nist["ion"] != model["ion"] or nist["passband"] != model["passband"]:
        return False
    if np.isnan(nist["lower_energy"]) and np.isnan(nist["upper_energy"]):
        # No levels to compare: within twice NIST's 0.01 A precision for such lines, or 3 sigma.
        return abs(nist["wavelength"] - model["wavelength"]) <= np.fmax(0.02, 3 * nist["wavelength_uncertainty"])
    # Wavenumbers within the tolerance (or 3 sigma), and identical J and energies for both levels.
    window = np.fmax(ENERGY_TOLERANCE, 3e8 * nist["wavelength_uncertainty"] / nist["wavelength"] ** 2)
    return abs(1e8 / nist["wavelength"] - 1e8 / model["wavelength"]) <= window and all(
        nist[f"{level}_j"] == model[f"{level}_j"]
        and abs(nist[f"{level}_energy"] - model[f"{level}_energy"]) <= ENERGY_TOLERANCE
        for level in ("lower", "upper")
    )


def merge_lines(rows, predictions, formation):
    """
    Attach predictions to NIST lines of the same transition and keep strong unmatched
    predictions.

    Then normalize each passband and model, attach the documented lines' references, and
    assign categories.
    """
    nist_rows = list(rows)
    matches, model_pairs = [], {}
    for predicted in predictions:
        candidates = [row for row in nist_rows if _same_transition(row, predicted)]
        # Distinct compatible NIST lines are ambiguous. Among radiative channels between the
        # same levels (e.g. Mg II E1 and M2), use the largest A-value.
        matched = None
        if len({_identity(row) for row in candidates}) == 1:
            matched = max(
                candidates,
                key=lambda row: (
                    np.nan_to_num(row["nist_transition_probability"]),
                    row["wavelength_source"] == "observed",
                    -abs(row["wavelength"] - predicted["wavelength"]),
                ),
            )
            model_pairs.setdefault(id(matched), set()).add(_identity(predicted))
        matches.append(matched)
    for predicted, matched in zip(predictions, matches, strict=True):
        # A NIST line compatible with several distinct model transitions is also ambiguous.
        if matched is not None and len(model_pairs[id(matched)]) == 1:
            for column in INTENSITY_COLUMNS:
                matched[column] = np.nan_to_num(matched[column]) + predicted[column]
        else:
            rows.append(predicted | CHIANTI_ONLY)
    for band in PASSBANDS:
        band_rows = [row for row in rows if row["passband"] == band]
        for column in INTENSITY_COLUMNS:
            maximum = max((row[column] for row in band_rows if np.isfinite(row[column])), default=0)
            if maximum > 0:
                for row in band_rows:
                    row[column] /= maximum
    for ion, wavelength, reference in DOCUMENTED:
        candidates = [row for row in rows if row["ion"] == ion and abs(row["wavelength"] - wavelength) <= 0.01]
        if not candidates:
            print(f"{ion} {wavelength} ({reference}) is in neither NIST nor CHIANTI; added from the reference.")  # noqa: T201
            element, stage = ion.split()
            candidates = [
                LITERATURE_ONLY
                | {
                    "ion": ion,
                    "element": element,
                    "ion_stage": roman.from_roman(stage),
                    "wavelength": wavelength,
                    "passband": _passband(wavelength),
                }
            ]
            rows.append(candidates[0])
        # The closest line; among channels at one wavelength (Mg II E1 and M2), the largest A-value.
        line = min(
            candidates,
            key=lambda row: (
                round(abs(row["wavelength"] - wavelength), 3),  # noqa: B023
                -np.nan_to_num(row["nist_transition_probability"]),
            ),
        )
        line["reference"], line["main"] = reference, True
    rows = [
        row
        for row in rows
        if row["wavelength_source"] != "chianti"
        or row["main"]
        or any(row[column] >= THRESHOLD for column in INTENSITY_COLUMNS)
    ]
    # Categories describe ions, so every row of an ion shares one category.
    ranked_ions = {row["ion"] for row in rows if any(np.isfinite(row[column]) for column in INTENSITY_COLUMNS)}
    for row in rows:
        # Round away floating-point drift from the 0.01-dex grid before applying boundaries.
        row["log_t_max"] = round(formation.get(row["ion"], np.nan), 2)
        row["category"] = next((name for name, boundary in CATEGORY_BOUNDARIES if row["log_t_max"] >= boundary), "")
        if row["category"]:
            continue
        if row["ion"] in ranked_ions:
            row["category"] = "chromospheric"
        elif row["ion_stage"] <= 2 and row["element"].lstrip("0123456789") in PHOTOSPHERIC_METALS:
            row["category"] = "photospheric"
    return rows


def _source_file(subdirectory, filename):
    import fiasco  # noqa: PLC0415

    path = Path(fiasco.defaults["ascii_dbase_root"]) / subdirectory / filename
    return {
        "filename": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference": fiasco.io.Parser(filename).parse().meta["footer"],
    }


def main():
    import fiasco  # noqa: PLC0415
    import h5py  # noqa: PLC0415

    with h5py.File(fiasco.defaults["hdf5_dbase_root"], "r") as database:
        hdf5_version = database["fe/fe_12/wgfa"].attrs["chianti_version"]
        builder_version = database.attrs.get("fiasco_version", "unknown")
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "astropy": astropy.__version__,
        "fiasco": fiasco.__version__,
        "chianti": (Path(fiasco.defaults["ascii_dbase_root"]) / "VERSION").read_text().strip(),
        "chianti_hdf5": str(hdf5_version),
        "chianti_hdf5_built_with_fiasco": str(builder_version),
    }
    for name, version in versions.items():
        print(f"{name}: {version}", flush=True)  # noqa: T201

    rows, queries = download_nist()
    predictions, formation, unranked = predict_lines()
    rows = merge_lines(rows, predictions, formation)
    # Python's sort is stable, so ties keep the deterministic input order on every platform.
    rows.sort(key=lambda row: (row["wavelength"], row["ion"], row["lower"], row["upper"]))
    table = QTable(rows=rows)
    for name in table.colnames:
        if "wavelength" in name and name != "wavelength_source":
            table[name] = table[name] * u.angstrom
    table["nist_transition_probability"] = table["nist_transition_probability"] / u.s
    for name in ("lower_energy", "upper_energy"):
        table[name] = table[name] / u.cm
    table.meta = {
        "generation_date": datetime.now(UTC).isoformat(),
        "versions": versions,
        "nist_queries": queries,
        "abundance_files": {label: _source_file("abundance", f"{name}.abund") for label, name in ABUNDANCES.items()},
        "dem_files": {region: _source_file("dem", f"{region}.dem") for region in PRESSURES},
        "electron_pressure_K_cm-3": PRESSURES,
        "passband_limits_angstrom": PASSBANDS,
        "passband_reference": "De Pontieu et al. (2014) Table 2",
        "category_boundaries_log_t": dict(CATEGORY_BOUNDARIES),
        "chianti_only_relative_threshold": THRESHOLD,
        "unranked_ions": unranked,
    }
    output = DATA_DIR / "iris_lines.ecsv"
    table.write(output, format="ascii.ecsv", overwrite=True)
    # Astropy emits trailing spaces on empty YAML comment lines.
    lines = output.read_text().splitlines()
    output.write_text("\n".join(line.rstrip() if line.startswith("#") else line for line in lines) + "\n")
    print(f"Wrote {len(table)} lines to {output}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
