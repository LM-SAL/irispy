"""
Regenerate irispy/data/iris_lines.ecsv from current NIST and CHIANTI data.

Requires ``pip install -e '.[density]'``, fiasco 0.8.2 or later, and a full CHIANTI database in fiasco's configured
location (https://fiasco.readthedocs.io/en/latest/quick_start.html).
Model assumptions and provenance are documented in docs/line_database.rst.

python tools/make_line_database.py
"""

import io
import re
import csv
import sys
import hashlib
from html import unescape
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
NIST_VERSION_URL = "https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml"
# Vacuum Angstrom limits from De Pontieu et al. (2014), Table 2.
# FUV2 readout below 1389 A falls outside the published passband.
PASSBANDS = {"FUV1": [1331.7, 1358.4], "FUV2": [1389.0, 1407.0], "NUV": [2782.7, 2835.1]}
PRESSURES = {"quiet_sun": 3e15, "active_region": 3e15, "flare": 1e16}  # n_e T in K cm^-3
ABUNDANCES = {"coronal": "sun_coronal_2021_chianti", "photospheric": "sun_photospheric_2021_asplund"}
# Ion, vacuum wavelength (Angstrom), and docs/references.bib key.
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
# Equilibrium ion-fraction peak boundaries, log10 K: Tian (2017); De Pontieu et al. (2021).
CATEGORY_BOUNDARIES = [("flare", 7.0), ("coronal", 5.9), ("transition_region", 4.3)]
INTENSITY_COLUMNS = [f"intensity_{region}_{abundance}" for region in PRESSURES for abundance in ABUNDANCES]
LEVELS = ("lower", "upper", "lower_energy", "upper_energy", "lower_j", "upper_j")
# Matching tolerance in cm^-1; accommodates Fe XII's 2 cm^-1 catalog difference.
ENERGY_TOLERANCE = 5.0
# Mg II: Leenaarts et al. (2013); C II: Rathore & Carlsson (2015);
# O I: Lin & Carlsson (2015); C I: Lin et al. (2017); Cl I: IRIS Technical Note 38.
CHROMOSPHERIC_IONS = {"Mg II", "C II", "O I", "C I", "Cl I"}
COOL_METALS = {"Li", "Be", "Na", "Mg", "Al", "Si", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
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
# Every catalog column with its empty value, in output order; each source overrides what it knows.
DEFAULTS = {
    "ion": "",
    "element": "",
    "ion_stage": 0,
    "wavelength": np.nan,
    "wavelength_source": "chianti",
    "wavelength_is_theoretical": False,
    "wavelength_uncertainty": np.nan,
    "observed_wavelength": np.nan,
    "observed_wavelength_uncertainty": np.nan,
    "ritz_wavelength": np.nan,
    "ritz_wavelength_uncertainty": np.nan,
    "transition_type": "",
    "lower": "",
    "upper": "",
    "lower_energy": np.nan,
    "upper_energy": np.nan,
    "lower_j": np.nan,
    "upper_j": np.nan,
    "nist_intensity": "",
    "nist_transition_probability": np.nan,
    "nist_line_reference": "",
    "log_t_max": np.nan,
    "passband": "",
    "category": "",
    "main": False,
    "reference": "",
    **dict.fromkeys(INTENSITY_COLUMNS, np.nan),
}


def _text(value):
    # Strip NIST's Excel string wrappers without evaluating them.
    value = value.strip()
    return value[2:-1] if value.startswith('="') and value.endswith('"') else value


def _number(value):
    value = _text(value).strip("[]?*+ ")
    return float(value) if value else np.nan


def _level_number(value):
    # Reject ambiguous or qualified level values: "1/2,3/2", "[123]", "123+x".
    try:
        return float(Fraction(_text(value)))
    except (ValueError, ZeroDivisionError):
        return np.nan


def _passband(wavelength):
    for band, (low, high) in PASSBANDS.items():
        if low <= wavelength <= high:
            return band
    return None


def _nist_version():
    request = Request(NIST_VERSION_URL, headers={"User-Agent": "Mozilla/5.0 (irispy line database generator)"})
    with urlopen(request, timeout=120) as response:  # noqa: S310
        content = unescape(response.read().decode("utf-8"))
    match = re.search(r"\(version\s+([0-9]+(?:\.[0-9]+)*)\)", content, flags=re.IGNORECASE)
    if match is None:
        msg = "Could not read the current NIST ASD version from its citation."
        raise ValueError(msg)
    return match.group(1)


def download_nist():
    """
    Return NIST lines and query provenance for each passband.
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
            observed_uncertainty = _number(record["unc_obs_wl"])
            ritz_uncertainty = _number(record["unc_ritz_wl"])
            # Prefer the smaller known uncertainty, falling back to observed wavelengths.
            prefer_ritz = np.isfinite(ritz) and (
                not np.isfinite(observed)
                or (
                    np.isfinite(ritz_uncertainty)
                    and (not np.isfinite(observed_uncertainty) or ritz_uncertainty < observed_uncertainty)
                )
            )
            source = "ritz" if prefer_ritz else "observed"
            wavelength = observed if source == "observed" else ritz
            if not low <= wavelength <= high:
                continue
            element = _text(record["element"])
            stage = int(record["sp_num"])
            rows.append(
                DEFAULTS
                | {
                    "ion": f"{element} {roman.to_roman(stage)}",
                    "element": element,
                    "ion_stage": stage,
                    "wavelength": wavelength,
                    "wavelength_source": source,
                    "wavelength_uncertainty": observed_uncertainty if source == "observed" else ritz_uncertainty,
                    "observed_wavelength": observed,
                    "observed_wavelength_uncertainty": observed_uncertainty,
                    "ritz_wavelength": ritz,
                    "ritz_wavelength_uncertainty": ritz_uncertainty,
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
                    "passband": band,
                }
            )
    return rows, queries


def predict_lines():
    """
    Predict reference optically thin intensities for eligible IRIS-band transitions.

    Retain excluded transitions and those missing excitation data without predictions.
    """
    import fiasco  # noqa: PLC0415
    from fiasco.util.exceptions import MissingDatasetException  # noqa: PLC0415

    dem = {region: fiasco.io.Parser(f"{region}.dem").parse() for region in PRESSURES}
    formation_grid = 10 ** np.arange(3.0, 8.55, 0.01) * u.K
    predictions, formation, unranked = [], {}, {}
    for name in fiasco.list_ions(sort=True):
        ion = fiasco.Ion(name, formation_grid, ionization_fraction="chianti")
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
        intensity = {column: np.full(len(indices), np.nan) for column in INTENSITY_COLUMNS}
        if ion.ionization_stage <= 2:
            unranked[name] = "Neutral and singly ionized lines are outside the prediction model's scope."
        else:
            print(f"Computing {name}: {len(indices)} transitions, {ion.n_levels} levels", flush=True)  # noqa: T201
            try:
                for region, model in dem.items():
                    temperature = model["temperature_bin_center"]
                    fraction = fiasco.Ion(name, temperature, ionization_fraction="chianti").ionization_fraction
                    if not np.all(np.isfinite(fraction)):
                        msg = f"Non-finite ionization fractions for {name}, {region}."
                        raise ValueError(msg)
                    keep = fraction > 0
                    for abundance, dataset in ABUNDANCES.items():
                        if not keep.any():
                            intensity[f"intensity_{region}_{abundance}"] = np.zeros(len(indices))
                            continue
                        density = PRESSURES[region] * u.K / u.cm**3 / temperature[keep]
                        # Abundances also determine fiasco's proton/electron ratio.
                        region_ion = fiasco.Ion(
                            name, temperature[keep], abundance=dataset, ionization_fraction="chianti"
                        )
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
                intensity = {column: np.full(len(indices), np.nan) for column in INTENSITY_COLUMNS}
        # CHIANTI level identifiers can be non-contiguous (e.g. S II).
        level_index = {int(level): index for index, level in enumerate(levels.level)}
        energy = levels.energy.to_value(u.cm**-1, equivalencies=u.spectral())
        j = levels.total_angular_momentum.value
        for position, index in enumerate(indices):
            lower = level_index[int(transitions.lower_level[bound][index])]
            upper = level_index[int(transitions.upper_level[bound][index])]
            predictions.append(
                DEFAULTS
                | {
                    "ion": ion.ion_name_roman,
                    "element": ion.atomic_symbol,
                    "ion_stage": ion.ionization_stage,
                    "wavelength": float(wavelengths[index]),
                    "wavelength_is_theoretical": not bool(transitions.is_observed[bound][index]),
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
    # Wavelength identifies NIST lines lacking complete level data (e.g. Ca II 1341.89).
    if np.isfinite(row["lower_energy"]) and np.isfinite(row["upper_energy"]):
        return tuple(row[key] for key in LEVELS)
    return row["wavelength"]


def _same_transition(nist, model):
    if nist["ion"] != model["ion"] or nist["passband"] != model["passband"]:
        return False
    if np.isnan(nist["lower_energy"]) and np.isnan(nist["upper_energy"]):
        # Twice NIST's 0.01 A precision for lines lacking levels, or 3 sigma.
        return abs(nist["wavelength"] - model["wavelength"]) <= np.fmax(0.02, 3 * nist["wavelength_uncertainty"])
    window = np.fmax(ENERGY_TOLERANCE, 3e8 * nist["wavelength_uncertainty"] / nist["wavelength"] ** 2)
    # NIST can leave one level energy unparsed (e.g. Si X 1344.09); compare the known one.
    return abs(1e8 / nist["wavelength"] - 1e8 / model["wavelength"]) <= window and all(
        nist[f"{level}_j"] == model[f"{level}_j"]
        and (
            np.isnan(nist[f"{level}_energy"])
            or abs(nist[f"{level}_energy"] - model[f"{level}_energy"]) <= ENERGY_TOLERANCE
        )
        for level in ("lower", "upper")
    )


def merge_lines(rows, predictions, formation):
    """
    Match NIST and CHIANTI transitions, retaining unmatched lines regardless of
    strength.

    Normalize strengths per passband and model, add published identifications, and
    assign categories.
    """
    nist_rows = list(rows)
    matches, model_pairs = [], {}
    for predicted in predictions:
        candidates = [row for row in nist_rows if _same_transition(row, predicted)]
        # Accept one level pair; prefer the largest A-value among its radiative channels.
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
        # Reject a NIST line matched to multiple CHIANTI level pairs.
        if matched is not None and len(model_pairs[id(matched)]) == 1:
            for column in INTENSITY_COLUMNS:
                matched[column] = np.nan_to_num(matched[column]) + predicted[column]
        else:
            rows.append(predicted)
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
                DEFAULTS
                | {
                    "ion": ion,
                    "element": element,
                    "ion_stage": roman.from_roman(stage),
                    "wavelength": wavelength,
                    "wavelength_source": "literature",
                    "passband": _passband(wavelength),
                }
            ]
            rows.append(candidates[0])
        # Prefer the nearest wavelength, then the largest A-value among channels.
        line = min(
            candidates,
            key=lambda row: (
                round(abs(row["wavelength"] - wavelength), 3),
                -np.nan_to_num(row["nist_transition_probability"]),
            ),
        )
        line["reference"], line["main"] = reference, True
    for row in rows:
        # Remove floating-point drift at the 0.01-dex category boundaries.
        row["log_t_max"] = round(formation.get(row["ion"], np.nan), 2)
        if row["ion"] in CHROMOSPHERIC_IONS:
            row["category"] = "chromospheric"
            continue
        row["category"] = next((name for name, boundary in CATEGORY_BOUNDARIES if row["log_t_max"] >= boundary), "")
        if row["category"]:
            continue
        if row["ion_stage"] <= 2 and row["element"].lstrip("0123456789") in COOL_METALS:
            row["category"] = "cool_metal"
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
        "nist_asd": _nist_version(),
    }
    for name, version in versions.items():
        print(f"{name}: {version}", flush=True)  # noqa: T201

    rows, queries = download_nist()
    predictions, formation, unranked = predict_lines()
    rows = merge_lines(rows, predictions, formation)
    rows.sort(key=lambda row: (row["wavelength"], row["ion"], row["lower"], row["upper"]))
    table = QTable(rows=rows)
    for name in table.colnames:
        if "wavelength" in name and name not in ("wavelength_source", "wavelength_is_theoretical"):
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
        "prediction_excluded_ion_stages": [1, 2],
        "ionization_fraction_dataset": "chianti",
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
