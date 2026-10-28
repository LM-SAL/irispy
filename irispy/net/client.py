from typing import NamedTuple
from pathlib import Path

import requests

import astropy.units as u
from astropy.table import MaskedColumn
from astropy.time import Time
from astropy.utils.masked import Masked

from sunpy.net import attrs as a
from sunpy.net.attr import AttrAnd, and_
from sunpy.net.base_client import BaseClient, QueryResponseTable
from sunpy.util.exceptions import warn_user

from irispy.net import attrs as iris_attrs

__all__ = ["IRISClient"]

# The Heliophysics Coverage Registry, which the IRIS search page also queries.
HCR_URL = "https://www.lmsal.com/hek/hcr"
# The HCR returns 400 observations unless asked for more, and at most 5000 (SolarSoft's iris_time2hcr.pro).
HCR_LIMIT = 5000


class IRISQueryResponseTable(QueryResponseTable):
    display_keys = (
        "Start Time",
        "End Time",
        "OBSID",
        "Target",
        "NOAA AR",
        "XCen",
        "YCen",
        "FOV X",
        "FOV Y",
        "SDO",
        "SOT",
        "Product",
        "Wavelength",
        "Filename",
        "Size",
    )
    size_column = "Size"


class _Missing(NamedTuple):
    # A value the HCR does not give, masked in the table.
    empty: object


def _get(mapping, key, empty):
    value = mapping.get(key)
    return _Missing(empty) if value is None else value


def _quantity(mapping, key, unit):
    value = mapping.get(key)
    # Most of the HCR's numbers are single precision: drop the digits that the conversion to
    # double precision added.
    return _Missing(0 * unit) if value is None else float(f"{value:.7g}") * unit


def _tarball_size(event, key):
    # The HCR gives sizes in whole MiB, and -1 or nothing when there is no tarball.
    return _quantity(event, f"{key}_compsize_mb", u.Mibyte) if f"{key}_compurl" in event else _Missing(0 * u.Mibyte)


def _row(event, product, group, url, size):
    """
    The table row of one file.

    ``group`` is the HCR's entry for the file's product, empty for the AIA and SOT
    tarballs.
    """
    raster = product == "raster"
    fov, fovx, fovy = (event, "raster_fovx", "raster_fovy") if raster else (group, "fovx", "fovy")
    lines = event.get("spectral_window_names") if raster else None
    return {
        "Start Time": Time(group.get("start_time") or event["startTime"]),
        "End Time": Time(group.get("stop_time") or event["stopTime"]),
        "OBSID": int(event["obsId"]),
        "Target": event["target"],
        "NOAA AR": int(event["noaaNum"]) if event.get("noaaNum") else _Missing(0),
        "XCen": _quantity(event, "xCen", u.arcsec),
        "YCen": _quantity(event, "yCen", u.arcsec),
        "FOV X": _quantity(fov, fovx, u.arcsec),
        "FOV Y": _quantity(fov, fovy, u.arcsec),
        "SDO": "martinaia_compurl" in event,
        "SOT": "martinsot_compurl" in event or "martinsp_compurl" in event,
        "Product": product,
        # The HCR's spectral_window_names lists the lines in a raster's windows, not the
        # windows, so it does not give the wavelength range of the raster.
        "Wavelength": _quantity(group, "wavelnth", u.AA),
        "Filename": url.rsplit("/", 1)[-1],
        "Size": _Missing(0 * u.Mibyte) if size is None else size * u.Mibyte,
        "Observation Start": Time(event["startTime"]),
        "Title": _get(event, "obsTitle", ""),
        "OBS Description": _get(event, "goal", ""),
        "Science Objectives": _get(event, "sciObjectives", ""),
        "Planners": _get(event, "planners", ""),
        "Roll Angle": _quantity(event, "roll_angle", u.deg),
        "Duration": _quantity(event, "duration_hours", u.h),
        "Repeats": _get(event, "iris_repeats", 0),
        "Total Frames": _get(event, "iris_totframes", 0),
        "Images": _get(group, "num_images", 0),
        "Rasters": _get(group, "numrasters", 0),
        "Steps": _get(group, "raster_numsteps", 0),
        "Step Size": _quantity(group, "raster_stepsize", u.arcsec),
        "Exposure Time": _quantity(group, "mean_exptime", u.s),
        "Min Exposure Time": _quantity(group, "min_exptime", u.s),
        "Max Exposure Time": _quantity(group, "max_exptime", u.s),
        "Cadence": _quantity(group, "cadence_avg_asrun", u.s),
        "Planned Cadence": _quantity(group, "cadence_avg_planned", u.s),
        "Step Cadence": _quantity(group, "basic_cadence_avg_asrun", u.s),
        "Spatial Summing": _get(event, "spat_sum", 0),
        "FUV Spectral Summing": _get(event, "fuv_spec_sum", 0),
        "NUV Spectral Summing": _get(event, "nuvsji_spec_sum", 0),
        "Lines": lines.strip("'") if lines else _Missing(""),
        "Data Fraction": _get(event, "data_fraction_found", 0.0),
        "Bad Data": _get(event, "iris_baddata", empty=False),
        "Rotation Tracking": _get(event, "rotation_tracking", empty=False),
        "SDO Size": _tarball_size(event, "martinaia"),
        "SOTFG Size": _tarball_size(event, "martinsot"),
        "SOTSP Size": _tarball_size(event, "martinsp"),
        "Movie URL": _get(group, "url_mp4", ""),
        "Thumbnail URL": _get(group, "url_thumb", ""),
        "Context URL": _get(group, "url_context", ""),
        "Timeline URL": _get(event, "timeline_gif_url", ""),
        "URL": url,
    }


class IRISClient(BaseClient):
    """
    Searches the IRIS archive at LMSAL for Level 2 data and downloads it.

    Search with `a.Time <sunpy.net.attrs.Time>` and ``a.Instrument("IRIS")`` or
    ``a.Provider("LMSAL")``, and narrow the search with the `irispy.net.attrs` (``a.iris``).
    The results are in ``result["iris"]``, one row per file: the raster tarball and the
    slit-jaw files of each observation, or the files of the products asked for with
    ``a.iris.Product``. ``result["iris"].keys()`` lists its columns, of which
    ``result["iris"].show(...)`` shows the hidden ones.

    `Fido.fetch <sunpy.net.fido_factory.UnifiedDownloaderFactory.fetch>` puts the files of each
    observation in a folder named after it, such as ``20211001_060925_3683602040``, which
    `irispy.io.read_files` reads in one go. :ref:`irispy-tutorial-acquiring-data` shows a
    search and a download.

    Examples
    --------
    >>> from sunpy.net import Fido, attrs as a
    >>> import irispy.net
    >>> result = Fido.search(
    ...     a.Time("2021-10-01 06:00", "2021-10-01 07:00"),
    ...     a.Provider("LMSAL"),
    ...     a.iris.Target("AR"),
    ... )  # doctest: +REMOTE_DATA
    >>> print(
    ...     result["iris"]["OBSID", "Product", "Filename", "Size"]
    ... )  # doctest: +REMOTE_DATA
      OBSID    Product                         Filename                          Size
                                                                                Mibyte
    ---------- ------- -------------------------------------------------------- ------
    3683602040     sji iris_l2_20211001_060925_3683602040_SJI_1330_t000.fits.gz    3.0
    3683602040     sji iris_l2_20211001_060925_3683602040_SJI_1400_t000.fits.gz    3.0
    3683602040     sji iris_l2_20211001_060925_3683602040_SJI_2796_t000.fits.gz    5.0
    3683602040     sji iris_l2_20211001_060925_3683602040_SJI_2832_t000.fits.gz    6.0
    3683602040  raster         iris_l2_20211001_060925_3683602040_raster.tar.gz   83.0
    >>> files = Fido.fetch(result["iris"])  # doctest: +SKIP
    """

    @property
    def info_url(self):
        return "https://iris.lmsal.com/search/"

    def search(self, *query):
        """
        Search the IRIS archive.

        Parameters
        ----------
        *query : `sunpy.net.attr.Attr`
            The attrs to search with, combined with AND.

        Returns
        -------
        `sunpy.net.base_client.QueryResponseTable`
            One row per file.
        """
        query = and_(*query)
        params = {
            "cmd": "search-events3",
            "outputformat": "json",
            "instrument": "IRIS",
            "hasData": "true",
            "limit": HCR_LIMIT,
        }
        products = {"raster", "sji"}
        text = None
        for attr in query.attrs if isinstance(query, AttrAnd) else [query]:
            match attr:
                case a.Time():
                    params["startTime"], params["stopTime"] = attr.start.isot, attr.end.isot
                case iris_attrs.ObsID():
                    params["obsid"] = attr.value
                case iris_attrs.Target():
                    params["target"] = attr.value
                case iris_attrs.XCen():
                    params["minxcen"], params["maxxcen"] = attr.min.to_value(u.arcsec), attr.max.to_value(u.arcsec)
                case iris_attrs.YCen():
                    params["minycen"], params["maxycen"] = attr.min.to_value(u.arcsec), attr.max.to_value(u.arcsec)
                case iris_attrs.Description():
                    # The HCR's own text search (obsdesc) looks only at the title.
                    text = attr.value.lower()
                case iris_attrs.Product():
                    products = {attr.value}
        # A search of three months takes about 200 seconds.
        response = requests.get(HCR_URL, params=params, timeout=600)
        response.raise_for_status()
        events = response.json()["Events"]
        if len(events) == HCR_LIMIT:
            warn_user(f"The IRIS archive returns at most {HCR_LIMIT} observations; search a shorter time range.")
        rows = []
        for event in events:
            # Full-disk mosaics have no observation name.
            if "iris_obsshort" not in event:
                continue
            description = " ".join(event.get(key) or "" for key in ("obsTitle", "goal", "sciObjectives"))
            if text and text not in description.lower():
                continue
            files = [
                (
                    "raster"
                    if group["group_name"] == "Raster"
                    else "deconvolved"
                    if group["group_name"].endswith("_deconvolved")
                    else "sji",
                    group,
                    group.get("comp_data_url"),
                    group.get("comp_data_size_mb"),
                )
                for group in event["groups"]
            ]
            files += [
                ("sdo", {}, event.get("martinaia_compurl"), event.get("martinaia_compsize_mb")),
                ("sot", {}, event.get("martinsot_compurl"), event.get("martinsot_compsize_mb")),
                ("sot", {}, event.get("martinsp_compurl"), event.get("martinsp_compsize_mb")),
            ]
            rows += [
                _row(event, product, group, url, size)
                for product, group, url, size in files
                if product in products and url
            ]
        # The HCR returns the observations in no fixed order; a file's name starts with the
        # start of its observation.
        rows.sort(key=lambda row: row["Filename"])
        table = IRISQueryResponseTable(
            rows=[
                {name: value.empty if isinstance(value, _Missing) else value for name, value in row.items()}
                for row in rows
            ],
            client=self,
        )
        for name in table.colnames:
            mask = [isinstance(row[name], _Missing) for row in rows]
            if any(mask):
                column = table[name]
                table[name] = (
                    Masked(column, mask=mask) if isinstance(column, u.Quantity) else MaskedColumn(column, mask=mask)
                )
        return table

    def fetch(self, query_results, *, path, downloader, **kwargs):  # NOQA: ARG002
        """
        Queue the files of ``query_results`` on ``downloader``, each in the folder of
        its observation.

        `Fido.fetch <sunpy.net.fido_factory.UnifiedDownloaderFactory.fetch>` calls this.
        """
        for row in query_results:
            # The archive keeps each observation in a folder named after it.
            observation = row["URL"].rsplit("/", 2)[-2]
            filename = str(path).format(file=Path(observation, row["Filename"]), **row.response_block_map)
            downloader.enqueue_file(row["URL"], filename=filename)

    @classmethod
    def _can_handle_query(cls, *query):
        optional = {
            a.Instrument,
            a.Provider,
            iris_attrs.ObsID,
            iris_attrs.Description,
            iris_attrs.Target,
            iris_attrs.XCen,
            iris_attrs.YCen,
            iris_attrs.Product,
        }
        # As sunpy's GOES XRSClient does, the client's own attrs only refine a search for the
        # instrument or the provider.
        return (
            cls.check_attr_types_in_query(query, {a.Time}, optional)
            and any(isinstance(q, a.Instrument | a.Provider) for q in query)
            and all(str(q.value).upper() == "IRIS" for q in query if isinstance(q, a.Instrument))
            and all(str(q.value).upper() == "LMSAL" for q in query if isinstance(q, a.Provider))
        )

    @classmethod
    def _attrs_module(cls):
        return "iris", "irispy.net.attrs"

    @classmethod
    def register_values(cls):
        return {
            a.Instrument: [("IRIS", "Interface Region Imaging Spectrograph")],
            a.Provider: [("LMSAL", "Lockheed Martin Solar and Astrophysics Laboratory")],
            iris_attrs.Product: [
                ("raster", "The raster tarball of the observation."),
                ("sji", "The slit-jaw files of the observation."),
                ("deconvolved", "The deconvolved slit-jaw files of the observation."),
                ("sdo", "The tarball of AIA cubes co-aligned with the observation."),
                ("sot", "The Hinode/SOT tarballs of the observation, SOTFG images and SOTSP maps."),
            ],
        }
