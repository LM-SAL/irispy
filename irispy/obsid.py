import importlib.resources
from pathlib import Path

import numpy as np
import pandas as pd

from astropy import units as u

dir_path = Path(__file__).parent

__all__ = ["ObsID"]


class ObsID(dict):
    """
    A class to convert the IRIS OBS ID to human-readable format.

    Parameters
    ----------
    obsid : `int`
        IRIS OBS ID to query. Needs to be a valid OBS ID, with 10 digits.

    Notes
    -----
    Currently, only OBS IDs with version numbers 36, 38, and 40 are
    supported. Calibration OBS IDs (version 42) are not yet supported.

    Examples
    --------
    Quickly show OBS ID parameters

    >>> from irispy import obsid
    >>> obsid.ObsID(3675508564)
    IRIS OBS ID 3675508564
    ----------------------
    Description:            Large dense 96-step raster 31.35x120 96s
    SJI filters:                                    C II   Mg II w s
    SJI field of view:                                       120x120
    Exposure time:                                             8.0 s
    Binning:                               Spatial x 1, Spectral x 1
    FUV binning:                         FUV spectrally rebinned x 4
    SJI cadence:                                 SJI cadence default
    Compression:                                Lossless compression
    Linelist:                                       Flare linelist 1

    The data can be accessed as in a dictionary:

    >>> data = obsid.ObsID(3675508564)
    >>> data["exptime"]
    <Quantity 8. s>
    >>> data["linelist"]
    'Flare linelist 1'
    >>> data["raster_fov"]
    '31.35x120'
    >>> data["raster_step"]
    np.float64(0.33)
    """

    def __init__(self, obsid) -> None:
        self.obsid = obsid
        data, options = self._read_obsid(obsid)
        super().__init__(data)
        self.options = options

    def __repr__(self) -> str:
        return (
            "IRIS OBS ID {obsid}\n"
            "----------------------\n"
            "Description:       {raster_fulldesc:>45}\n"
            "SJI filters:       {sjis:>45}\n"
            "SJI field of view: {sji_fov:>45}\n"
            "Exposure time:     {exptime:>45}\n"
            "Binning:           {binning:>45}\n"
            "FUV binning:       {fuv_binning:>45}\n"
            "SJI cadence:       {sji_cadence:>45}\n"
            "Compression:       {compression:>45}\n"
            "Linelist:          {linelist:>45}"
        ).format(**self)

    @staticmethod
    def _exptime_to_quant(exptime):
        """
        Converts an 'exptime' string (used in IRIS tables and OBS_DESC) to a Quantity
        instance in seconds.
        """
        if exptime == "Exposure 1s":
            return 1.0 * u.s
        return float(exptime.split(" x ")[1]) * u.s

    def _read_obsid(self, obsid):
        """
        Reads different fields from OBS ID number.
        """
        options = {}
        data = {"obsid": obsid}
        field_keys = {
            "Large linelist": "linelist",
            "Default compression": "compression",
            "Lossy compression": "compression",
            "Non-simultaneous readout": "readout",
            "SJI cadence default": "sji_cadence",
            "SJI cadence 10s": "sji_cadence",
            "FUV binned same as NUV": "fuv_binning",
            "Spatial x 1, Spectral x 1": "binning",
            "Exposure 1s": "exptime",
            "C II   Si IV   Mg II h/k   Mg II w   ": "sjis",
        }
        versions = [34, 36, 38, 40]
        if len(str(obsid)) != 10:
            msg = "Invalid OBS ID: must have 10 digits."
            raise ValueError(msg)
        # here choose between tables
        version = int(str(obsid)[:2])
        if version not in versions:
            msg = f"Invalid OBS ID: two first digits must one of {versions}"
            raise ValueError(msg)
        obsid = int(str(obsid)[2:])  # version digits are no longer needed
        with importlib.resources.as_file(importlib.resources.files("irispy") / f"data/v{version}-table10.csv") as path:
            table1 = pd.read_csv(path)
        with importlib.resources.as_file(
            importlib.resources.files("irispy") / f"data/v{version}-table2000.csv"
        ) as path:
            table2 = pd.read_csv(path)
        id_raster = int(str(obsid)[-2:])
        try:
            meta = table1.where(table1["OBS-ID"] == id_raster).dropna().iloc[0]
        except IndexError:
            msg = f"Invalid OBS ID: last two numbers must be between {table1['OBS-ID'].min()} and {table1['OBS-ID'].max()}"
            raise ValueError(
                msg,
            ) from None

        data["raster_step"] = meta["Raster step"]
        data["raster_fov"] = meta["Raster FOV"]
        data["spec_cadence"] = meta["Spectral cadence"]
        data["sji_fov"] = meta["SJI FOV"]
        data["raster_desc"] = meta["Description"]
        data["raster_fulldesc"] = f"{data['raster_desc']} {data['raster_fov']} {data['spec_cadence']}"
        field_ranges = np.concatenate(
            [  # find all dividers between fields
                table2.where(table2["OBS ID"] == 0).dropna(how="all").index,
                np.array([len(table2)]),
            ],
        )
        # field indices, start from largest and subtract
        for start, end in zip(field_ranges[-2::-1], field_ranges[:0:-1], strict=True):
            table = table2.iloc[start:end]
            for i in np.arange(start, end)[::-1]:
                index = i
                tmp = table["OBS ID"].loc[i]
                if (obsid - tmp) >= 0:
                    obsid -= tmp
                    break
            desc = table["Size + description"]
            # Save values for attributes but also table options as function of OBS ID
            if desc.iloc[0] in field_keys:
                attr_name = field_keys[desc.iloc[0]]
                if attr_name == "exptime":
                    opt = (self._exptime_to_quant(a) for a in list(desc.values))
                    opt = dict(zip(opt, table["OBS ID"], strict=True))
                    attr_value = self._exptime_to_quant(desc.loc[index])
                else:
                    opt = dict(zip(desc, table["OBS ID"], strict=True))
                    attr_value = desc.loc[index].strip()
                data[attr_name] = attr_value
                options[attr_name] = opt
        return data, options
