import textwrap
from copy import deepcopy
from functools import cached_property

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.meta import NDMeta
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.screens import SphericalScreen
from sunraster.meta import RemoteSensorMetaABC, SlitSpectrographMetaABC

from irispy._interpolation import _pad_axis0
from irispy.utils.constants import SPECTRAL_BAND

__all__ = ["BaseMeta", "SGMeta", "SJIMeta"]


def _stack_scan_aligned_values(values, target_length):
    values = [_pad_axis0(value, target_length) for value in values]
    first = values[0]
    if isinstance(first, SkyCoord):
        return SkyCoord(
            Tx=np.stack([value.Tx.to_value(u.arcsec) for value in values]) * u.arcsec,
            Ty=np.stack([value.Ty.to_value(u.arcsec) for value in values]) * u.arcsec,
            frame=first.frame,
        )
    return np.stack(values)


class BaseMeta(NDMeta):
    def __init__(self, header, **kwargs) -> None:
        super().__init__(header, **kwargs)

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{self!s}"

    def _construct_time(self, key):
        val = self.get(key)
        if val is not None:
            val = Time(val, format="fits", scale="utc")
        return val

    @property
    def fits_header(self):
        return self._fits_header

    @property
    def spectral_window(self):
        return self.get(f"TDESC{self._iwin}")

    @property
    def detector(self):
        return self.get(f"TDET{self._iwin}")

    @property
    def instrument(self):
        return self.get("INSTRUME")

    @property
    def observatory(self):
        return self.get("TELESCOP")

    @property
    def processing_level(self):
        return int(self.get("DATA_LEV"))

    @property
    def distance_to_sun(self):
        return (self.get("DSUN_OBS") * u.m).to(u.AU)

    @property
    def date_reference(self):
        return self._construct_time("DATE_OBS")

    @property
    def date_start(self):
        return self.date_reference

    @property
    def date_end(self):
        return self._construct_time("DATE_END")

    @property
    def observing_mode_id(self):
        return int(self.get("OBSID"))

    # ---------- IRIS-specific metadata properties ----------
    @property
    def observing_mode_description(self):
        return self.get("OBS_DESC")

    @property
    def observing_campaign_start(self):
        """
        Start time of observing campaign.
        """
        return self._construct_time("STARTOBS")

    @property
    def observing_campaign_end(self):
        """
        End time of observing mode.
        """
        return self._construct_time("ENDOBS")

    @property
    def observation_includes_saa(self):
        """
        Whether IRIS passed through SAA during observations.
        """
        return bool(self.get("SAA"))

    @property
    def satellite_rotation(self):
        """
        Satellite roll from solar north.
        """
        return self.get("SAT_ROT") * u.deg

    @property
    def exposure_control_triggers_in_observation(self):
        """
        Number of times automatic exposure control triggered during observing campaign.
        """
        return self.get("AECNOBS")

    @property
    def exposure_control_triggers_in_raster(self):
        """
        Number of times automatic exposure control was triggered during this raster.
        """
        return self.get("AECNRAS")

    @property
    def number_of_unique_raster_positions(self):
        """
        Number of unique positions in raster.
        """
        return self.get("NRASTERP")

    @property
    def number_of_raster_positions(self):
        """
        Number of positions in raster.
        """
        return self.get("RASNRPT")

    @property
    def spectral_range(self):
        """
        The spectral range of the spectral window.
        """
        return [self.get(f"TWMIN{self._iwin}"), self.get(f"TWMAX{self._iwin}")] * u.AA

    @property
    def spectral_band(self):
        """
        The spectral band of the spectral window.
        """
        return SPECTRAL_BAND.get(self.spectral_window, self.spectral_window)

    @property
    def raster_fov_width_y(self):
        """
        Width of the field of view of the raster in the Y (slit) direction.
        """
        return self.get("FOVY") * u.arcsec

    @property
    def raster_fov_width_x(self):
        """
        Width of the field of view of the raster in the X (rastering) direction.
        """
        return self.get("FOVX") * u.arcsec

    @property
    def fov_center(self):
        """
        Location of the center of the field of view.
        """
        return SkyCoord(
            Tx=self.get("XCEN"),
            Ty=self.get("YCEN"),
            unit=u.arcsec,
            frame=Helioprojective,
        )

    @property
    def automatic_exposure_control_enabled(self):
        return bool(self.get("IAECFLAG"))

    @property
    def tracking_mode_enabled(self):
        return bool(self.get("TR_MODE"))

    @property
    def observatory_at_high_latitude(self):
        """
        Whether IRIS passed through high Earth latitude during observations.
        """
        return bool(self.get("HLZ"))

    @property
    def spatial_summing_factor(self):
        """
        Number of pixels summed together in the spatial (Y/slit) direction.
        """
        return self.get("SUMSPAT")

    @property
    def spectral_summing_factor(self):
        """
        Number of pixels summed together in the spectral direction.
        """
        if "fuv" in self.detector.lower():
            return self.get("SUMSPTRF")
        return self.get("SUMSPTRN")


class SJIMeta(BaseMeta, RemoteSensorMetaABC):
    """
    Metadata class for IRIS slit-jaw images.
    """

    def __init__(self, header, **kwargs) -> None:
        super().__init__(header, **kwargs)
        self._iwin = 1
        self._fits_header = header

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
                SJIMeta
                -------
                Observatory:     {self.observatory}
                Instrument:      {self.instrument}
                Detector:        {self.detector}
                Spectral Window: {self.spectral_window}
                Spectral Range:  {self.spectral_range}
                Spectral Band:   {self.spectral_band}
                Dimensions:      {self.data_shape}
                Date:            {self.date_reference}
                OBS ID:          {self.observing_mode_id}
                OBS Description: {self.observing_mode_description}
                """,
        )

    @property
    def spectral_window(self):
        return super().spectral_window.replace("SJI_", "")


class SGMeta(BaseMeta, SlitSpectrographMetaABC):
    """
    Metadata class for IRIS slit spectrograph data.
    """

    def __init__(self, header, spectral_window, **kwargs) -> None:
        super().__init__(header, **kwargs)
        spectral_windows = np.array([self[f"TDESC{i}"] for i in range(1, self["NWIN"] + 1)])
        window_mask = np.array([spectral_window in window for window in spectral_windows])
        if window_mask.sum() < 1:
            msg = (
                "Spectral window not found. "
                f"Input spectral window: {spectral_window}; "
                f"Spectral windows in header: {spectral_windows}"
            )
            raise ValueError(
                msg,
            )
        if window_mask.sum() > 1:
            msg = (
                "Spectral window must be unique. "
                f"Input spectral window: {spectral_window}; "
                f"Ambiguous spectral windows in header: {spectral_windows[window_mask]}"
            )
            raise ValueError(
                msg,
            )
        self._iwin = np.arange(len(spectral_windows))[window_mask][0] + 1
        self._fits_header = header

    @cached_property
    def observer(self):
        """
        The IRIS observer location at the observation start, assumed to be at Earth.
        """
        base_time = self.date_reference or self.observing_campaign_start
        location = get_body_heliographic_stonyhurst("Earth", base_time.isot)
        observer = Helioprojective(
            self["XCEN"] * u.arcsec,
            self["YCEN"] * u.arcsec,
            observer=location,
            obstime=base_time,
        )
        with SphericalScreen(observer.observer):
            return observer.transform_to(HeliographicStonyhurst(obstime=base_time))

    @classmethod
    def combine(cls, metas, combined_shape):
        metas = tuple(metas)
        if not metas:
            msg = "Cannot combine an empty SGMeta sequence."
            raise ValueError(msg)

        target_steps = combined_shape[1]
        meta = deepcopy(metas[0])
        for key in ("memmap_path", "memmap_ext"):
            meta.pop(key, None)
        meta._data_shape = np.asarray(combined_shape, dtype=int)
        meta["NAXIS4"] = combined_shape[0]
        meta["NAXIS3"] = combined_shape[1]
        if "NAXIS3" in meta.fits_header:
            meta.fits_header["NAXIS3"] = combined_shape[1]
        meta.fits_header["NAXIS4"] = combined_shape[0]
        for key in ("DATE_END", "ENDOBS"):
            if metas[-1].get(key) is not None:
                meta[key] = metas[-1][key]
                if key in meta.fits_header:
                    meta.fits_header[key] = metas[-1][key]

        for key, axes in metas[0]._axes.items():
            if not np.array_equal(axes, [0]):
                continue
            meta.add(
                key,
                _stack_scan_aligned_values([source_meta[key] for source_meta in metas], target_steps),
                axes=(0, 1),
                overwrite=True,
            )
        return meta

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
                SGMeta
                ------
                Observatory:     {self.observatory}
                Instrument:      {self.instrument}
                Detector:        {self.detector}
                Spectral Window: {self.spectral_window}
                Spectral Range:  {self.spectral_range}
                Spectral Band:   {self.spectral_band}
                Dimensions:      {self.data_shape}
                Date:            {self.date_reference}
                OBS ID:          {self.observing_mode_id}
                OBS Description: {self.observing_mode_description}
                """,
        )
