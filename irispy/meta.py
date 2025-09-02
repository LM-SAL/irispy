import textwrap

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.meta import NDMeta
from sunpy.coordinates import Helioprojective
from sunraster.meta import RemoteSensorMetaABC, SlitSpectrographMetaABC

from irispy.utils.constants import SPECTRAL_BAND

__all__ = ["SGMeta", "SJIMeta"]


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
        Number of times automatic exposure control triggered during observing
        campaign.
        """
        return self.get("AECNOBS")

    @property
    def exposure_control_triggers_in_raster(self):
        """
        Number of times automatic exposure control was triggered during this
        raster.
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
        return SPECTRAL_BAND[self.spectral_window]

    @property
    def raster_fov_width_y(self):
        """
        Width of the field of view of the raster in the Y (slit) direction.
        """
        return self.get("FOVY") * u.arcsec

    @property
    def raster_fov_width_x(self):
        """
        Width of the field of view of the raster in the X (rastering)
        direction.
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
