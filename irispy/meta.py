import textwrap

import numpy as np

import astropy.units as u
from astropy.constants import R_sun as _R_SUN
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.meta import NDMeta
from sunpy.coordinates import Helioprojective
from sunraster.meta import RemoteSensorMetaABC, SlitSpectrographMetaABC

from irispy.utils.constants import SPECTRAL_BAND

__all__ = ["BaseMeta", "SGMeta", "SJIMeta"]


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
    def camera(self):
        """
        IRIS camera ID: 1 for FUV, 2 for NUV/SJI.
        """
        return self.get("CAMERA")

    @property
    def sun_angular_radius(self):
        """
        Apparent angular radius of the Sun at the observer location.

        Read from ``RSUN_OBS`` (arcsec) if present, otherwise computed
        from ``DSUN_OBS``.
        """
        rsun = self.get("RSUN_OBS")
        if rsun is not None:
            return float(rsun) * u.arcsec
        return np.arctan(_R_SUN.to(u.m).value / float(self.get("DSUN_OBS"))) * u.rad

    @property
    def observer_radial_velocity(self):
        """
        Radial velocity of the observer relative to the Sun (m/s).

        Read from ``OBS_VR``.
        """
        return float(self.get("OBS_VR")) * u.m / u.s

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
    def temporal_cadence(self):
        """
        Average time between exposures.
        """
        return float(self.get("CADEX_AV")) * u.s

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
    def detector_band(self):
        """
        Detector band: ``'FUV'``, ``'NUV'``, or ``'SJI'``.
        """
        det_upper = self.detector.upper()
        return next(
            (band for band in ("FUV", "NUV", "SJI") if det_upper.startswith(band)),
            det_upper,
        )

    @property
    def rest_wavelength(self):
        """
        Rest wavelength of the spectral line for this window.
        """
        return (float(self.get(f"TWAVE{self._iwin}")) * u.AA).to(u.nm)

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

    @property
    def exposure_time(self):
        """
        Mean exposure duration (shutter open time).
        """
        return float(self.get("EXPTIME")) * u.s

    @property
    def exposure_time_min(self):
        """
        Minimum exposure duration in this raster/SJI.
        """
        return float(self.get("EXPMIN")) * u.s

    @property
    def exposure_time_max(self):
        """
        Maximum exposure duration in this raster/SJI.
        """
        return float(self.get("EXPMAX")) * u.s

    @property
    def data_type(self):
        """
        Type of data, e.g. ``'Intensity'``.
        """
        return self.get("BTYPE")

    @property
    def data_unit(self):
        """
        Unit of the data values.
        """
        return self.get("BUNIT")

    @property
    def data_status(self):
        """
        Processing status: ``'Quicklook'`` or ``'Final'``.
        """
        return self.get("STATUS")

    @property
    def build_version(self):
        """
        Build version from ``jsoc_version.h``.
        """
        return self.get("BLD_VERS")

    @property
    def reformat_version(self):
        """
        Version of the software that reformatted the data to Level 2.
        """
        return self.get("VER_RF2")

    @property
    def reformat_date(self):
        """
        Date of reformatting to Level 2.
        """
        return self._construct_time("DATE_RF2")

    @property
    def observing_label(self):
        """
        Observing list string.
        """
        return self.get("OBSLABEL")

    @property
    def observing_title(self):
        """
        Title given by the planner.
        """
        return self.get("OBSTITLE")

    @property
    def lut_id(self):
        """
        Look-up table ID.
        """
        return self.get("LUTID")

    @property
    def number_of_exposures(self):
        """
        Number of exposures in this raster/SJI.
        """
        return self.get("NEXP")

    @property
    def number_of_exposures_planned(self):
        """
        Number of planned exposures in this raster/SJI.
        """
        return self.get("NEXP_PRP")

    @property
    def number_of_exposures_observation(self):
        """
        Expected total number of exposures in the whole observation.
        """
        return self.get("NEXPOBS")

    @property
    def number_of_saturated_pixels(self):
        """
        Number of saturated pixels.
        """
        return self.get("NSATPIX")

    @property
    def number_of_spikes(self):
        """
        Number of pixels identified as noise (cosmic-ray) spikes.
        """
        return self.get("NSPIKES")

    @property
    def percent_data(self):
        """
        Percentage of valid data values.
        """
        return self.get("PERCENTD")

    @property
    def data_mean(self):
        """
        Mean value of all pixels.
        """
        return self.get("DATAMEAN")

    @property
    def data_rms(self):
        """
        RMS deviation from the mean value of all pixels.
        """
        return self.get("DATARMS")

    @property
    def data_median(self):
        """
        Median value of all pixels.
        """
        return self.get("DATAMEDN")

    @property
    def data_min(self):
        """
        Minimum value of all pixels.
        """
        return self.get("DATAMIN")

    @property
    def data_max(self):
        """
        Maximum value of all pixels.
        """
        return self.get("DATAMAX")


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

    @property
    def number_of_spectral_windows(self):
        """
        Number of spectral windows in this observation.
        """
        return self.get("NWIN")

    @property
    def raster_repetition(self):
        """
        Current raster repetition counter.
        """
        return self.get("RASRPT")

    @property
    def step_size_average(self):
        """
        Average of the basic raster step size.
        """
        return self.get("STEPS_AV")

    @property
    def step_size_stddev(self):
        """
        Standard deviation of the basic raster step size.
        """
        return self.get("STEPS_DV")

    @property
    def step_time_average(self):
        """
        Average of the basic raster step time.
        """
        return self.get("STEPT_AV")

    @property
    def step_time_stddev(self):
        """
        Standard deviation of the basic raster step time.
        """
        return self.get("STEPT_DV")

    @property
    def cadence_planned_average(self):
        """
        Mean cadence of the raster as planned.
        """
        return float(self.get("CADPL_AV")) * u.s

    @property
    def cadence_planned_stddev(self):
        """
        Standard deviation of the planned raster cadence.
        """
        return float(self.get("CADPL_DV")) * u.s

    @property
    def cadence_executed_stddev(self):
        """
        Standard deviation of the executed raster cadence.
        """
        return float(self.get("CADEX_DV")) * u.s

    @property
    def raster_type_index(self):
        """
        Raster type number.
        """
        return self.get("RASTYPDX")

    @property
    def raster_type_total(self):
        """
        Total number of raster types.
        """
        return self.get("RASTYPNX")

    @property
    def number_of_missing_raster_files(self):
        """
        Number of missing Level 1 files in this raster.
        """
        return self.get("MISSRAS")

    @property
    def number_of_missing_observation_files(self):
        """
        Number of missing Level 1 files in the whole observation.
        """
        return self.get("MISSOBS")

    @property
    def window_mean(self):
        """
        Mean value of all pixels in this spectral window.
        """
        return self.get(f"TDMEAN{self._iwin}")

    @property
    def window_rms(self):
        """
        RMS deviation from the mean value of all pixels in this spectral window.
        """
        return self.get(f"TDRMS{self._iwin}")

    @property
    def window_median(self):
        """
        Median value of all pixels in this spectral window.
        """
        return self.get(f"TDMEDN{self._iwin}")

    @property
    def window_min(self):
        """
        Minimum value of all pixels in this spectral window.
        """
        return self.get(f"TDMIN{self._iwin}")

    @property
    def window_max(self):
        """
        Maximum value of all pixels in this spectral window.
        """
        return self.get(f"TDMAX{self._iwin}")

    @property
    def window_saturated_pixels(self):
        """
        Number of saturated pixels in this spectral window.
        """
        return self.get(f"TSATPX{self._iwin}")

    @property
    def window_spikes(self):
        """
        Number of pixels identified as noise spikes in this spectral window.
        """
        return self.get(f"TSPIKE{self._iwin}")

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
