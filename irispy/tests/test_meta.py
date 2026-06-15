import astropy.units as u
from astropy.io import fits

from irispy.io.utils import read_files
from irispy.meta import SGMeta, SJIMeta


def _make_sg_header():
    header = fits.Header()
    header["INSTRUME"] = "IRIS"
    header["TELESCOP"] = "IRIS"
    header["DATA_LEV"] = 2
    header["OBSID"] = 1
    header["OBS_DESC"] = "Test obs"
    header["NWIN"] = 1
    header["TDESC1"] = "Si IV 1403"
    header["TWAVE1"] = 1402.77
    header["TWMIN1"] = 1398.61
    header["TWMAX1"] = 1406.03
    header["TDET1"] = "FUV2"
    return header


def test_sgmeta_rest_wavelength():
    meta = SGMeta(_make_sg_header(), "Si IV 1403")
    assert meta.rest_wavelength.unit == u.nm
    assert u.isclose(meta.rest_wavelength, 140.277 * u.nm, rtol=1e-4)


def test_sgmeta_detector_band_fuv():
    meta = SGMeta(_make_sg_header(), "Si IV 1403")
    assert meta.detector_band == "FUV"


def test_sgmeta_detector_band_nuv():
    header = _make_sg_header()
    header["TDET1"] = "NUV"
    meta = SGMeta(header, "Si IV 1403")
    assert meta.detector_band == "NUV"


def test_sgmeta_detector_band_sji():
    header = fits.Header()
    header["INSTRUME"] = "IRIS"
    header["TELESCOP"] = "IRIS"
    header["TDET1"] = "SJI"
    header["TDESC1"] = "SJI_1330"
    header["TWAVE1"] = 1330.0
    header["TWMIN1"] = 1330.0
    header["TWMAX1"] = 1330.0
    meta = SJIMeta(header)
    assert meta.detector_band == "SJI"


def test_sgmeta_observer_uses_campaign_start_when_date_obs_is_blank():
    header = _make_sg_header()
    header["DATE_OBS"] = ""
    header["STARTOBS"] = "2014-03-29T14:09:38.830"
    header["XCEN"] = 0.0
    header["YCEN"] = 0.0
    meta = SGMeta(header, "Si IV 1403")

    assert meta.date_reference is None
    assert meta.observer.obstime.isot == "2014-03-29T14:09:38.830"


def test_sgmeta_temporal_cadence():
    header = _make_sg_header()
    header["CADEX_AV"] = 9.264
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    cadence = meta.temporal_cadence
    assert u.isclose(cadence, 9.264 * u.s, rtol=1e-4)


def test_sgmeta_real_sns_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"][0]
    meta = cube.meta

    assert meta.detector_band == "FUV"
    assert meta.rest_wavelength.unit == u.nm
    assert u.isclose(meta.rest_wavelength, 140.277 * u.nm, rtol=1e-3)
    assert meta.temporal_cadence.unit.is_equivalent(u.s)
    assert meta.camera == 1
    assert meta.number_of_spectral_windows is not None


def test_sgmeta_camera():
    header = _make_sg_header()
    header["CAMERA"] = 1
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.camera == 1


def test_sgmeta_sun_angular_radius_from_rsun():
    header = _make_sg_header()
    header["RSUN_OBS"] = 975.0
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert u.isclose(meta.sun_angular_radius, 975.0 * u.arcsec)


def test_sgmeta_sun_angular_radius_from_dsun():
    header = _make_sg_header()
    header["DSUN_OBS"] = 1.5e11
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    radius = meta.sun_angular_radius
    assert radius.unit.is_equivalent(u.arcsec)


def test_sgmeta_observer_radial_velocity():
    header = _make_sg_header()
    header["OBS_VR"] = 3500.0
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert u.isclose(meta.observer_radial_velocity, 3500.0 * u.m / u.s)


def test_sgmeta_number_of_spectral_windows():
    meta = SGMeta(_make_sg_header(), "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.number_of_spectral_windows == 1


def test_sgmeta_raster_repetition():
    header = _make_sg_header()
    header["RASRPT"] = 3
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.raster_repetition == 3


def test_meta_exposure_time():
    header = _make_sg_header()
    header["EXPTIME"] = 12.5
    header["EXPMIN"] = 10.0
    header["EXPMAX"] = 15.0
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert u.isclose(meta.exposure_time, 12.5 * u.s)
    assert u.isclose(meta.exposure_time_min, 10.0 * u.s)
    assert u.isclose(meta.exposure_time_max, 15.0 * u.s)


def test_meta_data_type_and_unit():
    header = _make_sg_header()
    header["BTYPE"] = "Intensity"
    header["BUNIT"] = "DN/s"
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.data_type == "Intensity"
    assert meta.data_unit == "DN/s"


def test_meta_data_status():
    header = _make_sg_header()
    header["STATUS"] = "Final"
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.data_status == "Final"


def test_meta_build_and_reformat_info():
    header = _make_sg_header()
    header["BLD_VERS"] = "v9.4"
    header["VER_RF2"] = "1.2.3"
    header["DATE_RF2"] = "2024-01-15T10:30:00"
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.build_version == "v9.4"
    assert meta.reformat_version == "1.2.3"
    assert meta.reformat_date.scale == "utc"


def test_meta_observing_label_and_title():
    header = _make_sg_header()
    header["OBSLABEL"] = "OBS_12345"
    header["OBSTITLE"] = "Test Campaign"
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.observing_label == "OBS_12345"
    assert meta.observing_title == "Test Campaign"


def test_meta_lut_id():
    header = _make_sg_header()
    header["LUTID"] = 7
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.lut_id == 7


def test_meta_number_of_exposures():
    header = _make_sg_header()
    header["NEXP"] = 180
    header["NEXP_PRP"] = 200
    header["NEXPOBS"] = 400
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.number_of_exposures == 180
    assert meta.number_of_exposures_planned == 200
    assert meta.number_of_exposures_observation == 400


def test_meta_data_quality_counts():
    header = _make_sg_header()
    header["NSATPIX"] = 42
    header["NSPIKES"] = 15
    header["PERCENTD"] = 98.5
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.number_of_saturated_pixels == 42
    assert meta.number_of_spikes == 15
    assert meta.percent_data == 98.5


def test_meta_data_statistics():
    header = _make_sg_header()
    header["DATAMEAN"] = 120.5
    header["DATARMS"] = 45.2
    header["DATAMEDN"] = 110.0
    header["DATAMIN"] = 5.0
    header["DATAMAX"] = 500.0
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.data_mean == 120.5
    assert meta.data_rms == 45.2
    assert meta.data_median == 110.0
    assert meta.data_min == 5.0
    assert meta.data_max == 500.0


def test_sgmeta_raster_step_properties():
    header = _make_sg_header()
    header["STEPS_AV"] = 0.35
    header["STEPS_DV"] = 0.02
    header["STEPT_AV"] = 18.5
    header["STEPT_DV"] = 1.2
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.step_size_average == 0.35
    assert meta.step_size_stddev == 0.02
    assert meta.step_time_average == 18.5
    assert meta.step_time_stddev == 1.2


def test_sgmeta_cadence_planned():
    header = _make_sg_header()
    header["CADPL_AV"] = 15.0
    header["CADPL_DV"] = 0.5
    header["CADEX_DV"] = 0.3
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert u.isclose(meta.cadence_planned_average, 15.0 * u.s)
    assert u.isclose(meta.cadence_planned_stddev, 0.5 * u.s)
    assert u.isclose(meta.cadence_executed_stddev, 0.3 * u.s)


def test_sgmeta_raster_type():
    header = _make_sg_header()
    header["RASTYPDX"] = 3
    header["RASTYPNX"] = 8
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.raster_type_index == 3
    assert meta.raster_type_total == 8


def test_sgmeta_missing_files():
    header = _make_sg_header()
    header["MISSRAS"] = 2
    header["MISSOBS"] = 5
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.number_of_missing_raster_files == 2
    assert meta.number_of_missing_observation_files == 5


def test_sgmeta_window_statistics():
    header = _make_sg_header()
    header["TDMEAN1"] = 150.0
    header["TDRMS1"] = 30.0
    header["TDMEDN1"] = 145.0
    header["TDMIN1"] = 10.0
    header["TDMAX1"] = 400.0
    header["TSATPX1"] = 3
    header["TSPIKE1"] = 7
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    assert meta.window_mean == 150.0
    assert meta.window_rms == 30.0
    assert meta.window_median == 145.0
    assert meta.window_min == 10.0
    assert meta.window_max == 400.0
    assert meta.window_saturated_pixels == 3
    assert meta.window_spikes == 7


def test_sjmeta_base_properties():
    header = fits.Header()
    header["INSTRUME"] = "IRIS"
    header["TELESCOP"] = "IRIS"
    header["DATA_LEV"] = 2
    header["TDET1"] = "SJI"
    header["TDESC1"] = "SJI_1330"
    header["TWAVE1"] = 1330.0
    header["TWMIN1"] = 1330.0
    header["TWMAX1"] = 1330.0
    header["EXPTIME"] = 8.0
    header["BTYPE"] = "Intensity"
    header["NSATPIX"] = 0
    meta = SJIMeta(header, data_shape=(10, 10))
    assert u.isclose(meta.exposure_time, 8.0 * u.s)
    assert meta.data_type == "Intensity"
    assert meta.number_of_saturated_pixels == 0
