from copy import copy
from pathlib import Path

import dask
import dask.array as da
import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.frames import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.screens import SphericalScreen
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube, SpectrogramCubeSequence
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE, SLIT_WIDTH

__all__ = ["read_spectrograph_lvl2"]


class _IRISRasterGWCS(gwcs.WCS):
    def __init__(self, *args, basic_wcs=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._basic_wcs = basic_wcs

    @staticmethod
    def _supports_legacy_basic_wcs_bridge(world_objects):
        return (
            len(world_objects) == 2
            and isinstance(world_objects[0], SpectralCoord)
            and isinstance(world_objects[1], SkyCoord)
        )

    @property
    def celestial(self):
        if self._basic_wcs is None:
            msg = "This raster gWCS does not have an associated basic celestial WCS."
            raise AttributeError(msg)
        return self._basic_wcs.celestial

    def world_to_pixel(self, *world_objects):
        if self._basic_wcs is not None and self._supports_legacy_basic_wcs_bridge(world_objects):
            return self._basic_wcs.world_to_pixel(*world_objects)
        return super().world_to_pixel(*world_objects)

    def world_to_array_index(self, *world_objects):
        if self._basic_wcs is not None and self._supports_legacy_basic_wcs_bridge(world_objects):
            return self._basic_wcs.world_to_array_index(*world_objects)
        return super().world_to_array_index(*world_objects)


def _load_raster_window(filename, ext_idx, flip_step_axis):
    """
    Load and return scaled data for one spectral window from one raster FITS file.

    Used as a ``dask.delayed`` callable so that data is only read from disk
    when the resulting dask array is actually computed. The explicit cast to
    ``float32`` guarantees the dtype matches the ``da.from_delayed`` declaration.
    """
    data = fits.getdata(filename, ext=ext_idx).astype(np.float32)
    if flip_step_axis:
        return np.flip(data, axis=0).copy()
    return data


def _header_time(header, *keys):
    for key in keys:
        value = header.get(key)
        if value:
            return Time(value, format="isot", scale="utc")
    msg = f"Header is missing all usable time keys: {keys}"
    raise ValueError(msg)


def _make_observer(primary_header):
    base_time = _header_time(primary_header, "DATE_OBS", "STARTOBS")
    location = get_body_heliographic_stonyhurst("Earth", base_time.isot)
    observer = Helioprojective(
        primary_header["XCEN"] * u.arcsec,
        primary_header["YCEN"] * u.arcsec,
        observer=location,
        obstime=base_time,
    )
    with SphericalScreen(observer.observer):
        return observer.transform_to(HeliographicStonyhurst(obstime=base_time))


def _pc_matrix(lam, angle_1, angle_2):
    return angle_1, -1 * lam * angle_2, 1 / lam * angle_2, angle_1


def _prepare_raster_wcs_header(header, aux_data, spectral_band, *, flip):
    header = copy(header)
    if header.get("CUNIT1", "").lower() == "angstrom":
        header["CUNIT1"] = "nm"
        header["CRVAL1"] *= 0.1
        header["CDELT1"] *= 0.1
    offset_index = 34 if spectral_band == "FUV" else 45
    header["CRVAL3"] -= aux_data[:, offset_index].mean() * (SLIT_WIDTH.value / 2)
    if header["CDELT3"] == 0:
        header["CDELT3"] = 1e-10
        ang1, ang2, ang3, ang4 = _pc_matrix(
            header["CDELT3"] / header["CDELT2"],
            aux_data[:, 20].mean(),
            aux_data[:, 22].mean(),
        )
        header["PC2_2"] = ang1
        header["PC2_3"] = ang2
        header["PC3_2"] = ang3
        header["PC3_3"] = ang4
    if flip:
        header["PC1_3"] = 0
        header["PC2_3"] = -header["PC2_3"]
        header["PC3_2"] = -header["PC3_2"]
        header["CDELT3"] = -header["CDELT3"]
        header["CRPIX3"] = header["NAXIS3"] - header["CRPIX3"] + 1
    return header


def _create_basic_raster_wcs(header, observer):
    wcs = WCS(header)
    _set_wcs_aux_obs_coord(wcs, observer)
    return wcs


def _create_raster_gwcs(window_header, pc_all, crval_all, dt_all, t_ref, observer, basic_wcs):
    """
    Build a gWCS for an IRIS raster or sit-and-stare spectrograph window.

    Pixel axes: (0=dispersion, 1=slit, 2=step) in FITS order.
    World axes: (wavelength, helioprojective lon, lat, time, step).

    Parameters
    ----------
    window_header : FITS header
        Header from the window HDU of the first raster file.
    pc_all : `~astropy.units.Quantity`
        Per-step PC matrix, shape ``(N_steps, 2, 2)``, in ``u.pix``.
    crval_all : `~astropy.units.Quantity`
        Per-step celestial reference values, shape ``(N_steps, 2)``, in ``u.arcsec``.
    dt_all : `~astropy.units.Quantity`
        Per-step time offsets from ``t_ref``, shape ``(N_steps,)``, in ``u.s``.
    t_ref : `~astropy.time.Time`
        Reference time for the temporal frame.

    Returns
    -------
    `gwcs.WCS`
    """
    from dkist.wcs.models import AsymmetricMapping, CoupledCompoundModel, VaryingCelestialTransform  # NOQA: PLC0415
    from sunpy.time import parse_time  # NOQA: PLC0415

    if window_header.get("CUNIT1", "").lower() == "angstrom":
        cdelt1 = window_header["CDELT1"] * 0.1
        crval1 = window_header["CRVAL1"] * 0.1
    else:
        cdelt1 = window_header["CDELT1"]
        crval1 = window_header["CRVAL1"]
    crpix1 = window_header.get("CRPIX1", 1.0)
    spectral = m.Linear1D(
        slope=cdelt1 * u.nm / u.pix,
        intercept=(crval1 - cdelt1 * (crpix1 - 1)) * u.nm,
        name="Wavelength",
    )

    crpix_table = np.array([window_header["CRPIX2"], window_header["CRPIX3"]]) * u.pix
    cdelt = np.array([window_header["CDELT2"], window_header["CDELT3"]]) * (u.arcsec / u.pix)
    celestial_raw = VaryingCelestialTransform(
        cdelt=cdelt,
        pc_table=pc_all,
        crval_table=crval_all,
        crpix_table=crpix_table,
    )
    if np.isclose(window_header["CDELT3"], 1e-10):
        celestial = celestial_raw
    else:
        celestial = celestial_raw | m.Mapping((1, 0), name="SwapHelioprojectiveAxes")
        celestial.inverse = m.Mapping((1, 0, 2), n_inputs=3, name="SwapHelioprojectiveAxesInverseInputs") | (
            celestial_raw.inverse
        )

    temporal = m.Tabular1D(
        np.arange(pc_all.shape[0]) * u.pix,
        lookup_table=dt_all,
        fill_value=np.nan,
        bounds_error=False,
        method="linear",
        name="Time",
    )

    # (slit_pix, step_pix) → (slit, step, step, step): feeds celestial+temporal+step passthrough
    slit_step_mapping = AsymmetricMapping([0, 1, 1, 1], [0, 1], name="SlitStepMapping")
    non_spectral_rhs = CoupledCompoundModel("&", left=celestial, right=temporal, shared_inputs=1) & m.Identity(
        1, name="step"
    )
    non_spectral = slit_step_mapping | non_spectral_rhs
    # The explicit scan-step axis is the authoritative inverse for the raster/time dimension.
    # This keeps round-trips stable when sky coordinates repeat, e.g. sit-and-stare exposures.
    non_spectral.inverse = non_spectral_rhs.inverse | m.Mapping(
        (0, 3),
        n_inputs=4,
        name="SelectSlitAndExplicitStep",
    )
    forward_transform = spectral & non_spectral
    forward_transform.inverse = spectral.inverse & non_spectral.inverse

    base_time = parse_time(t_ref)
    spectral_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm, name="wavelength", axes_names=("wavelength",))
    celestial_frame = cf.CelestialFrame(
        axes_order=(1, 2),
        unit=(u.arcsec, u.arcsec),
        reference_frame=Helioprojective(observer=observer, obstime=observer.obstime),
        axis_physical_types=["custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"],
        axes_names=("helioprojective longitude", "helioprojective latitude"),
    )
    temporal_frame = cf.TemporalFrame(base_time, unit=(u.s,), axes_order=(3,), axes_names=("Time (UTC)",))
    step_frame = cf.CoordinateFrame(
        naxes=1,
        axes_order=(4,),
        axes_names=("scan_step",),
        axes_type=("STEP",),
        unit=(u.pix,),
        name="step",
    )
    output_frame = cf.CompositeFrame([spectral_frame, celestial_frame, temporal_frame, step_frame])
    pixel_frame = cf.CoordinateFrame(
        naxes=3,
        axes_order=(0, 1, 2),
        axes_names=["dispersion axis", "spatial along slit", "scan/step number"],
        axes_type=["PIXEL", "PIXEL", "PIXEL"],
        unit=(u.pix, u.pix, u.pix),
    )
    return _IRISRasterGWCS(
        forward_transform,
        input_frame=pixel_frame,
        output_frame=output_frame,
        basic_wcs=basic_wcs,
    )


def read_spectrograph_lvl2(
    filenames,
    *,
    spectral_windows=None,
    uncertainty=False,
    memmap=False,
    revert_v34=False,
):
    """
    Reads either a SINGLE IRIS level 2 spectrograph FITs or a list of them.

    Does not handle tar files.

    Parameters
    ----------
    filenames: `list` of `str` or `str`
        Filename of filenames to be read. They must all be associated with the same
        OBS number.
    spectral_windows: iterable of `str` or `str`
        Spectral windows to extract from files. Default=None, implies, extract all
        spectral windows.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), data is loaded lazily using Dask — arrays
        are not read from disk until you access them. This keeps memory usage low
        when loading many raster files. BSCALE/BZERO scaling is applied on first
        access. A lazy bad-pixel mask is built. Uncertainties are not computed in this mode.
    revert_v34 : `bool`, optional.
        Will undo the data and WCS flipping made to V34 observations.
        Defaults to `False`.

    Returns
    -------
        `RasterCollection`
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = sorted(str(f) for f in filenames)

    if len(filenames) > 1:
        date_obs_times = []
        has_missing_date_obs = False
        for fn in filenames:
            with fits.open(fn, memmap=False) as h:
                value = h[0].header.get("DATE_OBS")
                if value:
                    date_obs_times.append((Time(value, format="isot", scale="utc").unix, fn))
                else:
                    has_missing_date_obs = True
                    break
        if not has_missing_date_obs:
            filenames = [fn for _, fn in sorted(date_obs_times)]

    with fits.open(filenames[0], memmap=False, do_not_scale_image_data=False) as hdulist:
        hdulist.verify("silentfix")
        v34 = hdulist[0].header["STEPS_AV"] < -0.01
        primary_header = hdulist[0].header.copy()
        obsid = primary_header.get("OBSID", None)
        windows_in_obs = np.array(
            [primary_header[f"TDESC{i}"] for i in range(1, primary_header["NWIN"] + 1)],
        )
        window_name_to_hdu = {name: i + 1 for i, name in enumerate(windows_in_obs)}
        if not spectral_windows:
            spectral_windows_req = list(windows_in_obs)
        else:
            spectral_windows_req = [spectral_windows] if isinstance(spectral_windows, str) else list(spectral_windows)
            missing = [w for w in spectral_windows_req if w not in window_name_to_hdu]
            if missing:
                msg = f"Spectral windows {missing} not in file {filenames[0]}"
                raise ValueError(msg)
        window_fits_indices = [window_name_to_hdu[w] for w in spectral_windows_req]
        ref_shapes = {wi: (hdulist[wi].header["NAXIS1"], hdulist[wi].header["NAXIS2"]) for wi in window_fits_indices}
        observer = _make_observer(primary_header)

    for filename in filenames[1:]:
        with fits.open(filename, memmap=False) as h:
            h.verify("silentfix")
            file_startobs = h[0].header.get("STARTOBS", None)
            file_obsid = h[0].header.get("OBSID", None)
            if file_startobs != primary_header.get("STARTOBS"):
                msg = (
                    f"File {filename} has STARTOBS={file_startobs!r} which differs from "
                    f"{filenames[0]} ({primary_header.get('STARTOBS')!r}). "
                    "All files must belong to the same observation."
                )
                raise ValueError(msg)
            if obsid is not None and file_obsid != obsid:
                msg = (
                    f"File {filename} has OBSID={file_obsid!r} which differs from "
                    f"{filenames[0]} ({obsid!r}). "
                    "All files must belong to the same observation."
                )
                raise ValueError(msg)
            for wi, wname in zip(window_fits_indices, spectral_windows_req, strict=False):
                shape = (h[wi].header["NAXIS1"], h[wi].header["NAXIS2"])
                if shape != ref_shapes[wi]:
                    msg = (
                        f"Window {wname!r} in {filename} has shape {shape} "
                        f"but expected {ref_shapes[wi]} (from {filenames[0]}). "
                        "All files must have the same per-window dispersion/slit dimensions."
                    )
                    raise ValueError(msg)

    data_dict = {window_name: [] for window_name in spectral_windows_req}

    for filename in filenames:
        with fits.open(filename, memmap=False, do_not_scale_image_data=False) as hdulist:
            hdulist.verify("silentfix")
            file_primary_header = hdulist[0].header.copy()
            aux = hdulist[-2]
            file_startobs = _header_time(file_primary_header, "STARTOBS", "DATE_OBS")

            pc_indices = [aux.header[key] for key in ("PC2_2IX", "PC2_3IX", "PC3_2IX", "PC3_3IX")]
            pc = aux.data[:, pc_indices].reshape(-1, 2, 2) * u.pix
            times = file_startobs + TimeDelta(aux.data[:, aux.header["TIME"]] * u.s)
            exp_fuv = aux.data[:, aux.header["EXPTIMEF"]] * u.s
            exp_nuv = aux.data[:, aux.header["EXPTIMEN"]] * u.s
            obs_vr = aux.data[:, aux.header["OBS_VRIX"]] * u.m / u.s
            ophase = aux.data[:, aux.header["OPHASEIX"]] * u.one

            if v34 and not revert_v34:
                pc = pc[::-1]
                times = times[::-1]
                exp_fuv = exp_fuv[::-1]
                exp_nuv = exp_nuv[::-1]
                obs_vr = obs_vr[::-1]
                ophase = ophase[::-1]

            flip = v34 and not revert_v34
            t_ref = times[0]
            dt = (times - t_ref).to_value(u.s) * u.s
            for wi, window_name in zip(window_fits_indices, spectral_windows_req, strict=False):
                window_header = hdulist[wi].header.copy()
                meta = SGMeta(
                    file_primary_header,
                    window_name,
                    data_shape=(window_header["NAXIS3"], window_header["NAXIS2"], window_header["NAXIS1"]),
                )
                prepared_wcs_header = _prepare_raster_wcs_header(
                    window_header,
                    aux.data,
                    meta.spectral_band,
                    flip=flip,
                )
                if np.isclose(window_header["CDELT3"], 0):
                    crval = np.repeat(
                        [[prepared_wcs_header["CRVAL3"], prepared_wcs_header["CRVAL2"]]],
                        len(times),
                        axis=0,
                    ) * u.arcsec
                else:
                    offset_index = 34 if meta.spectral_band == "FUV" else 45
                    xcen = aux.data[:, aux.header["XCENIX"]] - aux.data[:, offset_index] * (SLIT_WIDTH.value / 2)
                    ycen = aux.data[:, aux.header["YCENIX"]]
                    crval = np.column_stack((ycen, xcen)) * u.arcsec
                    if flip:
                        crval = crval[::-1]
                basic_wcs = _create_basic_raster_wcs(prepared_wcs_header, observer)

                is_fuv = "FUV" in meta.detector
                dn_unit = DN_UNIT["FUV"] if is_fuv else DN_UNIT["NUV"]
                readout_noise = READOUT_NOISE["FUV"] if is_fuv else READOUT_NOISE["NUV"]
                exposure_times = exp_fuv if is_fuv else exp_nuv

                if memmap:
                    shape = (window_header["NAXIS3"], window_header["NAXIS2"], window_header["NAXIS1"])
                    delayed = dask.delayed(_load_raster_window)(filename, wi, flip)
                    data = da.from_delayed(delayed, shape=shape, dtype=np.float32)
                    data_mask = data == BAD_PIXEL_VALUE_SCALED
                    out_uncertainty = None
                else:
                    data = hdulist[wi].data.copy()
                    if flip:
                        data = np.flip(data, axis=0).copy()
                    data_mask = data == BAD_PIXEL_VALUE_SCALED
                    out_uncertainty = calculate_uncertainty(data, readout_noise, dn_unit) if uncertainty else None

                cube = SpectrogramCube(
                    data,
                    wcs=_create_raster_gwcs(
                        prepared_wcs_header,
                        pc,
                        crval,
                        dt,
                        t_ref,
                        observer,
                        basic_wcs,
                    ),
                    uncertainty=out_uncertainty,
                    unit=dn_unit,
                    meta=meta,
                    mask=data_mask,
                    _basic_wcs=basic_wcs,
                )
                cube.extra_coords.add("time", 0, times, None)
                cube.extra_coords.add("exposure time", 0, exposure_times, None)
                cube.extra_coords.add("observer radial velocity", 0, obs_vr, None)
                cube.extra_coords.add("orbital phase", 0, ophase, None)
                data_dict[window_name].append(cube)

    window_data_pairs = [
        (window_name, SpectrogramCubeSequence(data_dict[window_name], common_axis=0, meta=primary_header))
        for window_name in spectral_windows_req
    ]
    return RasterCollection(window_data_pairs, aligned_axes=(0, 1, 2))
