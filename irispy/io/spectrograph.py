import warnings
from copy import copy
from pathlib import Path

import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from dkist.wcs.models import AsymmetricMapping, CoupledCompoundModel, VaryingCelestialTransform
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.frames import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.screens import SphericalScreen
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord
from sunpy.time import parse_time

from irispy.io._raster_combine import _finalize_window_object
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE, SLIT_WIDTH

__all__ = ["read_spectrograph_lvl2"]


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


OBSERVATION_COMPATIBILITY_KEYS = (
    "TELESCOP",
    "INSTRUME",
    "OBSID",
    "STARTOBS",
    "OBS_DESC",
    "NWIN",
)

WINDOW_COMPATIBILITY_KEYS = (
    "NAXIS1",
    "NAXIS2",
    "CTYPE1",
    "CUNIT1",
    "CDELT1",
    "CRPIX1",
    "CUNIT2",
    "CUNIT3",
    "CDELT2",
    "CDELT3",
    "CRPIX2",
    "CRPIX3",
)


def _spectral_windows_from_header(header):
    return np.array([header[f"TDESC{i}"] for i in range(1, header["NWIN"] + 1)], dtype=str)


def _validate_spectrograph_file_compatible(
    reference_header,
    reference_window_headers,
    hdulist,
    filename,
    window_fits_indices,
    window_names,
):
    header = hdulist[0].header
    for key in OBSERVATION_COMPATIBILITY_KEYS:
        expected = reference_header.get(key)
        actual = header.get(key)
        if actual != expected:
            msg = (
                "Spectrograph files must belong to one compatible observation; "
                f"{filename} has {key}={actual!r}, expected {expected!r}."
            )
            raise ValueError(msg)

    reference_windows = _spectral_windows_from_header(reference_header)
    windows = _spectral_windows_from_header(header)
    if not np.array_equal(windows, reference_windows):
        msg = (
            "Spectrograph files must have the same spectral-window order; "
            f"{filename} has {windows.tolist()}, expected {reference_windows.tolist()}."
        )
        raise ValueError(msg)

    for reference_window_header, window_index, window_name in zip(
        reference_window_headers,
        window_fits_indices,
        window_names,
        strict=True,
    ):
        window_header = hdulist[window_index].header
        for key in WINDOW_COMPATIBILITY_KEYS:
            expected = reference_window_header.get(key)
            actual = window_header.get(key)
            if actual != expected:
                msg = (
                    f"Spectral window {window_name!r} in {filename} is not compatible with the first file: "
                    f"{key}={actual!r}, expected {expected!r}."
                )
                raise ValueError(msg)


def _raster_wcs_bad_row_mask(pc, crval):
    """Return AUX rows whose PC or CRVAL table entries are unusable."""
    pc_values = pc.to_value(u.pix) if hasattr(pc, "to_value") else np.asarray(pc)
    crval_values = crval.to_value(u.arcsec) if hasattr(crval, "to_value") else np.asarray(crval)
    pc_bad = np.isclose(pc_values, 0).all(axis=(1, 2))
    crval_bad = np.isclose(crval_values, 0).all(axis=1)
    return pc_bad | crval_bad


def _sanitize_raster_wcs_tables(pc, crval, fallback_pc=None, fallback_crval=None, *, bad_rows=None):
    """Replace all-zero PC/crval rows via neighbour interpolation or fallback."""
    if bad_rows is None:
        bad_rows = _raster_wcs_bad_row_mask(pc, crval)

    if not bad_rows.any():
        return pc, crval

    n_bad = int(bad_rows.sum())
    warnings.warn(
        f"Found {n_bad} step(s) with all-zero WCS tables in raster aux data. Interpolating from neighbouring steps.",
        UserWarning,
        stacklevel=3,
    )

    good_indices = np.nonzero(~bad_rows)[0]
    if good_indices.size == 0:
        if fallback_pc is None or fallback_crval is None:
            msg = "All WCS table rows are bad and no fallback values are available."
            raise ValueError(msg)
        warnings.warn(
            "All steps in this file have bad WCS tables. Using fallback values from observation mean.",
            UserWarning,
            stacklevel=3,
        )
        for i in range(pc.shape[0]):
            pc[i] = fallback_pc
            crval[i] = fallback_crval
        return pc, crval

    for i in np.where(bad_rows)[0]:
        before = good_indices[good_indices < i]
        after = good_indices[good_indices > i]
        if before.size > 0 and after.size > 0:
            b = before[-1]
            a = after[0]
            weight = (i - b) / (a - b)
            pc[i] = pc[b] * (1 - weight) + pc[a] * weight
            crval[i] = crval[b] * (1 - weight) + crval[a] * weight
        elif before.size > 0:
            pc[i] = pc[before[-1]]
            crval[i] = crval[before[-1]]
        elif after.size > 0:
            pc[i] = pc[after[0]]
            crval[i] = crval[after[0]]
    return pc, crval


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
        dispersion_ratio = header["CDELT3"] / header["CDELT2"]
        angle_1 = aux_data[:, 20].mean()
        angle_2 = aux_data[:, 22].mean()
        header["PC2_2"] = angle_1
        header["PC2_3"] = -dispersion_ratio * angle_2
        header["PC3_2"] = angle_2 / dispersion_ratio
        header["PC3_3"] = angle_1
    if flip:
        header["PC1_3"] = 0
        header["PC2_3"] = -header["PC2_3"]
        header["PC3_2"] = -header["PC3_2"]
        header["CDELT3"] = -header["CDELT3"]
        header["CRPIX3"] = header["NAXIS3"] - header["CRPIX3"] + 1
    return header


def _create_raster_gwcs(window_header, pc_all, crval_all, dt_all, t_ref, observer):
    """
    Build the raster gWCS from per-step AUX sky and timing tables.

    The spectral axis stays a 1D linear transform. The scan axis expands into
    helioprojective sky coordinates, elapsed time from ``t_ref``, and an
    explicit scan-step coordinate so the inverse stays stable for crops and
    other round-trips where sky or time alone are not unique.

    Notes
    -----
    The gWCS inverse deliberately ignores the time component (axis 3) and
    uses sky position plus the explicit scan-step coordinate to determine the
    pixel index. This is required because time may not be monotonic across
    flipped or repeated rasters. Consequently, ``world_to_array_index`` will
    return the same pixel regardless of the time value passed in the world
    tuple.
    """
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
    slit_step_mapping = AsymmetricMapping([0, 1, 1, 1], [0, 1], name="SlitStepMapping")
    non_spectral_rhs = CoupledCompoundModel("&", left=celestial, right=temporal, shared_inputs=1) & m.Identity(
        1, name="step"
    )
    non_spectral = slit_step_mapping | non_spectral_rhs
    # Time is redundant with the explicit scan-step output and may not be monotonic
    # across flipped or repeated rasters, so the inverse keys off sky + step only.
    non_spectral.inverse = (
        m.Mapping(
            (0, 1, 3),
            n_inputs=4,
            name="SelectSkyAndExplicitStep",
        )
        | celestial.inverse
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
    temporal_frame = cf.TemporalFrame(base_time, unit=(u.s,), axes_order=(3,), axes_names=("Seconds from Start (s)",))
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
    return gwcs.WCS(
        forward_transform,
        input_frame=pixel_frame,
        output_frame=output_frame,
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
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        the data units will be unscaled, not the usual data numbers (DN).
    revert_v34 : `bool`, optional.
        Will undo the data and WCS flipping made to V34 observations.
        Defaults to `False`.

    Returns
    -------
    `RasterCollection`
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = [str(f) for f in filenames]
    compute_uncertainty = uncertainty and not memmap
    with fits.open(filenames[0], memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
        v34 = hdulist[0].header["STEPS_AV"] < -0.01
        hdulist.verify("silentfix")
        primary_header = hdulist[0].header.copy()
        windows_in_obs = np.array(
            [primary_header[f"TDESC{i}"] for i in range(1, primary_header["NWIN"] + 1)],
        )
        if not spectral_windows:
            spectral_windows_req = windows_in_obs
            window_fits_indices = range(1, len(hdulist) - 2)
        else:
            spectral_windows_req = [spectral_windows] if isinstance(spectral_windows, str) else spectral_windows
            spectral_windows_req = np.asarray(spectral_windows_req, dtype="U")
            window_is_in_obs = np.asarray([window in windows_in_obs for window in spectral_windows_req])
            if not all(window_is_in_obs):
                missing_windows = spectral_windows_req[~window_is_in_obs]
                msg = f"Spectral windows {missing_windows.tolist()} not in file {filenames[0]}"
                raise ValueError(msg)
            window_fits_indices = [int(np.where(windows_in_obs == window)[0][0]) + 1 for window in spectral_windows_req]
        data_dict = {window_name: [] for window_name in spectral_windows_req}
        reference_window_headers = [hdulist[index].header.copy() for index in window_fits_indices]
        observer = _make_observer(primary_header)

    # Per-window running means of good WCS table rows across files, used as
    # fallback only when a later file has no good rows for that window.
    running_wcs_fallbacks = {window_name: [np.zeros((2, 2)), np.zeros(2), 0] for window_name in spectral_windows_req}

    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
            _validate_spectrograph_file_compatible(
                primary_header,
                reference_window_headers,
                hdulist,
                filename,
                window_fits_indices,
                spectral_windows_req,
            )
            aux = hdulist[-2]
            file_startobs = _header_time(hdulist[0].header, "STARTOBS", "DATE_OBS")
            times = file_startobs + TimeDelta(aux.data[:, aux.header["TIME"]] * u.s)
            fov_center = SkyCoord(
                Tx=aux.data[:, aux.header["XCENIX"]],
                Ty=aux.data[:, aux.header["YCENIX"]],
                unit=u.arcsec,
                frame=Helioprojective,
            )
            obs_vrix = aux.data[:, aux.header["OBS_VRIX"]] * u.m / u.s
            ophaseix = aux.data[:, aux.header["OPHASEIX"]]
            exposure_times_fuv = aux.data[:, aux.header["EXPTIMEF"]] * u.s
            exposure_times_nuv = aux.data[:, aux.header["EXPTIMEN"]] * u.s
            pc_indices = [aux.header[key] for key in ("PC2_2IX", "PC2_3IX", "PC3_2IX", "PC3_3IX")]
            pc = aux.data[:, pc_indices].reshape(-1, 2, 2) * u.pix

            flip = v34 and not revert_v34
            if flip:
                times = times[::-1]
                obs_vrix = obs_vrix[::-1]
                ophaseix = ophaseix[::-1]
                exposure_times_fuv = exposure_times_fuv[::-1]
                exposure_times_nuv = exposure_times_nuv[::-1]
                pc = pc[::-1]
            t_ref = times[0]
            dt = (times - t_ref).to_value(u.s) * u.s

            for i, window_name in enumerate(spectral_windows_req):
                window_header = hdulist[window_fits_indices[i]].header.copy()
                meta = SGMeta(
                    hdulist[0].header,
                    window_name,
                    data_shape=hdulist[window_fits_indices[i]].data.shape,
                )
                exposure_times = exposure_times_nuv
                dn_unit = DN_UNIT["NUV"]
                readout_noise = READOUT_NOISE["NUV"]
                if "FUV" in meta.detector:
                    exposure_times = exposure_times_fuv
                    dn_unit = DN_UNIT["FUV"]
                    readout_noise = READOUT_NOISE["FUV"]
                meta.add("exposure time", exposure_times, None, 0)
                meta.add("exposure FOV center", fov_center, None, 0)
                meta.add("observer radial velocity", obs_vrix, None, 0)
                meta.add("orbital phase", ophaseix, None, 0)
                prepared_wcs_header = _prepare_raster_wcs_header(
                    window_header,
                    aux.data,
                    meta.spectral_band,
                    flip=flip,
                )
                if np.isclose(window_header["CDELT3"], 0):
                    # Sit-and-stare: CRVAL comes from the header, only PC may need fallback.
                    crval = (
                        np.repeat(
                            [[prepared_wcs_header["CRVAL3"], prepared_wcs_header["CRVAL2"]]],
                            len(times),
                            axis=0,
                        )
                        * u.arcsec
                    )
                else:
                    offset_index = 34 if meta.spectral_band == "FUV" else 45
                    xcen = aux.data[:, aux.header["XCENIX"]] - aux.data[:, offset_index] * (SLIT_WIDTH.value / 2)
                    ycen = aux.data[:, aux.header["YCENIX"]]
                    crval = np.column_stack((ycen, xcen)) * u.arcsec
                    if flip:
                        crval = crval[::-1]

                running_pc_sum, running_crval_sum, running_count = running_wcs_fallbacks[window_name]
                fallback_pc = (running_pc_sum / running_count * u.pix) if running_count > 0 else None
                fallback_crval = (running_crval_sum / running_count * u.arcsec) if running_count > 0 else None
                bad_rows = _raster_wcs_bad_row_mask(pc, crval)
                good_mask = ~bad_rows
                pc_sanitized, crval = _sanitize_raster_wcs_tables(
                    pc.copy(),
                    crval,
                    fallback_pc,
                    fallback_crval,
                    bad_rows=bad_rows,
                )
                if good_mask.any():
                    running_wcs_fallbacks[window_name][0] += pc[good_mask].sum(axis=0).to_value(u.pix)
                    running_wcs_fallbacks[window_name][1] += crval[good_mask].sum(axis=0).to_value(u.arcsec)
                    running_wcs_fallbacks[window_name][2] += int(good_mask.sum())

                basic_wcs = WCS(prepared_wcs_header)
                _set_wcs_aux_obs_coord(basic_wcs, observer)

                out_uncertainty = None
                data_mask = None
                if not memmap:
                    data_mask = hdulist[window_fits_indices[i]].data == BAD_PIXEL_VALUE_SCALED
                if compute_uncertainty:
                    out_uncertainty = calculate_uncertainty(
                        hdulist[window_fits_indices[i]].data,
                        readout_noise,
                        dn_unit,
                    )
                if flip:
                    data = np.flip(hdulist[window_fits_indices[i]].data, axis=0)
                else:
                    data = hdulist[window_fits_indices[i]].data
                cube = SpectrogramCube(
                    data,
                    wcs=_create_raster_gwcs(
                        prepared_wcs_header,
                        pc_sanitized,
                        crval,
                        dt,
                        t_ref,
                        observer,
                    ),
                    uncertainty=out_uncertainty,
                    unit=dn_unit,
                    meta=meta,
                    mask=data_mask,
                    _basic_wcs=basic_wcs,
                    _memmap=memmap,
                )
                cube.extra_coords.add("time", 0, times, physical_types="time")
                cube._raster_wcs_header = prepared_wcs_header
                cube._raster_pc_table = pc_sanitized
                cube._raster_crval_table = crval
                cube._raster_observer = observer
                cube._memmap_path = filename
                cube._memmap_ext = window_fits_indices[i]
                cube._flip = flip
                data_dict[window_name].append(cube)
    window_data_pairs = [
        (_window_name, _finalize_window_object(cubes, memmap=memmap, create_raster_gwcs=_create_raster_gwcs))
        for _window_name, cubes in data_dict.items()
    ]
    return RasterCollection(window_data_pairs, aligned_axes=(0, 1))
