import warnings
from copy import copy, deepcopy
from pathlib import Path

import dask.array as da
import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from dask import delayed
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

from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube, SpectrogramCubeSequence
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE, SLIT_WIDTH

__all__ = ["read_spectrograph_lvl2"]

LAZY_RASTER_TARGET_CHUNK_BYTES = 32 * 1024


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


def _raster_wcs_bad_row_mask(pc, crval):
    """Return AUX rows whose PC or CRVAL table entries are unusable."""
    pc_bad = np.array([np.allclose(pc[i], 0) for i in range(pc.shape[0])])
    crval_bad = np.array([np.allclose(crval[i], 0) for i in range(crval.shape[0])])
    return pc_bad | crval_bad


def _sanitize_raster_wcs_tables(pc, crval, fallback_pc=None, fallback_crval=None):
    """Replace all-zero PC/crval rows via neighbour interpolation or fallback."""
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


def _concatenate_scan_aligned_values(values):
    first = values[0]
    if isinstance(first, SkyCoord):
        return SkyCoord(np.concatenate(values))
    return np.concatenate(values)


def _concatenate_uncertainty(cubes):
    if all(cube.uncertainty is None for cube in cubes):
        return None
    if any(cube.uncertainty is None for cube in cubes):
        msg = "Cannot combine a raster sequence when only some cubes contain uncertainty."
        raise ValueError(msg)
    uncertainty_type = type(cubes[0].uncertainty)
    return uncertainty_type(np.concatenate([cube.uncertainty.array for cube in cubes], axis=0))


def _concatenate_mask(cubes):
    if all(cube.mask is None for cube in cubes):
        return None
    return np.concatenate(
        [np.zeros(cube.shape, dtype=bool) if cube.mask is None else np.asarray(cube.mask, dtype=bool) for cube in cubes],
        axis=0,
    )


def _combine_raster_meta(cubes, combined_shape):
    meta = deepcopy(cubes[0].meta)
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    meta["NAXIS3"] = combined_shape[0]
    for key in ("DATE_END", "ENDOBS"):
        if cubes[-1].meta.get(key) is not None:
            meta[key] = cubes[-1].meta[key]
    for key in ("exposure time", "exposure FOV center", "observer radial velocity", "orbital phase"):
        meta.add(
            key,
            _concatenate_scan_aligned_values([cube.meta[key] for cube in cubes]),
            axes=0,
            overwrite=True,
        )
    return meta


def _validate_combinable_raster_sequence(sequence):
    cubes = list(sequence)
    if not cubes:
        msg = "Cannot combine an empty raster sequence."
        raise ValueError(msg)
    if len(cubes) == 1:
        return cubes
    if getattr(sequence, "_common_axis", None) != 0:
        msg = "Only raster sequences with common_axis=0 can be combined into one cube."
        raise NotImplementedError(msg)
    if any(cube.shape[1:] != cubes[0].shape[1:] for cube in cubes[1:]):
        msg = "All rasters in the sequence must have the same slit and wavelength dimensions."
        raise ValueError(msg)
    if any(cube.unit != cubes[0].unit for cube in cubes[1:]):
        msg = "All rasters in the sequence must have the same data unit."
        raise ValueError(msg)
    required_attrs = (
        "_raster_wcs_header",
        "_raster_pc_table",
        "_raster_crval_table",
        "_raster_observer",
    )
    if any(not all(hasattr(cube, attr) for attr in required_attrs) for cube in cubes):
        msg = "Sequence cubes do not expose the raster WCS metadata needed to build a combined cube."
        raise ValueError(msg)
    return cubes


def _build_combined_raster_cube(cubes, data, *, mask, memmap):
    times = Time(np.concatenate([cube.time for cube in cubes]))
    pc_all = np.concatenate([cube._raster_pc_table for cube in cubes], axis=0)
    crval_all = np.concatenate([cube._raster_crval_table for cube in cubes], axis=0)
    starts = np.cumsum([0, *[c.shape[0] for c in cubes[:-1]]])
    raster_memmap_segments = None
    if memmap:
        raster_memmap_segments = [
            (
                start,
                start + cube.shape[0],
                cube._memmap_path,
                cube._memmap_ext,
                cube._flip,
                getattr(cube, "_memmap_slice", slice(0, cube.shape[0])),
            )
            for start, cube in zip(starts, cubes, strict=True)
        ]
    combined_cube = SpectrogramCube(
        data,
        wcs=_create_raster_gwcs(
            cubes[0]._raster_wcs_header,
            pc_all,
            crval_all,
            (times - times[0]).to_value(u.s) * u.s,
            times[0],
            cubes[0]._raster_observer,
        ),
        uncertainty=_concatenate_uncertainty(cubes),
        unit=cubes[0].unit,
        meta=_combine_raster_meta(cubes, data.shape),
        mask=mask,
        _basic_wcs_segments=[
            (start, start + cube.shape[0], cube.basic_wcs) for start, cube in zip(starts, cubes, strict=True)
        ],
        _raster_boundaries=[(start, start + cube.shape[0]) for start, cube in zip(starts, cubes, strict=True)],
        _memmap=memmap,
        _raster_memmap_segments=raster_memmap_segments,
    )
    combined_cube.extra_coords.add("time", 0, times, physical_types="time")
    combined_cube._raster_wcs_header = cubes[0]._raster_wcs_header
    combined_cube._raster_pc_table = pc_all
    combined_cube._raster_crval_table = crval_all
    combined_cube._raster_observer = cubes[0]._raster_observer
    return combined_cube


def _lazy_raster_scan_chunk_rows(cube):
    row_bytes = int(np.prod(cube.shape[1:]) * np.dtype(cube.data.dtype).itemsize)
    return max(1, min(cube.shape[0], LAZY_RASTER_TARGET_CHUNK_BYTES // max(row_bytes, 1)))


def _load_memmap_raster_chunk(filename, ext, *, scan_slice, flip):
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as hdulist:
        data = hdulist[ext].data
        if flip:
            data = np.flip(data, axis=0)
        return np.array(data[scan_slice], copy=True)


def _build_lazy_raster_data(cubes):
    chunks = []
    for cube in cubes:
        memmap_slice = getattr(cube, "_memmap_slice", slice(0, cube.shape[0]))
        if memmap_slice.step not in (None, 1):
            msg = "Lazy raster memmap chunks only support contiguous scan-axis slices."
            raise NotImplementedError(msg)
        chunk_rows = _lazy_raster_scan_chunk_rows(cube)
        base_start = 0 if memmap_slice.start is None else memmap_slice.start
        for local_start in range(0, cube.shape[0], chunk_rows):
            local_stop = min(local_start + chunk_rows, cube.shape[0])
            chunk = da.from_delayed(
                delayed(_load_memmap_raster_chunk)(
                    cube._memmap_path,
                    cube._memmap_ext,
                    scan_slice=slice(base_start + local_start, base_start + local_stop),
                    flip=cube._flip,
                ),
                shape=(local_stop - local_start, *cube.shape[1:]),
                dtype=cube.data.dtype,
            )
            chunks.append(chunk)
    return da.concatenate(chunks, axis=0)


def _combine_raster_sequence_lazy(sequence):
    cubes = _validate_combinable_raster_sequence(sequence)
    if len(cubes) == 1:
        return cubes[0]
    data = _build_lazy_raster_data(cubes)
    mask = data == BAD_PIXEL_VALUE_SCALED
    return _build_combined_raster_cube(cubes, data, mask=mask, memmap=True)


def _combine_raster_sequence(sequence):
    cubes = _validate_combinable_raster_sequence(sequence)
    if len(cubes) == 1:
        return cubes[0]
    if any(getattr(cube, "_memmap", False) or isinstance(cube.data, np.memmap) for cube in cubes):
        msg = "Use _combine_raster_sequence_lazy() for memmap-backed raster sequences."
        raise NotImplementedError(msg)
    data = np.concatenate([cube.data for cube in cubes], axis=0)
    return _build_combined_raster_cube(cubes, data, mask=_concatenate_mask(cubes), memmap=False)


def _finalize_window_object(cubes, primary_header, *, memmap):
    if len(cubes) == 1:
        cube = cubes[0]
        cube._raster_boundaries = [(0, cube.shape[0])]
        return cube
    if memmap:
        return _combine_raster_sequence_lazy(SpectrogramCubeSequence(cubes, common_axis=0, meta=primary_header))
    return _combine_raster_sequence(SpectrogramCubeSequence(cubes, common_axis=0, meta=primary_header))


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
            window_fits_indices = np.nonzero(np.isin(windows_in_obs, spectral_windows))[0] + 1
        data_dict = {window_name: [] for window_name in spectral_windows_req}
        observer = _make_observer(primary_header)

    # Running mean of good WCS table rows across files (used as fallback).
    running_pc_sum = np.zeros((2, 2))
    running_crval_sum = np.zeros(2)
    running_count = 0

    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
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

                # Update running mean from good rows in the first window of each file.
                if i == 0:
                    good_mask = ~_raster_wcs_bad_row_mask(pc, crval)
                    if good_mask.any():
                        running_pc_sum += pc[good_mask].sum(axis=0).to_value(u.pix)
                        running_crval_sum += crval[good_mask].sum(axis=0).to_value(u.arcsec)
                        running_count += good_mask.sum()

                fallback_pc = (running_pc_sum / running_count * u.pix) if running_count > 0 else None
                fallback_crval = (running_crval_sum / running_count * u.arcsec) if running_count > 0 else None
                pc_sanitized, crval = _sanitize_raster_wcs_tables(pc.copy(), crval, fallback_pc, fallback_crval)
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
                cube._memmap_slice = slice(0, cube.shape[0])
                data_dict[window_name].append(cube)
    window_data_pairs = [
        (_window_name, _finalize_window_object(cubes, primary_header, memmap=memmap))
        for _window_name, cubes in data_dict.items()
    ]
    return RasterCollection(window_data_pairs, aligned_axes=(0, 1))
