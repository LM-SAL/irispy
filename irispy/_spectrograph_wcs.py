import warnings
from copy import copy
from numbers import Integral

import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

from dkist.wcs.models import (
    AsymmetricMapping,
    BaseVaryingCelestialTransform,
    CoupledCompoundModel,
    VaryingCelestialTransform,
)
from sunpy import log as logger
from sunpy.coordinates.frames import Helioprojective
from sunpy.time import parse_time

from irispy._interpolation import _interpolate_bad_axis0_rows
from irispy.utils.constants import SLIT_WIDTH

AUX_FUV_SLIT_OFFSET_COLUMN = 34
AUX_NUV_SLIT_OFFSET_COLUMN = 45
AUX_PC_ANGLE_1_COLUMN = 20
AUX_PC_ANGLE_2_COLUMN = 22
SIT_AND_STARE_CDELT3_PLACEHOLDER = 1e-10
_SPECTROGRAM_CUBE_METADATA_KWARGS = (
    "_fits_wcs",
    "_fits_wcs_segments",
    "_raster_boundaries",
    "_memmap",
    "_raster_wcs_header",
    "_raster_pc_table",
    "_raster_crval_table",
    "_raster_observer",
    "_memmap_path",
    "_memmap_ext",
    "_flip",
    "_separate_raster_axis",
    "_sit_and_stare",
)
_SPECTROGRAM_CUBE_METADATA_DEFAULTS = {
    "_memmap": False,
    "_flip": False,
    "_separate_raster_axis": False,
    "_sit_and_stare": False,
}


class _RasterSequenceCelestialTransform(BaseVaryingCelestialTransform):
    """
    Celestial transform with separate raster-step and raster-scan lookup axes.
    """

    n_inputs = 4

    @property
    def inverse(self):
        return _InverseRasterSequenceCelestialTransform(
            crpix_table=self.crpix_table,
            cdelt=self.cdelt,
            lon_pole=self.lon_pole,
            pc_table=self.pc_table,
            crval_table=self.crval_table,
            projection=self.projection,
        )


class _InverseRasterSequenceCelestialTransform(BaseVaryingCelestialTransform):
    """
    Inverse of `_RasterSequenceCelestialTransform`.
    """

    n_inputs = 4
    _is_inverse = True


def _safe_slice_wcs(wcs, item, context):
    """
    Slice a WCS, logging debug on failure instead of raising.
    """
    try:
        if hasattr(wcs, "slice"):
            return wcs.slice(item, numpy_order=True)
        llwcs = wcs.low_level_wcs if hasattr(wcs, "low_level_wcs") else wcs
        item = sanitize_slices(item, len(llwcs.pixel_shape))
        return HighLevelWCSWrapper(SlicedLowLevelWCS(llwcs, item))
    except (IndexError, NotImplementedError, TypeError, ValueError) as e:
        logger.debug(f"Unable to slice {context} with item {item!r}: {e}")
        return None


def _spectrogram_cube_metadata_kwargs_for_copy(cube):
    """
    Return `to_nddata` kwargs that preserve SpectrogramCube-specific metadata.
    """
    return {attr: "copy" for attr in _SPECTROGRAM_CUBE_METADATA_KWARGS if hasattr(cube, attr)}


class _SpectrogramCubeWCSMixin:
    """
    Mixin that handles ``fits_wcs`` and raster-metadata slicing for `SpectrogramCube`.
    """

    def _normalize_fits_wcs_item(self, item):
        """
        Normalize index to tuple of length ndim with only int/slice entries.
        """
        try:
            normalized = tuple(sanitize_slices(item, self.data.ndim))
        except (IndexError, TypeError, ValueError):
            return None
        if not all(isinstance(s, (Integral, slice)) for s in normalized):
            return None
        return normalized

    def _slice_fits_wcs(self, item):  # NOQA: PLR0911
        normalized_item = self._normalize_fits_wcs_item(item)
        if normalized_item is None:
            return None

        if self._fits_wcs is not None:
            sliced = _safe_slice_wcs(self._fits_wcs, normalized_item, "SpectrogramCube fits_wcs")
            if sliced is not None:
                return sliced

        if not self._fits_wcs_segments:
            return None

        if self._separate_raster_axis:
            scan_item = normalized_item[0]
            if isinstance(scan_item, Integral):
                scan_index = scan_item if scan_item >= 0 else self.shape[0] + scan_item
                segment = self._fits_wcs_segment_for_index(scan_index)
                if segment is not None:
                    return _safe_slice_wcs(
                        segment[2],
                        normalized_item[1:],
                        "SpectrogramCube segment fits_wcs",
                    )
            return None

        scan_item = normalized_item[0]
        if isinstance(scan_item, Integral):
            scan_index = scan_item if scan_item >= 0 else self.shape[0] + scan_item
            segment = self._fits_wcs_segment_for_index(scan_index)
            if segment is not None:
                relative = (scan_index - segment[0], *normalized_item[1:])
                return _safe_slice_wcs(segment[2], relative, "SpectrogramCube segment fits_wcs")
            return None

        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None
        segment = self._fits_wcs_segment_for_slice(scan_start, scan_stop)
        if segment is not None:
            relative = (slice(scan_start - segment[0], scan_stop - segment[0]), *normalized_item[1:])
            return _safe_slice_wcs(segment[2], relative, "SpectrogramCube segment fits_wcs")
        return None

    def _fits_wcs_segment_for_index(self, scan_index):
        for seg_start, seg_stop, seg_wcs in self._fits_wcs_segments:
            if seg_start <= scan_index < seg_stop:
                return seg_start, seg_stop, seg_wcs
        return None

    def _fits_wcs_segment_for_slice(self, scan_start, scan_stop):
        for seg_start, seg_stop, seg_wcs in self._fits_wcs_segments:
            if seg_start <= scan_start and scan_stop <= seg_stop:
                return seg_start, seg_stop, seg_wcs
        return None

    def _slice_fits_wcs_segments_for_slice(self, scan_item):
        """
        Rebuild segment list for a slice; returns None for stepped slices.
        """
        if not self._fits_wcs_segments:
            return None

        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None

        sliced_segments = []
        for seg_start, seg_stop, seg_wcs in self._fits_wcs_segments:
            overlap_start = max(seg_start, scan_start)
            overlap_stop = min(seg_stop, scan_stop)
            if overlap_start >= overlap_stop:
                continue
            relative = (
                slice(overlap_start - seg_start, overlap_stop - seg_start),
                slice(None),
                slice(None),
            )
            overlap_wcs = _safe_slice_wcs(seg_wcs, relative, "SpectrogramCube segment fits_wcs")
            if overlap_wcs is None:
                return None
            sliced_segments.append((overlap_start - scan_start, overlap_stop - scan_start, overlap_wcs))
        return sliced_segments or None

    def _slice_raster_boundaries_for_slice(self, scan_item):
        if not self._raster_boundaries:
            return None

        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None

        boundaries = []
        for boundary_start, boundary_stop in self._raster_boundaries:
            overlap_start = max(boundary_start, scan_start)
            overlap_stop = min(boundary_stop, scan_stop)
            if overlap_start < overlap_stop:
                boundaries.append((overlap_start - scan_start, overlap_stop - scan_start))
        return boundaries or None

    def _slice_raster_wcs_header_for_step_slice(self, step_item, step_axis_length):
        header = getattr(self, "_raster_wcs_header", None)
        if header is None:
            return None
        if isinstance(step_item, Integral):
            return header

        step_start, step_stop, step = step_item.indices(step_axis_length)
        sliced_header = copy(header)
        sliced_header["CRPIX3"] = (sliced_header["CRPIX3"] - 1 - step_start) / step + 1
        sliced_header["CDELT3"] *= step
        if "NAXIS3" in sliced_header:
            sliced_header["NAXIS3"] = len(range(step_start, step_stop, step))
        return sliced_header

    def _slice_raster_metadata(self, item, sliced_self):
        normalized_item = self._normalize_fits_wcs_item(item)
        if normalized_item is None:
            sliced_self._fits_wcs_segments = None
            sliced_self._raster_boundaries = None
            sliced_self._separate_raster_axis = False
            sliced_self._sit_and_stare = False
            return

        scan_item = normalized_item[0]

        sliced_self._raster_observer = getattr(self, "_raster_observer", None)
        sliced_self._memmap = self._memmap
        sliced_self._separate_raster_axis = getattr(self, "_separate_raster_axis", False)
        sliced_self._sit_and_stare = getattr(self, "_sit_and_stare", False)

        if self._separate_raster_axis:
            step_item = normalized_item[1]
            sliced_self._raster_wcs_header = self._slice_raster_wcs_header_for_step_slice(step_item, self.shape[1])
            for attr in ("_raster_pc_table", "_raster_crval_table"):
                value = getattr(self, attr, None)
                if value is not None:
                    setattr(sliced_self, attr, value[scan_item, step_item])

            if isinstance(step_item, Integral):
                sliced_self._separate_raster_axis = False
                sliced_self._fits_wcs_segments = None
                sliced_self._raster_boundaries = None
                return

            if isinstance(scan_item, Integral):
                sliced_self._separate_raster_axis = False
                sliced_self._fits_wcs_segments = None
                sliced_self._raster_boundaries = [(0, sliced_self.shape[0])] if sliced_self.data.ndim >= 3 else None
                return

            scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
            if scan_step != 1 or scan_start >= scan_stop:
                sliced_self._fits_wcs_segments = None
                sliced_self._raster_boundaries = None
                return
            if self._fits_wcs_segments:
                sliced_self._fits_wcs_segments = [
                    (start - scan_start, stop - scan_start, seg_wcs)
                    for start, stop, seg_wcs in self._fits_wcs_segments
                    if scan_start <= start and stop <= scan_stop
                ]
            sliced_self._raster_boundaries = [(index, index + 1) for index in range(sliced_self.shape[0])]
            return

        if isinstance(scan_item, Integral):
            sliced_self._fits_wcs_segments = None
            sliced_self._raster_boundaries = None
            return

        sliced_self._raster_wcs_header = self._slice_raster_wcs_header_for_step_slice(scan_item, self.shape[0])
        for attr in ("_raster_pc_table", "_raster_crval_table"):
            value = getattr(self, attr, None)
            if value is not None:
                setattr(sliced_self, attr, value[scan_item])

        sliced_self._fits_wcs_segments = self._slice_fits_wcs_segments_for_slice(scan_item)
        sliced_self._raster_boundaries = self._slice_raster_boundaries_for_slice(scan_item)


def _raster_wcs_bad_row_mask(pc, crval, *, pc_only=False):
    """
    Return AUX rows whose PC or CRVAL table entries are unusable.
    """
    pc_values = pc.to_value(u.pix) if hasattr(pc, "to_value") else np.asarray(pc)
    crval_values = crval.to_value(u.arcsec) if hasattr(crval, "to_value") else np.asarray(crval)
    pc_bad = np.isclose(pc_values, 0).all(axis=(1, 2))
    if pc_only:
        return pc_bad
    crval_bad = np.isclose(crval_values, 0).all(axis=1)
    # Require BOTH pc and crval to be all-zero: crval=(0,0) alone is valid for
    # disk-centre pointings.  A truly unfilled row will have an all-zero PC matrix.
    return pc_bad & crval_bad


def _interpolate_wcs_bad_rows(pc, crval, bad_rows):
    """
    Fill bad rows by linear interpolation between nearest good neighbours.
    """
    _interpolate_bad_axis0_rows(pc, bad_rows)
    _interpolate_bad_axis0_rows(crval, bad_rows)
    return pc, crval


def _apply_wcs_fallback(pc, crval, fallback_pc, fallback_crval):
    """
    Replace every row with the fallback mean values.
    """
    warnings.warn(
        "All steps in this file have bad WCS tables. Using fallback values from observation mean.",
        UserWarning,
        stacklevel=3,
    )
    for i in range(pc.shape[0]):
        pc[i] = fallback_pc
        crval[i] = fallback_crval
    return pc, crval


def _sanitize_raster_wcs_tables(pc, crval, fallback_pc=None, fallback_crval=None, *, bad_rows=None):
    """
    Replace all-zero PC/crval rows via neighbour interpolation or fallback.
    """
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

    if not np.any(~bad_rows):
        if fallback_pc is None or fallback_crval is None:
            msg = "All WCS table rows are bad and no fallback values are available."
            raise ValueError(msg)
        return _apply_wcs_fallback(pc, crval, fallback_pc, fallback_crval)

    return _interpolate_wcs_bad_rows(pc, crval, bad_rows)


def _normalize_spectral_axis_units(header):
    """
    Convert spectral axis values from Angstrom to nm in-place when needed.
    """
    if header.get("CUNIT1", "").lower() == "angstrom":
        header["CUNIT1"] = "nm"
        header["CRVAL1"] *= 0.1
        header["CDELT1"] *= 0.1


def _pc_table_to_lon_lat_order(pc_table, cdelt_lat_lon):
    """
    Convert FITS lat/lon PC matrices to the lon/lat order expected by gWCS.
    """
    pc_values = pc_table.to_value(u.pix) if hasattr(pc_table, "to_value") else np.asarray(pc_table)
    cdelt_lat_lon = np.asarray(cdelt_lat_lon)
    cd_matrix = cdelt_lat_lon[:, np.newaxis] * pc_values
    cd_matrix_lon_lat = cd_matrix[..., [1, 0], :][..., :, [1, 0]]
    cdelt_lon_lat = cdelt_lat_lon[[1, 0]]
    pc_lon_lat = cd_matrix_lon_lat / cdelt_lon_lat[:, np.newaxis]
    return pc_lon_lat * u.pix if hasattr(pc_table, "unit") else pc_lon_lat


def _prepare_raster_wcs_header(header, aux_data, spectral_band, *, sit_and_stare, flip):
    header = copy(header)
    _normalize_spectral_axis_units(header)
    offset_index = AUX_FUV_SLIT_OFFSET_COLUMN if spectral_band == "FUV" else AUX_NUV_SLIT_OFFSET_COLUMN
    header["CRVAL3"] -= aux_data[:, offset_index].mean() * (SLIT_WIDTH.value / 2)
    if sit_and_stare:
        header["CDELT3"] = SIT_AND_STARE_CDELT3_PLACEHOLDER
        dispersion_ratio = header["CDELT3"] / header["CDELT2"]
        angle_1 = aux_data[:, AUX_PC_ANGLE_1_COLUMN].mean()
        angle_2 = aux_data[:, AUX_PC_ANGLE_2_COLUMN].mean()
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


def _as_value_array(value, unit):
    return np.asarray(value.to_value(unit) if hasattr(value, "to_value") else value)


def _validate_raster_wcs_inputs(window_header, pc_all, crval_all, dt_all):
    """
    Validate the cheap table/header invariants needed to build the raster gWCS.
    """
    for key in ("CDELT1", "CRVAL1", "CDELT2", "CDELT3", "CRPIX2", "CRPIX3"):
        try:
            value = float(window_header[key])
        except KeyError as e:
            msg = f"Missing WCS header key {key!r}."
            raise ValueError(msg) from e
        except (TypeError, ValueError) as e:
            msg = f"WCS header key {key!r} must be numeric."
            raise ValueError(msg) from e
        if not np.isfinite(value):
            msg = f"WCS header key {key!r} must be finite."
            raise ValueError(msg)

    pc_values = _as_value_array(pc_all, u.pix)
    crval_values = _as_value_array(crval_all, u.arcsec)
    dt_values = _as_value_array(dt_all, u.s)
    table_shape = pc_values.shape[:-2]

    if pc_values.ndim not in (3, 4) or pc_values.shape[-2:] != (2, 2):
        msg = "Raster PC table must have shape (step, 2, 2) or (scan, step, 2, 2)."
        raise ValueError(msg)
    if crval_values.shape != (*table_shape, 2):
        msg = "Raster CRVAL table shape must match the PC table leading dimensions plus a 2-vector."
        raise ValueError(msg)
    if dt_values.shape != table_shape:
        msg = "Raster time-offset table shape must match the PC table leading dimensions."
        raise ValueError(msg)
    for name, values in (("PC", pc_values), ("CRVAL", crval_values), ("time-offset", dt_values)):
        if not np.all(np.isfinite(values)):
            msg = f"Raster {name} table must contain only finite values."
            raise ValueError(msg)


def _create_raster_gwcs(window_header, pc_all, crval_all, dt_all, t_ref, observer, *, sit_and_stare):
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
    _validate_raster_wcs_inputs(window_header, pc_all, crval_all, dt_all)
    _normalize_spectral_axis_units(window_header)
    spectral_unit = u.Unit(window_header.get("CUNIT1", "nm"))
    cdelt1 = window_header["CDELT1"]
    crval1 = window_header["CRVAL1"]
    crpix1 = window_header.get("CRPIX1", 1.0)
    spectral = m.Linear1D(
        slope=cdelt1 * spectral_unit / u.pix,
        intercept=(crval1 - cdelt1 * (crpix1 - 1)) * spectral_unit,
        name="Wavelength",
    )

    cdelt_lat_lon = np.array([window_header["CDELT2"], window_header["CDELT3"]])
    crpix_values = np.array([window_header["CRPIX3"], window_header["CRPIX2"]])
    crpix_values[0] -= 1
    if not sit_and_stare:
        crpix_values[1] -= 1
    crpix_table = crpix_values * u.pix
    cdelt = cdelt_lat_lon[[1, 0]] * (u.arcsec / u.pix)
    separate_raster_axis = pc_all.ndim == 4
    if separate_raster_axis:
        n_scans, n_steps = pc_all.shape[:2]
        # Pixel inputs are (wavelength, slit, step, scan), so lookup tables use (step, scan).
        pc_table = _pc_table_to_lon_lat_order(np.swapaxes(pc_all, 0, 1), cdelt_lat_lon)
        crval_table = np.swapaxes(crval_all, 0, 1)
        dt_table = np.swapaxes(dt_all, 0, 1)
        celestial = _RasterSequenceCelestialTransform(
            cdelt=cdelt,
            pc_table=pc_table,
            crval_table=crval_table,
            crpix_table=crpix_table,
        )

        temporal = m.Tabular2D(
            points=(np.arange(n_steps) * u.pix, np.arange(n_scans) * u.pix),
            lookup_table=dt_table,
            fill_value=np.nan,
            bounds_error=False,
            method="linear",
            name="Time",
        )
        celestial_mapping = m.Mapping((1, 0, 1, 2), n_inputs=3, name="StepSlitScanMapping")
        celestial_forward = celestial_mapping | celestial
        sky_time = CoupledCompoundModel("&", left=celestial_forward, right=temporal, shared_inputs=2)
        non_spectral = CoupledCompoundModel("&", left=sky_time, right=m.Identity(2, name="step_scan"), shared_inputs=2)
        non_spectral.inverse = (
            m.Mapping(
                (0, 1, 3, 4, 3, 4),
                n_inputs=5,
                name="SelectSkyStepScan",
            )
            | (celestial.inverse & m.Identity(2, name="step_scan"))
            | m.Mapping((1, 2, 3), n_inputs=4, name="SelectSlitStepScan")
        )
    else:
        pc_table = _pc_table_to_lon_lat_order(pc_all, cdelt_lat_lon)
        celestial = VaryingCelestialTransform(
            cdelt=cdelt,
            pc_table=pc_table,
            crval_table=crval_all,
            crpix_table=crpix_table,
        )

        temporal = m.Tabular1D(
            np.arange(pc_all.shape[0]) * u.pix,
            lookup_table=dt_all,
            fill_value=np.nan,
            bounds_error=False,
            method="linear",
            name="Time",
        )
        slit_step_mapping = AsymmetricMapping([1, 0, 1, 1], [1, 0], name="StepSlitMapping")
        non_spectral_rhs = CoupledCompoundModel("&", left=celestial, right=temporal, shared_inputs=1) & m.Identity(
            1, name="step"
        )
        non_spectral = slit_step_mapping | non_spectral_rhs
        non_spectral.inverse = (
            m.Mapping(
                (0, 1, 3, 3),
                n_inputs=4,
                name="SelectSkyAndExplicitStep",
            )
            | (celestial.inverse & m.Identity(1, name="step"))
            | m.Mapping((1, 2), n_inputs=3, name="SelectSlitStep")
        )
    forward_transform = spectral & non_spectral
    forward_transform.inverse = spectral.inverse & non_spectral.inverse

    base_time = parse_time(t_ref)
    spectral_frame = cf.SpectralFrame(
        axes_order=(0,), unit=spectral_unit, name="wavelength", axes_names=("wavelength",)
    )
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
        axes_names=("raster_step",),
        axes_type=("STEP",),
        unit=(u.pix,),
        name="step",
    )
    frames = [spectral_frame, celestial_frame, temporal_frame, step_frame]
    if separate_raster_axis:
        scan_frame = cf.CoordinateFrame(
            naxes=1,
            axes_order=(5,),
            axes_names=("raster_scan",),
            axes_type=("SCAN",),
            unit=(u.pix,),
            name="scan",
        )
        frames.append(scan_frame)
    output_frame = cf.CompositeFrame(frames)
    pixel_axis_names = ["dispersion axis", "spatial along slit", "raster step"]
    if separate_raster_axis:
        pixel_axis_names.append("raster scan")
    pixel_frame = cf.CoordinateFrame(
        naxes=len(pixel_axis_names),
        axes_order=tuple(range(len(pixel_axis_names))),
        axes_names=pixel_axis_names,
        axes_type=["PIXEL"] * len(pixel_axis_names),
        unit=(u.pix,) * len(pixel_axis_names),
    )
    return gwcs.WCS(
        forward_transform,
        input_frame=pixel_frame,
        output_frame=output_frame,
    )
