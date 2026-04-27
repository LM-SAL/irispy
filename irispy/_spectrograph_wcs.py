from numbers import Integral

import numpy as np

from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

from sunpy import log as logger


def _normalize_tuple_index(item, ndim):
    """
    Normalize a tuple index to explicit per-axis entries.

    Returns
    -------
    list or None
        A normalized list of length ``ndim`` when normalization is valid.
        Returns ``None`` when the tuple contains more than one ellipsis.
    """
    normalized_item = []
    ellipsis_seen = False
    for subitem in item:
        if subitem is Ellipsis:
            if ellipsis_seen:
                return None
            ellipsis_seen = True
            missing_dims = ndim - (len(item) - 1)
            normalized_item.extend([slice(None)] * missing_dims)
        else:
            normalized_item.append(subitem)
    if len(normalized_item) < ndim:
        normalized_item.extend([slice(None)] * (ndim - len(normalized_item)))
    if len(normalized_item) != ndim:
        return None
    return normalized_item


def _safe_slice_wcs(wcs, item, context):
    """Slice a WCS, logging debug on failure instead of raising."""
    try:
        if hasattr(wcs, "slice"):
            return wcs.slice(item, numpy_order=True)
        # Fallback for wcsapi wrappers (e.g. SlicedFITSWCS from astropy ≥7)
        # that lack .slice() but can be sliced via SlicedLowLevelWCS.
        llwcs = wcs.low_level_wcs if hasattr(wcs, "low_level_wcs") else wcs
        item = sanitize_slices(item, len(llwcs.pixel_shape))
        return HighLevelWCSWrapper(SlicedLowLevelWCS(llwcs, item))
    except (IndexError, NotImplementedError, TypeError, ValueError) as e:
        logger.debug(f"Unable to slice {context} with item {item!r}: {e}")
        return None


class _SpectrogramCubeWCSMixin:
    """Mixin that handles ``basic_wcs`` and raster-metadata slicing for `SpectrogramCube`."""

    def _normalize_basic_wcs_item(self, item):
        if isinstance(item, tuple):
            item = _normalize_tuple_index(item, self.data.ndim)
            if item is None or not all(isinstance(subitem, (Integral, slice)) for subitem in item):
                return None
            return tuple(item)
        if isinstance(item, (Integral, slice)):
            return (item, *([slice(None)] * (self.data.ndim - 1)))
        if item is Ellipsis:
            return tuple([slice(None)] * self.data.ndim)
        return None

    def _slice_basic_wcs(self, item):
        normalized_item = self._normalize_basic_wcs_item(item)
        if normalized_item is None:
            return None
        return self._slice_single_basic_wcs(normalized_item) or self._slice_segment_basic_wcs(normalized_item)

    def _slice_single_basic_wcs(self, normalized_item):
        if self._basic_wcs is None:
            return None
        return _safe_slice_wcs(self._basic_wcs, normalized_item, "SpectrogramCube basic_wcs")

    def _slice_segment_basic_wcs(self, normalized_item):
        if not self._basic_wcs_segments:
            return None
        scan_item = normalized_item[0]
        if isinstance(scan_item, Integral):
            scan_index = scan_item if scan_item >= 0 else self.shape[0] + scan_item
            return self._slice_segment_basic_wcs_index(scan_index, normalized_item)
        return self._slice_segment_basic_wcs_slice(scan_item, normalized_item)

    def _slice_segment_basic_wcs_index(self, scan_index, normalized_item):
        for segment_start, segment_stop, segment_wcs in self._basic_wcs_segments:
            if segment_start <= scan_index < segment_stop:
                relative_item = (scan_index - segment_start, *normalized_item[1:])
                return _safe_slice_wcs(segment_wcs, relative_item, "SpectrogramCube segment basic_wcs")
        return None

    def _slice_segment_basic_wcs_slice(self, scan_item, normalized_item):
        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_step != 1 or scan_start >= scan_stop:
            return None
        for segment_start, segment_stop, segment_wcs in self._basic_wcs_segments:
            if segment_start <= scan_start and scan_stop <= segment_stop:
                relative_item = (slice(scan_start - segment_start, scan_stop - segment_start), *normalized_item[1:])
                return _safe_slice_wcs(segment_wcs, relative_item, "SpectrogramCube segment basic_wcs")
        return None

    def _slice_basic_wcs_segments_for_slice(self, scan_item):
        if not self._basic_wcs_segments:
            return None
        scan_start, scan_stop, scan_step = scan_item.indices(self.shape[0])
        if scan_start >= scan_stop:
            return None
        sliced_segments = []
        for segment_start, segment_stop, segment_wcs in self._basic_wcs_segments:
            overlap_start = max(segment_start, scan_start)
            overlap_stop = min(segment_stop, scan_stop)
            if overlap_start >= overlap_stop:
                continue
            if scan_step != 1:
                retained = np.arange(overlap_start, overlap_stop, scan_step)
                if retained.size == 0:
                    continue
                relative_item = (
                    slice(retained[0] - segment_start, retained[-1] - segment_start + 1, scan_step),
                    slice(None),
                    slice(None),
                )
            else:
                relative_item = (
                    slice(overlap_start - segment_start, overlap_stop - segment_start),
                    slice(None),
                    slice(None),
                )
            overlap_wcs = _safe_slice_wcs(segment_wcs, relative_item, "SpectrogramCube segment basic_wcs")
            if overlap_wcs is None:
                return None
            sliced_segments.append((overlap_start - scan_start, overlap_stop - scan_start, overlap_wcs))
        return sliced_segments or None

    def _slice_raster_boundaries_for_slice(self, scan_item):
        if not self._raster_boundaries:
            return None
        scan_start, scan_stop, _ = scan_item.indices(self.shape[0])
        if scan_start >= scan_stop:
            return None
        boundaries = []
        for boundary_start, boundary_stop in self._raster_boundaries:
            overlap_start = max(boundary_start, scan_start)
            overlap_stop = min(boundary_stop, scan_stop)
            if overlap_start < overlap_stop:
                boundaries.append((overlap_start - scan_start, overlap_stop - scan_start))
        return boundaries or None

    def _slice_raster_metadata(self, item, sliced_self):
        normalized_item = self._normalize_basic_wcs_item(item)
        if normalized_item is None:
            sliced_self._basic_wcs_segments = None
            sliced_self._raster_boundaries = None
            return

        scan_item = normalized_item[0]
        for attr in ("_raster_wcs_header", "_raster_observer"):
            if hasattr(self, attr):
                setattr(sliced_self, attr, getattr(self, attr))
        sliced_self._memmap = self._memmap

        if isinstance(scan_item, Integral):
            sliced_self._basic_wcs_segments = None
            sliced_self._raster_boundaries = None
            return

        for attr in ("_raster_pc_table", "_raster_crval_table"):
            value = getattr(self, attr, None)
            if value is not None:
                setattr(sliced_self, attr, value[scan_item])
        sliced_self._basic_wcs_segments = self._slice_basic_wcs_segments_for_slice(scan_item)
        sliced_self._raster_boundaries = self._slice_raster_boundaries_for_slice(scan_item)
