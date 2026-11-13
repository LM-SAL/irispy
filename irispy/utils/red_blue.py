"""
Red-blue asymmetry utilities for IRIS spectrogram cubes.
"""

from copy import deepcopy

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

import astropy.units as u
from astropy import constants
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS

from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils._spectral import (
    _QualityFlag,
    _to_unit,
    check_velocity_range,
    default_rest_wavelength,
    drop_extra_coords_dependent_on_axis,
    in_chunks,
    make_map_cube,
    make_spatial_template,
    standard_deviation,
)
from irispy.utils.utils import _boolean_mask, _saturated_samples

__all__ = ["RBAQualityFlag", "calculate_red_blue_asymmetry"]

# Minimum fraction of each wing, by weight, whose interpolated values must be finite.
# This is a choice of this module, not a value from the literature: a pixel with less is
# flagged INCOMPLETE_WINGS.
_MIN_WING_COVERAGE = 0.8
# Spatial pixels whose profiles are interpolated at a time: without return_profiles only one
# chunk of interpolated profiles is held. 1000 is about 1.5 times faster per profile than 20000
# on one thread, and below the about 8192 / k profiles from which OpenBLAS solves the spline of
# degree k >= 2 on several threads, for two to three times the CPU time and no faster.
_CHUNK_ROWS = 1000


class RBAQualityFlag(_QualityFlag):
    """
    Quality flags for the per-pixel RBA computation.
    """

    OK = (0, "ok")
    NO_FINITE_DATA = (1, "no finite sample within velocity_range[1] of rest_wavelength")
    PEAK_AT_EDGE = (
        2,
        (
            "peak at the first or last sample within velocity_range[1] of rest_wavelength, "
            "with the profile still rising beyond it"
        ),
    )
    TOO_FEW_POINTS = (3, "too few finite points")
    INTERP_FAILED = (4, "interpolation failed")
    PEAK_IS_ZERO = (5, "peak is zero or non-finite")
    INCOMPLETE_WINGS = (6, "incomplete red or blue wing coverage")
    LOW_SIGNAL = (7, "below min_intensity")
    SATURATED = (8, "saturated")


def _make_velocity_wcs(array_shape, velocity_axis, velocity_grid):
    """
    Return a profile WCS whose only meaningful world axis is Doppler velocity.

    Do not copy the input cube WCS here: profile cubes may have fewer axes than
    the source cube, so stale FITS-WCS axes can warn or fail during construction.
    """
    naxis = len(array_shape)
    wcs_axis = naxis - 1 - velocity_axis
    cdelt = float(np.nanmean(np.diff(velocity_grid))) if velocity_grid.size > 1 else 1.0

    wcs = WCS(naxis=naxis)
    wcs.wcs.ctype[wcs_axis] = "VELO"
    wcs.wcs.cunit[wcs_axis] = "km/s"
    wcs.wcs.crpix[wcs_axis] = 1.0
    wcs.wcs.crval[wcs_axis] = float(velocity_grid[0])
    wcs.wcs.cdelt[wcs_axis] = cdelt
    wcs.array_shape = tuple(array_shape)
    wcs.wcs.set()
    return wcs


def _make_profile_cube(cube, *, data, velocity_grid, wavelength_axis, meta, uncertainty=None, mask=None):
    return SpectrogramCube(
        data,
        wcs=_make_velocity_wcs(data.shape, wavelength_axis, velocity_grid),
        uncertainty=uncertainty,
        unit=cube.unit,
        meta=meta,
        mask=mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=False),
    )


def _prepare_data(data, mask, errors, wavelength_axis):
    """
    Data and errors, negatives masked, spectral axis last.
    """
    data = np.asarray(data, dtype=float)
    # Negative samples are left out
    data = np.where((data < 0) | mask, np.nan, data)

    if errors is not None:
        errors = np.where(mask, np.nan, errors)

    data = np.moveaxis(data, wavelength_axis, -1)
    if errors is not None:
        errors = np.moveaxis(errors, wavelength_axis, -1)
    return data, errors


@u.quantity_input
def calculate_red_blue_asymmetry(
    cube,
    *,
    rest_wavelength: u.AA = None,
    velocity_range: u.km / u.s = (50, 150) * u.km / u.s,
    dv: u.km / u.s = 10 * u.km / u.s,
    spline_degree=3,
    min_intensity=None,
    return_profiles=True,
):
    """
    Calculate red-blue asymmetry maps from a spectrogram cube.

    The asymmetry is computed for each spatial pixel as
    :math:`(I_R - I_B) / I_p`, where ``I_R`` and ``I_B`` are mean intensities
    in matching red and blue velocity ranges and ``I_p`` is the interpolated
    peak intensity (see Notes).

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        Input spectrogram cube.
    rest_wavelength : `astropy.units.Quantity`, optional
        Rest wavelength used to convert the spectral axis to Doppler velocity. Defaults to
        ``cube.meta.rest_wavelength``, or to the one documented line the cube covers
        (`~irispy.utils.lines.get_lines`) if it does not cover ``cube.meta.rest_wavelength``;
        a blend of lines within 0.1 Å of each other counts as its strongest line.
        Required if there is neither.
    velocity_range : `astropy.units.Quantity`, optional
        Two positive velocities. The red wing spans these velocities and the blue wing their
        negatives, both measured from the peak of each profile, which is looked for no farther
        than the upper velocity from ``rest_wavelength`` (see Notes).
    dv : `astropy.units.Quantity`, optional
        Velocity spacing of the interpolated profile, which is sampled at multiples of ``dv``
        from the peak of each profile.
    spline_degree : `int`, optional
        Spline degree for `scipy.interpolate.make_interp_spline`.
    min_intensity : `astropy.units.Quantity`, optional
        Minimum intensity of the peak sample (see Notes) required for a pixel to be processed.
        Pixels below this threshold are assigned quality flag
        `~irispy.utils.red_blue.RBAQualityFlag.LOW_SIGNAL`. ``u.DN`` stands for the data's DN.
    return_profiles : `bool`, optional
        If `True`, include the 3D ``"interpolated_profile"`` cube in the output.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        Always contains ``"red_blue_asymmetry"`` and ``"quality"`` 2D maps. The asymmetry is NaN and
        masked wherever ``"quality"`` is not `~irispy.utils.red_blue.RBAQualityFlag.OK`.
        When the input cube has uncertainty, the ``"red_blue_asymmetry"`` map carries
        its error as a `~astropy.nddata.StdDevUncertainty`.
        The ``"interpolated_profile"`` 3D cube is added when ``return_profiles=True``;
        its velocity axis is measured from the peak of each profile. Slicing the
        collection slices the spatial axes of every cube.

    Raises
    ------
    ValueError
        If no sample lies within ``velocity_range[1]`` of ``rest_wavelength``, as when the cube
        does not cover it.

    Notes
    -----
    Red-blue asymmetries were used to find the upflows of the transition region and corona by
    :cite:t:`depontieu2009` and :cite:t:`tian2011`. This function is not a port of either method.
    These choices are its own, and the papers should be consulted before comparing values with
    theirs: the profile is interpolated with a spline on a regular velocity grid centered on its
    peak, the asymmetry is the difference of the mean intensities of the two wings over the
    interpolated peak intensity, and the default wing range of 50 to 150 km/s and the requirement
    that 80% of each wing is finite have no source.

    The peak sample of a profile is its brightest sample within ``velocity_range[1]`` of
    ``rest_wavelength``. A pixel with no finite sample there is flagged
    `~irispy.utils.red_blue.RBAQualityFlag.NO_FINITE_DATA`. A peak sample that is the first or last
    there is measured when the next sample beyond it is lower, and flagged
    `~irispy.utils.red_blue.RBAQualityFlag.PEAK_AT_EDGE` when that sample is as bright or brighter,
    missing or past the end of the window, as the profile may still rise there. The peak is the
    vertex of the parabola through the peak sample and the samples on either side, or the peak
    sample when one of those is missing. The spline goes through all the samples of the profile,
    and ``I_p`` is the interpolated profile at the peak. ``I_R`` and ``I_B`` are its means over
    ``velocity_range`` either side of the peak, each of its velocities weighted by how much of the
    ``dv``-wide band centered on it lies in the range, so that they change little with ``dv`` for
    a range several ``dv`` wide; use a smaller ``dv`` for a narrower range.

    Pixels with a saturated sample (see :ref:`irispy-tutorial-saturated-data`) within
    ``velocity_range[1]`` of ``rest_wavelength``, where the peak is looked for, or within
    ``velocity_range[1]`` of the peak plus ``dv`` or the cube's sample spacing, whichever is larger,
    are flagged `~irispy.utils.red_blue.RBAQualityFlag.SATURATED`.
    Subtract a background first with `~irispy.utils.spectrograph.subtract_background` if the line
    sits on one.

    For the asymmetry as a function of velocity, as :cite:t:`tian2011` show it
    (:cite:t:`depontieu2009` do not divide), divide the difference of the wings of each
    interpolated profile, in 20 km/s bands, by its value at the peak:

    .. code-block:: python

        rba = calculate_red_blue_asymmetry(cube, velocity_range=(0, 160) * u.km / u.s)
        ok = rba["quality"].data == RBAQualityFlag.OK
        profiles = rba["interpolated_profile"].data[ok]
        velocity = (
            rba["interpolated_profile"].axis_world_coords(-1)[0].to_value(u.km / u.s)
        )
        peak = profiles[:, np.isclose(velocity, 0)]


        def wing(low, high):
            # Each velocity weighted by the overlap of the 10 km/s about it with the band
            overlap = np.clip(
                np.minimum(velocity + 5, high) - np.maximum(velocity - 5, low), 0, None
            )
            inside = overlap > 0
            return np.average(
                profiles[:, inside], axis=-1, weights=overlap[inside], keepdims=True
            )


        rb_v = np.hstack(
            [(wing(v, v + 20) - wing(-v - 20, -v)) / peak for v in range(0, 150, 10)]
        )

    ``rb_v`` has a row for each pixel whose ``"quality"`` is OK and a column for each band, centered
    on ``v + 10`` km/s; a band with a missing velocity is NaN. Calling this function for each band
    instead takes a peak for each band, within its upper velocity of ``rest_wavelength``.
    """
    if min_intensity is not None:
        min_intensity = _to_unit(min_intensity, cube.unit, "min_intensity")
    if rest_wavelength is None:
        rest_wavelength = default_rest_wavelength(cube)
        if rest_wavelength is None:
            msg = (
                "rest_wavelength was not provided and could not be read from cube.meta. "
                "Pass rest_wavelength explicitly or provide a cube with SGMeta containing TWAVE keywords."
            )
            raise ValueError(msg)
    rest_wavelength = rest_wavelength.to(u.nm)
    if rest_wavelength.shape != () or not np.isfinite(rest_wavelength.value) or rest_wavelength <= 0 * u.nm:
        msg = "rest_wavelength must be a positive finite scalar wavelength"
        raise ValueError(msg)

    check_velocity_range(velocity_range)
    if velocity_range[0] < 0:
        msg = f"velocity_range must be positive, not {velocity_range}"
        raise ValueError(msg)

    dv = dv.to_value(u.km / u.s)
    if dv <= 0:
        msg = "dv must be positive"
        raise ValueError(msg)
    spline_degree = int(spline_degree)
    if spline_degree < 0:
        msg = "spline_degree must be a non-negative integer"
        raise ValueError(msg)

    wavelength_axis = cube.wavelength_axis
    wavelengths = cube.axis_world_coords(wavelength_axis)[0].to(u.nm)
    rest_wavelength = rest_wavelength.to(wavelengths.unit)
    velocity = ((wavelengths - rest_wavelength) / rest_wavelength * constants.c).to_value(u.km / u.s)
    velocity_range_kms = tuple(velocity_range.to_value(u.km / u.s))
    low, high = velocity_range_kms
    # Multiples of dv from the peak, out to at least velocity_range[1] + dv: the grid holds the peak,
    # which I_p is read at, and its wings are mirror images
    steps = np.ceil(high / dv) + 1
    interp_velocity = dv * np.arange(-steps, steps + 1)
    peak_bin = interp_velocity.size // 2
    # Each velocity stands for the dv about it, and a wing weights it by the overlap with
    # velocity_range, so that the wing is the mean over the range whatever dv
    red_weight = np.clip(
        np.minimum(interp_velocity + dv / 2, high) - np.maximum(interp_velocity - dv / 2, low), 0, None
    )
    blue_weight = red_weight[::-1]
    # The samples the peak is looked for in, so that another line, the end of the window or a
    # spike brighter than the line is not taken for it
    searched = np.flatnonzero(np.abs(velocity) <= high)
    if not searched.size:
        covered = np.sort(wavelengths.to_value(u.AA)[[0, -1]])
        msg = (
            f"No wavelength of the cube ({covered[0]:.2f} to {covered[1]:.2f} Å) lies within "
            f"velocity_range[1]={high} km/s of rest_wavelength={rest_wavelength.to_value(u.AA):.3f} Å"
        )
        raise ValueError(msg)
    first, last = searched[0], searched[-1]
    in_range = slice(first, last + 1)

    def asymmetry(data, mask, clipped, sigma, *, axis):
        """
        The asymmetry of each spectrum of ``data``, with its error, flag and
        interpolated profile.
        """
        data, errors = _prepare_data(data, mask, sigma, axis)
        output_shape = data.shape[:-1]
        # One row per spatial pixel
        profiles = data.reshape(-1, data.shape[-1])
        profile_errors = None if errors is None else errors.reshape(profiles.shape)

        quality = np.full(len(profiles), RBAQualityFlag.OK, dtype=np.uint8)
        # The largest finite sample, NaN for none. np.nanmax warns of that, and warnings.catch_warnings is not safe
        # in the threads this runs in.
        raw_peak = np.fmax.reduce(profiles[:, in_range], axis=-1)
        if min_intensity is not None:
            quality[raw_peak < min_intensity.value] = RBAQualityFlag.LOW_SIGNAL

        min_points = spline_degree + 1
        finite = np.isfinite(profiles)
        finite_velocity = np.isfinite(velocity)
        todo = quality == RBAQualityFlag.OK
        quality[todo & ~finite[:, in_range].any(axis=-1)] = RBAQualityFlag.NO_FINITE_DATA
        todo &= quality == RBAQualityFlag.OK
        peak_index = first + np.argmax(np.where(finite[:, in_range], profiles[:, in_range], -np.inf), axis=-1)
        # The peak sample and its neighbors; at an end of the window, the peak sample stands in for the
        # missing neighbor
        around = np.clip(peak_index[:, None] + [-1, 0, 1], 0, profiles.shape[1] - 1)
        below, top, above = np.take_along_axis(profiles, around, axis=-1).T
        rising = ((peak_index == first) & ~(below < top)) | ((peak_index == last) & ~(above < top))
        quality[todo & rising] = RBAQualityFlag.PEAK_AT_EDGE
        todo &= quality == RBAQualityFlag.OK
        # The spline goes through all the finite samples at finite velocities
        points = finite & finite_velocity
        quality[todo & (points.sum(axis=-1) < min_points)] = RBAQualityFlag.TOO_FEW_POINTS
        todo &= quality == RBAQualityFlag.OK
        # Each profile is centered on the vertex of the parabola through its peak sample and the two
        # neighbors, or on the peak sample when a neighbor is missing. Centered on the peak sample, a
        # symmetric line between two samples had an asymmetry as large as real ones.
        # A flagged peak sample at an end of the search range can have three samples in line
        with np.errstate(divide="ignore", invalid="ignore"):
            offset = 0.5 * (below - above) / (below - 2 * top + above)
        center = velocity[peak_index] + np.nan_to_num(offset * (velocity[around[:, 2]] - velocity[around[:, 0]]) / 2)

        # The samples the peak is looked for in, as a masked saturated peak leaves a fainter one to be
        # found, and those the wings and I_p are interpolated from, with dv or a sample to spare, as the
        # spline at the end of a wing depends on the next sample, measured from the peak sample where a
        # flag leaves the center meaningless. A saturated line elsewhere in the window, as C II 1334 is
        # for C II 1336, does not flag the pixel.
        saturated = np.moveaxis(clipped, axis, -1).reshape(profiles.shape)
        rows = np.flatnonzero(saturated.any(axis=-1))
        anchor = np.where(todo, center, velocity[peak_index])
        near = np.abs(velocity - anchor[rows, None]) <= high + max(dv, np.abs(np.diff(velocity)).max(initial=0))
        near[:, in_range] = True
        quality[rows[(saturated[rows] & near).any(axis=-1)]] = RBAQualityFlag.SATURATED
        todo &= quality == RBAQualityFlag.OK

        def interpolate(x, y):
            return make_interp_spline(x, y, k=spline_degree, axis=-1)(interp_velocity, extrapolate=False)

        def interpolate_rows(x, y, center):
            """
            The splines through the rows of ``y`` at ``x``, each at ``interp_velocity``
            from its own ``center``.
            """
            t, c, k = make_interp_spline(x, y, k=spline_degree, axis=-1).tck
            at = center[:, None] + interp_velocity
            inside = (at >= x[0]) & (at <= x[-1])
            # The k + 1 B-splines that are nonzero at each velocity, to weight with the coefficients of its row
            basis = BSpline.design_matrix(np.where(inside, at, x[0]).ravel(), t, k)
            shape = (*at.shape, k + 1)
            terms = basis.data.reshape(shape) * c[basis.indices.reshape(shape), np.arange(len(at))[:, None, None]]
            return np.where(inside, terms.sum(axis=-1), np.nan)

        def asymmetries(profiles, profile_errors, center, points, todo, quality):
            """
            The asymmetries and errors of a chunk of rows, flagging ``quality`` in
            place, with their interpolated profiles and errors.
            """
            interpolated = np.full((len(profiles), interp_velocity.size), np.nan)
            interpolated_errors = None if profile_errors is None else np.full_like(interpolated, np.nan)
            # Profiles without gaps share their velocity grid, so one spline fits all of them;
            # a gap leaves a profile its own grid, and its own spline.
            complete = todo & points.all(axis=-1)
            if profile_errors is not None:
                complete &= np.isfinite(profile_errors).all(axis=-1)
            order = np.argsort(velocity)
            rows = np.flatnonzero(complete)
            try:
                interpolated[rows] = interpolate_rows(velocity[order], profiles[rows][:, order], center[rows])
                if profile_errors is not None:
                    interpolated_errors[rows] = interpolate_rows(
                        velocity[order], profile_errors[rows][:, order], center[rows]
                    )
            except ValueError:
                quality[rows] = RBAQualityFlag.INTERP_FAILED
                interpolated[rows] = np.nan
            for row in np.flatnonzero(todo & ~complete):
                kept = points[row]
                shifted_velocity = velocity[kept] - center[row]
                order = np.argsort(shifted_velocity)
                x = shifted_velocity[order]
                samples = profiles[row][kept][order]
                row_errors = None if profile_errors is None else profile_errors[row][kept][order]
                finite_errors = None if row_errors is None else np.isfinite(row_errors)
                try:
                    if row_errors is not None and finite_errors.all():
                        # Errors on the same grid: one spline through both
                        interpolated[row], interpolated_errors[row] = interpolate(x, [samples, row_errors])
                    else:
                        interpolated[row] = interpolate(x, samples)
                        if row_errors is not None and finite_errors.sum() >= min_points:
                            interpolated_errors[row] = interpolate(x[finite_errors], row_errors[finite_errors])
                except ValueError:
                    quality[row] = RBAQualityFlag.INTERP_FAILED
                    interpolated[row] = np.nan
            if interpolated_errors is not None:
                interpolated_errors = np.where(interpolated_errors >= 0, interpolated_errors, np.nan)

            finite_interpolated = np.isfinite(interpolated)
            incomplete = ((finite_interpolated * red_weight).sum(axis=-1) < _MIN_WING_COVERAGE * red_weight.sum()) | (
                (finite_interpolated * blue_weight).sum(axis=-1) < _MIN_WING_COVERAGE * blue_weight.sum()
            )
            quality[(quality == RBAQualityFlag.OK) & incomplete] = RBAQualityFlag.INCOMPLETE_WINGS
            rows = np.flatnonzero(quality == RBAQualityFlag.OK)  # with finite samples in both wings
            profile = interpolated[rows]
            red_intensity, blue_intensity = (
                np.nansum(profile * weight, axis=-1) / (np.isfinite(profile) * weight).sum(axis=-1)
                for weight in (red_weight, blue_weight)
            )
            # I_p is the profile at its center, not its maximum, which can be a spike or an overshoot of
            # the spline across a gap; SolarSoft's sgf_rbp_1lp.pro centers on the spline's maximum and
            # takes its value
            peak = profile[:, peak_bin]
            zero = ~np.isfinite(peak) | np.isclose(peak, 0)
            quality[rows[zero]] = RBAQualityFlag.PEAK_IS_ZERO
            rows, peak = rows[~zero], peak[~zero]
            numerator = (red_intensity - blue_intensity)[~zero]
            red_blue = np.full(len(profiles), np.nan)
            red_blue[rows] = numerator / peak
            red_blue_error = np.full(len(profiles), np.nan)
            if interpolated_errors is not None:
                error = interpolated_errors[rows]

                red_norm, blue_norm = (
                    (np.isfinite(error) * weight).sum(axis=-1) for weight in (red_weight, blue_weight)
                )
                # Each bin counts by its red less its blue weight, each over its wing's total, so that a
                # bin in both wings, the center bin when velocity_range[0] < dv / 2, cancels
                with np.errstate(divide="ignore", invalid="ignore"):
                    weight = red_weight / red_norm[:, None] - blue_weight / blue_norm[:, None]
                    numerator_error = np.sqrt(np.nansum((weight * error) ** 2, axis=-1))
                numerator_error[(red_norm == 0) | (blue_norm == 0)] = np.nan
                peak_error = error[:, peak_bin]
                numerator_is_zero = np.isclose(numerator, 0)
                valid = np.isfinite(numerator_error) & (numerator_is_zero | np.isfinite(peak_error))
                variance = (numerator_error / peak) ** 2
                variance = variance + np.where(numerator_is_zero, 0, (numerator * peak_error / peak**2) ** 2)
                red_blue_error[rows[valid]] = np.sqrt(variance[valid])
            return red_blue, red_blue_error, interpolated, interpolated_errors

        red_blue = np.full(len(profiles), np.nan)
        red_blue_error = np.full(len(profiles), np.nan)
        interpolated = interpolated_errors = None
        if return_profiles:
            interpolated = np.full((len(profiles), interp_velocity.size), np.nan)
            interpolated_errors = None if profile_errors is None else np.full_like(interpolated, np.nan)
        for start in range(0, len(profiles), _CHUNK_ROWS):
            chunk = slice(start, start + _CHUNK_ROWS)
            red_blue[chunk], red_blue_error[chunk], chunk_profiles, chunk_errors = asymmetries(
                profiles[chunk],
                None if profile_errors is None else profile_errors[chunk],
                center[chunk],
                points[chunk],
                todo[chunk],
                quality[chunk],
            )
            if return_profiles:
                interpolated[chunk] = chunk_profiles
                if interpolated_errors is not None:
                    interpolated_errors[chunk] = chunk_errors

        return {
            "red_blue": red_blue.reshape(output_shape),
            "error": red_blue_error.reshape(output_shape),
            "quality": quality.reshape(output_shape),
            "profiles": None if interpolated is None else interpolated.reshape(*output_shape, -1),
            "profile_errors": None if interpolated_errors is None else interpolated_errors.reshape(*output_shape, -1),
        }

    sigma = standard_deviation(cube)
    maps = in_chunks(asymmetry, cube.data, _boolean_mask(cube), _saturated_samples(cube), sigma, axis=wavelength_axis)
    red_blue, red_blue_error, quality = maps["red_blue"], maps["error"], maps["quality"]
    interpolated, interpolated_errors = maps["profiles"], maps["profile_errors"]
    output_shape = red_blue.shape

    meta = {
        "rba_rest_wavelength": rest_wavelength.to_value(u.nm),
        "rba_rest_wavelength_unit": "nm",
        "rba_velocity_range": velocity_range_kms,
        "rba_dv": dv,
        "rba_spline_degree": spline_degree,
    }

    template = make_spatial_template(cube, wavelength_axis)
    uncertainty = None if sigma is None else StdDevUncertainty(red_blue_error)
    result_cubes = [
        (
            "red_blue_asymmetry",
            make_map_cube(template, red_blue, u.dimensionless_unscaled, mask_invalid=True, uncertainty=uncertainty),
        ),
        ("quality", make_map_cube(template, quality, u.dimensionless_unscaled)),
    ]
    for _, map_cube in result_cubes:
        map_cube.meta.update(meta)

    if return_profiles:
        interpolated_data = np.moveaxis(interpolated, -1, wavelength_axis)
        interpolated_uncertainty = (
            StdDevUncertainty(np.moveaxis(interpolated_errors, -1, wavelength_axis))
            if interpolated_errors is not None
            else None
        )
        result_cubes.append(
            (
                "interpolated_profile",
                _make_profile_cube(
                    cube,
                    data=interpolated_data,
                    velocity_grid=interp_velocity,
                    wavelength_axis=wavelength_axis,
                    meta=deepcopy({**cube.meta, **meta, "rba_profile": "interpolated_peak_centered"}),
                    uncertainty=interpolated_uncertainty,
                    mask=~np.isfinite(interpolated_data),
                ),
            )
        )

    map_axes = tuple(range(len(output_shape)))
    profile_axes = tuple(axis for axis in range(cube.data.ndim) if axis != wavelength_axis)
    return RasterCollection(result_cubes, aligned_axes=(map_axes,) * 2 + (profile_axes,) * (len(result_cubes) - 2))
