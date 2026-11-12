import operator
from functools import reduce
from contextlib import suppress, nullcontext

import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs.utils import celestial_frame_to_wcs
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

from sunpy.util import MetaDict

from irispy._wcs import _table_lookup
from irispy.io._level2 import _check_level2
from irispy.io._mask import _check_scaled, _fill_mask_values, _is_scaled, _memmap_fill_mask
from irispy.meta import SJIMeta
from irispy.sji import AIACube, SJICube, SOTCube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import (
    BAD_PIXEL_VALUE_SCALED,
    BAD_PIXEL_VALUES_SCALED,
    DN_UNIT,
    READOUT_NOISE,
    SATURATION_LIMIT,
)

__all__ = ["read_sji_lvl2"]

# SOT unit descriptions that Astropy cannot parse directly.
_SOT_BUNIT = {
    "DN = 64 photoelectrons": "DN",
    "dimensionless ratio": "",
    "unitless number between 0 and 1": "",
}


def _t_obs(hdulist):
    """
    Return the SJI exposure midpoints (T_OBS): ``STARTOBS + AUX TIME + EXPTIMES / 2``.
    """
    frame_count = len(hdulist[0].data)
    auxiliary_count = len(hdulist[1].data)
    if auxiliary_count != frame_count:
        msg = f"Expected {frame_count} SJI auxiliary rows, found {auxiliary_count}"
        raise ValueError(msg)

    return Time(hdulist[0].header["STARTOBS"], format="isot", scale="utc") + (
        (hdulist[1].data[:, hdulist[1].header["TIME"]] + hdulist[1].data[:, hdulist[1].header["EXPTIMES"]] / 2) * u.s
    )


def _fill_dropped_pointing_rows(hdulist) -> None:
    """
    Fill the auxiliary pointing columns of dropped exposures, in place.

    A dropped exposure zeroes its entire pointing row (XCENIX, YCENIX and the PC
    matrix). Only rows where every pointing value is zero are treated as gaps. Gaps are
    filled with the average of the nearest valid neighbor on each side (clamped at the
    ends), so filled values never leave the neighbors' range and an unrotated PC matrix
    stays exactly identity.
    """
    columns = [
        hdulist[1].header["XCENIX"],
        hdulist[1].header["YCENIX"],
        hdulist[1].header["PC1_1IX"],
        hdulist[1].header["PC1_2IX"],
        hdulist[1].header["PC2_1IX"],
        hdulist[1].header["PC2_2IX"],
    ]
    pointing = hdulist[1].data[:, columns]
    dropped_rows = np.flatnonzero(np.all(pointing == 0, axis=1))
    valid_rows = np.flatnonzero(np.any(pointing != 0, axis=1))
    if dropped_rows.size == 0 or valid_rows.size == 0:
        return
    if not hdulist[1].data.flags["W"]:
        hdulist[1].data = hdulist[1].data.copy()
    right = np.clip(np.searchsorted(valid_rows, dropped_rows), 0, valid_rows.size - 1)
    left = np.clip(right - 1, 0, valid_rows.size - 1)
    for column in columns:
        values = hdulist[1].data[:, column]
        values[dropped_rows] = (values[valid_rows[left]] + values[valid_rows[right]]) / 2


def _rows(tables, offset=None):
    """
    The model ``offset[frame] + a * tables[0][frame] + b * tables[1][frame] + ...`` of
    the inputs ``(a, b, ..., frame)``, from lookup tables with a row for each frame.

    It is how a value that is linear in its inputs, with coefficients that change from
    frame to frame, is written with astropy models (``irispy._wcs`` has the operators).
    """
    count = len(tables)
    # The frame is the last input, and ``frame | lookup`` the row of a table at that frame
    frame = m.Mapping((count,), n_inputs=count + 1)
    terms = [m.Mapping((k,), n_inputs=count + 1) * (frame | _table_lookup(table)) for k, table in enumerate(tables)]
    if offset is not None:
        terms.insert(0, frame | _table_lookup(offset))
    return reduce(operator.add, terms)


def _create_gwcs(hdulist: fits.HDUList, t_obs) -> gwcs.WCS:
    """
    The gWCS of a slit-jaw, AIA or SOT cube, from the pixels ``(x, y, frame)`` to the
    helioprojective longitude and latitude and the time.

    Each frame has a FITS WCS of its own: a TAN projection about the pointing of the
    frame (Calabretta & Greisen 2002, A&A 395, 1077). The gWCS gives the sky positions
    of that WCS. The file records the pointing (``XCENIX``, ``YCENIX``) and the ``PC``
    matrix of every frame, so each name below with ``[frame]`` is a lookup table with
    a row for each frame.

    **Pixel to sky**, in three steps:

    1. Where the pixel is on the plane of the projection. A TAN projection draws the
       image on the plane that touches the sky sphere, of radius 1, at the pointing.
       The linear part of the FITS WCS puts a pixel at ``(X, Y)`` on that plane: its
       distances east and north of the pointing (toward greater longitude and
       latitude). ``CDELT`` gives them in arcsec, and in radians they are lengths on
       the plane. FITS scales row i of ``PC`` by ``CDELT`` i::

           (X, Y) = to_plane[frame] @ (pixel - crpix)

           to_plane = | CDELT1 * PC1_1    CDELT1 * PC1_2 |
                      | CDELT2 * PC2_1    CDELT2 * PC2_2 |

    2. Where that point is in space, from the observer at the center of the sphere::

           direction = toward[frame] + X * east[frame] + Y * north[frame]

       ``toward`` is the unit vector from the observer to the pointing, at longitude
       ``lon`` and latitude ``lat``, and ``east`` and ``north`` are the unit vectors
       along the plane. The first axis of the three points to longitude and latitude
       0, the second to longitude 90 degrees and the third to the north pole::

           toward = (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
           east = (-sin(lon), cos(lon), 0)
           north = (-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))

    3. The sky position is the direction to that point, whatever its distance::

           longitude = arctan2(direction[1], direction[0])
           latitude = arctan2(direction[2], hypot(direction[0], direction[1]))

    Steps 1 and 2 together are linear in the pixel, so lookup tables can hold them::

        direction = origin[frame] + x * per_x[frame] + y * per_y[frame]

    ``per_x`` and ``per_y`` are the change of the direction for one pixel along x and
    y, and ``origin`` is the direction of pixel (0, 0). Each has three components, so
    there are nine tables. Between frames, the tables and the time are interpolated.
    astropy's own TAN and rotation models, which the mosaic gWCS uses, take one
    pointing, not one for each frame.

    **Sky to pixel.** The time gives the frame, and the sky position a unit vector
    ``d``, by step 3 backward. The plane of the frame is every point with
    ``point @ toward = 1``, so the line from the observer along ``d`` meets it at
    ``d / (d @ toward)``. Then steps 2 and 1 backward::

        X = (d @ east[frame]) / (d @ toward[frame])
        Y = (d @ north[frame]) / (d @ toward[frame])
        pixel = crpix + inverse(to_plane[frame]) @ (X, Y)
    """
    from sunpy.coordinates.frames import Helioprojective  # NOQA: PLC0415

    # The tables of the file, by frame: (frame, 2, 2) and (frame, 2) in arcsec
    pc_table = hdulist[1].data[:, hdulist[1].header["PC1_1IX"] : hdulist[1].header["PC2_2IX"] + 1].reshape(-1, 2, 2)
    crval_table = hdulist[1].data[:, hdulist[1].header["XCENIX"] : hdulist[1].header["YCENIX"] + 1]
    # FITS CRPIX is 1-based and the transform takes 0-based pixels
    crpix = np.array([hdulist[0].header["CRPIX1"] - 1, hdulist[0].header["CRPIX2"] - 1])
    cdelt = np.array([hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]])
    # Step 2: the unit vectors at the pointing of each frame, each (frame, 3)
    lon, lat = np.deg2rad(crval_table / 3600).T
    toward = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    east_north = np.stack([east, north], axis=1)  # (frame, 2, 3)
    # Step 1: (frame, 2, 2), from arcsec to radians
    to_plane = np.deg2rad(pc_table * cdelt[:, np.newaxis] / 3600)
    # Steps 1 and 2 as one. per_pixel[frame, j] = to_plane[frame, 0, j] * east + to_plane[frame, 1, j] * north is
    # per_x at j = 0 and per_y at j = 1, and (frame, 2, 3) in all.
    per_pixel = np.swapaxes(to_plane, 1, 2) @ east_north
    origin = toward - crpix @ per_pixel
    # Component k of the direction, from its three tables
    direction = [_rows(list(per_pixel[..., k].T / u.pix), origin[:, k] * u.one) for k in range(3)]
    start_time = t_obs[0]
    cadence = (t_obs - start_time).to_value(u.s) * u.s
    temporal = _table_lookup(cadence)
    # (x, y, frame) to (lon, lat, time): each component of the direction takes (x, y, frame) and the time takes
    # the frame. CartesianToSpherical is step 3.
    forward_transform = (
        m.Mapping((0, 1, 2) * 3 + (2,))
        | (direction[0] & direction[1] & direction[2] | CartesianToSpherical(wrap_lon_at=180)) & temporal
    )
    # Sky to pixel. ``to_pixel[frame, k]`` is ``inverse(to_plane)[k, 0] * east + inverse(to_plane)[k, 1] * north``,
    # so ``pixel[k]`` is one division of two sums: ``crpix[k] + (d @ to_pixel[frame, k]) / (d @ toward[frame])``
    to_pixel = np.linalg.inv(to_plane) @ east_north
    pixel = [
        _rows(list(to_pixel[:, k].T * u.pix)) / _rows(list(toward.T * u.one)) | m.Shift(crpix[k] * u.pix)
        for k in range(2)
    ]
    # The times of a one-frame cube do not invert
    with suppress(NotImplementedError):
        # (lon, lat, time) to (d, frame), then each pixel coordinate takes (d, frame) and the frame passes through
        forward_transform.inverse = (
            SphericalToCartesian(wrap_lon_at=180) & temporal.inverse
            | m.Mapping((0, 1, 2, 3, 0, 1, 2, 3, 3))
            | pixel[0] & pixel[1] & m.Identity(1)
        )
    celestial_frame = cf.CelestialFrame(
        axes_order=(0, 1),
        unit=(u.arcsec, u.arcsec),
        reference_frame=Helioprojective(observer="earth", obstime=start_time),
        axis_physical_types=[
            "custom:pos.helioprojective.lon",
            "custom:pos.helioprojective.lat",
        ],
        axes_names=("Longitude", "Latitude"),
    )
    temporal_frame = cf.TemporalFrame(start_time, unit=(u.s,), axes_order=(2,), axes_names=("Time (UTC)",))
    output_frame = cf.CompositeFrame([celestial_frame, temporal_frame])
    input_frame = cf.CoordinateFrame(
        axes_order=(0, 1, 2),
        naxes=3,
        axes_type=["PIXEL", "PIXEL", "PIXEL"],
        unit=(u.pix, u.pix, u.pix),
    )
    return gwcs.WCS(forward_transform, input_frame=input_frame, output_frame=output_frame)


def _create_headers_wcs(hdulist, t_obs):
    """
    This is required as occasionally we need a normal WCS instead of a gWCS due to
    compatibility issues.

    This has been set to have an Earth Observer at the time of the observation.

    However, this only creates the WCS headers, not the full WCS objects. Those are
    created in the SJICube class property fits_wcs, and `irispy.sji.SJICube.to_maps`
    builds its maps from them, so they carry the keys sunpy reads: the instrument and
    telescope that pick the map class and the frame's exposure time.
    """
    from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst  # NOQA: PLC0415
    from sunpy.coordinates.frames import Helioprojective  # NOQA: PLC0415

    aux = {key: hdulist[1].data[:, hdulist[1].header[key]] for key in ("XCENIX", "YCENIX", "EXPTIMES")}
    pc = [hdulist[1].data[:, hdulist[1].header[key]] for key in ("PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX")]
    # Earth is looked up at millisecond precision, as t_obs[i].isot would for each frame.
    earth = get_body_heliographic_stonyhurst("Earth", Time(t_obs.isot))
    pointing = Helioprojective(aux["XCENIX"] * u.arcsec, aux["YCENIX"] * u.arcsec, observer=earth, obstime=t_obs)
    # Build the same SJI header using Astropy's registered solar WCS converter,
    # without importing sunpy.map just to prepare headers at open.
    template = MetaDict(celestial_frame_to_wcs(pointing[0], "TAN").to_header())
    height, width = hdulist[0].data.shape[1:]
    template.update(
        {
            "naxis": 2,
            "naxis1": width,
            "naxis2": height,
            # The reference pixel of the file, where XCENIX/YCENIX point; it need not be the array center
            "crpix1": hdulist[0].header["CRPIX1"],
            "crpix2": hdulist[0].header["CRPIX2"],
            "cunit1": "arcsec",
            "cunit2": "arcsec",
            "cdelt1": hdulist[0].header["CDELT1"],
            "cdelt2": hdulist[0].header["CDELT2"],
            "lonpole": 180.0 if pointing[0].spherical.lat.to_value(u.deg) < 90 else 0.0,
            "instrume": hdulist[0].header["INSTRUME"],
            "telescop": hdulist[0].header["TELESCOP"],
            "twave1": hdulist[0].header["TWAVE1"],
        }
    )
    # Hinode/SOT cubes have no wavelength (TWAVE1 is 0)
    if hdulist[0].header["TWAVE1"]:
        template.update({"wavelnth": int(hdulist[0].header["TWAVE1"]), "waveunit": "Angstrom"})
    per_frame = {
        "crval1": pointing.spherical.lon.to_value(template["cunit1"]),
        "crval2": pointing.spherical.lat.to_value(template["cunit2"]),
        "date-obs": [str(date) for date in t_obs.isot],
        # make_fitswcs_header reads these back from a FITS-WCS header, which keeps 14 significant digits.
        "dsun_obs": [float(f"{value:.14G}") for value in earth.radius.to_value(u.m)],
        "hgln_obs": [float(f"{value:.14G}") for value in earth.lon.to_value(u.deg)],
        "hglt_obs": [float(f"{value:.14G}") for value in earth.lat.to_value(u.deg)],
        "exptime": aux["EXPTIMES"],
        "pc1_1": pc[0],
        "pc1_2": pc[1],
        "pc2_1": pc[2],
        "pc2_2": pc[3],
        "rsun_obs": np.arcsin(pointing.rsun / earth.radius).to_value(u.arcsec),
    }
    wcses = []
    for i in range(hdulist[0].header["NAXIS3"]):
        header = template.copy()
        header.update({key: values[i] for key, values in per_frame.items()})
        wcses.append(header)
    # Object array, so SJICube.fits_wcs can take the frames of any slice.
    headers = np.empty(len(wcses), dtype=object)
    headers[:] = wcses
    return headers


def read_sji_lvl2(filename, *, uncertainty=False, memmap=False):
    """
    Read one SJI, AIA, or Hinode/SOT FITS file supplied with IRIS data.

    Does not handle multiple files nor tar files.

    Parameters
    ----------
    filename : `str`, `pathlib.Path`, file-like or `astropy.io.fits.HDUList`
        File to read. A supplied HDU list is left open, its auxiliary table is edited in
        place (dropped pointing rows are filled) and, with ``memmap=False``, its image is
        copied before the fill values are replaced. Open it with ``do_not_scale_image_data=True``
        when using ``memmap=True``, and without it otherwise.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        IRIS data are the raw integers of the file, in ``DN_IRIS_SJI_UNSCALED``, not the usual
        data numbers (DN); AIA data are stored in DN and keep that unit.
        When ``memmap=True``, missing pixels (the ``-200`` DN fill value and the ``-199`` DN floor
        that lower samples are clipped to, the raw integers ``-32768`` and ``-32764`` in IRIS files)
        retain their original values and are marked in a lazy Dask mask, computed only for the
        slices that are used.
        With ``memmap=False``, missing pixels are marked in the mask and replaced with ``NaN`` for
        floating-point data or ``-200`` for integer data, and IRIS pixels at the level 2 ceiling,
        `irispy.utils.constants.SATURATION_LIMIT`, keep their value and are marked in the mask
        and, if there are any, in ``meta["saturated"]`` (see :ref:`irispy-tutorial-saturated-data`).
        Hinode/SOT data retain their units and ``NaN`` values with either setting.
        Compressed filenames are decompressed into memory once and cannot be memory-mapped.

    Returns
    -------
    `irispy.sji.SJICube`
        The data cube, using a gWCS. AIA and Hinode/SOT cubes are an
        `irispy.sji.AIACube` and an `irispy.sji.SOTCube`.

    Examples
    --------
    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import read_sji_lvl2
    >>> filename = get_test_filepath(
    ...     "sns/iris_l2_20210905_001833_3620258102_SJI_1400_t000_test.fits"
    ... )
    >>> cube = read_sji_lvl2(filename)
    >>> cube.shape, cube.unit
    ((62, 40, 37), Unit("DN_IRIS_SJI_1400"))
    >>> cube.axis_world_coords("time")[0][[0, -1]].isot
    array(['2021-09-05T00:18:47.390', '2021-09-05T05:07:22.210'], dtype='<U23')

    FITS data already in memory are read as an HDU list:

    >>> from pathlib import Path
    >>> from astropy.io import fits
    >>> content = Path(filename).read_bytes()
    >>> read_sji_lvl2(fits.HDUList.fromstring(content)).shape
    (62, 40, 37)
    """
    if isinstance(filename, fits.HDUList):
        context = nullcontext(filename)
    else:
        try:
            context = fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap, decompress_in_memory=True)
        except (OSError, ValueError, TypeError):
            # A directory or a file that is not FITS: raises the level 2 error, chained to this one
            _check_level2(filename, None, "SJI")
    with context as hdulist:
        _check_level2(hdulist.filename() or filename, hdulist[0].header, "SJI")
        # An unquoted NAN, such as CADPL_DV of a two-frame SJI: see read_spectrograph_lvl2
        hdulist[0].verify("silentfix")
        instrume = hdulist[0].header["INSTRUME"]
        is_sot = instrume.startswith("SOT")
        if hdulist[1].data.ndim == 1:
            # A one-frame cube, such as a Hinode/SOT-SP map, stores its auxiliary row as a 1D array
            hdulist[1].data = hdulist[1].data[np.newaxis]
        t_obs = _t_obs(hdulist)
        _fill_dropped_pointing_rows(hdulist)
        extra_coords = [
            (
                "exposure time",
                0,
                hdulist[1].data[:, hdulist[1].header["EXPTIMES"]] * u.s,
            ),
            (
                "obs_vrix",
                0,
                hdulist[1].data[:, hdulist[1].header["OBS_VRIX"]] * u.m / u.s,
            ),
            (
                "ophaseix",
                0,
                hdulist[1].data[:, hdulist[1].header["OPHASEIX"]] * u.one,
            ),
            ("pztx", 0, hdulist[1].data[:, hdulist[1].header["PZTX"]] * u.arcsec),
            ("pzty", 0, hdulist[1].data[:, hdulist[1].header["PZTY"]] * u.arcsec),
            # SLTPX1IX and SLTPX2IX are 1-based FITS pixels (the level 1 CRPIX); the cube's pixels are 0-based
            (
                "slit x position",
                0,
                (hdulist[1].data[:, hdulist[1].header["SLTPX1IX"]] - 1) * u.pix,
            ),
            (
                "slit y position",
                0,
                (hdulist[1].data[:, hdulist[1].header["SLTPX2IX"]] - 1) * u.pix,
            ),
            ("xcenix", 0, hdulist[1].data[:, hdulist[1].header["XCENIX"]] * u.arcsec),
            ("ycenix", 0, hdulist[1].data[:, hdulist[1].header["YCENIX"]] * u.arcsec),
        ]
        data = hdulist[0].data
        data_nan_masked = hdulist[0].data
        out_uncertainty = None
        saturated = None
        scaled = _is_scaled(hdulist[0].header)
        # The passband, e.g. SJI_1400, sets the photons per DN; only the four main passbands have one
        # published (Wülser et al. 2018, Table 1), so the others, such as SJI_5000W, and AIA cubes stay in DN
        passband = hdulist[0].header["TDESC1"]
        unit = DN_UNIT.get(passband, u.DN) if scaled else DN_UNIT["SJI_UNSCALED"]
        # Negative SOT magnetic values are valid measurements; only NaN marks missing pixels.
        if memmap:
            mask = _memmap_fill_mask(data, hdulist[0].header, nan_only=is_sot)
        else:
            _check_scaled(hdulist[0].header)
            # This is a workaround for the AIA cubes being in int and not float
            mask = np.isnan(data) if is_sot else _fill_mask_values(data, *BAD_PIXEL_VALUES_SCALED)
            mask_value = BAD_PIXEL_VALUE_SCALED if np.issubdtype(data.dtype, np.integer) else np.nan
            # A caller's HDU list keeps its image: the fill values are replaced in a copy.
            if isinstance(filename, fits.HDUList) or not data_nan_masked.flags["W"]:
                data_nan_masked = data_nan_masked.copy()
            data_nan_masked[mask] = mask_value
            if instrume in ["IRIS", "SJI"]:
                # Level 2 clips at the ceiling, where iris_prep's +Inf saturation flags were lost
                saturated = data_nan_masked >= SATURATION_LIMIT.value
                mask |= saturated
                if uncertainty:
                    if passband not in READOUT_NOISE:
                        msg = f"No photons per DN is published for {passband}; read it with uncertainty=False"
                        raise ValueError(msg)
                    out_uncertainty = StdDevUncertainty(
                        calculate_uncertainty(data_nan_masked, READOUT_NOISE[passband], unit)
                    )
        cube_class = SJICube if instrume in ["IRIS", "SJI"] else AIACube
        if is_sot:
            # Hinode/SOT cubes are floats without FITS scaling
            cube_class = SOTCube
            bunit = hdulist[0].header.get("BUNIT", "")
            unit = u.Unit(_SOT_BUNIT.get(bunit, bunit), parse_strict="silent")
        meta = SJIMeta(hdulist[0].header)
        meta["frame_wcs_headers"] = _create_headers_wcs(hdulist, t_obs)  # root-relative, not axis-aware
        meta["scaled"] = scaled
        if saturated is not None and saturated.any():
            meta.add("saturated", saturated, "Samples at the level 2 ceiling, also masked", (0, 1, 2))
        map_cube = cube_class(
            data_nan_masked,
            _create_gwcs(hdulist, t_obs),
            uncertainty=out_uncertainty,
            unit=unit,
            meta=meta,
            mask=mask,
        )
        [map_cube.extra_coords.add(*extra_coord) for extra_coord in extra_coords]
    return map_cube
