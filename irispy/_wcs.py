from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective


def _celestial_frame_from_cube(cube):
    """
    The `~sunpy.coordinates.frames.Helioprojective` frame of this observation, as
    seen by the IRIS observer.

    This does not depend on slicing: it works on combined multi-file cubes and on
    cubes sliced along the scan, step, slit, time, or wavelength axes.
    """
    observer = getattr(cube.meta, "observer", None)
    if observer is not None:
        return Helioprojective(observer=observer, obstime=observer.obstime)

    fits_wcs = cube.fits_wcs
    if fits_wcs is None:
        segments = getattr(cube, "_fits_wcs_segments", None)
        if segments:
            fits_wcs = segments[0][2]
    if isinstance(fits_wcs, list):
        fits_wcs = fits_wcs[0] if fits_wcs else None
    if fits_wcs is None or not hasattr(fits_wcs, "celestial"):
        msg = "This cube does not carry the WCS metadata needed to derive a celestial frame."
        raise ValueError(msg)
    return wcs_to_celestial_frame(fits_wcs.celestial)
