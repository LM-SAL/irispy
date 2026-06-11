import warnings

from astropy.wcs.utils import wcs_to_celestial_frame

from sunpy.coordinates.frames import Helioprojective


class _FitsWCSCompatMixin:
    """
    Mixin providing the deprecated ``basic_wcs`` alias and the ``celestial_frame``
    convenience shared by `SpectrogramCube` and `SJICube`.
    """

    @property
    def basic_wcs(self):
        """
        Deprecated alias of ``fits_wcs``.
        """
        warnings.warn(
            "'basic_wcs' is deprecated and will be removed in a future release; use 'fits_wcs' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fits_wcs

    @property
    def celestial_frame(self):
        """
        The `~sunpy.coordinates.frames.Helioprojective` frame of this observation, as
        seen by the IRIS observer.

        This does not depend on slicing: it works on combined multi-file
        cubes (where ``fits_wcs`` is `None`) and on cubes sliced along the
        scan, step, slit, time, or wavelength axes.
        """
        observer = getattr(self, "_raster_observer", None)
        if observer is not None:
            return Helioprojective(observer=observer, obstime=observer.obstime)
        fits_wcs = self.fits_wcs
        if fits_wcs is None:
            segments = getattr(self, "_fits_wcs_segments", None)
            if segments:
                fits_wcs = segments[0][2]
        if isinstance(fits_wcs, list):
            fits_wcs = fits_wcs[0] if fits_wcs else None
        if fits_wcs is None or not hasattr(fits_wcs, "celestial"):
            msg = "This cube does not carry the WCS metadata needed to derive a celestial frame."
            raise ValueError(msg)
        return wcs_to_celestial_frame(fits_wcs.celestial)
