Add `irispy.utils.spectrograph.radiation_temperature`, which converts a radiometrically calibrated cube to radiation temperature in K by inverting the Planck function, accepting radiance per unit wavelength or frequency and propagating the uncertainty.
Unlike astropy's `~astropy.units.brightness_temperature` equivalency, a Rayleigh-Jeans limit, it holds in the ultraviolet.
