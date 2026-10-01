import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

import sunpy.map
from sunpy.coordinates import Helioprojective

from irispy.io.spectrograph import read_spectrograph_lvl2

AXIS = [
    (
        "custom:pos.helioprojective.lon",
        "custom:pos.helioprojective.lat",
        "time",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:PIXEL",  # slit x position
        "custom:PIXEL",  # slit y position
        "custom:CUSTOM",
        "custom:CUSTOM",
    ),
    ("custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"),
    ("custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"),
]


def test_world_axis_physical_types_sjicube_2832(sns_sjicube_2832):
    assert np.all(sns_sjicube_2832.shape == (10, 40, 37))
    assert sns_sjicube_2832.array_axis_physical_types == AXIS


def test_world_axis_physical_types_sjicube_2796(sns_sjicube_2796):
    assert np.all(sns_sjicube_2796.shape == (62, 40, 37))
    assert sns_sjicube_2796.array_axis_physical_types == AXIS


def test_world_axis_physical_types_sjicube_1400(sns_sjicube_1400):
    assert np.all(sns_sjicube_1400.shape == (62, 40, 37))
    assert sns_sjicube_1400.array_axis_physical_types == AXIS


def test_world_axis_physical_types_sjicube_1330(sns_sjicube_1330):
    assert np.all(sns_sjicube_1330.shape == (52, 40, 37))
    assert sns_sjicube_1330.array_axis_physical_types == AXIS


def test_sji_plotter_does_not_expose_plot_rgb(sns_sjicube_1400):
    assert not hasattr(sns_sjicube_1400.plotter, "plot_rgb")


def test_to_map(sns_sjicube_1330):
    # Basic smoke tests
    output = sns_sjicube_1330.to_maps(0)
    assert isinstance(output, sunpy.map.GenericMap)
    assert output.data.shape == (40, 37)
    assert output.reference_date is not None
    assert output.wavelength == 1330 * u.AA
    assert "1330" in output.name

    output = sns_sjicube_1330.to_maps([0, 2])
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 2)
    assert np.all([output.reference_date is not None for output in output])

    output = sns_sjicube_1330.to_maps(range(1, 3))
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 2)
    assert np.all([output.reference_date is not None for output in output])

    output = sns_sjicube_1330.to_maps(range(0, 12, 4))
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 3)
    assert np.all([output.reference_date is not None for output in output])

    output = sns_sjicube_1330.to_maps()
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 52)
    assert np.all([output.reference_date is not None for output in output])


def test_sji_plot_uses_short_slider_label(sns_sjicube_1330):
    fig = plt.figure()
    animator = sns_sjicube_1330.plot(fig=fig)

    assert animator.slider_labels == ["Time"]
    plt.close(fig)


def test_sji_plot_accepts_custom_slider_label(sns_sjicube_1330):
    fig = plt.figure()
    animator = sns_sjicube_1330.plot(fig=fig, slider_labels=["Frame"])

    assert animator.slider_labels == ["Frame"]
    plt.close(fig)


def test_sji_plot_falls_back_to_viridis_without_an_iris_colormap(sns_sjicube_1330):
    image = sns_sjicube_1330[0]
    image.meta["TWAVE1"] = 1234  # There is no irissji1234 colormap.
    ax = image.plot()
    assert ax.images[0].get_cmap().name == "viridis"
    plt.close(ax.figure)


def test_rebinned_cube_has_no_fits_wcs(sns_sjicube_1330):
    rebinned = sns_sjicube_1330[:, :, :36].rebin((1, 2, 2))

    assert rebinned.fits_wcs is None
    assert rebinned.celestial_frame == sns_sjicube_1330.celestial_frame
    with pytest.raises(ValueError, match="no FITS WCS"):
        rebinned.to_maps(0)


def test_spatial_slice_fits_wcs_describes_sliced_pixels(sns_sjicube_1330):
    sliced = sns_sjicube_1330[:, 10:, 20:].fits_wcs[0].pixel_to_world(0, 0)
    full = sns_sjicube_1330.fits_wcs[0].pixel_to_world(20, 10)
    assert sliced.separation(full) < 1e-6 * u.arcsec


def test_to_maps_negative_index(sns_sjicube_1330):
    last = sns_sjicube_1330.shape[0] - 1
    assert sns_sjicube_1330.to_maps(-1).meta["DATE-OBS"] == sns_sjicube_1330.to_maps(last).meta["DATE-OBS"]
    dates = [m.meta["DATE-OBS"] for m in sns_sjicube_1330.to_maps([-1, 0])]
    assert dates == [sns_sjicube_1330.to_maps(i).meta["DATE-OBS"] for i in (last, 0)]


def test_to_maps_accepts_numpy_integers(sns_sjicube_1330):
    assert sns_sjicube_1330.to_maps(np.int64(2)).meta["DATE-OBS"] == sns_sjicube_1330.to_maps(2).meta["DATE-OBS"]


def test_sji_celestial_frame_is_slicing_invariant_and_keeps_sg_points_in_place(sns_sg_file, sns_sjicube_1400):
    """
    SG and SJI see the Sun from the same place, so an SG point must not move in the SJI
    frame.
    """
    frame = sns_sjicube_1400.celestial_frame
    assert isinstance(frame, Helioprojective)
    assert sns_sjicube_1400[0].celestial_frame == frame

    cube = read_spectrograph_lvl2(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"]
    _, point, _, _ = cube.wcs.pixel_to_world(5, 20, 93)
    moved = point.transform_to(frame)
    assert_quantity_allclose(moved.Tx, point.Tx, atol=0.1 * u.arcsec)
    assert_quantity_allclose(moved.Ty, point.Ty, atol=0.1 * u.arcsec)
