import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS

import sunpy.map
from sunpy.coordinates import Helioprojective

from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.utils import read_files
from irispy.sji import SJICube
from irispy.tests.helpers import figure_test
from irispy.utils.bursts import find_bright_image_events
from irispy.utils.constants import RADIANCE_UNIT
from irispy.utils.spectrograph import radiometric_calibration
from irispy.visualization import LAT_AXIS_LABEL, LON_AXIS_LABEL


def test_to_maps_rejects_unscaled_data(bursts_sji_1400_file):
    # The maps give the IRIS DN units as DN, which the raw integers of a memmap read are not
    with pytest.raises(ValueError, match=r"unscaled.*memmap=False"):
        read_files(bursts_sji_1400_file, memmap=True).to_maps(0)
    assert read_files(bursts_sji_1400_file).to_maps(0).unit == u.DN


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


@pytest.mark.parametrize(
    ("cube_fixture", "n_frames"),
    [("sns_sjicube_2832", 10), ("sns_sjicube_2796", 62), ("sns_sjicube_1400", 62), ("sns_sjicube_1330", 52)],
)
def test_world_axis_physical_types_sjicube(request, cube_fixture, n_frames):
    cube = request.getfixturevalue(cube_fixture)
    assert cube.shape == (n_frames, 40, 37)
    assert cube.array_axis_physical_types == AXIS


def test_sjicube_str(sns_sjicube_1330, sot_fg_file):
    """
    The summary names the passband, also of a SOT cube, whose TWAVE1 is 0, gives the
    times of the first and last frames of a slice along them, and the axis types of the
    WCS alone.
    """
    times = sns_sjicube_1330.time
    summary = str(sns_sjicube_1330[:, 5])
    assert "Bandpass:              1330\n" in summary
    assert f"Obs Date:              {times[0]} -- {times[-1]}\n" in summary
    assert "Total Frames" not in summary
    assert f"Axis Types:            {[AXIS[0][:3], AXIS[2]]}\n" in summary
    assert "Bandpass:              G band 4305\n" in str(read_files(sot_fg_file))


def test_sji_frame_plot_clip_interval(sns_sjicube_1400):
    """
    ``clip_interval`` sets the stretch of a single frame, as of each frame of an
    animation.
    """
    frame = sns_sjicube_1400[0]
    ax = frame.plot(clip_interval=(1, 99.5) * u.percent)
    np.testing.assert_allclose(ax.images[0].get_clim(), np.percentile(frame.data[~frame.mask], [1, 99.5]))
    plt.close("all")
    with pytest.raises(ValueError, match="A range of 2 values"):
        frame.plot(clip_interval=(1, 50, 99.5) * u.percent)


def test_to_maps_are_sji_maps_with_the_frame_metadata(sns_sjicube_1330):
    # Maps were built from the WCS alone: GenericMap, no unit, the file-level EXPTIME in every frame
    exposure_times = sns_sjicube_1330.axis_world_coords(wcs=sns_sjicube_1330.extra_coords["exposure time"])[0]
    sji_map = sns_sjicube_1330.to_maps(2)
    assert isinstance(sji_map, sunpy.map.sources.SJIMap)
    assert sji_map.unit == u.DN
    assert sji_map.exposure_time == exposure_times[2]
    assert (sji_map.observatory, sji_map.instrument) == ("IRIS", "SJI")
    assert sji_map.date.utc.isot == sns_sjicube_1330.wcs.pixel_to_world(0, 0, 2)[-1].utc.isot
    assert sji_map.plot_settings["cmap"] == "irissji1330"
    sliced_map = sns_sjicube_1330[2, 10:, 20:].to_maps()
    assert isinstance(sliced_map, sunpy.map.sources.SJIMap)
    assert sliced_map.data.shape == (30, 17)
    assert sliced_map.wcs.pixel_to_world(0, 0).separation(sji_map.wcs.pixel_to_world(20, 10)) < 1e-6 * u.arcsec


def test_to_maps_keep_the_unit_and_mask_of_the_cube(bursts_sjicube_1400):
    # The maps were in DN whatever the cube's unit, and had no mask
    frame = 1
    assert bursts_sjicube_1400.mask[frame].any()  # fill values and the clipped pixels of the flare
    for cube, unit in (
        (bursts_sjicube_1400, u.DN),
        (bursts_sjicube_1400.apply_exposure_time_correction(), u.DN / u.s),
        (radiometric_calibration(bursts_sjicube_1400), RADIANCE_UNIT),
    ):
        for sji_map in (cube.to_maps(frame), cube[frame].to_maps()):
            assert sji_map.unit == unit
            np.testing.assert_array_equal(sji_map.mask, cube.mask[frame])


def test_to_maps_of_a_cube_without_a_mask(bursts_sjicube_1400):
    # The burst labels have no mask
    labels, _ = find_bright_image_events(bursts_sjicube_1400[:3])
    assert labels.to_maps(1).mask is None
    assert labels[1].to_maps().mask is None


def test_aia_cube_and_maps_plot_with_the_sdoaia_color_table(aia_1700_file):
    # The maps asked matplotlib for the unknown aia1700, so plotting them raised, and the cube plotted in viridis
    aia = read_files(aia_1700_file)
    assert aia[0].plot().images[0].get_cmap().name == "sdoaia1700"
    plt.close("all")
    assert aia.to_maps(0).plot().get_cmap().name == "sdoaia1700"


@pytest.mark.parametrize("sot_file", ["sot_fg_file", "sot_mg_file", "sot_sp_file"])
def test_sot_cube_plots_with_the_hinodesotintensity_color_table(sot_file, request):
    # The cubes plotted in viridis, and the maps of the broadband and SP files in gray
    sot = read_files(request.getfixturevalue(sot_file))
    assert sot[0].plot().images[0].get_cmap().name == "hinodesotintensity"
    plt.close("all")
    assert sot.to_maps(0).plot().get_cmap().name == "hinodesotintensity"
    plt.close("all")


def test_to_map(sns_sjicube_1330):
    # Basic smoke tests
    output = sns_sjicube_1330.to_maps(0)
    assert isinstance(output, sunpy.map.GenericMap)
    assert output.data.shape == (40, 37)
    assert output.reference_date is not None
    assert output.wavelength == 1330 * u.AA
    assert "1330" in output.name
    sliced = sns_sjicube_1330[0].to_maps()
    assert sliced.meta == output.meta
    assert sliced.plot_settings["cmap"] == output.plot_settings["cmap"]

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


@pytest.mark.parametrize(("kwargs", "expected"), [({}, ["Time"]), ({"slider_labels": ["Frame"]}, ["Frame"])])
def test_sji_plot_slider_labels(sns_sjicube_1330, kwargs, expected):
    fig = plt.figure()
    animator = sns_sjicube_1330.plot(fig=fig, **kwargs)
    assert animator.slider_labels == expected
    plt.close(fig)


def test_sji_plot_slider_updates_the_frame(sns_sjicube_1330):
    fig = plt.figure()
    animator = sns_sjicube_1330.plot(fig=fig)

    animator.sliders[0]._slider.set_val(3)

    expected = np.ma.masked_array(sns_sjicube_1330.data[3], sns_sjicube_1330.mask[3])
    np.testing.assert_array_equal(animator.im.get_array(), expected)
    # The update resets the WCS axes, so the IRIS labels are applied again
    assert {axis.get_axislabel() for axis in animator.axes.coords} >= {LON_AXIS_LABEL, LAT_AXIS_LABEL}
    plt.close(fig)


@figure_test
def test_sji_frame_figure(sns_sjicube_1400):
    sns_sjicube_1400[0].plot()


def test_sji_image_along_time_labels_its_time(sns_sjicube_1400):
    # Unlike a raster plot, which shows the time only when asked for
    time = sns_sjicube_1400[:, :, 10].plot().coords["time"]
    assert time.get_ticklabel_visible()
    assert time.get_ticklabel_position()
    plt.close("all")


def test_sji_plot_falls_back_to_viridis_without_an_iris_colormap(sns_sjicube_1330):
    image = sns_sjicube_1330[0]
    image.meta["TWAVE1"] = 1234  # There is no irissji1234 colormap.
    ax = image.plot()
    assert ax.images[0].get_cmap().name == "viridis"
    plt.close(ax.figure)


def test_apply_exposure_time_correction(sns_sjicube_1330):
    # The exposure time of a slit-jaw cube is an extra coordinate, which raised in sunraster
    corrected = sns_sjicube_1330.apply_exposure_time_correction()
    exposure_time = sns_sjicube_1330.exposure_time.to_value(u.s)[:, np.newaxis, np.newaxis]
    assert corrected.unit == sns_sjicube_1330.unit / u.s
    np.testing.assert_allclose(corrected.data, sns_sjicube_1330.data / exposure_time)
    undone = corrected.apply_exposure_time_correction(undo=True)
    assert undone.unit == sns_sjicube_1330.unit
    np.testing.assert_allclose(undone.data, sns_sjicube_1330.data)
    frame = sns_sjicube_1330[1]
    assert frame.exposure_time == sns_sjicube_1330.exposure_time[1]
    np.testing.assert_allclose(frame.apply_exposure_time_correction().data, corrected.data[1])


def test_time_of_one_frame(sns_sjicube_1330):
    frame, expected = sns_sjicube_1330[1].time, sns_sjicube_1330.time[1]
    assert (frame.shape, frame.scale, frame.format) == ((), expected.scale, expected.format)
    assert frame == expected


def test_array_index_raises_index_error(sns_sjicube_1330):
    # As ndcube: a list reads as a tuple of indices, an array is rejected
    assert sns_sjicube_1330[[0, 1]].shape == sns_sjicube_1330[0, 1].shape
    for item in (np.array([0, 1]), (slice(None), np.array([0, 1]))):
        with pytest.raises(IndexError, match="only integer or range slices"):
            sns_sjicube_1330[item]


def test_rebinned_cube_has_no_fits_wcs(sns_sjicube_1330):
    rebinned = sns_sjicube_1330[:, :, :36].rebin((1, 2, 2))

    assert rebinned.fits_wcs is None
    assert rebinned.celestial_frame == sns_sjicube_1330.celestial_frame
    with pytest.raises(ValueError, match="no FITS WCS"):
        rebinned.to_maps(0)


@pytest.mark.parametrize("item", [np.s_[5], np.s_[:, 0], np.s_[:, 0, 0]])
def test_celestial_frame_does_not_depend_on_slicing(sns_sjicube_1330, item):
    assert sns_sjicube_1330[item].celestial_frame == sns_sjicube_1330.celestial_frame


def test_spatial_slice_fits_wcs_describes_sliced_pixels(sns_sjicube_1330):
    sliced = sns_sjicube_1330[:, 10:, 20:].fits_wcs[0].pixel_to_world(0, 0)
    full = sns_sjicube_1330.fits_wcs[0].pixel_to_world(20, 10)
    assert sliced.separation(full) < 1e-6 * u.arcsec


def test_integer_spatial_index_gives_the_sliced_fits_wcs(sns_sjicube_1330):
    # No header describes a 1-D image: fits_wcs raised AttributeError, with no map to build
    row = sns_sjicube_1330[:, 0, :]
    assert len(row.fits_wcs) == sns_sjicube_1330.shape[0]
    from_frame, from_cube = row.fits_wcs[3].pixel_to_world(5), row.wcs.pixel_to_world(5, 3)[0]
    assert u.allclose((from_frame.Tx, from_frame.Ty), (from_cube.Tx, from_cube.Ty), atol=1e-6 * u.arcsec)
    assert sns_sjicube_1330[3, 0, :].fits_wcs.pixel_n_dim == 1
    with pytest.raises(ValueError, match="no FITS WCS"):
        row.to_maps()


def test_sliced_frame_headers_keep_the_file_units(sns_sjicube_1330):
    rows, columns = sns_sjicube_1330.data.shape[1:]
    original = sns_sjicube_1330._frame_headers[0]
    sliced = sns_sjicube_1330[:, 2 : rows - 3, 4 : columns - 5]._frame_headers[0]
    assert sliced["cunit1"] == sliced["cunit2"] == original["cunit1"] == "arcsec"
    assert sliced["crpix1"] == pytest.approx(original["crpix1"] - 4)
    assert sliced["crpix2"] == pytest.approx(original["crpix2"] - 2)
    assert (sliced["naxis1"], sliced["naxis2"]) == (columns - 9, rows - 5)
    # to_header writes 14 significant digits
    assert sliced["cdelt1"] == pytest.approx(original["cdelt1"], rel=1e-12)
    assert sliced["crval1"] == pytest.approx(original["crval1"], rel=1e-12)


@pytest.mark.parametrize(
    "file_fixture",
    [
        "sns_sji_1330_file",
        "sns_sji_1400_file",
        "sns_sji_2796_file",
        "sns_sji_2832_file",
        "raster_sji_1400_file",
        "raster_sji_2796_file",
        "raster_sji_2832_file",
        "bursts_sji_1400_file",
    ],
)
def test_to_maps_wcs_matches_the_cube_wcs(request, file_fixture):
    # The frame headers put the pointing at the array center; these files' CRPIX is off it by up to a pixel
    cube = read_files(request.getfixturevalue(file_fixture))
    n_frames, ny, nx = cube.shape
    x = np.array([0, nx - 1, 0, nx - 1, (nx - 1) / 2])
    y = np.array([0, 0, ny - 1, ny - 1, (ny - 1) / 2])
    for frame in (0, n_frames - 1):
        from_map = cube.to_maps(frame).wcs.pixel_to_world(x, y)
        from_cube = cube.wcs.pixel_to_world(x, y, frame)[0]
        np.testing.assert_allclose(from_map.Tx.to_value(u.arcsec), from_cube.Tx.to_value(u.arcsec), atol=1e-6)
        np.testing.assert_allclose(from_map.Ty.to_value(u.arcsec), from_cube.Ty.to_value(u.arcsec), atol=1e-6)


def test_to_maps_negative_index(sns_sjicube_1330):
    last = sns_sjicube_1330.shape[0] - 1
    assert sns_sjicube_1330.to_maps(-1).meta["DATE-OBS"] == sns_sjicube_1330.to_maps(last).meta["DATE-OBS"]
    maps = sns_sjicube_1330.to_maps([-1, 0])
    # The frames come in the requested order; the sequence used to sort them by date but label them in request order
    assert [m.meta["DATE-OBS"] for m in maps] == [sns_sjicube_1330.to_maps(i).meta["DATE-OBS"] for i in (last, 0)]
    for m, i in zip(maps, (last, 0), strict=True):
        np.testing.assert_array_equal(m.data, sns_sjicube_1330.data[i])


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

    cube = read_spectrograph_lvl2(sns_sg_file, spectral_windows="Si IV 1403")["Si IV 1403"][0]
    _, point, _, _ = cube.wcs.pixel_to_world(5, 20, 93)
    moved = point.transform_to(frame)
    assert_quantity_allclose(moved.Tx, point.Tx, atol=0.1 * u.arcsec)
    assert_quantity_allclose(moved.Ty, point.Ty, atol=0.1 * u.arcsec)


def test_sjicube_takes_only_data_and_wcs_positionally(sns_sjicube_1330):
    with pytest.raises(TypeError, match="positional"):
        SJICube(sns_sjicube_1330.data, sns_sjicube_1330.wcs, sns_sjicube_1330.unit)


@pytest.mark.parametrize(
    ("item", "expected_len"),
    [(-1, None), ((slice(0, 3), slice(0, 10)), 3), ((Ellipsis, slice(0, 10)), 52), ((0, Ellipsis), None)],
    ids=["frame", "ranges", "ellipsis_first", "ellipsis_last"],
)
def test_sjicube_slice_keeps_the_fits_wcs_of_its_frames(sns_sjicube_1330, item, expected_len):
    subset = sns_sjicube_1330[item]

    if expected_len is None:
        assert isinstance(subset.fits_wcs, WCS)
    else:
        assert len(subset.fits_wcs) == expected_len
        assert isinstance(subset.fits_wcs[0], WCS)


def test_sjicube_slice_rejects_multiple_ellipsis(sns_sjicube_1330):
    with pytest.raises((IndexError, ValueError), match=r"single ellipsis|only have a single ellipsis"):
        sns_sjicube_1330[(Ellipsis, Ellipsis)]


def test_sjicube_slice_rejects_too_many_indices(sns_sjicube_1330):
    with pytest.raises((IndexError, ValueError), match=r"can not be greater than the dimensionality"):
        sns_sjicube_1330[(slice(0, 3), slice(0, 10), slice(0, 10), slice(None))]
