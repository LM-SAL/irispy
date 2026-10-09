import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u

import sunpy.map

from irispy.io.utils import read_files


def test_apply_dust_mask_with_lazy_mask(sns_sjicube_1330):
    cube = sns_sjicube_1330[:1, :5, :5]
    cube.data[:] = 10
    cube.data[0, 2, 2] = 0
    original_mask = np.zeros(cube.shape, dtype=bool)
    original_mask[0, 0, 0] = True
    cube.mask = da.from_array(original_mask, chunks=(1, 5, 5))

    cube.apply_dust_mask()
    expected = original_mask.copy()
    expected[0, 1:4, 1:4] = True
    assert isinstance(cube.mask, da.Array)
    np.testing.assert_array_equal(cube.mask.compute(), expected)
    assert cube.dust_masked

    cube.apply_dust_mask(undo=True)
    assert isinstance(cube.mask, da.Array)
    np.testing.assert_array_equal(cube.mask.compute(), original_mask)
    assert not cube.dust_masked


def test_apply_dust_mask_rejects_unscaled_data(sns_sji_1330_file):
    cube = read_files(sns_sji_1330_file, memmap=True)
    with pytest.raises(ValueError, match=r"unscaled.*memmap=False"):
        cube.apply_dust_mask()


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


def test_negative_slices(sns_sjicube_1330):
    assert sns_sjicube_1330[-3:].shape == (3, 40, 37)
    assert len(sns_sjicube_1330[-3:].fits_wcs) == 3
    assert sns_sjicube_1330[:-2].shape == (50, 40, 37)
    assert len(sns_sjicube_1330[:-2].fits_wcs) == 50


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
