import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import ndimage

from astropy import units as u

from irispy.io import read_files
from irispy.tests.helpers import figure_test
from irispy.utils.dust import _local_median_fill, calculate_dust_mask, remove_dust


def test_remove_dust_repairs_pixels_from_neighboring_frames(sns_sjicube_1330):
    cube = sns_sjicube_1330[:4, :3, :3]
    cube.data[...] = np.array(
        [
            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],
            [[20, 20, 20], [20, 0, 20], [20, 20, 20]],
            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],
            [[20, 20, 20], [20, 20, 20], [20, 20, 20]],
        ],
        dtype=float,
    )
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 1, 1] = True
    original_exposure_times = cube.axis_world_coords(0, wcs=cube.extra_coords["exposure time"])[0].to_value(u.s)

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=2)

    assert cleaned.data[1, 1, 1] == pytest.approx(10, rel=1e-4)
    assert cleaned.mask[1, 1, 1] is np.False_
    assert cube.data[1, 1, 1] == 0
    np.testing.assert_allclose(
        cleaned.axis_world_coords(0, wcs=cleaned.extra_coords["exposure time"])[0].to_value(u.s),
        original_exposure_times,
    )


def test_remove_dust_preserves_sjicube_state(sns_sjicube_1330):
    cube = sns_sjicube_1330[:4, :3, :3]
    cube.data[...] = 1.0
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 1, 1] = True
    cube.data[1, 1, 1] = 0.0

    original_fits_wcs = cube.fits_wcs
    original_extra_coord_keys = list(cube.extra_coords.keys())
    original_global_coord_keys = list(cube.global_coords.keys())

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=1)

    assert type(cleaned) is type(cube)
    assert cleaned.data.dtype == np.float32
    assert list(cleaned.extra_coords.keys()) == original_extra_coord_keys
    assert list(cleaned.global_coords.keys()) == original_global_coord_keys
    assert cleaned.fits_wcs is not None
    assert len(cleaned.fits_wcs) == len(original_fits_wcs)
    assert cleaned.to_maps(0).wcs is not None


def test_remove_dust_does_not_share_meta_or_coordinates(sns_sjicube_1330):
    cube = sns_sjicube_1330
    image = cube[0]
    cleaned, cleaned_image = remove_dust(cube), remove_dust(image)
    # The copied coordinates still place the dust in each frame
    np.testing.assert_array_equal(calculate_dust_mask(cleaned), calculate_dust_mask(cube))
    np.testing.assert_array_equal(calculate_dust_mask(cleaned_image), calculate_dust_mask(image))
    cleaned.meta["note"] = "changed"
    cleaned.extra_coords.add("sequence", 0, np.arange(len(cleaned.data)) * u.one)
    cleaned_image.global_coords.add("note", "time", image.global_coords["Time (UTC)"])
    assert "note" not in cube.meta
    assert "sequence" not in tuple(cube.extra_coords.keys())
    assert "note" not in image.global_coords


@pytest.mark.parametrize(
    ("unit", "values", "expected"),
    [(u.DN, [10, 20, 40, 80], 20), (u.DN / u.s, [10, 10, 10, 10], 10)],
)
def test_remove_dust_scales_counts_to_the_exposure_time(sns_sjicube_1330, monkeypatch, unit, values, expected):
    # 10 DN/s through exposures of 1, 2, 4 and 8 s, as under automatic exposure control
    monkeypatch.setattr(type(sns_sjicube_1330), "exposure_time", property(lambda _self: [1, 2, 4, 8] * u.s))
    cube = sns_sjicube_1330[:4, :1, :1]
    cube = cube.to_nddata(
        data=np.reshape(values, cube.shape).astype(float),
        mask=np.zeros(cube.shape, dtype=bool),
        unit=unit,
        nddata_type=type(cube),
        extra_coords="copy",
        global_coords="copy",
    )
    dust_mask = np.zeros(cube.shape, dtype=bool)
    dust_mask[1, 0, 0] = True

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=1, fallback=None)

    assert cleaned.data[1, 0, 0] == expected


def test_remove_dust_broadcasts_2d_mask_over_time(sns_sjicube_1330):
    cube = sns_sjicube_1330[:4, :3, :3]
    cube.data[...] = np.array(
        [
            [[10, 10, 10], [10, 0, 10], [10, 10, 10]],
            [[20, 20, 20], [20, 0, 20], [20, 20, 20]],
            [[30, 30, 30], [30, 0, 30], [30, 30, 30]],
            [[40, 40, 40], [40, 0, 40], [40, 40, 40]],
        ],
        dtype=float,
    )
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask_2d = np.zeros(cube.data.shape[1:], dtype=bool)
    dust_mask_2d[1, 1] = True

    cleaned = remove_dust(cube, dust_mask=dust_mask_2d, temporal_window=1)

    expected_broadcast_mask = np.broadcast_to(dust_mask_2d, cube.data.shape)
    # All broadcast dust pixels are repaired by spatial fallback, so they should be unmasked.
    np.testing.assert_array_equal(
        cleaned.mask[expected_broadcast_mask], np.zeros(expected_broadcast_mask.sum(), dtype=bool)
    )
    np.testing.assert_allclose(cleaned.data[:, 1, 1], np.array([10.0, 20.0, 30.0, 40.0]))


@pytest.mark.parametrize(
    "mask_shape",
    [
        (4, 3),
        (3, 4),
        (4, 4, 3),
        (4, 3, 4),
    ],
)
def test_remove_dust_raises_for_incompatible_mask_shapes(sns_sjicube_1330, mask_shape):
    cube = sns_sjicube_1330[:4, :3, :3]
    dust_mask = np.zeros(mask_shape, dtype=bool)

    with pytest.raises(ValueError, match=r"dust_mask must have shape") as excinfo:
        remove_dust(cube, dust_mask=dust_mask)

    message = str(excinfo.value)
    assert "dust_mask" in message
    assert "shape" in message


@pytest.mark.parametrize(("fallback", "expected", "unfilled"), [("spatial", 5, False), (None, 0, True)])
def test_remove_dust_spatial_fallback_for_single_images(sns_sjicube_1330, fallback, expected, unfilled):
    cube = sns_sjicube_1330[0, :3, :3]
    cube.data[...] = [[5, 5, 5], [5, 0, 9], [5, 5, 5]]
    cube.mask = np.zeros(cube.shape, dtype=bool)
    dust_mask = np.zeros(cube.shape, dtype=bool)
    dust_mask[1, 1] = True

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=0, spatial_box=3, fallback=fallback)

    assert cleaned.data[1, 1] == expected
    np.testing.assert_array_equal(cleaned.mask, dust_mask if unfilled else np.zeros_like(dust_mask))


def test_remove_dust_rejects_string_none_fallback(sns_sjicube_1330):
    cube = sns_sjicube_1330[0, :3, :3]
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 1] = True

    with pytest.raises(ValueError, match=r"fallback must be 'spatial' or None\."):
        remove_dust(cube, dust_mask=dust_mask, fallback="none")


@figure_test
def test_remove_dust_before_after_figure(sns_sjicube_1330):
    cube = sns_sjicube_1330[:4, :10, :10]
    cube.data[...] = np.array(
        [
            np.full((10, 10), 10.0),
            np.full((10, 10), 20.0),
            np.full((10, 10), 10.0),
            np.full((10, 10), 20.0),
        ]
    )
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 5, 5] = True
    cube.data[1, 5, 5] = 0.0

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=2)

    original_frame = cube[1]
    cleaned_frame = cleaned[1]

    fig = plt.figure(figsize=(8, 4))
    fig.subplots_adjust(wspace=0.03)

    ax0 = fig.add_subplot(121, projection=original_frame.wcs)
    original_frame.plot(axes=ax0, vmin=0, vmax=20)
    ax0.set_title("Original")

    ax1 = fig.add_subplot(122, projection=cleaned_frame.wcs)
    cleaned_frame.plot(axes=ax1, vmin=0, vmax=20)
    ax1.set_title("Dust Removed")

    for ax in (ax0, ax1):
        ax.coords[0].set_ticks_visible(False)
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticks_visible(False)
        ax.coords[1].set_ticklabel_visible(False)

    return fig


@pytest.mark.parametrize("spatial_box", [3, 4, 5])
def test_local_median_fill_matches_generic_filter(spatial_box):
    rng = np.random.default_rng(0)
    frame = rng.normal(size=(30, 40))
    invalid = rng.random(frame.shape) < 0.3
    target = rng.random(frame.shape) < 0.5
    expected = ndimage.generic_filter(np.where(invalid, np.nan, frame), np.nanmedian, size=spatial_box, mode="nearest")
    result = _local_median_fill(frame, invalid, spatial_box, target)
    np.testing.assert_array_equal(result[target], expected[target])
    assert np.isnan(result[~target]).all()


def _dark_fraction(frame, where):
    """
    The share of the pixels at ``where`` darker than half their 15 by 15 neighborhood.
    """
    frame = np.asarray(frame, dtype=float)
    local = ndimage.median_filter(np.nan_to_num(frame, nan=np.nanmedian(frame)), 15)
    return float(((frame / np.maximum(local, 1)) < 0.5)[where].mean())


@pytest.mark.remote_data
@pytest.mark.parametrize("channel", ["SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"])
def test_remove_dust_repairs_the_dust_of_each_channel(channel):
    from irispy.data import sample  # NOQA: PLC0415

    cube = read_files(str(getattr(sample, channel)))
    dust_mask = calculate_dust_mask(cube)
    cleaned = remove_dust(cube)

    core = ndimage.binary_erosion(dust_mask[0], np.ones((3, 3)))
    elsewhere = ~dust_mask[0] & np.isfinite(np.asarray(cube.data[0]))
    assert _dark_fraction(cube.data[0], core) > _dark_fraction(cube.data[0], elsewhere) + 0.2
    assert _dark_fraction(cleaned.data[0], core) < _dark_fraction(cube.data[0], elsewhere) + 0.05


@pytest.mark.remote_data
def test_calculate_dust_mask_follows_slicing():
    from irispy.data import sample  # NOQA: PLC0415

    cube = read_files(str(sample.SJI_2832))
    dust_mask = calculate_dust_mask(cube)
    assert dust_mask.shape == cube.shape
    assert dust_mask.any()
    np.testing.assert_array_equal(calculate_dust_mask(cube[2:5, 30:, 40:]), dust_mask[2:5, 30:, 40:])
    np.testing.assert_array_equal(calculate_dust_mask(cube[3]), dust_mask[3])
    # A crop whose edge is next to a bad pixel still gets the dilation into it
    rows, columns = np.nonzero(dust_mask[3])
    y0, x0 = rows.min() + 1, columns.min() + 1
    np.testing.assert_array_equal(calculate_dust_mask(cube[3, y0:, x0:]), dust_mask[3, y0:, x0:])


def test_calculate_dust_mask_has_the_cube_shape(sns_sjicube_2832):
    dust_mask = calculate_dust_mask(sns_sjicube_2832)
    assert dust_mask.shape == sns_sjicube_2832.shape
    assert dust_mask.dtype == bool
    assert not dust_mask[np.isnan(np.asarray(sns_sjicube_2832.data))].any()


def test_calculate_dust_mask_rejects_other_instruments(aia_1700_file):
    with pytest.raises(ValueError, match="No bad-pixel table"):
        calculate_dust_mask(read_files(aia_1700_file))


def test_remove_dust_leaves_pixels_without_values(sns_sjicube_1330):
    cube = sns_sjicube_1330[:3, :4, :4]
    cube.data[...] = 10.0
    cube.data[1, 1, 1] = np.nan
    cube.mask = np.isnan(cube.data)
    dust_mask = np.zeros(cube.shape, dtype=bool)
    dust_mask[1, 1, 1] = True
    dust_mask[1, 2, 2] = True

    cleaned = remove_dust(cube, dust_mask=dust_mask)

    assert np.isnan(cleaned.data[1, 1, 1])
    assert cleaned.mask[1, 1, 1]
    assert cleaned.data[1, 2, 2] == pytest.approx(10, rel=1e-4)
    assert not cleaned.mask[1, 2, 2]
