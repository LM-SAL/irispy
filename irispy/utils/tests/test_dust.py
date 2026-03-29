import matplotlib.pyplot as plt
import numpy as np
import pytest

from astropy import units as u

from irispy.tests.helpers import figure_test
from irispy.utils.dust import remove_dust


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

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=2, exposure_normalize=False)

    assert cleaned.data[1, 1, 1] == 10
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

    original_basic_wcs = cube.basic_wcs
    original_extra_coord_keys = list(cube.extra_coords.keys())
    original_global_coord_keys = list(cube.global_coords.keys())

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=1, exposure_normalize=False)

    assert type(cleaned) is type(cube)
    assert cleaned.scaled is cube.scaled
    assert list(cleaned.extra_coords.keys()) == original_extra_coord_keys
    assert list(cleaned.global_coords.keys()) == original_global_coord_keys
    assert cleaned.basic_wcs is not None
    assert len(cleaned.basic_wcs) == len(original_basic_wcs)
    assert cleaned.to_maps(0).wcs is not None


def test_remove_dust_uses_spatial_fallback_for_single_images(sns_sjicube_1330):
    cube = sns_sjicube_1330[0, :3, :3]
    cube.data[...] = np.array(
        [
            [5, 5, 5],
            [5, 0, 9],
            [5, 5, 5],
        ],
        dtype=float,
    )
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 1] = True

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=0, spatial_box=3)

    assert cleaned.data[1, 1] == 5
    assert cleaned.mask[1, 1] is np.False_


def test_remove_dust_keeps_unfilled_pixels_masked_when_fallback_disabled(sns_sjicube_1330):
    cube = sns_sjicube_1330[0, :3, :3]
    cube.data[...] = np.array(
        [
            [5, 5, 5],
            [5, 0, 9],
            [5, 5, 5],
        ],
        dtype=float,
    )
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 1] = True

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=0, fallback=None)

    assert cleaned.data[1, 1] == 0
    assert cleaned.mask[1, 1] is np.True_


def test_remove_dust_rejects_string_none_fallback(sns_sjicube_1330):
    cube = sns_sjicube_1330[0, :3, :3]
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 1] = True

    with pytest.raises(ValueError, match=r"fallback must be 'spatial' or None\."):
        remove_dust(cube, dust_mask=dust_mask, fallback="none")


def test_sjicube_remove_dust_method_matches_function(sns_sjicube_1330):
    cube = sns_sjicube_1330[:3, :3, :3]
    cube.data[...] = np.array(
        [
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        ],
        dtype=float,
    )
    cube.mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[0, 1, 1] = True

    cleaned_by_function = remove_dust(cube, dust_mask=dust_mask, temporal_window=1, exposure_normalize=False)
    cleaned_by_method = cube.remove_dust(dust_mask=dust_mask, temporal_window=1, exposure_normalize=False)

    np.testing.assert_allclose(cleaned_by_method.data, cleaned_by_function.data)
    np.testing.assert_array_equal(cleaned_by_method.mask, cleaned_by_function.mask)


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

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=2, exposure_normalize=False)

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
