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

    original_fits_wcs = cube.fits_wcs
    original_extra_coord_keys = list(cube.extra_coords.keys())
    original_global_coord_keys = list(cube.global_coords.keys())

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=1, exposure_normalize=False)

    assert type(cleaned) is type(cube)
    assert cleaned.scaled is cube.scaled
    assert list(cleaned.extra_coords.keys()) == original_extra_coord_keys
    assert list(cleaned.global_coords.keys()) == original_global_coord_keys
    assert cleaned.fits_wcs is not None
    assert len(cleaned.fits_wcs) == len(original_fits_wcs)
    assert cleaned.to_maps(0).wcs is not None


def test_remove_dust_exposure_normalize_uses_exposure_time(sns_sjicube_1330, monkeypatch):
    # Use a small cube with a single spatial pixel so temporal behavior is clear
    cube = sns_sjicube_1330[:4, :1, :1]
    cube.data[...] = 0.0
    cube.mask = np.zeros_like(cube.data, dtype=bool)

    # Exposure times vary so a simple median in data space would differ
    exposure_times = u.Quantity([1, 2, 4, 8], u.s)
    monkeypatch.setattr(type(cube), "exposure_time", property(lambda _self: exposure_times))

    scale = 10.0
    for i, t in enumerate(exposure_times.value):
        cube.data[i, 0, 0] = scale * t

    # Introduce a dust pixel in the second frame
    dust_mask = np.zeros_like(cube.data, dtype=bool)
    dust_mask[1, 0, 0] = True
    cube.data[1, 0, 0] = 0.0
    cube.mask[1, 0, 0] = True

    cleaned = remove_dust(
        cube,
        dust_mask=dust_mask,
        temporal_window=1,
        exposure_normalize=True,
    )

    # If exposure normalization is used, the repaired value should scale with the
    # local exposure time (i.e. be proportional to exposure_time[1])
    expected_value = scale * exposure_times[1].value
    assert cleaned.data[1, 0, 0] == expected_value

    # And it should *not* equal the median of the un-normalized neighboring values
    # (which would be 10, 40, 80 -> median 40)
    assert cleaned.data[1, 0, 0] != np.median([10.0, 40.0, 80.0])


def test_remove_dust_exposure_normalize_scalar_exposure_time_falls_back(sns_sjicube_1330, monkeypatch):
    # Two otherwise-identical cubes: one with exposure_normalize=True and a bad
    # exposure_time, and one with exposure_normalize=False
    cube_norm = sns_sjicube_1330[:4, :1, :1]
    cube_no_norm = sns_sjicube_1330[:4, :1, :1]

    for cube in (cube_norm, cube_no_norm):
        cube.data[...] = 1.0
        cube.mask = np.zeros_like(cube.data, dtype=bool)

    dust_mask = np.zeros_like(cube_norm.data, dtype=bool)
    dust_mask[1, 0, 0] = True
    cube_norm.data[1, 0, 0] = 0.0
    cube_no_norm.data[1, 0, 0] = 0.0
    cube_norm.mask[1, 0, 0] = True
    cube_no_norm.mask[1, 0, 0] = True

    # Bad exposure_time: scalar instead of 1D array matching the time axis
    monkeypatch.setattr(type(cube_norm), "exposure_time", property(lambda _self: 2.0 * u.s))

    cleaned_no_norm = remove_dust(
        cube_no_norm,
        dust_mask=dust_mask,
        temporal_window=1,
        exposure_normalize=False,
    )
    cleaned_norm = remove_dust(
        cube_norm,
        dust_mask=dust_mask,
        temporal_window=1,
        exposure_normalize=True,
    )

    # With an invalid exposure_time, the function should fall back to the
    # non-normalized behavior without raising
    np.testing.assert_allclose(cleaned_norm.data, cleaned_no_norm.data)


def test_remove_dust_exposure_normalize_wrong_length_exposure_time_falls_back(sns_sjicube_1330, monkeypatch):
    cube_norm = sns_sjicube_1330[:4, :1, :1]
    cube_no_norm = sns_sjicube_1330[:4, :1, :1]

    for cube in (cube_norm, cube_no_norm):
        cube.data[...] = 1.0
        cube.mask = np.zeros_like(cube.data, dtype=bool)

    dust_mask = np.zeros_like(cube_norm.data, dtype=bool)
    dust_mask[2, 0, 0] = True
    cube_norm.data[2, 0, 0] = 0.0
    cube_no_norm.data[2, 0, 0] = 0.0
    cube_norm.mask[2, 0, 0] = True
    cube_no_norm.mask[2, 0, 0] = True

    # Bad exposure_time: array length does not match number of time steps (4)
    monkeypatch.setattr(type(cube_norm), "exposure_time", property(lambda _self: u.Quantity([1, 2, 3], u.s)))

    cleaned_no_norm = remove_dust(
        cube_no_norm,
        dust_mask=dust_mask,
        temporal_window=1,
        exposure_normalize=False,
    )
    with pytest.warns(UserWarning, match=r"exposure_normalize=True but the number of exposure_time values"):
        cleaned_norm = remove_dust(
            cube_norm,
            dust_mask=dust_mask,
            temporal_window=1,
            exposure_normalize=True,
        )

    # Again, we should fall back to the same behavior as exposure_normalize=False
    np.testing.assert_allclose(cleaned_norm.data, cleaned_no_norm.data)


def test_remove_dust_warns_when_exposure_time_length_mismatches_frames(sns_sjicube_1330, monkeypatch):
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

    monkeypatch.setattr(type(cube), "exposure_time", property(lambda _self: np.array([1.0, 2.0]) * u.s))

    with pytest.warns(UserWarning, match=r"exposure_normalize=True but the number of exposure_time values"):
        cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=1, exposure_normalize=True)

    assert type(cleaned) is type(cube)
    assert cleaned.data.shape == cube.data.shape


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

    cleaned = remove_dust(cube, dust_mask=dust_mask_2d, temporal_window=1, exposure_normalize=False)

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
    assert cleaned.dust_masked is False


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
    assert cleaned.dust_masked is True

    cleaned = remove_dust(cube, dust_mask=dust_mask, temporal_window=0, spatial_box=3)

    assert cleaned.data[1, 1] == 5
    assert cleaned.mask[1, 1] is np.False_
    assert cleaned.dust_masked is False


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
