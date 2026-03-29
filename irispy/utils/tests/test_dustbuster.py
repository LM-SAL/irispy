from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u

import irispy.utils.dustbuster as dustbuster_module
from irispy.tests.helpers import figure_test
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED
from irispy.utils.dustbuster import clean_sji, clean_sji_regions, get_sji_dust_params


def _clean_test_cube(cube, *, align, fill_mode="blur", dust_ids=None):
    return dustbuster_module._clean_sji_with_params(
        cube,
        dust_ids=[2073] if dust_ids is None else dust_ids,
        slit_center=(1.0, 0.5),
        mask_scale=1.0,
        roll_deg=0.0,
        align=align,
        fill_mode=fill_mode,
    )


def _clean_test_region_cube(cube, *, align, fill_mode="blur", dust_ids=None):
    return dustbuster_module._clean_sji_regions_with_params(
        cube,
        dust_ids=[2073] if dust_ids is None else dust_ids,
        slit_center=(1.0, 0.5),
        mask_scale=1.0,
        roll_deg=0.0,
        align=align,
        fill_mode=fill_mode,
    )


def _first_changed_frame(original_cube, *cleaned_cubes):
    changed_frames = np.zeros(original_cube.data.shape[0], dtype=bool)
    for cleaned_cube in cleaned_cubes:
        changed_frames |= np.any(~np.isclose(cleaned_cube.data, original_cube.data), axis=(1, 2))
    changed_idx = np.flatnonzero(changed_frames)
    assert changed_idx.size > 0
    return int(changed_idx[0])


def _changed_region(original_frame, *cleaned_frames, margin=20):
    changed = np.zeros(original_frame.shape, dtype=bool)
    for cleaned_frame in cleaned_frames:
        changed |= ~np.isclose(cleaned_frame, original_frame)

    y_idx, x_idx = np.nonzero(changed)
    assert y_idx.size > 0
    y_min = max(int(y_idx.min()) - margin, 0)
    y_max = min(int(y_idx.max()) + margin + 1, original_frame.shape[0])
    x_min = max(int(x_idx.min()) - margin, 0)
    x_max = min(int(x_idx.max()) + margin + 1, original_frame.shape[1])
    return slice(y_min, y_max), slice(x_min, x_max)


def test_align_frame_idx_covers_short_and_long_cubes():
    np.testing.assert_array_equal(dustbuster_module._align_frame_idx(3), np.arange(3))

    frame_idx = dustbuster_module._align_frame_idx(9)

    assert len(frame_idx) <= dustbuster_module._MAX_ALIGNMENT_FRAMES
    assert frame_idx[0] == 0
    assert frame_idx[-1] == 8


def test_clean_sji_replaces_2d_dust_pixel_and_clears_mask(make_sji_cube):
    cube = make_sji_cube(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 0.1, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    cleaned_cube = _clean_test_cube(cube, align=False)

    assert cube.data[1, 1] == 0.1
    assert cleaned_cube.data[1, 1] != cube.data[1, 1]
    assert np.isfinite(cleaned_cube.data[1, 1])
    assert not cleaned_cube.mask[1, 1]


def test_clean_sji_aligns_and_uses_temporal_fill_for_3d_data(make_sji_cube):
    data = np.full((9, 4, 4), 10.0)
    slit_x = np.array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0])
    for frame_idx, dust_x in enumerate((slit_x - 1).astype(int)):
        data[frame_idx, 1, dust_x] = 0.1

    cube = make_sji_cube(data, slit_x=slit_x)

    cleaned_cube = _clean_test_cube(cube, align=True)

    assert cleaned_cube.data[0, 1, 1] == pytest.approx(10.0)
    assert cleaned_cube.data[1, 1, 2] == pytest.approx(10.0)
    assert not cleaned_cube.mask[0, 1, 1]
    assert not cleaned_cube.mask[1, 1, 2]


def test_clean_sji_uses_global_fill_when_temporal_and_spatial_fill_fail(make_sji_cube):
    data = np.full((2, 4, 4), BAD_PIXEL_VALUE_SCALED, dtype=float)
    data[0, 1, 1] = 1.0
    data[1, 0, 0] = 7.0
    data[1, 1, 2] = 3.0

    cube = make_sji_cube(data, slit_x=[2.0, 3.0])

    cleaned_cube = _clean_test_cube(cube, align=False)

    assert cleaned_cube.data[0, 1, 1] == pytest.approx(3.0)


def test_clean_sji_fill_mode_global_skips_blur_fill(make_sji_cube):
    data = np.ones((2, 20, 20), dtype=float)
    data[:, :6, :6] = 10.0
    data[:, 1, 1] = 0.1

    cube = make_sji_cube(data)

    cleaned_blur = _clean_test_cube(cube, align=False, fill_mode="blur")
    cleaned_global = _clean_test_cube(cube, align=False, fill_mode="global")

    assert cleaned_blur.data[0, 1, 1] > cleaned_global.data[0, 1, 1]
    assert cleaned_global.data[0, 1, 1] == pytest.approx(1.0)


def test_clean_sji_fills_off_slit_negative_island(make_sji_cube):
    data = np.full((7, 7), 5.0)
    data[3, 4] = -3.0

    cube = make_sji_cube(data)
    cleaned_cube = _clean_test_cube(cube, align=False, dust_ids=[])

    assert cleaned_cube.data[3, 4] == pytest.approx(5.0)


def test_clean_sji_leaves_negative_pixel_on_slit(make_sji_cube):
    data = np.full((7, 7), 5.0)
    data[3, 1] = -3.0

    cube = make_sji_cube(data)
    cleaned_cube = _clean_test_cube(cube, align=False, dust_ids=[])

    assert cleaned_cube.data[3, 1] == pytest.approx(-3.0)


def test_clean_sji_regions_expands_seed_into_connected_negative_region(make_sji_cube):
    data = np.full((7, 7), 5.0)
    data[1, 1] = BAD_PIXEL_VALUE_SCALED
    data[1, 2] = -4.0
    data[2, 1] = -3.0

    cube = make_sji_cube(data)
    cleaned_default = _clean_test_cube(cube, align=False)
    cleaned_region = _clean_test_region_cube(cube, align=False)

    assert cleaned_default.data[1, 1] == pytest.approx(BAD_PIXEL_VALUE_SCALED)
    assert cleaned_default.data[2, 1] == pytest.approx(-3.0)
    assert cleaned_region.data[1, 1] == pytest.approx(5.0)
    assert cleaned_region.data[1, 2] == pytest.approx(5.0)
    assert cleaned_region.data[2, 1] == pytest.approx(5.0)


@figure_test
def test_clean_sji_compare_fill_modes(make_sji_cube):
    data = np.ones((2, 20, 20), dtype=float)
    data[:, :6, :6] = 10.0
    data[:, 1, 1] = 0.1

    cube = make_sji_cube(data)
    cleaned_blur = _clean_test_cube(cube, align=False, fill_mode="blur")
    cleaned_global = _clean_test_cube(cube, align=False, fill_mode="global")

    original_frame = cube.data[0]
    blur_frame = cleaned_blur.data[0]
    global_frame = cleaned_global.data[0]
    y_slice, x_slice = _changed_region(original_frame, blur_frame, global_frame, margin=8)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        (original_frame, blur_frame, global_frame),
        ("Original", 'fill_mode="blur"', 'fill_mode="global"'),
        strict=True,
    ):
        ax.imshow(image[y_slice, x_slice], origin="lower", cmap="gray", vmin=0, vmax=10)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Dust fill fallback comparison")
    return fig


def test_clean_sji_validates_fill_mode(make_sji_cube):
    cube = make_sji_cube(np.zeros((2, 4, 4)))

    with pytest.raises(ValueError, match="fill_mode must be 'blur' or 'global'"):
        _clean_test_cube(cube, align=False, fill_mode="bad")


def test_clean_sji_validates_input_shape(make_sji_cube):
    cube = make_sji_cube(np.zeros((2, 4, 4)))[:, 0, 0]

    with pytest.raises(ValueError, match=r"cube.data must have shape"):
        _clean_test_cube(cube, align=False)


def test_clean_sji_requires_basic_wcs(make_sji_cube):
    cube = make_sji_cube(np.zeros((2, 4, 4)))
    cube._basic_wcs = None

    with pytest.raises(ValueError, match=r"cube.basic_wcs is required"):
        _clean_test_cube(cube, align=False)


def test_clean_sji_requires_one_basic_wcs_per_frame(make_sji_cube):
    cube = make_sji_cube(np.zeros((2, 4, 4)))
    cube._basic_wcs = cube._basic_wcs[:1]

    with pytest.raises(ValueError, match=r"cube.basic_wcs must contain one WCS per frame"):
        _clean_test_cube(cube, align=False)


def test_clean_sji_requires_per_frame_extra_coords(make_sji_cube):
    cube = make_sji_cube(np.zeros((2, 4, 4)))
    cube.__dict__["__extra_coords"] = {
        "exposure time": SimpleNamespace(wcs=SimpleNamespace(pixel_to_world=lambda _pixels: np.array([1.0]) * u.s)),
        "slit x position": SimpleNamespace(
            wcs=SimpleNamespace(pixel_to_world=lambda _pixels: np.array([2.0, 2.0]) * u.arcsec)
        ),
        "slit y position": SimpleNamespace(
            wcs=SimpleNamespace(pixel_to_world=lambda _pixels: np.array([2.0, 2.0]) * u.arcsec)
        ),
    }

    with pytest.raises(ValueError, match="required per-frame extra coordinates"):
        _clean_test_cube(cube, align=False)


@pytest.mark.remote_data
@pytest.mark.parametrize(
    ("fixture_name", "expected_slit_center", "expected_mask_scale", "expected_roll_deg"),
    [
        ("sns_sjicube_1330", (536.29999, 523.03003), 0.1656, -0.2),
        ("sns_sjicube_1400", (528.32001, 509.45999), 0.1656, -0.22400001),
        ("sns_sjicube_2796", (503.69, 502.40201), 0.1679, 0.27399999),
        ("sns_sjicube_2832", (505.47, 501.22), 0.1679, 0.28600001),
    ],
)
def test_get_sji_dust_params_real_sji_values(
    request,
    fixture_name,
    expected_slit_center,
    expected_mask_scale,
    expected_roll_deg,
):
    cube = request.getfixturevalue(fixture_name)
    params = get_sji_dust_params(
        date_obs=cube.meta["DATE_OBS"],
        sji_name=cube.meta["TDESC1"],
    )

    assert set(params) == {"dust_ids", "slit_center", "mask_scale", "roll_deg"}
    assert params["dust_ids"].ndim == 1
    assert params["dust_ids"].size > 0
    assert params["dust_ids"].dtype == np.int64
    assert params["slit_center"] == pytest.approx(expected_slit_center)
    assert params["mask_scale"] == pytest.approx(expected_mask_scale)
    assert params["roll_deg"] == pytest.approx(expected_roll_deg)


@pytest.mark.remote_data
def test_clean_sji_smoke_real_sji_cube(sns_sjicube_2796):
    cube = sns_sjicube_2796[:5]
    cleaned_cube = clean_sji(cube, align=False)

    assert cleaned_cube.data.shape == cube.data.shape
    assert cleaned_cube.basic_wcs is not None


@pytest.mark.remote_data
def test_clean_sji_regions_smoke_real_sji_cube(sns_sjicube_2796):
    cube = sns_sjicube_2796[:5]
    cleaned_cube = clean_sji_regions(cube, align=False)

    assert cleaned_cube.data.shape == cube.data.shape
    assert cleaned_cube.basic_wcs is not None


@pytest.mark.remote_data
@figure_test
def test_clean_sji_compare_options(example_sjicube_2832):
    cube = example_sjicube_2832[:5]
    cleaned_fast = clean_sji(cube, align=False)
    cleaned_aligned = clean_sji(cube, align=True)

    frame_idx = _first_changed_frame(cube, cleaned_fast, cleaned_aligned)
    original_frame = cube.data[frame_idx]
    fast_frame = cleaned_fast.data[frame_idx]
    aligned_frame = cleaned_aligned.data[frame_idx]
    y_slice, x_slice = _changed_region(original_frame, fast_frame, aligned_frame)

    vmin, vmax = np.nanpercentile(aligned_frame[y_slice, x_slice], [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        (original_frame, fast_frame, aligned_frame),
        ("Original", "align=False", "align=True"),
        strict=True,
    ):
        ax.imshow(image[y_slice, x_slice], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Dust removal comparison, frame {frame_idx}")
    return fig


@pytest.mark.remote_data
@pytest.mark.parametrize(
    ("sji_name", "message"),
    [
        ("2796", "Unsupported TDESC1"),
        ("SJI_9999", "Unsupported SJI channel"),
        ("SJI_1600", "No flat-index rows matched"),
        ("SJI_5000", "No flat-index rows matched"),
    ],
)
def test_get_sji_dust_params_validates_real_inputs(sji_name, message):
    with pytest.raises(ValueError, match=message):
        get_sji_dust_params(
            date_obs="2024-01-01T00:00:00.000",
            sji_name=sji_name,
        )
