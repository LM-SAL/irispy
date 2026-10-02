import sys
import types

import dask.array as da
import numpy as np
import pytest

from irispy.utils.cosmic_rays import remove_cosmic_rays


def test_remove_cosmic_rays_rsliding(sns_sjicube_1330):
    pytest.importorskip("rsliding")
    cube = sns_sjicube_1330[0, :3, :3]
    data = np.array([[1.0, 2.0, 3.0], [4.0, 100.0, 6.0], [7.0, 8.0, 9.0]])
    mask = np.array([[False, False, False], [False, False, False], [True, False, False]])
    cube.data[...] = data
    cube.mask = mask.copy()
    original_dust_masked = cube.dust_masked

    cleaned_cube = remove_cosmic_rays(
        cube,
        sigma=2.5,
        max_iters=7,
        method_kwargs={"kernel": 3, "threads": 1},
    )

    np.testing.assert_allclose(cleaned_cube.data[1, 1], 5.0, atol=2.0)
    np.testing.assert_array_equal(cleaned_cube.mask, mask)
    assert cleaned_cube.dust_masked == original_dust_masked


def test_remove_cosmic_rays_astroscrappy(sns_sjicube_1330):
    pytest.importorskip("astroscrappy")
    cube = sns_sjicube_1330[0, :10, :10]
    data = np.full((10, 10), 10.0, dtype=float)
    data[5, 5] = 500.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    cube.data[...] = data
    cube.mask = mask.copy()
    original_dust_masked = cube.dust_masked

    cleaned_cube = remove_cosmic_rays(
        cube,
        method="astroscrappy",
        sigma=2.0,
        max_iters=3,
        method_kwargs={"readnoise": 1.0},
    )

    assert not np.isclose(cleaned_cube.data[5, 5], 500.0)
    np.testing.assert_allclose(cleaned_cube.data[5, 5], 10.0, atol=1.0)
    np.testing.assert_array_equal(cleaned_cube.mask, mask)
    assert cleaned_cube.dust_masked == original_dust_masked


def test_remove_cosmic_rays_rsliding_kwargs_forwarded(sns_sjicube_1330, monkeypatch):
    captured = {}

    class FakeSlidingSigmaClipping:
        def __init__(self, data, **kwargs):
            captured["kwargs"] = kwargs.copy()
            cosmic_ray_mask = data > 9
            cleaned_data = np.where(cosmic_ray_mask, 5.0, data)
            self.clipped = np.ma.masked_array(cleaned_data, mask=cosmic_ray_mask)

    fake_module = types.SimpleNamespace(SlidingSigmaClipping=FakeSlidingSigmaClipping)
    monkeypatch.setitem(sys.modules, "rsliding", fake_module)

    cube = sns_sjicube_1330[0, :2, :2]
    cube.data[...] = np.array([[1.0, 2.0], [3.0, 4.0]])
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    user_kwargs = {"kernel": 5, "threads": 2}

    remove_cosmic_rays(
        cube,
        sigma=2.5,
        max_iters=7,
        method_kwargs=user_kwargs,
    )

    assert captured["kwargs"]["sigma"] == 2.5
    assert captured["kwargs"]["max_iters"] == 7
    assert captured["kwargs"]["kernel"] == 5
    assert captured["kwargs"]["threads"] == 2
    assert captured["kwargs"]["masked_array"] is True
    assert "sigma" not in user_kwargs
    assert "max_iters" not in user_kwargs
    assert "masked_array" not in user_kwargs


def test_remove_cosmic_rays_astroscrappy_kwargs_forwarded(sns_sjicube_1330, monkeypatch):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append(kwargs.copy())
        return np.zeros_like(frame, dtype=bool), frame.copy()

    fake_module = types.SimpleNamespace(detect_cosmics=fake_detect_cosmics)
    monkeypatch.setitem(sys.modules, "astroscrappy", fake_module)

    cube = sns_sjicube_1330[0, :3, :3]
    cube.data[...] = np.ones((3, 3), dtype=float)
    cube.mask = np.zeros((3, 3), dtype=bool)
    user_kwargs = {"readnoise": 4.0}

    remove_cosmic_rays(
        cube,
        method="astroscrappy",
        sigma=2.0,
        max_iters=3,
        method_kwargs=user_kwargs,
    )

    assert len(calls) == 1
    assert calls[0]["sigclip"] == 2.0
    assert calls[0]["niter"] == 3
    assert calls[0]["readnoise"] == 4.0
    assert calls[0]["verbose"] is False
    assert "sigclip" not in user_kwargs
    assert "niter" not in user_kwargs
    assert "verbose" not in user_kwargs


@pytest.mark.parametrize("dask_backed", [False, True], ids=["numpy", "dask"])
def test_remove_cosmic_rays_astroscrappy_backend(sns_sjicube_1330, monkeypatch, dask_backed):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):
        calls.append((frame.copy(), inmask.copy(), kwargs.copy()))
        return frame > 10, frame - 1

    monkeypatch.setitem(sys.modules, "astroscrappy", types.SimpleNamespace(detect_cosmics=fake_detect_cosmics))
    cube = sns_sjicube_1330[:2, :3, :4]
    data = np.arange(24, dtype=float).reshape(cube.shape)
    data[1, 2, 3] = np.nan
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0, 0] = True
    cube.data[...] = data
    cube.mask = mask.copy()
    if dask_backed:
        cube = cube.to_nddata(
            data=da.from_array(data, chunks=(1, 3, 4)),
            mask=da.from_array(mask, chunks=(1, 3, 4)),
            nddata_type=type(cube),
            extra_coords="copy",
            global_coords="copy",
        )
    cube.dust_masked = True

    cleaned = remove_cosmic_rays(cube, method="astroscrappy")

    expected_mask = mask | np.isnan(data)
    expected_frames = np.where(expected_mask, 0, data)
    assert len(calls) == 2
    for index, (frame, inmask, kwargs) in enumerate(calls):
        np.testing.assert_array_equal(frame, expected_frames[index])
        np.testing.assert_array_equal(inmask, expected_mask[index])
        assert kwargs == {"verbose": False}
    assert type(cleaned) is type(cube)
    assert isinstance(cleaned.data, np.ndarray)
    np.testing.assert_array_equal(cleaned.data, expected_frames - 1)
    np.testing.assert_array_equal(cleaned.mask, mask)
    assert cleaned.dust_masked is True
    assert cleaned.unit == cube.unit
    assert cleaned.meta["scaled"] == cube.meta["scaled"]
    assert list(cleaned.extra_coords.keys()) == list(cube.extra_coords.keys())
    assert list(cleaned.global_coords) == list(cube.global_coords)
    np.testing.assert_array_equal(cube.data, data)
    np.testing.assert_array_equal(cube.mask, mask)


@pytest.mark.parametrize("inmask_ndim", [2, 3])
def test_remove_cosmic_rays_astroscrappy_inmask_combined(sns_sjicube_1330, monkeypatch, inmask_ndim):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append(inmask.copy())
        return np.zeros_like(frame, dtype=bool), frame.copy()

    monkeypatch.setitem(sys.modules, "astroscrappy", types.SimpleNamespace(detect_cosmics=fake_detect_cosmics))
    cube = sns_sjicube_1330[:2, :3, :4]
    cube.data[...] = 1.0
    mask = np.zeros(cube.shape, dtype=bool)
    mask[0, 1, 2] = True
    cube.mask = mask.copy()
    inmask = np.zeros(cube.shape if inmask_ndim == 3 else cube.shape[1:], dtype=bool)
    inmask.flat[0] = True
    if inmask_ndim == 3:
        inmask[1, 2, 3] = True
    method_kwargs = {"inmask": inmask}

    remove_cosmic_rays(cube, method="astroscrappy", method_kwargs=method_kwargs)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls, mask | np.broadcast_to(inmask, cube.shape))
    assert method_kwargs["inmask"] is inmask


def test_remove_cosmic_rays_rsliding_rejects_dask_cube(sns_sjicube_1330):
    cube = sns_sjicube_1330[:2, :3, :4]
    cube = cube.to_nddata(data=da.ones(cube.shape, chunks=(1, 3, 4)), nddata_type=type(cube))

    with pytest.raises(ValueError, match="requires the full cube in memory"):
        remove_cosmic_rays(cube, method="rsliding")


@pytest.mark.parametrize("method", ["rsliding", "astroscrappy"])
def test_remove_cosmic_rays_missing_optional_dependency(sns_sjicube_1330, monkeypatch, method):
    def missing_module(module_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr("irispy.utils.utils.import_module", missing_module)
    cube = sns_sjicube_1330[0, :3, :3]

    with pytest.raises(ImportError, match=r"irispy-lmsal\[cosmic-rays\]") as excinfo:
        remove_cosmic_rays(cube, method=method)

    assert excinfo.value.__cause__.name == method
    assert f"method='{method}'" in str(excinfo.value)
    assert f"pip install {method}" in str(excinfo.value)
