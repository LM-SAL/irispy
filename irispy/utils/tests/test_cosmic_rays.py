import sys
import types

import dask.array as da
import numpy as np
import pytest

import astropy.units as u

from irispy.io.sji import read_sji_lvl2
from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.tests.helpers import mark_saturated
from irispy.utils.constants import SATURATION_LIMIT
from irispy.utils.cosmic_rays import remove_cosmic_rays


def test_remove_cosmic_rays_rsliding(sns_sjicube_1330):
    pytest.importorskip("rsliding")
    cube = sns_sjicube_1330[0, :3, :3]
    data = np.array([[1.0, 2.0, 3.0], [4.0, 100.0, 6.0], [7.0, 8.0, 9.0]])
    mask = np.array([[False, False, False], [False, False, False], [True, False, False]])
    cube.data[...] = data
    cube.mask = mask.copy()

    cleaned_cube = remove_cosmic_rays(cube, method_kwargs={"kernel": 3, "threads": 1, "sigma": 2.5, "max_iters": 7})

    np.testing.assert_allclose(cleaned_cube.data[1, 1], 5.0, atol=2.0)
    np.testing.assert_array_equal(cleaned_cube.mask, mask)


def test_remove_cosmic_rays_astroscrappy(sns_sjicube_1330):
    pytest.importorskip("astroscrappy")
    cube = sns_sjicube_1330[0, :10, :10]
    data = np.full((10, 10), 10.0, dtype=float)
    data[5, 5] = 500.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    cube.data[...] = data
    cube.mask = mask.copy()

    cleaned_cube = remove_cosmic_rays(
        cube, method="astroscrappy", method_kwargs={"readnoise": 1.0, "sigclip": 2.0, "niter": 3}
    )

    assert not np.isclose(cleaned_cube.data[5, 5], 500.0)
    np.testing.assert_allclose(cleaned_cube.data[5, 5], 10.0, atol=1.0)
    np.testing.assert_array_equal(cleaned_cube.mask, mask)


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
    user_kwargs = {"kernel": 5, "threads": 2, "sigma": 2.5, "max_iters": 7}

    remove_cosmic_rays(cube, method_kwargs=user_kwargs)

    assert captured["kwargs"]["sigma"] == 2.5
    assert captured["kwargs"]["max_iters"] == 7
    assert captured["kwargs"]["kernel"] == 5
    assert captured["kwargs"]["threads"] == 2
    assert captured["kwargs"]["masked_array"] is True
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
    user_kwargs = {"readnoise": 4.0, "sigclip": 2.0, "niter": 3}

    remove_cosmic_rays(cube, method="astroscrappy", method_kwargs=user_kwargs)

    assert len(calls) == 1
    assert calls[0]["sigclip"] == 2.0
    assert calls[0]["niter"] == 3
    assert calls[0]["readnoise"] == 4.0
    assert calls[0]["verbose"] is False
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
    np.testing.assert_array_equal(cleaned.data, np.where(expected_mask, data, data - 1))
    np.testing.assert_array_equal(cleaned.mask, mask)
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
        return np.zeros_like(frame, dtype=bool), frame - 1

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

    cleaned = remove_cosmic_rays(cube, method="astroscrappy", method_kwargs=method_kwargs)

    assert len(calls) == 2
    kept = mask | np.broadcast_to(inmask, cube.shape)
    np.testing.assert_array_equal(calls, kept)
    np.testing.assert_array_equal(cleaned.data, np.where(kept, 1.0, 0.0))
    assert method_kwargs["inmask"] is inmask


def test_remove_cosmic_rays_does_not_share_meta_coordinates_or_mask(sns_sjicube_1330):
    pytest.importorskip("rsliding")
    cube = sns_sjicube_1330
    mask = cube.mask.copy()
    cleaned = remove_cosmic_rays(cube)
    cleaned.meta["note"] = "changed"
    cleaned.extra_coords.add("sequence", 0, np.arange(len(cleaned.data)) * u.one)
    cleaned.mask[...] = ~cleaned.mask
    assert "note" not in cube.meta
    assert "sequence" not in tuple(cube.extra_coords.keys())
    np.testing.assert_array_equal(cube.mask, mask)


def test_remove_cosmic_rays_rejects_unscaled_data(bursts_sji_1400_file):
    with pytest.raises(ValueError, match="unscaled"):
        remove_cosmic_rays(read_sji_lvl2(bursts_sji_1400_file, memmap=True))


@pytest.mark.parametrize("method", ["rsliding", "astroscrappy"])
def test_remove_cosmic_rays_keeps_masked_and_nan_samples(sns_sg_file, sns_sjicube_1330, method):
    pytest.importorskip(method)
    spectrogram = read_spectrograph_lvl2(sns_sg_file)["Si IV 1403"][0]  # masked -200 fills
    for cube in (spectrogram, sns_sjicube_1330[0]):  # NaN outside the field of view
        kept = np.asarray(cube.mask) | ~np.isfinite(cube.data)
        assert kept.any()

        cleaned = remove_cosmic_rays(cube, method=method)

        np.testing.assert_array_equal(cleaned.data[kept], cube.data[kept])


def test_remove_cosmic_rays_rsliding_sot_memmap(sot_fg_file):
    pytest.importorskip("rsliding")
    lazy = read_sji_lvl2(sot_fg_file, memmap=True)
    assert isinstance(lazy.mask, da.Array)

    cleaned = remove_cosmic_rays(lazy)

    np.testing.assert_array_equal(cleaned.data, remove_cosmic_rays(read_sji_lvl2(sot_fg_file)).data)


def test_remove_cosmic_rays_rejects_unknown_method(sns_sjicube_1330):
    with pytest.raises(ValueError, match=r"Unsupported method 'median'.*\['astroscrappy', 'rsliding'\]"):
        remove_cosmic_rays(sns_sjicube_1330[0, :3, :3], method="median")


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


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("rsliding", {"kernel": 3, "threads": 1, "sigma": 2.5, "max_iters": 7}),
        ("astroscrappy", {"readnoise": 1.0, "sigclip": 2.0, "niter": 3}),
    ],
)
def test_remove_cosmic_rays_keeps_clipped_pixels(sns_sjicube_1330, method, kwargs):
    pytest.importorskip(method)
    cube = sns_sjicube_1330[0, :10, :10]
    data = np.full((10, 10), 10.0)
    data[5, 5] = 500.0
    data[2, 2] = np.inf
    data[7, 7] = SATURATION_LIMIT.value
    cube.data[...] = data
    cube.mask = np.zeros((10, 10), dtype=bool)
    mark_saturated(cube, data == SATURATION_LIMIT.value)  # as the readers mark clipped pixels
    cleaned = remove_cosmic_rays(cube, method=method, method_kwargs=kwargs)
    assert np.isposinf(cleaned.data[2, 2])
    assert cleaned.data[7, 7] == SATURATION_LIMIT.value
    np.testing.assert_allclose(cleaned.data[5, 5], 10.0, atol=2.0)
    assert np.isfinite(np.delete(cleaned.data, 22)).all()


def test_remove_cosmic_rays_spectrogram_cube(sns_sg_file, monkeypatch):
    captured = {}

    def fake_remove_cosmic_rays(data, mask, *, method_kwargs):
        captured["mask"] = mask.copy()
        captured["method_kwargs"] = method_kwargs
        return data + 2

    monkeypatch.setattr("irispy.utils.cosmic_rays._remove_cosmic_rays_astroscrappy", fake_remove_cosmic_rays)

    raster = read_spectrograph_lvl2(sns_sg_file)
    key = next(iter(raster.keys()))
    cube = raster[key][0]
    cleaned_cube = remove_cosmic_rays(cube, method="astroscrappy", method_kwargs={"batch_size": 16})

    np.testing.assert_array_equal(cleaned_cube.data, cube.data + 2)
    np.testing.assert_array_equal(cleaned_cube.mask, cube.mask)
    assert list(cleaned_cube.extra_coords.keys()) == list(cube.extra_coords.keys())
    assert captured["method_kwargs"]["batch_size"] == 16
    np.testing.assert_array_equal(captured["mask"], cube.mask)
    array_index = (3, 20, 10)
    assert cleaned_cube.wcs.world_to_array_index(*cube.wcs.array_index_to_world(*array_index)) == array_index
    assert list(cleaned_cube.global_coords) == list(cube.global_coords)
    assert cleaned_cube.unit == cube.unit
    # meta is an NDMeta subclass that may contain arrays; compare keys only
    assert set(cleaned_cube.meta.keys()) == set(cube.meta.keys())
