import sys
import types

import numpy as np
import pytest

from irispy.utils.cosmic_rays import remove_cosmic_rays


def test_remove_cosmic_rays_rsliding_backend(sns_sjicube_1330, monkeypatch):
    captured = {}

    class FakeSlidingSigmaClipping:
        def __init__(self, data, **kwargs):
            captured["data"] = data.copy()
            captured["kwargs"] = kwargs
            cosmic_ray_mask = data > 9
            cleaned_data = np.where(cosmic_ray_mask, 5.0, data)
            self.clipped = np.ma.masked_array(cleaned_data, mask=cosmic_ray_mask)

    monkeypatch.setitem(
        sys.modules,
        "rsliding",
        types.SimpleNamespace(SlidingSigmaClipping=FakeSlidingSigmaClipping),
    )

    cube = sns_sjicube_1330[0, :2, :2]
    data = np.array([[1.0, 12.0], [3.0, np.nan]])
    mask = np.array([[False, False], [True, False]])
    cube.data[...] = data
    cube.mask = mask.copy()
    original_dust_masked = cube.dust_masked
    cleaned_cube = remove_cosmic_rays(
        cube,
        sigma=2.5,
        max_iters=7,
        method_kwargs={"kernel": 5, "threads": 2},
    )

    np.testing.assert_array_equal(np.isnan(captured["data"]), [[False, False], [True, True]])
    assert captured["kwargs"]["kernel"] == 5
    assert captured["kwargs"]["threads"] == 2
    assert captured["kwargs"]["sigma"] == 2.5
    assert captured["kwargs"]["max_iters"] == 7
    assert captured["kwargs"]["masked_array"] is True
    assert cleaned_cube.data[0, 1] == pytest.approx(5.0)
    assert np.isnan(cleaned_cube.data[1, 0])
    assert np.isnan(cleaned_cube.data[1, 1])
    np.testing.assert_array_equal(cleaned_cube.mask, mask)
    assert cleaned_cube.dust_masked == original_dust_masked


def test_remove_cosmic_rays_rsliding_backend_defaults(sns_sjicube_1330, monkeypatch):
    captured = {}

    class FakeSlidingSigmaClipping:
        def __init__(self, data, **kwargs):
            captured["data"] = data.copy()
            captured["kwargs"] = kwargs
            self.clipped = np.ma.masked_array(data.copy(), mask=np.zeros_like(data, dtype=bool))

    monkeypatch.setitem(
        sys.modules,
        "rsliding",
        types.SimpleNamespace(SlidingSigmaClipping=FakeSlidingSigmaClipping),
    )

    cube = sns_sjicube_1330[0, :2, :2]
    cube.data[...] = np.array([[1.0, 2.0], [3.0, 4.0]])
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    remove_cosmic_rays(cube, method="rsliding")

    assert captured["kwargs"]["sigma"] == 3.0
    assert captured["kwargs"]["max_iters"] == 5
    for key in ("kernel", "threads", "center_choice", "borders"):
        assert key in captured["kwargs"], f"Default key {key!r} missing from backend kwargs"


def test_remove_cosmic_rays_astroscrappy_backend(sns_sjicube_1330, monkeypatch):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):
        calls.append((frame.copy(), inmask.copy(), kwargs.copy()))
        frame_mask = frame > 10
        return frame_mask, frame - 1

    monkeypatch.setitem(sys.modules, "astroscrappy", types.SimpleNamespace(detect_cosmics=fake_detect_cosmics))

    cube = sns_sjicube_1330[:2, :3, :4]
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    data[1, 2, 3] = np.nan
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0, 0] = True
    cube.data[...] = data
    cube.mask = mask.copy()

    cleaned_cube = remove_cosmic_rays(
        cube,
        method="astroscrappy",
        sigma=2.0,
        max_iters=3,
        method_kwargs={"readnoise": 4.0},
    )

    assert len(calls) == 2
    assert calls[0][2]["sigclip"] == 2.0
    assert calls[0][2]["niter"] == 3
    assert calls[0][2]["readnoise"] == 4.0
    assert calls[0][2]["verbose"] is False
    assert calls[0][1][0, 0]
    assert calls[1][1][2, 3]
    assert calls[1][0][2, 3] == pytest.approx(0.0)
    assert cleaned_cube.data[0, 0, 0] == pytest.approx(-1.0)
    assert cleaned_cube.data[1, 2, 3] == pytest.approx(-1.0)


def test_remove_cosmic_rays_astroscrappy_inmask_combined(sns_sjicube_1330, monkeypatch):
    """
    Inmask from method_kwargs must be OR-merged with mask before each detect_cosmics
    call.
    """
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append(inmask.copy())
        return np.zeros_like(frame, dtype=bool), frame.copy()

    monkeypatch.setitem(
        sys.modules,
        "astroscrappy",
        types.SimpleNamespace(detect_cosmics=fake_detect_cosmics),
    )

    cube = sns_sjicube_1330[:2, :3, :4]
    data = np.ones((2, 3, 4), dtype=float)
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, 1, 2] = True
    inmask = np.zeros((2, 3, 4), dtype=bool)
    inmask[0, 0, 0] = True
    inmask[1, 2, 3] = True
    cube.data[...] = data
    cube.mask = mask.copy()

    remove_cosmic_rays(cube, method="astroscrappy", method_kwargs={"inmask": inmask})

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], mask[0] | inmask[0])
    np.testing.assert_array_equal(calls[1], mask[1] | inmask[1])


@pytest.mark.parametrize("method", ["rsliding", "astroscrappy"])
def test_remove_cosmic_rays_missing_optional_dependency(sns_sjicube_1330, monkeypatch, method):
    def fake_import_optional_backend(module_name, *, method):
        msg = (
            f"{module_name} is an optional dependency required for method='{method}'. "
            f"Install it with `pip install {module_name}` or "
            "`pip install 'irispy-lmsal[cosmic-rays]'`."
        )
        raise ImportError(msg)

    monkeypatch.setattr("irispy.utils.cosmic_rays._import_optional_backend", fake_import_optional_backend)

    cube = sns_sjicube_1330[0, :3, :3]
    cube.data[...] = np.ones((3, 3), dtype=float)
    cube.mask = np.zeros(cube.data.shape, dtype=bool)

    expected_module = "rsliding" if method == "rsliding" else "astroscrappy"
    with pytest.raises(ImportError, match=expected_module):
        remove_cosmic_rays(cube, method=method)
