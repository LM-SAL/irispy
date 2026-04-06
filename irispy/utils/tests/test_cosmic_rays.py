import sys
import types

import numpy as np
import pytest

from irispy.utils.cosmic_rays import remove_cosmic_rays


def test_remove_cosmic_rays_rsliding_backend(monkeypatch):
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

    data = np.array([[1.0, 12.0], [3.0, np.nan]])
    mask = np.array([[False, False], [True, False]])
    cleaned_data, cosmic_ray_mask = remove_cosmic_rays(
        data,
        mask=mask,
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
    np.testing.assert_array_equal(cosmic_ray_mask, [[False, True], [False, False]])
    assert cleaned_data[0, 1] == pytest.approx(5.0)
    assert np.isnan(cleaned_data[1, 0])
    assert np.isnan(cleaned_data[1, 1])


def test_remove_cosmic_rays_astroscrappy_backend(monkeypatch):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):
        calls.append((frame.copy(), inmask.copy(), kwargs.copy()))
        frame_mask = frame > 10
        return frame_mask, frame - 1

    monkeypatch.setitem(sys.modules, "astroscrappy", types.SimpleNamespace(detect_cosmics=fake_detect_cosmics))

    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    data[1, 2, 3] = np.nan
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0, 0] = True

    cleaned_data, cosmic_ray_mask = remove_cosmic_rays(
        data,
        method="astroscrappy",
        mask=mask,
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
    np.testing.assert_array_equal(cosmic_ray_mask[0], calls[0][0] > 10)
    np.testing.assert_array_equal(cosmic_ray_mask[1], calls[1][0] > 10)
    assert cleaned_data[0, 0, 0] == pytest.approx(-1.0)
    assert cleaned_data[1, 2, 3] == pytest.approx(-1.0)


@pytest.mark.parametrize("method", ["rsliding", "astroscrappy"])
def test_remove_cosmic_rays_missing_optional_dependency(monkeypatch, method):
    def fake_import_optional_backend(module_name, *, method):
        msg = (
            f"{module_name} is an optional dependency required for method='{method}'. "
            f"Install it with `pip install {module_name}` or "
            "`pip install 'irispy-lmsal[cosmic-rays]'`."
        )
        raise ImportError(msg)

    monkeypatch.setattr("irispy.utils.cosmic_rays._import_optional_backend", fake_import_optional_backend)

    expected_module = "rsliding" if method == "rsliding" else "astroscrappy"
    with pytest.raises(ImportError, match=expected_module):
        remove_cosmic_rays(np.ones((3, 3)), method=method)
