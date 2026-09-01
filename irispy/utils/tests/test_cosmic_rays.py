import types

import dask.array as da
import numpy as np
import pytest

from irispy.utils.cosmic_rays import remove_cosmic_rays


class FakeCube:
    def __init__(self, data, mask=None):
        self.data = data
        self.mask = mask

    def to_nddata(self, **kwargs):
        return types.SimpleNamespace(data=kwargs["data"], mask=self.mask)


@pytest.fixture
def mock_cosmic_ray_backend(monkeypatch):
    """
    Return a helper that patches ``_import_optional`` to a fake module.
    """

    def _patch(fake_module):
        def fake_import_optional(_module_name, *, reason, extra):
            _ = reason, extra
            return fake_module

        monkeypatch.setattr(
            "irispy.utils.cosmic_rays._import_optional",
            fake_import_optional,
        )

    return _patch


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


def test_remove_cosmic_rays_rsliding_kwargs_forwarded(sns_sjicube_1330, mock_cosmic_ray_backend):
    captured = {}

    class FakeSlidingSigmaClipping:
        def __init__(self, data, **kwargs):
            captured["kwargs"] = kwargs.copy()
            cosmic_ray_mask = data > 9
            cleaned_data = np.where(cosmic_ray_mask, 5.0, data)
            self.clipped = np.ma.masked_array(cleaned_data, mask=cosmic_ray_mask)

    fake_module = types.SimpleNamespace(SlidingSigmaClipping=FakeSlidingSigmaClipping)
    mock_cosmic_ray_backend(fake_module)

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


def test_remove_cosmic_rays_astroscrappy_kwargs_forwarded(sns_sjicube_1330, mock_cosmic_ray_backend):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append(kwargs.copy())
        return np.zeros_like(frame, dtype=bool), frame.copy()

    fake_module = types.SimpleNamespace(detect_cosmics=fake_detect_cosmics)
    mock_cosmic_ray_backend(fake_module)

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


def test_remove_cosmic_rays_astroscrappy_backend(sns_sjicube_1330, mock_cosmic_ray_backend):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):
        calls.append((frame.copy(), inmask.copy(), kwargs.copy()))
        frame_mask = frame > 10
        return frame_mask, frame - 1

    fake_module = types.SimpleNamespace(detect_cosmics=fake_detect_cosmics)
    mock_cosmic_ray_backend(fake_module)

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
    np.testing.assert_allclose(calls[1][0][2, 3], 0.0)
    np.testing.assert_allclose(cleaned_cube.data[0, 0, 0], -1.0)
    np.testing.assert_allclose(cleaned_cube.data[1, 2, 3], -1.0)


def test_remove_cosmic_rays_astroscrappy_inmask_combined(sns_sjicube_1330, mock_cosmic_ray_backend):
    """
    Inmask from method_kwargs must be OR-merged with mask before each detect_cosmics
    call.
    """
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append(inmask.copy())
        return np.zeros_like(frame, dtype=bool), frame.copy()

    fake_module = types.SimpleNamespace(detect_cosmics=fake_detect_cosmics)
    mock_cosmic_ray_backend(fake_module)

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


def test_remove_cosmic_rays_astroscrappy_broadcasts_2d_inmask(sns_sjicube_1330, mock_cosmic_ray_backend):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append(inmask.copy())
        return np.zeros_like(frame, dtype=bool), frame.copy()

    fake_module = types.SimpleNamespace(detect_cosmics=fake_detect_cosmics)
    mock_cosmic_ray_backend(fake_module)

    cube = sns_sjicube_1330[:2, :3, :4]
    cube.data[...] = np.ones((2, 3, 4), dtype=float)
    cube.mask = np.zeros(cube.data.shape, dtype=bool)
    inmask = np.zeros((3, 4), dtype=bool)
    inmask[0, 0] = True

    remove_cosmic_rays(cube, method="astroscrappy", method_kwargs={"inmask": inmask})

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], inmask)
    np.testing.assert_array_equal(calls[1], inmask)


def test_remove_cosmic_rays_astroscrappy_handles_dask_frames(mock_cosmic_ray_backend):
    calls = []

    def fake_detect_cosmics(frame, *, inmask=None, **kwargs):  # NOQA: ARG001
        calls.append((frame.copy(), inmask.copy()))
        return frame > 10, frame - 1

    fake_module = types.SimpleNamespace(detect_cosmics=fake_detect_cosmics)
    mock_cosmic_ray_backend(fake_module)

    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    data[1, 2, 3] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0, 0] = True
    cube = FakeCube(da.from_array(data, chunks=(1, 3, 4)), da.from_array(mask, chunks=(1, 3, 4)))

    cleaned_cube = remove_cosmic_rays(cube, method="astroscrappy")

    assert len(calls) == 2
    assert calls[0][1][0, 0]
    assert calls[1][1][2, 3]
    assert calls[1][0][2, 3] == pytest.approx(0.0)
    assert cleaned_cube.data[1, 2, 3] == pytest.approx(-1.0)


def test_remove_cosmic_rays_rsliding_rejects_dask_cube():
    cube = FakeCube(da.from_array(np.ones((2, 3, 4)), chunks=(1, 3, 4)))

    with pytest.raises(ValueError, match="requires the full cube in memory"):
        remove_cosmic_rays(cube, method="rsliding")


@pytest.mark.parametrize("method", ["rsliding", "astroscrappy"])
def test_remove_cosmic_rays_missing_optional_dependency(sns_sjicube_1330, monkeypatch, method):
    def fake_import_optional(module_name, *, reason, extra):
        msg = (
            f"{module_name} is an optional dependency required for {reason}. "
            f"Install it with `pip install {module_name}` or "
            f"`pip install 'irispy-lmsal[{extra}]'`."
        )
        raise ImportError(msg)

    monkeypatch.setattr("irispy.utils.cosmic_rays._import_optional", fake_import_optional)

    cube = sns_sjicube_1330[0, :3, :3]
    cube.data[...] = np.ones((3, 3), dtype=float)
    cube.mask = np.zeros(cube.data.shape, dtype=bool)

    expected_module = "rsliding" if method == "rsliding" else "astroscrappy"
    with pytest.raises(ImportError, match=expected_module):
        remove_cosmic_rays(cube, method=method)
