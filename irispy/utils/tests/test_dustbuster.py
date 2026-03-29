from types import SimpleNamespace

import numpy as np
import pytest

import astropy.units as u

import irispy.utils as iris_utils
import irispy.utils.dustbuster as dustbuster_module
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED
from irispy.utils.dustbuster import clean_sji_dust, get_sji_dust_params

_DEFAULT = object()


def _fake_time(*_args, **_kwargs):
    return SimpleNamespace(unix_tai=1000.0)


def _read_genx_with_rows(rows):
    def _reader(_path):
        return {"SAVEGEN0": rows}

    return _reader


def _read_geny_with_ids(dust_ids):
    def _reader(_path):
        return {
            "p0": np.rec.array(
                [(np.array(dust_ids, dtype=np.int64),)],
                dtype=[("F7", object)],
            ),
        }

    return _reader


def _fake_frame_wcs():
    return SimpleNamespace(
        wcs=SimpleNamespace(
            crpix=np.array([1.0, 1.0]),
            cdelt=np.array([1.0, 1.0]),
            crval=np.array([0.0, 0.0]),
        )
    )


def _make_cube(
    data,
    *,
    exposure_s=None,
    slit_x=None,
    slit_y=None,
    basic_wcs=_DEFAULT,
    meta=None,
):
    data = np.asarray(data, dtype=float)
    n_frames = data.shape[0] if data.ndim == 3 else 1
    exposure_s = np.ones(n_frames) if exposure_s is None else np.asarray(exposure_s, dtype=float)
    slit_x = np.full(n_frames, 2.0) if slit_x is None else np.asarray(slit_x, dtype=float)
    slit_y = np.full(n_frames, 2.0) if slit_y is None else np.asarray(slit_y, dtype=float)
    if basic_wcs is _DEFAULT:
        basic_wcs = [_fake_frame_wcs() for _ in range(n_frames)] if data.ndim == 3 else _fake_frame_wcs()
    if meta is None:
        meta = {
            "DATE_OBS": "2024-01-01T00:00:00.000",
            "TDESC1": "SJI_2796",
            "SUMSPAT": 1,
            "SUMSPTRL": 1,
        }

    def _extra_coord(values, unit):
        def _pixel_to_world(_pixels, *, values=values, unit=unit):
            return values * unit

        return SimpleNamespace(wcs=SimpleNamespace(pixel_to_world=_pixel_to_world))

    return SimpleNamespace(
        data=data,
        mask=data < 0.5,
        meta=meta,
        extra_coords={
            "exposure time": _extra_coord(exposure_s, u.s),
            "slit x position": _extra_coord(slit_x, u.arcsec),
            "slit y position": _extra_coord(slit_y, u.arcsec),
        },
        basic_wcs=basic_wcs,
    )


def _write_managed_file_stubs(tmp_path):
    flat_index_path = tmp_path / "flat.genx"
    bad_pixel_path = tmp_path / "badpix.geny"
    flat_index_path.write_text("flat", encoding="ascii")
    bad_pixel_path.write_text("badpix", encoding="ascii")
    return flat_index_path, bad_pixel_path


def test_align_frame_idx_covers_short_and_long_cubes():
    np.testing.assert_array_equal(dustbuster_module._align_frame_idx(3), np.arange(3))

    frame_idx = dustbuster_module._align_frame_idx(9)

    assert len(frame_idx) <= dustbuster_module._MAX_ALIGNMENT_FRAMES
    assert frame_idx[0] == 0
    assert frame_idx[-1] == 8


def test_clean_sji_dust_replaces_2d_dust_pixel_and_clears_mask():
    cube = _make_cube(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 0.1, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    cleaned_cube = clean_sji_dust(
        cube,
        dust_ids=[2073],
        slit_center=(1.0, 0.5),
        mask_scale=1.0,
        roll_deg=0.0,
        align=False,
    )

    assert cube.data[1, 1] == 0.1
    assert cleaned_cube.data[1, 1] != cube.data[1, 1]
    assert np.isfinite(cleaned_cube.data[1, 1])
    assert not cleaned_cube.mask[1, 1]


def test_clean_sji_dust_aligns_and_uses_temporal_fill_for_3d_data():
    data = np.full((9, 4, 4), 10.0)
    slit_x = np.array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0])
    for frame_idx, dust_x in enumerate((slit_x - 1).astype(int)):
        data[frame_idx, 1, dust_x] = 0.1

    cube = _make_cube(data, slit_x=slit_x)

    cleaned_cube = clean_sji_dust(
        cube,
        dust_ids=[2073],
        slit_center=(1.0, 0.5),
        mask_scale=1.0,
        roll_deg=0.0,
        align=True,
    )

    assert cleaned_cube.data[0, 1, 1] == pytest.approx(10.0)
    assert cleaned_cube.data[1, 1, 2] == pytest.approx(10.0)
    assert not cleaned_cube.mask[0, 1, 1]
    assert not cleaned_cube.mask[1, 1, 2]


def test_clean_sji_dust_uses_global_fill_when_temporal_and_spatial_fill_fail():
    data = np.full((2, 4, 4), BAD_PIXEL_VALUE_SCALED, dtype=float)
    data[0, 1, 1] = 1.0
    data[1, 0, 0] = 7.0
    data[1, 1, 2] = 3.0

    cube = _make_cube(data, slit_x=[2.0, 3.0])

    cleaned_cube = clean_sji_dust(
        cube,
        dust_ids=[2073],
        slit_center=(1.0, 0.5),
        mask_scale=1.0,
        roll_deg=0.0,
        align=False,
    )

    assert cleaned_cube.data[0, 1, 1] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("cube", "message"),
    [
        (_make_cube(np.zeros(4)), "cube.data must have shape"),
        (_make_cube(np.zeros((2, 4, 4)), basic_wcs=None), "cube.basic_wcs is required"),
        (
            _make_cube(np.zeros((2, 4, 4)), basic_wcs=[_fake_frame_wcs()]),
            "cube.basic_wcs must contain one WCS per frame",
        ),
        (_make_cube(np.zeros((2, 4, 4)), exposure_s=[1.0]), "required per-frame extra coordinates"),
    ],
)
def test_clean_sji_dust_validates_input_shape_and_metadata(cube, message):
    with pytest.raises(ValueError, match=message):
        clean_sji_dust(
            cube,
            dust_ids=[2073],
            slit_center=(1.0, 0.5),
            mask_scale=1.0,
            roll_deg=0.0,
            align=False,
        )


def test_get_sji_dust_params_uses_sji_header_values(monkeypatch, tmp_path):
    flat_index_path, bad_pixel_path = _write_managed_file_stubs(tmp_path)

    monkeypatch.setattr(dustbuster_module, "Time", _fake_time)
    monkeypatch.setattr(
        dustbuster_module,
        "read_genx",
        _read_genx_with_rows(
            [
                {"IMG_PATH": "SJI_2796", "FILETAI": 900.0, "RECNUM": 1},
                {"IMG_PATH": "SJI_2796", "FILETAI": 1002.0, "RECNUM": 7},
            ]
        ),
    )
    monkeypatch.setattr(dustbuster_module, "read_geny", _read_geny_with_ids([11, 12]))

    with (
        dustbuster_module.data_manager.override_file("iris_sji_flat_index", str(flat_index_path)),
        dustbuster_module.data_manager.override_file("iris_sji_bad_pixel_map", str(bad_pixel_path)),
    ):
        params = get_sji_dust_params(
            date_obs="2024-01-01T00:00:00.000",
            sji_name="SJI_2796",
        )

    np.testing.assert_array_equal(params["dust_ids"], np.array([11, 12], dtype=np.int64))
    assert set(params) == {"dust_ids", "slit_center", "mask_scale", "roll_deg"}
    assert params["slit_center"] == (503.69, 502.40201)
    assert params["mask_scale"] == 0.1679
    assert params["roll_deg"] == 0.27399999


@pytest.mark.remote_data
def test_clean_sji_dust_smoke_real_sji_cube(sns_sjicube_2796):
    cube = sns_sjicube_2796[:5]

    params = get_sji_dust_params(
        date_obs=cube.meta["DATE_OBS"],
        sji_name=cube.meta["TDESC1"],
    )
    cleaned_cube = clean_sji_dust(cube, align=False, **params)

    assert cleaned_cube.data.shape == cube.data.shape
    assert cleaned_cube.basic_wcs is not None
    assert params["dust_ids"].ndim == 1
    assert params["dust_ids"].size > 0


@pytest.mark.parametrize(
    ("sji_name", "flat_rows", "message"),
    [
        ("2796", [], "Unsupported TDESC1"),
        ("SJI_9999", [], "Unsupported SJI channel"),
        ("SJI_2796", [{"IMG_PATH": "SJI_2832", "FILETAI": 1000.0, "RECNUM": 7}], "No flat-index rows matched"),
    ],
)
def test_get_sji_dust_params_validates_sji_name_and_matching_rows(monkeypatch, tmp_path, sji_name, flat_rows, message):
    flat_index_path, bad_pixel_path = _write_managed_file_stubs(tmp_path)

    monkeypatch.setattr(dustbuster_module, "Time", _fake_time)
    monkeypatch.setattr(dustbuster_module, "read_genx", _read_genx_with_rows(flat_rows))
    monkeypatch.setattr(dustbuster_module, "read_geny", _read_geny_with_ids([11, 12]))

    with (
        dustbuster_module.data_manager.override_file("iris_sji_flat_index", str(flat_index_path)),
        dustbuster_module.data_manager.override_file("iris_sji_bad_pixel_map", str(bad_pixel_path)),
        pytest.raises(ValueError, match=message),
    ):
        get_sji_dust_params(
            date_obs="2024-01-01T00:00:00.000",
            sji_name=sji_name,
        )


def test_dustbuster_public_utils_exports():
    assert iris_utils.dustbuster is dustbuster_module
    assert iris_utils.clean_sji_dust is clean_sji_dust
    assert iris_utils.get_sji_dust_params is get_sji_dust_params
