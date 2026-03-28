from types import SimpleNamespace

import numpy as np
import irispy.utils.dustbuster as dustbuster_module
import irispy.utils as iris_utils
from irispy.utils.dustbuster import clean_sji_dust, get_sji_dust_params


class _FakeCoord:
    def __init__(self, values):
        self.value = np.asarray(values, dtype=float)


class _FakeMeta(dict):
    @property
    def spatial_summing_factor(self):
        return self.get("SUMSPAT")

    @property
    def spectral_summing_factor(self):
        return None


class _FakeCube:
    def __init__(self):
        self.data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 0.1, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        )
        self.mask = self.data < 0.5
        self.meta = _FakeMeta(
            {
                "DATE_OBS": "2024-01-01T00:00:00.000",
                "TDESC1": "SJI_2796",
                "SUMSPAT": 1,
                "SUMSPTRL": 1,
            },
        )
        self.extra_coords = object()
        self.basic_wcs = SimpleNamespace(
            wcs=SimpleNamespace(
                crpix=np.array([1.0, 1.0]),
                cdelt=np.array([1.0, 1.0]),
                crval=np.array([0.0, 0.0]),
            ),
        )

    def axis_world_coords(self, name, *, wcs=None):
        assert wcs is self.extra_coords
        values = {
            "exposure time": [1.0],
            "slit x position": [2.0],
            "slit y position": [2.0],
        }
        return (_FakeCoord(values[name]),)

    def __deepcopy__(self, memo):
        new_cube = type(self).__new__(type(self))
        memo[id(self)] = new_cube
        new_cube.data = self.data.copy()
        new_cube.mask = self.mask.copy()
        new_cube.meta = _FakeMeta(dict(self.meta))
        new_cube.extra_coords = self.extra_coords
        new_cube.basic_wcs = self.basic_wcs
        return new_cube


def test_clean_sji_dust_supports_sji_header_fallbacks_and_2d_data():
    cube = _FakeCube()

    cleaned_cube = clean_sji_dust(
        cube,
        dust_ids=[2073],
        slit_center=(1.0, 1.0),
        mask_scale=1.0,
        roll_deg=0.0,
        align=False,
    )

    assert cube.data[1, 1] == 0.1
    assert cleaned_cube.data[1, 1] != cube.data[1, 1]
    assert np.isfinite(cleaned_cube.data[1, 1])
    assert not cleaned_cube.mask[1, 1]


def test_get_sji_dust_params_uses_sji_header_values(monkeypatch, tmp_path):
    cube = _FakeCube()

    class _FakeTime:
        def __init__(self, value, format=None, scale=None):
            self.unix_tai = 1000.0

    flat_index_path = tmp_path / "flat.genx"
    bad_pixel_path = tmp_path / "badpix.geny"
    flat_index_path.write_text("flat", encoding="ascii")
    bad_pixel_path.write_text("badpix", encoding="ascii")

    monkeypatch.setattr(dustbuster_module, "Time", _FakeTime)
    monkeypatch.setattr(
        dustbuster_module,
        "read_genx",
        lambda path: {
            "SAVEGEN0": [
                {"IMG_PATH": "SJI_2796", "FILETAI": 900.0, "RECNUM": 1},
                {"IMG_PATH": "SJI_2796", "FILETAI": 1002.0, "RECNUM": 7},
            ],
        },
    )
    monkeypatch.setattr(
        dustbuster_module,
        "read_geny",
        lambda path: {"p0": {"F7": np.array([11, 12], dtype=np.int64)}},
    )

    with dustbuster_module.data_manager.override_file("iris_sji_flat_index", str(flat_index_path)):
        with dustbuster_module.data_manager.override_file("iris_sji_bad_pixel_map", str(bad_pixel_path)):
            params = get_sji_dust_params(cube)

    np.testing.assert_array_equal(params["dust_ids"], np.array([11, 12], dtype=np.int64))
    assert set(params) == {"dust_ids", "slit_center", "mask_scale", "roll_deg"}
    assert params["slit_center"] == (503.69, 502.40201)
    assert params["mask_scale"] == 0.1679
    assert params["roll_deg"] == 0.27399999


def test_dustbuster_public_utils_exports():
    assert iris_utils.dustbuster is dustbuster_module
    assert iris_utils.clean_sji_dust is clean_sji_dust
    assert iris_utils.get_sji_dust_params is get_sji_dust_params
