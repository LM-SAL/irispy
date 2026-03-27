from types import SimpleNamespace

import numpy as np
import irispy.utils.dustbuster as dustbuster_module
import irispy.utils as iris_utils
from irispy.utils.dustbuster import dustbuster_sji_cube, get_sji_dust_metadata_from_ssw


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


def test_dustbuster_sji_cube_supports_sji_header_fallbacks_and_2d_data():
    cube = _FakeCube()

    cleaned_cube, bad_pixel_indices, replacement_values, applied_shift = dustbuster_sji_cube(
        cube,
        bad_pixel_addresses=[5],
        slit_center_mask=(1.0, 1.0),
        mask_plate_scale=1.0,
        roll_deg=0.0,
        mask_shape=(4, 4),
        manual_offset=(0.0, 0.0),
        align_mask=False,
        spatial_window=3,
    )

    assert applied_shift == (0, 0)
    np.testing.assert_array_equal(bad_pixel_indices, np.array([[0, 1, 1]]))
    np.testing.assert_allclose(replacement_values, np.array([6.0]))
    np.testing.assert_allclose(cleaned_cube.data[1, 1], 6.0)
    assert not cleaned_cube.mask[1, 1]


def test_get_sji_dust_metadata_from_ssw_uses_sji_header_summing_factor(monkeypatch):
    cube = _FakeCube()

    class _FakeTime:
        def __init__(self, value, format=None, scale=None):
            self.unix_tai = 1000.0

    monkeypatch.setattr(dustbuster_module, "Time", _FakeTime)

    metadata = get_sji_dust_metadata_from_ssw(
        cube,
        flat_genx_path="flat.genx",
        badpix_geny_path="badpix.geny",
        read_genx=lambda path: {
            "SAVEGEN0": [
                {"IMG_PATH": "SJI_2796", "FILETAI": 900.0, "RECNUM": 1},
                {"IMG_PATH": "SJI_2796", "FILETAI": 1002.0, "RECNUM": 7},
            ],
        },
        read_geny=lambda path: {"p0": {"F7": np.array([11, 12], dtype=np.int64)}},
    )

    np.testing.assert_array_equal(metadata["bad_pixel_addresses"], np.array([11, 12], dtype=np.int64))
    assert metadata["sumspat"] == 1
    assert metadata["sumsptrl"] == 1
    assert metadata["recnum"] == 7


def test_dustbuster_public_utils_exports():
    assert iris_utils.dustbuster is dustbuster_module
    assert iris_utils.dustbuster_sji_cube is dustbuster_sji_cube
    assert iris_utils.get_sji_dust_metadata_from_ssw is get_sji_dust_metadata_from_ssw
