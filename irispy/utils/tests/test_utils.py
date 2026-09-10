import numpy as np
import numpy.testing as np_test
import pytest

from irispy import utils
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED

data_dust = np.array(
    [
        [[-1, 2, -3, 4], [2, BAD_PIXEL_VALUE_SCALED, 5, 3], [0, 1, 2, -300]],
        [[2, BAD_PIXEL_VALUE_SCALED, 5, 1], [10, -5, 2, 2], [10, -3, 3, 0]],
    ],
)
dust_mask_expected = np.array(
    [
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, False, False],
        ],
        [[True, True, True, False], [True, True, True, True], [True, True, True, True]],
    ],
)


@pytest.mark.parametrize(
    ("test_input", "expected_output"),
    [
        ({"detector type": "FUV1"}, "FUV"),
        ({"detector type": "NUV"}, "NUV"),
        ({"detector type": "SJI"}, "SJI"),
    ],
)
def test_get_detector_type(test_input, expected_output):
    assert utils.get_detector_type(test_input) == expected_output


@pytest.mark.parametrize(("input_array", "expected_array"), [(data_dust, dust_mask_expected)])
def test_calculate_dust_mask(input_array, expected_array):
    np_test.assert_array_equal(utils.calculate_dust_mask(input_array), expected_array)


def test_calculate_dust_mask_2d():
    data = np.ones((3, 3))
    data[1, 1] = 0

    np_test.assert_array_equal(utils.calculate_dust_mask(data), np.ones((3, 3), dtype=bool))


def test_import_optional_missing_module_names_the_extra():
    with pytest.raises(ImportError, match=r"pip install 'irispy-lmsal\[rgb\]'"):
        utils.utils._import_optional("definitely_not_a_module", reason="testing", extra="rgb")


def test_import_optional_preserves_transitive_import_error(monkeypatch):
    def import_module(_):
        msg = "No module named 'transitive_dependency'"
        raise ModuleNotFoundError(msg, name="transitive_dependency")

    monkeypatch.setattr(utils.utils, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="transitive_dependency"):
        utils.utils._import_optional("optional_module", reason="testing", extra="tests")
