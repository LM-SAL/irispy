import numpy as np

import sunpy.map

AXIS = [
    (
        "custom:pos.helioprojective.lon",
        "custom:pos.helioprojective.lat",
        "time",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
        "custom:CUSTOM",
    ),
    ("custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"),
    ("custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"),
]


def test_world_axis_physical_types_sjicube_2832(sns_sjicube_2832):
    assert np.all(sns_sjicube_2832.shape == (10, 40, 37))
    assert sns_sjicube_2832.array_axis_physical_types == AXIS


def test_world_axis_physical_types_sjicube_2796(sns_sjicube_2796):
    assert np.all(sns_sjicube_2796.shape == (62, 40, 37))
    assert sns_sjicube_2796.array_axis_physical_types == AXIS


def test_world_axis_physical_types_sjicube_1400(sns_sjicube_1400):
    assert np.all(sns_sjicube_1400.shape == (62, 40, 37))
    assert sns_sjicube_1400.array_axis_physical_types == AXIS


def test_world_axis_physical_types_sjicube_1330(sns_sjicube_1330):
    assert np.all(sns_sjicube_1330.shape == (52, 40, 37))
    assert sns_sjicube_1330.array_axis_physical_types == AXIS


def test_to_map(sns_sjicube_1330):
    # Basic smoke tests
    output = sns_sjicube_1330.to_maps(0)
    assert isinstance(output, sunpy.map.GenericMap)
    assert output.data.shape == (40, 37)
    assert output.reference_date is not None

    output = sns_sjicube_1330.to_maps([0, 2])
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 2)
    assert np.all([output.reference_date is not None for output in output])

    output = sns_sjicube_1330.to_maps(range(1, 3))
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 2)
    assert np.all([output.reference_date is not None for output in output])

    output = sns_sjicube_1330.to_maps(range(0, 12, 4))
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 3)
    assert np.all([output.reference_date is not None for output in output])

    output = sns_sjicube_1330.to_maps()
    assert isinstance(output, sunpy.map.mapsequence.MapSequence)
    assert output.data.shape == (40, 37, 52)
    assert np.all([output.reference_date is not None for output in output])
