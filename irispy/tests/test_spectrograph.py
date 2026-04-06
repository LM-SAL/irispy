import copy

import numpy as np

from astropy.io import fits

from irispy.io.spectrograph import read_spectrograph_lvl2


def test_fits_data_comparison(sns_sg_file):
    """
    Make sure the data is the same in pyfits and irispy.
    """
    iris_l2_test_raster = read_spectrograph_lvl2(sns_sg_file)
    with fits.open(sns_sg_file) as hdulist:
        spectral_window1 = hdulist[0].header["TDESC1"]
        spectral_window2 = hdulist[0].header["TDESC2"]
        spectral_window3 = hdulist[0].header["TDESC3"]
        data1 = copy.deepcopy(hdulist[1].data)
        data2 = copy.deepcopy(hdulist[2].data)
        data3 = copy.deepcopy(hdulist[3].data)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window1].data[0].data, data1)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window2].data[0].data, data2)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window3].data[0].data, data3)


def test_spectrogram_cube_remove_cosmic_rays(sns_sg_file, monkeypatch):
    captured = {}

    def fake_remove_cosmic_rays(data, *, method, mask, sigma, max_iters, method_kwargs):
        captured["method"] = method
        captured["mask"] = mask.copy()
        captured["sigma"] = sigma
        captured["max_iters"] = max_iters
        captured["method_kwargs"] = method_kwargs
        return data + 2, np.zeros_like(data, dtype=bool)

    monkeypatch.setattr("irispy.spectrograph.remove_cosmic_rays", fake_remove_cosmic_rays)

    raster = read_spectrograph_lvl2(sns_sg_file)
    key = next(iter(raster.keys()))
    cube = raster[key][0]
    cleaned_cube = cube.remove_cosmic_rays(
        method="astroscrappy",
        sigma=5.0,
        max_iters=5,
        method_kwargs={"batch_size": 16},
    )

    np.testing.assert_array_equal(cleaned_cube.data, cube.data + 2)
    np.testing.assert_array_equal(cleaned_cube.mask, cube.mask)
    assert list(cleaned_cube.extra_coords.keys()) == list(cube.extra_coords.keys())
    assert captured["method"] == "astroscrappy"
    assert captured["sigma"] == 5.0
    assert captured["max_iters"] == 5
    assert captured["method_kwargs"]["batch_size"] == 16
    np.testing.assert_array_equal(captured["mask"], cube.mask)
    # WCS is deep-copied, so check equivalence not identity
    assert dict(cleaned_cube.wcs.to_header()) == dict(cube.wcs.to_header())
    assert list(cleaned_cube.global_coords) == list(cube.global_coords)
    assert cleaned_cube.unit == cube.unit
    # meta is an NDMeta subclass that may contain arrays; compare keys only
    assert set(cleaned_cube.meta.keys()) == set(cube.meta.keys())


def test_raster_collection_remove_cosmic_rays(sns_sg_file, monkeypatch):
    monkeypatch.setattr(
        "irispy.spectrograph.remove_cosmic_rays",
        lambda data, **_kwargs: (data + 1, np.zeros_like(data, dtype=bool)),
    )

    raster = read_spectrograph_lvl2(sns_sg_file)
    key = next(iter(raster.keys()))
    cleaned_sequence = raster[key].remove_cosmic_rays(method="astroscrappy")
    cleaned_collection = raster.remove_cosmic_rays(method="astroscrappy")

    np.testing.assert_array_equal(cleaned_sequence[0].data, raster[key][0].data + 1)
    np.testing.assert_array_equal(cleaned_collection[key][0].data, raster[key][0].data + 1)

    # Collection: key order, aligned_axes mapping, and meta are preserved
    assert list(cleaned_collection.keys()) == list(raster.keys())
    assert cleaned_collection.aligned_axes == raster.aligned_axes
    assert cleaned_collection.meta == raster.meta

    # Sequence: length, per-cube WCS, and sequence-level meta are preserved
    assert len(cleaned_sequence) == len(raster[key])
    assert dict(cleaned_sequence[0].wcs.to_header()) == dict(raster[key][0].wcs.to_header())
    assert dict(cleaned_sequence.meta) == dict(raster[key].meta)
