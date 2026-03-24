"""Tests for the measurement module."""

import pandas as pd
import pytest
from ngio import ChannelSelectionModel, create_empty_ome_zarr
from ngio.experimental.iterators import FeatureExtractorIterator

import fractal_tasks_utils.measurement
from fractal_tasks_utils.measurement import (
    compute_measurement,
    join_tables,
    setup_measurement_iterator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ome_zarr_2d(tmp_path):
    """Minimal 2-channel 2D ome-Zarr with a derived label image."""
    zarr_path = str(tmp_path / "test.zarr")
    ome_zarr = create_empty_ome_zarr(
        store=zarr_path,
        shape=(2, 64, 64),
        pixelsize=0.5,
        channels_meta=["DAPI", "GFP"],
        axes_names=["c", "y", "x"],
    )
    ome_zarr.derive_label(name="nuclei", overwrite=True)
    return zarr_path


@pytest.fixture
def ome_zarr_with_roi_table(tmp_path):
    """Minimal 2-channel 2D ome-Zarr with a label and an image-level ROI table."""
    zarr_path = str(tmp_path / "test.zarr")
    ome_zarr = create_empty_ome_zarr(
        store=zarr_path,
        shape=(2, 64, 64),
        pixelsize=0.5,
        channels_meta=["DAPI", "GFP"],
        axes_names=["c", "y", "x"],
    )
    ome_zarr.derive_label(name="nuclei", overwrite=True)
    roi_table = ome_zarr.build_image_roi_table()
    ome_zarr.add_table(name="roi_table", table=roi_table)
    return zarr_path


def test_measurement_module_is_importable():
    assert fractal_tasks_utils.measurement is not None


def test_join_tables_single():
    table = {"label": [1, 2], "area": [10.0, 20.0], "region": ["roi_0", "roi_0"]}
    df = join_tables([table])
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "label"
    assert list(df.index) == [1, 2]
    assert list(df["area"]) == [10.0, 20.0]


def test_join_tables_multiple():
    t1 = {"label": [1, 2], "area": [10.0, 20.0], "region": ["roi_0", "roi_0"]}
    t2 = {"label": [3, 4], "area": [30.0, 40.0], "region": ["roi_1", "roi_1"]}
    df = join_tables([t1, t2])
    assert list(df.index) == [1, 2, 3, 4]
    assert list(df["area"]) == [10.0, 20.0, 30.0, 40.0]
    assert list(df["region"]) == ["roi_0", "roi_0", "roi_1", "roi_1"]


def test_join_tables_requires_at_least_one():
    with pytest.raises(ValueError):
        join_tables([])


def test_join_tables_custom_index():
    table = {"id": [10, 20], "value": [1.0, 2.0]}
    df = join_tables([table], index_key="id")
    assert df.index.name == "id"
    assert list(df.index) == [10, 20]


# ---------------------------------------------------------------------------
# setup_measurement_iterator tests
# ---------------------------------------------------------------------------


def test_setup_measurement_iterator_default(ome_zarr_2d):
    iterator = setup_measurement_iterator(ome_zarr_2d, "nuclei")
    assert isinstance(iterator, FeatureExtractorIterator)
    assert len(iterator.rois) > 0


def test_setup_measurement_iterator_channel_selection(ome_zarr_2d):
    channels = [ChannelSelectionModel(mode="label", identifier="DAPI")]
    iterator = setup_measurement_iterator(ome_zarr_2d, "nuclei", channels=channels)
    assert isinstance(iterator, FeatureExtractorIterator)
    # Iterating should yield images with only 1 channel in the last axis
    for img_chunk, _lbl_chunk, _roi in iterator.iter_as_numpy():
        assert img_chunk.shape[-1] == 1
        break


def test_setup_measurement_iterator_channels_none_includes_all(ome_zarr_2d):
    iterator = setup_measurement_iterator(ome_zarr_2d, "nuclei", channels=None)
    for img_chunk, _lbl_chunk, _roi in iterator.iter_as_numpy():
        assert img_chunk.shape[-1] == 2
        break


def test_setup_measurement_iterator_with_table(ome_zarr_with_roi_table):
    iterator = setup_measurement_iterator(
        ome_zarr_with_roi_table, "nuclei", roi_table_names=["roi_table"]
    )
    assert isinstance(iterator, FeatureExtractorIterator)
    assert len(iterator.rois) > 0


def test_setup_measurement_iterator_tables_none(ome_zarr_2d):
    iterator_no_table = setup_measurement_iterator(
        ome_zarr_2d, "nuclei", roi_table_names=None
    )
    iterator_empty = setup_measurement_iterator(
        ome_zarr_2d, "nuclei", roi_table_names=[]
    )
    assert len(iterator_no_table.rois) == len(iterator_empty.rois)


# ---------------------------------------------------------------------------
# compute_measurement tests
# ---------------------------------------------------------------------------


def test_compute_measurement_basic(ome_zarr_2d):
    iterator = setup_measurement_iterator(ome_zarr_2d, "nuclei")

    def simple_func(img, label, roi):
        return {"label": [1, 2], "area": [10.0, 20.0]}

    df = compute_measurement(measurement_func=simple_func, iterator=iterator)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "label"
    assert "area" in df.columns


def test_compute_measurement_empty_raises(ome_zarr_2d):
    """An iterator that produces no ROIs causes compute_measurement to raise."""
    # Build an iterator and force its ROI list to be empty via product with empty ROIs
    iterator = setup_measurement_iterator(ome_zarr_2d, "nuclei")
    # Manually empty the ROI list by taking the product with an empty set
    empty_iterator = iterator.product([])

    def dummy_func(img, label, roi):
        return {"label": [], "area": []}

    with pytest.raises(ValueError):
        compute_measurement(measurement_func=dummy_func, iterator=empty_iterator)
