"""Tests for the measurement module."""

import pandas as pd
import pytest

import fractal_tasks_utils.measurement
from fractal_tasks_utils.measurement import join_tables


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
