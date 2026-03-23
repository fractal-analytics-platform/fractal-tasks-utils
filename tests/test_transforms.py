import numpy as np
import pytest
from pydantic import ValidationError

from fractal_tasks_utils.transforms._transforms import (
    GaussianBlurConfig,
    GaussianBlurTransform,
    HistogramEqualizationConfig,
    HistogramEqualizationTransform,
    MedianFilterConfig,
    MedianFilterTransform,
    SizeFilterConfig,
    SizeFilterTransform,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_2d():
    rng = np.random.default_rng(42)
    return rng.random((32, 32), dtype=np.float32)


@pytest.fixture
def image_3d():
    rng = np.random.default_rng(42)
    return rng.random((4, 32, 32), dtype=np.float32)


@pytest.fixture
def image_4d():
    rng = np.random.default_rng(42)
    return rng.random((2, 4, 32, 32), dtype=np.float32)


@pytest.fixture
def label_2d():
    """10x10 label array: label 1 = 25px region, label 2 = 3px region."""
    arr = np.zeros((10, 10), dtype=np.int32)
    arr[1:6, 1:6] = 1  # 5x5 = 25 pixels
    arr[8, 8] = 2
    arr[8, 9] = 2
    arr[9, 8] = 2  # 3 pixels
    return arr


# ---------------------------------------------------------------------------
# GaussianBlurTransform
# ---------------------------------------------------------------------------


def test_gaussian_blur_axes_yx(image_2d):
    result = GaussianBlurTransform(sigma_xy=1.0).apply(image_2d, axes=("y", "x"))
    assert result.shape == image_2d.shape
    assert not np.array_equal(result, image_2d)


def test_gaussian_blur_axes_zyx(image_3d):
    result = GaussianBlurTransform(sigma_xy=1.0, sigma_z=1.0).apply(
        image_3d, axes=("z", "y", "x")
    )
    assert result.shape == image_3d.shape
    assert not np.array_equal(result, image_3d)


def test_gaussian_blur_axes_czyx(image_4d):
    result = GaussianBlurTransform(sigma_xy=1.0, sigma_z=1.0).apply(
        image_4d, axes=("c", "z", "y", "x")
    )
    assert result.shape == image_4d.shape


def test_gaussian_blur_axes_cyx():
    rng = np.random.default_rng(0)
    image_cyx = rng.random((3, 16, 16)).astype(np.float32)
    result = GaussianBlurTransform(sigma_xy=1.0).apply(image_cyx, axes=("c", "y", "x"))
    assert result.shape == image_cyx.shape
    assert not np.array_equal(result, image_cyx)


def test_gaussian_blur_preserves_range():
    rng = np.random.default_rng(0)
    image = (rng.random((16, 16)) * 100 + 100).astype(np.uint16)
    result = GaussianBlurTransform(sigma_xy=1.0).apply(image, axes=("y", "x"))
    # preserve_range=True: output stays in original scale, not normalized to [0, 1]
    assert result.max() > 1.0


def test_gaussian_blur_mismatched_axes_raises():
    arr = np.zeros((16, 16))
    with pytest.raises(ValueError):
        GaussianBlurTransform().apply(arr, axes=("z", "y", "x"))


# ---------------------------------------------------------------------------
# GaussianBlurConfig
# ---------------------------------------------------------------------------


def test_gaussian_blur_config_to_transform():
    t = GaussianBlurConfig(sigma_xy=3.0, sigma_z=1.5).to_transform()
    assert isinstance(t, GaussianBlurTransform)
    assert t.sigma_xy == 3.0
    assert t.sigma_z == 1.5


def test_gaussian_blur_config_invalid_sigma():
    with pytest.raises(ValidationError):
        GaussianBlurConfig(sigma_xy=0)
    with pytest.raises(ValidationError):
        GaussianBlurConfig(sigma_xy=-1.0)


# ---------------------------------------------------------------------------
# MedianFilterTransform
# ---------------------------------------------------------------------------


def test_median_filter_2d(image_2d):
    result = MedianFilterTransform(size_xy=3).apply(image_2d, axes=("y", "x"))
    assert result.shape == image_2d.shape


def test_median_filter_axes_zyx(image_3d):
    result = MedianFilterTransform(size_xy=3, size_z=2).apply(
        image_3d, axes=("z", "y", "x")
    )
    assert result.shape == image_3d.shape


def test_median_filter_axes_czyx(image_4d):
    result = MedianFilterTransform(size_xy=3, size_z=2).apply(
        image_4d, axes=("c", "z", "y", "x")
    )
    assert result.shape == image_4d.shape


def test_median_filter_axes_cyx():
    rng = np.random.default_rng(0)
    image_cyx = rng.random((3, 16, 16)).astype(np.float32)
    result = MedianFilterTransform(size_xy=3).apply(image_cyx, axes=("c", "y", "x"))
    assert result.shape == image_cyx.shape


def test_median_filter_axes_zyx_no_size_z(image_3d):
    # size_z=None with z axis present: falls back to 1 for z dimension
    result = MedianFilterTransform(size_xy=3).apply(image_3d, axes=("z", "y", "x"))
    assert result.shape == image_3d.shape


def test_median_filter_mismatched_axes_raises():
    arr = np.zeros((16, 16))
    with pytest.raises(ValueError):
        MedianFilterTransform().apply(arr, axes=("z", "y", "x"))


# ---------------------------------------------------------------------------
# MedianFilterConfig
# ---------------------------------------------------------------------------


def test_median_filter_config_to_transform():
    t = MedianFilterConfig(size_xy=5, size_z=2).to_transform()
    assert isinstance(t, MedianFilterTransform)
    assert t.size_xy == 5
    assert t.size_z == 2


def test_median_filter_config_invalid_size():
    with pytest.raises(ValidationError):
        MedianFilterConfig(size_xy=0)


# ---------------------------------------------------------------------------
# HistogramEqualizationTransform
# ---------------------------------------------------------------------------


def test_histogram_equalization_2d(image_2d):
    result = HistogramEqualizationTransform().apply(image_2d, axes=("y", "x"))
    assert result.shape == image_2d.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_histogram_equalization_3d(image_3d):
    result = HistogramEqualizationTransform().apply(image_3d, axes=("z", "y", "x"))
    assert result.shape == image_3d.shape


def test_histogram_equalization_axes_czyx(image_4d):
    result = HistogramEqualizationTransform().apply(image_4d, axes=("c", "z", "y", "x"))
    assert result.shape == image_4d.shape


def test_histogram_equalization_axes_cyx():
    rng = np.random.default_rng(0)
    image_cyx = rng.random((2, 16, 16))
    result = HistogramEqualizationTransform().apply(image_cyx, axes=("c", "y", "x"))
    assert result.shape == image_cyx.shape


def test_histogram_equalization_mismatched_axes_raises():
    arr = np.zeros((16, 16))
    with pytest.raises(ValueError):
        HistogramEqualizationTransform().apply(arr, axes=("z", "y", "x"))


# ---------------------------------------------------------------------------
# HistogramEqualizationConfig
# ---------------------------------------------------------------------------


def test_histogram_equalization_config_to_transform():
    t = HistogramEqualizationConfig(clip_limit=0.05, nbins=128).to_transform()
    assert isinstance(t, HistogramEqualizationTransform)
    assert t.clip_limit == 0.05
    assert t.nbins == 128


def test_histogram_equalization_config_clip_limit_bounds():
    with pytest.raises(ValidationError):
        HistogramEqualizationConfig(clip_limit=-0.1)
    with pytest.raises(ValidationError):
        HistogramEqualizationConfig(clip_limit=1.1)
    # Boundary values must be accepted
    HistogramEqualizationConfig(clip_limit=0.0)
    HistogramEqualizationConfig(clip_limit=1.0)


# ---------------------------------------------------------------------------
# SizeFilterTransform
# ---------------------------------------------------------------------------


def test_size_filter_removes_small_objects(label_2d):
    result = SizeFilterTransform(min_size=5).apply(label_2d)
    assert 1 in result  # 25px object survives
    assert 2 not in result  # 3px object is removed


def test_size_filter_min_size_zero_keeps_all(label_2d):
    result = SizeFilterTransform(min_size=0).apply(label_2d)
    assert 1 in result
    assert 2 in result


# ---------------------------------------------------------------------------
# SizeFilterConfig
# ---------------------------------------------------------------------------


def test_size_filter_config_to_transform():
    t = SizeFilterConfig(min_size=10).to_transform()
    assert isinstance(t, SizeFilterTransform)
    assert t.min_size == 10


def test_size_filter_config_invalid_min_size():
    with pytest.raises(ValidationError):
        SizeFilterConfig(min_size=-1)
