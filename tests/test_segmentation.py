import numpy as np
import pytest
from ngio import ChannelSelectionModel, create_empty_ome_zarr, open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage

from fractal_tasks_utils.segmentation._compute import (
    _load_masked_image,
    compute_segmentation,
    setup_segmentation_iterator,
)
from fractal_tasks_utils.segmentation._models import (
    IteratorConfig,
    MaskingConfig,
)
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from fractal_tasks_utils.transforms._transforms import (
    GaussianBlurConfig,
    GaussianBlurTransform,
    MedianFilterConfig,
    SizeFilterConfig,
    SizeFilterTransform,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CHANNELS = [ChannelSelectionModel(mode="label", identifier="DAPI")]


def _make_2d_zarr(tmp_path, name="test.zarr"):
    zarr_path = str(tmp_path / name)
    ome_zarr = create_empty_ome_zarr(
        store=zarr_path,
        shape=(2, 64, 64),
        pixelsize=0.5,
        channels_meta=["DAPI", "GFP"],
        axes_names=["c", "y", "x"],
    )
    return zarr_path, ome_zarr


@pytest.fixture
def ome_zarr_2d(tmp_path):
    """Minimal 2-channel 2D OME-Zarr."""
    zarr_path, _ = _make_2d_zarr(tmp_path)
    return zarr_path


@pytest.fixture
def ome_zarr_3d(tmp_path):
    """Minimal 2-channel 3D OME-Zarr."""
    zarr_path = str(tmp_path / "test3d.zarr")
    create_empty_ome_zarr(
        store=zarr_path,
        shape=(2, 4, 32, 32),
        pixelsize=0.5,
        z_spacing=1.0,
        channels_meta=["DAPI", "GFP"],
        axes_names=["c", "z", "y", "x"],
    )
    return zarr_path


@pytest.fixture
def ome_zarr_with_masking_label(tmp_path):
    """2D OME-Zarr with a masking label ('organoids') containing 2 non-zero regions."""
    zarr_path, ome_zarr = _make_2d_zarr(tmp_path)
    ome_zarr.derive_label(name="organoids", overwrite=True)
    lbl = ome_zarr.get_label("organoids")
    data = lbl.get_array()
    data[5:25, 5:25] = 1
    data[35:55, 35:55] = 2
    lbl.set_array(data)
    lbl.consolidate()
    return zarr_path


@pytest.fixture
def ome_zarr_with_masking_table(tmp_path):
    """2D OME-Zarr with a masking ROI table built from the 'organoids' label."""
    zarr_path, ome_zarr = _make_2d_zarr(tmp_path)
    ome_zarr.derive_label(name="organoids", overwrite=True)
    lbl = ome_zarr.get_label("organoids")
    data = lbl.get_array()
    data[5:25, 5:25] = 1
    data[35:55, 35:55] = 2
    lbl.set_array(data)
    lbl.consolidate()
    masking_table = ome_zarr.build_masking_roi_table("organoids")
    ome_zarr.add_table(name="masking_table", table=masking_table)
    return zarr_path


@pytest.fixture
def ome_zarr_with_roi_table(tmp_path):
    """2D OME-Zarr with an image-level iteration ROI table."""
    zarr_path, ome_zarr = _make_2d_zarr(tmp_path)
    roi_table = ome_zarr.build_image_roi_table()
    ome_zarr.add_table(name="roi_table", table=roi_table)
    return zarr_path


# ---------------------------------------------------------------------------
# MaskingConfig
# ---------------------------------------------------------------------------


def test_masking_configuration_defaults():
    cfg = MaskingConfig()
    assert cfg.mode == "Table Name"
    assert cfg.identifier is None


def test_masking_configuration_label_name_mode():
    cfg = MaskingConfig(mode="Label Name", identifier="nuclei")
    assert cfg.mode == "Label Name"
    assert cfg.identifier == "nuclei"


# ---------------------------------------------------------------------------
# IteratorConfig
# ---------------------------------------------------------------------------


def test_iterator_configuration_defaults():
    cfg = IteratorConfig()
    assert cfg.masking is None
    assert cfg.roi_table is None


def test_iterator_configuration_with_masking():
    cfg = IteratorConfig(masking=MaskingConfig(mode="Table Name"))
    assert isinstance(cfg.masking, MaskingConfig)
    assert cfg.masking.mode == "Table Name"


def test_iterator_configuration_with_roi_table():
    cfg = IteratorConfig(roi_table="FOV_ROI_table")
    assert cfg.roi_table == "FOV_ROI_table"
    assert cfg.masking is None


# ---------------------------------------------------------------------------
# SegmentationTransformConfig
# ---------------------------------------------------------------------------


def test_segmentation_transform_config_empty_defaults():
    cfg = SegmentationTransformConfig()
    assert cfg.pre_process == []
    assert cfg.post_process == []
    assert cfg.to_pre_transforms() == []
    assert cfg.to_post_transforms() == []


def test_segmentation_transform_config_pre_process_round_trip():
    cfg = SegmentationTransformConfig(pre_process=[GaussianBlurConfig(sigma_xy=2.0)])
    result = cfg.to_pre_transforms()
    assert len(result) == 1
    assert isinstance(result[0], GaussianBlurTransform)
    assert result[0].sigma_xy == 2.0


def test_segmentation_transform_config_post_process_round_trip():
    cfg = SegmentationTransformConfig(post_process=[SizeFilterConfig(min_size=50)])
    result = cfg.to_post_transforms()
    assert len(result) == 1
    assert isinstance(result[0], SizeFilterTransform)
    assert result[0].min_size == 50


def test_segmentation_transform_config_mixed_pre_process():
    cfg = SegmentationTransformConfig(
        pre_process=[
            GaussianBlurConfig(sigma_xy=1.0),
            MedianFilterConfig(size_xy=3),
        ]
    )
    result = cfg.to_pre_transforms()
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _load_masked_image
# ---------------------------------------------------------------------------


def test__load_masked_image_by_table_name(ome_zarr_with_masking_table, caplog):
    import logging

    ome_zarr = open_ome_zarr_container(ome_zarr_with_masking_table)
    cfg = MaskingConfig(mode="Table Name", identifier="masking_table")
    logger = logging.getLogger("test")
    masked = _load_masked_image(ome_zarr, cfg, logger)
    assert isinstance(masked, MaskedImage)


def test__load_masked_image_by_label_name(ome_zarr_with_masking_label):
    import logging

    ome_zarr = open_ome_zarr_container(ome_zarr_with_masking_label)
    cfg = MaskingConfig(mode="Label Name", identifier="organoids")
    logger = logging.getLogger("test")
    masked = _load_masked_image(ome_zarr, cfg, logger)
    assert isinstance(masked, MaskedImage)


# ---------------------------------------------------------------------------
# setup_segmentation_iterator
# ---------------------------------------------------------------------------


def test_setup_segmentation_iterator_2d(ome_zarr_2d):
    iterator = setup_segmentation_iterator(ome_zarr_2d, channels=_CHANNELS)
    assert isinstance(iterator, SegmentationIterator)
    assert len(iterator.rois) > 0


def test_setup_segmentation_iterator_3d(ome_zarr_3d):
    iterator = setup_segmentation_iterator(ome_zarr_3d, channels=_CHANNELS)
    assert isinstance(iterator, SegmentationIterator)
    assert len(iterator.rois) > 0


def test_setup_segmentation_iterator_masked_table(ome_zarr_with_masking_table):
    mc = MaskingConfig(mode="Table Name", identifier="masking_table")
    ic = IteratorConfig(masking=mc)
    iterator = setup_segmentation_iterator(
        ome_zarr_with_masking_table, channels=_CHANNELS, iterator_configuration=ic
    )
    assert isinstance(iterator, MaskedSegmentationIterator)
    assert len(iterator.rois) > 0


def test_setup_segmentation_iterator_masked_label(ome_zarr_with_masking_label):
    mc = MaskingConfig(mode="Label Name", identifier="organoids")
    ic = IteratorConfig(masking=mc)
    iterator = setup_segmentation_iterator(
        ome_zarr_with_masking_label, channels=_CHANNELS, iterator_configuration=ic
    )
    assert isinstance(iterator, MaskedSegmentationIterator)
    assert len(iterator.rois) > 0


def test_setup_segmentation_iterator_with_roi_table(ome_zarr_with_roi_table):
    ic = IteratorConfig(roi_table="roi_table")
    iterator = setup_segmentation_iterator(
        ome_zarr_with_roi_table, channels=_CHANNELS, iterator_configuration=ic
    )
    assert isinstance(iterator, SegmentationIterator)
    assert len(iterator.rois) > 0


def test_setup_segmentation_iterator_default_configs(ome_zarr_2d):
    """Passing None for optional configs should not raise."""
    iterator = setup_segmentation_iterator(
        ome_zarr_2d,
        channels=_CHANNELS,
        iterator_configuration=None,
        segmentation_transform_config=None,
    )
    assert isinstance(iterator, SegmentationIterator)


def test_setup_segmentation_iterator_custom_label_name(ome_zarr_2d):
    setup_segmentation_iterator(
        ome_zarr_2d, channels=_CHANNELS, output_label_name="my_label"
    )
    ome_zarr = open_ome_zarr_container(ome_zarr_2d)
    assert "my_label" in ome_zarr.list_labels()


# ---------------------------------------------------------------------------
# compute_segmentation
# ---------------------------------------------------------------------------


def test_compute_segmentation_basic(ome_zarr_2d):
    """Segmentation function is called and results are written without error."""
    iterator = setup_segmentation_iterator(ome_zarr_2d, channels=_CHANNELS)
    call_count = [0]

    def seg_func(img):
        call_count[0] += 1
        return np.zeros_like(img)

    compute_segmentation(segmentation_func=seg_func, iterator=iterator)
    assert call_count[0] == len(iterator.rois)


def test_compute_segmentation_label_uniqueness(ome_zarr_with_masking_table):
    """Labels from different ROIs must be offset so they don't collide."""
    mc = MaskingConfig(mode="Table Name", identifier="masking_table")
    ic = IteratorConfig(masking=mc)
    iterator = setup_segmentation_iterator(
        ome_zarr_with_masking_table, channels=_CHANNELS, iterator_configuration=ic
    )
    # Each ROI returns labels 1 and 2; with 2 ROIs the second should be offset to 3, 4
    call_count = [0]

    def seg_func(img):
        call_count[0] += 1
        result = np.zeros_like(img)
        result[..., 2:5, 2:5] = 1
        result[..., 6:9, 6:9] = 2
        return result

    compute_segmentation(segmentation_func=seg_func, iterator=iterator)
    assert call_count[0] == 2

    ome_zarr = open_ome_zarr_container(ome_zarr_with_masking_table)
    written = ome_zarr.get_label("segmentation").get_array()
    non_zero = sorted(np.unique(written[written > 0]))
    # Labels should be disjoint across the two ROIs (1, 2 from first; 3, 4 from second)
    assert len(non_zero) == len(set(non_zero))


def test_compute_segmentation_zero_background_preserved(ome_zarr_2d):
    """Zero-valued (background) pixels must remain 0 after the label offset."""
    iterator = setup_segmentation_iterator(ome_zarr_2d, channels=_CHANNELS)

    def seg_func(img):
        result = np.zeros_like(img)
        result[..., 10:20, 10:20] = 3
        return result

    compute_segmentation(segmentation_func=seg_func, iterator=iterator)
    ome_zarr = open_ome_zarr_container(ome_zarr_2d)
    written = ome_zarr.get_label("segmentation").get_array()
    assert (written == 0).any()
