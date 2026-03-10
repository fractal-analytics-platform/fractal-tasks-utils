from fractal_tasks_utils.segmentation._models import (
    IteratorConfiguration,
    MaskingConfiguration,
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
# MaskingConfiguration
# ---------------------------------------------------------------------------


def test_masking_configuration_defaults():
    cfg = MaskingConfiguration()
    assert cfg.mode == "Table Name"
    assert cfg.identifier is None


def test_masking_configuration_label_name_mode():
    cfg = MaskingConfiguration(mode="Label Name", identifier="nuclei")
    assert cfg.mode == "Label Name"
    assert cfg.identifier == "nuclei"


# ---------------------------------------------------------------------------
# IteratorConfiguration
# ---------------------------------------------------------------------------


def test_iterator_configuration_defaults():
    cfg = IteratorConfiguration()
    assert cfg.masking is None
    assert cfg.roi_table is None


def test_iterator_configuration_with_masking():
    cfg = IteratorConfiguration(masking=MaskingConfiguration(mode="Table Name"))
    assert isinstance(cfg.masking, MaskingConfiguration)
    assert cfg.masking.mode == "Table Name"


def test_iterator_configuration_with_roi_table():
    cfg = IteratorConfiguration(roi_table="FOV_ROI_table")
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
