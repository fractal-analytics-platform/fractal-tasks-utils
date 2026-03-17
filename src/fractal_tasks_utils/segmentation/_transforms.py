from typing import Annotated

from pydantic import BaseModel, Field

from fractal_tasks_utils.transforms import (
    GaussianBlurConfig,
    HistogramEqualizationConfig,
    MedianFilterConfig,
    SizeFilterConfig,
)

PreProcess = Annotated[
    GaussianBlurConfig | MedianFilterConfig | HistogramEqualizationConfig,
    Field(discriminator="type"),
]

PostProcess = Annotated[
    SizeFilterConfig,
    Field(discriminator="type"),
]


class SegmentationTransformConfig(BaseModel):
    """Configuration for pre- and post-processing steps."""

    pre_process: list[PreProcess] = Field(default_factory=list)
    """
    List of pre-processing steps to be applied before segmentation.
    """
    post_process: list[PostProcess] = Field(default_factory=list)
    """
    List of post-processing steps to be applied after segmentation.
    """

    def to_pre_transforms(self) -> list:
        """Get the list of pre-processing transformations."""
        return [pre.to_transform() for pre in self.pre_process]

    def to_post_transforms(self) -> list:
        """Get the list of post-processing transformations."""
        return [post.to_transform() for post in self.post_process]
