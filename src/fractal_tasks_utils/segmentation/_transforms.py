from typing_extensions import Annotated

from fractal_tasks_utils.transforms import (
    HistogramEqualizationConfig,
    GaussianBlurConfig,
    MedianFilterConfig,
    SizeFilterConfig,
)
from pydantic import BaseModel, Field

PreProcess = Annotated[
    GaussianBlurConfig | MedianFilterConfig | HistogramEqualizationConfig,
    Field(discriminator="type"),
]

PostProcess = Annotated[
    SizeFilterConfig,
    Field(discriminator="type"),
]


class SegmentationTransformConfig(BaseModel):
    """Configuration for pre- and post-processing steps.

    Attributes:
        pre_process (list[PreProcess]): List of pre-processing steps.
        post_process (list[PostProcess]): List of post-processing steps.
    """

    pre_process: list[PreProcess] = Field(default_factory=list)
    post_process: list[PostProcess] = Field(default_factory=list)

    def to_pre_transforms(self) -> list:
        """Get the list of pre-processing transformations."""
        return [pre.to_transform() for pre in self.pre_process]

    def to_post_transforms(self) -> list:
        """Get the list of post-processing transformations."""
        return [post.to_transform() for post in self.post_process]
