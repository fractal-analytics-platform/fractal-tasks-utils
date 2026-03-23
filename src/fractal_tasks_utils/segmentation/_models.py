"""Pydantic models for advanced iterator configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NoMaskingConfig(BaseModel):
    """No masking configuration."""

    mode: Literal["No Masking"] = "No Masking"
    """
    If set to "No Masking", the segmentation will be performed without any masking,
    while if set to "Masking", the segmentation will be performed only within the
    the confines of the specified mask.
    """


class MaskingConfig(BaseModel):
    """Masking configuration."""

    mode: Literal["Masking"] = "Masking"
    """
    If set to "No Masking", the segmentation will be performed without any masking,
    while if set to "Masking", the segmentation will be performed only within the
    the confines of the specified mask.
    """

    masking_source: Literal["Table Name", "Label Name"] = "Table Name"
    """
    Mode of masking to be applied.
        - If "Table Name", the identifier refers to a table name. This must identify
        a valid masking ROI table in the OME-Zarr.
        - If "Label Name", the identifier refers to a label image name in the OME-Zarr.
        In this case, the masking roi table will be built on the fly from the label
        image.
    """
    identifier: str | None = None
    """
    Name of the masking table or label image depending on the mode.
    """


MaskingConfigUnion = Annotated[
    NoMaskingConfig | MaskingConfig,
    Field(discriminator="mode"),
]


class IteratorConfig(BaseModel):
    """Advanced iterator configuration."""

    masking: MaskingConfigUnion = Field(
        default=NoMaskingConfig(), title="Masking Iterator Configuration"
    )
    """
    If set, the segmentation will be performed only within the confines of
    the specified mask. A mask can be specified either by a label image or a
    Masking ROI table.
    """
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
    """
    Name of a ROI table. If set, the segmentation will be applied to each ROI
    in the table individually. This option can be combined with masking.
    """
