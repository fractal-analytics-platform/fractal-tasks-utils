"""Pydantic models for advanced iterator configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class MaskingConfig(BaseModel):
    """Masking configuration.

    Attributes:
        mode (Literal["Table Name", "Label Name"]): Mode of masking to be applied.
            If "Table Name", the identifier refers to a masking table name.
            If "Label Name", the identifier refers to a label image name.
        identifier (str | None): Name of the masking table or label image
            depending on the mode.
    """

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    identifier: str | None = None


class IteratorConfig(BaseModel):
    """Advanced iterator configuration.

    Attributes:
        masking (MaskingConfig | None): If set, the segmentation will be
            performed only within the confines of the specified mask. A mask can be
            specified either by a label image or a Masking ROI table.
        roi_table (str | None): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    masking: MaskingConfig | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
