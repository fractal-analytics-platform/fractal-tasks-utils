"""Core computation logic for segmentation tasks."""

import logging
import time
from typing import Protocol

import numpy as np
from ngio import ChannelSelectionModel, OmeZarrContainer, open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage

from fractal_tasks_utils.segmentation._models import (
    IteratorConfiguration,
    MaskingConfiguration,
)
from fractal_tasks_utils.segmentation._transforms import (
    SegmentationTransformConfig,
)


class SegmentationFunction(Protocol):
    def __call__(
        self,
        input_image: np.ndarray,
    ) -> np.ndarray:
        """Segmentation function protocol definition."""
        ...


def load_masked_image(
    ome_zarr: OmeZarrContainer,
    masking_configuration: MaskingConfiguration,
    logger: logging.Logger,
    level_path: str | None = None,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (str | None): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logger.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_image


def setup_segmentation_iterator(
    zarr_url: str,
    # Segmentation parameters
    channels: list[ChannelSelectionModel],
    label_name: str = "segmentation",
    level_path: str | None = None,
    # Iteration parameters
    iterator_configuration: IteratorConfiguration | None = None,
    segmentation_transform_config: SegmentationTransformConfig | None = None,
    # Other parameters
    overwrite: bool = True,
) -> SegmentationIterator | MaskedSegmentationIterator:
    """Set up the segmentation iterator based on the provided configuration.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channels (CellposeChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        label_name (str): Name of the resulting label image.
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        segmentation_transform_config (SegmentationTransformConfig | None):
            Configuration for pre- and post-processing transformations. If not
            provided, no additional transformations will be applied.
        custom_model (str | None): Path to a custom Cellpose model. If not
            set, the default "cpsam" model will be used.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger = logging.getLogger("fractal_tasks_utils.setup_iterator")
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    # Validate that the specified channels are present in the image
    # if _skip_segmentation(channels=channels, ome_zarr=ome_zarr):
    #    return None
    logger.info(f"Formatted label name: {label_name=}")

    # Derive the label and an get it at the specified level path
    ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    label = ome_zarr.get_label(name=label_name, path=level_path)
    logger.info(f"Derived label image: {label=}")

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    # Determine if we are doing 3D segmentation or 2D
    axes_order = "czyx" if ome_zarr.is_3d else "cyx"
    logger.info(f"Segmenting using {axes_order=}")

    if segmentation_transform_config is None:
        segmentation_transform_config = SegmentationTransformConfig()

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        image = ome_zarr.get_image(path=level_path)
        logger.info(f"{image=}")
        iterator = SegmentationIterator(
            input_image=image,
            output_label=label,
            channel_selection=channels,
            axes_order=axes_order,
            input_transforms=segmentation_transform_config.to_pre_transforms(),
            output_transforms=segmentation_transform_config.to_post_transforms(),
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
            logger=logger,
        )
        logger.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedSegmentationIterator(
            input_image=masked_image,
            output_label=label,
            channel_selection=channels,
            axes_order=axes_order,
            input_transforms=segmentation_transform_config.to_pre_transforms(),
            output_transforms=segmentation_transform_config.to_post_transforms(),
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logger.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logger.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logger.info(f"Iterator updated with ROI table: {iterator=}")
    return iterator


def compute_segmentation(
    *,
    func: SegmentationFunction,
    iterator: SegmentationIterator | MaskedSegmentationIterator,
) -> None:
    """Core computation loop for applying the segmentation function.

    This function iterates over the image over the specifed patterns in
    the iterator, applies the segmentation function to each chunk of the image,
    and writes the resulting label images back to the OME-Zarr.

    Args:
        func: The segmentation function to apply to each chunk of the image.
            This function should take an image chunk as input and return a label
            image as output.
        func_kwargs: Keyword arguments to pass to the segmentation function.
        iterator: An iterator that yields image chunks and corresponding writers.
    """
    logger = logging.getLogger("fractal_tasks_utils.compute_segmentation")

    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    #
    logger.info("Starting processing...")
    run_times: list[float] = []
    num_rois = len(iterator.rois)
    logging_step = max(1, num_rois // 10)
    for it, (image_data, writer) in enumerate(iterator.iter_as_numpy()):
        start_time = time.time()
        label_img = func(
            input_image=image_data,
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = max(max_label, label_img.max())
        writer(label_img)
        iteration_time = time.time() - start_time
        run_times.append(iteration_time)

        # Only log the progress every logging_step iterations
        if it % logging_step == 0 or it == num_rois - 1:
            avg_time = sum(run_times) / len(run_times)
            logger.info(
                f"Processed ROI {it + 1}/{num_rois} "
                f"(avg time per ROI: {avg_time:.2f} s)"
            )
    return None
