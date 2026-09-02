"""Core computation logic for segmentation tasks."""

import logging
import time
from collections.abc import Callable

import numpy as np
from ngio import (
    ChannelSelectionModel,
    MaskedImage,
    OmeZarrContainer,
    open_ome_zarr_container,
)
from ngio.common import ConsolidationMode
from ngio.iterators import MaskedSegmentationIterator, SegmentationIterator

from fractal_tasks_utils._iteration import (
    IterateBy,
    apply_iterate_by,
    resolve_iterate_by,
    validate_axes_order,
)
from fractal_tasks_utils.segmentation._models import (
    IteratorConfig,
    MaskingConfig,
    NoMaskingConfig,
)
from fractal_tasks_utils.segmentation._transforms import (
    SegmentationTransformConfig,
)


def _load_masked_image(
    ome_zarr: OmeZarrContainer,
    masking_configuration: MaskingConfig,
    logger: logging.Logger,
    level_path: str | None = None,
) -> MaskedImage:
    """Load a masked image from an ome-Zarr based on the masking configuration.

    Args:
        ome_zarr: The ome-Zarr container.
        masking_configuration (MaskingConfig): Configuration for masking.
        level_path (str | None): Optional path to a specific resolution level.

    """
    if masking_configuration.masking_source == "Table Name":
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
    output_label_name: str = "segmentation",
    level_path: str | None = None,
    # Iteration parameters
    iterator_configuration: IteratorConfig | None = None,
    segmentation_transform_config: SegmentationTransformConfig | None = None,
    axes_order: str | None = None,
    iterate_by: IterateBy | None = None,
    # Other parameters
    consolidation_mode: ConsolidationMode = "auto",
    overwrite: bool = True,
) -> SegmentationIterator | MaskedSegmentationIterator:
    """Set up the segmentation iterator based on the provided configuration.

    Args:
        zarr_url (str): URL to the ome-Zarr container
        channels (CellposeChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        output_label_name (str): Name of the resulting label image.
        level_path (str | None): If the ome-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        segmentation_transform_config (SegmentationTransformConfig | None):
            Configuration for pre- and post-processing transformations. If not
            provided, no additional transformations will be applied.
        axes_order (str | None): Axes order of the patches handed to the
            segmentation function. If not provided, "czyx" is used for 3D
            images and "cyx" for 2D ones. An axis the image does not have is
            added as a singleton, so a function that always wants a 4D patch
            can ask for "czyx" regardless of whether the data is 2D or 3D.
        iterate_by (IterateBy | None): How much of the image a single
            iteration covers. "by_zyx" hands over the full z/y/x extent,
            "by_yx" hands over one z plane at a time. If not provided, it is
            inferred from `axes_order`: an order carrying "z" asks for
            "by_zyx", one without it asks for "by_yx". Pass it explicitly to
            combine the two, e.g. axes_order="czyx" with iterate_by="by_yx"
            runs a 2D function on every plane while keeping the singleton z
            axis in the patch.
        custom_model (str | None): Path to a custom Cellpose model. If not
            set, the default "cpsam" model will be used.
        consolidation_mode (ConsolidationMode): How the output pyramid is
            rebuilt after iteration. "auto" (the default) builds a small
            pyramid in memory and falls back to the chunked dask path above
            ngio's `consolidation.numpy_max_bytes` (256 MB by default).
            "dask" always takes the chunked path, "numpy" always takes the
            in-memory one, and "coarsen" is the cheapest option for label
            images. On very large images, pass an explicit mode if the
            "auto" choice does not fit the available memory.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger = logging.getLogger("fractal_tasks_utils.setup_iterator")
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the ome-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    logger.info(f"Formatted label name: {output_label_name=}")

    # Derive the label and an get it at the specified level path
    ome_zarr.derive_label(name=output_label_name, overwrite=overwrite)
    label = ome_zarr.get_label(name=output_label_name, path=level_path)
    logger.info(f"Derived label image: {label=}")

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfig()

    # Determine if we are doing 3D segmentation or 2D
    # Determine if we are doing 3D segmentation or 2D
    if axes_order is None:
        axes_order = "czyx" if ome_zarr.is_3d else "cyx"
    else:
        validate_axes_order(axes_order)
    iterate_by = resolve_iterate_by(
        axes_order=axes_order, iterate_by=iterate_by, is_3d=ome_zarr.is_3d
    )
    logger.info(f"Segmenting using {axes_order=} {iterate_by=}")

    if segmentation_transform_config is None:
        segmentation_transform_config = SegmentationTransformConfig()

    if isinstance(iterator_configuration.masking, NoMaskingConfig):
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
            consolidation_mode=consolidation_mode,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = _load_masked_image(
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
            consolidation_mode=consolidation_mode,
        )
    # Split the ROIs into the requested iteration unit. Either unit also
    # splits a time axis, so we always iterate over it if there is one.
    iterator = apply_iterate_by(iterator, iterate_by)
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
    segmentation_func: Callable[[np.ndarray], np.ndarray],
    iterator: SegmentationIterator | MaskedSegmentationIterator,
) -> None:
    """Core computation loop for applying the segmentation function.

    This function iterates over the image over the specified patterns in
    the iterator, applies the segmentation function to each chunk of the image,
    and writes the resulting label images back to the ome-Zarr.

    Args:
        segmentation_func: The segmentation function to apply to each image chunk.
            This function should take an image chunk as input and return a label
            image as output.
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
    for it, (input_img, writer) in enumerate(iterator.iter_as_numpy()):
        start_time = time.time()
        label_img = segmentation_func(input_img)
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
