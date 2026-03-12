"""Core computation logic for measurement tasks."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from ngio import ChannelSelectionModel, Roi, open_ome_zarr_container
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.transforms import ZoomTransform


def join_tables(
    tables: list[dict[str, list]], index_key: str = "label"
) -> pd.DataFrame:
    """Join a list of per-ROI result dicts into a single DataFrame.

    Args:
        tables: List of dicts returned by the per-ROI extraction function.
        index_key: Column to use as the DataFrame index. Defaults to "label".

    Returns:
        A DataFrame with all ROI results concatenated and indexed by `index_key`.
    """
    if len(tables) == 0:
        raise ValueError("At least one table is required to join into a DataFrame.")
    out_dict: dict[str, list] = {}
    for table in tables:
        for key, value in table.items():
            if key not in out_dict:
                out_dict[key] = []
            out_dict[key].extend(value)
    df = pd.DataFrame(out_dict)
    df = df.set_index(index_key)
    return df


def setup_measurement_iterator(
    zarr_url: str,
    label_image_name: str,
    level_path: str | None = None,
    channels: list[ChannelSelectionModel] | None = None,
    roi_table_names: list[str] | None = None,
) -> FeatureExtractorIterator:
    """Set up a FeatureExtractorIterator for measurement tasks.

    Args:
        zarr_url: URL to the OME-Zarr container.
        label_image_name: Name of the label image to analyze.
        level_path: Optional path to a specific resolution level of the image.
            If not provided, the highest resolution level is used.
        channels: Optional list of ChannelSelectionModel to specify which channels
            to include. If None, all channels are included.
        roi_table_names: Optional list of ROI table names to include in the
            iterator. If None, no table is included.

    Returns:
        A FeatureExtractorIterator that yields (image, label, roi)
    """
    logger = logging.getLogger("fractal_tasks_utils.setup_measurement_iterator")
    logger.info(f"{zarr_url=}")

    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    image = ome_zarr.get_image(path=level_path)
    logger.info(f"{image=}")

    # Get the label at the closest resolution to the image (strict=False allows
    # a ZoomTransform to handle any resolution mismatch)
    label_image = ome_zarr.get_label(
        name=label_image_name, pixel_size=image.pixel_size, strict=False
    )
    logger.info(f"{label_image=}")

    # For 2D images, squeeze the singleton z-axis (yxc); keep z for 3D (yxzc)
    axes_order = "yxc" if image.is_2d else "yxzc"

    # The ZoomTransform will handle any necessary rescaling of the label image to
    # match the image resolution (if not necessary, it will be a no-op)
    label_zoom_transform = ZoomTransform(
        input_image=label_image,
        target_image=image,
        order="nearest",
    )

    iterator = FeatureExtractorIterator(
        input_image=image,
        input_label=label_image,
        axes_order=axes_order,
        channel_selection=channels,
        label_transforms=[label_zoom_transform],
    )
    # by_zyx(strict=False): works for both 2D and 3D data
    iterator = iterator.by_zyx(strict=False)

    tables = roi_table_names if roi_table_names is not None else []
    for table_name in tables:
        table = ome_zarr.get_generic_roi_table(table_name)
        iterator = iterator.product(table)

    logger.info(f"Iterator created: {iterator=}")
    return iterator


def compute_measurement(
    *,
    measurement_func: Callable[[np.ndarray, np.ndarray, Roi], dict],
    iterator: FeatureExtractorIterator,
) -> pd.DataFrame:
    """Run the measurement computation loop.

    Iterates over ROIs using the provided iterator, applies `measurement_func`
    to each chunk, and returns all results joined into a single DataFrame.

    Args:
        measurement_func: Consumer-provided extraction function with signature
            ``(image, label, roi) -> dict``. The dict keys become DataFrame
            columns; values must be lists of equal length.
        iterator: A `FeatureExtractorIterator` (with `.by_zyx` already applied)
            that yields ``(image, label, roi)`` tuples.
    Returns:
        A DataFrame with all per-ROI results concatenated, indexed by "label".
    """
    logger = logging.getLogger("fractal_tasks_utils.compute_measurement")

    tables = []
    num_rois = len(iterator.rois)
    logging_step = max(1, num_rois // 10)

    logger.info("Starting measurement...")
    for it, (input_data, label_data, roi) in enumerate(iterator.iter_as_numpy()):
        table_dict = measurement_func(input_data, label_data, roi)
        tables.append(table_dict)

        if it % logging_step == 0 or it == num_rois - 1:
            logger.info(f"Processed ROI {it + 1}/{num_rois}")

    return join_tables(tables, index_key="label")
