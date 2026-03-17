"""Pydantic models for advanced iterator configuration."""

import logging
from typing import Annotated, Literal

import numpy as np
from dask import array as da
from ngio.io_pipes._ops_axes import AxesOps
from ngio.io_pipes._ops_slices import SlicingOps
from pydantic import BaseModel, Field
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, median
from skimage.morphology import remove_small_objects

logger = logging.getLogger("fractal_tasks_utils.transforms")

class GaussianBlurTransform:
    """Gaussian pre-processing configuration."""

    def __init__(self, sigma_xy: float = 2.0, sigma_z: float | None = None):
        self.sigma_xy = sigma_xy
        self.sigma_z = sigma_z

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Filtered image.
        """
        if image.ndim == 2:
            if self.sigma_z is not None:
                logger.warning(
                    "sigma_z is specified but the input image is 2D. Ignoring sigma_z."
                )
            return gaussian(image, sigma=self.sigma_xy)
        elif image.ndim == 3:
            sigma = (
                self.sigma_z if self.sigma_z is not None else 0,
                self.sigma_xy,
                self.sigma_xy,
            )
            return gaussian(image, sigma=sigma)
        elif image.ndim == 4:
            sigma = (
                0,
                self.sigma_z if self.sigma_z is not None else 0,
                self.sigma_xy,
                self.sigma_xy,
            )
            return gaussian(image, sigma=sigma)
        else:
            raise ValueError("Input to Gaussian filter image must be 2D, 3D, or 4D.")

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply Gaussian blur transformation to a numpy array."""
        return self.apply(array)

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply Gaussian blur transformation to a dask array."""
        # apply the Gaussian filter to each chunk of the dask array using map_blocks
        raise NotImplementedError(
            "Gaussian blur transformation is not implemented for dask arrays yet."
        )

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Get Gaussian blur transformation applied before writing a numpy array."""
        return array

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Get Gaussian blur transformation applied before writing a dask array."""
        return array


class GaussianBlurConfig(BaseModel):
    """Configuration for Gaussian blur transformation."""

    type: Literal["Gaussian Blur Filter"] = "Gaussian Blur Filter"
    """
    Transformation type identifier for Gaussian blur filtering.
    """
    sigma_xy: float = Field(default=2.0, gt=0, title="Sigma XY")
    """
    Standard deviation for Gaussian kernel in XY plane.
    """
    sigma_z: float | None = None
    """
    Standard deviation for Gaussian kernel in Z axis.
    If not specified, no blurring is applied in Z axis.
    """

    def to_transform(self) -> GaussianBlurTransform:
        """Convert the configuration to a GaussianBlurTransform instance."""
        return GaussianBlurTransform(sigma_xy=self.sigma_xy, sigma_z=self.sigma_z)


class MedianFilterTransform:
    """Median filter pre-processing transformation."""

    def __init__(self, size_xy: int = 2, size_z: int | None = None):
        self.size_xy = size_xy
        self.size_z = size_z

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Median filter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Filtered image.
        """
        if image.ndim == 2:
            if self.size_z is not None:
                logger.warning(
                    "size_z is specified but the input image is 2D. Ignoring size_z."
                )
            return median(image, footprint=np.ones((self.size_xy, self.size_xy)))
        elif image.ndim == 3:
            size = (
                self.size_z if self.size_z is not None else 1,
                self.size_xy,
                self.size_xy,
            )
            return median(image, footprint=np.ones(size))
        elif image.ndim == 4:
            size = (
                1,
                self.size_z if self.size_z is not None else 1,
                self.size_xy,
                self.size_xy,
            )
            return median(image, footprint=np.ones(size))
        else:
            raise ValueError("Input to median filter image must be 2D, 3D, or 4D.")

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply Median filter transformation to a numpy array."""
        return self.apply(array)

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply Median filter transformation to a dask array."""
        raise NotImplementedError(
            "Median filter transformation is not implemented for dask arrays yet."
        )

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Get Median filter transformation applied before writing a numpy array."""
        return array

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Get Median filter transformation applied before writing a dask array."""
        return array


class MedianFilterConfig(BaseModel):
    """Configuration for Median filter transformation.

    Attributes:
        type (Literal["Median Filter"]): Type of transformation.
        size_xy (int): Size in pixels of the median filter in XY plane.
        size_z (int | None): Size in pixels of the median filter in Z axis.
            If not specified, no filtering is applied in Z axis.
    """

    type: Literal["Median Filter"] = "Median Filter"
    """
    Transformation type identifier for median filtering.
    """
    size_xy: int = Field(default=2, gt=0, title="Size XY")
    """
    Size in pixels of the median filter in XY plane.
    """
    size_z: int | None = None
    """
    Size in pixels of the median filter in Z axis.
    If not specified, no filtering is applied in Z axis.
    """

    def to_transform(self) -> MedianFilterTransform:
        """Convert the configuration to a MedianFilterTransform instance."""
        return MedianFilterTransform(size_xy=self.size_xy, size_z=self.size_z)


class HistogramEqualizationTransform:
    """Histogram equalization pre-processing transformation."""

    def __init__(
        self,
        kernel_size_xy: int | None = None,
        kernel_size_z: int | None = None,
        clip_limit: float = 0.01,
        nbins: int = 256,
    ):
        self.kernel_size_xy = kernel_size_xy
        self.kernel_size_z = kernel_size_z
        self.clip_limit = clip_limit
        self.nbins = nbins

    def _build_kernel_size(self, image: np.ndarray) -> np.ndarray | None:
        """Build kernel size tuple based on image dimensions."""
        if image.ndim == 2:
            if self.kernel_size_z is not None:
                logger.warning(
                    "kernel_size_z is specified but the input image is 2D. "
                    "Ignoring kernel_size_z."
                )
            kernel_size = (self.kernel_size_xy, self.kernel_size_xy)

        elif image.ndim == 3:
            kernel_size = (
                self.kernel_size_z,
                self.kernel_size_xy,
                self.kernel_size_xy,
            )

        elif image.ndim == 4:
            kernel_size = (
                1,
                self.kernel_size_z,
                self.kernel_size_xy,
                self.kernel_size_xy,
            )
        else:
            raise ValueError("Input to histogram equalization must be 2D, 3D, or 4D.")

        # Return None if any kernel size component is None (use scikit-image defaults)
        # Return np.array only if all values are specified
        return (
            np.array(kernel_size) if all(k is not None for k in kernel_size) else None
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Histogram equalized image.
        """
        kernel_size = self._build_kernel_size(image)

        return equalize_adapthist(
            image,
            kernel_size=kernel_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply histogram equalization transformation to a numpy array."""
        return self.apply(array)

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply histogram equalization transformation to a dask array."""
        raise NotImplementedError(
            "Histogram equalization is not implemented for dask arrays yet."
        )

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Get histogram equalization applied before writing a numpy array."""
        return array

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Get histogram equalization applied before writing a dask array."""
        return array


class HistogramEqualizationConfig(BaseModel):
    """Configuration for Contrast Limited Adaptive Histogram Equalization (CLAHE)."""

    type: Literal["Histogram Equalization"] = "Histogram Equalization"
    """
    Transformation type identifier for histogram equalization.
    """
    kernel_size_xy: int | None = Field(
        default=None,
        title="Kernel Size XY",
    )
    """
    Shape of kernel in XY plane. If None, the default is 1/8 of image height by
    1/8 of its width.
    """
    kernel_size_z: int | None = None
    """
    Shape of kernel in Z axis. If None, the default is 1/8 of image height by
    1/8 of its width.
    """
    clip_limit: float = Field(default=0.01, ge=0, le=1)
    """
    Clipping limit, normalized between 0 and 1 (higher values give more contrast).
    """
    nbins: int = 256
    """
    Number of gray bins for histogram ("data range").
    """

    def to_transform(self) -> HistogramEqualizationTransform:
        """Convert the configuration to a HistogramEqualizationTransform instance."""
        return HistogramEqualizationTransform(
            kernel_size_xy=self.kernel_size_xy,
            kernel_size_z=self.kernel_size_z,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )


class SizeFilterTransform:
    """Size filter post-processing transformation."""

    def __init__(self, min_size: int = 0):
        self.min_size = min_size

    def apply(self, labels: np.ndarray) -> np.ndarray:
        """Apply size filtering to the labeled image.

        Args:
            labels (np.ndarray): Labeled image.

        Returns:
            np.ndarray: Size-filtered labeled image.
        """
        return remove_small_objects(labels, max_size=self.min_size)

    def get_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Apply size filter transformation to a numpy array."""
        return self.apply(array)

    def get_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Apply size filter transformation to a dask array."""
        raise NotImplementedError(
            "Size filter transformation is not implemented for dask arrays yet."
        )

    def set_as_numpy_transform(
        self, array: np.ndarray, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> np.ndarray:
        """Get size filter transformation applied before writing a numpy array."""
        return self.apply(array)

    def set_as_dask_transform(
        self, array: da.Array, slicing_ops: SlicingOps, axes_ops: AxesOps
    ) -> da.Array:
        """Get size filter transformation applied before writing a dask array."""
        raise NotImplementedError(
            "Size filter transformation is not implemented for dask arrays yet."
        )


class SizeFilterConfig(BaseModel):
    """Configuration for size filter post-processing."""

    type: Literal["Size Filter"] = "Size Filter"
    """
    Transformation type identifier for size filtering.
    """
    min_size: int = Field(ge=0)
    """
    Minimum size in pixels for objects to keep.
    Objects smaller than this size will be removed.
    """

    def to_transform(self) -> SizeFilterTransform:
        """Convert the configuration to a SizeFilterTransform instance."""
        return SizeFilterTransform(min_size=self.min_size)


PreProcess = Annotated[
    GaussianBlurConfig | MedianFilterConfig | HistogramEqualizationConfig,
    Field(discriminator="type"),
]

PostProcess = Annotated[
    SizeFilterConfig,
    Field(discriminator="type"),
]
