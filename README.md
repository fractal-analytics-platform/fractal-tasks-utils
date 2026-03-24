# fractal-tasks-utils

General utilities for building [Fractal](https://fractal-analytics-platform.github.io/) tasks.

## Modules

- **segmentation** — utilities for running segmentation models over SOME-Zarr images (`compute_segmentation`, `setup_segmentation_iterator`, `IteratorConfiguration`, `MaskingConfiguration`)
- **transforms** — standard image transforms (`GaussianBlurConfig`, `MedianFilterConfig`, `HistogramEqualizationConfig`, `SizeFilterConfig`)
- **measurement** — utilities for computing measurements (WIP)

## Development

Pixi is used for development tasks such as formatting, type-checking, and testing. To run these tasks, use the following commands:

```bash
pixi run format    # format code with ruff
pixi run typecheck # type-check with ty
pixi run test      # run tests with pytest
```
