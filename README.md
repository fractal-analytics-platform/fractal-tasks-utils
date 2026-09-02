# fractal-tasks-utils

General utilities for building [Fractal](https://fractal-analytics-platform.github.io/) tasks.

Requires `ngio >=1.1,<1.2` (since 0.2.0; 0.1.x required `ngio <0.6`).

## Modules

- **segmentation** — utilities for running segmentation models over OME-Zarr images (`compute_segmentation`, `setup_segmentation_iterator`, `IteratorConfig`, `MaskingConfig`)
- **transforms** — standard image transforms (`GaussianBlurConfig`, `MedianFilterConfig`, `HistogramEqualizationConfig`, `SizeFilterConfig`)
- **measurement** — utilities for computing measurements (WIP)

## Migrating to 0.2.0

0.2.0 moves to the ngio 1.1 APIs. The public API of this package is unchanged, but
custom transforms must now implement ngio 1.1's `on_get(array, ctx)` / `on_set(array, ctx)`
instead of the old `get_as_numpy_transform` / `get_as_dask_transform` /
`set_as_numpy_transform` / `set_as_dask_transform` methods.

## Development

Pixi is used for development tasks such as formatting, type-checking, and testing. To run these tasks, use the following commands:

```bash
pixi run format    # format code with ruff
pixi run typecheck # type-check with ty
pixi run test      # run tests with pytest
```
