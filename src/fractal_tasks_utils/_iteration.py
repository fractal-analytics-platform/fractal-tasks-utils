"""Shared helpers to configure the axes layout and iteration unit of iterators."""

from typing import Literal, TypeVar

from ngio.iterators import AbstractIteratorBuilder
from ngio.ome_zarr_meta.ngio_specs import canonical_axes_order

IterateBy = Literal["by_zyx", "by_yx"]
"""How much of an image a single iteration covers.

Named after the ngio iterator methods they resolve to.

- "by_zyx": one ROI per t, covering the full z/y/x extent.
- "by_yx": one ROI per (t, z) combination, covering the full y/x extent.

Channels are never iterated: ngio splits ROIs along t/z/y/x only, so the c axis
is carried whole into every patch, narrowed by `channels` rather than by the
iteration unit.
"""

_IteratorT = TypeVar("_IteratorT", bound=AbstractIteratorBuilder)


def validate_axes_order(axes_order: str) -> None:
    """Check that an axes order is well formed before it reaches ngio.

    Asking for a canonical axis the image does not have is deliberate: ngio
    inserts a singleton for it, which is what lets a task request "czyx" for
    both 2D and 3D data. ngio extends that to any string though, so a typo
    becomes a phantom axis rather than an error; checking the alphabet here is
    what separates the two. A repeated axis ngio does reject, and so is a
    dropped one, but only if it is non-singleton and only once the first patch
    is read, phrased in terms of the axes it dropped rather than the value that
    was typed.

    The canonical alphabet is narrower than ngio's, which also accepts
    non-canonical on-disk axis names. That is fine here: this validates a task
    parameter, before any image is opened.

    Args:
        axes_order: The requested axes order, e.g. "czyx".

    Raises:
        ValueError: If the order uses an unknown axis, repeats one, or omits
            "y" or "x".
    """
    known = set(canonical_axes_order())
    unknown = sorted(set(axes_order) - known)
    if unknown:
        raise ValueError(
            f"axes_order={axes_order!r} contains unknown axes {unknown}. "
            f"Valid axes are {sorted(known)}."
        )
    duplicated = sorted({axis for axis in axes_order if axes_order.count(axis) > 1})
    if duplicated:
        raise ValueError(
            f"axes_order={axes_order!r} repeats axes {duplicated}; "
            "each axis can appear at most once."
        )
    missing = [axis for axis in ("y", "x") if axis not in axes_order]
    if missing:
        raise ValueError(
            f"axes_order={axes_order!r} is missing the required axes {missing}."
        )


def resolve_iterate_by(
    axes_order: str,
    iterate_by: IterateBy | None,
    is_3d: bool,
) -> IterateBy:
    """Resolve how much of the image a single iteration should cover.

    When `iterate_by` is not given it is inferred from `axes_order`: an order
    carrying a z axis asks for "by_zyx", one without it asks for "by_yx". Note
    that the inference is a default, not a rule — an order carrying z is valid
    with either unit, and "by_yx" is what a 2D function that still wants a
    singleton z in its patch should ask for.

    Args:
        axes_order: The axes order the patches are handed in.
        iterate_by: The requested iteration unit, or None to infer one.
        is_3d: Whether the image being iterated has a real z axis.

    Returns:
        The resolved iteration unit.

    Raises:
        ValueError: If "by_zyx" is requested from an axes order that drops the
            z axis of a 3D image, which ngio cannot serve.
    """
    if iterate_by is None:
        return "by_zyx" if "z" in axes_order else "by_yx"

    if iterate_by == "by_zyx" and "z" not in axes_order and is_3d:
        raise ValueError(
            f"iterate_by='by_zyx' needs a z axis, but axes_order={axes_order!r} "
            "drops it and the image is 3D. Add 'z' to axes_order to segment "
            "whole volumes, or use iterate_by='by_yx' to iterate plane by plane."
        )
    return iterate_by


def apply_iterate_by(iterator: _IteratorT, iterate_by: IterateBy) -> _IteratorT:
    """Split the iterator's ROIs into the requested iteration unit.

    Args:
        iterator: The iterator to split.
        iterate_by: The iteration unit, as resolved by `resolve_iterate_by`.

    Returns:
        A new iterator over the split ROIs.
    """
    if iterate_by == "by_zyx":
        # strict=False: images without a real z axis fall back to 2D planes,
        # which is the same thing when z is absent or singleton.
        return iterator.by_zyx(strict=False)
    return iterator.by_yx()
