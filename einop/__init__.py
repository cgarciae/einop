__version__ = "0.0.1"
__all__ = ["einop"]

import functools
from typing import TypeVar

import einops
from einops.parsing import EinopsError, ParsedExpression

A = TypeVar("A")


@functools.lru_cache(256)
def _match_einop(pattern: str, reduction=None, **axes_lengths: int):
    """Find the corresponding operation matching the pattern"""
    left, rght = pattern.split("->")
    left = ParsedExpression(left)
    rght = ParsedExpression(rght)

    default_op = "rearrange"
    op = default_op

    for index in left.identifiers:
        if index not in rght.identifiers:
            op = "reduce"
            break

    for index in rght.identifiers:
        if index not in left.identifiers:
            if op != default_op:
                raise EinopsError(
                    "You must perform a reduce and repeat separately: {}".format(
                        pattern
                    )
                )
            op = "repeat"
            break

    return op


def einop(tensor: A, pattern: str, reduction=None, **axes_lengths: int) -> A:
    """Perform either reduce, rearrange, or repeat depending on pattern"""
    op = _match_einop(pattern, reduction, **axes_lengths)

    if op == "rearrange":
        if reduction is not None:
            raise EinopsError(
                'Got reduction operation but there is no dimension to reduce in pattern: "{}"'.format(
                    pattern
                )
            )
        return einops.rearrange(tensor, pattern, **axes_lengths)
    elif op == "reduce":
        if reduction is None:
            raise EinopsError(
                "Missing reduction operation for reduce pattern: {}".format(pattern)
            )
        return einops.reduce(tensor, pattern, reduction, **axes_lengths)
    elif op == "repeat":
        if reduction is not None:
            raise EinopsError(
                "Do not pass reduction for repeat pattern: {}".format(pattern)
            )
        return einops.repeat(tensor, pattern, **axes_lengths)
    else:
        raise ValueError(f"Unknown operation: {op}")
