"""
Run-time checkable numeric types for use with beartype.

Those types will not be verified by static type checkers.

Also, they are completely optional and can be ignored if you don't want to use them.

Simply use int and float if you don't want to use these types.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias

import numpy as np
import numpy.typing as npt
from beartype.vale import Is
from torch import Tensor

# An int that must be strictly greater than zero.
PositiveInt: TypeAlias = Annotated[int, Is[lambda n: n > 0]]

# A float that must be strictly greater than zero.
PositiveFloat: TypeAlias = Annotated[float, Is[lambda x: x > 0]]

# An int that must be zero or greater.
NonNegativeInt: TypeAlias = Annotated[int, Is[lambda n: n >= 0]]

# A float that must be zero or greater.
NonNegativeFloat: TypeAlias = Annotated[float, Is[lambda x: x >= 0]]


def _is_wrapped_angle_rad(angle_rad: Any) -> bool:
    """Tests whether every angle lies in [-pi, pi); works on numpy and torch."""
    return bool((angle_rad >= -np.pi).all()) and bool((angle_rad < np.pi).all())


# A single angle in radians that must lie in [-pi, pi).
WrappedAngleRad: TypeAlias = Annotated[float, Is[lambda a: -np.pi <= a < np.pi]]

# A float64 numpy array in which every element is an angle in radians in [-pi, pi).
WrappedAngleRadArray: TypeAlias = Annotated[
    npt.NDArray[np.float64],
    Is[_is_wrapped_angle_rad],
]

# A torch tensor in which every element is an angle in radians in [-pi, pi).
WrappedAngleRadTensor: TypeAlias = Annotated[Tensor, Is[_is_wrapped_angle_rad]]

# A float64 numpy array of (x, y, yaw) poses; only the last column, the yaw, must lie in [-pi, pi).
WrappedSE2Array: TypeAlias = Annotated[
    npt.NDArray[np.float64],
    Is[lambda a: _is_wrapped_angle_rad(a[..., 2])],
]
