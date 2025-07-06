from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

from numba import cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    pos = 0
    for idx, stride in zip(index, strides):
        pos += idx * stride

    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ordinal % shape[i]
        ordinal = ordinal // shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    dim_diff = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = (
                0  #  broadcasted dimensions (with size 1) don’t vary. The value at index 0 is repeated for all positions in that dimension.
            )
        else:
            out_index[i] = big_index[
                i + dim_diff
            ]  # The first dim_diff dimensions of big_shape are extra compared to shape => ignore that dimension for the smaller tensor by offsetting the index by dim_diff.


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    max_len = max(len(shape1), len(shape2))
    new_shape = []

    # We'll iterate from right to left
    for i in range(max_len):
        dim1 = shape1[-1 - i] if i < len(shape1) else 1
        dim2 = shape2[-1 - i] if i < len(shape2) else 1

        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            new_dim = max(dim1, dim2)
            new_shape.append(new_dim)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    # Reverse the list to get correct order
    new_shape.reverse()
    return tuple(new_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Compute the strides for a given tensor shape.

    Strides define how many elements in memory you need to skip to move to the next element
    in each dimension. For a contiguous array, this is calculated based on the product of
    dimensions to the right of each axis.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the tensor.

    Returns
    -------
    tuple[int]
        The computed strides corresponding to the shape.

    """
    layout = [
        1
    ]  # last dimension stride is always 1 (moving one step in last dimension moves 1 element)
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    # last stride removed: corresponds to moving beyond the entire tensor
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Move the internal storage to GPU memory using Numba (in-place)."""
        if not cuda.is_cuda_array(self._storage):
            self._storage = cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check whether the tensor's memory layout is contiguous.

        Returns
        -------
        bool
            True if memory layout is contiguous, False otherwise.

        """
        last = 1e9
        for stride in self._strides:  # strides should decrease or stay the same as you go from outer to inner dimensions
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Statically broadcast two shapes.

        Parameters
        ----------
        shape_a : Sequence[int]
            First shape to broadcast.
        shape_b : Sequence[int]
            Second shape to broadcast.

        Returns
        -------
        UserShape
            Broadcasted output shape.

        Raises
        ------
        IndexingError
            If shapes cannot be broadcast.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert a multidimensional index into a position in storage.

        Parameters
        ----------
        index : Union[int, Sequence[int]]
            Multidimensional index.

        Returns
        -------
        int
            Corresponding position in storage.

        Raises
        ------
        IndexingError
            If index is out of bounds or has incorrect dimensionality.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        elif isinstance(index, tuple):
            aindex = array(index)
        else:
            raise IndexingError(f"Invalid index type: {type(index)}")

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate all indices for this tensor.

        Yields
        ------
        Sequence[int]
            Each valid index in the tensor.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Randomly sample a valid index in the tensor.

        Returns
        -------
        Sequence[int]
            A randomly chosen valid index.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get the tensor value at a given index.

        Parameters
        ----------
        key : Sequence[int]
            The index to access.

        Returns
        -------
        float
            The value at the given index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the tensor value at a given index.

        Parameters
        ----------
        key : Sequence[int]
            The index to modify.
        val : float
            The value to assign.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return the internal tensor representation.

        Returns
        -------
        Tuple[Storage, Shape, Strides]
            The storage, shape, and strides of the tensor.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Return a new `TensorData` object with permuted dimensions.

        Parameters
        ----------
        *order : int
            A permutation of the dimensions.

        Returns
        -------
        TensorData
            A new TensorData with permuted shape and strides.

        Raises
        ------
        AssertionError
            If the provided order is not a valid permutation.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)

        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Pretty-print the tensor with correct formatting for nested dimensions.

        Returns
        -------
        str
            A string representation of the tensor.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
