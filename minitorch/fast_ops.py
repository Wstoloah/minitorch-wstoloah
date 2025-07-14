from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
import numpy.typing as npt
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A convenience wrapper around Numba's `njit` that sets `inline="always"` by default.

    This decorator compiles the given function using Numba's `njit` (no Python mode)
    with aggressive inlining to improve performance in inner loops.

    Args:
    ----
        fn (Callable): The function to be JIT-compiled.
        **kwargs: Additional keyword arguments to pass to Numba's `_njit`.

    Returns:
    -------
        Callable: The compiled function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        same_shape = (out_shape == in_shape).all()
        same_strides = (out_strides == in_strides).all()

        # Fast path: same shape and strides
        if same_shape and same_strides:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
            return

        # Slower path: need to compute index and broadcast
        out_index = np.empty(len(out_shape), dtype=np.int32)
        in_index = np.empty(len(in_shape), dtype=np.int32)

        for i in prange(len(out)):
            # === Inline to_index(i, out_shape, out_index)
            shape_strides = np.empty(len(out_shape), dtype=np.int32)
            acc = 1
            for j in range(len(out_shape) - 1, -1, -1):
                shape_strides[j] = acc
                acc *= out_shape[j]

            for dim in range(len(out_shape)):
                out_index[dim] = (i // shape_strides[dim]) % out_shape[dim]

            # === Inline broadcast_index(out_index, out_shape, in_shape, in_index)
            in_offset = len(in_shape) - len(out_shape)
            for j in range(len(in_shape)):
                if j < in_offset:
                    in_index[j] = 0
                else:
                    if in_shape[j] == 1:
                        in_index[j] = 0
                    else:
                        in_index[j] = out_index[j - in_offset]

            # === Inline index_to_position for in_index and out_index
            in_pos = 0
            for j in range(len(in_shape)):
                in_pos += in_index[j] * in_strides[j]

            out_pos = 0
            for j in range(len(out_shape)):
                out_pos += out_index[j] * out_strides[j]

            # === Apply function
            out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:
    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # === Inline is_contiguous for all tensors
        def _is_contiguous(
            shape: npt.NDArray[np.int32], strides: npt.NDArray[np.int32]
        ) -> bool:
            """Check whether the tensor's memory layout is contiguous.

            Returns
            -------
            bool
                True if memory layout is contiguous, False otherwise.

            """
            expected_stride = 1
            for i in range(len(shape) - 1, -1, -1):
                if shape[i] != 1 and strides[i] != expected_stride:
                    return False
                expected_stride *= shape[i]
            return True

        same_shape = len(out_shape) == len(a_shape) == len(b_shape)
        if same_shape:
            same_shape = True
            for i in range(len(out_shape)):
                if out_shape[i] != a_shape[i] or out_shape[i] != b_shape[i]:
                    same_shape = False
                    break

        stride_aligned = (
            _is_contiguous(out_shape, out_strides)
            and _is_contiguous(a_shape, a_strides)
            and _is_contiguous(b_shape, b_strides)
            and same_shape
        )

        if stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            out_index = np.empty(len(out_shape), dtype=np.int32)
            a_index = np.empty(len(a_shape), dtype=np.int32)
            b_index = np.empty(len(b_shape), dtype=np.int32)
            shape_strides = np.empty(len(out_shape), dtype=np.int32)

            acc = 1
            for j in range(len(out_shape) - 1, -1, -1):
                shape_strides[j] = acc
                acc *= out_shape[j]

            for i in prange(len(out)):
                for dim in range(len(out_shape)):
                    out_index[dim] = (i // shape_strides[dim]) % out_shape[dim]

                a_offset = len(a_shape) - len(out_shape)
                for j in range(len(a_shape)):
                    if j < a_offset or a_shape[j] == 1:
                        a_index[j] = 0
                    else:
                        a_index[j] = out_index[j - a_offset]

                b_offset = len(b_shape) - len(out_shape)
                for j in range(len(b_shape)):
                    if j < b_offset or b_shape[j] == 1:
                        b_index[j] = 0
                    else:
                        b_index[j] = out_index[j - b_offset]

                a_pos = 0
                for j in range(len(a_shape)):
                    a_pos += a_index[j] * a_strides[j]

                b_pos = 0
                for j in range(len(b_shape)):
                    b_pos += b_index[j] * b_strides[j]

                out_pos = 0
                for j in range(len(out_shape)):
                    out_pos += out_index[j] * out_strides[j]

                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.empty(len(out_shape), dtype=np.int32)
        a_index = np.empty(len(a_shape), dtype=np.int32)
        shape_strides = np.empty(len(out_shape), dtype=np.int32)

        # Precompute shape strides for to_index
        acc = 1
        for j in range(len(out_shape) - 1, -1, -1):
            shape_strides[j] = acc
            acc *= out_shape[j]

        reduce_size = a_shape[reduce_dim]

        for i in prange(len(out)):
            # === Inline to_index(i, out_shape, out_index)
            for j in range(len(out_shape)):
                out_index[j] = (i // shape_strides[j]) % out_shape[j]

            # === Copy out_index to a_index (broadcasting not needed for reduce)
            for j in range(len(out_shape)):
                a_index[j] = out_index[j]

            # === First element of reduction
            a_index[reduce_dim] = 0
            a_pos = 0
            for j in range(len(a_shape)):
                a_pos += a_index[j] * a_strides[j]
            acc = a_storage[a_pos]

            # === Accumulate over reduce dimension
            for j in range(1, reduce_size):
                a_index[reduce_dim] = j
                a_pos = 0
                for k in range(len(a_shape)):
                    a_pos += a_index[k] * a_strides[k]
                acc = fn(acc, a_storage[a_pos])

            # === Compute output position
            out_pos = 0
            for j in range(len(out_shape)):
                out_pos += out_index[j] * out_strides[j]

            out[out_pos] = acc

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Ensure inner dimensions align for matrix multiplication
    assert a_shape[-1] == b_shape[-2]

    # Extract dimensions
    batch = out_shape[0]  # After reshaping, tensors are always 3D
    out_i = out_shape[1]
    out_j = out_shape[2]
    inner_dim = a_shape[2]  # Shared dimension for dot product

    # Parallelize over batch dimension
    for n in prange(batch):
        for i in range(out_i):
            for j in range(out_j):
                acc = 0.0
                for k in range(inner_dim):
                    # Handle broadcasting: if shape is 1 in batch, reuse 0-th index
                    a_n = n if a_shape[0] > 1 else 0
                    b_n = n if b_shape[0] > 1 else 0

                    # Compute flat index into a[n, i, k]
                    a_pos = int(
                        a_n * a_strides[0] + i * a_strides[1] + k * a_strides[2]
                    )

                    # Compute flat index into b[n, k, j]
                    b_pos = int(
                        b_n * b_strides[0] + k * b_strides[1] + j * b_strides[2]
                    )

                    acc += a_storage[a_pos] * b_storage[b_pos]

                # Compute flat index into out[n, i, j]
                out_pos = int(
                    n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )

                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
