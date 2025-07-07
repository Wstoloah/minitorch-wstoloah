from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Apply a unary function elementwise to the tensor `x`, optionally writing
        results into `out`.

        Args:
        ----
            x (Tensor): Input tensor to map over.
            out (Optional[Tensor], optional): Optional output tensor to write to.
                If not provided, a new tensor with the same shape as `x` is created.

        Returns:
        -------
            Tensor: Result tensor after applying the function elementwise.

        """
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Create a higher-order unary elementwise map operation.

        Args:
        ----
            fn (Callable[[float], float]): Function mapping a single float to a float.

        Returns:
        -------
            MapProto: A callable that applies `fn` elementwise to an input tensor,
            optionally writing to an output tensor.

        """
        raise NotImplementedError("map() is not implemented yet")

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a higher-order unary elementwise map operation that takes explicit
        input and output tensors.

        Args:
        ----
            fn (Callable[[float], float]): Function mapping a single float to a float.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that applies `fn` elementwise
            to the first tensor and writes results into the second tensor.

        """

        def _unimplemented(a: Tensor, out: Tensor) -> Tensor:
            raise NotImplementedError("cmap is not yet implemented.")

        return _unimplemented

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a higher-order binary elementwise zip operation.

        Args:
        ----
            fn (Callable[[float, float], float]): Function mapping two floats to a float.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that applies `fn` elementwise
            to two input tensors, returning a new tensor with the results.

        """

        def _unimplemented(a: Tensor, out: Tensor) -> Tensor:
            raise NotImplementedError("zip is not yet implemented.")

        return _unimplemented

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Create a higher-order reduce operation over a specified dimension.

        Args:
        ----
            fn (Callable[[float, float], float]): Binary reduction function combining two floats.
            start (float, optional): Initial value to start the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that takes a tensor and a dimension index,
            and returns a tensor reduced along that dimension using `fn`.

        """

        def _unimplemented(a: Tensor, b: int) -> Tensor:
            raise NotImplementedError("reduce is not yet implemented.")

        return _unimplemented

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication between two tensors.

        Args:
        ----
            a (Tensor): Left matrix operand tensor.
            b (Tensor): Right matrix operand tensor.

        Returns:
        -------
            Tensor: Resulting tensor from matrix multiplication.

        Raises:
        ------
            NotImplementedError: Always, as this function is not implemented in this assignment.

        """
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
        Callable[[Tensor, Optional[Tensor]], Tensor]:
        A function that applies `fn` elementwise to a tensor, optionally writing to `out`.

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
        Callable[[Tensor, Tensor], Tensor]:
        A function that applies `fn` elementwise to two tensors.

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            start (float, optional): Initial value to start the reduction. Defaults to 0.0

        Returns:
        -------
            A function that takes a Tensor `a` and an integer `dim`, and returns
            a new Tensor reduced along the specified dimension applying fn.

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Perform matrix multiplication between two tensors.

        Args:
        ----
            a (Tensor): Left matrix operand tensor.
            b (Tensor): Right matrix operand tensor.

        Returns:
        -------
            Tensor: Resulting tensor from matrix multiplication.

        Raises:
        ------
            NotImplementedError: Always, as this function is not implemented in this assignment.

        """
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(fn: Callable[[float], float]) -> Any:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
    -------
        None : Fills in `out`

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.array([0] * len(out_shape))
        in_index = np.array([0] * len(in_shape))

        for i in range(len(out)):
            to_index(i, out_shape, out_index)  # Get the multi-dim index for out
            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # Get corresponding in_index (handles broadcasting)

            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)

            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

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
        out_index = np.array([0] * len(out_shape))
        a_index = np.array([0] * len(a_shape))
        b_index = np.array([0] * len(b_shape))

        for i in range(len(out)):
            to_index(i, out_shape, out_index)  # Get the multi-dim index for out

            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out_pos = index_to_position(out_index, out_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
    -------
        None : Fills in `out`

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
        out_index = np.array([0] * len(out_shape))
        a_index = np.array([0] * len(a_shape))

        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            a_index[:] = out_index[
                :
            ]  # Start from the same position as out, then iterate a_index[reduce_dim] to cover all values being reduced

            out_pos = index_to_position(
                out_index, out_strides
            )  # out_strides might not be contiguous, so the logical index i is not always equal to the physical position out_pos in memory

            for j in range(a_shape[reduce_dim]):
                # Walk along reduce dimension
                a_index[reduce_dim] = j

                # Flat position for a
                a_pos = index_to_position(a_index, a_strides)

                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
