"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Enables or disables gradient tracking for this tensor.

        Args:
        ----
            x (bool): If True, gradients will be tracked during operations.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns True if this tensor is tracking gradients.

        Returns
        -------
            bool: Whether gradient tracking is enabled.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns
        int : size of the tensor

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns
        int : dimensionality of the tensor

        """
        return self._tensor.dims

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(
            b, (int, float)
        ):  # ensures that the input b is a tensor, even if it starts as a plain Python number like an int or float
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks whether all (or along a given dimension) elements in the tensor are non-zero (truthy)"""
        if dim is None:
            return All.apply(
                self.view(self.size), self._ensure_tensor(0)
            )  # self.view(self.size) flattens the tensor to 1D
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Element-wise check if tensor values are close to those in `y`."""
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm function element-wise."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise."""
        return Exp.apply(self)

    def item(self) -> float:
        """Returns the single scalar value as a Python float."""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum over dimension `dim`"""
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over dimension `dim`"""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permute tensor dimensions to *order"""
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size"""
        return View.apply(self, tensor(list(shape)))

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.
        ie. it is used during backpropagation to reverse the effects of broadcasting

        Args:
        ----
            other (Tensor): The tensor from the backward pass to be broadcast.

        Returns:
        -------
            Tensor: A new tensor broadcast to match `self`'s shape.

        """
        # Case 1: Both the same shape.
        if (
            self.shape == other.shape
        ):  # If the original input and the gradient have the same shape, nothing needs to change
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(
            other, buf
        )  # id_map copies other into buf, broadcasting as needed
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor of zeros with the given shape.

        Args:
        ----
            shape (Optional[UserShape]): Desired shape of the tensor.

        Returns:
        -------
            Tensor: A tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                np.zeros(int(operators.prod(shape)), dtype=np.float64),
                shape,
                backend=self.backend,
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def ones(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor of ones with the given shape.

        Args:
        ----
            shape (Optional[UserShape]): Desired shape of the tensor.

        Returns:
        -------
            Tensor: A tensor filled with zeros.

        """

        def one(shape: UserShape) -> Tensor:
            return Tensor.make(
                [1.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = one(self.shape)
        else:
            out = one(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns the internal storage, shape, and strides of the tensor.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: Tensor metadata.

        """
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Returns a new tensor detached from the computation graph.

        Returns
        -------
            Tensor: A copy of the tensor without autograd history.

        """
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is a constant (not requiring gradients).

        Returns
        -------
            bool: True if tensor is constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the input variables that produced this tensor.

        Returns
        -------
            Iterable[Variable]: List of parent variables in the graph.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes gradients for each input using the chain rule.

        Given an output gradient `d_output`, this method calls the backward
        function of the operation that created this variable, expands the
        resulting gradients to match input shapes, and returns them paired
        with their corresponding input variables.

        Returns
        -------
            Iterable of (input_variable, input_gradient) pairs.

        """
        h = self.history
        assert h is not None  # not a constant
        assert h.last_fn is not None  # not a leaf node
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert (
            len(x) == len(h.inputs)
        ), (
            f"Bug in function {h.last_fn}"
        )  # sanity check: number of gradients = number of inputs
        return [
            (
                inp,
                inp.expand(self._ensure_tensor(d_in)),
            )  # pair each input variable inp with its gradient d_in
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Initiates backpropagation on this tensor.

        Args:
        ----
            grad_output (Optional[Tensor]): Gradient of the output. Defaults to 1 if tensor is scalar.

        Returns:
        -------
            None

        """
        if grad_output is None:  # starting point of a backpropagation
            assert (
                self.shape == (1,)
            ), "Must provide grad_output if non-scalar"  # you can’t backprop from a vector or matrix without knowing which direction
            grad_output = Tensor.make(
                [1.0], (1,), backend=self.backend
            )  # the gradient of the output with respect to itself is just 1.0
        backpropagate(self, grad_output)

    def zero_grad_(self) -> None:  # pragma: no cover
        """Reset the derivative on this variable."""
        self.grad = None
