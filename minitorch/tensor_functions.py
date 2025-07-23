"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> Tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Apply the function to the given tensors."""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(no_grad=not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negation of the input tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise reciprocal of the tensor t1.
        Saves the input for use in the backward pass.
        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Gradient of reciprocal: d/dx (1/x) = -1 / x^2"""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise addition of tensors t1 and t2."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Gradient of addition is just the incoming gradient passed to both inputs."""
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of tensors a and b.
        Saves both inputs for gradient computation.
        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        Gradient is grad_output * other_tensor
        """
        a, b = ctx.saved_values
        return grad_output.f.mul_zip(b, grad_output), grad_output.f.mul_zip(
            a, grad_output
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the sigmoid function element-wise: 1 / (1 + exp(-x))"""
        sig = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Derivative: sigmoid(x) * (1 - sigmoid(x))"""
        (output,) = ctx.saved_values
        ones = output.ones(output.shape)
        return grad_output.f.mul_zip(
            output, output.f.add_zip(ones, output.f.neg_map(output))
        )


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the ReLU function element-wise: max(0, x)"""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Derivative: 1 if x > 0 else 0"""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies natural logarithm element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Derivative: 1 / x"""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies the exponential function element-wise."""
        output = t1.f.exp_map(t1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Derivative of exp(x) is exp(x)"""
        (output,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Reduces tensor `a` by summing along the given dimension."""
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for sum distributes the gradient across the original shape."""
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Checks whether all elements are non-zero along a given dimension.
        Uses multiplication as a logical AND over boolean values.
        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise less-than comparison: returns tensor of booleans."""
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Less-than is non-differentiable, return zero gradients."""
        # Return zero tensors with same shape and backend
        zero = zeros(grad_output.shape, backend=grad_output.backend)
        return zero, zero


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise equality check: returns tensor of booleans."""
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Equality is non-differentiable, return zero gradients."""
        # Return zero tensors with same shape and backend
        zero = zeros(grad_output.shape, backend=grad_output.backend)
        return zero, zero


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Returns a boolean tensor where a and b are approximately equal."""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Rearranges dimensions of tensor `a` based on `order`."""
        order_data = [int(order[i]) for i in range(order.size)]
        ctx.save_for_backward(a.shape)
        tensor_data = a._tensor.permute(*order_data)
        return minitorch.Tensor(tensor_data, backend=a.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Gradient of permute is the inverse permutation."""
        (original_shape,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage,
                original_shape,
                backend=grad_output.backend,
            ),
            0.0,
        )


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor `a` to the new shape provided in `shape`."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Reshape the gradient to the original input shape stored in the context."""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Return a copy of the input tensor `a`."""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient as-is for the copy operation."""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform matrix multiplication between tensors `t1` and `t2`."""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for matrix multiplication with respect to both inputs."""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        np.array([0] * int(operators.prod(shape))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1, ind: UserIndex
) -> float:
    """Estimate the gradient using central difference approximation."""
    print(
        f"Estimating gradient for {f.__name__ if hasattr(f, "__name__") else str(f)} at index {ind} with epsilon {epsilon}"
    )
    x = vals[arg]
    up = zeros(x.shape)
    up._tensor._storage[up._tensor.index(ind)] = epsilon
    vals_up = [x if j != arg else x + up for j, x in enumerate(vals)]
    print(f"vals_up = {vals_up}")
    vals_down = [x if j != arg else x - up for j, x in enumerate(vals)]
    print(f"vals_down = {vals_down}")

    up_val = f(*vals_up).sum().item()
    print(f"up_val = {up_val:.10f}")
    down_val = f(*vals_down).sum().item()
    print(f"down_val = {down_val:.10f}")
    # name = f.__name__ if hasattr(f, "__name__") else str(f)
    # if name in ["cube", "square"]:
    #     print(f"------------------{name}---------------------")
    #     print(f"original val = {vals}")
    #     print(f"f(*vals_up) = {up_val:.10f}")
    #     print(f"f(*vals_down) = {down_val:.10f}")
    delta: Tensor = f(*vals_up).sum() - f(*vals_down).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check that the gradients computed match numerical estimates."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()

    random.seed(10)
    out = f(*vals)
    out.sum().backward()

    err_msg = """
-------------------- GRADIENT CHECK FAILED --------------------
Function: {func}
Input tensor {arg_index} (sample index = {ind})
Expected (numerical) derivative: {expected:.6f}
Received (autograd) derivative: {received:.6f}
---------------------------------------------------------------
"""
    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        actual = x.grad[ind]
        np.testing.assert_allclose(
            actual,
            check,
            rtol=1e-1,
            atol=1e-1,
            err_msg=err_msg.format(
                func=f.__name__ if hasattr(f, "__name__") else str(f),
                arg_index=i,
                ind=ind,
                expected=check,
                received=actual,
            ),
        )
