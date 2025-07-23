from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    permuted = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
    )
    return (
        permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(fn=operators.max, start=-1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax


    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Enables autograd support for max. Only max element gets gradient."""
        int_dim = int(dim.item())
        ctx.save_for_backward(argmax(input, int_dim))
        return max_reduce(input, int_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        (argmax,) = ctx.saved_values
        return grad_output * argmax, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Perform max over a given dimension with autograd support"""
    return Max.apply(input, input._ensure_tensor(dim))


class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute softmax along a given dimension."""
        int_dim = int(dim.item())
        max_val = max(input, int_dim)
        shifted = input - max_val
        exp = shifted.exp()
        sum_exp = exp.sum(int_dim)
        out = exp / sum_exp
        ctx.save_for_backward(out, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute gradient of softmax."""
        softmax_out, dim = ctx.saved_values
        int_dim = int(dim.item())

        # Derivative of softmax: Jacobian-vector product
        dot = (grad_output * softmax_out).sum(int_dim)
        new_shape = list(dot.shape)
        new_shape.insert(int_dim, 1)
        dot = dot.view(*new_shape)
        grad_input = softmax_out * (grad_output - dot)
        return grad_input, 0.0


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.

    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
         softmax tensor

    """
    return Softmax.apply(input, input._ensure_tensor(dim))


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute log softmax along a given dimension."""
        int_dim = int(dim.item())
        max_input = max(input, int_dim)
        shifted = input - max_input
        exp = shifted.exp()
        sum_exp = exp.sum(int_dim)
        logsumexp = sum_exp.log()
        output = shifted - logsumexp
        ctx.save_for_backward(output, dim)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute gradient of log softmax."""
        logsoftmax_out, dim = ctx.saved_values
        int_dim = int(dim.item())

        softmax = logsoftmax_out.exp()  # shape: same as input

        # grad_output: same shape as input
        # sum along int_dim (e.g., across rows)
        sum_grad = grad_output.sum(int_dim)

        # manually reshape sum_grad to be broadcastable
        shape = grad_output.shape
        new_shape = []
        for i, s in enumerate(shape):
            if i == int_dim:
                new_shape.append(1)
            else:
                new_shape.append(s)
        sum_grad = sum_grad.view(*new_shape)

        grad_input = grad_output - softmax * sum_grad
        return grad_input, 0.0


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    $z_i = \log\left(\frac{e^{x_i}}{\sum_i e^{x_i}}\right)$

    Args:
    ----
        input : input tensor
        dim : dimension to apply logsoftmax

    Returns:
    -------
         logsoftmax tensor

    """
    return LogSoftmax.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with randoom positions dropped out

    """
    if ignore:
        return input

    rand_values = rand(input.shape, input.backend, requires_grad=False)
    rand_gate = rand_values > rate
    return input * rand_gate
