from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple, Any

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: Any) -> Tuple:  # type: ignore
    """Converts an input into a tuple if it's not already.

    Args:
    ----
        x: The input, which may or may not be a tuple.

    Returns:
    -------
        A tuple containing the input.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: Tuple) -> Any:  # type: ignore
    """Unwraps a singleton tuple into its sole value, otherwise returns as-is.

    Args:
    ----
        x: A tuple to unwrap.

    Returns:
    -------
        The single element if tuple has one, otherwise the tuple itself.

    """
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Wraps the backward computation into a tuple.

        Args:
        ----
            ctx: Context object containing saved inputs.
            d_out: Derivative from higher in the graph.

        Returns:
        -------
            A tuple of floats representing partial derivatives.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Executes the forward pass of the function.

        Args:
        ----
            ctx: Context to store intermediate variables.
            *inps: Input values as floats.

        Returns:
        -------
            The output of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        """Applies the function to Scalar-like values.

        Args:
        ----
            *vals: A sequence of Scalar or scalar-like values.

        Returns:
        -------
            A Scalar object that wraps the result and tracks computation history.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): Autodiff context to store information for backward pass.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: Result of a + b.

        """
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): Context from forward pass (not used here).
            d_output (float): Gradient of output with respect to some scalar.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to `a` and `b`, both equal to `d_output`.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for natural logarithm.

        Args:
        ----
            ctx (Context): Context to save input `a` for backward computation.
            a (float): Input value.

        Returns:
        -------
            float: log(a)

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for natural logarithm.

        Args:
        ----
            ctx (Context): Context containing saved input `a`.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: Gradient with respect to input `a`.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): Context to save inputs `a` and `b`.
            a (float): First operand.
            b (float): Second operand.

        Returns:
        -------
            float: a * b

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): Context containing `a` and `b`.
            d_output (float): Gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients w.r.t. `a` and `b`.

        """
        a, b = ctx.saved_values
        return operators.mul(d_output, b), operators.mul(d_output, a)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse (1/a).

        Args:
        ----
            ctx (Context): Context to save `a`.
            a (float): Input value.

        Returns:
        -------
            float: 1 / a

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse.

        Args:
        ----
            ctx (Context): Context containing input `a`.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: Gradient w.r.t. `a`.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): Context to save input `a`.
            a (float): Input value.

        Returns:
        -------
            float: -a

        """
        ctx.save_for_backward(a)
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): Context containing input `a`.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: -d_output

        """
        (a,) = ctx.saved_values
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid function.

        Args:
        ----
            ctx (Context): Context to save output of sigmoid.
            a (float): Input value.

        Returns:
        -------
            float: Sigmoid(a)

        """
        result = float(operators.sigmoid(a))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx (Context): Context containing sigmoid output.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: Gradient w.r.t. input.

        """
        (s,) = ctx.saved_values
        return d_output * s * (1 - s)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU function.

        Args:
        ----
            ctx (Context): Context to save input `a`.
            a (float): Input value.

        Returns:
        -------
            float: max(0, a)

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU.

        Args:
        ----
            ctx (Context): Context containing input `a`.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: Gradient w.r.t. input.

        """
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential function.

        Args:
        ----
            ctx (Context): Context to save input `a`.
            a (float): Input value.

        Returns:
        -------
            float: e^a

        """
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential.

        Args:
        ----
            ctx (Context): Context containing input `a`.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: Gradient w.r.t. input `a`.

        """
        (a,) = ctx.saved_values
        return d_output * float(operators.exp(a))


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than comparison.

        Args:
        ----
            ctx (Context): Context to save inputs `a` and `b`.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0

        """
        ctx.save_for_backward(a, b)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than.

        Args:
        ----
            ctx (Context): Context (unused).
            d_output (float): Gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradient since comparison is not differentiable.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): Context to save inputs.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0

        """
        ctx.save_for_backward((a, b))
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality.

        Args:
        ----
            ctx (Context): Context (unused).
            d_output (float): Gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradients since equality is not differentiable.

        """
        return 0.0, 0.0
