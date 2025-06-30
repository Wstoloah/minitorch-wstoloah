from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the computation history for a `Scalar`.

    This includes the function that created the scalar, the context
    with intermediate saved values, and the scalar inputs to the function.

    Attributes
    ----------
        last_fn (Optional[Type[ScalarFunction]]): The last function used to compute the Scalar.
        ctx (Optional[Context]): Context used to store intermediate values.
        inputs (Sequence[Scalar]): Inputs to the function.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


class Scalar:
    """A differentiable scalar value used for reverse-mode automatic differentiation.

    Scalars behave like floats but track the operations applied to them
    so that gradients can be automatically computed via backpropagation.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        """Constructs a new Scalar.

        Args:
        ----
            v (float): The float value.
            back (ScalarHistory): The history of operations used to produce this value.
            name (Optional[str]): An optional name for identification.

        """
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        """String representation of Scalar."""
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiplies two Scalar values."""
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divides two Scalar values."""
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Computes b / self."""
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        """Adds two Scalar values."""
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        """Returns boolean truthiness of Scalar."""
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less-than comparison."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater-than comparison."""
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        """Equality comparison."""
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtracts one Scalar from another."""
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negates the Scalar value."""
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Right-hand side addition."""
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Right-hand side multiplication."""
        return self * b

    def log(self) -> Scalar:
        """Calculates log."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Calculates Exponential."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Calculates Sigmoid."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Calculates ReLu."""
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if variable is constant, False otherwise."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns iterable of parents of variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies chain rule."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        backward_outputs = h.last_fn._backward(h.ctx, d_output)
        return zip(h.inputs, backward_outputs)

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a Python function.

    This function compares the numerical derivative of `f` with its
    analytical gradient computed via automatic differentiation.

    Args:
    ----
        f (Any): A Python function that supports autodiff.
        *scalars (Scalar): One or more Scalar values to evaluate `f` at.

    Raises:
    ------
        AssertionError: If the numerical and autodiff gradients differ significantly.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
