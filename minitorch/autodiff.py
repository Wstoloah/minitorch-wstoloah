from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_forward = list(vals)
    vals_backward = list(vals)
    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon
    return (f(*vals_forward) - f(*vals_backward)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative value into the variable."""
        pass

    @property
    def unique_id(self) -> int:
        """Returns unique id."""
        return 0

    def is_leaf(self) -> bool:
        """Returns True if a leaf, False otherwise."""
        return False

    def is_constant(self) -> bool:
        """Returns true if variable is constant, False otherwise."""
        return False

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns parents of the variable."""
        return []

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Applies chain rule."""
        return []


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def dfs(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            dfs(parent)
        order.append(var)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation to compute derivatives for leaf nodes.

    Args:
    ----
        variable (Variable): The final variable in the computation graph.
        deriv (Any): The initial derivative to propagate backward.

    """
    # Get topological order of non-constant variables
    topo_order = list(topological_sort(variable))

    # Dictionary to hold the gradients of each variable
    derivatives = {var.unique_id: 0 for var in topo_order}
    derivatives[variable.unique_id] = deriv

    for var in topo_order:
        d_output = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for parent, d_input in var.chain_rule(d_output):
                derivatives[parent.unique_id] += d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns saved tensors."""
        return self.saved_values
