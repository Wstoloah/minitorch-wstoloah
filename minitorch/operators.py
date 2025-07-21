"""Collection of the core mathematical operators used throughout the code base."""

# ## Task 0.1
from typing import Callable, Iterable, Any
import numpy as np

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(num1: float, num2: float) -> float:
    """Multiplication of two numbers."""
    return num1 * num2


def id(input: Any) -> Any:
    """Identity function."""
    return input


def add(num1: float, num2: float) -> float:
    """Addition of two numbers."""
    return num1 + num2


def neg(num: float) -> float:
    """Negation of a number."""
    return -num


def lt(num1: float, num2: float) -> float:
    """Check if num1 is less than num2."""
    if num1 < num2:
        return 1.0
    return 0.0


def eq(num1: float, num2: float) -> float:
    """Check if num1 is equal to num2."""
    if num1 == num2:
        return 1.0
    return 0.0


def max(num1: float, num2: float) -> float:
    """Return the maximum of two numbers."""
    return num1 if num1 > num2 else num2


def is_close(num1: float, num2: float, tol: float = 1e-2) -> bool:
    """Check if two numbers are close within a tolerance."""
    return abs(num1 - num2) < tol


def sigmoid(num: float) -> float:
    """Sigmoid function."""
    if num >= 0:
        return 1 / (1 + np.exp(-num))
    return np.exp(num) / (1 + np.exp(num))


def relu(num: float) -> float:
    """ReLU function."""
    if 0 < num:
        return num
    return 0


def log(num: float) -> float:
    """Logarithm (base 10) of a number."""
    if num <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers.")
    return np.log10(num)


def exp(num: float) -> float:
    """Exponential function."""
    return np.exp(num)


def inv(num: float) -> float:
    """Reciprocal (1/num)."""
    if num == 0:
        raise ZeroDivisionError("Cannot compute reciprocal of zero.")
    return 1 / num


def log_back(num: float, d: float) -> float:
    """Derivative of the log function times a second argument."""
    return (1 / num) * d if num > 0 else 0


def inv_back(num: float, d: float) -> float:
    """Derivative of reciprocal times a second argument."""
    return (-1 / (num**2)) * d if num != 0 else 0


def relu_back(num: float, d: float) -> float:
    """Derivative of ReLU times a second argument."""
    return d if num > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(iterable: Iterable, f: Callable) -> Iterable:
    """Applies a given function to each element of an iterable"""
    return (f(x) for x in iterable)


def zipWith(it1: Iterable, it2: Iterable, f: Callable) -> Iterable:
    """ZipWith - Higher-order function that"""
    return (f(x, y) for x, y in zip(it1, it2))


def reduce(iterable: Iterable, f: Callable) -> Any:
    """Reduces an iterable to a single value using a given function"""
    if not iterable:
        return 0
    it = iter(iterable)
    result = next(it)
    for item in it:
        result = f(result, item)
    return result


def negList(list: Iterable[float]) -> Iterable:
    """Negate all elements in a list using map"""
    return map(list, lambda x: -x)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(lst1, lst2, lambda x, y: x + y)


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list using reduce."""
    return reduce(lst, lambda x, y: x + y)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(lst, lambda x, y: x * y)
