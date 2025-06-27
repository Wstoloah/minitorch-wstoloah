import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List:
    """Generate a list of `N` random 2D points, where each point has
    two floating-point coordinates between 0 and 1.

    Parameters
    ----------
    N : int
        The number of points to generate.

    Returns
    -------
    List[Tuple[float, float]]
        A list of `N` 2D points represented as tuples of floats.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a dataset of `N` points where each point is classified
    based on its first coordinate. If the first coordinate `x_1` is
    less than 0.5, the label is `1`, otherwise, the label is `0`.

    Parameters
    ----------
    N : int
        The number of points to generate.

    Returns
    -------
    Graph
        A `Graph` object containing the dataset, which includes the
        points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a dataset of `N` points where each point is classified
    based on the sum of its coordinates. If the sum `x_1 + x_2` is
    less than 0.5, the label is `1`, otherwise, the label is `0`.

    Parameters
    ----------
    N : int
        The number of points to generate.

    Returns
    -------
    Graph
        A `Graph` object containing the dataset, which includes the
        points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a dataset of `N` points where each point is classified
    based on its first coordinate. If the first coordinate `x_1` is
    less than 0.2 or greater than 0.8, the label is `1`, otherwise,
    the label is `0`.

    Parameters
    ----------
    N : int
        The number of points to generate.

    Returns
    -------
    Graph
        A `Graph` object containing the dataset, which includes the
        points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a dataset of `N` points where each point is classified
    based on the XOR of its coordinates. The label is `1` if either:
    - `x_1` is less than 0.5 and `x_2` is greater than 0.5, or
    - `x_1` is greater than 0.5 and `x_2` is less than 0.5.

    Parameters
    ----------
    N : int
        The number of points to generate.

    Returns
    -------
    Graph
        A `Graph` object containing the dataset, which includes the
        points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a dataset of `N` points where each point is classified
    based on its distance from the center `(0.5, 0.5)`. The points
    inside a circle of radius `0.1` are labeled `0`, and the points
    outside the circle are labeled `1`.

    Parameters
    ----------
    N : int
        The number of points to generate.

    Returns
    -------
    Graph
        A `Graph` object containing the dataset, which includes the
        points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a dataset of `N` points arranged in a spiral pattern.
    The first half of the points form one spiral, and the second
    half forms another. The first spiral is labeled `0` and the second
    spiral is labeled `1`.

    Parameters
    ----------
    N : int
        The number of points to generate (must be an even number).

    Returns
    -------
    Graph
        A `Graph` object containing the dataset, which includes the
        points and their labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
