"""Numerical oracles shared by tests."""

from __future__ import annotations

import numpy as np


def central_difference_jacobian(function, point, step=1e-5):
    columns = []
    for axis in np.eye(point.size):
        columns.append((function(point + step * axis) - function(point - step * axis)) / (2 * step))
    return np.stack(columns, axis=1)
