import numpy as np

from math import hypot


def rotation(matrix, row, col):
    a = matrix[col, col]
    b = matrix[row, col]
    n = len(matrix)

    r = hypot(a, b)
    c = a / r
    s = -b / r

    g = np.identity(n)
    g[[col, row], [col, row]] = c
    g[row, col] = s
    g[col, row] = -s

    return g
