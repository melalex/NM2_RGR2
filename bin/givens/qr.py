import numpy as np

from bin.givens.rotation import rotation


def qr(a):
    n, m = np.shape(a)

    q = np.identity(n)
    r = np.copy(a)

    rows, cols = np.tril_indices(n, -1, m)
    for row, col in zip(rows, cols):
        if r[row, col] != 0:
            g = rotation(r, row, col)

            r = np.dot(g, r)
            q = np.dot(q, g.T)

    return q, r
