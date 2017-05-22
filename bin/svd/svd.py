import numpy as np

from bin.householder.householder import householder


def __first_step_shape(n, m):
    u_row, u_col = np.triu_indices(n, 2, m)
    l_row, l_col = np.tril_indices(n, -1, m)
    return np.unique(l_col).size, np.unique(u_row).size


def svd(matrix):
    a = np.array(matrix)
    n, m = a.shape
    n_, m_ = __first_step_shape(n, m)

    # First step
    for i in range(max(m_, n_)):
        if i < n_:
            h = np.eye(n)
            h[i:, i:] = householder(a[i:, i])
            a = h.dot(a)

        if i < m_:
            h = np.eye(m)
            h[i + 1:, i + 1:] = householder(a[i, i + 1:])
            a = a.dot(h)

    return
