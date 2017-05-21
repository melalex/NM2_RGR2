import numpy as np

from math import sqrt


def householder(vec, k=0):
    a = np.empty_like(vec, dtype=float)
    a[k:] = vec[k:]
    beta = np.copysign(np.linalg.norm(a[k:]), -a[k])
    mu = 1 / sqrt(2 * beta * beta - 2 * beta * a[k])
    a[k] -= beta
    w = mu * a[np.newaxis]
    return np.eye(a.shape[0]) - 2 * w.T.dot(w)
