import numpy as np

from math import sqrt


def householder(a):
    # a = np.empty_like(vec, dtype=float)
    # a[k:] = vec[k:]
    # beta = np.copysign(np.linalg.norm(a[k:]), -a[k])
    # mu = 1 / sqrt(2 * beta * beta - 2 * beta * a[k])
    # a[k] -= beta
    # w = mu * a[np.newaxis]
    # return np.eye(a.shape[0]) - 2 * w.T.dot(w)
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    h = np.eye(a.shape[0])
    h -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return h