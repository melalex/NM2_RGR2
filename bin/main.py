import numpy as np

from bin.svd import *
from bin.householder.householder import *
from bin.constants import *
from bin.svd.svd import svd

# m = np.array(TEST_1)
# a_1 = np.copy(m[:, 0])
# h_1 = householder(a_1)
# m = h_1.dot(m)
# a_2 = np.copy(m[0, :])
# h_2 = householder(a_2, 1)
# m = m.dot(h_2)

print(svd(TEST_1))
