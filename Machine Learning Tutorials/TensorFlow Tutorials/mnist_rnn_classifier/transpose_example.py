import numpy as np

x = np.ones((1, 2, 3))  # A 1 by 2 by 3 array of 1s.

print(x)
print()
print(np.transpose(x, (1, 0, 2)))

'''

[[

    [ 1.  1.  1.],
    [ 1.  1.  1.]

]]

[

    [[ 1.  1.  1.]],
    [[ 1.  1.  1.]]

]

'''
