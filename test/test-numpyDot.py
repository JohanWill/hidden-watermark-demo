import numpy as np

a = np.array([
    [2,3],
    [4,7]
])
b = np.array([
    [7,9],
    [3,3]
])
c = a * b
print(c)
# [[14 27]
#  [12 21]]

d = np.dot(a,b)
print(d)
# [[23 27]
#  [49 57]]