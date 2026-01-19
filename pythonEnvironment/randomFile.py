import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')

def normalization_factor(x, y, z, currentN):
    tmp = 0.0
    if x < np.ceil(currentN / 2):
        tmp = (x + 1)
    else:
        tmp = (currentN - x)
    if y < np.ceil(currentN / 2):
        tmp *= (y + 1)
    else:
        tmp *= (currentN - y)
    if z < np.ceil(currentN / 2):
        tmp *= (z + 1)
    else:
        tmp *= (currentN - z)
    return tmp*tmp

currentN = 10
x_vals = np.arange(0, currentN)
y_vals = np.arange(0, currentN)
z_vals = np.arange(0, currentN)

X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
vectorized_func = np.vectorize(normalization_factor)
values = vectorized_func(X, Y, Z, currentN)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=values.ravel(), cmap='viridis')
plt.colorbar(scatter)
plt.show()
