import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np


xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints)
plt.show()