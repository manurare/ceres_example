import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")
import pandas as pd

data = pd.read_csv("results.txt", header=None).values
idxs = np.where(data[:, 0] == 0)
obs = data[idxs[0][0]:idxs[0][1], 1:]
curve = data[idxs[0][1]:, 1:]
plt.plot(obs[:, 0], obs[:, 1], 'r*')
plt.plot(curve[:, 0], curve[:, 1], 'b')
plt.legend(['observations', 'fitted curve'])
plt.show()
print("hola")

