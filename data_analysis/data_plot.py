import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data

by = data.read_Btimeseries()
i = 0
for By in by:
    i = i + 1

print(i)
By = np.array(by)
length = np.size(by[0])

x = np.linspace(0, 1, num=length)
plt.plot(x, By[3500])
plt.show()
