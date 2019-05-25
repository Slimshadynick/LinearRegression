import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

a=[1,2,3]
b=[4,5,6,7]
a,b=np.meshgrid(a,b)
print(a)
print(b)


fig=plt.figure()
axes=fig.gca(projection="3d")
axes.plot_surface(a,b,a**2+b**2)
plt.show()
