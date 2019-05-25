import matplotlib.pyplot as plt
import numpy as np
a=np.array([1,2,3,4,5])
b=a**2

plt.plot(a,b,color="orange",label="Stock Price")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sample Data")
plt.legend()
plt.show()