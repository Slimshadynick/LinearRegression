import matplotlib.pyplot as plt
import numpy as np

#Uniform distribution
x=np.random.randint(10,20,300)
plt.hist(x)

#Normal Distribution
mean=15
std=5
x=np.random.randn(300)*std+mean
noise=np.random.randn(300)
m=5
c=1
y=m*x+c
y=y+noise
plt.hist(x)
plt.scatter(x,y)
plt.show()