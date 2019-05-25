import matplotlib.pyplot as plt
import numpy as np

#Multivariate normal distribution is used when we have more than one feature
# Suppose we have two features, then we need to provide two means and 2*2 covariance matrix

u1=np.array([2,4])
cov_1=np.array([[1,-0.6],[-0.6,1]])
apple=np.random.multivariate_normal(u1,cov_1,400)
print(apple.shape)

u2=np.array([7,12])
cov_2=np.array([[1,0],[0,1]])
mango=np.random.multivariate_normal(u2,cov_2,300)
print(mango.shape)
plt.scatter(mango[:,0],mango[:,1],label="Mango")
plt.scatter(apple[:,0],apple[:,1],label="Apple")
plt.legend()
plt.show()

#Generating Y and combining apples and mangoes into fruits
Y=np.zeros(700)
Y[:400]=1
X=np.vstack((apple,mango))
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
print(X.shape)