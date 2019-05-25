import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readData(filename):
    df=pd.read_csv(filename)
    return df.values

# np.seterr(all="warn")
x=readData("Training Data/Linear_X_Train.csv")
#print(x)
# print(x.shape)
x.reshape((3750,))
y=readData("Training Data/Linear_Y_Train.csv")
# print(y.shape)
y.reshape((3750,))

#Normalisaion
x=x-x.mean()/x.std()


#Algorithm Linear Regression

def hypothesis(theta,x):
    return theta[0] + theta[1]*x

def error(X,Y,theta):
    m=X.shape[0]
    total_err=0

    for i in range(m+1):
        total_err+=(Y[i]-hypothesis(theta,X[i]))**2
    return total_err/2

def gradient(X,Y,theta):
    m=X.shape[0]
    grad=np.array([0.0,0.0])
    for i in range(m):
        h=hypothesis(theta,X[i])
        y=Y[i]
        grad[0]+=(h-y)
        grad[1]+=(h-y)*X[i]
    return grad

def gradientDescent(X,Y,alpha,Max_Iter):
    theta=np.array([0.0,0.0])
    grad=np.array([0.0,0.0])
    for i in range(Max_Iter):
        grad=gradient(X,Y,theta)
        theta[0]=theta[0]-grad[0]*alpha
        theta[1]=theta[1]-grad[1]*alpha
    return theta

theta=gradientDescent(x,y,0.001,500)
print(theta)

#Plotting
plt.scatter(x,y)
plt.scatter(x,hypothesis(theta,x))
plt.show()
