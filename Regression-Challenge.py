import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readData(filename):
    df=pd.read_csv(filename)
    return df.values

X=readData("/home/nikhil/PycharmProjects/LinearRegression/Training Data/Linear_X_Train.csv");
# print(X)
# print(X.shape)
X=X.reshape(3750)
# print(X)
Y=readData("Training Data/Linear_Y_Train.csv")
# print(Y)
# print(Y.shape)
Y=Y.reshape(3750)
 # print(Y)
plt.scatter(X,Y)
plt.show()

def hypothesis(x,theta):
    return theta[0]+theta[1]*x

def error(X,Y,theta):
    err=0
    m=X.shape[0]
    for i in range(m):
        h=hypothesis(X[0],theta)
        err+=h-Y[i]
    return err

def gradient(X,Y,theta):
    grad=np.array([0.0,0.0])
    m=X.shape[0]
    for i in range(m):
        # print(Y[i])
        h=hypothesis(X[i],theta)
        grad[0]=grad[0]+(h-Y[i])
        grad[1]=grad[1]+((h-Y[i])*X[i])
    return grad

def gradientDescent(X,Y,alpha,Max_Iterations):
    theta=np.array([0.0,0.0])
    grad=np.array([0.0,0.0])
    for i in range(Max_Iterations):
        grad=gradient(X,Y,theta)
        theta[0]=theta[0]-grad[0]*alpha
        theta[1]=theta[1]-grad[1]*alpha
    return theta

theta=gradientDescent(X,Y,0.00001,500)
YP=theta[0]+theta[1]*X
# print(YP.shape)
# print(Y)
# print(YP)
plt.scatter(X,Y,label="Original")
plt.scatter(X,YP,label="Line")
print(error(X,Y,theta))
XTest=readData("Test Cases/Linear_X_Test.csv")
XTest=XTest.reshape(1250)
f=open("Output.csv",'w')
YTest=theta[0]+theta[1]*XTest
tempList=[]
print(YTest.shape)
for i in range(1250):
    str1=str(YTest[i])+"\n"
    tempList.append(str1)
f.writelines(tempList)
plt.scatter(XTest,YTest,label="test")
f.close()
plt.legend()
plt.show()

