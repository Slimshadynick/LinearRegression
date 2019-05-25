import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df=pd.read_csv("/home/nikhil/PycharmProjects/LinearRegression/Multivariate-Regression/Train/Train.csv")
data=df.values
print(data.shape)
X=data[:,0:5]
print(X.shape)
Y=data[:,5]
print(Y.shape)
# print(Y)

XTrain,XTest,YTrain,Ytest=train_test_split(X,Y,test_size=0.25)
lr=LinearRegression(normalize=True)
lr.fit(XTrain,YTrain)
theta=lr.coef_
print(theta)
print(theta.shape)
print(lr.intercept_)
YOutput=np.dot(XTrain,theta)
print(YOutput.shape)
print(YOutput)
YOutput=YOutput+lr.intercept_
print(YOutput)
YTestOutput=np.dot(XTest,theta)
YTestOutput=YTestOutput+lr.intercept_
# plt.scatter(XTrain[:,1],YTrain,label="Original")
# plt.scatter(XTrain[:,1],YOutput,label="OutputTrain")
# plt.scatter(XTest[:,1],Ytest,label="Test")
# plt.scatter(XTest[:,1],YTestOutput,label="OutputTest")
print(lr.score(XTrain,YTrain))
print(lr.score(XTest,Ytest))
df=pd.read_csv("/home/nikhil/PycharmProjects/LinearRegression/Multivariate-Regression/Test.csv")
X=df.values
print(X.shape)
Y=np.dot(X,theta)
Y=Y+lr.intercept_
# f=open("/home/nikhil/PycharmProjects/LinearRegression/Multivariate-Regression/YOutput.csv",'w')
list=[]
for i in range(400):
    temp=[]
    temp.append(Y[i])
    list.append(temp)
df = pd.DataFrame(list, columns = [ 'Target'])
df.to_csv("/home/nikhil/PycharmProjects/LinearRegression/Multivariate-Regression/YOutput.csv", sep=',', encoding='utf-8')
plt.scatter(X[:,1],Y)
plt.legend()
plt.show()