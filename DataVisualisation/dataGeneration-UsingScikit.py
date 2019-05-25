from sklearn.datasets import make_regression,make_classification,make_blobs,make_moons
from matplotlib import pyplot as plt

#Regression
[X,Y]=make_regression(n_samples=100,n_features=1,n_informative=1,noise=0.8)
print(X.shape)
print(Y.shape)
plt.scatter(X,Y)
plt.show()

#Classification
[XC,YC]=make_classification(n_samples=1000,n_features=2,n_informative=2,n_redundant=0,n_classes=3,n_clusters_per_class=1,
                            random_state=5)
print(XC.shape)
print(YC.shape)
plt.scatter(XC[:,0],XC[:,1],c=YC)
plt.show()


#Clustering Generation
[XB,YB]=make_blobs(n_samples=400,n_features=2,centers=5)
plt.scatter(XB[:,0],XB[:,1],c=YB)
plt.show()

#Using Subplots
plt.subplot(221)
plt.scatter(X,Y)
plt.subplot(222)
plt.scatter(XC[:,0],XC[:,1],c=YC)
plt.subplot(223)
plt.scatter(XB[:,0],XB[:,1],c=YB)
plt.show()