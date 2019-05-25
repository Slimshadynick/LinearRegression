from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

mnist=load_digits()
X=mnist.data
Y=mnist.target
print(X.shape)
print(Y.shape)
image=X[0].reshape((8,8))
print(Y[0])
plt.imshow(image,cmap="gray")
plt.show()