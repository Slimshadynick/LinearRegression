from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

boston=load_boston();
X=boston.data
Y=boston.target

print(X.shape)
print(Y.shape)

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2)

print(X_Train.shape)
print(X_Test.shape)

# Train our Regression Model

#1 Create an object
lr=LinearRegression(normalize=True)

# 2 Train our training data
lr.fit(X_Train,Y_Train)

# 3 Output parameters
print(lr.coef_)
print(lr.intercept_)

# 4 Checking how well it has performed
print("Training score")
print(lr.score(X_Train,Y_Train))
print("Test Score")
print(lr.score(X_Test,Y_Test))

# 5 K-fold-CrossValidation

scores=cross_val_score(lr,X_Train,Y_Train,cv=10,scoring="r2")
print(scores)
print(scores.mean())
print(scores.std())