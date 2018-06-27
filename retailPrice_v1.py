import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
boston.data.shape
print(boston.feature_names)
print(boston.DESCR)

bos = pd.DataFrame(boston.data)
bos.head()
bos.columns = boston.feature_names
boston.target[:5]
bos['PRICE'] = boston.target
bos.head()

from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis=1)
lm = LinearRegression()
lm
lm.fit(X, bos.PRICE)
lm.intercept_
print(len(lm.coef_))
print(lm.coef_)
pd.DataFrame(X.columns, lm.coef_)

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms for dwelling (RM)")
plt.ylabel("Housing price")
plt.title("Relationship between RM and price")
plt.show()

lm.predict(X)[0:5]

plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("Observed prices")
plt.ylabel("Predicted prices")
plt.title("Observed prices Vs. Predicted prices")
plt.show()

mseFull = np.mean((bos.PRICE - lm.predict(X))**2)
print(mseFull)

lm = LinearRegression()
lm.fit(X[['PTRATIO']], bos.PRICE)
msePTRATIO = np.mean((bos.PRICE - lm.predict(X[['PTRATIO']]))**2)
print(msePTRATIO)

## Training and validation sets
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, bos.PRICE, 
                                                                             test_size = 0.33, random_state = 5)
X_train.shape
X_test.shape
y_train.shape
y_test.shape


lm = LinearRegression()
lm.fit(X_train, y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

np.mean((y_train - pred_train)**2)
np.mean((y_test - pred_test)**2)

## Use validation set approach and analyze train and test errors

# Compute RMSE using 10-fold x-validation
from sklearn.cross_validation import KFold

X = bos.drop('PRICE', axis=1)
y = bos.PRICE
X = np.array(X)
y = np.array(y)

kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

xval_err = 0
for train_index, test_index in kf.split(X):
    #print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lm.fit(X_train, y_train)
    p = lm.predict(X_test)
    e = p-y_test
    xval_err += np.dot(e,e)
rmse_10cv = np.sqrt(xval_err/len(x))
print(rmse_10cv)


################
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import numpy as np
import pylab as pl

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.feature_names)
print(boston.data.shape)
print(boston.target.shape)
np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)
print(boston.data)
# In order to do multiple regression we need to add a column of 1s for x0
x = np.array([np.concatenate((v,[1])) for v in boston.data])
y = boston.target
# First 10 elements of the data
print(x[:10])
# First 10 elements of the response variable
print(y[:10])
# Create linear regression object
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(x,y)
# Predictions for the first 10 instances
print(linreg.predict(x[:10]))
# Compute RMSE on training data
# p = np.array([linreg.predict(xi) for xi in x])
p = linreg.predict(x)
# Now we can constuct a vector of errors
err = abs(p-y)

# Let's see the error on the first 10 predictions
print(err[:10])
# Dot product of error vector with itself gives us the sum of squared errors
total_error = np.dot(err,err)
# Compute RMSE
rmse_train = np.sqrt(total_error/len(p))
print(rmse_train)
# We can view the regression coefficients
print('Regression Coefficients:') 
print(linreg.coef_)

# Plot outputs
pl.plot(p, y,'ro')
pl.plot([0,50],[0,50], 'g-')
pl.xlabel('predicted')
pl.ylabel('real')
pl.show()

# Now let's compute RMSE using 10-fold x-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
    linreg.fit(x[train],y[train])
    # p = np.array([linreg.predict(xi) for xi in x[test]])
    p = linreg.predict(x[test])
    e = p-y[test]
    xval_err += np.dot(e,e)
    
rmse_10cv = np.sqrt(xval_err/len(x))

method_name = 'Simple Linear Regression'
print('Method: %s' %method_name)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 10-fold CV: %.4f' %rmse_10cv)