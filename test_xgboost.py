# Compare different XGBoost models

from sklearn.ensemble import GradientBoostingRegressor as gbr1
import xgboost as xgb
from xgboost import XGBRegressor as gbr2
from sklearn.datasets import load_boston
from time import time
import pandas as pd
import numpy as np

# load_boston(return_X_y=True)
ntrain, ntest = 100000, 100
n = ntrain+ntest
p = 10
np.random.seed(n)
X = np.random.randn(ntrain+ntest, p)
y = np.cos(X.sum(1))
Xtrain, ytrain = X[:ntrain], y[:ntrain]
Xtest, ytest = X[ntrain:], y[ntrain:]

# xgb_yXtrain = xgb.DMatrix(data=Xtrain,label=ytrain)
# xgb_Xtest = xgb.DMatrix(data=Xtest)

# Define parameters
n_trees = 10
max_depth = 10
eta = 0.1
n_core = 10

# Set up models
mdl1 = gbr1(criterion='friedman_mse',random_state=n,
    learning_rate=eta, max_depth=max_depth, n_estimators=n_trees,
    max_features='auto')
mdl2 = gbr2(objective='reg:squarederror', random_state=n,
    learning_rate=eta, max_depth=max_depth, n_estimators=n_trees, 
    colsample_bytree=1,
    n_jobs=n_core, booster='gbtree')

# Fit different models
stime = time()
mdl2.fit(X=Xtrain,y=ytrain)
print('mdl2=%0.1f' % (100*(time() - stime)))

stime = time()
mdl1.fit(X=Xtrain,y=ytrain)
print('mdl1=%0.1f' % (100*(time() - stime)))

# Compare the fit
eta1 = mdl1.predict(Xtest)
eta2 = mdl2.predict(Xtest)
pd.DataFrame({'mdl1':eta1, 'mdl2':eta2}).corr()




