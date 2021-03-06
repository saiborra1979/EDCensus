import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR
from scipy.stats import spearmanr

# self=model(encoder=enc_yX, lead=lead, lag=lag, di_model=di_model)
class model():
    def __init__(self, encoder, lead=24, lag=24, di_model=None):
        self.enc = encoder
        self.lead = np.arange(1,lead+1)
        self.k = lead
        self.lag = lag
        self.isfit = False
        self.rdrop = min(self.lead)
        # Set up with the defaults
        self.di_model = {'n_trees':100, 'depth':3}
        if di_model is not None:
            for k, v in di_model.items():
                assert k in self.di_model
                self.di_model[k] = int(di_model[k])

    # self = regressor; X=Xtrain.copy(); y=ytrain.copy()
    def fit(self, X, y):
        assert len(X) == len(y)
        Xtil = self.enc.transform_X(X, rdrop=self.rdrop)
        Ytil = self.enc.transform_y(y, rdrop=self.rdrop)
        assert len(Xtil) == len(Ytil)
        self.di_mdl = {}
        for k in self.lead:
            ytil = Ytil[:,k-1]
            idx_k = ~np.isnan(ytil)
            self.di_mdl[k] = GBR(random_state=k, 
                n_estimators=self.di_model['n_trees'], 
                max_depth=self.di_model['depth'])
            self.di_mdl[k].fit(Xtil[idx_k],ytil[idx_k])
    
    # X = Xtrain[-25:].copy()
    def predict(self, X):
        Xtil = self.enc.transform_X(X,rdrop=0)
        pred = np.vstack([self.di_mdl[k].predict(Xtil) for k in self.lead]).T
        if hasattr(self.enc,'enc_Y'):
            pred = self.enc.inverse_transform_y(pred)
        return pred

# from sklearn.datasets import make_regression
# from sklearn.multioutput import MultiOutputRegressor, RegressorChain
# X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
# qq = MultiOutputRegressor(GBR(random_state=0))
# qq.fit(X, y, sample_weight=np.random.rand(10,3))
# qq.predict(X[0:2])

# X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
# # y[8,2] = np.NaN
# gbm = GBR(random_state=0)
# chain = RegressorChain(base_estimator=gbm, order=[0, 1, 2])
# chain.fit(X, y,sample_weights=None)


