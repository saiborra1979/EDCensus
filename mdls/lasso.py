"""
MODEL CLASS TO FIT LASSO WITH GLMNET MODULE
https://pypi.org/project/glmnet/
"""

import os
import pickle
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import norm
from glmnet import ElasticNet

class mdl():
    def __init__(self, model, lead, date, cn):
        self.enc = StandardScaler(copy=True)
        self.isfit, self.istuned = False, False
        self.model, self.lead, self.date, self.cn = model, lead, date, cn
        # Construct the unique filename
        self.fn = 'mdl_' + self.model + '_' + self.date.strftime('%Y_%m_%d') + '_lead_' + str(self.lead) + '.pkl'

    def fit(self, X, y, nlam=50):
        n, p = X.shape
        Xtil = self.enc.fit_transform(X)  # Normalize the data
        # Train model
        self.mdl = ElasticNet(alpha=1, standardize=False, fit_intercept=True, n_splits=0,
                           n_lambda=nlam, min_lambda_ratio=0.001)
        print('Fitting model')
        tstart = time()
        self.mdl.fit(X=Xtil, y=y)
        print('Model took %i seconds to train' % (time() - tstart))
        # Get residual distribution
        e2_holder = np.zeros(self.mdl.n_lambda_)
        for jj, ll in enumerate(self.mdl.lambda_path_):
            eta = Xtil.dot(self.mdl.coef_path_[:, jj]) + self.mdl.intercept_path_[jj]
            e2_holder[jj] = np.sqrt(np.sum(np.square(y - eta)))
        # Expected number of false discoveries
        lams2 = n * self.mdl.lambda_path_ / e2_holder
        nsupp = np.sum(self.mdl.coef_path_ != 0, 0)
        efd = norm.cdf(-lams2) * p
        fdr = efd / nsupp
        self.res = pd.DataFrame({'fdr': fdr, 'nsel': nsupp,
                                 'lam': self.mdl.lambda_path_, 'idx': range(nlam)})
        self.res = self.res[self.res.fdr.notnull()]
        self.isfit = True

    def tune(self, X, y, fdr=None):
        assert self.isfit
        if (X is None) & (y is None):
            print('Tuning using residuals only')
            jj_star = int(self.res.iloc[np.argmin((self.res.fdr - fdr) ** 2)].idx)
        elif fdr is None:
            print('Tuning based on validation set')
            Xtil = self.enc.transform(X)
            r2_holder = np.zeros(self.res.shape[0])
            for ii, jj in enumerate(self.res.idx):
                eta = Xtil.dot(self.mdl.coef_path_[:, jj]) + self.mdl.intercept_path_[jj]
                r2_holder[ii] = r2_score(y, eta)
            jj_star = np.argmax(r2_holder)
            self.res['r2'] = r2_holder
        else:
            print('Error! Set fdr to None, or X/y to None')
            assert False
        print(self.res.loc[jj_star])
        # print(self.res)
        self.bhat = self.mdl.coef_path_[:, jj_star]
        self.ahat = self.mdl.intercept_path_[jj_star]
        self.istuned = True

    def predict(self, X):
        assert self.isfit & self.istuned
        Xtil = self.enc.transform(X)
        eta = Xtil.dot(self.bhat) + self.ahat
        return eta

    def save(self, folder):
        """
        SAVES THE COEFFICIENTS OF THE MODEL
        """
        assert self.isfit & self.istuned
        self.df_bhat = pd.Series(self.bhat, index=self.cn).reset_index()
        self.df_bhat = self.df_bhat.rename(columns={'level_0': 'cn', 'level_1': 'lag', 0: 'bhat_z'})
        self.df_bhat['lag'] = self.df_bhat.lag.str.replace('lag_', '').astype(int)
        self.df_bhat = self.df_bhat.assign(bhat=lambda x: x.bhat_z / self.enc.scale_,
                                           model=self.model, lead=self.lead, date=self.date)
        print('Pickling!')
        with open(os.path.join(folder, self.fn), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        #df_bhat.to_csv(os.path.join(folder, fn), index=False)

    def load(self, folder):
        """
        SET EXISTING CLASS ATTRIBUTES TO MATCH
        """
        with open(os.path.join(folder, self.fn), 'rb') as input:
            tmp = pickle.load(input)
        for attr in list(vars(tmp)):
            setattr(self, attr, getattr(tmp, attr))

