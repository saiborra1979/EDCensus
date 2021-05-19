"""
FUNCTION TO IMPLEMENT NON-PARAMETRIC NADARYA-WATSON ESTIMATOR

https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0550
http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
http://www.cs.cmu.edu/~epxing/Class/10708-16/note/10708_scribe_lecture20.pdf
https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb
"""

import os
import pickle
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from scipy.stats import norm
from glmnet import ElasticNet
from funs_support import cvec
from joblib import Parallel, delayed
import warnings


def nadarya(x, X, y, ll=0.5):
    """
    FUNCTION TO IMPLEMENT NADARYA-WATSON LINEAR SMOOTHER
    """
    assert (x.shape[1] == X.shape[1]) & (X.shape[0] == y.shape[0])
    dist_l2 = pairwise_distances(X=x, Y=X, metric='l2')
    w_gauss = np.exp(-0.5 * dist_l2 ** 2 / ll).T
    ytil = np.sum((w_gauss * cvec(y)), 0) / w_gauss.sum(0)
    return ytil


# Can we parallelize this???
def local_reg(x, X, y, ll=0.5):
    assert (x.shape[1] == X.shape[1]) & (X.shape[0] == y.shape[0])
    dist_l2 = pairwise_distances(X=x, Y=X, metric='l2')
    w_gauss = np.exp(-0.5 * dist_l2 ** 2 / ll).T.flatten()
    # endog = x - X
    mdl = LinearRegression(fit_intercept=True).fit(X, y, w_gauss)
    return mdl.predict(x)

def dist_weight(x, X, ll=0.5):
    assert x.shape[1] == X.shape[1]
    dist_l2 = pairwise_distances(X=x, Y=X, metric='l2')
    w_gauss = np.exp(-0.5 * dist_l2 ** 2 / ll).T.flatten()


def vprint(stmt, verbose=True):
    if verbose:
        print(stmt)

def fpc(X, y, w, nlam=50, fdr=0.1, verbose=True):
    warnings.filterwarnings('ignore')
    # Train model
    elnet = ElasticNet(alpha=1, standardize=False, fit_intercept=True, n_splits=0,
                       n_lambda=nlam, min_lambda_ratio=0.001)
    vprint('Fitting model', verbose)
    tstart = time()
    elnet.fit(X=X, y=y, sample_weight=w)
    vprint('Model took %i seconds to train' % (time() - tstart), verbose)
    # Get residual distribution
    e2_holder = np.zeros(elnet.n_lambda_)
    for jj, ll in enumerate(elnet.lambda_path_):
        eta = X.dot(elnet.coef_path_[:, jj]) + elnet.intercept_path_[jj]
        e2_holder[jj] = np.sqrt(np.sum(np.square(y - eta)))
    # Expected number of false discoveries
    lams2 = X.shape[0] * elnet.lambda_path_ / e2_holder
    nsupp = np.sum(elnet.coef_path_ != 0, 0)
    efd = norm.cdf(-lams2) * X.shape[1]
    efdr = efd / nsupp
    res = pd.DataFrame({'fdr': efdr, 'nsel': nsupp, 'lam': elnet.lambda_path_, 'idx': range(nlam)})
    res = res[res.fdr.notnull()]
    jj = int(res.iloc[np.argmin((res.fdr - fdr)**2)].idx)
    bhat, ahat = elnet.coef_path_[:,jj], elnet.intercept_path_[jj]
    warnings.resetwarnings()
    return bhat, ahat

def local_fpc(wmat, xval, yval, X, y, verbose=True):
    """
    :param wmat: Each column corresponds to a weight for y
    :param xval, yval: out-of-sample observation to make prediction
    :param X, y: Original (normalized) design matrix and response
    """
    nval = xval.shape[0]
    assert (wmat.shape[0] == X.shape[0]) & (wmat.shape[1] == nval)
    assert (nval == len(yval)) & (X.shape[0] == len(y))
    yhat_val = np.zeros(nval)
    for ii in range(nval):
        if (ii + 1) % 10 == 0:
            vprint('Iteration %i of %i' % (ii+1, nval), verbose)
        subset = np.where(wmat[:,ii]>1e-3)[0]  # don't bother keeping weights that are low
        bhat, ahat = fpc(X[subset], y[subset], w=wmat[subset,ii], verbose=False)
        yhat_val[ii] = xval[ii].dot(bhat) + ahat
    r2 = r2_score(yval, yhat_val)
    return r2


def r2_from_ll(ll, wmat, xval, yval, X, y, verbose=True):
    """
    WRAPS local_fpc for a specific length scale with pre-existing euclidian distance (wmat)
    """
    gauss = np.exp(-0.5 * wmat ** 2 / ll)
    return local_fpc(wmat=gauss, xval=xval, yval=yval, X=X, y=y, verbose=verbose)



# self = mdl(model=model, lead=lead, date=d_test, cn=cn_X)
# self.fit(X=Xmat_train.copy(), y=y_train.copy())
# self.tune(Xmat_valid.copy(), y=y_valid.copy())
class mdl():
    def __init__(self, model, lead, date, cn):
        self.enc = StandardScaler(copy=True)
        self.isfit, self.istuned = False, False
        self.model, self.lead, self.date, self.cn = model, lead, date, cn
        # Construct the unique filename
        self.fn = 'mdl_' + self.model + '_' + self.date.strftime('%Y_%m_%d') + '_lead_' + str(self.lead) + '.pkl'

    def fit(self, X, y):
        """
        local regression is lazy learner. Coefficients are learned at inference time
        """
        n, p = X.shape
        assert n == len(y)
        self.Xtil = self.enc.fit_transform(X)  # Learn the normalizer and keep data
        self.idx_train = cvec(np.arange(self.Xtil.shape[0]))
        self.idx_enc = StandardScaler().fit(self.idx_train)
        self.y = y.copy()
        # Fit baseline model so we can compare in validation set
        self.bhat, self.ahat = fpc(self.Xtil, self.y, w=None)
        self.isfit = True

    # X = Xmat_valid.copy(); y = y_valid.copy()
    def tune(self, X, y):
        """
        USE FPC_LOCAL TO TUNE THE MODEL
        Assumes that X is the validation X
        """
        nval = X.shape[0]
        Xtil = self.enc.transform(X)
        idx_val = cvec(np.arange(nval) + self.idx_train.max() + 1)
        Wmat = pairwise_distances(self.idx_enc.transform(idx_val),
                                  self.idx_enc.transform(self.idx_train)).T
        ll_seq = np.exp(np.linspace(np.log(0.001), np.log(20), num=30))
        wmat_seq = [np.mean(np.exp(-0.5 * Wmat ** 2 / ll),1) for ll in ll_seq]
        # Fit accross different length scales
        stime = time()
        lst = Parallel(n_jobs=10)(delayed(fpc)(X=self.Xtil, y=self.y, w=w, verbose=False) for w in wmat_seq)
        print('Took %0.1f seconds to run length scale tuning' % (time() - stime))
        # Fit ridge regression to each
        vec_r2 = np.zeros(len(ll_seq))
        for ii, ll in enumerate(ll_seq):
            supp = np.where(lst[ii][0] !=0)[0]
            l2 = Ridge(alpha=0.01).fit(self.Xtil[:,supp], self.y, wmat_seq[ii])
            vec_r2[ii] = r2_score(y,l2.predict(Xtil[:,supp]))
        res = pd.DataFrame({'ll':ll_seq, 'r2':vec_r2})
        print('Optimal length-scale\n%s' % res.loc[res.r2.idxmax()])
        self.ll = res.loc[res.r2.idxmax()].ll
        # Add on the validation X, y and index to the model
        self.Xtil, self.y = np.vstack([self.Xtil, Xtil]), np.append(self.y, y)
        self.idx_train = cvec(np.append(self.idx_train, idx_val))
        self.istuned = True

    # X = Xmat_test.copy()
    def predict(self, X):
        assert self.isfit & self.istuned
        Xtil = self.enc.transform(X)
        ntest = Xtil.shape[0]
        idx_test = cvec(np.arange(ntest) + self.idx_train.max() + 1)
        Wmat = pairwise_distances(self.idx_enc.transform(idx_test),
                                  self.idx_enc.transform(self.idx_train)).T
        Wgauss = np.exp(-0.5 * Wmat ** 2 / self.ll)
        assert Wgauss.shape[1] <= 24
        Wgauss = Wgauss.mean(1)  # So few observations, that we can average weights out for now
        # Fit the local model based on learned length scale
        bhat, ahat = fpc(X=self.Xtil,y=self.y,w=Wgauss)
        supp = np.where(bhat!=0)[0]
        # Post-lasso
        l2 = Ridge(alpha=0.01).fit(self.Xtil[:,supp], self.y, Wgauss)
        eta = l2.predict(Xtil[:,supp])
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