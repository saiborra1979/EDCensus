import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def rvec(x):
    return np.atleast_2d(x)

def cvec(x):
    return rvec(x).T

# leads=24; n_bins=5
class yX_process():
    def __init__(self, cn, cn_ohe=None, cn_bin=None, cn_cont=None, leads=24, n_bins=5):
        self.leads = np.arange(leads)+1
        self.cn_all = pd.Series(cn)
        self.di_cn = {'ohe':cn_ohe, 'bin':cn_bin, 'cont':cn_cont}
        self.enc_X = {'ohe':OneHotEncoder(dtype=int), 
                    'bin':KBinsDiscretizer(n_bins=n_bins,strategy='quantile'), 
                    'cont':MinMaxScaler()}
        self.di_none = {k:v is None for k,v in self.di_cn.items()}
        for k in self.di_cn:  # Check either lists or empty
            assert isinstance(self.di_cn[k],list) or self.di_cn[k] is None
            if not self.di_none[k]:
                self.di_cn[k] = pd.Series(self.di_cn[k])
        # Prepare column mapping
        self.di_X = pd.concat([None if v is None else pd.DataFrame({'tt':k,'cn':v},index=range(len(v))) for k,v in self.di_cn.items()]).reset_index(None,True)
        self.di_X = self.di_X.assign(cidx=self.di_X.apply(lambda x: np.where(self.cn_all == x.cn)[0][0], 1))        

    # self=enc_yX; X=Xtrain.copy(); y=ytrain.copy()
    def fit(self, X, y):
        assert isinstance(X,np.ndarray)
        self.p = 0
        self.cn_X = []
        for k in self.di_cn:
            if not self.di_none[k]:
                tmp_cn = self.di_X.query('tt==@k')
                self.enc_X[k].fit(X[:,tmp_cn.cidx.values])
                if k == 'ohe':
                    self.p += sum([len(z) for z in self.enc_X[k].categories_])
                    tmp1, tmp2 = tmp_cn.cn.to_list(), self.enc_X[k].categories_
                    tmp3 = [[cn+'_'+str(int(v)) for v in vals] for cn, vals in zip(tmp1, tmp2)]
                if k == 'bin':
                    self.p += sum([len(z)-1 for z in self.enc_X[k].bin_edges_])
                    tmp1, tmp2 = tmp_cn.cn.to_list(), self.enc_X[k].bin_edges_
                    tmp3 = [[cn+'_'+str(int(vals[i]))+'-'+str(int(vals[i+1])) for i in range(len(vals)-1)] for cn, vals in zip(tmp1, tmp2)]
                if k == 'cont':
                    self.p += len(tmp_cn)
                    tmp3 = [tmp_cn.cn.to_list()]
                self.cn_X = self.cn_X + list(np.concatenate(tmp3))
        assert len(self.cn_X) == self.p
        # Fit the Y to standard normal
        self.enc_Y = StandardScaler()
        self.enc_Y.fit(cvec(y))

    # self = enc_yX; X=Xtrain.copy();y=ytrain.copy(); rdrop=1
    def transform_X(self, X, rdrop=1):
        assert isinstance(X,np.ndarray)
        lst = []
        for k in self.di_cn:
            if not self.di_none[k]:
                lst.append(self.enc_X[k].transform(X[:,self.di_X.query('tt==@k').cidx.values]))
        sX = sparse.hstack(lst).tocsr()
        assert sX.shape[1] == self.p
        if rdrop > 0:
            sX = sX[:-rdrop]
        return sX

    def transform_y(self, y, rdrop=1):
        assert isinstance(y,np.ndarray)
        y_leads = pd.concat([pd.Series(y).shift(-k) for k in self.leads],1).values
        if rdrop > 0:
            y_leads = y_leads[:-rdrop]
        y_leads = np.hstack([self.enc_Y.transform(y_leads[:,[k-1]]) for k in self.leads])
        return y_leads

    def inverse_transform_y(self, y):
        assert isinstance(y,np.ndarray)
        if len(y.shape) == 1:
            y = cvec(y)
        yrev = self.enc_Y.inverse_transform(y)
        return yrev

