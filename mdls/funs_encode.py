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

# cn=cn_use; lead=lead; lag=lag
# cn_ohe=di_cn['ohe']; cn_cont=di_cn['cont']; cn_bin=di_cn['bin']
# self = yX_process(cn=cn_use, lead=lead, lag=lag, cn_ohe=di_cn['ohe'], cn_cont=di_cn['cont'], cn_bin=di_cn['bin'])
# del cn, cn_ohe, cn_cont, cn_bin, self
class yX_process():
    def __init__(self, cn, cn_ohe=None, cn_bin=None, cn_cont=None, n_bins=5, lead=24, lag=24):
        self.lead, self.lag = lead, lag
        self.cn_all = pd.Series(cn)
        self.di_cn = {'ohe':cn_ohe, 'bin':cn_bin, 'cont':cn_cont}
        self.enc_X = {'ohe':OneHotEncoder(dtype=int, drop='if_binary'),
                    'bin':KBinsDiscretizer(n_bins=n_bins,strategy='quantile'), 
                    'cont':MinMaxScaler()}
        self.di_none = {k: (v is None) or (v == [])  for k,v in self.di_cn.items()}
        for k in self.di_cn:  # Check either lists or empty
            assert isinstance(self.di_cn[k],list) or self.di_cn[k] is None
            if not self.di_none[k]:
                self.di_cn[k] = pd.Series(self.di_cn[k])
        # Prepare column mapping
        self.di_X = pd.concat([None if v is None else pd.DataFrame({'tt':k,'cn':v},index=range(len(v))) for k,v in self.di_cn.items()]).reset_index(None,True)
        self.di_X = self.di_X.assign(cidx=self.di_X.apply(lambda x: np.where(self.cn_all == x.cn)[0][0], 1))        

    # self=enc_yX; X=Xtrain.copy(); y=None
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.p = 0
        self.cn_X = []
        for k in self.di_cn:
            if not self.di_none[k]:
                tmp_cn = self.di_X.query('tt==@k')
                self.enc_X[k].fit(X[tmp_cn.cn])
                if k == 'ohe':
                    tmp_cn = tmp_cn.cn.to_list()
                    tmp_cat = self.enc_X[k].categories_
                    if self.enc_X[k].drop == 'if_binary':
                        tmp_cat = [z[[1]] if len(z)==2 else z for z in tmp_cat]
                    self.p += sum([len(z) for z in tmp_cat])
                    tmp3 = [[cn+'_'+str(int(v)) for v in vals] for cn, vals in zip(tmp_cn, tmp_cat)]
                if k == 'bin':
                    self.p += sum([len(z)-1 for z in self.enc_X[k].bin_edges_])
                    tmp_cn = tmp_cn.cn.to_list()
                    tmp_edge = self.enc_X[k].bin_edges_
                    tmp3 = [[cn+'_'+str(int(vals[i]))+'-'+str(int(vals[i+1])) for i in range(len(vals)-1)] for cn, vals in zip(tmp_cn, tmp_edge)]
                if k == 'cont':
                    q1 = pd.Series(np.tile(tmp_cn.cn,self.lag+1))
                    q2 = pd.Series(np.repeat(np.arange(self.lag+1),len(tmp_cn.cn)))
                    tmp3 = [list(q1 + '_lag_' + q2.astype(str))]
                    self.p += len(tmp3[0])
                self.cn_X = self.cn_X + list(np.concatenate(tmp3))
        assert len(self.cn_X) == self.p
        # Fit the Y to standard normal
        if y is not None:
            self.enc_Y = StandardScaler()
            self.enc_Y.fit(cvec(y))

    # self = enc_yX; X=Xtrain.copy(); rdrop=1
    def transform_X(self, X, rdrop=1):
        assert isinstance(X, pd.DataFrame)
        lst = []
        for k in self.di_cn:
            if not self.di_none[k]:
                X_k = self.enc_X[k].transform(X[self.di_X.query('tt==@k').cn])
                if (k == 'cont') and (self.lag > 0):
                    X_k = pd.DataFrame(X_k)
                    X_k = pd.concat([X_k.shift(l) for l in range(self.lag+1)],1).values
                lst.append(X_k)
                del X_k
        if all([isinstance(x,np.ndarray) for x in lst]):
            X = np.concatenate(lst,axis=1)
        else:
            X = sparse.hstack(lst).toarray()
        assert X.shape[1] == len(self.cn_X)
        X = X[self.lag:]  # Remove missing values from lag
        if rdrop > 0:
            X = X[:-rdrop]
        return X

    def transform_y(self, y, rdrop=1):
        assert isinstance(y,np.ndarray)
        seq_leads = range(1,self.lead+1)
        y_leads = pd.concat([pd.Series(y).shift(-k) for k in seq_leads],1).values
        y_leads = y_leads[self.lag:]
        if rdrop > 0:
            y_leads = y_leads[:-rdrop]
        if hasattr(self, 'enc_Y'):
            y_leads = np.hstack([self.enc_Y.transform(y_leads[:,[k-1]]) for k in seq_leads])
        return y_leads

    def inverse_transform_y(self, y):
        assert isinstance(y,np.ndarray)
        if len(y.shape) == 1:
            y = cvec(y)
        yrev = np.hstack([self.enc_Y.inverse_transform(y[:,[j]]) for j in range(y.shape[1])])
        return yrev

