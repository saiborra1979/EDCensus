import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

def rvec(x):
    return np.atleast_2d(x)

def cvec(x):
    return rvec(x).T

class yX_process():
    """
    FUNCTION TO CONVERT CATEGORICAL + NUMERIC INTO BINARY FEATURES
    """
    def __init__(self, cn, cn_ohe, cn_num, leads=24):
        self.leads = np.arange(leads)+1
        self.cn = pd.Series(cn)
        self.cn_ohe = pd.Series(cn_ohe,dtype='object')
        self.cn_num = pd.Series(cn_num,dtype='object')
        if cn_ohe is not None:
            assert self.cn_ohe.isin(self.cn).all()
        if cn_num is not None:
            assert self.cn_num.isin(self.cn).all()
        self.di_X = self.cn_ohe.append(self.cn_num)
        # Find corresponding column index
        self.di_X = pd.DataFrame({'cn':self.di_X,'cidx':self.di_X.apply(lambda x: np.where(self.cn == x)[0][0])})
        self.di_X['tt'] = np.append(np.repeat('ohe',len(self.cn_ohe)),np.repeat('num',len(self.cn_num)))
        
    def fit(self, X):
        assert isinstance(X,np.ndarray)
        self.p = 0
        self.cn_X = []
        if len(self.cn_ohe)>0:
            #print('One-hot encoding')
            tmp_cn = self.di_X.query('tt=="ohe"')
            self.ohe = OneHotEncoder(dtype=int)
            self.ohe.fit(X[:,tmp_cn.cidx.values])
            self.p += sum([len(z) for z in self.ohe.categories_])
            tmp1, tmp2 = tmp_cn.cn.to_list(), self.ohe.categories_
            tmp3 = [[cn+'_'+str(int(v)) for v in vals] for cn, vals in zip(tmp1, tmp2)]
            self.cn_X = self.cn_X + list(np.concatenate(tmp3))

        if len(self.cn_num) > 0:
            #print('Discretizing')
            tmp_cn = self.di_X.query('tt=="num"')
            self.num = KBinsDiscretizer(n_bins=5,strategy='quantile')
            self.num.fit(X[:,tmp_cn.cidx.values])
            tmp1, tmp2 = tmp_cn.cn.to_list(), self.num.bin_edges_
            tmp3 = [[cn+'_'+str(int(vals[i]))+'-'+str(int(vals[i+1])) for i in range(len(vals)-1)] for cn, vals in zip(tmp1, tmp2)]
            self.cn_X = self.cn_X + list(np.concatenate(tmp3))
            self.p += sum([len(z)-1 for z in self.num.bin_edges_])
        assert len(self.cn_X) == self.p

    # self = processor; X=Xtrain.copy();y=ytrain.copy();rdrop=1
    def transform(self, X, y=None, rdrop=1):
        assert isinstance(X,np.ndarray)
        lst = []
        if len(self.cn_ohe)>0:
            lst.append(self.ohe.transform(X[:,self.di_X.query('tt=="ohe"').cidx.values]))
        if len(self.cn_num) > 0:
            lst.append(self.num.transform(X[:,self.di_X.query('tt=="num"').cidx.values]))
        sX = sparse.hstack(lst).tocsr()
        if rdrop > 0:
            sX = sX[:-rdrop]
        assert sX.shape[1] == self.p
        if y is not None:
            assert isinstance(y,np.ndarray)
            assert len(X) == len(y)
            y_leads = pd.concat([pd.Series(y).shift(-k) for k in self.leads],1)
            y_delta = y_leads.values[:-rdrop]
            return sX, y_delta
        else:
            return sX 
