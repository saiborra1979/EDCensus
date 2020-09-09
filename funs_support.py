import re
import os
import numpy as np
import pandas as pd
import itertools
from math import radians, cos, sin, asin, sqrt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def makeifnot(path):
    if not os.path.exists(path):
        print('making folder')
        try:
            os.mkdir(path)
        except:
            print('looks like folder already exists!')

def date2ymd(x):
    assert isinstance(x, pd.Series)
    dat = pd.DataFrame({'year':x.dt.strftime('%Y').astype(int),
                          'month':x.dt.strftime('%m').astype(int),
                          'day':x.dt.strftime('%d').astype(int)})
    return dat

def date2ymdh(x):
    assert isinstance(x, pd.Series)
    year, month, day, hour = x.dt.year, x.dt.month, x.dt.day, x.dt.hour
    return pd.DataFrame({'year':year, 'month':month, 'day':day, 'hour':hour})

def rho(x,y):
    return pd.DataFrame({'x':x,'y':y}).corr().iloc[0,1]

def r2_fun(y, x):
    idx = ~(np.isnan(y) | np.isnan(x))
    y, x = y[idx], x[idx]
    mdl = LinearRegression().fit(X=cvec(x),y=y)
    res = y - mdl.predict(cvec(x)).flatten()
    return 1 - res.var() / y.var()


class normalizer():
    def __init__(self):
        self.enc = StandardScaler(copy=True)
    def fit(self, x):
        ls = len(x.shape)
        assert ls <= 2
        if ls == 2:
            self.enc.fit(x)
        else:
            self.enc.fit(cvec(x))
    def transform(self, x):
        ls = len(x.shape)
        assert ls <= 2
        if ls == 2:
            return self.enc.transform(x)
        else:
            return self.enc.transform(cvec(x)).flatten()

    def inverse_transform(self, x):
        ls = len(x.shape)
        assert ls <= 2
        if ls == 2:
            return self.enc.inverse_transform(x)
        else:
            return self.enc.inverse_transform(cvec(x)).flatten()

def t2n(x):
    return x.cpu().detach().numpy()

def cvec(x):
    return np.atleast_2d(x).T

def smoother(x,lam=0.5):
    """
    FUNCTION TO SMOOTH TIME-SERIES x
    # del n, idx, mat, dist, weighted
    """
    n = len(x)
    idx = np.arange(n) + 1
    idx = (idx - idx.mean()) / idx.std()
    mat = np.tile(idx, [n, 1])
    dist = np.exp(-((cvec(idx) - mat) ** 2) / lam)
    weighted = np.sum(cvec(x) * dist,0) / dist.sum(0)
    return weighted


def cindex(y, pred):
    """
    C-index for continuous variables
    """
    df = pd.DataFrame({'y': y, 'pred': pred}).sort_values('y', ascending=False).reset_index(None, True)
    mat_pred = np.tile(df.pred, [df.shape[0], 1])
    mat_y = np.tile(df.y, [df.shape[0], 1])
    idx_y = cvec(df.y) > mat_y
    idx_act = cvec(df.pred) > mat_pred
    idx_equal = cvec(df.pred) == mat_pred
    nact = idx_act[idx_y].sum()
    nequal = idx_equal[idx_y].sum()
    val = (nact + nequal * 0.5) / idx_y.sum()
    return val


# del df, mat_pred, mat_y, idx_y, idx_act, idx_equal, nact, nequal


def add_lags(df, l):
    tmp = df.shift(l)
    tmp.columns = pd.MultiIndex.from_product([tmp.columns, ['lag_' + str(l)]])
    return tmp


def ljoin(x):
    return list(itertools.chain.from_iterable(x))


# Function to calculate haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def vhaversine(lat1, lon1, lat2, lon2):
    fun = np.vectorize(haversine)
    vals = fun(lat1, lon1, lat2, lon2)
    return (vals)


def pc_extract(ss, pat):
    hit = re.search(pat, ss)
    if hit is None:
        return 0
    else:
        return hit.end()
