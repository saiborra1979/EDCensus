import re
import numpy as np
import pandas as pd
import itertools
from math import radians, cos, sin, asin, sqrt


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
