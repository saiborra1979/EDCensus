import re
import os
import sys
import socket
import numpy as np
import pandas as pd
import itertools
import pickle
from math import radians, cos, sin, asin, sqrt
from colorspace.colorlib import HCL

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def find_dir_olu():
    dir_base = os.getcwd()
    cpu = socket.gethostname()
    print(cpu)
    hpf1 = pd.Series(cpu).str.contains('qlogin')[0]
    hpf2 = pd.Series(cpu).str.contains('node')[0]
    # Set directory based on CPU name
    if cpu == 'RT5362WL-GGB':
        if os.name == 'nt':  # In windows
            dir_olu = 'D:\\projects\\ED'
        elif os.name == 'posix':  # on WSL
            dir_olu = '/mnt/d/projects/ED'
        else:
            sys.exit('Huh?! I dont recognize this operating system')
        print('On predator machine')
    elif cpu == 'snowqueen':
        print('On snowqueen machine')
        dir_olu = '/data/ED/'  #os.path.join(dir_base, '..')
    elif hpf1 or hpf2:
        print('On HPF')
        dir_olu = os.path.join(dir_base, '..')
    else:
        sys.exit('Where are we?!')
    return dir_olu

# R-like functions
def str_subset(x,pat):
    return x[x.str.contains(pat,regex=True)]

# Remove column nnames with Unnamed
def drop_unnamed(x):
    assert isinstance(x, pd.DataFrame)
    cn = x.columns
    cn_drop = list(cn[cn.str.contains('Unnamed')])
    if len(cn_drop) > 0:
        x = x.drop(columns=cn_drop)
    return x


# ---- FUNCTIONS TO READ/WRITE PICKLES ---- #
def read_pickle(path):
    assert os.path.exists(path)
    with open(path, 'rb') as handle:
        di = pickle.load(handle)
    return di

def write_pickle(di, path):
    with open(path, 'wb') as handle:
        pickle.dump(di, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Check whether two arrays are the same
def any_diff(x, y):
    uu = np.union1d(x, y)
    ii = np.intersect1d(x, y)
    check = len(np.setdiff1d(uu, ii)) > 0
    return check

# Find the columns that have some variation
def find_nonzero_var(df):
    u_count = df.apply(lambda x: x.unique().shape[0],0)
    cn = list(u_count[u_count > 1].index)
    return cn

# Find columns with no variation
def find_zero_var(df):
    u_count = df.apply(lambda x: x.unique().shape[0],0)
    cn = list(u_count[u_count == 1].index)
    return cn

def drop_zero_var(df):
    df = df.copy()
    cn = find_nonzero_var(df)
    return df[cn]


def get_date_range(x):
    assert isinstance(x,pd.Series)
    dfmt = '%b %d , %H%P'
    if len(x) > 0:
        xmi, xmx = x.min(), x.max()
        x = '(' + xmi.strftime(dfmt) + ' - ' + xmx.strftime(dfmt) + ')'
        return x
    else:
        return None

def stopifnot(stmt,msg):
    if not stmt:
        print(msg)
    assert stmt

def gg_save(fn,fold,gg,width,height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height)


def gg_color_hue(n):
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(HCL(H=h, L=65, C=100).colors()[0])
    return hcl

def makeifnot(path):
    if not os.path.exists(path):
        print('making folder')
        try:
            os.mkdir(path)
        except:
            print('looks like folder already exists!')

def date2ymd(x):
    assert isinstance(x, pd.Series)
    dat = pd.DataFrame({'year':x.dt.year, 'month':x.dt.month,
                        'day':x.dt.strftime('%d').astype(int)})
    return dat

def date2ymdh(x):
    assert isinstance(x, pd.Series)
    year, month, day, hour = x.dt.year, x.dt.month, x.dt.day, x.dt.hour
    return pd.DataFrame({'year':year, 'month':month, 'day':day, 'hour':hour})

def date2ymw(x,week53=True):
    assert isinstance(x, pd.Series)
    dat = x.dt.isocalendar().drop(columns='day').rename(columns={'week':'woy'})
    dat.insert(1,'month',x.dt.month)
    if week53:  # Will ensure week 53 is not split over two years
        dat = dat.assign(year = lambda x: np.where((x.woy==53)&(x.month==1), x.year-1, x.year))
    return dat

def ymd2date(x):
    assert isinstance(x, pd.DataFrame)
    cn = ['year','month','day']
    assert x.columns.isin(cn).sum() == len(cn)
    dd  = pd.to_datetime(x.year.astype(str) + '-' + x.month.astype(str) + '-' + x.day.astype(str))
    return dd

def ymdh2date(x):
    assert isinstance(x, pd.DataFrame)
    cn = ['year','month','day','hour']
    assert x.columns.isin(cn).sum() == len(cn)
    dd  = pd.to_datetime(x.year.astype(str) + '-' + x.month.astype(str) + '-' + x.day.astype(str)+' '+x.hour.astype(str)+':00:00')
    return dd

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

# df=r2_mdl.copy();gg=cn_melt;lam=0.1
# del df, gg, lam, holder
def smoother_df(df, gg, lam=0.1):
    """
    WILL APPLY SMOOTHER FUNCTION ON df WITH GROUPS gg
    ASSUMES 'value' IN COLUMN
    """
    assert isinstance(df, pd.DataFrame)
    assert 'value' in df
    df['tmp'] = df.groupby(gg).cumcount()
    holder = df.groupby(gg).apply(
        lambda x: pd.Series({'smooth': smoother(x=x.value.values, lam=lam), 'idx': x.tmp.values}))
    holder = holder.explode('smooth').drop(columns='idx').reset_index().assign(tmp=holder.explode('idx').idx.values)
    holder['tmp'] = holder.tmp.astype(int)
    holder['smooth'] = holder.smooth.astype(float)
    df = df.merge(holder, 'left', on=gg + ['tmp']).drop(columns=['tmp'])
    return df


# del df, mat_pred, mat_y, idx_y, idx_act, idx_equal, nact, nequal
def add_lags(df, l):
    tmp = df.shift(l)
    tmp.columns = pd.MultiIndex.from_product([tmp.columns, ['lag_' + str(l)]])
    return tmp

# Join list of lists into one list
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

# Vectorized haversine
def vhaversine(lat1, lon1, lat2, lon2):
    fun = np.vectorize(haversine)
    vals = fun(lat1, lon1, lat2, lon2)
    return (vals)

# Function to extract postal codes
def pc_extract(ss, pat):
    hit = re.search(pat, ss)
    if hit is None:
        return 0
    else:
        return hit.end()

