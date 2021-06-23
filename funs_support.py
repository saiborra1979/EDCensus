import re
import os
import sys
import socket
import numpy as np
import pandas as pd
import itertools
from math import radians, cos, sin, asin, sqrt
from statsmodels.stats.proportion import proportion_confint as propCI
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error as MAE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse

import multiprocessing
from itertools import repeat

from scipy.stats import norm

def any_diff(x, y):
    uu = np.union1d(x, y)
    ii = np.intersect1d(x, y)
    check = len(np.setdiff1d(uu, ii)) > 0
    return check


def find_nonzero_var(df):
    u_count = df.apply(lambda x: x.unique().shape[0],0)
    cn = list(u_count[u_count > 1].index)
    return cn

def find_zero_var(df):
    u_count = df.apply(lambda x: x.unique().shape[0],0)
    cn = list(u_count[u_count == 1].index)
    return cn

def drop_zero_var(df):
    df = df.copy()
    cn = find_nonzero_var(df)
    return df[cn]


def get_reg_score(x):
    tmp = pd.Series({'spearman': spearmanr(x.y,x.pred)[0],'MAE':MAE(x.y,x.pred)})
    return tmp

def get_iqr(x,alpha=0.25, add_n=False, ret_df=True):
    tmp = x.quantile([1-alpha,0.5,alpha])
    tmp.index = ['lb','med','ub']
    if add_n:
        tmp = tmp.append(pd.Series({'n':len(x)}))
    if ret_df:
        tmp = pd.DataFrame(tmp).T
    return tmp


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


def get_level(groups, target, gg):
    cn = groups[0]
    df = groups[1]
    out = pd.DataFrame(cn).T.assign(level=find_prec(df, target, gg))
    return out

# data=res_rest.copy();gg=cn_multi;n_cpus=10
def parallel_find_prec(data, gg, target, n_cpus=None):
    data_split = data.groupby(gg)
    if n_cpus is None:
        n_cpus = os.cpu_count()-1
    print('Number of CPUs: %i' % n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus)
    data = pd.concat(pool.starmap(get_level, zip(data_split, repeat(target), repeat(gg))))
    pool.close()
    pool.join()
    return data


# Function that will apply ordinal_lbls() until it finds the precision target
# df = res_train.copy(); gg=['month']
def ret_prec(level, df, gg, ret_df=False):
    cn_sign = ['pred', 'y']
    dat2ord = ordinal_lbls(df.copy(), level=level)
    dat2ord[cn_sign] = np.sign(dat2ord[cn_sign])
    prec = sens_spec_df(df=dat2ord, gg=gg)
    if ret_df:
        return prec
    else:
        prec = prec.query('pred==1 & metric=="prec"').value.values[0]
    return prec

# df=res_train.copy(); target=0.8; gg=['month']
def find_prec(df, target, gg, tol=0.005, max_iter=50):
    level_lb, level_mid, level_ub = 0.01, 0.5, 0.99
    prec_lb, prec_mid, prec_ub = ret_prec(level_lb, df=df, gg=gg), ret_prec(level_mid, df=df, gg=gg), ret_prec(level_ub, df=df, gg=gg)
    for tick in range(max_iter):
        if target < prec_mid:
            #print('Mid becomes top')
            level_lb, level_mid, level_ub = level_lb, level_lb+(level_mid-level_lb)/2, level_mid
        else:
            #print('Mid becomes bottom')
            level_lb, level_mid, level_ub = level_mid, level_mid+(level_ub-level_mid)/2, level_ub
        prec_lb = ret_prec(level_lb, df=df, gg=gg)
        prec_mid = ret_prec(level_mid, df=df, gg=gg)
        prec_ub = ret_prec(level_ub, df=df, gg=gg)
        err_lb, err_mid, err_ub = np.abs(prec_lb-target), np.abs(prec_mid-target), np.abs(prec_ub-target)
        err = min(err_lb, err_mid, err_ub)
        # print('lb: %0.2f (%0.3f), mid: %0.2f (%0.3f), ub: %0.2f (%0.3f)' %
        #       (level_lb, prec_lb, level_mid, prec_mid, level_ub, prec_ub))
        if err < tol:
            #print('Tolerance met')
            break
    di_level = {'lb':level_lb, 'mid':level_mid, 'ub':level_ub}
    tt = pd.DataFrame({'tt':['lb','mid','ub'],'err':[err_lb, err_mid, err_ub]}).sort_values('err').tt.values[0]
    level_star = di_level[tt]
    return level_star



def gg_color_hue(n):
    from colorspace.colorlib import HCL
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(HCL(H=h, L=65, C=100).colors()[0])
    return hcl

def get_perf(groups):
    cn = groups[0]
    df = groups[1]
    res = pd.DataFrame(cn).T.assign(rmse=mse(df.y, df.pred, squared=False), r2=r2(df.y, df.pred), conc=cindex(df.y, df.pred))
    return res

# data=res_rest.copy();gg=cn_multi;n_cpus=10
def parallel_perf(data, gg, n_cpus=None):
    data_split = data.groupby(gg)
    if n_cpus is None:
        n_cpus = os.cpu_count()-1
    print('Number of CPUs: %i' % n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus)
    data = pd.concat(pool.map(get_perf, data_split))
    pool.close()
    pool.join()
    return data

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

