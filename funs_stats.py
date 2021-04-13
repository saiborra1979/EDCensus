import numpy as np
import pandas as pd
from scipy import stats

def get_CI(df,cn_mu,cn_se,alpha=0.05):
    critv = stats.norm.ppf(1-alpha/2)
    return df.assign(lb=lambda x: x[cn_mu]-critv*x[cn_se], ub=lambda x: x[cn_mu]+critv*x[cn_se])

def add_bin_CI(df, method='beta', alpha=0.05):
    assert df.columns.isin(['n','value']).sum() == 2
    holder = pd.concat(propCI(count=(df.n * df.value).astype(int), nobs=df.n, alpha=alpha, method=method), 1)
    holder = pd.concat([df, holder.rename(columns={0: 'lb', 1: 'ub'})],1)
    return holder
