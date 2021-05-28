import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint as propCI
import statsmodels.api as sm
from funs_support import stopifnot


def ttest_vec(mu1, mu2, se1, se2, n1, n2, var_eq=False):
    var1, var2 = se1**2, se2**2
    num = mu1 - mu2
    if var_eq:
        nu = n1 + n2 - 2
        sp2 = ((n1-1)*var1 + (n2-1)*var2) / nu
        den = np.sqrt(sp2*(1/n1 + 1/n2))
    else:
        nu = (var1/n1 + var2/n2)**2 / ( (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1) )
        den = np.sqrt(var1/n1 + var2/n2)
    dist_null = stats.t(df=nu)
    tstat = num / den
    pvals = 2*np.minimum(dist_null.sf(tstat), dist_null.cdf(tstat))
    return tstat, pvals


# Function to calculate escalation levels for different columns
def get_esc_levels(x, cn, esc_bins, esc_lbls):
    assert isinstance(x,pd.DataFrame)
    if not isinstance(cn, pd.Series):
        cn = pd.Series(cn)
    cn2 = 'esc_' + cn
    z = x[cn].apply(lambda w: pd.Categorical(pd.cut(w, esc_bins, False, esc_lbls)).codes,0)
    z.rename(columns=dict(zip(cn,cn2)),inplace=True)
    return pd.concat([x,z],axis=1)


def ols(y, x, alpha=0.05):
    assert isinstance(x,np.ndarray) and isinstance(y,np.ndarray)
    mod = sm.OLS(y, x).fit()
    p = 1
    if len(x.shape) == 2:
        p = x.shape[1]
    cn = 'x'+pd.Series(range(p)).astype(str)
    df1 = pd.DataFrame({'cn':cn,'bhat':mod.params})
    df2 = pd.DataFrame(mod.conf_int(alpha),columns=['lb','ub'])
    df = pd.concat([df1, df2],1)
    return df

def get_CI(df,cn_mu,cn_se,alpha=0.05):
    critv = stats.norm.ppf(1-alpha/2)
    return df.assign(lb=lambda x: x[cn_mu]-critv*x[cn_se], ub=lambda x: x[cn_mu]+critv*x[cn_se])

def add_bin_CI(df, cn_n, cn_val, method='beta', alpha=0.05):
    assert df.columns.isin([cn_n, cn_val]).sum() == 2
    holder = pd.concat(propCI(count=(df[cn_n] * df[cn_val]).astype(int), nobs=df[cn_n], alpha=alpha, method=method), 1)
    holder = pd.concat([df, holder.rename(columns={0: 'lb', 1: 'ub'})],1)
    return holder

# --------------------------------------------------------------------------- #
# --- Function to get ordinal labels for different lower-bounds of the CI --- #
# --------------------------------------------------------------------------- #
def ordinal_lbls(df, cn_date, cn_y, cn_y_rt, cn_pred, cn_se, level=0.5):
    # df=dat_pr.copy(); level=0.5; cn_date='dates'; cn_y='y'; cn_y_rt='y_rt'; cn_pred='pred'; cn_se='se'
    # del cn_check, crit, df, cn_val, ymx, ymi, esc_bins, esc_lvls,  cn_y, cn_y_rt, cn_pred, cn_se, cn_gg
    df = df.copy()
    cn_check = [cn_date,cn_y,cn_y_rt,cn_pred,cn_se]
    d_check = np.setdiff1d(cn_check, df.columns)
    stopifnot(len(d_check)==0, 'Missing columns: %s' % (', '.join(d_check)))
    assert (level > 0) & (level < 1) # Apply lower-bound of the level
    crit = stats.norm.ppf(level)
    df['hat_pred'] = df[cn_pred] - crit*df[cn_se]
    df.drop(columns=[cn_pred,cn_se],inplace=True)
    # Make the cuts
    cn_val = [cn_y_rt, cn_y, 'hat_pred']
    ymx = max(df[cn_val].max().max() + 1, 49)
    ymi = min(df[cn_val].min().min() - 1, -1)
    esc_bins = [ymi, 31, 38, 48, ymx]
    esc_lbls = ['≤30', '31-37', '38-47', '≥48']
    df[cn_val] = df[cn_val].apply(lambda x: pd.Categorical(pd.cut(x, esc_bins, False, esc_lbls)).codes)
    cn_gg = list(np.setdiff1d(df.columns,cn_check+['hat_pred']))
    df = df.melt(cn_gg+[cn_date,cn_y_rt],[cn_y,'hat_pred'],'tt')
    df['dd'] = df.value - df[cn_y_rt]
    df = df.pivot_table('dd', cn_gg + [cn_date,cn_y_rt], 'tt').reset_index()
    df.rename(columns={'hat_pred': 'pred', cn_y: 'y'}, inplace=True)
    return df


# ------------------------------------------------------------- #
# --- Function to get precision/recall for different levels --- #
# -------------------------------------------------------------- #
def prec_recall_lbls(x, cn_y, cn_pred, cn_idx, gg_nmax=53):
    # x = df[['y_delta','pred_delta','date_rt','lead']].copy()
    # cn_y='y_delta';cn_pred='pred_delta';cn_idx='date_rt';gg_nmax=100
    # del df, cn_y, cn_y_rt, cn_pred, cn_idx, gg_nmax, df_prec, df_sens, df_both, df_den
    x = x.copy()
    cn_check = [cn_y, cn_pred, cn_idx]
    assert x.columns.isin(cn_check).sum() == len(cn_check)
    cn_gg = list(np.setdiff1d(x.columns,cn_check))
    # Double check no highly imbalanced group
    assert np.all(x[cn_gg].apply(lambda x: np.unique(x).shape[0],0) <= gg_nmax)
    df_sens = x.pivot_table(cn_idx, cn_gg + [cn_pred], cn_y, 'count').fillna(0).astype(int)
    df_sens = df_sens.reset_index().melt(cn_gg + [cn_pred], None, None, 'n')
    df_sens = df_sens.assign(is_tp=lambda x: np.where(x[cn_pred] == x[cn_y], 'tp', 'fp'))
    # Calculate the precision
    df_prec = df_sens.groupby(cn_gg + [cn_pred, 'is_tp']).n.sum().reset_index()
    df_prec = df_prec.pivot_table('n', cn_gg + [cn_pred], 'is_tp', 'sum').fillna(0).astype(int).reset_index()
    df_prec = df_prec.assign(value=lambda x: x.tp / (x.tp + x.fp),
                             den=lambda x: x.tp + x.fp, metric='prec')
    df_prec.drop(columns=['tp', 'fp'], inplace=True)
    # Calculate sensitivity
    df_den = df_sens.groupby(cn_gg + [cn_y]).n.sum().reset_index().rename(columns={'n': 'den'})
    df_sens = df_sens.query('is_tp=="tp"').merge(df_den).drop(columns=['is_tp',cn_y])
    df_sens = df_sens.assign(value=lambda x: x.n / x.den, metric='sens').drop(columns='n')
    # Stack
    df_both = pd.concat([df_prec, df_sens]).reset_index(None, True)
    return df_both