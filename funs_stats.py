import os
import numpy as np
import pandas as pd
import multiprocessing
import statsmodels.api as sm
from funs_support import stopifnot, cvec
from itertools import repeat
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint as propCI
from sklearn.metrics import mean_absolute_error as MAE


##########################################
# ---- (1) BASIC SUMMARY STATISTICS ---- # 

# Calculate spearman's rho and MAE for a data.frame with columns "y"/"pred"
def get_reg_score(x, add_n=False):
    tmp = pd.Series({'spearman': spearmanr(x.y,x.pred)[0],'MAE':-MAE(x.y,x.pred)})
    if add_n:
        tmp = tmp.append(pd.Series({'n':len(x)}))
    return tmp

# Calculates interquartile range for any array
def get_iqr(x,alpha=0.25, add_n=False, ret_df=True):
    tmp = x.quantile([1-alpha,0.5,alpha])
    tmp.index = ['lb','med','ub']
    if add_n:
        tmp = tmp.append(pd.Series({'n':len(x)}))
    if ret_df:
        tmp = pd.DataFrame(tmp).T
    return tmp

# Vectorize t-tests for arrays or means, standard errors, and sample sizes
"""
mu{12}: means of groups{12}
se{12}: standard deviation for groups {12}
n{12}: sample size for groups{12}
"""
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


# Calculate the concordance for a continuous variable
def cindex(y, pred):
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

# Function for extracting 
def get_level(groups, target, gg):
    cn = groups[0]
    df = groups[1]
    out = pd.DataFrame(cn).T.assign(level=find_prec(df, target, gg))
    return out


######################################
# ---- (2) REGRESSION FUNCTIONS ---- #

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


#####################################
# ---- (3) PRECISION FUNCTIONS ---- #

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

# # Function that will apply ordinal_lbls() until it finds the precision target
# # df = res_train.copy(); gg=['month']
# def ret_prec(level, df, gg, ret_df=False):
#     cn_sign = ['pred', 'y']
#     dat2ord = ordinal_lbls(df.copy(), level=level)
#     dat2ord[cn_sign] = np.sign(dat2ord[cn_sign])
#     prec = sens_spec_df(df=dat2ord, gg=gg)
#     if ret_df:
#         return prec
#     else:
#         prec = prec.query('pred==1 & metric=="prec"').value.values[0]
#     return prec

# # df=res_train.copy(); target=0.8; gg=['month']
# def find_prec(df, target, gg, tol=0.005, max_iter=50):
#     level_lb, level_mid, level_ub = 0.01, 0.5, 0.99
#     prec_lb, prec_mid, prec_ub = ret_prec(level_lb, df=df, gg=gg), ret_prec(level_mid, df=df, gg=gg), ret_prec(level_ub, df=df, gg=gg)
#     for tick in range(max_iter):
#         if target < prec_mid:
#             #print('Mid becomes top')
#             level_lb, level_mid, level_ub = level_lb, level_lb+(level_mid-level_lb)/2, level_mid
#         else:
#             #print('Mid becomes bottom')
#             level_lb, level_mid, level_ub = level_mid, level_mid+(level_ub-level_mid)/2, level_ub
#         prec_lb = ret_prec(level_lb, df=df, gg=gg)
#         prec_mid = ret_prec(level_mid, df=df, gg=gg)
#         prec_ub = ret_prec(level_ub, df=df, gg=gg)
#         err_lb, err_mid, err_ub = np.abs(prec_lb-target), np.abs(prec_mid-target), np.abs(prec_ub-target)
#         err = min(err_lb, err_mid, err_ub)
#         # print('lb: %0.2f (%0.3f), mid: %0.2f (%0.3f), ub: %0.2f (%0.3f)' %
#         #       (level_lb, prec_lb, level_mid, prec_mid, level_ub, prec_ub))
#         if err < tol:
#             #print('Tolerance met')
#             break
#     di_level = {'lb':level_lb, 'mid':level_mid, 'ub':level_ub}
#     tt = pd.DataFrame({'tt':['lb','mid','ub'],'err':[err_lb, err_mid, err_ub]}).sort_values('err').tt.values[0]
#     level_star = di_level[tt]
#     return level_star


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