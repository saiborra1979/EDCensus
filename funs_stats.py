import sys
import numpy as np
import pandas as pd
from scipy import stats
from funs_support import stopifnot
from statsmodels.stats.proportion import proportion_confint as propCI

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
def prec_recall_lbls(df, cn_y, cn_y_rt, cn_pred, cn_idx, gg_nmax=50):
    # df=tmp.drop(columns=cn_drop);cn_y='y';cn_y_rt='y_rt';cn_pred='pred';cn_idx='date_rt';gg_nmax=50
    # del df, cn_y, cn_y_rt, cn_pred, cn_idx, gg_nmax, df_prec, df_sens, df_both, df_den
    df = df.copy()
    cn_check = [cn_y, cn_y_rt, cn_pred, cn_idx]
    assert df.columns.isin(cn_check).sum() == len(cn_check)
    cn_gg = list(np.setdiff1d(df.columns,cn_check))
    # Double check no highly imbalanced group
    assert np.all(df[cn_gg].apply(lambda x: np.unique(x).shape[0],0) < gg_nmax)
    df_sens = df.pivot_table(cn_idx, cn_gg + [cn_pred], cn_y, 'count').fillna(0).astype(int)
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