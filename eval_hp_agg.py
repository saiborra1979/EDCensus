import os
import pandas as pd
import numpy as np
from plotnine import *
from funs_support import find_dir_olu, find_zero_var, str_subset, gg_save
from funs_stats import get_esc_levels, prec_recall_lbls, fast_F
from funs_esc import esc_lbls, esc_bins

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
lst_dir = [dir_figures, dir_output, dir_flow, dir_test]
assert all([os.path.exists(z) for z in lst_dir])

#########################
# --- (1) LOAD DATA --- #

path_qsub = os.path.join(dir_test, 'res_merge_qsub.csv')
df_qsub = pd.read_csv(path_qsub, header=[0, 1])
df_qsub.rename(columns={'Unnamed: 0_level_1':'model','model':'base'},inplace=True)
df_qsub.columns = df_qsub.columns.droplevel(0)

# (ii) Load the benchmark result
path_bl = os.path.join(dir_test,'bl_hour.csv')
dat_bl = pd.read_csv(path_bl)

##############################
# --- (2) MERGE BY MONTH --- #

from funs_support import any_diff
import multiprocessing

def get_perf_month(groups):
    cn = groups[0]
    df = groups[1]
    print(cn)
    res = pd.DataFrame({'n':df.n.sum(), 'value':df.value.mean(),
            'se':np.sqrt( np.sum(df.se**2 / df.n)) },index=[0])
    return res

# data=res_rest.copy();gg=cn_multi;n_cpus=10
def parallel_perf(data, gg, fun, n_cpus=None):
    data_split = data.groupby(gg)
    if n_cpus is None:
        n_cpus = max(os.cpu_count()-2, 1)
    print('Number of CPUs: %i' % n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus)
    res = pd.concat(pool.map(fun, data_split))
    pool.close()
    pool.join()
    return res

cn_gg = ['model', 'metric', 'lead', 'dtrain', 'h_rtrain', 'nval'] 
cn_val = ['month', 'n', 'se', 'value']

assert not any_diff(df_qsub.columns, cn_gg + cn_val)

parallel_perf(data=df_qsub.head(1000), gg=cn_gg, fun=get_perf_month)



df_agg = df_qsub.groupby(cn_gg).apply(lambda x: pd.DataFrame({'n':x.n.sum(),'value':x.value.mean(), 'se':np.sqrt( np.sum(x.se**2 / x.n) )},index=[0]))
df_agg

#############################
# --- (2) BASELINE PERF --- #

from funs_stats import get_reg_score

cn_reg = ['lead']
cn_regn = cn_reg + ['n']
cn_gg = ['lead', 'metric']
cn_ggn = cn_gg + ['n']
cn_ord = ['y_delta','pred_delta','date_rt','lead']

# (1) Calculate spearman and MAE
perf_reg = dat_bl.groupby(cn_reg).apply(get_reg_score,add_n=True).reset_index()
perf_reg = perf_reg.melt(cn_regn,None,'metric')
perf_reg['n'] = perf_reg.n.astype(int)

# (2) Calculate the precision/recall
perf_ord = prec_recall_lbls(x=df_res[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
perf_ord = perf_ord.query('pred_delta == 1').reset_index(None, True)
perf_ord = perf_ord.drop(columns='pred_delta').rename(columns={'den':'n'})

# Merge regression + ordinal
perf_agg = pd.concat([perf_reg, perf_ord]).reset_index(None, True)


dat_bl

dat_bl[cn_dates] = dat_bl[cn_dates].apply(pd.to_datetime,0)


df_qsub



