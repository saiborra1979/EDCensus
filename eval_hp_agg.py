import os
import pandas as pd
import numpy as np
import plotnine as pn
from plotnine.facets.facet import facet
from plotnine.facets.facet_wrap import facet_wrap
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

cn_int = ['dtrain','h_rtrain','nval']
di_int = dict(zip(cn_int, ['# Train','# Retrain','# Validation']))
di_metric = {'MAE':'MAE', 'spearman':'Spearman', 'sens':'Sensivitiy','prec':'Precision'}

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
    res1 = pd.DataFrame({'n':df.n.sum(), 'value':df.value.mean(),
            'se':np.sqrt( np.sum(df.se**2 / df.n)) },index=[0])
    res2 = df.head(1).drop(columns=['n','value','se','month']).reset_index(None,True)
    res = pd.concat([res2, res1],axis=1)
    return res

# data=res_rest.copy();gg=cn_multi;n_cpus=10
def parallel_perf(data, gg, fun, n_cpus=None):
    data_split = data.groupby(gg)
    if n_cpus is None:
        n_cpus = max(os.cpu_count()-1, 1)
    print('Number of CPUs: %i' % n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus)
    res = pd.concat(pool.map(fun, data_split))
    pool.close()
    pool.join()
    return res

cn_gg = ['model', 'metric', 'lead', 'dtrain', 'h_rtrain', 'nval'] 
cn_val = ['month', 'n', 'se', 'value']
assert not any_diff(df_qsub.columns, cn_gg + cn_val)
df_agg = parallel_perf(data=df_qsub, gg=cn_gg, fun=get_perf_month)
df_agg.reset_index(None, True, True)
# Convert the retraining/validation set into days

# Convert to daily from hourly
df_agg.h_rtrain = (df_agg.h_rtrain/24).astype(int)
df_agg.nval = (df_agg.nval/24).astype(int)


#############################
# --- (3) FIND SUPREMUM --- #

df_sup = df_agg.groupby(['metric','lead']).apply(lambda x: x.loc[x.value.idxmax()])
df_sup.reset_index(None,True,True)
df_sup = df_sup.melt(['model','lead','metric'],cn_int,'msr','star')
df_sup.metric = pd.Categorical(df_sup.metric,list(di_metric)).map(di_metric)
df_sup.msr = pd.Categorical(df_sup.msr,list(di_int)).map(di_int)

####################
# --- (4) PLOT --- #

# -- (i) Supremum by lead -- #
gg_sup_hp = (pn.ggplot(df_sup,pn.aes(x='lead',y='star',color='metric')) + 
    pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
    pn.scale_color_discrete(name='Metric') + 
    pn.theme(subplots_adjust={'wspace': 0.25}) + 
    pn.facet_wrap('~msr',nrow=1,scales='free_y') + 
    pn.labs(x='Forecasting lead',y='Winning parameter (Days)'))
gg_save(fn='gg_sup_hp.png',fold=dir_figures,gg=gg_sup_hp,width=16,height=4)


# -- (ii) Full information -- #
posd = pn.position_dodge(0.8)
for metric in df_agg.metric.unique():
    print('~~~ Metric: %s ~~~' % metric)
    tmp_df = df_agg.query('metric==@metric').reset_index(None, True)
    # Make into cateogircal
    tmp_df[cn_int] = tmp_df[cn_int].apply(lambda x: pd.Categorical(x,np.sort(x.unique())),0)
    shpz = ['$'+z+'$' for z in tmp_df.nval.cat.categories.astype(str)]    

    tmp_fn = 'dtrain_rtrain_nval_' + metric + '.png'    
    tmp_gg = (pn.ggplot(tmp_df, pn.aes(x='dtrain',y='value',color='h_rtrain',shape='nval')) + 
        pn.theme_bw() + pn.theme(subplots_adjust={'wspace': 0.25}) + 
        pn.geom_point(size=2,position=posd) + 
        pn.scale_color_discrete(name='Days to retrain') + 
        pn.scale_shape_manual(name='# of validation days',values=shpz) + 
        facet_wrap('~lead',labeller=pn.label_both,scales='free_y',nrow=4))
    gg_save(fn=tmp_fn,fold=dir_figures,gg=tmp_gg,width=22,height=12)



# #############################
# # --- (2) BASELINE PERF --- #

# from funs_stats import get_reg_score

# cn_reg = ['lead']
# cn_regn = cn_reg + ['n']
# cn_gg = ['lead', 'metric']
# cn_ggn = cn_gg + ['n']
# cn_ord = ['y_delta','pred_delta','date_rt','lead']

# # (1) Calculate spearman and MAE
# perf_reg = dat_bl.groupby(cn_reg).apply(get_reg_score,add_n=True).reset_index()
# perf_reg = perf_reg.melt(cn_regn,None,'metric')
# perf_reg['n'] = perf_reg.n.astype(int)

# # (2) Calculate the precision/recall
# perf_ord = prec_recall_lbls(x=df_res[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
# perf_ord = perf_ord.query('pred_delta == 1').reset_index(None, True)
# perf_ord = perf_ord.drop(columns='pred_delta').rename(columns={'den':'n'})

# # Merge regression + ordinal
# perf_agg = pd.concat([perf_reg, perf_ord]).reset_index(None, True)

