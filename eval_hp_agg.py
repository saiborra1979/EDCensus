# Load modules
import os
import pandas as pd
import numpy as np
import plotnine as pn
from plotnine.geoms.geom_boxplot import geom_boxplot
from plotnine.geoms.geom_histogram import geom_histogram
from plotnine.labels import ggtitle
from plotnine.themes.theme_bw import theme_bw
from funs_parallel import parallel_perf, get_perf_month
from funs_support import find_dir_olu, gg_save, any_diff, drop_unnamed

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
# df_qsub.rename(columns={'Unnamed: 0_level_1':'model','model':'base'},inplace=True)
df_qsub.columns = df_qsub.columns.droplevel(0)

# # (ii) Load the benchmark result
# path_bl = os.path.join(dir_test,'bl_hour.csv')
# dat_bl = pd.read_csv(path_bl)

##############################
# --- (2) MERGE BY MONTH --- #

cn_gg = ['model', 'metric', 'lead', 'dtrain', 'h_rtrain', 'nval'] 
cn_val = ['month', 'n', 'se', 'value']
assert not any_diff(df_qsub.columns, cn_gg + cn_val)
df_agg = parallel_perf(data=df_qsub, gg=cn_gg, fun=get_perf_month)
df_agg.reset_index(None, True, True)
# Convert the retraining/validation set into days

# Convert to daily from hourly
df_agg['h_rtrain'] = (df_agg.h_rtrain/24).astype(int)
df_agg['nval'] = (df_agg.nval/24).astype(int)


#############################
# --- (3) FIND SUPREMUM --- #

df_sup = df_agg.groupby(['metric','lead']).apply(lambda x: x.loc[x.value.idxmax()])
df_sup.reset_index(None,True,True)
df_sup = df_sup.melt(['model','lead','metric'],cn_int,'msr','star')
df_sup.metric = pd.Categorical(df_sup.metric,list(di_metric)).map(di_metric)
df_sup.msr = pd.Categorical(df_sup.msr,list(di_int)).map(di_int)

df_sup_n = df_sup.groupby(['model','metric','msr','star']).size().reset_index()
df_sup_n.rename(columns={0:'n'},inplace=True)
df_sup_n = df_sup_n.assign(star=lambda x: pd.Categorical(x.star, np.sort(np.unique(x.star))))


############################
# --- (4) DISTRIBUTION --- #

dat_prec = df_agg.query('metric=="prec"').reset_index(None,True)
dat_prec.drop(columns=['model','metric','n','se'], inplace=True)

# (i) More training data is not always better
dat_prec_dtrain = dat_prec.query('h_rtrain==1 & nval==2').reset_index(None, True)
dat_prec_dtrain.drop(columns=['h_rtrain','nval'],inplace=True)
dat_prec_dtrain = dat_prec_dtrain.assign(dtrain=lambda x: pd.Categorical(x.dtrain, np.sort(x.dtrain.unique())))

gg_dtrain = (pn.ggplot(dat_prec_dtrain,pn.aes(y='value',x='dtrain')) + 
    pn.labs(y='Precision',x='# of training days') + 
    pn.theme_bw() + 
    pn.ggtitle('Distribution of forecasting horizon') + 
    pn.geom_boxplot())
gg_save(fn='gg_dtrain.png',fold=dir_figures,gg=gg_dtrain,width=6,height=4)

# (ii) More validation data is not always better
dat_prec_nval = dat_prec.query('dtrain==360 & h_rtrain==1').reset_index(None, True)
dat_prec_nval.drop(columns=['dtrain','h_rtrain'],inplace=True)
dat_prec_nval = dat_prec_nval.assign(nval=lambda x: pd.Categorical(x.nval, np.sort(x.nval.unique())))

gg_nval = (pn.ggplot(dat_prec_nval,pn.aes(y='value',x='nval')) + 
    pn.labs(y='Precision', x='# of validation days') + 
    pn.theme_bw() + 
    pn.ggtitle('Distribution of forecasting horizon') + 
    pn.geom_boxplot())
gg_save(fn='gg_nval.png',fold=dir_figures,gg=gg_nval,width=6,height=4)

# (iii) Faster retraining is usually better
dat_prec_rtrain = dat_prec.query('dtrain==360 & nval==2').reset_index(None, True)
dat_prec_rtrain.drop(columns=['dtrain','nval'],inplace=True)
dat_prec_rtrain.rename(columns={'h_rtrain':'rtrain'}, inplace=True)
dat_prec_rtrain = dat_prec_rtrain.assign(rtrain=lambda x: pd.Categorical(x.rtrain, np.sort(x.rtrain.unique())))

gg_rtrain = (pn.ggplot(dat_prec_rtrain,pn.aes(y='value',x='rtrain')) + 
    pn.labs(y='Precision', x='# of days before retraining') + 
    pn.theme_bw() + 
    pn.ggtitle('Distribution of forecasting horizon') + 
    pn.geom_boxplot())
gg_save(fn='gg_rtrain.png',fold=dir_figures,gg=gg_rtrain,width=6,height=4)


####################
# --- (5) PLOT --- #

# -- (i) Supremum by lead -- #
gg_sup_hp = (pn.ggplot(df_sup,pn.aes(x='lead',y='star',color='metric')) + 
    pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
    pn.scale_color_discrete(name='Metric') + 
    pn.theme(subplots_adjust={'wspace': 0.25}) + 
    pn.facet_wrap('~msr',nrow=1,scales='free_y') + 
    pn.labs(x='Forecasting lead',y='Winning parameter (Days)'))
gg_save(fn='gg_sup_hp.png',fold=dir_figures,gg=gg_sup_hp,width=16,height=4)

# -- (ii) Supremum by freq -- #
gg_sup_hp_n = (pn.ggplot(df_sup_n.query('n>0'),pn.aes(x='star',y='n',color='metric')) +
    pn.theme_bw() + 
    pn.geom_point() +
    pn.scale_y_continuous(limits=[0,24],breaks=list(range(2,25,2))) +
    pn.theme(subplots_adjust={'wspace': 0.25,'hspace': 0.25}) + 
    pn.facet_wrap('~msr',scales='free') + 
    pn.labs(x='Hyperparameter value',y='# of leads (/24)'))
gg_save(fn='gg_sup_hp_n.png',fold=dir_figures,gg=gg_sup_hp_n,width=14,height=3.5)

# -- (iii) Full information -- #
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
        pn.facet_wrap('~lead',labeller=pn.label_both,scales='free_y',nrow=4))
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

