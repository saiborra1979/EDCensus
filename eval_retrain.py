"""
SCRIPT TO DECOMPOSE MODEL RESULTS
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score
from funs_support import find_dir_olu, gg_save

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_dtrain = os.path.join(dir_test, 'dtrain')
assert os.path.exists(dir_dtrain)

##################################
# --- (1) COMPARE DTRAIN 1-7 --- #
# Note: Validation days are zero

cn_gg = ['model','groups','dtrain','lead']
cn_desc = ['mean','std','25%','75%']
lbl_desc = ['Mean','Std. Dev', 'Q25','Q75']
di_desc = dict(zip(cn_desc,lbl_desc))

fn_dtrain = pd.Series(os.listdir(dir_dtrain))

holder = []
for fn in fn_dtrain:
    holder.append(pd.read_csv(os.path.join(dir_dtrain, fn)))
df_iter = pd.concat(holder).reset_index(None,True)
df_iter = df_iter.assign(dtrain=lambda x: (x.ntrain/24).astype(int),
        date=lambda x: pd.to_datetime(x.date)).drop(columns='ntrain')
assert df_iter.groupby(['model','groups','dtrain','lead']).size().unique().shape[0] == 1
# Find the overlapping leads
tmp = df_iter.groupby(['model','groups','lead']).size().reset_index().lead.value_counts()
leads = np.sort(tmp[tmp > 1].index)
df_iter = df_iter.query('lead.isin(@leads) & dtrain>1').reset_index(None,True)
ntrain = df_iter.dtrain.unique()

# Aggregate R2
dat_r2_agg = df_iter.groupby(cn_gg).apply(lambda x: r2_score(x.y,x.pred)).reset_index().rename(columns={0:'r2'})
dat_r2_agg[['dtrain','lead']] = dat_r2_agg[['dtrain','lead']].apply(pd.Categorical,0)

# Variations in daily performance
dat_r2_daily = df_iter.groupby(cn_gg+['date']).apply(lambda x: r2_score(x.y,x.pred)).reset_index().rename(columns={0:'r2'})
qq = dat_r2_daily.query('model=="mgpy" & dtrain==3').copy()
dat_r2_daily = dat_r2_daily.groupby(cn_gg).r2.describe()[cn_desc].reset_index()
dat_r2_daily[['dtrain','lead']] = dat_r2_daily[['dtrain','lead']].apply(pd.Categorical,0)
dat_r2_daily = dat_r2_daily.melt(cn_gg,None,'moment','r2')

fn_dtrain[fn_dtrain.str.contains('mgpy')].to_list()
fn_tmp = 'res_mgpy_dstart_60_dtrain_2_dval_0_groups_mds-arr-CTAS.csv'
res_tmp = pd.read_csv(os.path.join(dir_dtrain, fn_tmp)).drop(columns=['model','ntrain','groups'])
res_tmp = res_tmp.query('date==date.min()')
res_tmp
res_tmp.query('date==date.min()').groupby('lead').apply(lambda x: r2_score(x.y,x.pred))
res_tmp.query('date==date.min()').groupby('hour').apply(lambda x: r2_score(x.y,x.pred))

#######################
# --- (2) PLOT IT --- #

shpz = list('$'+pd.Series(ntrain).astype(str)+'$')

posd = position_dodge(0.5)
gg_r2_agg = (ggplot(dat_r2_agg,aes(x='lead',y='r2',color='model',shape='dtrain')) + 
    theme_bw() + geom_point(position=posd,size=2.5) + 
    scale_shape_manual(values=shpz) + 
    scale_color_discrete(name='Model',labels=['GP','MultiTask']) + 
    guides(shape=False) + 
    ggtitle('Numbers indicate training days') + 
    labs(y='R-squared',x='Forecasting horizon (hours)') + 
    ggtitle('Aggregated R2 between models/training days'))
gg_save('gg_r2_agg.png',dir_figures,gg_r2_agg,8,5)

gg_r2_desc = (ggplot(dat_r2_daily,aes(x='lead',y='r2',color='model',shape='dtrain')) + 
    theme_bw() + geom_point(position=posd,size=2.0) + 
    facet_wrap('~moment',labeller=labeller(moment=di_desc),scales='free_y') + 
    theme(subplots_adjust={'wspace': 0.25}) + 
    guides(shape=False) + 
    scale_color_discrete(name='Model',labels=['GP','MultiTask']) + 
    labs(y='R-squared',x='Forecasting horizon (hours)') + 
    scale_shape_manual(values=shpz))
gg_save('gg_r2_desc.png',dir_figures,gg_r2_desc,12,8)



# ############################################
# # --- (1) Compare 72 hours pretraining --- #

# # NOTE! RETRAIN MEANS THERE IS NO PRETRAINING, WHEREAS ITERATIVE HAS PRETRAINING
# dir_iterative = os.path.join(dir_test,'iterative')
# dir_retrain = os.path.join(dir_test,'retrain')
# fn_iterative = pd.Series(os.listdir(dir_iterative))
# fn_retrain = pd.Series(os.listdir(dir_retrain))
# assert fn_iterative.isin(fn_retrain).all()
# di = {'iterative':fn_iterative, 'retrain':fn_retrain}

# holder = []
# for fold in di:
#     storage = []
#     for fn in di[fold]:
#         storage.append(pd.read_csv(os.path.join(dir_test, fold, fn)))
#     holder.append(pd.concat(storage).assign(tt=fold))
# df_iter = pd.concat(holder).reset_index(None,True)
# df_iter.date = pd.to_datetime(df_iter.date)
# # Lets get monthly performance by lead
# cn_gg = ['tt','ntrain','lead']
# r2_iter = df_iter.assign(month=lambda x: x.date.dt.month).groupby(cn_gg+['date']).apply(lambda x: r2_score(x.y, x.pred)).reset_index().rename(columns={0:'r2'})
# # Average daily performance over the months
# r2_iter = r2_iter.assign(month=lambda x: x.date.dt.month).groupby(cn_gg+['month']).r2.mean().reset_index()
# r2_iter = r2_iter.pivot_table('r2',list(r2_iter.columns.drop(['r2','tt'])),'tt').assign(spread=lambda x: x.iterative - x.retrain).reset_index()
# r2_iter_month = r2_iter.drop(columns='lead').groupby(['ntrain','month']).mean().reset_index()
# # Aggregate over all months
# r2_iter_agg = r2_iter.groupby(['ntrain','lead']).spread.mean().reset_index()

# tmp = r2_iter_agg.query('ntrain==72').reset_index(None, True)
# gg_iter_comp = (ggplot(tmp, aes(x='lead',y='spread')) +
#                 theme_bw() + geom_point(size=2,color='blue') + geom_line() +
#                 labs(x='Forecasting lead',y='R-squared (iterative - no pretraining)') +
#                 ggtitle('Average over one-day-ahead R2') +
#                 scale_color_discrete(name='Training hours') +
#                 scale_x_continuous(breaks=list(range(1,25,2))) +
#                 geom_hline(yintercept=0,linetype='--'))
# gg_iter_comp.save(os.path.join(dir_figures, 'gg_iter_comp.png'), height=3, width=4)
