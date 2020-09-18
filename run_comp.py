"""
SCRIPT TO DECOMPOSE MODEL RESULTS
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_figures = os.path.join(dir_base, '..', 'figures')

############################################
# --- (1) Compare 72 hours pretraining --- #

# NOTE! RETRAIN MEANS THERE IS NO PRETRAINING, WHEREAS ITERATIVE HAS PRETRAINING
dir_iterative = os.path.join(dir_test,'iterative')
dir_retrain = os.path.join(dir_test,'retrain')
fn_iterative = pd.Series(os.listdir(dir_iterative))
fn_retrain = pd.Series(os.listdir(dir_retrain))
assert fn_iterative.isin(fn_retrain).all()
di = {'iterative':fn_iterative, 'retrain':fn_retrain}

holder = []
for fold in di:
    storage = []
    for fn in di[fold]:
        storage.append(pd.read_csv(os.path.join(dir_test, fold, fn)))
    holder.append(pd.concat(storage).assign(tt=fold))
df_72 = pd.concat(holder).reset_index(None,True)
df_72.date = pd.to_datetime(df_72.date)
# Lets get monthly performance by lead
cn_gg = ['tt','ntrain','lead']
r2_72 = df_72.assign(month=lambda x: x.date.dt.month).groupby(cn_gg+['date']).apply(lambda x: r2_score(x.y, x.pred)).reset_index().rename(columns={0:'r2'})
# Average daily performance over the months
r2_72 = r2_72.assign(month=lambda x: x.date.dt.month).groupby(cn_gg+['month']).r2.mean().reset_index()
r2_72 = r2_72.pivot_table('r2',list(r2_72.columns.drop(['r2','tt'])),'tt').assign(spread=lambda x: x.iterative - x.retrain).reset_index()
#
gg_72 = (ggplot(r2_72, aes(x='lead',y='spread',color='ntrain.astype(str)')) + theme_bw() +
         geom_point(size=2) + facet_wrap('~month') +
         labs(x='Forecasting lead',y='R-squared (Pretraining - No pretraining)') +
         ggtitle('Monthly R2-performance by retraining strategy\nPretraining (iterative) dominates') +
         scale_color_discrete(name='Training hours') +
         scale_x_continuous(breaks=list(range(1,25))) +
         geom_hline(yintercept=0,linetype='--'))
gg_72.save(os.path.join(dir_figures, 'gg_72.png'), height=8, width=14)

