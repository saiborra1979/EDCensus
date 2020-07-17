"""
EVALUATE THE ONE-DAY-AHEAD PREDICTIONS
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from funs_support import cindex, smoother

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')

# Load data
cn_drop = 'Unnamed: 0'
res_mdl = pd.read_csv(os.path.join(dir_flow, 'res_model.csv'))
if cn_drop in res_mdl.columns:
    res_mdl.drop(columns=cn_drop ,inplace=True)
res_mdl.dates = pd.to_datetime(res_mdl.dates)

# bhat_lasso = pd.read_csv(os.path.join(dir_flow, 'bhat_lasso.csv')).drop(columns=['Unnamed: 0'])

#####################################
# --- STEP 1: AGGREGATE RESULTS --- #

# PERFORMANCE STARTS TO DETERIORATE ON MARCH 16
r2_mdl = res_mdl.groupby(['model','lead','year','month','day']).apply(lambda x: pd.Series({'r2':r2(x.y, x.pred), 'rmse':mse(x.y,x.pred,squared=False),'conc':cindex(x.y.values,x.pred.values)}))
r2_mdl = r2_mdl.reset_index().assign(date=lambda x: pd.to_datetime(x.year.astype(str)+'-'+x.month.astype(str)+'-'+x.day.astype(str)))
r2_mdl = r2_mdl.melt(r2_mdl.columns.drop(['r2','rmse','conc']), None, 'metric')
r2_mdl['tmp'] = r2_mdl.groupby(['model','lead','metric']).cumcount()
holder = r2_mdl.groupby(['model','lead','metric']).apply(lambda x: pd.Series({'smooth':smoother(x=x.value.values, lam=0.1),'idx':x.tmp.values}))
holder = holder.explode('smooth').drop(columns='idx').reset_index().assign(tmp=holder.explode('idx').idx.values)
holder['tmp'] = holder.tmp.astype(int)
holder['smooth'] = holder.smooth.astype(float)
r2_mdl = r2_mdl.merge(holder,'left',on=['model','lead','metric','tmp'])

# r2_mdl.loc[r2_mdl[r2_mdl.metric=='r2'].groupby('model').value.idxmin()]
# qq = res_mdl[(res_mdl.dates<pd.to_datetime('2020-03-20')) &
#         (res_mdl.dates>=pd.to_datetime('2020-03-19')) &
#         (res_mdl.lead==4) & (res_mdl.model=='local')]
# np.corrcoef(qq.y, qq.pred)[0,1]

###################################################
# --- STEP 2: CAN WE LEARN NON-PARA BOUNDARY? --- #

# Goal: Draw a threshold so that only 5% of of actual values are below it
sub = res_mdl[(res_mdl.lead==4) & (res_mdl.model=='local')].drop(columns=['lead','model','year','day','hour']).reset_index(None, True).sort_values('dates').reset_index(None,True)
sub_train = sub[sub.dates < pd.to_datetime('2020-03-19')]
# Pick a test year
sub_test = sub[(sub.dates<pd.to_datetime('2020-03-26')) & (sub.dates>=pd.to_datetime('2020-03-19'))]
# Compare the distribution of residuals by month....




################################
# --- STEP 3: PLOT RESULTS --- #

gg_r2 = (ggplot(r2_mdl, aes(x='date',y='value',color='model')) +
         geom_point(size=0.1, alpha=0.5) + geom_line(alpha=0.5) +
         theme_bw() + labs(y='Value') +
         theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
         ggtitle('One-day-ahead performance by lead') +
         facet_grid('metric~lead',labeller=label_both,scales='free_y') +
         scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y') +
         geom_line(aes(x='date',y='smooth',color='model')))
gg_r2.save(os.path.join(dir_figures, 'gg_r2.png'), height=9, width=10)



