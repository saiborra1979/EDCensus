"""
EVALUATE THE ONE-DAY-AHEAD PREDICTIONS
"""


import os
import pandas as pd
from plotnine import *
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from funs_support import cindex, smoother

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')

# Load data
res_lasso = pd.read_csv(os.path.join(dir_flow, 'res_lasso.csv')).drop(columns=['Unnamed: 0'])
# bhat_lasso = pd.read_csv(os.path.join(dir_flow, 'bhat_lasso.csv')).drop(columns=['Unnamed: 0'])

#####################################
# --- STEP 1: AGGREGATE RESULTS --- #

# PERFORMANCE STARTS TO DETERIORATE ON MARCH 16
r2_lasso = res_lasso.groupby(['lead','year','month','day']).apply(lambda x: pd.Series({'r2':r2(x.y, x.pred), 'rmse':mse(x.y,x.pred,squared=False),'conc':cindex(x.y.values,x.pred.values)}))
r2_lasso = r2_lasso.reset_index().assign(date=lambda x: pd.to_datetime(x.year.astype(str)+'-'+x.month.astype(str)+'-'+x.day.astype(str)))
r2_lasso = r2_lasso.melt(r2_lasso.columns.drop(['r2','rmse','conc']), None, 'metric')
r2_lasso['tmp'] = r2_lasso.groupby(['lead','metric']).cumcount()
holder = r2_lasso.groupby(['lead','metric']).apply(lambda x: pd.Series({'smooth':smoother(x=x.value.values, lam=0.1),'idx':x.tmp.values}))
holder = holder.explode('smooth').drop(columns='idx').reset_index().assign(tmp=holder.explode('idx').idx.values)
holder['tmp'] = holder.tmp.astype(int)
holder['smooth'] = holder.smooth.astype(float)
r2_lasso = r2_lasso.merge(holder,'left',on=['lead','metric','tmp'])

################################
# --- STEP 2: PLOT RESULTS --- #

gg_r2 = (ggplot(r2_lasso, aes(x='date',y='value')) +
         geom_point(size=0.1, alpha=0.5) + geom_line(alpha=0.5) +
         theme_bw() + labs(y='Value') +
         theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
         ggtitle('One-day-ahead performance by lead') +
         facet_grid('metric~lead',labeller=label_both,scales='free_y') +
         scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y') +
         geom_line(aes(x='date',y='smooth'),color='blue'))
gg_r2.save(os.path.join(dir_figures, 'gg_r2.png'), height=9, width=10)


