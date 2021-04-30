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

##################################
# --- (1) COMPARE DTRAIN 1-7 --- #
# Note: Validation days are zero






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
