"""
EVALUATE THE ONE-DAY-AHEAD PREDICTIONS
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from funs_support import cindex, smoother, date2ymd, cvec

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

# Load in the GP data
res_gp = res_mdl[res_mdl.model.str.contains('gpy')].reset_index(None,True)

# Combine
res_mdl.drop(columns = ['lb','ub'], inplace=True)

#####################################
# --- STEP 1: AGGREGATE RESULTS --- #

di_mdl = {'lasso':'Lasso', 'local':'Locally weighted','torch_gpy':'GP'}

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
r2_mdl = r2_mdl.assign(model=lambda x: x.model.map(di_mdl))

# r2_mdl.loc[r2_mdl[r2_mdl.metric=='r2'].groupby('model').value.idxmin()]
# qq = res_mdl[(res_mdl.dates<pd.to_datetime('2020-03-20')) &
#         (res_mdl.dates>=pd.to_datetime('2020-03-19')) &
#         (res_mdl.lead==4) & (res_mdl.model=='local')]
# np.corrcoef(qq.y, qq.pred)[0,1]

###################################################
# --- STEP 2: CAN WE LEARN NON-PARA BOUNDARY? --- #

# # Goal: Draw a threshold so that only 5% of of actual values are below it
# sub = res_mdl[(res_mdl.lead==4) & (res_mdl.model=='local')].drop(columns=['lead','model','year','day','hour']).reset_index(None, True).sort_values('dates').reset_index(None,True)
# sub_train = sub[sub.dates < pd.to_datetime('2020-03-19')]
# # Pick a test year
# sub_test = sub[(sub.dates<pd.to_datetime('2020-03-26')) & (sub.dates>=pd.to_datetime('2020-03-19'))]
# # Compare the distribution of residuals by month....

##############################
# --- STEP 3: GP RESULTS --- #

from statsmodels.stats.proportion import proportion_confint as propCI

res_gp = res_gp.sort_values(['lead','year','month','day']).reset_index(None,True)
res_gp.dates = pd.to_datetime(res_gp.dates.astype(str) + ' ' + res_gp.hour.astype(str)+':00:00')
print('UB viol: %.3f, LB voil: %.3f' % (np.mean(res_gp.y > res_gp.ub), np.mean(res_gp.y < res_gp.lb)))

assert np.all(res_gp.groupby(['lead','dates']).size() == 1)
res_gp_long = res_gp.melt(['lead','dates','y'],['pred','lb','ub'],'measure')

# Group the upper/lower bound violations by quntile
y_quints = np.quantile(res_gp.y,np.arange(0,1.01,0.1))
y_quints[-1] += 1
tmp = res_gp.assign(qq=pd.cut(res_gp.y, y_quints, False)).groupby(['qq','lead'])
tmp = tmp.apply(lambda x: pd.Series({'lb':np.sum(x.y < x.lb),'ub':np.sum(x.y > x.ub),'n':x.y.shape[0]})).reset_index()
dat_viol = pd.concat([tmp,pd.DataFrame(tmp[['lb','ub']].values / cvec(tmp.n),columns=['rate_lb','rate_ub'])],1)
tmp2 = tmp.groupby('lead')[['lb','ub','n']].sum().reset_index()
tmp2 = tmp2.assign(rate_lb=lambda x: x.lb/x.n, rate_ub=lambda x: x.ub/x.n)
cn = ['lead','rate_lb','rate_ub']
dat_viol = dat_viol.merge(tmp2[cn],'left','lead',suffixes=('','_agg'))
dat_viol_long = dat_viol.melt(['lead','qq','n'],['rate_lb','rate_ub'],'bound').merge(tmp2[cn],'left','lead')
tmp = pd.concat(propCI(count=(dat_viol_long.n * dat_viol_long.value).astype(int),
       nobs=dat_viol_long.n,alpha=0.05, method='beta'),1).rename(columns={0:'lb',1:'ub'})
dat_viol_long = pd.concat([dat_viol_long, tmp],1)

# Repeat for months
dat_viol_months = res_gp.groupby(['lead','month']).apply(lambda x: pd.Series({'lb':np.sum(x.y < x.lb),'ub':np.sum(x.y > x.ub),'n':x.y.shape[0]})).reset_index()
dat_viol_months = dat_viol_months.assign(rate_lb=lambda x: x.lb/x.n, rate_ub=lambda x: x.ub/x.n)
dat_months_long = dat_viol_months.melt(['lead','n','month'],['rate_lb','rate_ub'],'bound')
tmp = pd.concat(propCI(count=(dat_months_long.n * dat_months_long.value).astype(int),
       nobs=dat_months_long.n,alpha=0.05, method='beta'),1).rename(columns={0:'lb',1:'ub'})
dat_months_long = pd.concat([dat_months_long, tmp],1)


################################
# --- STEP 4: PLOT RESULTS --- #

di_measure = {'pred':'Mean','lb':'Lower-bound','ub':'Upper-bound'}

### SCATTERPLOT OF PREDICTED VS ACTUAL AND LB/UB ###
gg_viol = (ggplot(res_gp_long, aes(x='value',y='y')) + theme_bw() +
           geom_point(size=0.5, alpha=0.5) +
           geom_abline(intercept=0,slope=1,color='blue') +
           labs(x='Predicted',y='Actual') +
           facet_grid('lead~measure',labeller=labeller(measure=di_measure,lead=label_both),scales='free') +
           theme(panel_spacing_x=0.25) +
           ggtitle('Upper/lower-bound violations for Gaussian Process'))
gg_viol.save(os.path.join(dir_figures, 'gg_viol_scatter.png'), height=8, width=12)

### VIOLATION PERCENTAGE BY LEVEL OF RESPONSE ###
posd = position_dodge(1)
gg_quint = (ggplot(dat_viol_long, aes(x='qq',y='value',fill='bound')) + theme_bw() +
            # geom_bar(stat='identity',position=pd,color='black') +
            geom_point(position=posd,size=3) +
            facet_wrap('~lead',labeller=label_both) + guides(color=False) +
            geom_linerange(aes(x='qq',ymin='lb',ymax='ub',color='bound'),position=posd) +
            scale_fill_discrete(name='Bound',labels=['Lower-bound','Upper-bound']) +
            theme(axis_text_x=element_text(angle=90)) +
            labs(y='Violation rate',x='Max number of hourly patients') +
            geom_hline(yintercept=0.025) +
            ggtitle('GP violation rate by outcome level\n95% Prediction Interval'))
gg_quint.save(os.path.join(dir_figures, 'gg_viol_quint.png'), height=5.5, width=8)

### VIOLATION PERCENTAGE BY 2020 MONTH ###
gg_months = (ggplot(dat_months_long, aes(x='month.astype(str)',y='value',fill='bound')) +
            theme_bw() + geom_point(position=posd,size=3) +
            facet_wrap('~lead',labeller=label_both) + guides(color=False) +
            geom_linerange(aes(x='month.astype(str)',ymin='lb',ymax='ub',color='bound'),position=posd) +
            scale_fill_discrete(name='Bound',labels=['Lower-bound','Upper-bound']) +
            labs(y='Violation rate', x='Month (2020)') + geom_hline(yintercept=0.025) +
            ggtitle('GP violation rate by outcome month (2020)\n95% Prediction Interval'))
gg_months.save(os.path.join(dir_figures, 'gg_viol_month.png'), height=5, width=8)


### LEAD PERFORMANCE ###
di_metric = {'r2':'R-squared', 'conc':'Concordance', 'rmse':'RMSE'}

for metric in di_metric:
    print(metric)
    tmp = r2_mdl[(r2_mdl.metric==metric)]
    fn, lbl = 'gg_perf_' + metric + '.png', di_metric[metric]
    title = 'Performance by model: ' + lbl + '\nOne-day-ahead predictions'
    gg_tmp = (ggplot(tmp, aes(x='date',y='value',color='model')) +
                 geom_point(size=0.1, alpha=0.5) + geom_line(alpha=0.5) +
                 theme_bw() + labs(y=lbl) + scale_color_discrete(name='Model') +
                 theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
                 ggtitle(title) + facet_wrap('lead',labeller=label_both,scales='free_y') +
                 scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y') +
                 geom_line(aes(x='date',y='smooth',color='model')))
    gg_tmp.save(os.path.join(dir_figures, fn), height=5, width=8)




