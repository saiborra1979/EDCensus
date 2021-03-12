"""
EVALUATE THE ONE-DAY-AHEAD PREDICTIONS
"""

import os
import pandas as pd
import numpy as np
from plotnine import *

from funs_support import smoother_df, cvec, ymd2date, parallel_perf, ymdh2date, sens_spec_df
from scipy.stats import norm
from time import time
from sklearn.metrics import r2_score


dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')

# Load all model results
cn_drop = 'Unnamed: 0'
res_mdl = pd.read_csv(os.path.join(dir_flow, 'res_model.csv'))
if cn_drop in res_mdl.columns:
    res_mdl.drop(columns=cn_drop, inplace=True)
# Remove any rows with LB/UB
res_mdl = res_mdl[~(res_mdl.lb.notnull() | res_mdl.ub.notnull())].reset_index(None, True).drop(columns=['lb', 'ub'])
res_mdl.dates = pd.to_datetime(res_mdl.dates)

# Load the "actual" outcome
nleads = 24
tmp = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), nrows=2, usecols=range(nleads + 1 + 4), header=None)
assert np.all(tmp.iloc[0].fillna('y') == 'y')
act_y = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), skiprows=2, usecols=range(1 + 4))
act_y.columns = np.where(act_y.columns.str.contains('Unnamed'), '_'.join(tmp.iloc[:, 4]), act_y.columns)
assert act_y.columns[-1] == 'y_lead_0'
act_y = act_y.rename(columns={'y_lead_0': 'y'}).assign(date=ymdh2date(act_y))

# bhat_lasso = pd.read_csv(os.path.join(dir_flow, 'bhat_lasso.csv')).drop(columns=['Unnamed: 0'])
# Original run had only n=1000
mi_days = 45
res_mdl.ntrain = res_mdl.ntrain.fillna(int(mi_days * 24)).astype(int)
# groups is for GPy only
res_mdl.groups = res_mdl.groups.fillna('None')
print(res_mdl.groupby(['model', 'groups', 'lead']).size())

# Subset the GP data
res_gp = res_mdl[res_mdl.model.str.contains('gpy')].reset_index(None, True).drop(columns='model')
res_gp = res_gp.sort_values(['lead', 'year', 'month', 'day']).reset_index(None, True)
res_gp.dates = pd.to_datetime(res_gp.dates.astype(str) + ' ' + res_gp.hour.astype(str) + ':00:00')
res_gp = res_gp.reset_index(None, True)
# Save for later

# Create a comparison group
res_rest = res_mdl[res_mdl.groups == 'None']
# Subset to same lead
tmp = res_rest.groupby(['model']).lead.unique().to_list()
lead_intersect = pd.Series(list(set.intersection(*map(set, tmp)))).sort_values().reset_index(None, True)
res_rest = res_rest.query('lead.isin(@lead_intersect)', engine='python').reset_index(None, True)
# Subset to the shared date frame
res_rest.dates = pd.to_datetime(res_rest.dates.dt.strftime('%Y-%m-%d'))
tmp = res_rest.groupby('model').dates.unique().to_list()
date_intersect = pd.Series(list(set.intersection(*map(set, tmp)))).sort_values().reset_index(None, True)
res_rest = res_rest.query('dates.isin(@date_intersect)',engine='python').reset_index(None, True)

#####################################
# --- STEP 1: GP VS PARA MODELS --- #

di_mdl = {'lasso': 'Lasso', 'local': 'Locally weighted', 'gpy': 'GPR'}
cn_date = ['year', 'month', 'day']
cn_perf = ['r2', 'rmse', 'conc']

# PERFORMANCE STARTS TO DETERIORATE ON MARCH 16
cn_para = ['model', 'lead']
cn_multi = cn_para + cn_date
r2_mdl = parallel_perf(data=res_rest, gg=cn_multi)
r2_mdl.rename(columns=dict(zip(range(len(cn_multi)), cn_multi)), inplace=True)
r2_mdl.insert(0, 'date', ymd2date(r2_mdl[cn_date]))
r2_mdl = r2_mdl.melt(r2_mdl.columns.drop(cn_perf), None, 'metric')
cn_smooth = cn_para + ['metric']
r2_mdl = smoother_df(df=r2_mdl, gg=cn_smooth, lam=0.1).assign(model=lambda x: x.model.map(di_mdl))

################################
# --- STEP 2: GP GROUPINGS --- #

cn_desc = ['mean', '25%', '50%', '75%']
di_desc = dict(zip(cn_desc, ['mu', 'lb', 'med', 'ub']))
cn_gp = ['groups', 'ntrain', 'lead']
cn_multi = cn_gp + cn_date

# Load existing file?
stime = time()
perf_gp = parallel_perf(data=res_gp, gg=cn_multi)
print('Took %i seconds to calculate' % (time() - stime))
perf_gp.rename(columns=dict(zip(range(len(cn_multi)), cn_multi)), inplace=True)
perf_gp.insert(0, 'date', ymd2date(perf_gp[cn_date]))
perf_gp = perf_gp.melt(perf_gp.columns.drop(cn_perf), None, 'metric')
cn_smooth = cn_gp + ['metric']
perf_gp = smoother_df(df=perf_gp, gg=cn_smooth, lam=0.1)
# Average over the dates
grpz_gp = perf_gp.groupby(cn_smooth).value.describe()[cn_desc].reset_index().rename(columns=di_desc)
# Average over the leads
leads_gp = perf_gp.groupby(cn_gp[:-1] + ['metric']).value.describe()[cn_desc].reset_index().rename(columns=di_desc)
leads_gp = leads_gp.sort_values(['metric','mu']).reset_index(None,True)
# Average over the months
months_gp = perf_gp.assign(month=lambda x: x.date.dt.month).groupby(cn_gp[:-1] + ['month','metric'])
months_gp = months_gp.value.describe()[cn_desc].reset_index().rename(columns=di_desc)

####################################
# --- STEP 3: ESCLATION LEVELS --- #

# Compare predicted level to current one, and calculate sensitivity/precision
res_class_grpz = res_gp.drop(columns=cn_date + ['hour', 'se'])
res_class_grpz.rename(columns={'dates': 'date_rt', 'y': 'y_pred', 'pred': 'hat_pred'}, inplace=True)
res_class_grpz = res_class_grpz.assign(date_pred=lambda x: x.date_rt + pd.to_timedelta(x.lead, unit='H'))
res_class_grpz = res_class_grpz.merge(
    act_y.rename(columns={'y': 'y_rt', 'date': 'date_rt'}).drop(columns=cn_date + ['hour']),
    'left', 'date_rt')
# Make the cuts
cn_pred = ['y_rt', 'y_pred', 'hat_pred']
ymx = res_class_grpz[cn_pred].max().max()+1
ymi = res_class_grpz[cn_pred].min().min()-1
esc_bins = [ymi, 31, 38, 48, ymx]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
res_class_grpz[cn_pred] = res_class_grpz[cn_pred].apply(
    lambda x: pd.Categorical(pd.cut(x, esc_bins, False, esc_lbls)).codes)
res_class_grpz = res_class_grpz.melt(cn_gp + ['date_rt', 'y_rt'], ['y_pred', 'hat_pred'], 'tt', 'value')
res_class_grpz = res_class_grpz.assign(delta=lambda x: x.value - x.y_rt) #np.sign()
res_class_grpz = res_class_grpz.pivot_table('delta', cn_gp + ['date_rt', 'y_rt'], 'tt').reset_index()
res_class_grpz.rename(columns={'hat_pred': 'pred', 'y_pred': 'y'}, inplace=True)
cn_y = ['y_rt', 'pred', 'y']

# Get sensitivity/specificity by: 1) agg, 2) months, 3) full deltas
tmp = res_class_grpz.copy()
tmp[cn_y] = np.sign(tmp[cn_y])
res_sp_agg = sens_spec_df(df=tmp, gg=cn_gp)
tmp = tmp.assign(month=lambda x: x.date_rt.dt.month)
res_sp_month = sens_spec_df(df=tmp, gg=cn_gp + ['month'])
del tmp
res_sp_full = sens_spec_df(df=res_class_grpz, gg=cn_gp)

# # Rank
# cn_rank = ['metric', 'ntrain', 'lead', 'value']
# res_class_rank = res_class_both.query('pred==1').drop(columns='pred').sort_values(cn_rank).reset_index(None, True)
# res_class_rank['ridx'] = res_class_rank.groupby(cn_rank[:-1]).cumcount()
# res_class_rank = res_class_rank.groupby(['metric', 'groups']).ridx.mean().reset_index()

# Save the best model for later
best_mdl = res_sp_agg.query('pred==1').groupby(['groups','ntrain']).value.mean().sort_values(ascending=False).head(1).reset_index().drop(columns=['value'])
print('Best model: %s\n' % best_mdl.T)
best_ylbl = best_mdl.merge(res_class_grpz)
best_ypred = best_mdl.merge(res_gp)
best_ylbl.to_csv(os.path.join(dir_output, 'best_ylbl.csv'),index=False)
best_ypred.to_csv(os.path.join(dir_output, 'best_ypred.csv'),index=False)

# Why the hours of the day are so powerful...
tmp = act_y.query('date >= "2020-03-01"').drop(columns='year').sort_values(['month','hour','day'])
tmp = tmp.assign(hour=lambda x: pd.Categorical(x.hour, list(range(24))))
gg_hour = (ggplot(tmp, aes(x='hour',y='y')) +
           theme_bw() + facet_wrap('~month',labeller=label_both) +
           geom_jitter(random_state=1,size=0.5,alpha=0.5,height=0,width=0.1) +
           geom_boxplot() + labs(x='Hour of day') +
           ggtitle('Hour of day explains a lot of the variation (>Mar 2020)'))
gg_hour.save(os.path.join(dir_figures, 'gg_hour.png'), height=8, width=16)
# Compare today's hour to previous hour
y_mar2020 = act_y.query('date >= "2020-03-01"').drop(columns='year').assign(doy=lambda x: x.date.dt.dayofyear).drop(columns='date')
y_mar2020 = y_mar2020.sort_values(['hour','doy']).reset_index(None,True)
y_mar2020 = y_mar2020.assign(y1=y_mar2020.groupby('hour').y.shift(1,fill_value=-1)).query('y1>0')
print(y_mar2020.groupby('hour').apply(lambda x: np.corrcoef(x.y, x.y1)[0,1]))
# Scatter
gg_hour2 = (ggplot(y_mar2020, aes(x='y1',y='y',color='month')) +
           theme_bw() + facet_wrap('~hour',labeller=label_both,scales='free') +
           geom_point() + labs(x="Previous day's hour", y="Current day's hour") +
           ggtitle('Hour of day explains a lot of the variation (>Mar 2020)') +
           theme(subplots_adjust={'hspace': 0.20, 'wspace':0.15}) +
            geom_abline(intercept=0,slope=1))
gg_hour2.save(os.path.join(dir_figures, 'gg_hour2.png'), height=16, width=16)

# #################################
# # --- STEP X: GP STATISTICS --- #
#
# # CI for missing value
# res_gp = res_gp.assign(lb=lambda x: x.pred - norm.ppf(0.975) * x.se, ub=lambda x: x.pred + norm.ppf(0.975) * x.se)
# tmp = res_gp.groupby(['groups', 'lead']).apply(lambda x:
#                                                pd.Series({'lb': np.mean(x.y < x.lb),
#                                                           'ub': np.mean(x.y > x.ub)})).reset_index()
# print(tmp.head(50))
# # Remove any duplicate lead/group/days with the better performing one
# cn_date = ['year', 'month', 'day', 'hour']
# cn_gg = ['groups', 'ntrain', 'lead']
# cn_bound = ['lb', 'ub', 'n']
# cn_rate = ['rate_lb', 'rate_ub']
# res_gp['idx'] = res_gp.groupby(cn_gg + cn_date).cumcount()
# mse_idx = res_gp.groupby(cn_gg + cn_date[:-1] + ['idx']).apply(lambda x: mse(x.y, x.pred))
# mse_idx = mse_idx.reset_index().rename(columns={0: 'mse'})
# mse_idx = mse_idx.groupby(cn_gg + cn_date[:-1]).apply(lambda x: x.loc[x.mse.idxmin()]).reset_index(None, True)
# res_gp = mse_idx.drop(columns=['mse']).merge(res_gp, 'inner').drop(columns=['idx'])
# assert np.all(res_gp.groupby(cn_gg + cn_date).size() == 1)
# res_gp_long = res_gp.melt(cn_gg + ['dates', 'y'], ['pred', 'lb', 'ub'], 'measure')
#
# # Group the upper/lower bound violations by quntile
# p_seq = np.arange(0, 1.01, 0.2)
# y_quints = np.quantile(res_gp.y, p_seq)
# y_quints[-1] += 1
# tmp = res_gp.assign(qq=pd.cut(res_gp.y, y_quints, False)).groupby(cn_gg + ['qq'])
# tmp = tmp.apply(
#     lambda x: pd.Series({'lb': np.sum(x.y < x.lb), 'ub': np.sum(x.y > x.ub), 'n': x.y.shape[0]})).reset_index()
# tmp = tmp[tmp.isnull().sum(1) == 0].reset_index(None, True)
# tmp.qq = tmp.qq.astype('interval[int64]')
# tmp[cn_bound] = tmp[cn_bound].astype(int)
# dat_viol = pd.concat([tmp, pd.DataFrame(tmp[cn_bound[:-1]].values / cvec(tmp.n), columns=cn_rate)], 1)
# # tmp2 = tmp.groupby(cn_gg)[cn_bound].sum().reset_index()
# # tmp2 = tmp2.assign(rate_lb=lambda x: x.lb / x.n, rate_ub=lambda x: x.ub / x.n)
# # cn = cn_gg + cn_rate
# # dat_viol = dat_viol.merge(tmp2[cn], 'left', cn_gg, suffixes=('', '_agg'))
# dat_viol_long = dat_viol.melt(cn_gg + ['qq', 'n'], cn_rate, 'bound')  # .merge(tmp2[cn], 'left', cn_gg)
# tmp = pd.concat(propCI(count=(dat_viol_long.n * dat_viol_long.value).astype(int),
#                        nobs=dat_viol_long.n, alpha=0.05, method='beta'), 1).rename(columns={0: 'lb', 1: 'ub'})
# dat_viol_long = pd.concat([dat_viol_long, tmp], 1)
#
# # Repeat for months
# dat_viol_months = res_gp.groupby(cn_gg + ['month']).apply(
#     lambda x: pd.Series({'lb': np.sum(x.y < x.lb), 'ub': np.sum(x.y > x.ub), 'n': x.y.shape[0]})).reset_index()
# dat_viol_months = dat_viol_months.assign(rate_lb=lambda x: x.lb / x.n, rate_ub=lambda x: x.ub / x.n)
# dat_months_long = dat_viol_months.melt(cn_gg + ['n', 'month'], ['rate_lb', 'rate_ub'], 'bound')
# tmp = pd.concat(propCI(count=(dat_months_long.n * dat_months_long.value).astype(int),
#                        nobs=dat_months_long.n, alpha=0.05, method='beta'), 1).rename(columns={0: 'lb', 1: 'ub'})
# dat_months_long = pd.concat([dat_months_long, tmp], 1)

################################
# --- STEP 4: PLOT METRICS --- #

modian_n = perf_gp.ntrain.mode().values[0]
modian_grp = grpz_gp.groups.mode().values[0]
lead_sub = lead_intersect[0]

di_metric = {'r2': 'R-squared', 'conc': 'Concordance', 'rmse': 'RMSE'}

### PARA VS BL-GP ###
for metric in di_metric:
    print(metric)
    tmp = r2_mdl.query('metric == @metric & lead==@lead_sub')
    fn, lbl = 'gg_perf_' + metric + '.png', di_metric[metric]
    title = lbl + ': One-day-ahead prediction (' + str(lead_sub) + ' hour lead)'
    # title = 'Performance by model: ' + lbl + '\nOne-day-ahead predictions'
    gg_tmp = (ggplot(tmp, aes(x='date', y='value', color='model')) +
              geom_point(size=0.1, alpha=0.5) + geom_line(alpha=0.5) +
              theme_bw() + labs(y=lbl) + scale_color_discrete(name='Model') +
              theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90),
              legend_position=(0.60,0.80), subplots_adjust={'wspace': 0.25},
              legend_background=element_blank(),legend_direction='horizontal') +
              ggtitle(title) +
              scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y') +
              geom_line(aes(x='date', y='smooth', color='model')))
    # facet_wrap('lead', labeller=label_both, scales='free_y') +
    gg_tmp.save(os.path.join(dir_figures, fn), height=3, width=5)

### GPs for different groups ###

for metric in di_metric:
    print(metric)
    tmp = perf_gp.query('metric==@metric & ntrain==@modian_n')
    fn, lbl = 'gg_gpy_perf_' + metric + '.png', di_metric[metric]
    title = 'Performance by feature group (stat=%s, n=%i)' % (lbl, modian_n)
    print(title)
    gg_tmp = (ggplot(tmp, aes(x='date', y='smooth', color='groups')) +
              geom_line(size=1) + theme_bw() + labs(y=lbl) +
              scale_color_discrete(name='Model') + ggtitle(title) +
              theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90),
                    subplots_adjust={'wspace': 0.25}) +
              facet_wrap('~lead', scales='free_y', labeller=label_both) +
              scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y'))
    gg_tmp.save(os.path.join(dir_figures, fn), height=16, width=17)

### GPs perf by group/N over leads ###
tmp = grpz_gp.query('ntrain==@modian_n')
gg_grpz = (ggplot(tmp, aes(x='lead', y='med', color='groups')) +
           theme_bw() + geom_point() + geom_line() +
           facet_wrap('~metric', labeller=labeller(metric=di_metric), scales='free_y') +
           ggtitle('Performance by lead (n=%i)' % modian_n) +
           theme(subplots_adjust={'wspace': 0.1}, legend_position='bottom') +
           scale_x_continuous(breaks=list(range(0, 24 + 1))) +
           labs(y='Value', x='Forecasting lead'))
gg_grpz.save(os.path.join(dir_figures, 'gg_grpz_leads.png'), height=6, width=18)
# Repeat for differnt N
tmp = grpz_gp.query('groups==@modian_grp').assign(ntrain=lambda x: x.ntrain.astype(str))
tmp.ntrain = pd.Categorical(tmp.ntrain, pd.Series(np.sort(grpz_gp.ntrain.unique())).astype(str))
gg_ntrain = (ggplot(tmp, aes(x='lead', y='med', color='ntrain')) +
           theme_bw() + geom_point() + geom_line() +
           facet_wrap('~metric', labeller=labeller(metric=di_metric), scales='free_y') +
           ggtitle('Performance by lead (group=%s)' % modian_grp) +
           theme(subplots_adjust={'wspace': 0.1}) +
           scale_x_continuous(breaks=list(range(0, 24 + 1))) +
           labs(y='Value', x='Forecasting lead') +
            scale_color_discrete(name='Training hours'))
gg_ntrain.save(os.path.join(dir_figures, 'gg_ntrain_leads.png'), height=6, width=18)
# Create a version with only the R2
tmp = grpz_gp.query('groups==@modian_grp & metric=="r2"').assign(ntrain=lambda x: (x.ntrain/24).astype(int))
tmp.ntrain = pd.Categorical(tmp.ntrain, pd.Series(np.sort(tmp.ntrain.unique())))
gg_ntrain_r2 = (ggplot(tmp, aes(x='lead', y='med', color='ntrain')) +
                theme_bw() + geom_point() + geom_line() +
                scale_x_continuous(breaks=list(range(1, 25, 2))) +
                labs(y='R-Squared', x='Forecasting lead') +
                scale_color_discrete(name='Training days') +
                ggtitle('Average over one-day-ahead forecasts') +
                theme(legend_position='right',legend_text=element_text(size=10),
                      legend_key=element_rect(fill='white',color='white'),legend_key_height=0.5))
gg_ntrain_r2.save(os.path.join(dir_figures, 'gg_ntrain_leads_r2.png'), height=3, width=4.5)


### GPs perf by group/N over months ###
tmp = months_gp.query('ntrain==@modian_n')
gg_grpz_months = (ggplot(tmp, aes(x='month', y='med', color='groups')) +
                   theme_bw() + geom_point() + geom_line() +
                   facet_wrap('~metric', labeller=labeller(metric=di_metric), scales='free_y') +
                   ggtitle('Performance by month (n=%i)' % modian_n) +
                   theme(subplots_adjust={'wspace': 0.1}, legend_position='bottom') +
                   scale_x_continuous(breaks=list(range(0, 12 + 1))) +
                   labs(y='Value', x='Forecast month'))
gg_grpz_months.save(os.path.join(dir_figures, 'gg_grpz_months.png'), height=6, width=18)
# Repeat for differnt N
tmp = months_gp.query('groups==@modian_grp').assign(ntrain=lambda x: x.ntrain.astype(str))
tmp.ntrain = pd.Categorical(tmp.ntrain, pd.Series(np.sort(grpz_gp.ntrain.unique())).astype(str))

gg_ntrain_months = (ggplot(tmp, aes(x='month', y='med', color='ntrain')) +
                   theme_bw() + geom_point() + geom_line() +
                   facet_wrap('~metric', labeller=labeller(metric=di_metric), scales='free_y') +
                   ggtitle('Performance by lead (group=%s)' % modian_grp) +
                   theme(subplots_adjust={'wspace': 0.1}, legend_position='bottom') +
                   scale_x_continuous(breaks=list(range(0, 12 + 1))) +
                   labs(y='Value', x='Forecast month'))
gg_ntrain_months.save(os.path.join(dir_figures, 'gg_ntrain_months.png'), height=6, width=18)

### GPs perf by group agg ###
tmp = leads_gp.query('ntrain==@modian_n')
gg_leads_gp = (ggplot(tmp, aes(x='groups', y='med')) +
               theme_bw() + geom_point() +
               geom_linerange(aes(ymin='lb', ymax='ub')) +
               facet_wrap('~metric', labeller=labeller(metric=di_metric), scales='free_y') +
               ggtitle('Average performance over all leads\nLine range in IQR') +
               theme(subplots_adjust={'wspace': 0.1}, legend_position='bottom',
                     axis_text_x=element_text(angle=90)) +
               labs(y='Value', x='Grouping'))
gg_leads_gp.save(os.path.join(dir_figures, 'gg_leads_agg.png'), height=5, width=15)

########################################
# --- STEP 5: PLOT PRECISON/RECALL --- #

di_pr = {'prec': 'Precision', 'sens': 'Recall'}
### Precision/Recall tradeoff ###
tmp = res_sp_agg.query('pred==1 & ntrain==@modian_n')
gg_pr_groups_lead = (ggplot(tmp, aes(x='lead', y='value', color='groups')) + theme_bw() +
               geom_point() + geom_line() +
               facet_wrap('~metric', labeller=labeller(metric=di_pr)) +
               ggtitle('Precision/Recall for Δ>0 in Esc Level (n=%i)' % modian_n) +
               labs(x='Forecast lead', y='Value') +
               theme(subplots_adjust={'wspace': 0.1}) +
               scale_x_continuous(breaks=list(range(25))))
gg_pr_groups_lead.save(os.path.join(dir_figures, 'gg_pr_groups_lead.png'), height=5, width=12)

tmp = res_sp_agg.query('pred==1 & groups==@modian_grp').assign(ntrain=lambda x: x.ntrain.astype(str))
tmp.ntrain = pd.Categorical(tmp.ntrain, pd.Series(np.sort(grpz_gp.ntrain.unique())).astype(str))
gg_pr_ntrain_lead = (ggplot(tmp, aes(x='lead', y='value', color='ntrain')) + theme_bw() +
               geom_point() + geom_line() +
               facet_wrap('~metric', labeller=labeller(metric=di_pr)) +
               ggtitle('Precision/Recall for Δ>0 in Esc Level (group=%s)' % modian_grp) +
               labs(x='Forecast lead', y='Value') +
               theme(subplots_adjust={'wspace': 0.1}) +
               scale_x_continuous(breaks=list(range(25))))
gg_pr_ntrain_lead.save(os.path.join(dir_figures, 'gg_pr_ntrain_lead.png'), height=5, width=12)

# gg_pr_rank = (ggplot(res_class_rank, aes(x='groups', y='ridx', color='metric')) +
#               theme_bw() + geom_point(size=3, position=position_dodge(0.5)) +
#               labs(y='Average rank') +
#               ggtitle('Rank-order across the leads by group\nHigher is better') +
#               scale_color_discrete(name='metric', labels=['PPV', 'Recall']))
# gg_pr_rank.save(os.path.join(dir_figures, 'gg_pr_rank.png'), height=5, width=6)

# di_measure = {'pred': 'Mean', 'lb': 'Lower-bound', 'ub': 'Upper-bound'}

# ### SCATTERPLOT OF PREDICTED VS ACTUAL AND LB/UB ###
# gg_viol = (ggplot(res_gp_long, aes(x='value', y='y')) + theme_bw() +
#            geom_point(size=0.5, alpha=0.5) +
#            geom_abline(intercept=0, slope=1, color='blue') +
#            labs(x='Predicted', y='Actual') +
#            facet_grid('lead~measure', labeller=labeller(measure=di_measure, lead=label_both), scales='free') +
#            theme(panel_spacing_x=0.25) +
#            ggtitle('Upper/lower-bound violations for Gaussian Process'))
# gg_viol.save(os.path.join(dir_figures, 'gg_viol_scatter.png'), height=8, width=12)
#
# ### VIOLATION PERCENTAGE BY LEVEL OF RESPONSE ###
# posd = position_dodge(1)
# gg_quint = (ggplot(dat_viol_long, aes(x='qq', y='value', fill='bound')) + theme_bw() +
#             # geom_bar(stat='identity',position=pd,color='black') +
#             geom_point(position=posd, size=3) +
#             facet_wrap('~lead', labeller=label_both) + guides(color=False) +
#             geom_linerange(aes(x='qq', ymin='lb', ymax='ub', color='bound'), position=posd) +
#             scale_fill_discrete(name='Bound', labels=['Lower-bound', 'Upper-bound']) +
#             theme(axis_text_x=element_text(angle=90)) +
#             labs(y='Violation rate', x='Max number of hourly patients') +
#             geom_hline(yintercept=0.025) +
#             ggtitle('GP violation rate by outcome level\n95% Prediction Interval'))
# gg_quint.save(os.path.join(dir_figures, 'gg_viol_quint.png'), height=5.5, width=8)
#
# ### VIOLATION PERCENTAGE BY 2020 MONTH ###
# gg_months = (ggplot(dat_months_long, aes(x='month.astype(str)', y='value', fill='bound')) +
#              theme_bw() + geom_point(position=posd, size=3) +
#              facet_wrap('~lead', labeller=label_both) + guides(color=False) +
#              geom_linerange(aes(x='month.astype(str)', ymin='lb', ymax='ub', color='bound'), position=posd) +
#              scale_fill_discrete(name='Bound', labels=['Lower-bound', 'Upper-bound']) +
#              labs(y='Violation rate', x='Month (2020)') + geom_hline(yintercept=0.025) +
#              ggtitle('GP violation rate by outcome month (2020)\n95% Prediction Interval'))
# gg_months.save(os.path.join(dir_figures, 'gg_viol_month.png'), height=5, width=8)

###################################################
# --- STEP X: CAN WE LEARN NON-PARA BOUNDARY? --- #

# # Goal: Draw a threshold so that only 5% of of actual values are below it
# sub = res_mdl[(res_mdl.lead==4) & (res_mdl.model=='local')].drop(columns=['lead','model','year','day','hour']).reset_index(None, True).sort_values('dates').reset_index(None,True)
# sub_train = sub[sub.dates < pd.to_datetime('2020-03-19')]
# # Pick a test year
# sub_test = sub[(sub.dates<pd.to_datetime('2020-03-26')) & (sub.dates>=pd.to_datetime('2020-03-19'))]
# # Compare the distribution of residuals by month....
