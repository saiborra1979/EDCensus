"""
SCRIPT TO COMPARE MODEL PERFORMANCE TO THE ESLCATION LEVELS
"""

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--nleads', type=int, default=25, help='Number of leads from process_flow.py')
# args = parser.parse_args()
# print(args)
# nleads = args.nleads

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score as r2
from scipy.stats import norm
from mizani import palettes
from funs_support import cvec, ymdh2date, r2_fun, ymd2date, sens_spec_df, add_CI, gg_color_hue
from statsmodels.tsa.stattools import acf, pacf

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_figures = os.path.join(dir_base, '..', 'figures')

cn_ymd = ['year', 'month', 'day']
cn_ymdh = cn_ymd + ['hour']
cn_ymdl = cn_ymd + ['lead']
cn_gg = ['dates', 'lead']
cn_y = ['y', 'pred', 'se']
cn_all = cn_gg + cn_y + cn_ymdh
# Load in the "best" performing model (rom run_eval.py) (note y_rt is the current escalation level for the prediction)
best_ylbl = pd.read_csv(os.path.join(dir_output, 'best_ylbl.csv'))
best_ylbl.date_rt = pd.to_datetime(best_ylbl.date_rt)
best_ypred = pd.read_csv(os.path.join(dir_output, 'best_ypred.csv'))
best_ypred.dates = pd.to_datetime(best_ypred.dates)
drop_cn = best_ypred.columns[~best_ypred.columns.isin(cn_gg + cn_y + cn_ymdh)].to_list()
best_ylbl.drop(columns = drop_cn, inplace=True)
best_ypred.drop(columns = drop_cn, inplace=True)

dmin, dmax = best_ypred.dates.min(), best_ypred.dates.max()

# Check everyone has same horizon
mx_month = best_ypred.month.unique().max()
mx_leads = best_ypred.lead.unique().shape[0]
mi_leads = best_ypred.lead.unique().min()

# Remove any non-full days
qq = best_ypred.groupby(cn_ymdl).size()
df_pred = qq[qq == 24].reset_index().drop(columns=[0]).merge(best_ypred)
date_mi = df_pred.dates.min()

# Extract the y label for the model
tmp = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), nrows=2, usecols=range(24 + 1 + 4), header=None)
assert np.all(tmp.iloc[0].fillna('y') == 'y')
act_y = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), skiprows=2, usecols=range(1 + 4))
act_y.columns = np.where(act_y.columns.str.contains('Unnamed'), '_'.join(tmp.iloc[:, 4]), act_y.columns)
assert act_y.columns[-1] == 'y_lead_0'
act_y = act_y.rename(columns={'y_lead_0': 'y'}).assign(date=ymdh2date(act_y), doy=ymd2date(act_y))
act_y.drop(columns=cn_ymdh, inplace=True)
act_y = act_y.query('date >= @date_mi').reset_index(None,True).assign(doy2=lambda x: x.date.dt.dayofyear)

##################################
# --- (1) Summary statistics --- #

# Levels: pre-esclation (<= 30), escalation level 1 (31-37), escalation level 2 (38-47), escalation level 3 (>=48)
ymi, ymx = best_ypred.pred.min(), best_ypred.pred.max()
esc_bins = [ymi - 1, 31, 38, 48, ymx + 1]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
di_esc = dict(zip(esc_lbls, range(len(esc_lbls))))
# Add on true labels to ground truth
act_y = act_y.assign(esc=lambda x: pd.cut(x.y, esc_bins, False, esc_lbls))

# Find daily propotion for plot
act_esc = act_y.groupby('doy').apply(lambda x: x.esc.value_counts(True)).reset_index().rename(
    columns={'esc': 'share', 'level_1': 'esc'})

# Transition probability matrix
mat_trans = act_y.assign(esct1=lambda x: x.esc.shift(1)).groupby(['esc', 'esct1']).y.count().reset_index()
mat_trans = mat_trans.merge(mat_trans.groupby('esc').y.sum().reset_index().rename(columns={'y': 'n'})).assign(
    prob=lambda x: x.y / x.n).drop(columns=['y', 'n'])

### DAILY SHARE OF ESCLATION LEVELS
colz = ['green','#e8e409','orange','red']
lblz = ['Normal','Level 1','Level 2','Level 3']
lblz = pd.Categorical(lblz, lblz)
gg_esc_share = (ggplot(act_esc, aes(x='doy', y='share', color='esc')) + theme_bw() +
                geom_point(size=0.5) + geom_line(size=1) + labs(y='Share') +
                ggtitle('Daily share of esclation levels (max patients per hour)') +
                theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
                scale_x_datetime(breaks='1 month', date_labels='%b, %Y') +
                scale_color_manual(name='Level',values=colz))
gg_esc_share.save(os.path.join(dir_figures, 'gg_esc_share.png'), height=6, width=10)

tmp = pd.DataFrame({'x':dmin-pd.DateOffset(days=8),'y':[15,35,45,55], 'lbl':lblz})
gg_act_y = (ggplot(act_y, aes(x='date', y='y')) + theme_bw() +
                geom_line(size=0.25,color='blue',alpha=0.75) + labs(y='Census') +
                ggtitle('Max patients per hour') +
                theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
                scale_x_datetime(breaks='1 month', date_labels='%b, %Y') +
            geom_hline(yintercept=31) + geom_hline(yintercept=38) + geom_hline(yintercept=48) +
            annotate('rect',xmin=dmin,xmax=dmax,ymin=0,ymax=31,fill='green',alpha=0.25)+
            annotate('rect',xmin=dmin,xmax=dmax,ymin=31,ymax=38,fill='#e8e409',alpha=0.25)+
            annotate('rect',xmin=dmin,xmax=dmax,ymin=38,ymax=48,fill='orange',alpha=0.25)+
            annotate('rect',xmin=dmin,xmax=dmax,ymin=48,ymax=act_y.y.max(),fill='red',alpha=0.25) +
            geom_text(aes(x='x',y='y',label='lbl',color='lbl'),data=tmp) +
            scale_color_manual(values=colz) + guides(color=False))
gg_act_y.save(os.path.join(dir_figures, 'gg_act_y.png'), height=5, width=15)

### TRANSITION PROBABILITIES
tit = 'One-hour ahead empirical transition probability'
gg_trans = (ggplot(mat_trans, aes(x='esct1', y='esc', fill='prob')) +
            theme_bw() + ggtitle(tit) +
            geom_tile(aes(width=1, height=1)) +
            labs(y='State t0', x='State t1') +
            scale_fill_gradient2(name='Probability', limits=[0, 1.01], breaks=list(np.linspace(0, 1, 5)),
                                 low='cornflowerblue', mid='grey', high='indianred', midpoint=0.5) +
            geom_text(aes(label='prob.round(2)')))
gg_trans.save(os.path.join(dir_figures, 'gg_trans.png'), height=5, width=5.5)

########################################
# --- (1) Get daily R2 performance --- #

di_desc = {'25%': 'lb', '50%': 'med', '75%': 'ub'}

df_r2 = df_pred.groupby(cn_ymd+['lead']).apply(lambda x: r2(x.y, x.pred)).reset_index().rename(columns={0: 'r2'})
# Get a k day average
df_r2 = df_r2.rename_axis('level_1').reset_index().merge(df_r2.groupby('lead').r2.rolling(window=7,center=True).mean().reset_index().rename(columns={'r2':'trend'}))
df_r2 = df_r2.assign(date = ymd2date(df_r2), leads=lambda x: (x.lead-1)//6+1)
tmp0 = pd.Series(range(1,25))
tmp1 = (((tmp0-1) // 6)+1)
tmp2 = ((tmp1-1)*6+1).astype(str) + '-' + (tmp1*6).astype(str)
di_leads = dict(zip(tmp1, tmp2))
df_r2.leads = pd.Categorical(df_r2.leads.map(di_leads),list(di_leads.values()))

### DAILY R2 TREND ###
gg_r2_best = (ggplot(df_r2, aes(x='date', y='trend', color='lead', groups='lead.astype(str)')) +
              theme_bw() + labs(y='R-squared') + geom_line() +
              theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90),
                    subplots_adjust={'wspace': 0.25}) +
              ggtitle('Daily R2 performance by lead (7 day average)') +
              scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y') +
              scale_color_cmap(name='Lead',cmap_name='viridis') +
              facet_wrap('~leads',labeller=label_both))
gg_r2_best.save(os.path.join(dir_figures, 'gg_r2_best.png'), height=8, width=13)

##########################################
# --- (2) ESCLATION PRECISION/RECALL --- #

di_pr = {'prec': 'Precision', 'sens': 'Recall'}

tmp = best_ylbl.copy()
for cn in ['y_rt', 'pred', 'y']:
    tmp[cn] = np.sign(tmp[cn])
res_sp_agg = sens_spec_df(df=tmp, gg=['lead'])
tmp = tmp.assign(month=lambda x: x.date_rt.dt.month)
res_sp_month = sens_spec_df(df=tmp, gg=['lead', 'month'])
del tmp
res_sp_full = sens_spec_df(df=best_ylbl, gg=['lead'])

print(res_sp_full.head())
print(res_sp_agg.head())

# Calculate the breakdown of TPs/FPs
tmp0 = pd.concat([res_sp_agg.query('pred==1').assign(pred=0), res_sp_full.query('pred>0')])
tmp1 = tmp0.query('metric=="prec"').assign(tp=lambda x: np.round(x.value*x.den,0).astype(int)).assign(fp=lambda x: x.den-x.tp).drop(columns=['den','value','metric'])
tmp2 = tmp0.query('metric=="sens"').assign(fn=lambda x: x.den-np.round(x.value*x.den,0).astype(int)).rename(columns={'den':'pos'}).drop(columns=['value','metric'])
res_tp_full = tmp1.merge(tmp2)
assert res_tp_full.assign(check=lambda x: x.tp+x.fn == x.pos).check.all()
res_tp_full = res_tp_full.melt(['lead','pred'],None,'metric','n')

### PRECISION/RECALL DECOMPOSITION ###
posd = position_dodge(0.5)
tmp1 = add_CI(res_sp_agg.query('pred==1').rename(columns={'den':'n'})).assign(pred=0)
tmp2 = add_CI(res_sp_full.query('pred > 0').rename(columns={'den':'n'}))
tmp = pd.concat([tmp1, tmp2]).reset_index(None,True)
colz = ['black'] + gg_color_hue(3)
lblz = ['≥0', '1', '2', '3']

tit = 'Precision/Recall for predicting Δ>0 in escalation\n95% CI (beta method)'
gg_sp_full = (ggplot(tmp, aes(x='lead', y='value', color='pred.astype(str)')) +
              theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
              scale_x_continuous(breaks=list(range(1,25))) +
              labs(x='Forecasting lead', y='Precision/Recall') +
              geom_linerange(aes(ymin='lb', ymax='ub'),position=posd) +
              facet_wrap('~metric', labeller=labeller(metric=di_pr)) +
              scale_color_manual(name='Δ esclation',values=colz, labels=lblz) +
              theme(legend_position=(0.5, -0.05),legend_direction='horizontal'))
gg_sp_full.save(os.path.join(dir_figures, 'gg_sp_full.png'), height=5, width=13)

gg_sp_agg = (ggplot(tmp1, aes(x='lead', y='value', color='metric')) +
             theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
             scale_color_discrete(labels=['Precision','Recall'],name='Metric') +
             scale_x_continuous(breaks=list(range(1,25))) +
             labs(x='Forecasting lead', y='Precision/Recall') +
             geom_linerange(aes(ymin='lb', ymax='ub'),position=posd))
gg_sp_agg.save(os.path.join(dir_figures, 'gg_sp_agg.png'), height=5, width=9)
# scale_y_continuous(limits=[0.3, 1.0], breaks=list(np.arange(0.3, 0.9, 0.2)))

tmp = add_CI(res_sp_month.query('pred==1').rename(columns={'den':'n'}))
height = int(np.ceil(len(tmp.month.unique()) / 2)) * 3
gg_sp_month = (ggplot(tmp, aes(x='lead', y='value', color='metric')) +
             theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
             scale_color_discrete(labels=['Precision','Recall'],name='Metric') +
             scale_x_continuous(breaks=list(range(1,25))) +
             labs(x='Forecasting lead', y='Precision/Recall') +
             geom_linerange(aes(ymin='lb', ymax='ub'),position=posd) +
             facet_wrap('~month',ncol=3,labeller=label_both))
gg_sp_month.save(os.path.join(dir_figures, 'gg_sp_month.png'), height=height, width=16)


# NOTE THAT PRECISION FOR ACTUAL DELTA == 1, DELTA ==2, etc IS LOWER THAT DELTA > 0 BECAUSE YOU CAN PREDICT DELTA=1, BUT DELTA=2, AND THEREFORE NOT COUNT AS A TRUE POSITIVE IN THE LATTER CASE
# tmp1 = res_sp_agg.query('lead==9 & pred==1 & metric=="prec"')
# tmp2 = res_sp_full.query('lead==9 & pred>0 & metric=="prec"')
# ntp, np = int(tmp1.value*tmp1.den), int(tmp1.den)

tit = 'Prediction breakdown for Δ>0 in escalation'
di_lblz = dict(zip(range(4),lblz))
di_metric = {'tp':'TP', 'fp':'FP', 'fn':'FN', 'pos':'Positive'}
tmp = res_tp_full.assign(pred=lambda x: pd.Categorical(x.pred.map(di_lblz),list(di_lblz.values())),
                         metric=lambda x: pd.Categorical(x.metric.map(di_metric), list(di_metric.values())))
gg_tp_full = (ggplot(tmp, aes(x='lead', y='n', color='metric')) +
             theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
             scale_x_continuous(breaks=list(range(1,25))) +
             labs(x='Forecasting lead', y='Count') +
             facet_wrap('~pred',scales='free_y',labeller=label_both) +
              scale_color_manual(values=colz, name='Metric') +
              theme(subplots_adjust={'wspace': 0.10}))
gg_tp_full.save(os.path.join(dir_figures, 'gg_tp_full.png'), height=7, width=14)


##########################################
# --- (6) EXAMPLE OF TRAJECTORY PLOT --- #

perf_days = df_pred.drop(columns=cn_ymdh+['se']).rename(columns={'dates':'date_rt'}).sort_values(['lead','date_rt'])
perf_days = perf_days.assign(date_pred=perf_days.assign(lead2=lambda x: x.lead).groupby('lead').apply(lambda x: x.date_rt+pd.DateOffset(hours=int(x.lead2.min()))).values).assign(doy_pred=lambda x: x.date_pred.dt.dayofyear, doy_rt=lambda x: x.date_rt.dt.dayofyear)
perf_r2 = perf_days.groupby('doy_pred').apply(lambda x: r2(x.y, x.pred)).sort_values().reset_index().rename(columns={0:'r2'})
kk = 3
perf_r2 = pd.concat([perf_r2.head(kk), perf_r2.tail(kk)])
holder = []
for ii, rr in perf_r2.iterrows():
    doy = int(rr['doy_pred'])
    tmp1 = act_y.query('doy2==@doy').drop(columns='doy').rename(columns={'doy2':'doy'})
    tmp2 = act_y[act_y.doy2.isin(range(doy-3, doy))].drop(columns='doy').rename(columns={'doy2':'doy'})
    tmp = pd.concat([tmp1.assign(tt='curr'), tmp2.assign(tt='prev')]).reset_index(None, True)
    tmp = tmp.assign(hour=lambda x: x.date.dt.hour, doy2=doy, r2=rr['r2'] > 0.9)
    holder.append(tmp)
perf_facet = pd.concat(holder)

gg_perf = (ggplot(perf_facet, aes(x='hour',y='y',alpha='tt',groups='doy.astype(str)',color='r2.astype(str)')) +
           theme_bw() + geom_line() +
           facet_wrap('~r2+doy2') +
           scale_alpha_manual(values=[1.0,0.3]) +
           geom_point(aes(x='hour',y='y')))
gg_perf.save(os.path.join(dir_figures, 'gg_perf.png'), height=8, width=8)


# Find best day
best_day = df_r2.groupby(cn_ymd).r2.min().sort_values(ascending=False).head(1).reset_index().drop(columns=['r2'])
best_day = pd.to_datetime(list(best_day.astype(str).apply(lambda x: '-'.join(x), 1))[0])
drange = pd.date_range(best_day, periods=24, freq='1H')  # mx_leads
df_range = best_ypred.query('dates.isin(@drange)').drop(columns=cn_ymdh).reset_index(None,True)
df_range = pd.concat([df_range.query('lead==1').assign(lead=0, se=0),df_range])
df_range = df_range.assign(date_pred=df_range.apply(lambda x: x['dates']+pd.DateOffset(hours=x['lead']),1)).rename(columns={'dates':'date_rt','y':'y_pred'})
df_range = df_range.merge(act_y[['y','date']].rename(columns={'date':'date_pred'}))
df_range = df_range.assign(pred=lambda x: np.where(x.lead==0, x.y, x.pred), y_pred=lambda x: np.where(x.lead==0, x.y, x.y_pred)).drop(columns='y')

df_tp = df_range.groupby('date_pred').y_pred.mean().reset_index()
tit = 'Example of real-time trajectory for 24 hours ahead\nDay: %s\nBlock dots are actual values' % best_day
gg_trajectory = (ggplot(df_range, aes(x='date_pred', y='pred',
                                      color='date_rt.dt.hour', group='date_rt.dt.hour')) +
                 theme_bw() + geom_path(alpha=0.5, arrow=arrow(type='closed', length=0.1)) +
                 labs(y='Max patient per hour') + ggtitle(tit) +
                 scale_x_datetime(breaks='3 hours', date_labels='%H') +
                 theme(axis_title_x=element_blank()) +
                 scale_color_cmap(name='Point of forecast') +
                 geom_point(aes(x='date_pred', y='y_pred'), color='black',size=1.5,inherit_aes=False,
                            data=df_tp) +
                 geom_hline(yintercept=31, linetype='--') +
                 geom_hline(yintercept=38, linetype='--') +
                 geom_hline(yintercept=48, linetype='--'))
gg_trajectory.save(os.path.join(dir_figures, 'gg_trajectory.png'), height=6, width=12)

tit = 'Example of real-time trajectory for 24 hours ahead\nDay: %s\nBlue dots are actual values\nPrediction interval is 80%%CI' % best_day
udates = list(pd.Series(df_range.date_rt.unique()))
colz = [palettes.cmap_pal('inferno')(z) for z in range(len(udates))]
for ii, date in enumerate(udates[1:]):
    pp = 0.9
    tmp = df_range.query('date_rt==@date').assign(lb=lambda x: x.pred-norm.ppf(pp)*x.se,
                                                  ub=lambda x: x.pred+norm.ppf(pp)*x.se)
    gg = (ggplot(tmp, aes(x='date_pred', y='pred')) +
          geom_path(arrow=arrow(type='closed', length=0.1),color='black') +
          labs(y='Max patient per hour') + ggtitle(tit) + theme_bw() +
          scale_x_datetime(breaks='3 hours', date_labels='%H') +
          theme(axis_title_x=element_blank()) +
          geom_point(aes(x='date_pred', y='y_pred'), color='blue', size=2,data=df_tp) +
          geom_ribbon(aes(ymin='lb', ymax='ub'), alpha=0.25, fill='red') +
          geom_hline(yintercept=31, linetype='--') + geom_hline(yintercept=38, linetype='--') +
          geom_hline(yintercept=48, linetype='--') + scale_y_continuous(limits=[-10, 60]))
    gg.save(os.path.join(dir_figures,'traj', 'gg_trajectory_' + date.strftime('%H') + '.png'), height=6, width=12)

# gg_trajectory = (ggplot(df_range, aes(x='dates2', y='pred', color='dates', group='dates')) +
#                  theme_bw() + geom_path(alpha=0.5, arrow=arrow(type='closed', length=0.1)) +
#                  labs(y='Max patient per hour') + ggtitle(tit) +
#                  scale_x_datetime(breaks='3 hours', date_labels='%H') +
#                  theme(axis_title_x=element_blank()) +
#                  scale_color_cmap(name='Point of forecast') +
#                  geom_point(aes(x='date', y='y'), color='black', inherit_aes=False, data=tmp_act, size=2) +
#                  geom_hline(yintercept=31, linetype='--') +
#                  geom_hline(yintercept=38, linetype='--') +
#                  geom_hline(yintercept=48, linetype='--') +
#                  geom_point(data=df_range.groupby('dates').head(1)))
# gg_trajectory.save(os.path.join(dir_figures, 'gg_trajectory.png'), height=6, width=12)

#########################################
# --- (X1) Calculate violation rates --- #

# ....


#####################################
# --- (X2) Find best/worst weeks --- #

# cn_sub = ['y', 'pred', 'ub', 'lb', 'dates', 'model', 'lead']
#
# # Find weeks with violations
# viol_day = df_pred.groupby(cn_ml + cn_date).apply(
#     lambda x: pd.Series({'lb': np.sum(x.lb > x.y), 'ub': np.sum(x.ub < x.y)})).reset_index()
# viol_day = viol_day.melt(cn_ml + cn_date, None, 'bound', 'viol').sort_values(['model', 'lead', 'bound', 'viol'])
# viol_dates = viol_day.groupby(['model', 'lead', 'bound']).tail(1).reset_index(None, True)
# viol_dates = viol_dates.merge(df_pred[cn_sub + cn_date], 'left', cn_ml + cn_date).drop(columns=cn_date)
# viol_dates.dates = pd.to_datetime(viol_dates.dates)
# viol_dates.insert(viol_dates.shape[1], 'doy', viol_dates.dates.dt.strftime('%b %d'))
#
# # Find weeks with the best/worst fit
# tmp1 = df_r2.sort_values(cn_ml + ['r2']).groupby(cn_ml).head(1)
# tmp2 = df_r2.sort_values(cn_ml + ['r2']).groupby(cn_ml).tail(1)
# r2_dates = pd.concat([tmp1.assign(tt='worst'), tmp2.assign(tt='best')]).reset_index(None, True)
# r2_dates = r2_dates.merge(df_pred[cn_sub + cn_date], 'left', cn_ml + cn_date).drop(columns=cn_date)
# r2_dates.dates = pd.to_datetime(r2_dates.dates)
# r2_dates.insert(r2_dates.shape[1], 'doy', r2_dates.dates.dt.strftime('%b %d'))

########################################
# --- (4) Get esclation parameters --- #

# # Subset predictions to datestart
# df_ypost = df_pred[df_pred.dates >= esc_start].reset_index(None, True).drop(columns=cn_date + ['hour', 'ub'])
# # Map labels and predicted to thresholds
# df_predpost = df_ypost.assign(y=lambda x: pd.cut(x.y, esc_bins, False, esc_lbls),
#                               pred=lambda x: pd.cut(x.pred, esc_bins, False, esc_lbls),
#                               lb=lambda x: pd.cut(x.lb, esc_bins, False, esc_lbls))
# df_predpost = df_predpost.melt(['model', 'lead', 'y', 'dates'], None, 'tt')
# # Get the confusion matrix
# dat_conf = df_predpost.groupby(cn_ml + ['y', 'tt', 'value']).count().fillna(0).astype(int).reset_index()
# dat_conf = dat_conf.rename(columns={'value': 'pred', 'dates': 'n'}).assign(
#     pred=lambda x: pd.Categorical(x.pred, esc_lbls))
# # dat_conf.pivot_table('n',['model','lead','tt','y'],'pred')[esc_lbls]

######################################
# --- (5) Escalation sensitivity --- #

# crit = norm.ppf(0.975)  # Current quantile used in interval construction
#
# df_sens = df_ypost.assign(se=lambda x: (x.pred - x.lb) / crit,
#                           ylbl=lambda x: pd.cut(x.y, esc_bins, False, esc_lbls)).drop(columns=['lb'])
#
# ### Q1: Does the actual label end up being more,less,or the same severity relative to prediction/actual
# p_seq = np.linspace(0.05, 0.50, 10)
#
# holder = []
# for p in p_seq:
#     print('Lower bound: %0.2f' % p)
#     tmp = df_sens.assign(yhat=lambda x: pd.cut(x.pred - x.se * norm.ppf(1 - p), esc_bins, False, esc_lbls)).drop(
#         columns=['y', 'pred', 'se'])
#     tmp = tmp.groupby(['model', 'lead', 'yhat', 'ylbl']).count().fillna(0).astype(int).reset_index().rename(
#         columns={'dates': 'n'})
#     # Remove when both are pre-escalation levels
#     tmp = tmp[~((tmp.ylbl == esc_lbls[0]) & (tmp.yhat == esc_lbls[0]))].reset_index(None, True)
#     tmp = tmp.groupby(cn_ml).apply(lambda x: pd.Series(
#         {'less': np.sum((x.yhat > x.ylbl) * x.n), 'more': np.sum((x.yhat < x.ylbl) * x.n),
#          'same': np.sum((x.yhat == x.ylbl) * x.n)}))
#     tmp = tmp.div(tmp.sum(1), 0).reset_index().assign(p=p)
#     holder.append(tmp)
# # same, more, less
# dat_sml_pred = pd.concat(holder).reset_index(None, True).melt(['model', 'lead', 'p'], None, 'tt')
#
# # Calculate for the actual level
# holder = []
# sub = act_y[act_y.date >= esc_start].reset_index(None, True)
# for lead in np.arange(1, mx_leads + 1):
#     tmp = pd.crosstab(index=sub.esc.values, columns=sub.esc.shift(lead).values, normalize=0)
#     tmp = pd.DataFrame(tmp.values, columns=tmp.columns.astype(str), index=tmp.index.astype(str)).reset_index().melt(
#         'row_0').rename(columns={'row_0': 't0', 'col_0': 't1'}).assign(lead=lead)
#     holder.append(tmp)
# dat_sml_act = pd.concat(holder).assign(t0=lambda x: x.t0.map(di_esc), t1=lambda x: x.t1.map(di_esc)).sort_values(
#     ['lead', 't0', 't1']).reset_index(None, True)
#
# ### Q2: How much variation is there in the t+1,..t+7 levels?
# tmp_acf = acf(x=sub.y.values, nlags=mx_leads, alpha=0.05)
# tmp_pacf = pacf(x=sub.y.values, nlags=mx_leads, alpha=0.05)
# cn_acf = ['mu', 'lb', 'ub']
# tmp_acf = pd.DataFrame(np.hstack([cvec(tmp_acf[0]), tmp_acf[1]])[1:], columns=cn_acf)
# tmp_pacf = pd.DataFrame(np.hstack([cvec(tmp_pacf[0]), tmp_pacf[1]])[1:], columns=cn_acf)
# dat_acf = pd.concat([tmp_acf.assign(tt='ACF'), tmp_pacf.assign(tt='PACF')]).rename_axis('lead').reset_index().assign(
#     lead=lambda x: x.lead + 1)
# lead_seq = range(mx_leads + 1)[1:]
# e2 = [np.nanmean((sub.y.values - sub.y.shift(lead).values) ** 2) for lead in lead_seq]
# r2 = [r2_fun(sub.y.values, sub.y.shift(lead).values) for lead in lead_seq]
# dat_acf = pd.concat([dat_acf, pd.DataFrame({'lead': lead_seq, 'mu': r2, 'tt': 'R2'})])
#
# ### Q3: If there is a predicted level change, and such a change would be different than the current actual level, what does the actual level end up being?
#
# print(act_y[act_y.doy == '2020-01-01'].drop(columns=['doy']))
# print(df_pred[df_pred.dates == pd.to_datetime('2020-01-01 02:00:00')][['lead', 'dates', 'pred', 'y']])
# # act_y[act_y.date.isin(pd.date_range('2020-03-14','2020-03-16',freq='1H'))]
#
# tmp_act = pd.DataFrame(pd.Categorical(act_y.esc).codes, index=act_y.date,
#                        columns=pd.MultiIndex.from_product([['act'], [0]])).rename_axis('dates')
# holder = []
# for p in p_seq:
#     print('Lower bound: %0.2f' % p)
#     tmp = df_sens.assign(pred=lambda x: x.pred - x.se * norm.ppf(1 - p)).drop(columns='se')
#     tmp = tmp.pivot_table(['pred', 'y'], ['model', 'dates'], 'lead').apply(
#         lambda x: pd.cut(x, esc_bins, False, esc_lbls))
#     tmp = tmp.apply(lambda x: pd.Categorical(x).codes.astype(int))
#     tmp = tmp_act.join(tmp)
#     # Get change relative to current level
#     tmp = tmp.loc[:, ['pred', 'y']].subtract(tmp.loc[:, pd.IndexSlice['act', 0]].values, axis=0)
#     tmp = tmp.reset_index().melt(['model', 'dates']).rename(columns={'variable_0': 'tt', 'variable_1': 'lead'})
#     tmp = tmp.pivot_table('value', ['model', 'dates', 'lead'], 'tt').reset_index()
#     # Performance for higher/lower escalation prediction
#     tmp_gb = tmp.assign(psign=lambda x: np.sign(x.pred), ysign=lambda x: np.sign(x.y))
#     tmp_gb = tmp_gb.pivot_table('dates', cn_ml + ['psign'], 'ysign', 'count').fillna(0).astype(int).reset_index()
#     tmp_gb = tmp_gb.melt(cn_ml + ['psign'], None, None, 'n').sort_values(cn_ml + ['psign']).reset_index(None,
#                                                                                                         True).assign(
#         p=p)
#     holder.append(tmp_gb)
# # Merge
# cn_rel = cn_ml + ['CI', 'psign']
# dat_rel = pd.concat(holder).reset_index(None, True).assign(CI=lambda x: (1 - 2 * x.p).round(2))
# dat_rel = dat_rel.merge(dat_rel.groupby(cn_rel).n.sum().reset_index().rename(columns={'n': 'tot'}))
# dat_prec = dat_rel.assign(tp=lambda x: np.where(x.psign == x.ysign, 'tp', 'fp')).groupby(
#     cn_rel + ['tp']).n.sum().reset_index()
# # WARNING! I THINK THIS NEEDS TO BE A SUM
# # dat_prec = dat_prec.pivot_table('n',cn_rel,'tp').reset_index().assign(prec = lambda x: x.tp/(x.tp+x.fp))
# dat_prec = dat_prec.pivot_table('n', cn_rel, 'tp', 'sum').reset_index().assign(prec=lambda x: x.tp / (x.tp + x.fp))
# # Get the TPR
# tmp = dat_rel.groupby(cn_ml + ['CI', 'ysign']).n.sum().reset_index().rename(columns={'ysign': 'psign'})
# tmp1 = dat_prec.drop(columns=['tp', 'fp'])
# tmp2 = dat_prec[cn_rel + ['tp']].merge(tmp).assign(sens=lambda x: x.tp / x.n).drop(columns=['tp', 'n'])
# dat_prec = tmp1.merge(tmp2).melt(cn_rel, ['prec', 'sens'], 'metric')


##########################
# --- (7) MAKE PLOTS --- #

# ### EXAMPLE OF VIOLATION WEEKS
# for bound in viol_dates.bound.unique():
#     gg_viol_ex = (ggplot(viol_dates[viol_dates.bound == bound], aes(x='dates', color='lead')) +
#                   geom_point(aes(y='y')) + geom_line(aes(y='pred')) + theme_bw() +
#                   geom_ribbon(aes(ymin='lb', ymax='ub', fill='lead'), alpha=0.5) +
#                   facet_wrap('~bound+lead+doy+model', scales='free', labeller=label_both) +
#                   ggtitle('Example of violation weeks') + guides(color=False, fill=False) +
#                   labs(x='Hour of day', y='Max # of patients') +
#                   scale_x_datetime(breaks='3 hours', date_labels='%H') +
#                   theme(subplots_adjust={'wspace': 0.15, 'hspace': 0.35}))
#     gg_viol_ex.save(os.path.join(dir_figures, 'gg_viol_ex_' + bound + '.png'), height=mx_leads, width=mx_leads)
#
# ### EXAMPLE OF GOOD/BAD R2 WEEKS
# for tt in r2_dates.tt.unique():
#     gg_r2_ex = (ggplot(r2_dates[r2_dates.tt == tt], aes(x='dates', color='lead')) + theme_bw() +
#                 geom_point(aes(y='y')) + geom_line(aes(y='pred')) +
#                 geom_ribbon(aes(ymin='lb', ymax='ub', fill='lead'), alpha=0.5) +
#                 facet_wrap('~tt+lead+doy+model', scales='free', labeller=label_both) +
#                 ggtitle('Example of best/worst performing weeks') +
#                 labs(x='Hour of day', y='Max # of patients') +
#                 scale_x_datetime(breaks='3 hours', date_labels='%H') +
#                 theme(subplots_adjust={'wspace': 0.15, 'hspace': 0.35}) +
#                 guides(color=False, fill=False))
#     gg_r2_ex.save(os.path.join(dir_figures, 'gg_r2_ex_' + tt + '.png'), height=mx_leads, width=mx_leads)
#
# ### IQR DISTRIBUTION OF ONE-DAY-AHEAD RSQUARED BY LEAD
# tmp = r2_desc.assign(month2=lambda x: x.month - (int(mx_leads / 2) - x.lead) / (2 * mx_leads))
# gg_r2_month = (ggplot(tmp, aes(x='month2', y='med', color='lead')) + theme_bw() +
#                geom_point() + geom_linerange(aes(ymin='lb', ymax='ub')) +
#                labs(x='Month (2020)', y='One-day-ahead R2') +
#                ggtitle('IQR performance on R2') +
#                facet_wrap('~model', labeller=label_both) +
#                scale_x_continuous(breaks=list(np.arange(mx_month + 1))))
# gg_r2_month.save(os.path.join(dir_figures, 'gg_r2_month.png'), height=5, width=int(mx_leads * 0.7))
#
# ### VIOLATION RATES IN A GIVEN MONTH BY LEAD
# tmp = viol_desc.assign(month2=lambda x: x.month - (int(mx_leads / 2) - x.lead) / (2 * mx_leads)).sort_values(
#     ['bound', 'month', 'lead', ])  # .head(12).drop(columns=['n','prop'])
# gg_viol_month = (ggplot(tmp, aes(x='month2', y='prop', color='lead')) + theme_bw() +
#                  geom_point() +
#                  geom_linerange(aes(ymin='lb', ymax='ub')) +
#                  labs(x='Month (2020)', y='Violation %') +
#                  ggtitle('Coverage of 95% confidence intervals by lead/model/month') +
#                  facet_grid('model~bound', labeller=label_both) +
#                  geom_hline(yintercept=0.025) +
#                  scale_x_continuous(breaks=list(np.arange(mx_month + 1))))
# gg_viol_month.save(os.path.join(dir_figures, 'gg_viol_month2.png'), height=5, width=int(mx_leads * 0.7))
#
# di_tt = {'pred': 'Point', 'lb': 'LowerBound'}
# ### CONFUSION MATRIX BY LB/PRED AND MODEL
# gg_conf = (ggplot(dat_conf, aes(y='pred', x='y', fill='np.log(n+1)')) + theme_bw() +
#            geom_tile(aes(width=1, height=1)) +
#            labs(y='Predicted', x='Actual') +
#            facet_wrap('~tt+lead+model',
#                       labeller=labeller(tt=di_tt, lead=label_both, model=label_both)) +
#            ggtitle('Confusion matrix by model/lead/bound') +
#            geom_text(aes(label='n')) + guides(fill=False))
# gg_conf.save(os.path.join(dir_figures, 'gg_conf.png'), height=mx_leads, width=mx_leads)
#
#
# ### SAME, MORE LESS, RELATIVE TO PREDICTION
# tit = 'Actual escalation level relative to prediction\nIgnores pred&act==Normal state'
# gg_sml = (ggplot(dat_sml_pred, aes(x='1-p*2', y='value', color='lead', group='lead')) + theme_bw() +
#           geom_point() + geom_line() +
#           facet_wrap('~tt+model', labeller=label_both, ncol=3) +
#           ggtitle(tit) + labs(x='± CI (lowerbound)', y='Share'))
# gg_sml.save(os.path.join(dir_figures, 'gg_sml_pred.png'), height=5, width=10)
#
#
#
# #### TRANSITION PROBABILITIES RELATIVE TO ESCALATION LEVEL
# tmp = dat_sml_act.copy().rename(columns={'t0': 'Escalation_Level'})
# tit = 'Esclation transition by lead (%s)' % esc_start.strftime('≥%Y-%m-%d')
# gg_esctrans_act = (ggplot(tmp, aes(x='lead', y='value', color='t1.astype(str)')) +
#                    theme_bw() + geom_point(size=1) + geom_line() +
#                    labs(x='Lead', y='Share') + ggtitle(tit) +
#                    facet_wrap('~Escalation_Level', labeller=label_both, ncol=2) +
#                    scale_color_discrete(name='ESc in lead') +
#                    scale_x_continuous(breaks=list(range(mx_leads + 1))))
# gg_esctrans_act.save(os.path.join(dir_figures, 'gg_esctrans_act.png'), height=8, width=10)
#
# #### (PARTIAL) AUTO CORRELATION PROPERTIES
# gg_acf = (ggplot(dat_acf, aes(x='lead', y='mu')) + theme_bw() +
#           geom_point() + labs(x='Lead', y='Correlation') +
#           scale_x_continuous(breaks=list(range(mx_leads + 1))) +
#           ggtitle('ACF/PACF for max hourly patients') + facet_wrap('~tt') +
#           geom_linerange(aes(ymin='lb', ymax='ub')) +
#           geom_hline(yintercept=0, color='blue', linetype='--'))
# gg_acf.save(os.path.join(dir_figures, 'gg_acf.png'), height=5, width=mx_leads)
#
# ### HOW OFTEN DO WE GET THE CHANGE IN THE LEVEL CORRECT?
# # Let's examine how the distribution looks
# tit = 'TPR/PPV for change in predicted escalation level'  # \nTrue pos. implies change in actual esclation ≥ change in predicted escalation
# # di_metric = {'prec': 'Precision', 'sens': 'Sensitivity'}
# tmp = dat_prec.assign(psign=lambda x: x.psign.map({-1: 'Negative change', 0: 'No change', 1: 'Positive change'}))
# gg_rel = (ggplot(tmp, aes(x='lead', y='value', color='CI', group='CI')) +
#           theme_bw() + geom_point() + geom_line() +
#           facet_grid('metric~psign+model', scales='free_y',
#                      labeller=labeller(model=label_both, metric=di_metric)) +
#           scale_x_continuous(breaks=list(range(mx_leads + 1))) +
#           labs(x='Lead', y='Precision/TPs') +
#           scale_color_gradient(name='CI', limits=[0, 1.01]) +
#           ggtitle(tit))
# gg_rel.save(os.path.join(dir_figures, 'gg_rel.png'), height=8, width=int(mx_leads * 0.75))
