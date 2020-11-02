"""
SCRIPT TO LEAD THE PROSPECTIVE DATASET TO PREPARE FOR STATISTICAL ANALYSIS PLAN (SAP)
"""

import os
import pandas as pd
import numpy as np
from funs_support import ymdh2date, ymd2date, gg_color_hue, ordinal_lbls, sens_spec_df, find_prec, parallel_find_prec, ret_prec
from plotnine import *
from sklearn.utils import resample
from time import time
from scipy import stats
from scipy.stats import norm
# from sklearn.metrics import r2_score as r2

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_figures = os.path.join(dir_base, '..', 'figures')

cn_ymd = ['year', 'month', 'day']
cn_ymdh = cn_ymd + ['hour']

#########################
# --- (1) LOAD DATA --- #

dir_prosp = os.path.join(dir_test, 'prospective')
assert os.path.exists(dir_prosp)
fn_prosp = os.listdir(dir_prosp)
assert len(fn_prosp) == 24  # 24 leads
holder = []
for fn in fn_prosp:
    tmp = pd.read_csv(os.path.join(dir_prosp, fn)).drop(columns = ['model','groups','ntrain'])
    holder.append(tmp)
res = pd.concat(holder).sort_values(['lead','date','hour']).reset_index(None,True)
res.date = pd.to_datetime(res.date + ' ' + res.hour.astype(str) + ':00:00')
# Get date bounds
dmin, dmax = res.date.min(), res.date.max()
print('Minimum date: %s, maximum date: %s' % (dmin, dmax))

# Extract the y label for the model
tmp = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), nrows=2, usecols=range(24 + 1 + 4), header=None)
assert np.all(tmp.iloc[0].fillna('y') == 'y')
act_y = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), skiprows=2, usecols=range(1 + 4))
act_y.columns = np.where(act_y.columns.str.contains('Unnamed'), '_'.join(tmp.iloc[:, 4]), act_y.columns)
assert act_y.columns[-1] == 'y_lead_0'
act_y = act_y.rename(columns={'y_lead_0': 'y'}).assign(date=ymdh2date(act_y), doy=ymd2date(act_y))
act_y.drop(columns=cn_ymdh, inplace=True)
act_y = act_y.query('date >= @dmin').reset_index(None,True).assign(doy2=lambda x: x.date.dt.dayofyear)

##################################
# --- (2) GET ORDINAL LABELS --- #

di_acty = {'y': 'y_rt', 'date': 'date_rt'}
di_res = {'date': 'date_rt', 'y': 'y_pred', 'pred': 'hat_pred'}
res = res.rename(columns=di_res).merge(act_y.rename(columns=di_acty)[list(di_acty.values())],'left','date_rt')
res['date_pred'] = res.date_rt + pd.to_timedelta(res.lead, unit='H')

# Point estimate
df_ylbls = ordinal_lbls(df=res.copy(), level=0.5)
# Find the lead with the most label changes
dat_dlbls = df_ylbls.assign(y=lambda x: np.sign(x.y)).groupby(['lead','y']).size().reset_index().rename(columns={0:'n'})
lead_star = dat_dlbls.query('y==1').sort_values('n',ascending=False).head(1).lead.values[0]
print('A lead of %i has the most number of positive label changes' % lead_star)
cn_sign = ['pred', 'y']
# Calculate the precision recall curves and their bootstapped versions for the last four months
level_seq = np.round(np.arange(0.5, 1, 0.01), 2)
res_sub = res.query('lead == @lead_star').assign(month=lambda x: x.date_rt.dt.month)
res_sub = res_sub.query('month>=month.max()-3').reset_index(None, True)
holder_bl, holder_bs = [], []
n_bs = 250
for ii, level in enumerate(level_seq):
    print('Level %i of %i' % (ii+1, len(level_seq)))
    df_slbls = ordinal_lbls(df=res_sub.copy(), level=level)
    df_slbls[cn_sign] = np.sign(df_slbls[cn_sign])
    tmp_bl = sens_spec_df(df=df_slbls.copy(), gg=['month']).query('pred==1').assign(level=level).drop(columns='pred')
    holder_bl.append(tmp_bl)
    # # Run bootstrap....
    # for jj in range(n_bs):
    #     idx_jj = resample(df_slbls.index, stratify=df_slbls.month, random_state=jj)
    #     tmp_slbls = df_slbls.iloc[idx_jj].reset_index(None, True)
    #     tmp_slbls = sens_spec_df(df=tmp_slbls, gg=['month']).query('pred==1').assign(level=level, sim=jj).drop(columns='pred')
    #     holder_bs.append(tmp_slbls)

# Combine and plot
pr_bl = pd.concat(holder_bl).reset_index(None,True)
# pr_bl_bs = pd.concat(holder_bs).reset_index(None,True)
# pr_bl_bs.to_csv(os.path.join(dir_flow, 'pr_bl_bs.csv'), index=False)
pr_bl_bs = pd.read_csv(os.path.join(dir_flow, 'pr_bl_bs.csv'))
# Wide-cast
pr_bl = pr_bl.pivot_table('value',['month','level'],'metric').reset_index()
pr_bl_bs = pr_bl_bs.pivot_table('value',['month','level','sim'],'metric').reset_index()
pr_bl_bs = pr_bl_bs.assign(gg = lambda x: x.month.astype(str) + '-' + x.sim.astype(str))

# Make plot showing the number of positive events by month
dat_nlbls = df_ylbls.assign(month=lambda x: x.date_rt.dt.month).query('month>3 & y>0').groupby(['lead','month','y']).size()
dat_nlbls = dat_nlbls.reset_index().rename(columns={0:'n'})
dat_nlbls2 = dat_nlbls.groupby(['lead','month']).n.sum().reset_index().rename(columns={'n':'tot'})

gg_nlbls = (ggplot(dat_nlbls, aes(x='lead',y='n',fill='y.astype(str)')) + theme_bw() +
            geom_bar(color='black',stat='identity') + labs(x='Forecasting lead',y='Count') +
            facet_wrap('~month',labeller=label_both) +
            ggtitle('Number of labels by lead') +
            scale_fill_discrete(name='Δ level'))
gg_nlbls.save(os.path.join(dir_figures, 'gg_nlbls.png'),height=6,width=7)


colz = gg_color_hue(4)
# Make ggplot
gg_pr_SAP = (ggplot(pr_bl, aes(x='sens',y='prec',color='month.astype(str)',group='month.astype(str)')) +
             theme_bw() + geom_line() + labs(x='Recall', y='Precision') +
             ggtitle('Empirical PR Curves') +
             scale_x_continuous(limits=[0,1],breaks=list(np.arange(0,1.01,0.2))) +
             scale_y_continuous(limits=[0.8, 1], breaks=list(np.arange(0.8,1.01, 0.05))) +
             scale_color_manual(name='Month',values=colz))
gg_pr_SAP.save(os.path.join(dir_figures, 'gg_pr_SAP.png'),height=4,width=5)
gg_pr_bs = (gg_pr_SAP + facet_wrap('~month',labeller=label_both) + guides(color=False) +
            geom_line(color='black',size=2) +
            geom_line(aes(x='sens', y='prec', color='month.astype(str)', group='gg'),
                      alpha=0.1, size=0.5, data=pr_bl_bs, inherit_aes=False) +
            ggtitle('Distribution around PR curves by bootstrap'))
gg_pr_bs.save(os.path.join(dir_figures, 'gg_pr_bs.png'),height=6,width=7)

####################################
# --- (3) TARGET 90% PRECISION --- #

# !!! SET THE PRECISION TARGET !!! #
prec_target = 0.9
type1 = 0.2
num_bs = 1000

res_prec = res.query('lead == @lead_star').assign(month=lambda x: x.date_rt.dt.month).drop(columns='lead').reset_index(None,True)
# Get the training months
month_seq = np.array([5, 6, 7, 8])
holder_month = []
for month in month_seq:
    print('Establishing threshold on month: %i' % month)
    res_train = res_prec.query('month == @month').reset_index(None, True).drop(columns='hour')
    level_full = find_prec(res_train.copy(), prec_target, ['month'])
    # Great bootstrapped samples of res_train
    holder_bs = []
    for jj in range(num_bs):
        idx_jj = resample(res_train.index, random_state=jj)
        holder_bs.append(res_train.iloc[idx_jj].assign(bs=jj))
    res_train_bs = pd.concat(holder_bs).reset_index(None, True)
    stime = time()
    thresh_bs = parallel_find_prec(data=res_train_bs, gg=['bs','month'], target=prec_target)
    print('Took %0.1f seconds' % (time() - stime))
    thresh_bs = thresh_bs.rename(columns={0:'bs',1:'month'}).assign(full=level_full)
    holder_month.append(thresh_bs)
# Combine and save
dat_level_sim = pd.concat(holder_month)
dat_level_sim.to_csv(os.path.join(dir_flow, 'dat_level_sim.csv'), index=False)
dat_level_sim = pd.read_csv(os.path.join(dir_flow, 'dat_level_sim.csv'))

gg_level_sim = (ggplot(dat_level_sim, aes(x='1-level')) + theme_bw() +
                geom_histogram(bins=30,fill='grey',color='red') +
                facet_wrap('~month',labeller=label_both) +
                geom_vline(aes(xintercept='1-full'),data=dat_level_sim.groupby('month').full.mean().reset_index()) +
                labs(x='Percentile',y='Frequency') +
                ggtitle('Percentiles needed for 90% precision (bootstrap)\nVertical lines show point estimate'))
gg_level_sim.save(os.path.join(dir_figures, 'gg_level_sim.png'),height=7,width=9)

# Conservative estimate
month_test = max(month_seq) + 1
level_cons = dat_level_sim.groupby('month').apply(lambda x: np.quantile(x.level,type1)).max()
dat_month_test = res_prec.query('month >= @month_test')

# Simulate and compare
pr_month_test = ret_prec(level_cons, dat_month_test, ['month'], True).query('pred==1').reset_index(None, True)
nsim = 250
holder_sim = []
for ii in range(nsim):
    tmp = ret_prec(level_cons, dat_month_test.sample(frac=1,replace=True,random_state=ii), ['month'], True).query('pred==1')
    holder_sim.append(tmp.assign(sim=ii))
dat_pr_sim = pd.concat(holder_sim).reset_index(None, True)

tmp1 = dat_pr_sim.pivot_table('value',['sim','month'],'metric').reset_index()
tmp2 = pr_month_test.pivot('month','metric','value').reset_index()
gg_pr_sim = (ggplot(tmp1, aes(x='sens',y='prec')) + theme_bw() +
             geom_point(color='blue',alpha=0.5,size=0.5) + labs(x='Recall',y='Precision') +
             geom_point(aes(x='sens',y='prec'),data=tmp2,color='black',size=2) +
             ggtitle('PR on September 2020\nBootstrapped samples in blue') +
             facet_wrap('~month',labeller=label_both) + scale_y_continuous(limits=[0.9,1.0]))
gg_pr_sim.save(os.path.join(dir_figures, 'gg_pr_sim.png'),height=4,width=8)


#################################
# --- (4) POWER CALCULATION --- #

# Number of hours in a month
nhours = 24*30
# Get a lower bound on the predictive positive rate
npos = dat_pr_sim.query('metric=="prec"').den.quantile(type1)
ratio = npos / nhours  # Divide number of observations by this

def beta_fun(n, pt, pm, alpha):
    ta = stats.norm.ppf(1-alpha)
    sigma_t = np.sqrt(pt*(1-pt)/n)
    sigma_m = np.sqrt(pm*(1-pm)/n)
    Phi = stats.norm.cdf( (sigma_t*ta-(pm-pt))/sigma_m )
    return Phi

prec_target

target_seq = np.round(np.arange(0.75,0.851,0.05),2)
n_seq = np.arange(2,201,2)
tmp = pd.concat([pd.concat([pd.Series({'n':n,'trial':t,'beta':beta_fun(n,t,prec_target,0.05)}) for t in target_seq]) for n in n_seq]).reset_index()
dat_power = tmp.assign(qq=tmp.groupby('index').cumcount()).pivot('qq','index',0).reset_index().assign(power=lambda x: 1-x.beta, n=lambda x: x.n.astype(int), n2=lambda x: x.n/ratio/nhours)

gg_power = (ggplot(dat_power, aes(x='n2',y='power',color='trial.astype(str)',group='trial.astype(str)')) + theme_bw() + geom_line() + geom_hline(yintercept=0.8)  + scale_color_discrete(name='Trial precision') + ggtitle('Power calculations for 90% model precision') + labs(x='# of months', y='Power'))
gg_power.save(os.path.join(dir_figures, 'gg_power.png'),height=4,width=6)


# dat_level_sim.groupby('month').apply(lambda x: norm.ppf(np.mean(x.level < x.full)))
# for month in month_seq+1:
#     print('Making prediction on month: %i' % month)
#     res_test = res_prec.query('month == @month').reset_index(None, True).drop(columns='hour')
#     # Establish the threshold
#     level_full = dat_level_sim.query('month==@month-1').full.min()
#     prec_full = ret_prec(level_full, res_test.copy(), ['month'])
#     level_bs = dat_level_sim.query('month==@month-1').level.values
#     # The Jackknife is basically going to be zero, so we can set the acceleration parameter to 0
#     ahat = 0
#     num = np.sum((level_bs.mean() - level_bs) ** 3)
#     den = 6 * np.sum((level_bs.mean() - level_bs) ** 2) ** 1.5
#     ahat = num/den
#     zhat = norm.ppf(np.mean(level_bs < level_full))
#     ql, qu = norm.ppf(type1), norm.ppf(1-type1)
#     a1 = norm.cdf(zhat + (zhat + ql) / (1 - ahat * (zhat + ql)))
#     a2 = norm.cdf(zhat + (zhat + qu) / (1 - ahat * (zhat + qu)))
#     level_bca = np.quantile(level_bs, a2)
#     ret_prec(level_bca, res_test.sample(frac=1,replace=True), ['month'])











# for jj in range(res_train.shape[0]):
#     holder_jk.append(res_train.drop(jj).assign(jk=jj))
# res_train_jk = pd.concat(holder_jk).reset_index(None, True)
# stime = time()
# thresh_jk = parallel_find_prec(data=res_train_jk, gg=['jk','month'], target=prec_target)
# print('Took %0.1f seconds' % (time() - stime))