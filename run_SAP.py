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
from plydata.cat_tools import *

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

RUN_BOOT = False

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
    if RUN_BOOT:  # Run if set
        for jj in range(n_bs):
            idx_jj = resample(df_slbls.index, stratify=df_slbls.month, random_state=jj)
            tmp_slbls = df_slbls.iloc[idx_jj].reset_index(None, True)
            tmp_slbls = sens_spec_df(df=tmp_slbls, gg=['month']).query('pred==1').assign(level=level, sim=jj).drop(columns='pred')
            holder_bs.append(tmp_slbls)

# Combine and plot
pr_bl = pd.concat(holder_bl).reset_index(None,True)
if RUN_BOOT:
    pr_bl_bs = pd.concat(holder_bs).reset_index(None,True)
    pr_bl_bs.to_csv(os.path.join(dir_flow, 'pr_bl_bs.csv'), index=False)
else:
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
gg_pr_SAP = (ggplot(pr_bl.assign(month=lambda x: pd.Categorical(x.month)),
                    aes(x='sens',y='prec',color='month',group='month')) +
             theme_bw() + geom_line() + labs(x='Recall', y='Precision') +
             ggtitle('Empirical PR Curves') +
             scale_x_continuous(limits=[0,1],breaks=list(np.arange(0,1.01,0.2))) +
             scale_y_continuous(limits=[0.8, 1], breaks=list(np.arange(0.8,1.01, 0.05))) +
             scale_color_manual(name='Month',values=colz))
gg_pr_SAP.save(os.path.join(dir_figures, 'gg_pr_SAP.png'),height=4,width=5)
gg_pr_bs = (gg_pr_SAP + facet_wrap('~month',labeller=label_both) + guides(color=False) +
            geom_line(color='black',size=2) +
            geom_line(aes(x='sens', y='prec', color='month', group='gg'),
                      alpha=0.1, size=0.5, inherit_aes=False,
                      data=pr_bl_bs.assign(month=lambda x: pd.Categorical(x.month))) +
            ggtitle('Distribution around PR curves by bootstrap'))
gg_pr_bs.save(os.path.join(dir_figures, 'gg_pr_bs.png'),height=6,width=7)

####################################
# --- (3) TARGET 90% PRECISION --- #

RUN_PREC = False

# !!! SET THE PRECISION TARGET !!! #
prec_target = 0.9
type1 = 0.2
num_bs = 1000

res_prec = res.query('lead == @lead_star').assign(month=lambda x: x.date_rt.dt.month).drop(columns='lead').reset_index(None,True)
# Get the training months
month_seq = np.array([5, 6, 7, 8])

if RUN_PREC:
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
else:
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
dat_month_test = res_prec.query('month >= @month_test').reset_index(None,True)

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
             ggtitle('PR on 2020 test months\nBootstrapped samples in blue') +
             facet_wrap('~month',labeller=label_both) + scale_y_continuous(limits=[0.9,1.0]))
gg_pr_sim.save(os.path.join(dir_figures, 'gg_pr_sim.png'),height=4,width=8)

# Calculate contigency tables on test months
res_test = res.assign(month=lambda x: x.date_rt.dt.month).query('month >= @month_test').reset_index(None, True).drop(columns='hour')
yhat_test = ordinal_lbls(res_test.copy(), level=level_cons)
yhat_test = yhat_test.groupby(['lead','month','pred','y']).size().reset_index().rename(columns={0:'n'}).query('pred>=0 & y>=0')
yhat_test = yhat_test.pivot_table('n',['lead','month'],['pred','y']).fillna(0).astype(int).reset_index().melt(['lead','month'])
# Make plot (use four leads as as example)
gg_conmat = (ggplot(yhat_test.query('lead>3&lead<8'), aes(y='pred',x='y')) +
             theme_bw() + geom_tile(fill='grey',color='red') + geom_text(aes(label='value')) +
             facet_grid('lead~month',labeller=label_both) +
             labs(y='Predicted Δ esclation',x='Actual Δ esclation') +
             ggtitle('Confusion matrix for Δ>0 in esclation (leads 4-7)'))
gg_conmat.save(os.path.join(dir_figures, 'gg_conmat.png'),height=12,width=6)


################################
# --- (4) GENERATE FIGURES --- #

# --- (i) How close are the predicted probabilities from the GP to the actual level changes? --- #
# Subset to test months
dat_fig = res.assign(month=lambda x: x.date_rt.dt.strftime('%m').astype(int)).query('month>=@month_test')
dat_fig.reset_index(None,True,True)
# Test to see for calibration
ymi, ymx = dat_fig.hat_pred.min()-1, dat_fig.hat_pred.max()-1
esc_bins = [ymi, 31, 38, 48, ymx]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
dat_fig = dat_fig.assign(p1=lambda x: norm(loc=x.hat_pred, scale=x.se).cdf(31),
                   p2=lambda x: norm(loc=x.hat_pred, scale=x.se).cdf(38),
                   p3=lambda x: norm(loc=x.hat_pred, scale=x.se).cdf(48),
                   y_pred_ord=lambda x: pd.Categorical(pd.cut(x.y_pred, esc_bins, False, esc_lbls)).codes)
# Calculate the CDF between
dat_fig = dat_fig.assign(p1=lambda x: x.p2 - x.p1, p2=lambda x: x.p3-x.p2, p3=lambda x: 1-x.p3)

gg = ['lead','month']
tmp = dat_fig.melt(gg+['date_pred'],['p1','p2','p3'],'tmp')
tmp = tmp.assign(g_ord=lambda x: x.tmp.str.replace('p','').astype(int)).drop(columns='tmp')
tmp = tmp.groupby(gg+['g_ord']).apply(lambda x: pd.Series({'e_nlbl':x.value.sum()})).reset_index()
tmp2 = dat_fig.groupby(gg+['y_pred_ord']).size().reset_index().rename(columns={0:'a_nlbl','y_pred_ord':'g_ord'})
tmp3 = tmp.merge(tmp2,'left',gg+['g_ord']).melt(gg+['g_ord'],None,'tt').assign(lbl=lambda x: x.tt+'_'+x.g_ord.astype(str))
tmp3 = tmp3.assign(lbl=lambda x: pd.Categorical(x.lbl,sum([[l+'_'+str(g) for l in tmp3.tt.unique()] for g in tmp3.g_ord.unique()],[])))
lblz = pd.Categorical(tmp3.lbl).categories.str.replace('e_','Expected ').str.replace('a_','Actual ')
lblz = lblz.str.replace('nlbl_',' esc. lvl-')
colz = list(np.repeat(gg_color_hue(3),2))
shpz = list(np.tile(['$E$', '$A$'],3))
# Plot it
gg_gp_cal = (ggplot(tmp3, aes(x='lead',y='value',color='lbl',shape='lbl')) +
             theme_bw() + geom_point(size=2) +
             labs(x='Expected number',y='Actual number') +
             ggtitle('Escalation levels by GP distribution\nA==Actual, E==Expected') +
             scale_color_manual(values=colz,labels=lblz) +
             scale_shape_manual(values=shpz) + guides(shape=False) +
             facet_wrap('~month',scales='free_y',labeller=label_both) +
             theme(subplots_adjust={'wspace': 0.15}))
gg_gp_cal.save(os.path.join(dir_figures,'gg_gp_cal.png'), width=10,height=5)

# Repeat analysis but for the CHANGE in escalation levels
if 'p1' in list(dat_fig.columns):
    dat_fig = dat_fig.drop(columns = ['p1','p2','p3','y_pred_ord'])
dat_fig = dat_fig.assign(lvl_rt = lambda x: pd.Categorical(pd.cut(x.y_rt, esc_bins, False, esc_lbls)).codes,
               lvl_pred = lambda x: pd.Categorical(pd.cut(x.y_pred, esc_bins, False, esc_lbls)).codes,
               lvl_hat = lambda x: pd.Categorical(pd.cut(x.hat_pred, esc_bins, False, esc_lbls)).codes)
holder = np.zeros([dat_fig.shape[0],3])
for ii in range(dat_fig.shape[0]):
    if (ii+1) % 1000 == 0:
        print('%i of %i' % (ii+1, dat_fig.shape[0]))
    lvl_rt, mu, se = dat_fig.loc[ii,['lvl_rt','hat_pred','se']]
    dist = norm(loc=mu,scale=se)
    if lvl_rt == 0:
        pct1 = dist.cdf(38) - dist.cdf(31)
        pct2 = dist.cdf(48) - dist.cdf(38)
        pct3 = 1 - dist.cdf(48)
    elif lvl_rt == 1:
        pct1 = dist.cdf(48) - dist.cdf(38)
        pct2 = 1 - dist.cdf(48)
        pct3 = 0
    elif lvl_rt == 2:
        pct1 = 1 - dist.cdf(48)
        pct2, pct3 = 0, 0
    elif lvl_rt == 3:
        pct1, pct2, pct3 = 0, 0, 0
    else:
        assert False
    holder[ii] = [pct1, pct2, pct3]
tmp = pd.DataFrame(holder,columns=['d1','d2','d3'])
tmp = pd.concat([dat_fig, tmp],axis=1).assign(dlvl=lambda x: x.lvl_pred-x.lvl_rt)
gg = ['lead', 'month', 'lvl_rt']
tmp2 = tmp.melt(['date_rt']+gg,['d1','d2','d3'],'tt').groupby(gg+['tt']).value.sum().reset_index()
tmp2 = tmp2.assign(dlvl=lambda x: x.tt.str.replace('d','').astype(int)).rename(columns={'value':'e_nlbl'})
tmp3 = tmp.pivot_table('date_rt',gg,'dlvl',len).fillna(0).astype(int).reset_index().melt(gg)
tmp4 =  tmp2.merge(tmp3,'left',gg+['dlvl']).rename(columns={'value':'a_nlbl'})
tmp4 = tmp4.melt(gg+['dlvl'],['a_nlbl','e_nlbl'],'tt').assign(lbl=lambda x: x.tt+'_'+x.dlvl.astype(str))
tmp4 = tmp4.assign(lbl=lambda x: pd.Categorical(x.lbl,sum([[l+'_'+str(g) for l in x.tt.unique()] for g in x.dlvl.unique()],[])))
tmp4 = tmp4.query('lvl_rt <= 2').reset_index(None,True).rename(columns={'lvl_rt':'Esc_Level='})

# Plot it
gg_gp_cal_d = (ggplot(tmp4, aes(x='lead',y='value',color='lbl',shape='lbl')) +
             theme_bw() + geom_point(size=2) + geom_line() +
             labs(x='Expected number',y='Actual number') +
             ggtitle('Change in escalation levels by GP distribution\nA==Actual, E==Expected') +
             scale_color_manual(values=colz,labels=lblz) +
             scale_shape_manual(values=shpz) + guides(shape=False) +
             facet_grid('Esc_Level=~month',scales='free_y',labeller=label_both) +
             theme(subplots_adjust={'wspace': 0.15}))
gg_gp_cal_d.save(os.path.join(dir_figures,'gg_gp_cal_d.png'), width=10, height=8)


# (i) Pick a "good" hour in October (R2 vs precision?)
dat_path = dat_fig.melt(['lead','date_rt','lvl_rt'],['lvl_pred','lvl_hat'],'tt')
dat_path = dat_path.assign(dlvl=lambda x: x.value - x.lvl_rt)
dat_path = dat_path.pivot_table('dlvl',['lead','date_rt'],'tt').reset_index().assign(acc=lambda x: x.lvl_hat==x.lvl_pred) #ymd=lambda x: x.date_rt.dt.strftime('%Y-%m-%d'),
dat_path = dat_path.query('lvl_hat > 0').groupby(['date_rt']).acc.sum().reset_index().sort_values(['acc','date_rt'],ascending=[False,False]).reset_index(None,True)
ymd_best = dat_path.loc[0,'date_rt']
ymd_lb = ymd_best-pd.DateOffset(hours=1)
print('The candidate day is : %s' % ymd_best)

# (ii) Show the trajectory and uncertainty (focus on lowerbound)

dat_ymd = dat_fig.query('date_rt == @ymd_best')
tmp = dat_fig.query('date_rt == @ymd_lb & lead==1').assign(se=0.1,hat_pred=lambda x: x.y_pred)
dat_ymd = pd.concat([tmp, dat_ymd],0).reset_index(None,True)
print(dat_ymd.loc[0])
dat_ymd = dat_ymd.drop(columns=['month','hour','lvl_rt','lvl_pred','lvl_hat'])
dat_ymd = dat_ymd.assign(lb=lambda x: norm(loc=x.hat_pred, scale=x.se).ppf(1-level_cons),
                          ub=lambda x: norm(loc=x.hat_pred, scale=x.se).ppf(level_cons))

dat_hline = pd.DataFrame({'yint':[31, 38, 48]})
tit1 = 'Example of 24H trajectory made at %s' % ymd_best
tit2 = 'Red area shows %i%% CI' % (100*(2*level_cons-1))
title = tit1 + '\n' + tit2
gg_ymd_ribbon = (ggplot(dat_ymd, aes(x='date_pred',y='hat_pred')) + theme_bw() +
                 geom_point(data=tmp) +
                 geom_ribbon(aes(ymin='lb',ymax='ub'),fill='red',alpha=0.25,color='black') +
                 ggtitle(title) + labs(y='Predicted Census') +
                 theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) +
                 scale_x_datetime(date_breaks='1 hour',date_labels='%I%p') +
                 geom_hline(aes(yintercept='yint'),data=dat_hline,linetype='--'))
gg_ymd_ribbon.save(os.path.join(dir_figures,'gg_ymd_ribbon.png'), width=7, height=5)
# geom_line() + geom_point() +

# (iii) Make calculations for escalation levels
assert np.all(dat_ymd.y_rt < 31)
holder = np.zeros([dat_ymd.shape[0], 3])
for ii in range(dat_ymd.shape[0]):
    mu, se = dat_ymd.loc[ii, ['hat_pred', 'se']]
    dist = norm(loc=mu, scale=se)
    pct1 = dist.cdf(38) - dist.cdf(31)
    pct2 = dist.cdf(48) - dist.cdf(38)
    pct3 = 1 - dist.cdf(48)
    holder[ii] = [pct1, pct2, pct3]
dat_escp = pd.concat([pd.DataFrame(np.round(holder*100),columns=['p1','p2','p3']),
                      dat_ymd[['date_pred','hat_pred','se']]],1).iloc[1:]
dat_escp = dat_escp.melt(['date_pred','hat_pred','se'],None,'tt').assign(yval=lambda x: x.tt.map({'p1':35, 'p2':42, 'p3':52})).assign(lbl=lambda x: x.value.astype(int).astype(str)+'%').query('value>1')

colz = list(np.unique(colz))

gg_escp = gg_ymd_ribbon
for ii, cc in enumerate(colz):
    tt = "p"+str(ii + 1)
    gg_escp = (gg_escp + geom_text(aes(x='date_pred',y='yval',label='lbl'), data=dat_escp.query('tt==@tt'),size=8,color=cc,inherit_aes=False))
gg_escp.save(os.path.join(dir_figures,'gg_escp.png'), width=12, height=5)

# (iv) Autocorrelation of errors
dat_ac = dat_fig.assign(err=lambda x: x.y_pred - x.hat_pred)[['lead','date_rt','err']]
for jj in range(12):
    dat_ac.insert(dat_ac.shape[1],'err_'+str(jj+1),dat_ac.groupby('lead').err.shift(jj+1))
dat_ac = dat_ac[~dat_ac.isnull().any(1)].reset_index(None,True)

from statsmodels.regression.linear_model import OLS


cn = dat_ac.columns[dat_ac.columns.str.contains('err_')].to_list()
ll_seq = range(24)
r2_seq = np.zeros([len(ll_seq),3])
n_bs = 250
np.random.seed(1234)
for ll in ll_seq:
    print(ll)
    tmp = dat_ac.query('lead==@ll+1').reset_index(None,True)
    y, X = tmp.err.values, tmp[cn].values
    mdl = OLS(endog=y,exog=X).fit()
    n = X.shape[0]
    holder = np.zeros(n_bs)
    for ii in range(n_bs):
        idx = np.random.choice(range(n),n,replace=True)
        r2_bs = OLS(endog=y[idx], exog=X[idx]).fit().rsquared
        holder[ii] = r2_bs
    r2_seq[ll] = np.concatenate(([mdl.rsquared],np.quantile(holder, [0.025, 0.975])))
r2_seq = pd.DataFrame(r2_seq,columns=['r2','lb','ub'])


# Check the sign relationship
tmp = dat_ac.melt(['lead','date_rt','err'],None,'tt')
tmp[['err','value']] = np.sign(tmp[['err','value']]).astype(int)
tmp = tmp.groupby(['lead','tt']).apply(lambda x: pd.Series({'p':np.mean(x.err == x.value)})).reset_index()
tmp.tt = tmp.tt.str.replace('err_','').astype(int)

gg_sign_prob = (ggplot(tmp, aes(x='lead',y='p',color='tt')) + theme_bw() + geom_jitter(random_state=1) + labs(x='Forecasting lead',y='Sign equivalence',color='tt') + ggtitle('Autocorrelation of error signs') + geom_hline(yintercept=0.5,linetype='--') + scale_color_gradient(name='Lag of residual') + scale_y_continuous(limits=[0.45,0.6],breaks=list(np.arange(0.45,0.61,0.05))))
gg_sign_prob.save(os.path.join(dir_figures,'gg_sign_prob.png'), width=7, height=5)

#################################
# --- (5) POWER CALCULATION --- #

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