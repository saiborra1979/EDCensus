"""
SCRIPT TO COMPARE MODEL PERFORMANCE TO THE ESLCATION LEVELS
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nleads', type=int, default=25, help='Number of leads from process_flow.py')
args = parser.parse_args()
print(args)
nleads = args.nleads

# # For beta testing
# nleads = 25

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score as r2
from statsmodels.stats.proportion import proportion_confint as propCI
from scipy.stats import norm
from funs_support import cvec, rho, r2_fun
from statsmodels.tsa.stattools import acf, pacf

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_figures = os.path.join(dir_base, '..', 'figures')

models = ['gpy']
cn_ml = ['model', 'lead']
holder = []
for model in models:
    path = os.path.join(dir_flow,'res_'+model+'.csv')
    holder.append(pd.read_csv(path))
df_pred = pd.concat(holder).reset_index(None,True)

# Check everyone has same horizon
assert df_pred.groupby(cn_ml).size().unique().shape[0] == 1
mx_month = df_pred.month.unique().max()
mx_leads = df_pred.lead.unique().shape[0]
mi_leads = df_pred.lead.unique().min()

# Remove any non-full days
qq = df_pred.groupby(['model','lead','year','month','day']).size()
df_pred = qq[qq == 24].reset_index().drop(columns=[0]).merge(df_pred)
# Make datetime
df_pred.dates = pd.to_datetime(df_pred.dates)

# Extract the y label for the model
tmp = pd.read_csv(os.path.join(dir_flow,'df_lead_lags.csv'),nrows=2,usecols=range(nleads+4),header=None)
assert np.all(tmp.iloc[0].fillna('y') == 'y')
act_y = pd.read_csv(os.path.join(dir_flow,'df_lead_lags.csv'),skiprows=2,usecols=range(1+4))
act_y.columns = np.where(act_y.columns.str.contains('Unnamed'),'_'.join(tmp.iloc[:,4]),act_y.columns)
assert act_y.columns[-1] == 'y_lead_0'
act_y.rename(columns={'y_lead_0':'y'}, inplace=True)
act_y = act_y.assign(date=lambda x: pd.to_datetime(x.year.astype(str)+'-'+x.month.astype(str)+'-'+x.day.astype(str)+' '+x.hour.astype(str)+':00:00'))

# qq=df_pred.groupby(cn_ml).head(10)[['lead','dates','y']].reset_index(None,True)
# qq['date2'] = qq.apply(lambda x: x['dates']+pd.offsets.Hour(x['lead']),1)
# qq.pivot('date2','lead','y')

#########################################
# --- (1) Get aggregate performance --- #

di_desc = {'25%':'lb','50%':'med','75%':'ub'}
cn_gg = ['model','lead','month']
cn_date = ['year','month','day']

# Range of R2 performance by horizon/month
df_r2 = df_pred.groupby(cn_gg+['year','day']).apply(lambda x: r2(x.y, x.pred)).reset_index().rename(columns={0:'r2'})
r2_desc = df_r2.groupby(cn_gg).r2.describe()
r2_desc = r2_desc[list(di_desc)].rename(columns=di_desc).reset_index()

# Violation rate by horizon/month
viol_desc = df_pred.groupby(cn_gg).apply(lambda x: pd.Series({'lb':np.sum(x.lb > x.y),'ub':np.sum(x.ub < x.y),'n':len(x.y)})).reset_index().melt(cn_gg+['n'],None,'bound').assign(prop=lambda x: x.value/x.n)
tmp = pd.concat(propCI(count=viol_desc.value, nobs=viol_desc.n, alpha=0.05, method='beta'),1).rename(columns={0:'lb',1:'ub'})
viol_desc = pd.concat([viol_desc, tmp],1)

#####################################
# --- (3) Find best/worst weeks --- #

cn_sub = ['y','pred','ub','lb','dates','model','lead']

# Find weeks with violations
viol_day = df_pred.groupby(cn_ml+cn_date).apply(lambda x: pd.Series({'lb':np.sum(x.lb > x.y),'ub':np.sum(x.ub < x.y)})).reset_index()
viol_day = viol_day.melt(cn_ml+cn_date,None,'bound','viol').sort_values(['model','lead','bound','viol'])
viol_dates = viol_day.groupby(['model','lead','bound']).tail(1).reset_index(None,True)
viol_dates = viol_dates.merge(df_pred[cn_sub+cn_date],'left',cn_ml+cn_date).drop(columns=cn_date)
viol_dates.dates = pd.to_datetime(viol_dates.dates)
viol_dates.insert(viol_dates.shape[1],'doy',viol_dates.dates.dt.strftime('%b %d'))

# Find weeks with the best/worst fit
tmp1 = df_r2.sort_values(cn_ml+['r2']).groupby(cn_ml).head(1)
tmp2 = df_r2.sort_values(cn_ml+['r2']).groupby(cn_ml).tail(1)
r2_dates = pd.concat([tmp1.assign(tt='worst'),tmp2.assign(tt='best')]).reset_index(None,True)
r2_dates = r2_dates.merge(df_pred[cn_sub+cn_date],'left',cn_ml+cn_date).drop(columns=cn_date)
r2_dates.dates = pd.to_datetime(r2_dates.dates)
r2_dates.insert(r2_dates.shape[1],'doy',r2_dates.dates.dt.strftime('%b %d'))

########################################
# --- (4) Get esclation parameters --- #

# Levels: pre-esclation (<= 30), escalation level 1 (31-37), escalation level 2 (38-47), escalation level 3 (>=48)
ymx = df_pred.y.max()
esc_bins = [df_pred.lb.min()-1,31,38,48,ymx+2]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
esc_start = pd.to_datetime('2020-03-15')
di_esc = dict(zip(esc_lbls,range(len(esc_lbls))))
# Add on true labels to ground truth
act_y = act_y.assign(esc=lambda x: pd.cut(x.y,esc_bins,False,esc_lbls),
                     doy=lambda x: x.date.dt.strftime('%Y-%m-%d'))
act_y.drop(columns=['year','month','day','hour'],inplace=True)
# Find daily propotion for plot
dat_esc = act_y[act_y.date>=esc_start].groupby('doy').apply(lambda x: x.esc.value_counts(True)).reset_index().rename(columns={'esc':'share','level_1':'esc'}).assign(doy=lambda x: pd.to_datetime(x.doy))
# Esclation matters from
dat_ypost = act_y.loc[act_y.date >= esc_start].reset_index(None,True)

# Transition probability matrix
mat_trans = dat_ypost.assign(esct1=lambda x: x.esc.shift(1)).groupby(['esc','esct1']).y.count().reset_index()
mat_trans = mat_trans.merge(mat_trans.groupby('esc').y.sum().reset_index().rename(columns={'y':'n'})).assign(prob=lambda x: x.y/x.n).drop(columns=['y','n'])

# Subset predictions to datestart
df_ypost = df_pred[df_pred.dates >= esc_start].reset_index(None,True).drop(columns=cn_date+['hour','ub'])
# Map labels and predicted to thresholds
df_predpost = df_ypost.assign(y=lambda x: pd.cut(x.y,esc_bins,False,esc_lbls),
                                 pred=lambda x: pd.cut(x.pred,esc_bins,False,esc_lbls),
                                 lb=lambda x: pd.cut(x.lb,esc_bins,False,esc_lbls))
df_predpost = df_predpost.melt(['model','lead','y','dates'],None,'tt')
# Get the confusion matrix
dat_conf = df_predpost.groupby(cn_ml+['y','tt','value']).count().fillna(0).astype(int).reset_index()
dat_conf = dat_conf.rename(columns={'value':'pred','dates':'n'}).assign(pred=lambda x: pd.Categorical(x.pred,esc_lbls))
# dat_conf.pivot_table('n',['model','lead','tt','y'],'pred')[esc_lbls]

######################################
# --- (5) Escalation sensitivity --- #

crit = norm.ppf(0.975)  # Current quantile used in interval construction

df_sens = df_ypost.assign(se=lambda x: (x.pred-x.lb)/crit,
      ylbl=lambda x: pd.cut(x.y,esc_bins,False, esc_lbls)).drop(columns=['lb'])

### Q1: Does the actual label end up being more,less,or the same severity relative to prediction/actual
p_seq = np.linspace(0.05,0.50,10)

holder = []
for p in p_seq:
    print('Lower bound: %0.2f' % p)
    tmp = df_sens.assign(yhat=lambda x: pd.cut(x.pred-x.se*norm.ppf(1-p),esc_bins,False, esc_lbls)).drop(columns=['y','pred','se'])
    tmp = tmp.groupby(['model', 'lead', 'yhat', 'ylbl']).count().fillna(0).astype(int).reset_index().rename(columns={'dates':'n'})
    # Remove when both are pre-escalation levels
    tmp = tmp[~((tmp.ylbl == esc_lbls[0]) & (tmp.yhat == esc_lbls[0]))].reset_index(None,True)
    tmp = tmp.groupby(cn_ml).apply(lambda x: pd.Series({'less':np.sum((x.yhat > x.ylbl)*x.n),'more':np.sum((x.yhat < x.ylbl)*x.n),'same':np.sum((x.yhat == x.ylbl)*x.n)}))
    tmp = tmp.div(tmp.sum(1), 0).reset_index().assign(p=p)
    holder.append(tmp)
# same, more, less
dat_sml_pred = pd.concat(holder).reset_index(None,True).melt(['model','lead','p'],None,'tt')

# Calculate for the actual level
holder = []
sub = act_y[act_y.date >= esc_start].reset_index(None,True)
for lead in np.arange(1,mx_leads+1):
    tmp = pd.crosstab(index=sub.esc.values, columns=sub.esc.shift(lead).values, normalize=0)
    tmp = pd.DataFrame(tmp.values, columns=tmp.columns.astype(str), index=tmp.index.astype(str)).reset_index().melt('row_0').rename(columns={'row_0':'t0','col_0':'t1'}).assign(lead=lead)
    holder.append(tmp)
dat_sml_act = pd.concat(holder).assign(t0=lambda x: x.t0.map(di_esc), t1=lambda x: x.t1.map(di_esc)).sort_values(['lead','t0','t1']).reset_index(None,True)

### Q2: How much variation is there in the t+1,..t+7 levels?
tmp_acf = acf(x=sub.y.values,nlags=mx_leads,alpha=0.05)
tmp_pacf = pacf(x=sub.y.values,nlags=mx_leads,alpha=0.05)
cn_acf = ['mu','lb','ub']
tmp_acf = pd.DataFrame(np.hstack([cvec(tmp_acf[0]),tmp_acf[1]])[1:],columns=cn_acf)
tmp_pacf = pd.DataFrame(np.hstack([cvec(tmp_pacf[0]),tmp_pacf[1]])[1:],columns=cn_acf)
dat_acf = pd.concat([tmp_acf.assign(tt='ACF'), tmp_pacf.assign(tt='PACF')]).rename_axis('lead').reset_index().assign(lead=lambda x: x.lead+1)
lead_seq = range(mx_leads+1)[1:]
e2 = [np.nanmean((sub.y.values - sub.y.shift(lead).values)**2) for lead in lead_seq]
r2 = [r2_fun(sub.y.values, sub.y.shift(lead).values) for lead in lead_seq]
dat_acf = pd.concat([dat_acf,pd.DataFrame({'lead':lead_seq,'mu':r2,'tt':'R2'})])

### Q3: If there is a predicted level change, and such a change would be different than the current actual level, what does the actual level end up being?

print(act_y[act_y.doy=='2020-01-01'].drop(columns=['doy']))
print(df_pred[df_pred.dates==pd.to_datetime('2020-01-01 02:00:00')][['lead','dates','pred','y']])
# act_y[act_y.date.isin(pd.date_range('2020-03-14','2020-03-16',freq='1H'))]

tmp_act = pd.DataFrame(pd.Categorical(act_y.esc).codes, index=act_y.date,
                       columns=pd.MultiIndex.from_product([['act'],[0]])).rename_axis('dates')
holder = []
for p in p_seq:
    print('Lower bound: %0.2f' % p)
    tmp = df_sens.assign(pred=lambda x: x.pred-x.se*norm.ppf(1-p)).drop(columns='se')
    tmp = tmp.pivot_table(['pred','y'], ['model', 'dates'], 'lead').apply(lambda x: pd.cut(x,esc_bins,False, esc_lbls))
    tmp = tmp.apply(lambda x: pd.Categorical(x).codes.astype(int))
    tmp = tmp_act.join(tmp)
    # Get change relative to current level
    tmp = tmp.loc[:, ['pred','y']].subtract(tmp.loc[:, pd.IndexSlice['act', 0]].values, axis=0)
    tmp = tmp.reset_index().melt(['model', 'dates']).rename(columns={'variable_0':'tt','variable_1':'lead'})
    tmp = tmp.pivot_table('value', ['model', 'dates', 'lead'], 'tt').reset_index()
    # Performance for higher/lower escalation prediction
    tmp_gb = tmp.assign(psign=lambda x: np.sign(x.pred), ysign=lambda x: np.sign(x.y))
    tmp_gb = tmp_gb.pivot_table('dates',cn_ml+['psign'],'ysign','count').fillna(0).astype(int).reset_index()
    tmp_gb = tmp_gb.melt(cn_ml + ['psign'], None, None, 'n').sort_values(cn_ml+['psign']).reset_index(None,True).assign(p=p)
    holder.append(tmp_gb)
# Merge
cn_rel = cn_ml+['CI', 'psign']
dat_rel = pd.concat(holder).reset_index(None,True).assign(CI=lambda x: (1-2*x.p).round(2))
dat_rel = dat_rel.merge(dat_rel.groupby(cn_rel).n.sum().reset_index().rename(columns={'n':'tot'}))
dat_prec = dat_rel.assign(tp=lambda x: np.where(x.psign==x.ysign,'tp','fp')).groupby(cn_rel+['tp']).n.sum().reset_index()
dat_prec = dat_prec.pivot_table('n',cn_rel,'tp').reset_index().assign(prec = lambda x: x.tp/(x.tp+x.fp))
# Get the TPR
tmp = dat_rel.groupby(cn_ml+['CI','ysign']).n.sum().reset_index().rename(columns={'ysign':'psign'})
tmp1 = dat_prec.drop(columns=['tp','fp'])
tmp2 = dat_prec[cn_rel+['tp']].merge(tmp).assign(sens=lambda x: x.tp / x.n).drop(columns=['tp','n'])
dat_prec = tmp1.merge(tmp2).melt(cn_rel,['prec','sens'],'metric')

##########################################
# --- (6) EXAMPLE OF TRAJECTORY PLOT --- #

best_day = df_r2[(df_r2.model==models[0]) & (df_r2.month>=3)].groupby(cn_date).r2.min().sort_values(ascending=False).head(1).reset_index().drop(columns=['r2'])
best_day = pd.to_datetime(list(best_day.astype(str).apply(lambda x: '-'.join(x),1))[0])
drange = pd.date_range(best_day,periods=24,freq='1H')  #mx_leads
df_range = df_sens[(df_sens.model==models[0]) & (df_sens.dates.isin(drange))][['lead','pred','dates','se']].reset_index(None,True)
df_range = df_range.assign(lb=lambda x: x.pred-1.2*x.se, ub=lambda x: x.pred+1.2*x.se)
df_range = df_range.assign(dates2=df_range.apply(lambda x: x['dates'] + pd.DateOffset(hours=x['lead']),1))
df_range.dates = df_range.dates.dt.strftime('%H').astype(int)
tmp_act = act_y[act_y.date.isin(df_range.dates2.unique())]

tit = 'Example of real-time trajectory for %i hours ahead\nDay %s\nBlock dots are actual values' % (mx_leads,best_day.strftime('%b %d, %Y'))
for date in df_range.dates.unique():
    tmp = df_range[df_range.dates==date]
    gg = (ggplot(tmp, aes(x='dates2',y='pred',group='dates')) +
                 theme_bw() + geom_path(alpha=0.5,arrow=arrow(type='closed',length=0.1)) +
                 labs(y='Max patient per hour') + ggtitle(tit) +
                 scale_x_datetime(breaks='3 hours',date_labels='%H') +
                 theme(axis_title_x=element_blank()) +
                 geom_point(aes(x='date',y='y'),color='black',inherit_aes=False,data=tmp_act,size=2) +
            geom_ribbon(aes(ymin='lb',ymax='ub'),alpha=0.25,fill='blue') +
                 geom_hline(yintercept=31,linetype='--') +
                 geom_hline(yintercept=38,linetype='--') +
                    scale_y_continuous(limits=[-10,60]) +
                 geom_hline(yintercept=48,linetype='--'))
    gg.save(os.path.join(dir_figures, 'gg_trajectory_' + str(date) + '.png'), height=6, width=12)

gg_trajectory = (ggplot(df_range, aes(x='dates2',y='pred',color='dates',group='dates')) +
                 theme_bw() + geom_path(alpha=0.5,arrow=arrow(type='closed',length=0.1)) +
                 labs(y='Max patient per hour') + ggtitle(tit) +
                 scale_x_datetime(breaks='3 hours',date_labels='%H') +
                 theme(axis_title_x=element_blank()) +
                 scale_color_cmap(name='Point of forecast') +
                 geom_point(aes(x='date',y='y'),color='black',inherit_aes=False,data=tmp_act,size=2) +
                 geom_hline(yintercept=31,linetype='--') +
                 geom_hline(yintercept=38,linetype='--') +
                 geom_hline(yintercept=48,linetype='--') +
                 geom_point(data=df_range.groupby('dates').head(1)))
gg_trajectory.save(os.path.join(dir_figures, 'gg_trajectory.png'), height=6, width=12)


##########################
# --- (7) MAKE PLOTS --- #

### EXAMPLE OF VIOLATION WEEKS
for bound in viol_dates.bound.unique():
    gg_viol_ex = (ggplot(viol_dates[viol_dates.bound==bound],aes(x='dates',color='lead')) +
                  geom_point(aes(y='y')) + geom_line(aes(y='pred')) + theme_bw() +
                  geom_ribbon(aes(ymin='lb',ymax='ub',fill='lead'),alpha=0.5) +
                  facet_wrap('~bound+lead+doy+model',scales='free',labeller=label_both) +
                  ggtitle('Example of violation weeks') + guides(color=False,fill=False) +
                  labs(x='Hour of day', y='Max # of patients') +
                  scale_x_datetime(breaks='3 hours',date_labels='%H') +
                  theme(subplots_adjust={'wspace': 0.15,'hspace': 0.35}))
    gg_viol_ex.save(os.path.join(dir_figures,'gg_viol_ex_'+bound+'.png'),height=mx_leads,width=mx_leads)

### EXAMPLE OF GOOD/BAD R2 WEEKS
for tt in r2_dates.tt.unique():
    gg_r2_ex = (ggplot(r2_dates[r2_dates.tt==tt],aes(x='dates',color='lead')) + theme_bw() +
                  geom_point(aes(y='y')) + geom_line(aes(y='pred')) +
                  geom_ribbon(aes(ymin='lb',ymax='ub',fill='lead'),alpha=0.5) +
                  facet_wrap('~tt+lead+doy+model',scales='free',labeller=label_both) +
                  ggtitle('Example of best/worst performing weeks') +
                  labs(x='Hour of day', y='Max # of patients') +
                  scale_x_datetime(breaks='3 hours',date_labels='%H') +
                  theme(subplots_adjust={'wspace': 0.15,'hspace': 0.35}) +
                  guides(color=False,fill=False))
    gg_r2_ex.save(os.path.join(dir_figures,'gg_r2_ex_'+tt+'.png'),height=mx_leads,width=mx_leads)

### IQR DISTRIBUTION OF ONE-DAY-AHEAD RSQUARED BY LEAD
tmp = r2_desc.assign(month2=lambda x: x.month-(int(mx_leads/2)-x.lead)/(2*mx_leads))
gg_r2_month = (ggplot(tmp, aes(x='month2',y='med',color='lead')) + theme_bw() +
               geom_point() + geom_linerange(aes(ymin='lb',ymax='ub')) +
               labs(x='Month (2020)', y='One-day-ahead R2') +
               ggtitle('IQR performance on R2') +
               facet_wrap('~model',labeller=label_both) +
               scale_x_continuous(breaks=list(np.arange(mx_month+1))))
gg_r2_month.save(os.path.join(dir_figures,'gg_r2_month.png'),height=5,width=int(mx_leads*0.7))

### VIOLATION RATES IN A GIVEN MONTH BY LEAD
tmp = viol_desc.assign(month2=lambda x: x.month-(int(mx_leads/2)-x.lead)/(2*mx_leads)).sort_values(['bound','month','lead',])#.head(12).drop(columns=['n','prop'])
gg_viol_month = (ggplot(tmp,aes(x='month2',y='prop',color='lead')) + theme_bw() +
                 geom_point() +
                 geom_linerange(aes(ymin='lb',ymax='ub')) +
                 labs(x='Month (2020)', y='Violation %') +
                 ggtitle('Coverage of 95% confidence intervals by lead/model/month') +
                 facet_grid('model~bound',labeller=label_both) +
                 geom_hline(yintercept=0.025) +
                 scale_x_continuous(breaks=list(np.arange(mx_month+1))))
gg_viol_month.save(os.path.join(dir_figures,'gg_viol_month2.png'),height=5,width=int(mx_leads*0.7))

di_tt = {'pred':'Point', 'lb':'LowerBound'}
### CONFUSION MATRIX BY LB/PRED AND MODEL
gg_conf = (ggplot(dat_conf,aes(y='pred',x='y',fill='np.log(n+1)')) + theme_bw() +
           geom_tile(aes(width=1,height=1)) +
           labs(y='Predicted',x='Actual') +
           facet_wrap('~tt+lead+model',
                      labeller=labeller(tt=di_tt,lead=label_both,model=label_both)) +
           ggtitle('Confusion matrix by model/lead/bound') +
           geom_text(aes(label='n')) + guides(fill=False))
gg_conf.save(os.path.join(dir_figures,'gg_conf.png'),height=mx_leads,width=mx_leads)

### TRANSITION PROBABILITIES
tit = 'One-hour ahead empirical transition probability (%s)' % esc_start.strftime('≥%Y-%m-%d')
gg_trans = (ggplot(mat_trans,aes(x='esct1',y='esc',fill='prob')) +
            theme_bw() + ggtitle(tit) +
            geom_tile(aes(width=1,height=1)) +
            labs(y='State t0', x='State t1') +
            scale_fill_gradient2(name='Probability',limits=[0,1.01],breaks=list(np.linspace(0,1,5)),
                                 low='cornflowerblue',mid='grey',high='indianred',midpoint=0.5) +
            geom_text(aes(label='prob.round(2)')))
gg_trans.save(os.path.join(dir_figures,'gg_trans.png'),height=5,width=5.5)

### SAME, MORE LESS, RELATIVE TO PREDICTION
tit = 'Actual escalation level relative to prediction\nIgnores pred&act==Normal state'
gg_sml = (ggplot(dat_sml_pred, aes(x='1-p*2',y='value',color='lead',group='lead')) + theme_bw() +
          geom_point() + geom_line() +
          facet_wrap('~tt+model',labeller=label_both,ncol=3) +
          ggtitle(tit) + labs(x='± CI (lowerbound)',y='Share'))
gg_sml.save(os.path.join(dir_figures,'gg_sml_pred.png'),height=5,width=10)

### DAILY SHARE OF ESCLATION LEVELS
gg_esc_share = (ggplot(dat_esc,aes(x='doy',y='share',color='esc')) + theme_bw() +
                geom_point(size=0.5) + geom_line(size=1) + labs(y='Share') +
                ggtitle('Daily share of esclation levels (max patients per hour)') +
                theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
                scale_x_datetime(breaks='1 month', date_labels='%b, %Y') +
                scale_color_discrete(name='Level'))
gg_esc_share.save(os.path.join(dir_figures,'gg_esc_share.png'),height=6,width=10)

#### TRANSITION PROBABILITIES RELATIVE TO ESCALATION LEVEL
tmp = dat_sml_act.copy().rename(columns={'t0':'Escalation_Level'})
tit = 'Esclation transition by lead (%s)' % esc_start.strftime('≥%Y-%m-%d')
gg_esctrans_act = (ggplot(tmp, aes(x='lead',y='value',color='t1.astype(str)')) +
                   theme_bw() + geom_point(size=1) + geom_line() +
                   labs(x='Lead',y='Share') + ggtitle(tit) +
                   facet_wrap('~Escalation_Level',labeller=label_both, ncol=2) +
                   scale_color_discrete(name='ESc in lead') +
                   scale_x_continuous(breaks=list(range(mx_leads+1))))
gg_esctrans_act.save(os.path.join(dir_figures,'gg_esctrans_act.png'),height=8,width=10)

#### (PARTIAL) AUTO CORRELATION PROPERTIES
gg_acf = (ggplot(dat_acf, aes(x='lead',y='mu')) + theme_bw() +
          geom_point() + labs(x='Lead', y='Correlation') +
          scale_x_continuous(breaks=list(range(mx_leads+1))) +
          ggtitle('ACF/PACF for max hourly patients') + facet_wrap('~tt') +
          geom_linerange(aes(ymin='lb',ymax='ub')) +
          geom_hline(yintercept=0,color='blue',linetype='--'))
gg_acf.save(os.path.join(dir_figures,'gg_acf.png'),height=5,width=mx_leads)


### HOW OFTEN DO WE GET THE CHANGE IN THE LEVEL CORRECT?
# Let's examine how the distribution looks
tit = 'TPR/PPV for change in predicted escalation level' #\nTrue pos. implies change in actual esclation ≥ change in predicted escalation
di_metric = {'prec':'Precision', 'sens':'Sensitivity'}
tmp = dat_prec.assign(psign = lambda x: x.psign.map({-1:'Negative change',0:'No change',1:'Positive change'}))
gg_rel = (ggplot(tmp, aes(x='lead',y='value',color='CI',group='CI')) +
          theme_bw() + geom_point() + geom_line() +
          facet_grid('metric~psign+model',scales='free_y',
                     labeller=labeller(model=label_both,metric=di_metric)) +
          scale_x_continuous(breaks=list(range(mx_leads+1))) +
          labs(x='Lead',y='Precision/TPs') +
          scale_color_gradient(name='CI',limits=[0,1.01]) +
          ggtitle(tit))
gg_rel.save(os.path.join(dir_figures,'gg_rel.png'),height=8,width=int(mx_leads*0.75))


