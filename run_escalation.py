"""
SCRIPT TO COMPARE MODEL PERFORMANCE TO THE ESLCATION LEVELS
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import r2_score as r2
from statsmodels.stats.proportion import proportion_confint as propCI
from scipy.stats import norm

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_figures = os.path.join(dir_base, '..', 'figures')

models = ['gpy']
cn_ml = ['model','lead']

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
tmp = df_pred.loc[df_pred.model==models[0],['dates','y','lead']]
tmp['dates'] = tmp.apply(lambda x: x['dates']+pd.offsets.Hour(x['lead']),1)
tmp = tmp.pivot('dates','lead','y').fillna(method='ffill',axis=1).fillna(method='bfill',axis=1).astype(int).iloc[:,0]
act_y = tmp.reset_index().rename(columns={1:'y'})

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

ymx = df_pred.y.max()
esc_bins = [df_pred.lb.min()-1,31,38,48,ymx+2]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
esc_start = pd.to_datetime('2020-03-15')

act_y = act_y.assign(ylbl=lambda x: pd.cut(x.y,esc_bins,False,esc_lbls))

# Levels: pre-esclation (<= 30), escalation level 1 (31-37), escalation level 2 (38-47), escalation level 3 (>=48)
dat_y = df_pred.loc[(df_pred.model==models[0]) & (df_pred.lead==mi_leads),['y','dates','lead']]
dat_y.y = dat_y.y.shift(-mi_leads)
dat_y = dat_y.assign(esc=lambda x: pd.cut(x.y,esc_bins,False,esc_lbls),
                     doy=lambda x: x.dates.dt.strftime('%Y-%m-%d'))
dat_esc = dat_y.groupby('doy').apply(lambda x: x.esc.value_counts(True)).reset_index().rename(columns={'esc':'share','level_1':'esc'}).assign(doy=lambda x: pd.to_datetime(x.doy))
# Esclation matters from
dat_ypost = dat_y[dat_y.dates >= esc_start].reset_index(None,True)

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

### Q1: Does the actual label end up being more,less,or the same severity relative to prediction
p_seq = np.linspace(0.05,0.50,10)

holder = []
for p in p_seq:
    print('Lower bound: %0.2f' % p)
    tmp = df_sens.assign(yhat=lambda x: pd.cut(x.pred-x.se*norm.ppf(1-p),esc_bins,False, esc_lbls)).drop(columns=['y','pred','se'])
    tmp = tmp.groupby(['model', 'lead', 'yhat', 'ylbl']).count().fillna(0).astype(int).reset_index().rename(columns={'dates':'n'})
    print(tmp.head(12))
    # Remove when both are pre-escalation levels
    tmp = tmp[~((tmp.ylbl == esc_lbls[0]) & (tmp.yhat == esc_lbls[0]))].reset_index(None,True)
    tmp = tmp.groupby(cn_ml).apply(lambda x: pd.Series({'less':np.sum((x.yhat > x.ylbl)*x.n),'more':np.sum((x.yhat < x.ylbl)*x.n),'same':np.sum((x.yhat == x.ylbl)*x.n)}))
    tmp = tmp.div(tmp.sum(1), 0).reset_index().assign(p=p)
    holder.append(tmp)
# same, more, less
dat_sml = pd.concat(holder).reset_index(None,True).melt(['model','lead','p'],None,'tt')

### Q2: If there is a predicted level change, and such a change would be higher than the current actual level, what does the actual level end up being?


print(act_y.head(7))
print(df_pred[df_pred.dates==pd.to_datetime('2020-01-01 02:00:00')][['lead','dates','pred','y']])

tmp_act = pd.DataFrame(pd.Categorical(act_y.ylbl).codes, index=act_y.dates,
                       columns=pd.MultiIndex.from_product([['act'],[0]]))  #{'act':}
holder = []
for p in p_seq:
    print('Lower bound: %0.2f' % p)
    tmp = df_sens.assign(pred=lambda x: x.pred-x.se*norm.ppf(1-p)).drop(columns='se')
    tmp = tmp.pivot_table(['pred','y'], ['model', 'dates'], 'lead').apply(lambda x: pd.cut(x,esc_bins,False, esc_lbls))
    tmp = tmp.apply(lambda x: pd.Categorical(x).codes.astype(int))
    tmp = tmp_act.join(tmp)
    # Get change relative to current level
    tmp = tmp.loc[:, ['pred','y']].subtract(tmp.loc[:, pd.IndexSlice['act', 0]].values, axis=0)
    # tmp = tmp.loc[:,pd.IndexSlice[['pred']]].diff(axis=1).fillna(0).astype(int).join(tmp.loc[:, pd.IndexSlice[['y']]].diff(axis=1).fillna(0).astype(int))
    tmp = tmp.reset_index().melt(['model', 'dates']).rename(columns={'variable_0':'tt','variable_1':'lead'})
    tmp = tmp.pivot_table('value', ['model', 'dates', 'lead'], 'tt').reset_index()
    tmp_prec = tmp[tmp.pred>0].groupby(cn_ml).apply(lambda x: pd.Series({'tp':np.sum(x.y >= x.pred),'fp':np.sum(x.y < x.pred)})).assign(p=p).reset_index()
    holder.append(tmp_prec)

    # # Count the pred/y combinations relative to baseline
    # tmp2 = tmp.pivot_table('dates', cn_ml+['y'], ['pred'], 'count', 0).reset_index().melt(cn_ml + ['y'])
    # tmp2 = tmp2.pivot_table('value', cn_ml + ['pred'], ['y'], 'sum', 0).reset_index().melt(cn_ml+['pred'])
    # tmp2.rename(columns={'value':'n'},inplace=True)
    # # Normalize by prediction
    # tmp2 = tmp2.merge(tmp2.groupby(cn_ml+['pred']).n.sum().reset_index().rename(columns={'n':'tot'})).assign(share=lambda x: x.n/x.tot, p=p)
# Merge
dat_rel = pd.concat(holder).reset_index(None,True).assign(CI=lambda x: (1-2*x.p).round(2), prec=lambda x: x.tp/(x.tp+x.fp)).drop(columns='p').melt(cn_ml+['CI'],['tp','prec'],'metric')

# Let's examine how the distribution looks
tit = 'Positive in predicted escalation level relative to current level\nTrue pos. implies change in actual esclation ≥ change in predicted escalation'
gg_rel = (ggplot(dat_rel, aes(x='lead',y='value',color='CI',group='CI')) + theme_bw() +
          geom_point() + geom_line() +
          facet_grid('metric~model',scales='free_y',
                     labeller=labeller(model=label_both,metric={'tp':'True pos.','prec':'Precision'})) +
          scale_x_continuous(breaks=list(range(mx_leads+1))) +
          labs(x='Lead',y='Precision/TPs') +
          scale_color_gradient(name='CI',limits=[0,1.01]) +
          ggtitle(tit))
gg_rel.save(os.path.join(dir_figures,'gg_rel.png'),height=8,width=8)


### Q3: How much variation is there in the t+1,..t+7 levels?

##########################
# --- (X) MAKE PLOTS --- #

### SAME, MORE LESS, RELATIVE TO PREDICTION
tit = 'Actual escalation level relative to prediction\nIgnores pred&act==Normal state'
gg_sml = (ggplot(dat_sml, aes(x='1-p*2',y='value',color='lead',group='lead')) + theme_bw() +
          geom_point() + geom_line() +
          facet_wrap('~tt+model',labeller=label_both,ncol=3) +
          ggtitle(tit) + labs(x='± CI (lowerbound)',y='Share'))
gg_sml.save(os.path.join(dir_figures,'gg_sml.png'),height=5,width=10)


di_tt = {'pred':'Point', 'lb':'LowerBound'}
### CONFUSION MATRIX BY LB/PRED AND MODEL
gg_conf = (ggplot(dat_conf,aes(y='pred',x='y',fill='np.log(n+1)')) + theme_bw() +
           geom_tile(aes(width=1,height=1)) +
           labs(y='Predicted',x='Actual') +
           facet_wrap('~tt+lead+model',ncol=mx_leads,
                      labeller=labeller(tt=di_tt,lead=label_both,model=label_both)) +
           ggtitle('Confusion matrix by model/lead/bound') +
           geom_text(aes(label='n')) + guides(fill=False))
gg_conf.save(os.path.join(dir_figures,'gg_conf.png'),height=7,width=21)


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

### DAILY SHARE OF ESCLATION LEVELS
gg_esc_share = (ggplot(dat_esc,aes(x='doy',y='share',color='esc')) + theme_bw() +
                geom_point(size=0.5) + geom_line(size=1) + labs(y='Share') +
                ggtitle('Daily share of esclation levels (max patients per hour)') +
                theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
                scale_x_datetime(breaks='1 month', date_labels='%b, %Y') +
                scale_color_discrete(name='Level'))
gg_esc_share.save(os.path.join(dir_figures,'gg_esc_share.png'),height=6,width=10)

### EXAMPLE OF VIOLATION WEEKS
gg_viol_ex = (ggplot(viol_dates,aes(x='dates',color='lead')) + theme_bw() +
              geom_point(aes(y='y')) +
              geom_line(aes(y='pred')) +
              geom_ribbon(aes(ymin='lb',ymax='ub',fill='lead'),alpha=0.5) +
              facet_wrap('~bound+lead+doy+model',scales='free',ncol=mx_leads,labeller=label_both) +
              ggtitle('Example of violation weeks') +
              labs(x='Hour of day', y='Max # of patients') +
              scale_x_datetime(breaks='3 hours',date_labels='%H') +
              theme(subplots_adjust={'wspace': 0.15,'hspace': 0.35}) +
              guides(color=False,fill=False))
gg_viol_ex.save(os.path.join(dir_figures,'gg_viol_ex.png'),height=8,width=16)

### EXAMPLE OF GOOD/BAD R2 WEEKS
gg_r2_ex = (ggplot(r2_dates,aes(x='dates',color='lead')) + theme_bw() +
              geom_point(aes(y='y')) +
              geom_line(aes(y='pred')) +
              geom_ribbon(aes(ymin='lb',ymax='ub',fill='lead'),alpha=0.5) +
              facet_wrap('~tt+lead+doy+model',scales='free',ncol=mx_leads,labeller=label_both) +
              ggtitle('Example of best/worst performing weeks') +
              labs(x='Hour of day', y='Max # of patients') +
              scale_x_datetime(breaks='3 hours',date_labels='%H') +
              theme(subplots_adjust={'wspace': 0.15,'hspace': 0.35}) +
              guides(color=False,fill=False))
gg_r2_ex.save(os.path.join(dir_figures,'gg_r2_ex.png'),height=8,width=16)

### IQR DISTRIBUTION OF ONE-DAY-AHEAD RSQUARED BY LEAD
tmp = r2_desc.assign(month2=lambda x: x.month-(x.lead-4)/14)
gg_r2_month = (ggplot(tmp, aes(x='month2',y='med',color='lead')) + theme_bw() +
               geom_point() + geom_linerange(aes(ymin='lb',ymax='ub')) +
               labs(x='Month (2020)', y='One-day-ahead R2') +
               ggtitle('IQR performance on R2') +
               facet_wrap('~model',labeller=label_both) +
               scale_x_continuous(breaks=list(np.arange(mx_month+1))))
gg_r2_month.save(os.path.join(dir_figures,'gg_r2_month.png'),height=5,width=7)

### VIOLATION RATES IN A GIVEN MONTH BY LEAD
tmp = viol_desc.assign(month2=lambda x: x.month-(x.lead-4)/14)
gg_viol_month = (ggplot(tmp,aes(x='month2',y='prop',color='lead')) + theme_bw() +
                 geom_point() +
                 geom_linerange(aes(ymin='lb',ymax='ub')) +
                 labs(x='Month (2020)', y='Violation %') +
                 ggtitle('Coverage of 95% confidence intervals by lead/model/month') +
                 facet_grid('model~bound',labeller=label_both) +
                 geom_hline(yintercept=0.025) +
                 scale_x_continuous(breaks=list(np.arange(mx_month+1))))
gg_viol_month.save(os.path.join(dir_figures,'gg_viol_month2.png'),height=5,width=10)






