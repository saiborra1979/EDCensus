import os
import numpy as np
import pandas as pd
from plotnine import *
from funs_support import gg_save, find_dir_olu, ymdh2date, ymd2date
from funs_stats import get_CI

from sklearn import metrics
from sklearn.metrics import r2_score

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
lst_dir = [dir_figures, dir_output]
assert all([os.path.exists(z) for z in lst_dir])

#########################
# --- (1) LOAD DATA --- #

# (i) processed yX data
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), header=[0,1], index_col=[0,1,2,3])
# Get datetime from index
dates = pd.to_datetime(df_lead_lags.index.to_frame().astype(str).apply(lambda x: x.year+'-'+x.month+'-'+x.day+' '+x.hour+':00:00',1))
df_lead_lags.index = pd.MultiIndex.from_frame(pd.concat([df_lead_lags.index.to_frame(),pd.DataFrame(dates).rename(columns={0:'date'})],1))
#Ancillary
idx = pd.IndexSlice
cn_ymd = ['year', 'month', 'day']
cn_ymdh = cn_ymd + ['hour']

dfmt = '%Y-%m-%d'
dmin = pd.to_datetime((dates.min() + pd.DateOffset(days=2)).strftime(dfmt))
dmax = pd.to_datetime((dates.max() - pd.DateOffset(days=2)).strftime(dfmt))


###################################
# --- (2) CENSUS & ESCALATION --- #

# Time series of max patients in a given hour #
dat_census = df_lead_lags.loc[:,idx['y','lead_0']].reset_index().droplevel(1,axis=1).assign(smooth= lambda x: x.y.rolling(24).mean().fillna(method='bfill'))
dat_census = dat_census.assign(doy=lambda x: ymd2date(x))

gg_census = (ggplot(dat_census, aes(x='date',y='y')) +
    geom_point(size=0.1) + geom_line(size=0.1) +
    labs(y='Max patients') + theme_bw() +
    geom_line(aes(x='date',y='smooth'),color='blue') +
    ggtitle('Number of maximum patients in ED by hour') +
    theme(axis_text_x=element_text(angle=90), axis_title_x=element_blank()) +
    scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y'))
gg_save('gg_census.png',dir_figures,gg_census,12,6)

# Compare to the escalation levels
ymi, ymx = dat_census.y.min(), dat_census.y.max()
esc_bins = [ymi - 1, 31, 38, 48, ymx + 1]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
di_esc = dict(zip(esc_lbls, range(len(esc_lbls))))
# Add on true labels to ground truth
dat_census = dat_census.assign(esc=lambda x: pd.cut(x.y, esc_bins, False, esc_lbls))

# Find daily propotion for plot
act_esc = dat_census.groupby('doy').apply(lambda x: x.esc.value_counts(True))
act_esc = act_esc.reset_index().rename(columns={'esc': 'share', 'level_1': 'esc'})

# Transition probability matrix
mat_trans = dat_census.assign(esct1=lambda x: x.esc.shift(1)).groupby(['esc', 'esct1']).y.count().reset_index()
mat_trans = mat_trans.merge(mat_trans.groupby('esc').y.sum().reset_index().rename(columns={'y': 'n'}))
mat_trans = mat_trans.assign(prob=lambda x: x.y / x.n).drop(columns=['y', 'n'])

### DAILY SHARE OF ESCLATION LEVELS
colz_esc = ['green','#e8e409','orange','red']
lblz = ['Normal','Level 1','Level 2','Level 3']
lblz = pd.Categorical(lblz, lblz)
gg_esc_share = (ggplot(act_esc, aes(x='doy', y='share', color='esc')) + theme_bw() +
                geom_point(size=0.25) + geom_line(size=0.5) + labs(y='Share') +
                ggtitle('Daily share of esclation levels (max patients per hour)') +
                theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
                scale_x_datetime(breaks='1 month', date_labels='%b, %Y') +
                scale_color_manual(name='Level',values=colz_esc))
gg_esc_share.save(os.path.join(dir_figures, 'gg_esc_share.png'), height=6, width=14)

tmp = pd.DataFrame({'x':dmin-pd.DateOffset(days=25),'y':[18,34,43,55], 'lbl':lblz})
gg_act_y = (ggplot(dat_census, aes(x='date', y='y')) + theme_bw() +
    geom_line(size=0.25,color='blue',alpha=0.75) + labs(y='Max patients per hour') +
    #ggtitle('Max patients per hour') +
    theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
    scale_x_datetime(breaks='1 month', date_labels='%b, %Y',
    limits=(dmin-pd.DateOffset(days=25),dmax)) +
    geom_hline(yintercept=31) + geom_hline(yintercept=38) + geom_hline(yintercept=48) +
    annotate('rect',xmin=dmin,xmax=dmax,ymin=0,ymax=31,fill='green',alpha=0.25)+
    annotate('rect',xmin=dmin,xmax=dmax,ymin=31,ymax=38,fill='#e8e409',alpha=0.25)+
    annotate('rect',xmin=dmin,xmax=dmax,ymin=38,ymax=48,fill='orange',alpha=0.25)+
    annotate('rect',xmin=dmin,xmax=dmax,ymin=48,ymax=ymx,fill='red',alpha=0.25) +
    geom_text(aes(x='x',y='y',label='lbl',color='lbl'),data=tmp) +
    scale_color_manual(values=colz_esc) + guides(color=False))
gg_act_y.save(os.path.join(dir_figures, 'gg_act_y.png'), height=5, width=20)

### TRANSITION PROBABILITIES
tit = 'One-hour ahead empirical transition probability'
gg_trans = (ggplot(mat_trans, aes(x='esct1', y='esc', fill='prob')) +
    theme_bw() + ggtitle(tit) +
    geom_tile(aes(width=1, height=1)) +
    labs(y='State t0', x='State t1') +
    scale_fill_gradient2(name='Probability', limits=[0, 1.01], breaks=list(np.linspace(0, 1, 5)),low='cornflowerblue', mid='grey', high='indianred', midpoint=0.5) +
    geom_text(aes(label='prob.round(2)')))
gg_trans.save(os.path.join(dir_figures, 'gg_trans.png'), height=5, width=5.5)


############################
# --- (2) R2 FROM HOUR --- #

nsim = 250
alpha = 0.05

# (i) Calculate the R2
dat_hour = dat_census.drop(columns=['date','smooth']).sort_values(['hour']+cn_ymd).reset_index(None,True)
dat_hour = dat_hour.assign(ly=lambda x: x.groupby('hour').y.shift(1)).dropna().astype(int)
dat_r2_hour = dat_hour.groupby('hour').apply(lambda x: r2_score(x.y,x.ly)).reset_index().rename(columns={0:'r2'})
# Get the CI
dat_bs_hour = dat_hour.drop(columns=cn_ymd).groupby('hour')
holder = []
for i in range(nsim):
    tmp_i = dat_bs_hour.apply(lambda x: x.sample(frac=1,replace=True,random_state=i)).reset_index(drop=True).groupby('hour').apply(lambda x: r2_score(x.y,x.ly)).reset_index().rename(columns={0:'r2'})
    holder.append(tmp_i)
sim_bs_hour = pd.concat(holder).reset_index(None,True)
dat_r2_hour = dat_r2_hour.merge(sim_bs_hour.groupby('hour').r2.std(ddof=1).reset_index().rename(columns={'r2':'se'}))
dat_r2_hour = get_CI(dat_r2_hour,'r2','se',alpha)

gtit = "R2 from using previous day's hour\nVertical lines show 95% CI"
gg_r2_hour = (ggplot(dat_r2_hour, aes(x='hour',y='r2')) +
    geom_point(size=2) + theme_bw() + ggtitle(gtit) +
    labs(y='R-squared',x='Hour of day') + 
    theme(axis_text_x=element_text(angle=90)) +
    geom_linerange(aes(ymin='lb',ymax='ub')) + 
    scale_x_continuous(breaks=range(0,24)) + 
    scale_y_continuous(limits=[0,1]))
gg_save('gg_r2_hour.png',dir_figures,gg_r2_hour,7,4)


# (ii) Calculated the scatter plot
gg_hour_scatter = (ggplot(dat_hour,aes(x='ly',y='y',color='month')) + 
    theme_bw() + geom_point(size=0.5,alpha=0.5) + 
    labs(x="Previous day's hour",y='Current hour') + 
    facet_wrap('~hour',labeller=label_both,nrow=4) + 
    geom_abline(slope=1,intercept=0))
gg_save('gg_hour_scatter.png',dir_figures,gg_hour_scatter,16,10)

# (iii) Calculate the rolling R2
n_h_w = 7*24
dat_rolling = dat_hour.assign(e=lambda x: (x.y-x.ly),date=lambda x: ymdh2date(x))
dat_rolling = dat_rolling.query('date>=@dmin & date<@dmax').reset_index(None,True)
dat_rolling_mape = dat_rolling.groupby(['year','month','day']).apply(lambda x: x.e.abs().sum()/x.y.abs().sum() )
dat_rolling_mape = dat_rolling_mape.reset_index().rename(columns={0:'mape'})
dat_rolling_mape['date'] = ymd2date(dat_rolling_mape)
ymax = np.ceil(dat_rolling_mape.mape.max()*10)/10

# Why the hours of the day are so powerful...
gtit = "Daily mean absolute percentage error using previous day's hour"
gg_daily_MAPE = (ggplot(dat_rolling_mape, aes(x='date',y='mape')) +
    theme_bw() + ggtitle(gtit) + 
    geom_line(size=0.5,alpha=0.5) + 
    labs(y='MAPE') + 
    # geom_hline(yintercept=0) + 
    scale_x_datetime(date_breaks='2 months',date_labels='%b, %y') + 
    theme(axis_text_x=element_text(angle=90),axis_title_x=element_blank()) + 
    scale_y_continuous(limits=[0,ymax]))
gg_daily_MAPE.save(os.path.join(dir_figures, 'gg_daily_MAPE.png'), height=5, width=10)





