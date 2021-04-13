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
dmin = (dates.min() + pd.DateOffset(days=2)).strftime(dfmt)
dmax = (dates.max() - pd.DateOffset(days=2)).strftime(dfmt)


######################
# --- (2) CENSUS --- #

# Time series of max patients in a given hour #
dat_census = df_lead_lags.loc[:,idx['y','lead_0']].reset_index().droplevel(1,axis=1).assign(smooth= lambda x: x.y.rolling(24).mean().fillna(method='bfill'))

gg_census = (ggplot(dat_census, aes(x='date',y='y')) +
    geom_point(size=0.1) + geom_line(size=0.1) +
    labs(y='Max patients') + theme_bw() +
    geom_line(aes(x='date',y='smooth'),color='blue') +
    ggtitle('Number of maximum patients in ED by hour') +
    theme(axis_text_x=element_text(angle=90), axis_title_x=element_blank()) +
    scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y'))
gg_save('gg_census.png',dir_figures,gg_census,12,6)

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





