import os
import numpy as np
import pandas as pd
import plotnine as pn
from funs_support import gg_save, find_dir_olu

import warnings
from plotnine.exceptions import PlotnineWarning
warnings.filterwarnings("ignore", category=PlotnineWarning)

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_data = os.path.join(dir_olu, 'data')
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
lst_dir = [dir_figures, dir_output]
assert all([os.path.exists(z) for z in lst_dir])

#########################
# --- (1) LOAD DATA --- #

cn_y = 'census_max'
# (i) processed yX data
df_yX = pd.read_csv(os.path.join(dir_flow, 'hourly_yX.csv'))
df_yX.date = pd.to_datetime(df_yX.date)
df_yX.rename(columns={cn_y:'y'},inplace=True)

dfmt = '%Y-%m-%d'
dmin = pd.to_datetime((df_yX.date.min() + pd.DateOffset(days=2)).strftime(dfmt))
dmax = pd.to_datetime((df_yX.date.max() - pd.DateOffset(days=2)).strftime(dfmt))
print('dmin: %s, dmax: %s' % (dmin, dmax))

######################################
# --- (3) COMPARISON TO REALTIME --- #

# Load the long census
df_wb = pd.read_csv(os.path.join(dir_flow,'long_census.csv'))
df_wb = df_wb[df_wb.date>='2021-01-01'].reset_index(None,True)
df_wb['date'] = pd.to_datetime(df_wb.date)
df_wb['date'] = df_wb.date.dt.tz_localize(tz='America/Toronto',nonexistent='shift_forward')
# Find the last arrival time
wb_max = df_wb.query('ticker == 1').date.max()
df_wb = df_wb[df_wb.date <= wb_max]

# Load the real-time database
df_rt = pd.read_csv(os.path.join(dir_data,'census_rt.csv'))
df_rt.rename(columns={'metric_value':'census', 'processed_time':'date'}, inplace=True)
df_rt['date'] = pd.to_datetime(df_rt.date,utc=True)
df_rt['date'] = df_rt.date.dt.tz_convert(tz='America/Toronto')
df_rt = df_rt.assign(ticker=lambda x: x.census.diff().fillna(0).astype(int))

# Look at first full day
dstart = pd.to_datetime(df_rt.date.min().strftime('%Y-%m-%d')).tz_localize(tz='EST')
dend = dstart + pd.DateOffset(days=3)

rt_sub = df_rt.query('date >= @dstart & date < @dend')
wb_sub = df_wb.query('date >= @dstart & date < @dend')
rt_wb_sub = pd.concat([rt_sub.assign(tt='rt'),wb_sub.assign(tt='wb')]).reset_index(None,True)

rt_wb_sub = rt_wb_sub.melt(['date','tt'],None,'msr')
di_rt = {'census':'Census', 'ticker':'Δ in Census'}

tmp_vlines = pd.DataFrame({'date':pd.date_range(dstart,dend,freq='1D')})
tmp_vlines = tmp_vlines.assign(lbl=lambda x: x.date.dt.strftime('%b, %d'))
tmp_vlines = pd.concat([tmp_vlines.assign(msr='census'),tmp_vlines.assign(msr='ticker')])
tmp_vlines = tmp_vlines.merge(rt_wb_sub.groupby('msr').value.quantile(0.9).reset_index())

# tmp1 = rt_wb_sub.query('msr=="census" & tt!="rt"')
# tmp2 = rt_wb_sub.query('msr=="census" & tt=="rt"')
# size=2,alpha=0.5

gg_rt_wb = (pn.ggplot(rt_wb_sub, pn.aes(y='value',x='date',color='tt')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.facet_wrap('~msr',scales='free_y',ncol=1,labeller=pn.labeller(msr=di_rt)) + 
    pn.geom_text(pn.aes(x='date',y='value',label='lbl'),data=tmp_vlines, inherit_aes=False) + 
    pn.geom_vline(pn.aes(xintercept='date'),linetype='--',data=tmp_vlines, inherit_aes=False) + 
    pn.scale_color_discrete(name='Dataset', labels=['HeroAI', 'WorkBench']) + 
    pn.theme(axis_title_x=pn.element_blank(),axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_x_datetime(date_breaks='3 hours',date_labels='%I%p'))
gg_save('gg_rt_wb',dir_figures,gg_rt_wb, 9, 8)

q1 = rt_wb_sub.query('msr=="census"').drop(columns='msr').sort_values('date').reset_index(None,True)
q1 = q1.assign(date=lambda x: x.date.dt.strftime('%Y-%m-%d %H:%M')).groupby(['date','tt']).value.mean().reset_index()
q1 = q1.pivot('date','tt','value').fillna(method='ffill')
q1 = q1[q1.notnull().all(1)]

m_seq = range(-10,11) #[pd.DateOffset(hours=l) for l in range(-10,11)]

holder = []
for m in m_seq:
    tmp = pd.Series({'shift':m,'rho':q1.assign(wb=lambda x: x.wb.shift(m)).corr().iloc[0,1]})
    holder.append(tmp)
pd.concat(holder,1).T.sort_values('rho',ascending=False).reset_index(None,True)


###################################
# --- (2) CENSUS & ESCALATION --- #

# Time series of max patients in a given hour #
dat_census = df_yX[['date','y']].copy().assign(smooth= lambda x: x.y.rolling(24).mean().fillna(method='bfill'))
dat_census = dat_census.assign(doy=lambda x: pd.to_datetime(x.date.dt.strftime(dfmt)))
dat_census['hour'] = dat_census.date.dt.hour

gg_census = (pn.ggplot(dat_census, pn.aes(x='date',y='y')) +
    pn.geom_point(size=0.1) + pn.geom_line(size=0.1) +
    pn.labs(y='Max hourly census') + pn.theme_bw() +
    pn.geom_line(pn.aes(x='date',y='smooth'),color='blue') +
    pn.ggtitle('ED hourly census') +
    pn.theme(axis_text_x=pn.element_text(angle=90), axis_title_x=pn.element_blank()) +
    pn.scale_x_datetime(date_breaks='2 month', date_labels='%b, %Y'))
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
gg_esc_share = (pn.ggplot(act_esc, pn.aes(x='doy', y='share', color='esc')) + pn.theme_bw() +
                pn.geom_point(size=0.25) + pn.geom_line(size=0.5) + pn.labs(y='Share') +
                pn.ggtitle('Daily share of esclation levels (max patients per hour)') +
                pn.theme(axis_title_x=pn.element_blank(), axis_text_x=pn.element_text(angle=90)) +
                pn.scale_x_datetime(breaks='1 month', date_labels='%b, %Y') +
                pn.scale_color_manual(name='Level',values=colz_esc))
gg_save('gg_esc_share.png',dir_figures,gg_esc_share,14,6)

tmp = pd.DataFrame({'x':dmin-pd.DateOffset(days=25),'y':[18,34,43,55], 'lbl':lblz})
gg_act_y = (pn.ggplot(dat_census, pn.aes(x='date', y='y')) + pn.theme_bw() +
    pn.geom_line(size=0.25,color='blue',alpha=0.75) + pn.labs(y='Max patients per hour') +
    pn.theme(axis_title_x=pn.element_blank(), axis_text_x=pn.element_text(angle=90)) +
    pn.scale_x_datetime(breaks='1 month', date_labels='%b, %Y',
    limits=(dmin-pd.DateOffset(days=25),dmax)) +
    pn.geom_hline(yintercept=31) + pn.geom_hline(yintercept=38) + pn.geom_hline(yintercept=48) +
    pn.annotate('rect',xmin=dmin,xmax=dmax,ymin=0,ymax=31,fill='green',alpha=0.25)+
    pn.annotate('rect',xmin=dmin,xmax=dmax,ymin=31,ymax=38,fill='#e8e409',alpha=0.25)+
    pn.annotate('rect',xmin=dmin,xmax=dmax,ymin=38,ymax=48,fill='orange',alpha=0.25)+
    pn.annotate('rect',xmin=dmin,xmax=dmax,ymin=48,ymax=ymx,fill='red',alpha=0.25) +
    pn.geom_text(pn.aes(x='x',y='y',label='lbl',color='lbl'),data=tmp) +
    pn.scale_color_manual(values=colz_esc) + pn.guides(color=False))
gg_save('gg_act_y.png',dir_figures,gg_act_y,20,5)

### TRANSITION PROBABILITIES
tit = 'One-hour ahead empirical transition probability'
gg_trans = (pn.ggplot(mat_trans, pn.aes(x='esct1', y='esc', fill='prob')) +
    pn.theme_bw() + pn.ggtitle(tit) +
    pn.geom_tile(pn.aes(width=1, height=1)) +
    pn.labs(y='State t0', x='State t1') +
    pn.scale_fill_gradient2(name='Probability', limits=[0, 1.01], breaks=list(np.linspace(0, 1, 5)),low='cornflowerblue', mid='grey', high='indianred', midpoint=0.5) +
    pn.geom_text(pn.aes(label='prob.round(2)')))
gg_save('gg_trans.png',dir_figures,gg_trans,5.5,5)








