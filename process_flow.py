"""
SCRIPT TO PERFORM RAW PROCESSING
NUMBER OF PATIENT AT EACH TIME POINT THAT A NEW PATIENT ARRIVES OR IS DISCHARGED
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bfreq', type=str, default='1 hour', help='Time binning frequency')
parser.add_argument('--ylbl', type=str, default='census_max', help='Target label')
parser.add_argument('--nlags', type=int, default=10, help='Max number of lags')
args = parser.parse_args()
print(args)
bfreq, ylbl, nlags = args.bfreq, args.ylbl, args.nlags

# Debugging in PyCharm
bfreq = '1 hour'; ylbl = 'census_max'; nlags = 10


# Load in the required modules
import os
import numpy as np
import pandas as pd
from funs_support import add_lags, ljoin

dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '..', 'data')
dir_pulls = os.path.join(dir_base, '..', 'pulls')
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_clin = os.path.join(dir_pulls, 'triage_clin')
dir_labs = os.path.join(dir_pulls,'labs')
dir_DI = os.path.join(dir_pulls,'DI')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')

idx = pd.IndexSlice

######################################
# --- STEP 1: LOAD CLINICAL DATA --- #

print('STEP 1A: Loading Flow variables')
cn_clin_use = ['CSN', 'MRN', 'Arrived', 'Triage Comp to PIA', 'Treatment Team',
               'ED Completed Length of Stay (Minutes)', 'LOS']
cn_clin_triage = ['CSN', 'MRN', 'arrived', 'time2pia', 'md_team', 'los_min', 'los_clock']
di_clin_map = dict(zip(cn_clin_use, cn_clin_triage))
print(pd.DataFrame({'orig': cn_clin_use, 'trans': cn_clin_triage}))
fn_clin = os.listdir(dir_clin)
holder = []
for fn in fn_clin:
    tmp = pd.read_csv(os.path.join(dir_clin, fn), encoding='ISO-8859-1', usecols=cn_clin_use).rename(
        columns=di_clin_map)
    holder.append(tmp)
dat_clin = pd.concat(holder).reset_index(None, True)
del holder
# Convert to datetime
dat_clin['arrived'] = dat_clin.arrived.str.strip().str.replace('\\s', '-')
dat_clin['arrived'] = pd.to_datetime(dat_clin.arrived, format='%d/%m/%y-%H%M')
dat_clin = dat_clin.sort_values('arrived').reset_index(None, True)

# Remove rows if missing
cn_nodrop = ['arrived', 'los_min', 'los_clock']
idx_drop = ~dat_clin[cn_nodrop].isnull().any(axis=1)
print('A total of %i rows had to be dropped (no arrived or LOS field)' % sum(~idx_drop))
dat_clin = dat_clin[idx_drop].reset_index(None, True)

# Ensure all days are found
fmt = '%Y-%m-%d'
first_day = '2018-06-01'
dt_min, dt_max = dat_clin.arrived.min().strftime(fmt), dat_clin.arrived.max().strftime(fmt)
dt_range = pd.date_range(dt_min, dt_max,freq='d').astype(str)
udt = [first_day] + list(dat_clin.arrived.dt.strftime(fmt).unique())
assert dt_range.isin(udt).all()

# Time to PIA
dat_clin['time2pia'] = dat_clin.time2pia.str.split('\\:', expand=True).astype(float).apply(
    lambda x: x[0] * 60 + x[1], axis=1)
# LOS in clock
dat_clin['los_clock'] = dat_clin.los_clock.str.split('\\:', expand=True).astype(int).apply(lambda x: x[0] * 60 + x[1],
                                                                                           axis=1)
dat_clin['los'] = dat_clin[['los_min', 'los_clock']].min(axis=1).astype(int)
# For LOS=0 to LOS=1
dat_clin['los'] = np.where(dat_clin.los == 0, 1, dat_clin.los)
los_max = dat_clin[['los_min', 'los_clock']].max(axis=1).astype(int)
dat_clin['los'] = np.where(dat_clin.los < dat_clin.time2pia, los_max, dat_clin.los)
dat_clin.drop(columns=['los_min', 'los_clock'], inplace=True)
# Discharge time based on LOS
dat_clin['discharged'] = [dat_clin.arrived[i] + pd.Timedelta(minutes=dat_clin.los[i]) for i in range(dat_clin.shape[0])]

# Get the size/composition of the clinical team
mds = dat_clin.md_team.str.strip().str.replace('\\;$|^\\;', '')
mds = mds.str.replace('\\;\\s', ';').str.replace('^\\s', '')
mds = mds.str.replace('[^A-Z\\;]', '_')
# mds = mds.str.split('\\;')  # For counting below
dat_clin['md_team'] = mds.copy()

# Merge on demographics
dat_demo = pd.read_csv(os.path.join(dir_flow, 'demo4flow.csv'))
dat_clin = dat_clin.merge(dat_demo,'left',['CSN','MRN'])
dat_clin.drop(columns = ['MRN','time2pia','los'],inplace=True)

###################################
# --- STEP 2: LOAD LAB ORDERS --- #

print('# --- STEP 2: LOAD LAB ORDERS --- #')
cn_labs = {'MRN':'MRN','Order ID':'order_id','Order Date':'date',
         'Order Time':'time','Order Name':'name'}
fn_labs = os.listdir(dir_labs)
holder = []
for fn in fn_labs:
    tmp = pd.read_csv(os.path.join(dir_labs, fn), encoding='ISO-8859-1', usecols=list(cn_labs)).rename(
        columns=cn_labs)
    holder.append(tmp)
dat_labs = pd.concat(holder).reset_index(None, True)
del holder, tmp
dat_labs['order_time'] = pd.to_datetime(dat_labs.date + ' ' + dat_labs.time,format='%d/%m/%Y %I:%M:%S %p')
dat_labs.drop(columns=['date','time'],inplace=True)
pd.DataFrame({'lab':dat_labs.name.unique()}).to_csv(os.path.join(dir_output,'u_labs.csv'))
# Ensure all days are found
udt = [first_day] + list(dat_labs.order_time.dt.strftime(fmt).unique())
assert dt_range.isin(udt).all()
print(dt_range[~dt_range.isin(udt)])
# Fast parse
dat_labs.name = dat_labs.name.str.split('\\s',1,True).iloc[:,0].str.lower()
freq_labs = dat_labs.name.value_counts(True)
top_labs = list(freq_labs[freq_labs > 0.01].index)
dat_labs.name = np.where(dat_labs.name.isin(top_labs), dat_labs.name, 'other')
# Put into wide form
dat_labs_wide = dat_labs.groupby(['order_time','name']).size().reset_index().pivot('order_time','name',0).fillna(0).astype(int).add_prefix('labs_',).reset_index()

##################################
# --- STEP 3: LOAD DI ORDERS --- #

print('# --- STEP 3: LOAD DO ORDERS --- #')
cn_DI = {'MRN':'MRN','Order ID':'order_id','Order Date':'date',
         'Order Time':'time','Order Name':'name'}
fn_DI = os.listdir(dir_DI)
holder = []
for fn in fn_DI:
    tmp = pd.read_csv(os.path.join(dir_DI, fn), encoding='ISO-8859-1', usecols=list(cn_DI)).rename(
        columns=cn_DI)
    holder.append(tmp)
dat_DI = pd.concat(holder).reset_index(None, True)
pd.DataFrame({'lab':dat_DI.name.unique()}).to_csv(os.path.join(dir_output,'u_DI.csv'))
del holder, tmp
dat_DI['order_time'] = pd.to_datetime(dat_DI.date + ' ' + dat_DI.time,format='%d/%m/%Y %I:%M:%S %p')
dat_DI.drop(columns=['date','time'],inplace=True)
# Ensure all days are found
udt = [first_day] + list(dat_DI.order_time.dt.strftime(fmt).unique())
assert dt_range.isin(udt).all()
print(dt_range[~dt_range.isin(udt)])
# Fast parse
u_DI = pd.Series(dat_DI.name.unique())
u_DI = pd.DataFrame(u_DI.str.split('\\s',2,True).iloc[:,0:2]).rename(columns={0:'device',1:'organ'}).assign(term=u_DI)
u_DI['device'] = u_DI.device.str.lower().str.replace('\\-|\\/','')
u_DI.device[u_DI.device.str.contains('[^a-z]')] = 'other'
u_DI.organ = u_DI.organ.fillna('other')
u_DI.organ[u_DI.organ.str.contains('[0-9]')] = 'other'
u_DI.organ[u_DI.organ.str.len()==2] = 'Other'
di_device = dict(zip(u_DI.term, u_DI.device))
di_organ = dict(zip(u_DI.term, u_DI.organ))
dat_DI = dat_DI.assign(device=lambda x: x.name.map(di_device), organ=lambda x: x.name.map(di_organ))
dat_DI = dat_DI.drop(columns = ['MRN','order_id','name']).sort_values('order_time').reset_index(None,True)
# Put into wide form
dat_DI_wide = dat_DI.groupby(['order_time','device']).size().reset_index().pivot('order_time','device',0).fillna(0).astype(int).add_prefix('DI_',).reset_index()
# Aggregate if too lother
di_tot = dat_DI_wide.sum(0)
di_low = list(di_tot[di_tot < 100].index)
dat_DI_wide.DI_other += dat_DI_wide[di_low].sum(1)
dat_DI_wide.drop(columns=di_low,inplace=True)

# dat_DI_wide = dat_DI_wide.pivot_table(0,['order_time'],['device','organ']).fillna(0).astype(int)
# dat_DI_wide.columns = dat_DI_wide.columns.to_frame().apply(lambda x: x[0]+'_'+x[1],1)

##########################################
# --- STEP 5: HOURLY FLOW ON LABS/DI --- #

cn_date = ['year', 'month', 'day', 'hour']

# Get year/month/day/hour
dat_DI_flow = dat_DI_wide.assign(year=lambda x: x.order_time.dt.year, month=lambda x: x.order_time.dt.month, day=lambda x: x.order_time.dt.day, hour=lambda x: x.order_time.dt.hour).drop(columns=['order_time'])
dat_labs_flow = dat_labs_wide.assign(year=lambda x: x.order_time.dt.year, month=lambda x: x.order_time.dt.month, day=lambda x: x.order_time.dt.day, hour=lambda x: x.order_time.dt.hour).drop(columns=['order_time'])
# Aggregate by hour
dat_DI_flow = dat_DI_flow.groupby(cn_date)[dat_DI_wide.columns.drop('order_time')].sum().reset_index()
dat_labs_flow = dat_labs_flow.groupby(cn_date)[dat_labs_wide.columns.drop('order_time')].sum().reset_index()

#######################################################
# --- STEP 6: FLOW ON ARRIVAL AND DISCHARGE TIMES --- #

cn_clin = dat_clin.columns.drop(['CSN','arrived','discharged'])
print('Clinical features: %s' % ', '.join(cn_clin))
# (iii) Columns that can be one-hot encoded
dt_clin = dat_clin[cn_clin].dtypes
cn_num = list(dt_clin[dt_clin != 'object'].index)
cn_fac = list(dt_clin[dt_clin == 'object'].index.drop('md_team'))

cn_idx = ['CSN'] + list(cn_clin)  # All features to index on
# Long-form
dat_long = dat_clin.melt(cn_idx, ['arrived', 'discharged'], 'tt', 'date').sort_values('date').reset_index(None, True).assign(ticker=lambda x: np.where(x.tt == 'arrived', +1, -1))
# Actual patient count
dat_long.insert(0, 'census', dat_long.ticker.cumsum())
# Hourly results
dat_long = dat_long.assign(year=lambda x: x.date.dt.year, month=lambda x: x.date.dt.month, day=lambda x: x.date.dt.day, hour=lambda x: x.date.dt.hour)
num_days = dat_long.groupby(cn_date[:-1]).size().shape[0]
print('There are %i total days in the dataset' % num_days)
# Get the last day/hour
mx_year, mx_month, mx_day, mx_hour = list(dat_long.tail(1)[cn_date].values.flatten())
mi_year, mi_month, mi_day, mi_hour = list(dat_long.head(1)[cn_date].values.flatten())
dt_max = pd.to_datetime(str(mx_year)+'-'+str(mx_month)+'-'+str(mx_day)+' '+str(mx_hour)+':00:00')
dt_min = pd.to_datetime(str(mi_year)+'-'+str(mi_month)+'-'+str(mi_day)+' '+str(mi_hour)+':00:00')
# # Create a one-day buffer
# dt_min = dt_min + pd.DateOffset(days=1)
dt_max = dt_max - pd.DateOffset(days=1)
print('Start date: %s, end date: %s' % (dt_min, dt_max))

# (i) Hourly arrival/discharge
hourly_tt = dat_long.groupby(cn_date + ['tt']).size().reset_index()
hourly_tt = hourly_tt.pivot_table(0, cn_date[:-1] + ['tt'], 'hour').fillna(0).reset_index().melt(cn_date[:-1] + ['tt'])
hourly_tt = hourly_tt.pivot_table('value', cn_date, 'tt').fillna(0).add_prefix('tt_').astype(int).reset_index()
hourly_tt = hourly_tt.assign(date=lambda x: pd.to_datetime(x.year.astype(str)+'-'+x.month.astype(str)+'-'+x.day.astype(str)+' '+x.hour.astype(str)+':00:00'))
# Subset
hourly_tt = hourly_tt[(hourly_tt.date >= dt_min) & (hourly_tt.date <= dt_max)].reset_index(None,True)
# Long version for benchmarks
hourly_tt_long = hourly_tt.melt(['date']+cn_date, ['tt_arrived', 'tt_discharged']).assign(tt=lambda x: x.tt.str.replace('tt_','')).rename(columns={'value':'census'})

# (ii) Census moments in that hour  ['max','var']
cn_moment = ['max','var']
di_cn = dict(zip(cn_moment,['census_'+z for z in cn_moment]))
hourly_census = dat_long.pivot_table('census', cn_date[:-1], 'hour', cn_moment)
hourly_census = hourly_census.reset_index().melt(['year','month','day']).rename(columns={None:'moment'})
assert hourly_census.shape[0] == num_days*24*len(cn_moment)
hourly_census = hourly_census.pivot_table('value',cn_date,'moment',dropna=False).reset_index().rename(columns=di_cn)
# Subset the year/month/day counter
# Forward fill any census values (i.e. if there are 22 patients at hour 1, and no one arrives/leaves at hour 2, then there are 22 patients at hour 2
hourly_census = hourly_tt[cn_date].merge(hourly_census,'left',cn_date).fillna(method='ffill')
hourly_census['census_max'] = hourly_census.census_max.astype(int)
hourly_census['census_var'] = hourly_census.census_var.fillna(0)

holder = []
for ii, cn in enumerate(cn_fac):
    print('column: %s (%i of %i)' % (cn, ii + 1, len(cn_fac)))
    # Get into long-format
    tmp = dat_long.groupby(cn_date + ['tt', cn]).size().reset_index()
    tmp = tmp.pivot_table(0,cn_date[:-1],['tt','hour',cn]).fillna(0).astype(int).reset_index().melt(cn_date[:-1])
    tmp = tmp.sort_values([cn] + cn_date + ['tt']).reset_index(None, True)
    # Get share at each hour
    tmp = tmp.pivot_table('value', cn_date + ['tt'], cn)
    # Factor should exactly equal arrival/discharge
    qq = tmp.sum(1).reset_index().rename(columns={0:'n'}).merge(hourly_tt_long,'right',cn_date+['tt']).drop(columns=['date'])
    assert np.all(qq.n == qq.census)
    # den = np.atleast_2d(tmp.sum(1).values).T
    # vals = tmp.values / den
    # vals[np.where(np.isnan(vals))] = 1 / tmp.columns.shape[0]
    # tmp = pd.DataFrame(vals, columns=tmp.columns, index=tmp.index)
    # Re-expand into wide form
    tmp = tmp.reset_index().pivot_table(tmp.columns,cn_date,'tt')
    tmp.columns = tmp.columns.to_frame().apply(lambda x: cn + '_' + x[0] + '_' + x[1], 1).to_list()
    # Merge with existing date
    tmp = hourly_tt[cn_date].merge(tmp.reset_index(), 'left', cn_date)
    tmp = pd.DataFrame(tmp.drop(columns=cn_date).values,
                       columns=tmp.columns.drop(cn_date),
                       index=pd.MultiIndex.from_frame(tmp[cn_date]))
    tmp = tmp.fillna(0).astype(int)
    holder.append(tmp)
    del tmp
hourly_fac = pd.concat(holder, 1)
cn_X_fac = list(hourly_fac.columns)
hourly_fac.reset_index(inplace=True)

# (iv) Columns with numerical features (mean, var)
holder_num = dat_long.groupby(cn_date + ['tt'])[cn_num].mean().reset_index()
holder_num = hourly_tt_long[cn_date+ ['tt']].merge(holder_num, 'left', cn_date+ ['tt'])
holder_num[cn_num] = holder_num.groupby('tt')[cn_num].fillna(method='ffill').fillna(method='bfill')
holder_num = holder_num.pivot_table(cn_num, cn_date, 'tt')
holder_num.columns = holder_num.columns.to_frame().apply(lambda x: x[0]+'_'+x['tt'],1).to_list()
holder_num.reset_index(inplace=True)

# (v) MD team
num_mds = (dat_long.md_team.str.count(';')+1).fillna(0).astype(int)
holder_avg_mds = dat_long.assign(avg_mds=num_mds).groupby(cn_date+['tt']).avg_mds.mean().reset_index()
# Merge and widen
holder_avg_mds = hourly_tt_long[cn_date+['tt']].merge(holder_avg_mds, 'left', cn_date+['tt'])
holder_avg_mds['avg_mds'] = holder_avg_mds.groupby('tt').avg_mds.fillna(method='ffill').fillna(method='bfill')
holder_avg_mds = holder_avg_mds.pivot_table('avg_mds',cn_date,'tt').add_prefix('avgmd_').reset_index()

# Now apply for doctor count
tmp = dat_clin[['arrived','md_team']].assign(year=lambda x: x.arrived.dt.year, month=lambda x: x.arrived.dt.month, day=lambda x: x.arrived.dt.day, hour=lambda x: x.arrived.dt.hour).drop(columns=['arrived'])
tmp = hourly_tt[cn_date].merge(tmp[tmp.md_team.notnull()], 'left', cn_date)
tmp['md_team'] = tmp.md_team.fillna('').str.split(';')
tmp.index = tmp[cn_date].astype(str).assign(date=lambda x: pd.to_datetime(x.year+'-'+x.month+'-'+x.day+' '+x.hour+':00:00')).date
tmp = tmp.groupby('date').md_team.apply(lambda x: set(ljoin(x)))
holder = np.repeat(np.NaN,tmp.shape[0])
for ii in range(nlags,tmp.shape[0]):
    if ((ii+1) % 5000) == 0:
        print(ii+1)
    holder[ii] = len(set(ljoin(tmp.iloc[ii-10:ii])))
holder = pd.Series(holder, index=tmp.index).fillna(method='bfill').astype(int)
holder_u_mds = pd.DataFrame(holder,columns=['u_mds10h']).reset_index().assign(year=lambda x: x.date.dt.year, month=lambda x: x.date.dt.month, day=lambda x: x.date.dt.day, hour=lambda x: x.date.dt.hour).drop(columns=['date'])

# (v) Merge labs/DI with exiting hour range
dat_labs_flow = hourly_tt[cn_date].merge(dat_labs_flow, 'left', cn_date).fillna(method='ffill').fillna(method='bfill').astype(int)
dat_DI_flow = hourly_tt[cn_date].merge(dat_DI_flow, 'left', cn_date).fillna(method='ffill').fillna(method='bfill').astype(int)

########################################
# --- STEP 7: MERGE ALL DATA TYPES --- #

# X features
hourly_X = hourly_census.merge(hourly_tt.drop(columns='date'),'left',cn_date)
hourly_X = hourly_X.merge(holder_avg_mds,'left',cn_date).merge(holder_u_mds,'left',cn_date)
hourly_X = hourly_X.merge(holder_num,'left',cn_date).merge(hourly_fac,'left',cn_date)
hourly_X = hourly_X.merge(dat_labs_flow,'left',cn_date).merge(dat_DI_flow,'left',cn_date)
assert hourly_X.notnull().all().all()
# Create "lags" for the X features: t+0, t-1, ..., t-2
lags = np.arange(nlags+1)
cn_X = list(hourly_X.columns.drop(cn_date))
hourly_X = pd.concat([add_lags(hourly_X[cn_X], ll) for ll in lags], 1).set_index(pd.MultiIndex.from_frame(hourly_X[cn_date])).iloc[nlags:-nlags]

# y-labels
leads = -(np.arange(nlags)+1)
hourly_Y = pd.DataFrame(np.vstack([hourly_census[ylbl].shift(z).values for z in leads]).T).set_index(pd.MultiIndex.from_frame(hourly_census[cn_date]))
hourly_Y.columns = pd.MultiIndex.from_product([['y'],['lead_'+str(z) for z in np.abs(leads)]])
hourly_Y = hourly_Y.iloc[nlags:-nlags].astype(int)

# Merge data and save for later
df_lead_lags = hourly_Y.join(hourly_X)
df_lead_lags.to_csv(os.path.join(dir_flow, 'df_lead_lags.csv'),index=True)
