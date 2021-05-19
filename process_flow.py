"""
SCRIPT TO PERFORM RAW PROCESSING
NUMBER OF PATIENT AT EACH TIME POINT THAT A NEW PATIENT ARRIVES OR IS DISCHARGED
"""

# Load in the required modules
import os
import numpy as np
import pandas as pd
from funs_support import ljoin, date2ymdh, find_dir_olu, ymdh2date
from Mappings import DI_tags

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_data = os.path.join(dir_olu, 'data')
dir_pulls = os.path.join(dir_olu, 'pulls')
dir_clin = os.path.join(dir_pulls, 'triage_clin')
dir_labs = os.path.join(dir_pulls, 'labs')
dir_DI = os.path.join(dir_pulls, 'DI')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')

lst_dir = [dir_data, dir_pulls, dir_clin, dir_labs, dir_DI,dir_output, dir_flow]
assert all([os.path.exists(z) for z in lst_dir])

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
dat_clin = dat_clin.sort_values('arrived')

# Remove rows if missing
cn_nodrop = ['arrived', 'los_min', 'los_clock']
idx_drop = ~dat_clin[cn_nodrop].isnull().any(axis=1)
print('A total of %i rows had to be dropped (no arrived or LOS field)' % sum(~idx_drop))
dat_clin = dat_clin[idx_drop].reset_index(None, True)

# Drop duplicate CSNs
dat_clin = dat_clin[~dat_clin.CSN.duplicated()].reset_index(None, True)
assert ~dat_clin.CSN.duplicated().any()

# Ensure all days are found (note we substract 1 minute off incase someone comes in at midnight)
fmt = '%Y-%m-%d'
first_day = '2018-06-01'
dt_min = dat_clin.arrived.min().strftime(fmt)
dt_max = (dat_clin.arrived.max() - pd.DateOffset(minutes=1)).strftime(fmt)
dt_range = pd.date_range(dt_min, dt_max, freq='d').astype(str)
udt = [first_day] + list((dat_clin.arrived - pd.DateOffset(minutes=1)).dt.strftime(fmt).unique())
assert dt_range.isin(udt).all()
print('First day: %s, last day: %s' % (dt_min, dt_max))

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
dat_clin = dat_clin.merge(dat_demo, 'inner', ['CSN', 'MRN'])
dat_clin.drop(columns=['MRN', 'time2pia', 'los'], inplace=True)

###################################
# --- STEP 2: LOAD LAB ORDERS --- #

print('# --- STEP 2: LOAD LAB ORDERS --- #')
cn_labs = {'MRN': 'MRN', 'Order ID': 'order_id', 'Order Date': 'date',
           'Order Time': 'time', 'Order Name': 'name'}
fn_labs = os.listdir(dir_labs)
holder = []
for fn in fn_labs:
    tmp = pd.read_csv(os.path.join(dir_labs, fn), encoding='ISO-8859-1', usecols=list(cn_labs)).rename(
        columns=cn_labs)
    holder.append(tmp)
dat_labs = pd.concat(holder).reset_index(None, True)
del holder, tmp
dat_labs['order_time'] = pd.to_datetime(dat_labs.date + ' ' + dat_labs.time, format='%d/%m/%Y %I:%M:%S %p')
dat_labs.drop(columns=['date', 'time'], inplace=True)
u_labs = pd.DataFrame({'lab': dat_labs.name.unique()})
u_labs.to_csv(os.path.join(dir_flow, 'u_labs.csv'))
# Remove any duplicate rows
cn_sub = ['MRN', 'order_time', 'name']
dat_labs = dat_labs[~dat_labs[cn_sub].duplicated()].reset_index(None, True)
# old_labs = dat_labs.copy() # Can be used later for Hillary's analysis

# Ensure all days are found
udt = [first_day] + list((dat_labs.order_time - pd.DateOffset(minutes=1)).dt.strftime(fmt).unique())
print(dt_range[~dt_range.isin(udt)])
assert dt_range.isin(udt).all()
# Fast parse
dat_labs.name = dat_labs.name.str.split('\\s', 1, True).iloc[:, 0].str.lower()
freq_labs = dat_labs.name.value_counts(True)
top_labs = list(freq_labs[freq_labs > 0.01].index)
dat_labs.name = np.where(dat_labs.name.isin(top_labs), dat_labs.name, 'other')
# Put into wide form
dat_labs_wide = dat_labs.groupby(['order_time', 'name']).size().reset_index().pivot('order_time', 'name', 0).fillna(0).astype(int).add_prefix('labs_', ).reset_index()

##################################
# --- STEP 3: LOAD DI ORDERS --- #

print('# --- STEP 3: LOAD DO ORDERS --- #')
cn_DI = {'MRN': 'MRN', 'Order ID': 'order_id', 'Order Date': 'date',
         'Order Time': 'time', 'Order Name': 'name'}
fn_DI = os.listdir(dir_DI)
holder = []
for fn in fn_DI:
    tmp = pd.read_csv(os.path.join(dir_DI, fn), encoding='ISO-8859-1', usecols=list(cn_DI)).rename(
        columns=cn_DI)
    holder.append(tmp)
dat_DI = pd.concat(holder).reset_index(None, True)
del holder, tmp
dat_DI['order_time'] = pd.to_datetime(dat_DI.date + ' ' + dat_DI.time, format='%d/%m/%Y %I:%M:%S %p')
dat_DI.drop(columns=['date', 'time'], inplace=True)
dat_DI = dat_DI[~dat_DI[cn_sub].duplicated()].reset_index(None, True)
# old_DI = dat_DI.copy()

# Ensure all days are found
udt = [first_day] + list((dat_DI.order_time - pd.DateOffset(minutes=1)).dt.strftime(fmt).unique())
print(dt_range[~dt_range.isin(udt)])
assert dt_range.isin(udt).all()
# Fast parse
u_DI = pd.DataFrame({'DI': dat_DI.name.unique()})
u_DI.to_csv(os.path.join(dir_flow, 'u_DI.csv'))
tmp1 = u_DI.rename_axis('idx').reset_index()
tmp2 = u_DI.DI.map(DI_tags).explode().rename_axis('idx').reset_index().fillna('Missing')
u_DI = tmp1.merge(tmp2.rename(columns={'DI':'term'}),'left','idx').drop(columns='idx')
u_DI['idx'] = u_DI.groupby('DI').cumcount()
u_DI = u_DI.pivot('DI','idx','term').reset_index().fillna('Missing')
u_DI.rename(columns={'DI':'name',0:'DI',1:'organ'},inplace=True)
# Merge with existing data
dat_DI = dat_DI.merge(u_DI,'left','name')
dat_DI = dat_DI.drop(columns=['MRN', 'order_id', 'name']).sort_values('order_time').reset_index(None, True)
# Put into wide form
dat_DI_wide = dat_DI.groupby(['order_time', 'DI']).size().reset_index().pivot('order_time', 'DI', 0).fillna(0).astype(int).add_prefix('DI_', ).reset_index()
# Aggregate if too lother
di_tot = dat_DI_wide.sum(0)
di_low = list(di_tot[di_tot < 100].index)
dat_DI_wide['DI_other'] = dat_DI_wide[di_low].sum(1)
dat_DI_wide.drop(columns=di_low, inplace=True)

# dat_DI_wide = dat_DI_wide.pivot_table(0,['order_time'],['device','organ']).fillna(0).astype(int)
# dat_DI_wide.columns = dat_DI_wide.columns.to_frame().apply(lambda x: x[0]+'_'+x[1],1)

##########################################
# --- STEP 5: HOURLY FLOW ON LABS/DI --- #

cn_date = ['year', 'month', 'day', 'hour']

# Get year/month/day/hour
dat_DI_flow = pd.concat([dat_DI_wide, date2ymdh(dat_DI_wide.order_time)], 1).drop(columns=['order_time'])
dat_labs_flow = pd.concat([dat_labs_wide, date2ymdh(dat_labs_wide.order_time)], 1).drop(columns=['order_time'])
# Aggregate by hour
cn_DI = dat_DI_wide.columns.drop('order_time')
cn_labs = dat_labs_wide.columns.drop('order_time')
dat_DI_flow = dat_DI_flow.groupby(cn_date)[cn_DI].sum().reset_index()
dat_labs_flow = dat_labs_flow.groupby(cn_date)[cn_labs].sum().reset_index()

#######################################################
# --- STEP 6: FLOW ON ARRIVAL AND DISCHARGE TIMES --- #

cn_clin = dat_clin.columns.drop(['CSN', 'arrived', 'discharged'])
print('Clinical features: %s' % ', '.join(cn_clin))
# (iii) Columns that can be one-hot encoded
dt_clin = dat_clin[cn_clin].dtypes
cn_num = list(dt_clin[dt_clin != 'object'].index)
cn_fac = list(dt_clin[dt_clin == 'object'].index.drop('md_team'))
cn_idx = ['CSN'] + list(cn_clin)  # All features to index on
# Long-form
dat_long = dat_clin.melt(cn_idx, ['arrived', 'discharged'], 'tt', 'date').sort_values('date').reset_index(None,True)
dat_long = dat_long.assign(ticker=lambda x: np.where(x.tt == 'arrived', +1, -1))
# Actual patient count
dat_long.insert(0, 'census', dat_long.ticker.cumsum())
# Hourly results
dat_long = pd.concat([dat_long, date2ymdh(dat_long.date)], 1)
num_days = dat_long.groupby(cn_date[:-1]).size().shape[0]
print('There are %i total days in the dataset' % num_days)
# Get the last day/hour
mx_year, mx_month, mx_day, mx_hour = list(dat_long.tail(1)[cn_date].values.flatten())
mi_year, mi_month, mi_day, mi_hour = list(dat_long.head(1)[cn_date].values.flatten())
dt_max = pd.to_datetime(str(mx_year) + '-' + str(mx_month) + '-' + str(mx_day) + ' ' + str(mx_hour) + ':00:00')
dt_min = pd.to_datetime(str(mi_year) + '-' + str(mi_month) + '-' + str(mi_day) + ' ' + str(mi_hour) + ':00:00')
# # Create a one-day buffer
# dt_min = dt_min + pd.DateOffset(days=1)
dt_max = dt_max - pd.DateOffset(days=1)
print('Start date: %s, end date: %s' % (dt_min, dt_max))

# (i) Hourly arrival/discharge
hourly_tt = dat_long.groupby(cn_date + ['tt']).size().reset_index()
hourly_tt = hourly_tt.pivot_table(0, cn_date[:-1] + ['tt'], 'hour').fillna(0).reset_index().melt(cn_date[:-1] + ['tt'])
hourly_tt = hourly_tt.pivot_table('value', cn_date, 'tt').fillna(0).add_prefix('tt_').astype(int).reset_index()
hourly_tt = hourly_tt.assign(date=lambda x: pd.to_datetime(
    x.year.astype(str) + '-' + x.month.astype(str) + '-' + x.day.astype(str) + ' ' + x.hour.astype(str) + ':00:00'))
# Subset
hourly_tt = hourly_tt[(hourly_tt.date >= dt_min) & (hourly_tt.date <= dt_max)].reset_index(None, True)
# Long version for benchmarks
hourly_tt_long = hourly_tt.melt(['date'] + cn_date, ['tt_arrived', 'tt_discharged']).assign(
    tt=lambda x: x.tt.str.replace('tt_', '')).rename(columns={'value': 'census'})

# (ii) Census moments in that hour  ['max','var']
cn_moment = ['max', 'var']
di_cn = dict(zip(cn_moment, ['census_' + z for z in cn_moment]))
hourly_census = dat_long.pivot_table('census', cn_date[:-1], 'hour', cn_moment)
hourly_census = hourly_census.reset_index().melt(['year', 'month', 'day']).rename(columns={None: 'moment'})
assert hourly_census.shape[0] == num_days * 24 * len(cn_moment)
hourly_census = hourly_census.pivot_table('value', cn_date, 'moment', dropna=False).reset_index().rename(columns=di_cn)
# Subset the year/month/day counter
# Forward fill any census values (i.e. if there are 22 patients at hour 1, and no one arrives/leaves at hour 2, then there are 22 patients at hour 2
hourly_census = hourly_tt[cn_date].merge(hourly_census, 'left', cn_date)
hourly_census = hourly_census.fillna(method='ffill').fillna(0)
hourly_census.census_max = hourly_census.census_max.astype(int)

holder = []
for ii, cn in enumerate(cn_fac):
    print('column: %s (%i of %i)' % (cn, ii + 1, len(cn_fac)))
    # Get into long-format
    tmp = dat_long.groupby(cn_date + ['tt', cn]).size().reset_index()
    tmp = tmp.pivot_table(0, cn_date[:-1], ['tt', 'hour', cn])
    tmp = tmp.fillna(0).astype(int).reset_index().melt(cn_date[:-1])
    tmp = tmp.sort_values([cn] + cn_date + ['tt']).reset_index(None, True)
    # Get share at each hour
    tmp = tmp.pivot_table('value', cn_date + ['tt'], cn)
    cn_tmp = tmp.columns
    # Factor should exactly equal arrival/discharge
    qq = tmp.sum(1).reset_index().rename(columns={0: 'n'}).merge(hourly_tt_long, 'right', cn_date + ['tt']).drop(columns=['date'])
    assert np.all(qq.n == qq.census)
    # CALCULATE AS A PERCENTAGE
    tmp = tmp.merge(hourly_tt_long, 'right', cn_date + ['tt'])
    tmp_census = tmp.census.copy()
    tmp = pd.concat([tmp.drop(columns=list(cn_tmp)+['census']),
                     tmp[cn_tmp].divide(tmp_census.clip(1,tmp_census.max()),axis=0)],1)
    # Re-expand into wide form
    tmp = tmp.reset_index(None,True).pivot_table(tmp.columns, cn_date, 'tt')
    tmp.columns = pd.DataFrame(tmp.columns.to_frame().values).apply(lambda x: cn + '_' + x[0] + '_' + x[1], axis=1).to_list()
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
holder_num = hourly_tt_long[cn_date + ['tt']].merge(holder_num, 'left', cn_date + ['tt'])
holder_num[cn_num] = holder_num.groupby('tt')[cn_num].fillna(method='ffill').fillna(0)
holder_num = holder_num.pivot_table(cn_num, cn_date, 'tt')
holder_num.columns = holder_num.columns.to_frame().apply(lambda x: x[0] + '_' + x['tt'], 1).to_list()
holder_num.reset_index(inplace=True)

# (v) MD team
num_mds = (dat_long.md_team.str.count(';') + 1).fillna(0).astype(int)
holder_avg_mds = dat_long.assign(avg_mds=num_mds).groupby(cn_date + ['tt']).avg_mds.mean().reset_index()
# Merge and widen
holder_avg_mds = hourly_tt_long[cn_date + ['tt']].merge(holder_avg_mds, 'left', cn_date + ['tt'])
holder_avg_mds['avg_mds'] = holder_avg_mds.groupby('tt').avg_mds.fillna(method='ffill').fillna(0)
holder_avg_mds = holder_avg_mds.pivot_table('avg_mds', cn_date, 'tt').add_prefix('avgmd_').reset_index()

# Now apply for doctor count
tmp = dat_clin[['arrived', 'md_team']].copy()
tmp = pd.concat([tmp,date2ymdh(tmp.arrived)],1).drop(columns=['arrived'])
tmp = hourly_tt[cn_date].merge(tmp[tmp.md_team.notnull()], 'left', cn_date)
tmp['md_team'] = tmp.md_team.fillna('').str.split(';')
tmp.index = tmp[cn_date].astype(str).assign(
    date=lambda x: pd.to_datetime(x.year + '-' + x.month + '-' + x.day + ' ' + x.hour + ':00:00')).date
tmp = tmp.groupby('date').md_team.apply(lambda x: set(ljoin(x)))
holder = np.repeat(np.NaN, tmp.shape[0])
nhours = 10
for ii in range(nhours, tmp.shape[0]):
    if ((ii + 1) % 5000) == 0:
        print(ii + 1)
    # Remove no doctors from the set
    holder[ii] = len(set(ljoin(tmp.iloc[ii - nhours:ii])) - {''})
holder = pd.Series(holder, index=tmp.index)
assert holder.isnull().sum() == nhours
holder = holder.fillna(0).astype(int)
holder_u_mds = pd.DataFrame(holder, columns=['u_mds10h']).reset_index()
holder_u_mds = pd.concat([holder_u_mds,date2ymdh(holder_u_mds.date)],1).drop(columns=['date'])

# (v) Merge labs/DI with exiting hour range
dat_labs_flow = hourly_tt[cn_date].merge(dat_labs_flow, 'left', cn_date).fillna(0).astype(int)
dat_DI_flow = hourly_tt[cn_date].merge(dat_DI_flow, 'left', cn_date).fillna(0).astype(int)

# # (vi) Full DI/raw labs for Hillary
# q1 = date2ymdh(DI.order_time).assign(name=DI.name).groupby(cn_date + ['name']).size().reset_index().pivot_table(0,cn_date,'name').fillna(0).astype(int).reset_index()
# q1 = hourly_census.drop(columns=['census_var']).merge(q1, 'left', cn_date).fillna(0).astype(int)
# q1.to_csv(os.path.join(dir_flow, 'all_DI.csv'), index=True)
# q2 = date2ymdh(labs.order_time).assign(name=labs.name).groupby(cn_date + ['name']).size().reset_index().pivot_table(0,cn_date,'name').fillna(0).astype(int).reset_index()
# q2 = hourly_census.drop(columns=['census_var']).merge(q2, 'left', cn_date).fillna(0).astype(int)
# q2.to_csv(os.path.join(dir_flow, 'all_labs.csv'), index=True)

########################################
# --- STEP 7: MERGE ALL DATA TYPES --- #

# All features/labels
hourly_yX = hourly_census.merge(hourly_tt.drop(columns='date'), 'left', cn_date)
hourly_yX = hourly_yX.merge(holder_avg_mds, 'left', cn_date).merge(holder_u_mds, 'left', cn_date)
hourly_yX = hourly_yX.merge(holder_num, 'left', cn_date).merge(hourly_fac, 'left', cn_date)
hourly_yX = hourly_yX.merge(dat_labs_flow, 'left', cn_date).merge(dat_DI_flow, 'left', cn_date)
assert hourly_yX.notnull().all().all()
# Add on day_of_week and date
hourly_yX.insert(0,'date',ymdh2date(hourly_yX[cn_date]))
hourly_yX.insert(1,'dow',hourly_yX.date.dt.dayofweek)

# Save for later
hourly_yX.to_csv(os.path.join(dir_flow, 'hourly_yX.csv'), index=False)
print('End of process_flow.py script')
