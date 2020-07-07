"""
SCRIPT TO PERFORM RAW PROCESSING
NUMBER OF PATIENT AT EACH TIME POINT THAT A NEW PATIENT ARRIVES OR IS DISCHARGED
"""

import sys

if sys.stdout.isatty():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bfreq', type=str, default='1 hour', help='Time binning frequency')
    parser.add_argument('--ylbl', type=str, default='census_max', help='Target label')
    parser.add_argument('--nlags', type=int, default=10, help='Max number of lags')
    args = parser.parse_args()
    bfreq = args.bfreq
    ylbl = args.ylbl
    nlags = args.nlags
    print(args)
else:  # Debugging in PyCharm
    bfreq = '1 hour'
    ylbl = 'census_max'
    nlags = 10

# sys.exit('done argparse')

# Load in the required modules
import os
import numpy as np
import pandas as pd

dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '..', 'data')
dir_pulls = os.path.join(dir_base, '..', 'pulls')
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_clin = os.path.join(dir_pulls, 'triage_clin')
lst_dir = [dir_data, dir_pulls, dir_clin]
assert all([os.path.exists(z) for z in lst_dir])
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
folders = [dir_output, dir_flow]
for dd in folders:
    if not os.path.exists(dd):
        print('making output directory: %s' % dd)
        os.mkdir(dd)

#############################
# --- STEP 1: LOAD DATA --- #

print('STEP 1A: Loading Flow variables')
cn_clin_use = ['CSN', 'MRN', 'CTAS', 'Arrived', 'Triage Comp to PIA', 'Treatment Team',
               'ED Completed Length of Stay (Minutes)', 'LOS']
cn_clin_triage = ['CSN', 'MRN', 'CTAS', 'arrived', 'time2pia', 'md_team', 'los_min', 'los_clock']
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

# Remove rows if missing
cn_nodrop = ['arrived', 'los_min', 'los_clock']
idx_drop = ~dat_clin[cn_nodrop].isnull().any(axis=1)
print('A total of %i rows had to be dropped (no arrived or LOS field)' % sum(~idx_drop))
dat_clin = dat_clin[idx_drop].reset_index(None, True)

######################################
# --- STEP 2: FEATURE TRANSFORMS --- #

dat_clin['arrived'] = dat_clin.arrived.str.strip().str.replace('\\s', '-')
dat_clin['arrived'] = pd.to_datetime(dat_clin.arrived, format='%d/%m/%y-%H%M')
dat_clin = dat_clin.sort_values('arrived').reset_index(None, True)

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
mds = mds.str.split('\\;')  # For counting below
dat_clin['md_team'] = mds.copy()

# Merge with pre-processed clinical data
dat_clin['CTAS'] = dat_clin.CTAS.fillna('missing').astype(str).str[0:1]

#######################################################
# --- STEP 3: FLOW ON ARRIVAL AND DISCHARGE TIMES --- #

idx = pd.IndexSlice

cn_date = ['year', 'month', 'day', 'hour']
cn_idx = ['CSN', 'CTAS']  # All features to index on
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

# (ii) Census moments in that hour  ['max','min','var']
cn_moment = ['min','max','mean']
di_cn = dict(zip(cn_moment,['census_'+z for z in cn_moment]))
hourly_census = dat_long.pivot_table('census', cn_date[:-1], 'hour', cn_moment)
hourly_census = hourly_census.reset_index().melt(['year','month','day']).rename(columns={None:'moment'})
assert hourly_census.shape[0] == num_days*24*len(cn_moment)
hourly_census = hourly_census.pivot_table('value',cn_date,'moment',dropna=False).reset_index().rename(columns=di_cn)
# Subset the year/month/day counter
# Forward fill any census values (i.e. if there are 22 patients at hour 1, and no one arrives/leaves at hour 2, then there are 22 patients at hour 2
hourly_census = hourly_tt[cn_date].merge(hourly_census,'left',cn_date).fillna(method='ffill')
hourly_census[['census_min', 'census_max']] = hourly_census[['census_min', 'census_max']].astype(int)

# Merge
hourly_both = hourly_census.merge(hourly_tt,'left',cn_date).drop(columns=['date'])

# (iii) Columns that can be one-hot encoded
cn_fac = ['CTAS']

holder = []
for ii, cn in enumerate(cn_fac):
    print('column: %s (%i of %i)' % (cn, ii + 1, len(cn_fac)))
    # Get into long-format
    tmp = dat_long.groupby(cn_date + ['tt', cn]).size().reset_index()
    tmp = tmp.pivot_table(0,cn_date[:-1],['tt','hour',cn]).fillna(0).astype(int).reset_index().melt(cn_date[:-1])
    tmp = tmp.sort_values([cn] + cn_date + ['tt']).reset_index(None, True)
    # Get share at each hour
    tmp = tmp.pivot_table('value', cn_date + ['tt'], 'CTAS')
    # Factor should exactly equal arrival/discharge
    qq = tmp.sum(1).reset_index().rename(columns={0:'n'}).merge(hourly_tt_long,'right',cn_date+['tt']).drop(columns=['date'])
    assert np.all(qq.n == qq.census)
    den = np.atleast_2d(tmp.sum(1).values).T
    vals = tmp.values / den
    vals[np.where(np.isnan(vals))] = 1 / tmp.columns.shape[0]
    tmp = pd.DataFrame(vals, columns=tmp.columns, index=tmp.index)
    # Re-expand into wide form
    tmp = tmp.reset_index().pivot_table(tmp.columns,cn_date,'tt')
    tmp.columns = tmp.columns.to_frame().apply(lambda x: cn + '_' + x[0] + '_' + x[1], 1).to_list()

    # # ! only for cumulative counts ! #
    # tmp['value'] = np.where(tmp.tt == 'arrived', tmp.value, -tmp.value)
    # tmp['ticker'] = tmp.groupby(cn).value.cumsum()
    # tmp = tmp.pivot_table('ticker', cn_date + ['tt'], 'CTAS')
    # # The factors per hour should be between the min/max census
    # check1 = tmp.pivot_table('ticker',cn_date,'CTAS').add_prefix('CTAS_')
    # check1 = check1.sum(1).reset_index().rename(columns={0:'n'}).merge(hourly_census)
    # assert np.all((check1.n >= check1.census_min) & ((check1.n <= check1.census_max)))
    # .pivot_table('ticker',cn_date,'CTAS').add_prefix('CTAS_')

    # Merge with existing date
    tmp = hourly_tt[cn_date].merge(tmp.reset_index(), 'left', cn_date)
    holder.append(tmp)
    del tmp
hourly_fac = pd.concat(holder, 1)#.reset_index()
cn_fac = list(hourly_fac.columns.drop(cn_date))

# from plotnine import *
# gg = (ggplot(hourly_fac.melt(cn_date),aes(x='value')) +
#       geom_histogram(bins=50) +
#       facet_wrap('~variable') + theme_bw())
# gg.save(os.path.join(dir_figures,'dist.png'))

# (iii) Columns with numerical features (mean, var)


###############################
# --- STEP 4: CREATE LAGS --- #

def add_lags(df,l):
    tmp = df.shift(l)
    tmp.columns = pd.MultiIndex.from_product([tmp.columns,['lag_'+str(l)]])
    return tmp

# (i) Create "leads": t+1, t+2, ... for the label
leads = -(np.arange(nlags)+1)
df_y = pd.DataFrame(np.vstack([hourly_census[ylbl].shift(z).values for z in leads]).T).set_index(pd.MultiIndex.from_frame(hourly_census[cn_date]))
df_y.columns = pd.MultiIndex.from_product([['y'],['lead_'+str(z) for z in np.abs(leads)]])
df_y = df_y.iloc[:-nlags].astype(int)

# (ii) Create "lags" for the X features: t+0, t-1, ..., t-2
lags = np.arange(nlags+1)
df_X = pd.concat([add_lags(hourly_fac[cn_fac], ll) for ll in lags], 1)
df_X = df_X.set_index(pd.MultiIndex.from_frame(hourly_fac[cn_date])).iloc[nlags:].astype(int)

# (iii) Create lags for the arrival/discharge values
df_flow = pd.concat([add_lags(hourly_tt[['tt_arrived','tt_discharged']], ll) for ll in lags], 1)
df_flow = df_flow.set_index(pd.MultiIndex.from_frame(hourly_tt[cn_date])).iloc[nlags:].astype(int)

# (iv) Create lags for label (max number)
df_ylags = pd.concat([add_lags(hourly_census[[ylbl]], ll) for ll in lags], 1)
df_ylags = df_ylags.set_index(pd.MultiIndex.from_frame(hourly_census[cn_date])).iloc[nlags:].astype(int)
df_ylags.columns = df_ylags.columns.set_levels(['y'],0)

# Merge data and save for later
df_lead_lags = df_y.merge(df_ylags,'inner',cn_date).merge(df_flow,'inner',cn_date).merge(df_X,'inner',cn_date)
df_lead_lags.to_csv(os.path.join(dir_output, 'df_lead_lags.csv'),index=True)
