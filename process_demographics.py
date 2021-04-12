# Load in the required modules
import os
import numpy as np
import pandas as pd

from funs_support import vhaversine, pc_extract, find_dir_olu

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_data = os.path.join(dir_olu, 'data')
dir_pulls = os.path.join(dir_olu, 'pulls')
dir_figures = os.path.join(dir_olu, 'figures')
dir_clin = os.path.join(dir_pulls, 'triage_clin')
lst_dir = [dir_data, dir_pulls, dir_clin, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
folders = [dir_output, dir_flow]
for dd in folders:
    if not os.path.exists(dd):
        print('making output directory: %s' % dd)
        os.mkdir(dd)

#############################
# --- STEP 1: LOAD DATA --- #

#'Care Area', MAY BE CONFOUNDED SO REMOVING!
# , 'patient_CC', 'nurse_CC',

print('STEP 1A: Loading clinical data')
cn_clin_use = \
    ['CSN', 'MRN', 'Arrived', 'Address',
     'Age at Visit', 'Gender',
     'Last Weight', 'Pref Language', 'CTAS', 'Arrival Method',
     'Current Medications',
     'Pulse', 'BP', 'Resp', 'Temp']
cn_clin_triage = \
    ['CSN', 'MRN', 'arrived', 'address',
     'age', 'sex', 'weight', 'language',
     'CTAS', 'arr_method',
     'meds', 'pulse', 'BP', 'resp', 'temp']
di_clin_map = dict(zip(cn_clin_use, cn_clin_triage))
print(pd.DataFrame({'orig': cn_clin_use, 'trans': cn_clin_triage}))
# possible? 'Triage Comp to PIA','Diagnosis','Arrival to Room'
fn_clin = os.listdir(dir_clin)
holder = []
for fn in fn_clin:
    tmp = pd.read_csv(os.path.join(dir_clin, fn), encoding='ISO-8859-1', usecols=cn_clin_use)
    tmp.rename(columns=di_clin_map, inplace=True)
    holder.append(tmp)
dat_clin = pd.concat(holder).reset_index(None, True)
del holder
print(dat_clin.head())

# Concert to datetime
dat_clin['arrived'] = pd.to_datetime(dat_clin.arrived.str.strip().str.replace('\\s', '-'),format='%d/%m/%y-%H%M')
# Drop rows if they are arrival time data
dat_clin = dat_clin[dat_clin.arrived.notnull()]
# Remove any duplicated patient visits
dat_clin = dat_clin[~dat_clin.CSN.duplicated()].reset_index(None,True)

#######################################
# --- STEP 2: PROCESS GEOGRAPHIES --- #
print('STEP 2: PROCESSING GEOGRAPHIES')

# Postalcode dataset provided by <geodata@geocoder.ca> under a non-for-profit licence
pc_lookup = pd.read_csv(os.path.join(dir_data, 'zipcodeset.txt'), usecols=[0, 1, 2])
pc_all = pc_lookup.PostCode
pc_lookup['PostCode5'] = pd.Series([x[0:5] for x in pc_all])

# Pull out the postal code info
prov_code = ['ON', 'AB', 'QC', 'BC', 'NL', 'NL', 'SK',
             'MB', 'NB', 'PE', 'YT', 'NU', 'NT']
prov_code = '|'.join(['\\s' + str(x) + '\\s' for x in prov_code])
ss_index = pd.Series([pc_extract(ss=x, pat=prov_code) for
                      x in dat_clin.address.fillna(value='')])
pc_vec = pd.Series([ss[idx:] for ss, idx in
                    zip(dat_clin.address.fillna(''), ss_index)])
# Remove any spaces and keep only
pc_vec = pc_vec.str.replace(' ', '')
pc_vec = pd.Series(np.where(pc_vec.str.len() == 6, pc_vec, ''))
assert len(pc_vec) == dat_clin.shape[0]
dat_clin['PostalCode'] = pc_vec
# Try match for all six
u_pcs = pd.Series(pc_vec[pc_vec != ''].unique())
u_pcs5 = pd.Series([x[0:5] for x in u_pcs])
# Find perfect matches
u_match1 = pd.DataFrame({'pc_sk': u_pcs}).merge(pc_lookup, left_on='pc_sk', right_on='PostCode')
# Find no match hits
diff_pcs = np.setdiff1d(u_pcs.to_list(), u_match1.pc_sk.to_list())
u_match2 = pd.DataFrame({'pc_sk': diff_pcs,
                         'pc_sk5': pd.Series([x[0:5] for x in diff_pcs])})
u_match2 = u_match2.merge(pc_lookup, left_on='pc_sk5', right_on='PostCode5')
# Average over lat/lonu
u_match2 = u_match2.groupby('pc_sk').mean().reset_index()
# Merge
u_match = pd.concat([u_match1.drop(columns=['PostCode', 'PostCode5']),
                     u_match2], axis=0)
dat_clin = dat_clin.merge(u_match, 'left', left_on='PostalCode', right_on='pc_sk')
# Calculate the distance to the hospital
lat_sk = 43.656988
lon_sk = -79.388015
dat_clin['DistSK'] = vhaversine(lat1=dat_clin.Latitude,
                lon1=dat_clin.Longitude, lat2=lat_sk, lon2=lon_sk)
dat_clin['DistSK'] = np.log(dat_clin['DistSK'])
dat_clin.drop(columns=['pc_sk', 'PostalCode', 'Latitude', 'Longitude', 'address'],
              inplace=True)
# # Fill missing with median
# dat_clin.DistSK.fillna(dat_clin.DistSK.median())

# Bin the log(Distance) to a categorical: 0-1, 1-2, 3-4, 4+, missing
assert dat_clin.DistSK.min() > -2.5
assert dat_clin.DistSK.max()+1 < 10
lvls = [-2.5, 1, 2, 3, 4, 10, 20]
lbls = ['<1','1_2','2_3','3_4','4+','missing']
dat_clin['DistSK'] = pd.cut(x=dat_clin.DistSK.fillna(15),bins=lvls,labels=lbls)

##################################################
# --- STEP 3: SIMPLE FEATURE TRANSFORMATIONS --- #
print('STEP 3: SIMPLE FEATURE TRANSFORMATIONS')

# Columns where NaNs can be filled with "missing"
cn_missing = ['language', 'CTAS', 'arr_method']
dat_clin[cn_missing] = dat_clin[cn_missing].fillna('missing')

# meds
tmp = dat_clin.meds.copy()
tmp = tmp.str.strip().str.replace('\\;$', '', regex=True)
tmp = tmp.str.count('\\;').fillna(0).astype(int)
dat_clin['num_meds'] = tmp
dat_clin.drop(columns=['meds'], inplace=True)

# weight
tmp = np.where(dat_clin.weight == 'None', np.NaN, dat_clin.weight)
tmp = pd.Series(tmp).str.replace('\\skg', '')
tmp = tmp.str.replace('\\([P!S]\\)\\s', '').astype(float)
dat_clin['weight'] = tmp

# pulse/resp/temp
dat_clin['resp'] = dat_clin.resp.str.replace('[^0-9]', '')
dat_clin['resp'] = np.where(dat_clin.resp == '', np.NaN,
                            dat_clin.resp).astype(float)
dat_clin['pulse'] = dat_clin.pulse.str.replace('[^0-9]', '')
dat_clin['pulse'] = np.where(dat_clin.pulse == '', np.NaN,
                            dat_clin.pulse).astype(float)
tmp = dat_clin.temp.str.replace('\\(.\\)\\s', '')
tmp = tmp.str.split('\\s', n=1, expand=True).iloc[:, 0]
tmp = tmp.str.replace('[^0-9\\.]', '')
dat_clin['temp'] = np.where(tmp == '', np.NaN, tmp).astype(float)

# force respitory rate and pulse to reasonable values
pulse, resp = dat_clin.pulse, dat_clin.resp
resp[~((resp > 0) & (resp < 80))] = np.NaN
pulse[~((pulse >= 20) & (pulse <= 220))] = np.NaN

# convert blood pressure to Systolic/Diastolic measures
tmp_sys_dis = dat_clin.BP.str.replace('\\(.\\)\\s','')
tmp_sys_dis = tmp_sys_dis.str.split('\\s', expand=True).iloc[:, 0]
tmp_sys_dis = tmp_sys_dis.str.split('\\/', expand=True)
tmp_sys_dis = tmp_sys_dis.apply(lambda x: x.str.replace('[^0-9]','') ,axis=0)
tmp_sys_dis = np.where(tmp_sys_dis == '', np.NaN, tmp_sys_dis).astype(float)
tmp_sys_dis = pd.DataFrame(tmp_sys_dis, columns=['systolic', 'diastolic'])
dat_clin = pd.concat([dat_clin, tmp_sys_dis], axis=1).drop(columns=['BP'])

# convert
age_type = dat_clin.age.str.replace('[0-9\\s]','')
age_num = dat_clin.age.str.replace('[^0-9]','').astype(int)
age_class = ['days', 'y.o.', 'm.o.', 'wk.o.']
assert all(age_type.isin(age_class))
age_num[age_type == 'days'] = age_num[age_type == 'days'] / 365
age_num[age_type == 'm.o.'] = age_num[age_type == 'm.o.'] / 12
age_num[age_type == 'wk.o.'] = age_num[age_type == 'wk.o'] / 52
dat_clin['age'] = age_num
# dat_clin = dat_clin[dat_clin.age <= 24].reset_index(None,True)

# Return visit
tmp = dat_clin[['CSN','MRN','arrived']].sort_values(['MRN','arrived']).reset_index(None,True)
tmp['lag_arrived'] = tmp.groupby('MRN').arrived.shift(+1)
tmp['dhours'] = (tmp.arrived - tmp.lag_arrived) / pd.Timedelta('1 minute') / 60
dat_clin = dat_clin.merge(tmp[['CSN','dhours']],'left','CSN')
dat_clin['ret72'] = (dat_clin.dhours < 72).astype(int)

# Aggregate sex
print(dat_clin.sex.value_counts())
dat_clin['sex'] = np.where(dat_clin.sex == 'U', np.random.choice(['M','F'],dat_clin.shape[0]),dat_clin.sex)
print(dat_clin.sex.value_counts())

# Fill missing numeric with means
cn_num = ['age','weight','pulse','resp','temp','num_meds','systolic','diastolic']
for cn in cn_num:
    if dat_clin[cn].isnull().any():
        dat_clin[cn] = dat_clin[cn].fillna(dat_clin[cn].median())

#################################################
# --- STEP 4: AGGREGATE CATEGORICAL FACTORS --- #
print('# --- STEP 4: AGGREGATE CATEGORICAL FACTORS --- #')

cn_fac = ['sex', 'language', 'CTAS', 'arr_method', 'DistSK']
assert dat_clin[cn_fac].notnull().all().all()

# Arrival method
dat_clin['arr_method'] = dat_clin.arr_method.str.split('\\s\\(',1,True).iloc[:,0]
di_arr = {'Ambulatory':'Ambulance', 'Land Ambulance':'Ambulance',
          'Transfer In':'Transfer', 'missing':'Other',
          'Air Ambulance':'Air', 'Air & Ground Ambulance':'Air',
          'Walk':'Walk', 'Car':'Car', 'Other':'Other',
          'Unknown':'Other', 'Helicopter':'Air',
          'Police':'Other', 'Bus':'Car', 'Taxi':'Car', 'Stretcher':'Other'}
print(np.unique(list(di_arr.values())))
assert dat_clin.arr_method.isin(list(di_arr)).all()
dat_clin['arr_method'] = dat_clin.arr_method.map(di_arr)

# Languages (chinsese only)
dat_clin['language'] = dat_clin.language.str.split('\\s\\-',1,True).iloc[:,0]
lang_prop = dat_clin.language.value_counts(True)
lang_keep = list(lang_prop[lang_prop>0.005].index)
dat_clin['language'] = np.where(dat_clin.language.isin(lang_keep), dat_clin.language, 'Other')

#############################
# --- STEP 5: SAVE DATA --- #
print('STEP 5: SAVE DATA')

# Check that there is data for each day
z1 = dat_clin.arrived.min()
z2 = dat_clin.arrived.max()
dfmt = '%Y-%m-%d'
# noinspection PyTypeChecker
dseq = np.arange(np.datetime64(z1.strftime(dfmt)),
          np.datetime64(z2.strftime(dfmt))+1).astype(str)
dact = pd.Series(dat_clin.arrived.dt.strftime('%Y-%m-%d').unique())
assert all(dact.isin(dseq))

cn_drop = ['arrived', 'dhours']
dat_clin.drop(columns = cn_drop, inplace=True)
# Write csv
dat_clin.to_csv(os.path.join(dir_flow,'demo4flow.csv'),index=False)
print('--- END OF SCRIPT ---')
