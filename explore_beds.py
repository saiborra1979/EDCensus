"""
SCRIPT TO CHECK BEDS DATA AND OTHER POTENTIAL DATA SOURCES
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from funs_support import find_dir_olu

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_beds = os.path.join(dir_olu, 'beds')
dir_pulls = os.path.join(dir_olu, 'pulls')
dir_notes = os.path.join(dir_pulls, 'triage_notes')

#########################
# --- (1) LOAD DATA --- #

fn_beds = pd.Series(os.listdir(dir_beds))
cn_from = ['Room','Bed','Pt Name or Bed Status','Sex','Age','Isolation']
cn_to = ['room','bed','name','sex','age','iso']

holder = []
for fn in fn_beds:
    tmp = pd.read_csv(os.path.join(dir_beds,fn),usecols=cn_from)
    day, hour = fn.split('_')[1], fn.split('_')[2].split('.')[0]
    tmp.insert(0,'date',pd.to_datetime(day + ' ' + hour,format='%Y%m%d %H%M'))
    holder.append(tmp)
df = pd.concat(holder).reset_index(None,True).rename(columns=dict(zip(cn_from,cn_to)))
# Check that number of missing values is equal for every query
cn_gg = ['room','bed','name']
dat_n = df.assign(name=lambda x: x.name.isnull()).groupby(cn_gg+['date']).size().reset_index().rename(columns={0:'n'})
assert np.all(dat_n.groupby(cn_gg).n.var()==0)
# If successful, then missingness can be dropped
df = df.query('name.notnull()').reset_index(None, True)
assert np.all(df.date.value_counts().var() == 0)
# There are four statuses: Ready, Dity, Cleaning, and Maintenance
lst_status = ['Ready', 'Dirty', 'Cleaning', 'Maintenance']
df = df.assign(status=lambda x: np.where(x.name.isin(lst_status),x.name,'Occupied'))
df = df.assign(name = lambda x: np.where(x.name.isin(lst_status),np.NaN,x.name))
df = df.assign(is_iso = lambda x: x.iso.notnull())
# Variations over time
print(df.groupby(['status','date']).size())

###########################
# --- (2) ROOM VALUES --- #

df.bed = df.bed + '_' + df.groupby(['date','room']).cumcount().astype(str)
dat_n_bed = df.pivot_table('date',['bed'],'status','count').fillna(0).astype(int)
dat_n_bed = dat_n_bed.reset_index().melt('bed',None,'status','n')
n_bed_occ = dat_n_bed.query('status == "Occupied" & n>0').shape[0]
n_bed_nocc = dat_n_bed.query('status == "Occupied" & n==0').shape[0]
print('%i of %i unique beds have had a patient' % (n_bed_occ, n_bed_occ+n_bed_nocc))

print(df.groupby(['date','status','is_iso']).size())

##############################
# --- (3) PATIENT CHECKS --- #

print(pd.Series(df.date.unique()))

cn_from = ['Patient Name', 'File Time']
cn_to = ['name', 'time']
di_clin_map = dict(zip(cn_from, cn_to))
print(pd.DataFrame({'orig': cn_from, 'trans': cn_to}))
fn_clin = os.listdir(dir_notes)
holder = []
for fn in fn_clin:
    tmp = pd.read_csv(os.path.join(dir_notes, fn), encoding='ISO-8859-1',  usecols=cn_from).rename(columns=di_clin_map)
    holder.append(tmp)
dat_clin = pd.concat(holder).reset_index(None, True)
dat_clin.time = pd.to_datetime(dat_clin.time,format='%d/%m/%y %H%M')

def first2names(x):
    tmp = x.str.split('\\,\\s',1,True).rename(columns={0:'f',1:'l'})
    last = tmp.l.str.split('\\s',1,True).iloc[:,0]
    return tmp.f + ', ' + last

u_names = pd.Series(df.query('date<"2021-03-01"').name.unique())
u_names = u_names[u_names.notnull()].reset_index(None, True)
all_names = pd.Series(dat_clin.name.unique())
u_names = pd.Series(first2names(u_names).unique())
all_names = pd.Series(first2names(all_names).unique())
assert len(np.setdiff1d(u_names,all_names)) == 0

dat_sub = dat_clin.assign(name=lambda x: first2names(x.name)).query('name.isin(@u_names)')
dat_sub = dat_sub.sort_values(['name','time']).reset_index(None, True)
print(dat_sub.groupby('name').time.max().describe()[['first','last']])




