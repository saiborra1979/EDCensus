#!/usr/bin/env python
# coding: utf-8

# # process_rt.py

# In[23]:


# Script to process HeroAI data extracts
import os
import numpy as np
import pandas as pd
from funs_support import find_dir_olu, find_dir_rt, drop_zero_var

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_rt = find_dir_rt()


# In[151]:


di_cn = {'means of arrival':'arr_method', 'CTAS(Acuity)':'CTAS',
         'Care_Area':'care_area', 'Department':'department',
         'Disposition':'dispo', 'disposition_selected':'dispo_selected',
         'Arrival_Date_Time':'arrived', 'Depart_Date_Time':'discharged',
         'Length_of_stay':'LOS',
         'roomed_in_ed':'roomed',
         'Triage_1_start_time':'stime1', 'Triage_1_end_time':'etime1',
         'Triage_2_start_time':'stime2', 'Triage_2_end_time':'etime2'}


# In[152]:


#############################
# --- STEP 1: LOAD DATA --- #

# Existing output data
df_demo = pd.read_csv(os.path.join(dir_flow,'demo4flow.csv'))

# JSON real-time data
fn_rt = 'raw_head.json'
df_json = pd.read_json(os.path.join(dir_rt,fn_rt),lines=True)
df_json.rename(columns=di_cn, inplace=True)

##############################
# --- STEP 2: PARSE JSON --- #

# Process the dicionaries
df_json['_id'] = df_json['_id'].apply(lambda x: x['$oid'],1)
df_json['file_timestamp'] = df_json['file_timestamp'].apply(lambda x: x['$date'],1)
df_json['inserted_at'] = df_json['inserted_at'].apply(lambda x: x['$date'],1)

# The _id field should always be unique
assert not df_json._id.duplicated().any()

# Convert to datetime where relevant
cn_date_fmt1 = ['roomed', 'registration_complete', 'md_assign', 'PIA', 'dispo_selected']
df_json[cn_date_fmt1] = df_json[cn_date_fmt1].mask(df_json[cn_date_fmt1]=='')
df_json[cn_date_fmt1] = df_json[cn_date_fmt1].apply(lambda x: pd.to_datetime(x,format='%d/%m/%Y %H%M'),0)

cn_date_fmt2 = ['file_timestamp', 'inserted_at']
df_json[cn_date_fmt2] = df_json[cn_date_fmt2].apply(lambda x: x.str.replace('T','').str.split('.',1,True)[0])
df_json[cn_date_fmt2] = df_json[cn_date_fmt2].apply(lambda x: pd.to_datetime(x,format='%Y-%m-%d%H:%M:%S'))
# timestamp and inserted time should be 1:1
dat_timestamp = df_json.groupby(cn_date_fmt2).size().reset_index()
dat_timestamp.sort_values('file_timestamp',inplace=True)
dat_timestamp.reset_index(None, True, True)
dat_timestamp.rename(columns={0:'n'}, inplace=True)
assert not dat_timestamp.drop(columns='n').duplicated().any()

# Ensure file records are within 10 seconds of being sent
dsecs = (df_json.inserted_at - df_json.file_timestamp).dt.total_seconds()
print(dsecs.describe()[['min','max']])
assert dsecs.max() <= 10


# In[153]:


df_json.loc[0]


# In[ ]:


# Fields to investigate
# Dispostion
# Bed
# dispo_selected
# bed_requested
# bed_ready
# Length_of_stay


# In[119]:


dat_timestamp


# In[155]:


df_json.sort_values(['file_timestamp','arrived']).index


# In[168]:


#############################################
# --- STEP 3: CALCULATE FIELD STABILITY --- #

# What fields experience variation across patients?
u_count = df_json.groupby('CSN').apply(lambda x: x.apply(lambda z: z.unique().shape[0]))
u_count = u_count[(u_count == 1).all(0).reset_index().rename(columns={0:'n'}).query('n==False')['index']]


# In[179]:


# arr_method
# CTAS
# Bed


############################################
# --- STEP 3: CALCULATE ROLLING CENSUS --- #


cn_keep = cn_date_fmt2 + ['CSN','Bed','arrived','discharged','stime1','stime2','etime1','etime2',
                          'LOS','dispo','dispo_selected']
for ii, rr in qq.iterrows():
    print('~~~~~ ITERATION %i ~~~~~~' % (ii+1))


