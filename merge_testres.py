"""
SCRIPT TO QUICKLY AGGREGATE THE FILES IN THE FLOW/TEST FOLDER
"""

import os
import pandas as pd
from sklearn.metrics import r2_score as r2
from funs_support import date2ymd

dir_base = os.getcwd()
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
# Get the dates associated with the test output
dates = pd.Series(os.listdir(dir_test))
dates = dates[dates.str.contains('^[0-9]{4}')].reset_index(None,True)
print(dates)

##################################
# --- STEP 1: AGGREGATE DATA --- #

cn_drop = 'Unnamed: 0'
holder = []
for date in dates:
    print('Date: %s' % date)
    fold = os.path.join(dir_test, date)
    files = pd.Series(os.listdir(fold))
    if len(files) > 0:
        files = files[files.str.contains('^res_')].reset_index(None, True)
        holder_mdl = []
        for file in files:
            tmp = pd.read_csv(os.path.join(fold, file))
            if cn_drop in tmp.columns:
                tmp.drop(columns=cn_drop, inplace=True)
            if 'date' in tmp.columns:
                tmp.rename(columns={'date': 'dates'}, inplace=True)
                tmp.dates = pd.to_datetime(tmp.dates)
                tmp = pd.concat([date2ymd(tmp.dates), tmp], 1)
            holder_mdl.append(tmp)
        res = pd.concat(holder_mdl)
        holder.append(res)
    else:
        print('No files in this folder')

res_model = pd.concat(holder).reset_index(None, True)
print(res_model.isnull().sum(0))

print(res_model.groupby(['model','lead','year','month','day']).apply(lambda x: r2(x.y, x.pred)))
res_model.to_csv(os.path.join(dir_flow, 'res_model.csv'),index=False)

