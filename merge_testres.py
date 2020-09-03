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

# fn_lasso = fn_test[fn_test.str.contains('lasso_res')].reset_index(None, True)
# fn_bhat = fn_test[fn_test.str.contains('lasso_bhat')].reset_index(None, True)
# # Get the model types
# models = list(fn_test.str.split('_',3,True).iloc[:,1].unique())
# print(models)

##################################
# --- STEP 1: AGGREGATE DATA --- #

cn_drop = 'Unnamed: 0'
holder = []
for date in dates:
    print('Date: %s' % date)
    fold = os.path.join(dir_test, date)
    files = pd.Series(os.listdir(fold))
    files = files[files.str.contains('^res_')].reset_index(None, True)
    holder_mdl = []
    if len(files) > 0:
        for file in files:
            tmp = pd.read_csv(os.path.join(fold, file))
            if cn_drop in tmp.columns:
                tmp.drop(columns=cn_drop, inplace=True)
            if 'date' in tmp.columns:
                tmp.rename(columns={'date': 'dates'}, inplace=True)
                tmp.dates = pd.to_datetime(tmp.dates)
                tmp = pd.concat([date2ymd(tmp.dates), tmp], 1)
            holder_mdl.append(tmp)
    else:
        print('No files in this folder')
    res = pd.concat(holder_mdl)
    holder.append(res)
    # path = os.path.join(dir_flow,'res_'+model+'.csv')
    # res.to_csv(path,index=False)
    # print(res.columns)

res_model = pd.concat(holder).reset_index(None, True)
print(res_model.isnull().sum(0))

print(res_model.groupby(['model','lead','year','month','day']).apply(lambda x: r2(x.y, x.pred)))
res_model.to_csv(os.path.join(dir_flow, 'res_model.csv'),index=False)


# # Get coefficients
# holder = []
# for fn in fn_bhat:
#     holder.append(pd.read_csv(os.path.join(dir_test, fn)))
# bhat_lasso = pd.concat(holder).reset_index(None,True)
# if 'Unnamed: 0' in bhat_lasso.columns:
#     bhat_lasso.drop(columns = ['Unnamed: 0'],inplace=True)
# print(bhat_lasso.groupby(['lead','lag','day']).apply(lambda x: (x.bhat != 0).mean()))
# bhat_lasso.to_csv(os.path.join(dir_flow, 'bhat_lasso.csv'))