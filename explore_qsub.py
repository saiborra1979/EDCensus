"""
SCRIPT TO ANALYZE HOW THE GPU RUNS FAILED
"""

import os
import pandas as pd
import numpy as np

dir_base = os.getcwd()
dir_qsub = os.path.join(dir_base, '..', 'qsub')
assert os.path.exists(dir_qsub)

# Get the GPU runs
fn_gpu = pd.Series(os.listdir(dir_qsub))
fn_gpu = fn_gpu[fn_gpu.str.contains('^gpu')]
# Check that we have equal number of leads
assert (fn_gpu.str.split('-', 1, True).iloc[:, 1].value_counts().value_counts().shape[0] == 1)
# Subset to "error" only scripts
fn_gpu = fn_gpu[fn_gpu.str.contains('\\.e')].reset_index(None, True)
# Put into dataframe with meta info
df_gpu = pd.DataFrame({'fn': fn_gpu,
                       'tt': fn_gpu.str.split('\\_|\\.', 2, True).iloc[:, 1],
                       'array': fn_gpu.str.split('-', 1, True).iloc[:, 1],
                       'jobid': fn_gpu.str.split('\\.e|\\-', 2, True).iloc[:, 1]})
cn_int = ['array','jobid']
df_gpu[cn_int] = df_gpu[cn_int].astype(int)

# Loop through and parse
holder = []
for ii, rr in df_gpu.iterrows():
    fn = rr['fn']
    # Read in file
    with open(os.path.join(dir_qsub, fn), 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
    holder.append(last_line)
errors = pd.Series(holder)

tmp = errors.str.split('\\/[0-9]|python',2,True).iloc[:,0:2]
tmp = tmp.iloc[:,0] + ' ' + tmp.iloc[:,1].str.replace('[0-9]*\\-[0-9]{1,2}','jobid')
tmp = tmp.str.replace('\\:\\s{1,}[0-9]+',':').str.split('\\s{2,}',1,True).iloc[:,0]
errors = pd.Series(np.where(errors.str.contains('mom_priv'), tmp, errors))
errors = errors.str.replace('vmem\\s[0-9]+','vmem #')
errors = errors.str.strip()
# Combine
df_gpu = df_gpu.assign(error=errors).sort_values(['jobid','array']).reset_index(None,True)
df_gpu.to_csv(os.path.join(dir_qsub,'errors.csv'),index=False)
