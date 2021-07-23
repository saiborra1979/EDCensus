# Calls class from ~/mdls folder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_list', nargs='+', help='Model classes to evaluate (xgboost lasso)')
args = parser.parse_args()
model_list = args.model_list
print(model_list)
assert isinstance(model_list, list)

model_list = ['gp_stacker']

# Load modules
import os
import pandas as pd
import numpy as np
from funs_support import find_dir_olu, read_pickle, drop_zero_var

# Set up folders
dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
lst_dir = [dir_figures, dir_output, dir_flow, dir_test]
assert all([os.path.exists(z) for z in lst_dir])

# Get the model list
fn_test = os.listdir(dir_test)

assert all([mdl in fn_test for mdl in model_list])

idx = pd.IndexSlice


##################################
# --- (1) MERGE AGG BY MONTH --- #

holder = []
for model in model_list:
    print('Model: %s' % model)
    dir_model = os.path.join(dir_test, model)
    fn_model = pd.Series(os.listdir(dir_model))
    fn_model = fn_model[fn_model.str.contains('\\.pickle$')]
    holder_model = []
    for fn in fn_model:
        path_fn = os.path.join(dir_model, fn)
        df_hp, df_agg = read_pickle(path_fn).values()
        # Multicolumn
        cn_agg = pd.MultiIndex.from_product([['base'],df_agg.columns])
        cn_hp = pd.MultiIndex.from_frame(df_hp.drop(columns='val'))
        df_agg.columns = cn_agg
        df_hp = df_hp.drop(columns=['tt','hp']).T.values
        df_hp = pd.DataFrame(np.tile(df_hp,[len(df_agg),1]),columns=cn_hp)
        df_agg = pd.concat([df_agg,df_hp],1)
        holder_model.append(df_agg)
    # Merge model
    res_agg = pd.concat(holder_model).reset_index(None, True)
    res_agg = drop_zero_var(res_agg)
    # Add model to the base column
    res_agg.insert(0,'model',model)
    res_agg.columns = res_agg.columns.insert(0,('base','model')).drop(('model',''))    
    holder.append(res_agg)
    del holder_model, res_agg
# Merge all
res_all = pd.concat(holder).reset_index(None, True)
res_all.to_csv(os.path.join(dir_test, 'res_merge_qsub.csv'), index=False)


####################################
# --- (2) MISSING PERMUTATIONS --- #

cn_int = ['lead','month','dtrain','h_rtrain','nval']
res_all.columns = res_all.columns.droplevel(0)
res_all[cn_int] = res_all[cn_int].astype(int)

# Find the missing values

print('~~~ END OF merge_qsub.py ~~~')