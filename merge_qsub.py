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
from funs_support import find_dir_olu, find_zero_var

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

##################################
# --- (1) MERGE AGG BY MONTH --- #

holder = []
for model in model_list:
    print('Model: %s' % model)
    path_model = os.path.join(dir_test, model)
    path_model_agg = os.path.join(path_model, 'agg')
    fn_model_agg = pd.Series(os.listdir(path_model_agg))
    fn_model_agg = fn_model_agg[fn_model_agg.str.contains('\\.csv$')]
    # Split into the different categories
    dat_fn = fn_model_agg.str.split('\\+',1,True).rename(columns={0:'month',1:'hp'})
    dat_fn = dat_fn[dat_fn.month.str.contains('^month')].reset_index(None, True)
    # Specific hyperparameters
    hp_fn = pd.Series(dat_fn.hp.unique())
    print('Model %s has %i hyperparameter configurations' % (model, len(hp_fn)))
    for hp in hp_fn:
        u_months = pd.Series(dat_fn.query('hp==@hp').month.unique())
        tmp_fn = u_months + '+' + hp
        tmp_holder_agg = []
        for j, fn in enumerate(tmp_fn):
            tmp_agg = pd.read_csv(os.path.join(path_model_agg,fn))
            tmp_agg.insert(0,'month',u_months[j])
            tmp_holder_agg.append(tmp_agg)
        # Merge and overwrite
        tmp_agg = pd.concat(tmp_holder_agg).reset_index(None, True)
        # Calculate the average over months
        cn_z = find_zero_var(tmp_agg)
        cn_gg = cn_z + ['lead','metric']
        tmp_agg.drop(columns='model_args').loc[0]
        tmp_agg = tmp_agg.groupby(cn_gg).apply(lambda x:  
            pd.Series({'value':x.value.mean(),'se':x.se.mean(),'n':x.n.sum()})).reset_index()
        # Recalculate the upper/lower bound

        # Process the model args

        # Add on hp/model
        holder.append(tmp_agg)


############################
# --- (2) LOAD ORD/REG --- #

holder = []
for model in model_list:
    print('Model: %s' % model)
    path_model = os.path.join(dir_test, model)
    path_model_ord = os.path.join(path_model, 'ord')
    path_model_reg = os.path.join(path_model, 'reg')
    assert os.path.exists(path_model_ord) & os.path.exists(path_model_reg)
    fn_model_ord = pd.Series(os.listdir(path_model_ord))
    fn_model_reg = pd.Series(os.listdir(path_model_reg))
    # Find the overlap
    fn_model = pd.Series(np.intersect1d(fn_model_reg, fn_model_ord))
    fn_model = fn_model[fn_model.str.contains('\\.csv')].reset_index(None,True)
    # Split into the different categories
    dat_fn = fn_model.str.split('\\+',1,True).rename(columns={0:'month',1:'hp'})
    dat_fn = dat_fn[dat_fn.month.str.contains('^month')].reset_index(None, True)
    # Specific hyperparameters
    hp_fn = pd.Series(dat_fn.hp.unique())
    print('Model %s has %i hyperparameter configurations' % (model, len(hp_fn)))
    # Load by hp
    for hp in hp_fn:
        u_months = pd.Series(dat_fn.query('hp==@hp').month.unique())
        tmp_fn = u_months + '+' + hp
        tmp_holder_reg, tmp_holder_ord = [], []
        for j, fn in enumerate(tmp_fn):
            tmp_ord = pd.read_csv(os.path.join(path_model_ord,fn))
            tmp_reg = pd.read_csv(os.path.join(path_model_reg,fn))
            tmp_ord.insert(0,'month',u_months[j])
            tmp_reg.insert(0,'month',u_months[j])
            tmp_holder_ord.append(tmp_ord)
            tmp_holder_reg.append(tmp_reg)
        tmp_ord = pd.concat(tmp_holder_ord).reset_index(None, True)
        tmp_reg = pd.concat(tmp_holder_reg).reset_index(None, True)
        cn_z_ord = find_zero_var(tmp_ord)
        cn_z_reg = find_zero_var(tmp_reg)
        # Define the grouping
        cn_gg_ord = cn_z_ord + ['lead','metric']
        cn_gg_reg = cn_z_ord + ['lead','metric','iqr']
        # Aggregate over the months        
        tmp_ord_agg = tmp_ord.groupby(cn_gg_ord).apply(lambda x:  
            pd.Series({'value':x.value.mean(),'se':x.se.mean(),'den':x.den.sum()})).reset_index()
        tmp_ord_agg = tmp_ord_agg.assign(den = lambda x: x.den.astype(int), iqr='med')
        tmp_reg_agg = tmp_reg.groupby(cn_gg_reg).apply(lambda x:  
            pd.Series({'value':x.value.mean(),'se':x.se.mean()})).reset_index()
        # Combine into single
        tmp_agg = pd.concat([tmp_ord_agg, tmp_reg_agg])
        tmp_agg[['hp','model']] = [hp, model]
        holder.append(tmp_agg)



