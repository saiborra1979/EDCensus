# # Calls class from ~/mdls folder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lead', type=int, default=None, help='Number of leads to forecast')
parser.add_argument('--lag', type=int, default=None, help='Number of lags to use in X')
parser.add_argument('--month', type=int, default=1, help='Which month to use since March-2020 onwards')
parser.add_argument('--dtrain', type=int, default=5, help='# of training days')
parser.add_argument('--h_rtrain', type=int, default=24, help='Frequency of retraining')
parser.add_argument('--ylbl', type=str, default=None, help='Column from hourly_yX.csv to forecast')
parser.add_argument('--model_name', type=str, default=None, help='Model to use from ~/mdls')
parser.add_argument('--model_args', type=str, default=None, 
    help='Optional arguments for model class (e.g. n_trees=100,depth=3,...)')
parser.add_argument('--write_scores', default=False, action='store_true')
parser.add_argument('--write_model', default=False, action='store_true')

args = parser.parse_args()
print(args)
lead, lag, month, dtrain, h_rtrain = args.lead, args.lag, args.month, args.dtrain, args.h_rtrain
ylbl, model_name, model_args = args.ylbl, args.model_name, args.model_args
write_scores = args.write_scores
write_model = args.write_model

# # For debugging
# dtrain=366; h_rtrain=24; lag=24; lead=24; month=9
# model_args='base=rxgboost,nval=48,max_iter=100,lr=0.1,max_cg=10000,n_trees=100,depth=3,n_jobs=3'
# model_name='gp_stacker'; ylbl='census_max'; write_scores=False; write_model=False

# Load modules
import os
import numpy as np
import pandas as pd
from time import time
from funs_support import find_dir_olu, get_date_range, makeifnot, date2ymw, write_pickle
from funs_stats import prec_recall_lbls, get_reg_score
from mdls.funs_encode import yX_process
from funs_esc import esc_bins, esc_lbls, get_esc_levels


# (i) Model class should be in mdls folder
assert model_name in list(pd.Series(os.listdir('mdls')).str.replace('.py','',regex=True))

# (ii) Put optional arguments in dict (if any)
if model_args is not None:
    di_model = {}
    for opt in model_args.split(','):
        opt1, opt2 = opt.split('=')
        di_model[opt1] = opt2
else:
    di_model = None
print('di_model: %s' % di_model)

# (iii) Import the "mdl" class from the mdl folder
model_class = __import__('mdls.' + model_name)
assert hasattr(model_class, model_name)
assert hasattr(getattr(model_class, model_name), 'model')
model = getattr(getattr(model_class, model_name), 'model')

# Models must have four mandatory attributes
attr_mand = ['fit','predict','update_Xy','pickle_me']
assert all([hasattr(model, attr) for attr in attr_mand])

# (iv) Set up folders
dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_class = os.path.join(dir_test, model_name)
dir_model = os.path.join(dir_class, 'model')
makeifnot(dir_class)
makeifnot(dir_model)


#############################
# --- STEP 1: LOAD DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_X = pd.read_csv(os.path.join(dir_flow, 'hourly_yX.csv'))
df_X.date = pd.to_datetime(df_X.date)
# Extract y
yval = df_X[ylbl].values
dates = df_X.date.copy()


######################################
# --- STEP 2: CREATE DATE-SPLITS --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

assert isinstance(dtrain,int) & isinstance(h_rtrain,int)

dfmt = '%Y-%m-%d'
dmin = pd.to_datetime('2020-03-01')
dmax = dmin + pd.DateOffset(months=month) - pd.DateOffset(seconds=1)
dmin = pd.to_datetime(dmax.strftime('%Y-%m')+'-01')
nhours = int(np.ceil((dmax - dmin).total_seconds()/(60*60)))
ndays = int(np.ceil(nhours / 24))
print('day start: %s, day stop: %s (hours=%i, days=%i)' % 
    (dmin.strftime(dfmt),dmax.strftime(dfmt),nhours, ndays))

offset_train = pd.DateOffset(days=dtrain)
print('Training offset: %s' % offset_train)


########################################
# --- STEP 3: TRAIN BASELINE MODEL --- #

# All model classes a cn dictionary with ohe, cont, and bin
cn_cont = ['census_max','census_var','tt_arrived','tt_discharged']
cn_ohe = ['month','day','hour','dow','is_holiday']
cn_bin = []
di_cn = {'ohe':cn_ohe, 'bin':cn_bin, 'cont':cn_cont}
cn_all = list(df_X.columns)
cn_use = sum(list(di_cn.values()), [])
# Check that all columns can be bound
assert all([cn in cn_all for cn in cn_use])


holder = []
stime = time()
for ii in range(nhours):
    # break
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    s_train = time_ii - offset_train  # start time
    idx_train = ((dates >= s_train) & (dates <= time_ii)).values
    dates_train = dates[idx_train].reset_index(None,True)
    ytrain = yval[idx_train].copy()
    Xtrain = df_X.loc[idx_train, cn_use].copy()
    X_now = Xtrain[-(lag+1):]  # Ensure enough rows to calculate lags
    if ii % h_rtrain == 0:
        print('Training range: %s' % (get_date_range(dates_train)))
        print('Current time: %s' % time_ii)
        enc_yX = yX_process(cn=cn_use, lead=lead, lag=lag,
                cn_ohe=di_cn['ohe'], cn_cont=di_cn['cont'], cn_bin=di_cn['bin'])
        enc_yX.fit(X=Xtrain)
        # break
        regressor = model(encoder=enc_yX, di_model=di_model)
        regressor.fit(X=Xtrain, y=ytrain)
        nleft, nsec = nhours-(ii+1), time() - stime
        rate = (ii + 1) / nsec
        eta = nleft/rate
        print('ETA: %.1f minutes' % (eta/60))
    else:
        # Update X/y where relevant
        regressor.update_Xy(Xnew=Xtrain, ynew=ytrain)
    # Do inference
    tmp_pred = regressor.predict(X_now)
    if isinstance(tmp_pred, np.ndarray):
        tmp_pred = pd.DataFrame(tmp_pred).melt(None,None,'lead','pred')
    tmp_pred['date_rt'] = time_ii
    if tmp_pred.lead.min() == 0:
        tmp_pred['lead'] += 1
    holder.append(tmp_pred)


########################
# --- STEP 3: HASH --- #

lst_hp = ['month','lead', 'lag', 'dtrain', 'h_rtrain', 'ylbl', 'model_name']
lst_val = [month, lead, lag, dtrain, h_rtrain, ylbl, model_name]
df_hp = pd.DataFrame({'tt':'shared','hp':lst_hp, 'val':lst_val})

# Process model_args
if model_args is not None:
    df_margs = pd.DataFrame([z.split('=') for z in model_args.split(',')])
    df_margs.rename(columns={0:'hp',1:'val'}, inplace=True)
    # Number of jobs should not be part of hyperparameter as it does not affect results
    df_margs = df_margs[df_margs.hp != 'n_jobs'].reset_index(None, True)
    df_hp = df_hp.append(df_margs.assign(tt='margs'))
# Get the hash based on the string of the hyperparmater values
hp_str = df_hp.val.astype(str).str.cat(sep=',')
hash_str = str(pd.util.hash_pandas_object(pd.Series([hp_str]))[0])
print('~~~~ Hash for this run: %s ~~~~' % hash_str)

###################################
# --- STEP 4: MERGE AND LABEL --- #

df_res = pd.concat(holder).reset_index(None,True)
df_res = df_res.assign(date_pred=lambda x: x.date_rt + x.lead*pd.offsets.Hour(1))
df_res = df_res.merge(pd.DataFrame({'date_rt':dates,'y_rt':yval}))
df_res = df_res.merge(pd.DataFrame({'date_pred':dates,'y':yval}))
df_res = df_res.sort_values(['date_rt','lead']).reset_index(None,True)
df_res = pd.concat([date2ymw(df_res.date_rt), df_res],1).drop(columns='month')
# Add on the escalation levels
df_res = get_esc_levels(df_res,['y','y_rt','pred'],esc_bins, esc_lbls)
df_res = df_res.assign(y_delta=lambda x: np.sign(x.esc_y - x.esc_y_rt),
                pred_delta = lambda x: np.sign(x.esc_pred - x.esc_y_rt) )


#########################################
# --- STEP 5: GET MODEL PERFORMANCE --- #

cn_reg = ['lead']
cn_regn = cn_reg + ['n']
cn_gg = ['lead', 'metric']
cn_ggn = cn_gg + ['n']
cn_ord = ['y_delta','pred_delta','date_rt','lead']

# (1) Calculate spearman and MAE
perf_reg = df_res.groupby(cn_reg).apply(get_reg_score,add_n=True).reset_index()
perf_reg = perf_reg.melt(cn_regn,None,'metric')
perf_reg['n'] = perf_reg.n.astype(int)

# (2) Calculate the precision/recall
perf_ord = prec_recall_lbls(x=df_res[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
perf_ord = perf_ord.query('pred_delta == 1').reset_index(None, True)
perf_ord = perf_ord.drop(columns='pred_delta').rename(columns={'den':'n'})

# Merge regression + ordinal
perf_agg = pd.concat([perf_reg, perf_ord]).reset_index(None, True)

# (3) Do boostrap to get the standard errors
n_bs = 1000
holder_agg = []
stime = time()
for i in range(n_bs):
    if (i + 1) % 5 == 0:
        print(i+1)
        dtime, nleft = time() - stime, n_bs-(i+1)
        rate = (i+i)/dtime
        seta = nleft / rate
        print('bootstrap ETA: %i seconds (%i left)' % (seta, nleft))
    # Stratify bootstrap by lead
    bs_res = df_res.groupby('lead').sample(frac=1,replace=True,random_state=i).reset_index(None,True)
    # Regression
    bs_reg = bs_res.groupby(cn_reg).apply(get_reg_score,add_n=True).reset_index()
    bs_reg = bs_reg.melt(cn_regn,None,'metric').assign(n=lambda x: x.n.astype(int))
    # Classification
    bs_ord = prec_recall_lbls(x=bs_res[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
    bs_ord = bs_ord.query('pred_delta == 1').reset_index(None, True)
    bs_ord = bs_ord.drop(columns='pred_delta').rename(columns={'den':'n'})
    # Save
    bs_agg = pd.concat([bs_reg, bs_ord]).assign(bidx=i)
    holder_agg.append(bs_agg)

# Calculate bootstrap standard error
bs_agg = pd.concat(holder_agg)
bs_agg = bs_agg.groupby(cn_gg).value.std(ddof=1).reset_index()
bs_agg.rename(columns={'value':'se'}, inplace=True)
# Merge with existing
perf_agg = perf_agg.merge(bs_agg,'left')

########################
# --- STEP 6: SAVE --- #

# File named is just hash
fn_hash = hash_str + '.pickle'
path_hash = os.path.join(dir_class, fn_hash)

# Initialize dictionary
di_res = {'hp':df_hp, 'agg':perf_agg}
if write_scores:
    print('Writing scores')
    di_res['scores'] = df_res

# Save dictionary
write_pickle(di=di_res, path=path_hash)

# Write model if specified
if write_model:
    print('Writing model')
    fn_model = 'mdl_' + fn_hash
    path_pickle = os.path.join(dir_model, fn_model)
    regressor.pickle_me(path=path_pickle)

print('~~~ End of run_mdl.py ~~~')