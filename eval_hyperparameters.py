"""
SEARCHES THROUGH ~/test FOLDER TO FIND PERFORMANCE FOR DIFFERENT MODEL CONFIGS
"""

# # Calls class from ~/mdls folder
# import argparse
# from numpy.lib.arraysetops import isin
# from sklearn import metrics
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_list', nargs='+', help='Model classes to evaluate (xgboost lasso)')
# args = parser.parse_args()
# model_list = args.model_list
# print(model_list)
# assert isinstance(model_list, list)

# For debugging
model_list = ['xgboost']

import os
import pandas as pd
import numpy as np
from scipy import stats
from time import time
from plotnine import *
from funs_support import date2ymw, find_dir_olu, get_reg_score, get_iqr, gg_save #gg_color_hue
from funs_stats import get_esc_levels, ttest_vec, prec_recall_lbls

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
lst_dir = [dir_figures, dir_output, dir_flow, dir_test]
assert all([os.path.exists(z) for z in lst_dir])

cn_ymd = ['year', 'month', 'day']
cn_ymdh = cn_ymd + ['hour']
cn_ymdl = cn_ymd + ['lead']

def drop_zero_var(df):
    df = df.copy()
    u_count = df.apply(lambda x: x.unique().shape[0],0)
    df = df[u_count[u_count > 1].index]
    return df

################################
# --- (1) PREPARE BASELINE --- #

cn_ymdh = ['year', 'month', 'day', 'hour']
cn_dates = ['date_rt','date_pred']
idx = pd.IndexSlice

fn_test = os.listdir(dir_test)

# (i) Load the real-time y
act_y = pd.read_csv(os.path.join(dir_flow, 'hourly_yX.csv'),usecols=['date','census_max'])
act_y.rename(columns={'census_max':'y_rt', 'date':'date_rt'},inplace=True)
act_y.date_rt = pd.to_datetime(act_y.date_rt)

# (ii) Load the benchmark result
dat_bl = pd.read_csv(os.path.join(dir_test,'bl_hour.csv'))
dat_bl[cn_dates] = dat_bl[cn_dates].apply(pd.to_datetime,0)
dat_bl = pd.concat([date2ymw(dat_bl.date_rt), dat_bl],1)
# dat_bl[['date_rt','y_rt','lead']].merge(act_y,'left',on='date_rt').query('y_rt_x != y_rt_y')

# (iii) WOY lookup dictionary
freq_woy = dat_bl.groupby(['woy','year']).size().reset_index()
freq_woy = freq_woy.rename(columns={0:'n'}).sort_values(['year','woy'])
freq_woy = freq_woy.query('n == n.max()').reset_index(None,True).drop(columns='n')
# Mappings of year/woy to date
lookup_woy = dat_bl.groupby(['year','woy']).date_rt.min().reset_index()
lookup_woy = lookup_woy.merge(freq_woy,'inner')
# Subset to date range
dat_bl = dat_bl.merge(freq_woy,'inner')

# (iv) Get escalation levels
esc_bins = [-1000, 31, 38, 48, 1000]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
act_y = get_esc_levels(act_y,['y_rt'],esc_bins, esc_lbls)
# Create prediction versoin
pred_y = act_y.rename(columns={'date_rt':'date_pred','y_rt':'y','esc_y_rt':'esc_y'})

# act_y.query('date_rt>@dat_bl.date_rt.min()').esc_y_rt.value_counts(True)

#################################
# --- (2) MODEL PERFORMANCE --- #

cn_model = ['date_rt','date_pred','lead','pred']
cn_reg = ['year','woy','lead']
cn_ord = ['y_delta','pred_delta','date_rt','lead']
cn_gg = ['lead','metric']
cn_iqr = ['med','lb','ub']
di_iqr = dict(zip(cn_iqr,['Median','Q25','Q75']))
di_swap = {'med':'med', 'lb':'ub', 'ub':'lb'}
cn_hp = ['dtrain','h_retrain']
cn_gg_reg = cn_gg + ['iqr']

# (i) Baseline performance
res_bl = dat_bl[cn_model+['year','woy']].copy()
res_bl = res_bl.merge(act_y,'inner',on='date_rt')
res_bl = res_bl.merge(pred_y,'inner','date_pred')

res_bl = get_esc_levels(res_bl,['pred'],esc_bins, esc_lbls)
res_bl = res_bl.assign(y_delta=lambda x: np.sign(x.esc_y - x.esc_y_rt),
                pred_delta = lambda x: np.sign(x.esc_pred - x.esc_y_rt) )
bl_reg = res_bl.groupby(cn_reg).apply(get_reg_score).reset_index()
bl_reg = bl_reg.melt(cn_reg,None,'metric').groupby(cn_gg).value.apply(get_iqr)
bl_reg = bl_reg.reset_index().rename(columns={'level_2':'iqr'})
bl_reg = bl_reg.assign(value=lambda x: np.where(x.metric == 'MAE',-x.value,x.value),
                       iqr=lambda x: np.where(x.metric == 'MAE',x.iqr.map(di_swap),x.iqr),
                       model_name='bl')
# classification
bl_ord = prec_recall_lbls(x=res_bl[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
bl_ord = bl_ord.query('pred_delta == 1').reset_index(None, True)

# (ii) Load in the different model classes
n_bs = 250
holder_reg, holder_ord, holder_bs = [], [], []
for model in model_list:
    assert model in fn_test
    path_model = os.path.join(dir_test, model)
    fn_model = pd.Series(os.listdir(path_model))
    fn_model = fn_model[fn_model.str.contains('\\.csv')].reset_index(None,True)
    for i, fn in enumerate(fn_model):
        stime = time()
        print('fn: %s (%i)' % (fn,i+1))
        params = fn.replace('.csv','').split('+')
        params = pd.DataFrame([param.split('=') for param in params])
        params.rename(columns={0:'tt',1:'val'},inplace=True)
        params = params.query('tt != "lead"')
        # Remove the number of jobs
        params.val = params.val.str.replace('\\_n\\_jobs\\-[0-9]{1,2}','',regex=True)
        # Load data
        df = pd.read_csv(os.path.join(path_model, fn), usecols=cn_model)
        df[cn_dates] = df[cn_dates].apply(pd.to_datetime, 0)
        df = pd.concat([date2ymw(df.date_rt), df],1)
        df = df.merge(freq_woy,'inner')
        assert len(df) == len(dat_bl)
        # Add on y's
        df = df.merge(act_y,'inner',on='date_rt')
        df = df.merge(pred_y,'inner','date_pred')
        df = get_esc_levels(df,['pred'],esc_bins, esc_lbls)
        df = df.assign(y_delta=lambda x: np.sign(x.esc_y - x.esc_y_rt),
                        pred_delta = lambda x: np.sign(x.esc_pred - x.esc_y_rt) )
        # Calculate spearman's correlation
        perf_reg = df.groupby(cn_reg).apply(get_reg_score).reset_index()
        perf_reg = perf_reg.melt(cn_reg,None,'metric').groupby(cn_gg).value.apply(get_iqr)
        perf_reg = perf_reg.reset_index().rename(columns={'level_2':'iqr'})
        tmp_reg = pd.DataFrame(np.tile(params.val.values,[perf_reg.shape[0],1]),columns=params.tt)
        perf_reg = pd.concat([perf_reg,tmp_reg],1)
        perf_reg = perf_reg.assign(value=lambda x: np.where(x.metric == 'MAE',-x.value,x.value),
              iqr=lambda x: np.where(x.metric == 'MAE',x.iqr.map(di_swap),x.iqr))
        # Calculate the precision/recall
        perf_ord = prec_recall_lbls(x=df[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
        perf_ord = perf_ord.query('pred_delta == 1').reset_index(None, True)
        tmp_ord = pd.DataFrame(np.tile(params.val.values,[perf_ord.shape[0],1]),columns=params.tt)
        perf_ord = pd.concat([perf_ord,tmp_ord],1)
        # Do inference via the boostrap
        tmp_aerr = df.assign(aerr=lambda x: np.abs(x.y-x.pred))[cn_reg+['aerr']]
        tmp_aerr['idx'] = tmp_aerr.groupby(cn_reg).cumcount()
        tmp_aerr = tmp_aerr.pivot_table('aerr',cn_reg[:-1]+['idx'],'lead').droplevel(2)
        tmp_aerr_bs = tmp_aerr.sample(frac=n_bs,replace=True,random_state=n_bs)
        tmp_aerr_bs.insert(0,'idx',np.repeat(range(n_bs),tmp_aerr.shape[0]))
        tmp_aerr_bs = tmp_aerr_bs.set_index('idx',append=True)
        # Average over the rows within the idx/year/woy
        tmp_aerr_bs = tmp_aerr_bs.groupby(cn_reg[:-1]+['idx']).mean()
        tmp_aerr_bs = tmp_aerr_bs.reset_index().melt(cn_reg[:-1]+['idx'])
        tmp_aerr_bs = tmp_aerr_bs.groupby(['lead','idx']).value.apply(get_iqr).reset_index()
        tmp_aerr_bs = tmp_aerr_bs.rename(columns={'level_2':'iqr'})
        tmp_aerr_bs = tmp_aerr_bs.groupby(['lead','iqr']).value.std(ddof=1).reset_index()
        tmp_aerr_bs = tmp_aerr_bs.rename(columns={'value':'se'}).assign(metric='MAE')
        tmp_aerr_bs = tmp_aerr_bs.merge(perf_reg,'inner',['lead','iqr','metric'])
        # Save all
        holder_reg.append(perf_reg)
        holder_ord.append(perf_ord)
        holder_bs.append(tmp_aerr_bs)
        # Run time
        dtime, nleft = time() - stime, len(fn_model)-(i+1)
        seta = nleft*dtime
        print('ETA: %0.1f seconds' % seta)
# MAE
model_reg = pd.concat(holder_reg).reset_index(None,True)
model_reg.lead = pd.Categorical(model_reg.lead)
model_reg = drop_zero_var(model_reg)
# Ordinal
model_ord = pd.concat(holder_ord).reset_index(None,True)
model_ord.lead = pd.Categorical(model_ord.lead)
model_ord = drop_zero_var(model_ord)
# Bootstrap
model_bs = pd.concat(holder_bs).reset_index(None,True)
model_bs = drop_zero_var(model_bs)
model_bs[cn_hp] = model_bs[cn_hp].astype(int).apply(pd.Categorical)

###################################
# --- (3) COMPARE PERFORMANCE --- #

sort_reg = model_reg.sort_values(cn_gg_reg+['value'],ascending=False)
sort_reg['idx'] = sort_reg.groupby(cn_gg_reg).cumcount() + 1
sort_reg.reset_index(None,True,True)
sort_reg[cn_hp] = sort_reg[cn_hp].apply(lambda x: pd.Categorical(x.astype(int)))
best_reg = sort_reg.query('idx==1').drop(columns='idx')
best_reg = best_reg.sort_values(cn_gg_reg).reset_index(None,True)

sort_ord = model_ord.sort_values(cn_gg+['value'],ascending=False)
sort_ord['idx'] = sort_ord.groupby(cn_gg).cumcount() + 1
sort_ord.reset_index(None,True,True)
sort_ord[cn_hp] = sort_ord[cn_hp].apply(lambda x: pd.Categorical(x.astype(int)))
best_ord = sort_ord.query('idx==1').drop(columns='idx')
best_ord = best_ord.sort_values(['metric','lead']).reset_index(None,True)

# Remove the worst outliers
gg_hpf_reg = (ggplot(model_reg,aes(x='lead',y='value')) + 
    theme_bw() + geom_boxplot() + 
    facet_grid('metric~iqr',scales='free_y') + 
    labs(y='Regression metric',x='Lead') + 
    ggtitle('Red dot shows baseline model (hour of day)') + 
    theme(subplots_adjust={'wspace': 0.05},axis_text_x=element_text(angle=90)) + 
    geom_point(aes(x='lead',y='value'),color='red',data=bl_reg))
gg_save('gg_hpf_reg.png', dir_figures, gg_hpf_reg, 12, 6)

gg_hpf_ord = (ggplot(model_ord,aes(x='lead',y='value')) + 
    theme_bw() + geom_boxplot() + 
    facet_wrap('~metric') + 
    ggtitle('Red dot shows baseline model (hour of day)') + 
    scale_y_continuous(limits=[0,1]) + 
    labs(y='Precision/Recall',x='Lead') + 
    theme(subplots_adjust={'wspace': 0.15}) + 
    geom_point(color='red',data=bl_ord))
gg_save('gg_hpf_ord.png', dir_figures, gg_hpf_ord, 12, 5)

###################################
# --- (4) SELECTION FREQUENCY --- #

for metric in ['MAE','spearman']:
    fn = 'gg_dayhour_grid_' + metric + '.png'
    tmp = sort_reg.query('metric==@metric')
    tmp = tmp.groupby(cn_gg+cn_hp).idx.mean().reset_index()
    tmp2 = sort_reg.query('metric==@metric & idx <= 3')
    gg_dayhour_grid = (ggplot(tmp, aes(x='dtrain',y='h_retrain',fill='idx')) + 
        theme_bw() + geom_tile(color='black') + 
        labs(x='Training days',y='Retraining (hours)') + 
        facet_wrap('~lead',labeller=label_both,nrow=4) + 
        scale_fill_continuous(name='Rank (1==best)') + 
        ggtitle('MAE') + 
        geom_text(aes(label='idx',color='iqr'),data=tmp2,size=8) + 
        theme(axis_text_x=element_text(angle=90)) + 
        scale_color_manual(name='IQR',labels=['25%','Mean','75%'],values=['white','grey','red']))
    gg_save(fn, dir_figures, gg_dayhour_grid, 12, 7)
    
# Compare to bootstrap distribution
tmp = model_bs.assign(metric='MAE',lead=lambda x: pd.Categorical(x.lead)).drop(columns='value')
tmp[cn_hp] = tmp[cn_hp].astype(int).apply(pd.Categorical)
tmp = tmp.merge(best_reg,'inner').rename(columns={'value':'value_best','se':'se_best'})
tmp = tmp[['lead','iqr','value_best','se_best']]
z_bs = model_bs.merge(tmp,'left').sort_values(['iqr','lead','dtrain']).reset_index(None,True)
# Calculate whether means are 1-sd apart
z_bs = z_bs.assign(z=lambda x: (x.value-x.value_best)/np.sqrt((x.se**2+x.se_best**2)/2))
z_bs = z_bs.assign(pv = lambda x: 2*stats.norm.cdf(-x.z.abs()))
z_bs = z_bs.assign(is_sig = lambda x: x.pv < 0.05)
# Find the smallest dtrain
z_dtrain = z_bs.groupby(['lead','iqr','is_sig']).apply(lambda x: x.dtrain.astype(int).min())
z_dtrain = z_dtrain.reset_index().rename(columns={0:'dtrain'}).query('is_sig==False').drop(columns='is_sig')

posd = position_dodge(0.5)
tmp_gg = z_bs.query('iqr == "med" & lead>=5').reset_index(None,True)
gg_bs_MAE = (ggplot(tmp_gg,aes(x='dtrain',y='value',color='h_retrain',alpha='is_sig')) + 
    theme_bw() + geom_point(position=posd) + 
    scale_color_discrete(name='Hours2Retrain') + 
    facet_wrap('~lead',scales='free_y',labeller=label_both,ncol=5) + 
    labs(y='(Negative) Mean Absolute Error',x='# of training days') + 
    theme(subplots_adjust={'wspace': 0.20}) + 
    scale_alpha_manual(name='>2 sd',values=[1,0.3]) + 
    geom_linerange(aes(ymin='value-2*se',ymax='value+2*se'),position=posd))
gg_save('gg_bs_MAE.png', dir_figures, gg_bs_MAE, 18, 12)

# Plot it!
gg_dtrain_inf = (ggplot(z_dtrain,aes(x='lead',y='dtrain',color='iqr')) + 
    theme_bw() + geom_line() + geom_point() + 
    scale_color_discrete(name='IQR',labels=['Q25','Median','Q75']) + 
    labs(x='Forecasting lead',y='infimum training days within 2sd') + 
    scale_x_continuous(breaks=list(np.arange(1,24,2))) + 
    scale_y_continuous(breaks=list(np.sort(z_dtrain.dtrain.unique()))))
gg_save('gg_dtrain_inf.png', dir_figures, gg_dtrain_inf, 6, 4)







