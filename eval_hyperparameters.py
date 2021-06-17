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
model_list = ['xgboost', 'rxgboost', 'rxgboost2', 'gp_stacker']

import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy import stats
from time import time
from plotnine import *
from funs_support import date2ymw, find_dir_olu, get_reg_score, get_iqr, gg_save, drop_zero_var
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


from scipy.stats import norm, chi2
from funs_support import cvec
# Average difference
def fast_F(x):
    assert x.columns.get_level_values(0).isin(['value','se']).all()
    k = x.value.shape[1]
    assert k == x.se.shape[1]
    mu = cvec(x.value.mean(1).values)
    se = cvec(np.sqrt((x.se**2).mean(1)).values)
    Zhat = (x.value.values - mu) / se
    Zhat2 = np.sum(Zhat**2, 1)
    pval = chi2(df=k-1).cdf(np.sum(Zhat**2,1))
    pval = 2 * np.minimum(pval, 1-pval)
    zscore = norm.ppf(pval)
    zscore = np.where(np.abs(zscore) == np.inf, 0, zscore)
    return zscore, pval


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
cn_iqr = ['lb','med','ub']
di_iqr = dict(zip(cn_iqr,['Q25','Median','Q75']))
di_swap = {'lb':'ub', 'med':'med', 'ub':'lb'}
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
holder_reg, holder_ord = [], []
for model in model_list:
    print('Model: %s' % model)
    assert model in fn_test
    path_model = os.path.join(dir_test, model)
    path_model_ord = os.path.join(path_model, 'ord')
    path_model_reg = os.path.join(path_model, 'reg')
    assert os.path.exists(path_model_ord) & os.path.exists(path_model_reg)
    fn_model_ord = pd.Series(os.listdir(path_model_ord))
    fn_model_reg = pd.Series(os.listdir(path_model_reg))
    # Find the overlapp
    fn_model = pd.Series(np.intersect1d(fn_model_reg, fn_model_ord))
    fn_model = fn_model[fn_model.str.contains('\\.csv')].reset_index(None,True)
    for i, fn in enumerate(fn_model):
        fn_ord = os.path.join(path_model_ord, fn)
        fn_reg = os.path.join(path_model_reg, fn)
        perf_ord = pd.read_csv(fn_ord)
        perf_reg = pd.read_csv(fn_reg)
        holder_ord.append(perf_ord)
        holder_reg.append(perf_reg)
# MAE
model_reg = pd.concat(holder_reg).reset_index(None,True)
model_reg.lead = pd.Categorical(model_reg.lead)
model_reg = drop_zero_var(model_reg)
# Ordinal
model_ord = pd.concat(holder_ord).reset_index(None,True)
model_ord.lead = pd.Categorical(model_ord.lead)
model_ord = drop_zero_var(model_ord)

assert model_reg.notnull().all().all()
assert model_ord.notnull().all().all()

# qq = model_ord.query('model_name=="gp_stacker" & metric=="prec"').reset_index(None,True)
# q1 = qq[cn_hp]
# q2 = drop_zero_var(drop_zero_var(qq.model_args.str.split('\\_',6,True))[1].str.split('\\-',1,True)).rename(columns={1:'hval'})
# qq = pd.concat([q1,q2],1)
# qq = qq.groupby(list(qq.columns)).size().reset_index().rename(columns={0:'n'})
# qq.dtrain.value_counts()
# qq.pivot_table('n',['h_retrain','hval'],'dtrain')


#####################################
# --- (3) ALL MODEL PERFORMANCE --- #

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

#######################################
# --- (4) VARIATION ACROSS MODELS --- #

# Find the dtrain/h_retrain that overlap
count_model = model_reg.groupby(['model_name','dtrain','h_retrain']).size().reset_index()
count_model = count_model.pivot_table(0,cn_hp,'model_name')
count_model = count_model[count_model.notnull().all(1)].reset_index()[cn_hp]

reg_sub = model_reg.query('dtrain.isin(@count_model.dtrain) & h_retrain.isin(@count_model.h_retrain)').reset_index(None,True)
ord_sub = model_ord.query('dtrain.isin(@count_model.dtrain) & h_retrain.isin(@count_model.h_retrain)').reset_index(None,True)
# Calculate the z-score differences
tmp = reg_sub.pivot_table(['value','se'],cn_gg_reg+cn_hp,'model_name')
reg_sub_wide = pd.DataFrame(np.c_[fast_F(tmp)],columns=['zscore','pval'])
reg_sub_wide.index=tmp.index
reg_sub_wide.reset_index(None,inplace=True)

posd = position_dodge(0.5)
gg_reg_models = (ggplot(reg_sub,aes(x='lead',y='value',color='model_name',shape='iqr')) + 
    theme_bw() + geom_point(position=posd,size=3) + 
    geom_linerange(aes(ymin='CI_lb',ymax='CI_ub'),position=posd) + 
    facet_grid('metric~dtrain+h_retrain',scales='free_y',labeller=label_both) + 
    scale_shape_manual(name='IQR',labels=['25%','50%','75%'],values=['$L$','$M$','$U$']) + 
    scale_color_discrete(name='Models'))
gg_save('gg_reg_models.png', dir_figures, gg_reg_models, 7, 7)

gg_reg_zscore = (ggplot(reg_sub_wide,aes(x='lead',y='zscore',shape='iqr')) + 
    theme_bw() + geom_point(position=posd,size=3) + 
    geom_hline(yintercept=0,color='red',linetype='--') + 
    facet_grid('metric~dtrain+h_retrain',scales='free_y',labeller=label_both) + 
    scale_shape_manual(name='IQR',labels=['25%','50%','75%'],values=['$L$','$M$','$U$']))
gg_save('gg_reg_zscore.png', dir_figures, gg_reg_zscore, 7, 7)

gg_ord_models = (ggplot(ord_sub,aes(x='lead',y='value',color='model_name')) + 
    theme_bw() + geom_point(position=posd,size=1) + 
    geom_linerange(aes(ymin='CI_lb',ymax='CI_ub'),position=posd) + 
    facet_grid('metric~dtrain+h_retrain',scales='free_y',labeller=label_both) + 
    scale_color_discrete(name='Models'))
gg_save('gg_ord_models.png', dir_figures, gg_ord_models, 7, 7)


##############################################
# --- (5) VARIATION ACROSS DTRAIN/RTRAIN --- #

cn_metric = ['MAE','spearman']

n_dtrain_htrain = model_reg.groupby(cn_hp).size().reset_index().rename(columns={0:'n'})
retrain_star = n_dtrain_htrain.h_retrain.value_counts().reset_index().query('h_retrain>1')
retrain_star = retrain_star['index'].max()
print('Sup retraining hours with > 1 obs: %s' % retrain_star)

# (i) Number of training days
tmp1 = sort_reg[sort_reg.h_retrain == retrain_star].reset_index(None,True)
tmp2 = sort_ord[sort_ord.h_retrain == retrain_star].reset_index(None,True).assign(iqr='med')
cn_isec = np.intersect1d(tmp1.columns, tmp2.columns)
res_dtrain = pd.concat([tmp1[cn_isec],tmp2[cn_isec]]).reset_index(None,True)
res_dtrain = drop_zero_var(res_dtrain).rename(columns={'dtrain':'train'}).assign(tt='dtrain')

# (ii) Number of retraining days
dtrain_star = n_dtrain_htrain.dtrain.value_counts().reset_index()
dtrain_star = dtrain_star.sort_values('dtrain',ascending=False).iloc[0]['index']

# Number of training days
tmp1 = sort_reg[sort_reg.dtrain == dtrain_star].reset_index(None,True)
tmp2 = sort_ord[sort_ord.dtrain == dtrain_star].reset_index(None,True).assign(iqr='med')
cn_isec = np.intersect1d(tmp1.columns, tmp2.columns)
res_rtrain = pd.concat([tmp1[cn_isec],tmp2[cn_isec]]).reset_index(None,True)
res_rtrain = drop_zero_var(res_rtrain).rename(columns={'h_retrain':'train'}).assign(tt='rtrain')
# Put retraining frequency to days
res_rtrain.train = (res_rtrain.train.astype(float)/24).astype(int)

# (iii) Merge
res_train = pd.concat([res_dtrain,res_rtrain]).reset_index(None,True)

# (iv) Loop over training types and measures
cn_keep = ['lead','iqr','value','se','train']
cn_bounds = ['value','ub','lb']
cn_li = ['lead','iqr']

holder = []
for tt in res_train.tt.unique():
    print('tt: %s' % tt)
    tmp1 = res_train.query('tt == @tt').drop(columns='tt')
    for metric in tmp1.metric.unique():
        print('metric: %s' % metric)
        tmp2 = tmp1.query('metric==@metric').drop(columns='metric')
        tmp2.reset_index(None,True,True)
        u_iqr = tmp2.iqr.unique()
        n_iqr = u_iqr.shape[0]
        # Index to smallest dtrain/retrain
        tmp3 = tmp2.groupby(cn_li).apply(lambda x: x.loc[x.train.idxmin(),cn_keep[:-1]])
        tmp3 = tmp3.reset_index(None,True).rename(columns={'value':'t1'}).drop(columns='se')
        # Merge back on with original
        tmp4 = tmp2[cn_keep].merge(tmp3)
        tmp4 = tmp4.assign(ub=lambda x: x.value+x.se, lb=lambda x: x.value-x.se).drop(columns='se')
        tmp4[cn_bounds] = tmp4[cn_bounds].divide(tmp4.t1,axis=0)
        tmp4 = tmp4.assign(train=lambda x: pd.Categorical(x.train,np.sort(x.train.unique())))
        
        # Find the "best" 
        tmp5 = tmp2.groupby(cn_li).apply(lambda x: x.loc[x.value.idxmax(),cn_keep[:-1]])
        tmp5  = tmp5.reset_index(None,True).rename(columns={'value':'best'}).drop(columns='se')
        tmp6 = tmp2[cn_keep].merge(tmp3).merge(tmp5).assign(gain=lambda x: x.best - x.value)
        tmp6 = tmp6.assign(within=lambda x: x.gain <= 0.5*x.se)
        # tmp6.groupby('lead').within.sum()

        # Find the smallest/largest within
        tmp7 = tmp6.groupby(cn_li+['within']).train.apply(lambda x: 
                    pd.DataFrame({'mi':x.min(),'mx':x.max()},index=[0]))
        tmp7 = tmp7.reset_index().drop(columns='level_'+str(len(cn_li)+1))
        tmp7 = tmp7.query('within == True').assign(val=lambda x: np.where(tt=='rtrain',x.mx,x.mi))
        tmp7 = tmp7.drop(columns=['within','mi','mx']).reset_index(None,True)
        # Save for later
        tmp7 = tmp7.assign(tt=tt,metric=metric)
        holder.append(tmp7)

        # Set the title
        gtit = 'metric=%s, training=%s' % (metric, tt)
        tmp_di = {k:v for k,v in di_iqr.items() if k in list(u_iqr)}

        # (i) box plot
        fn1 = 'gg_' + tt + '_box_' + metric + '.png'
        gg1 = (ggplot(tmp4,aes(x='train',y='value*100',color='iqr')) + 
            theme_bw() + geom_boxplot() + 
            facet_wrap('~iqr',labeller=labeller(iqr=tmp_di),nrow=1) + 
            labs(x='Training (days)',y='Index (day 1 == 100)') + 
            theme(axis_text_x=element_text(angle=90)) + 
            guides(color=False) + ggtitle(gtit) + 
            geom_hline(yintercept=100) + 
            scale_color_discrete(name='IQR',labels=list(tmp_di.values())))
        gg_save(fn1, dir_figures, gg1, int(n_iqr*5), 4)
        
        # (ii) Time trend
        fn2 = 'gg_' + tt + '_trend_' + metric + '.png'
        gg2 = (ggplot(tmp4, aes(x='train.astype(int)',y='value*100',color='iqr')) + 
            theme_bw() + geom_line() + ggtitle(gtit) + 
            geom_hline(yintercept=100) + 
            labs(x='Training (days)',y='Index (day 1 == 100)') + 
            facet_wrap('~lead',labeller=label_both,nrow=4) + 
            theme(axis_text_x=element_text(angle=90)) + 
            scale_color_discrete(name='IQR',labels=list(tmp_di.values())))
        gg_save(fn2, dir_figures, gg2, 12, 7)

# Merge the infimum/supremum for analysis later
di_tt = {'dtrain':'# Training days', 'rtrain':'# Retraining days'}
di_metric = {'spearman':"Spearman's rho", 'MAE':'Mean Abs Error','sens':'Sensivitiy','prec':'Precision'}

dat_infimum = pd.concat(holder).reset_index(None,True)
tmp = dat_infimum.query('metric.isin(["MAE","prec"])')
gg_infimum = (ggplot(tmp,aes(x='lead',y='val',color='iqr')) + 
    theme_bw() + geom_point(size=0.5) + geom_line() + 
    labs(y='Smallest/largest training/retraining',x='Forecasting horizon') + 
    ggtitle('Within 0.5SDs of optimal') + 
    scale_x_continuous(breaks=list(range(1,25,1))) + 
    theme(axis_text_x=element_text(angle=90)) + 
    facet_grid('tt~metric',scales='free_y',labeller=labeller(tt=di_tt,metric=di_metric)) +
    scale_color_discrete(name='IQR',labels=list(di_iqr.values())))
gg_save('drtain_infimum.png', dir_figures, gg_infimum, 10, 6)







