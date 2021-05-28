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
from plotnine import *
from funs_support import date2ymw, find_dir_olu, get_reg_score, get_iqr, gg_save #gg_color_hue
from funs_stats import get_esc_levels, add_bin_CI, prec_recall_lbls

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
cn_iqr = ['mu','lb','ub']
di_swap = {'mu':'mu', 'lb':'ub', 'ub':'lb'}

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
    assert model in fn_test
    path_model = os.path.join(dir_test, model)
    fn_model = pd.Series(os.listdir(path_model))
    fn_model = fn_model[fn_model.str.contains('\\.csv')].reset_index(None,True)
    for i, fn in enumerate(fn_model):
        print('fn: %s (%i)' % (fn,i+1))
        params = fn.replace('.csv','').split('+')
        params = pd.DataFrame([param.split('=') for param in params])
        params.rename(columns={0:'tt',1:'val'},inplace=True)
        params = params.query('tt != "lead"')
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
        # Save both
        holder_reg.append(perf_reg)
        holder_ord.append(perf_ord)
# Merge
model_reg = pd.concat(holder_reg).reset_index(None,True)
model_ord = pd.concat(holder_ord).reset_index(None,True)
model_reg.lead = pd.Categorical(model_reg.lead)
model_ord.lead = pd.Categorical(model_ord.lead)
# Remove the number of jobs
model_reg.model_args = model_reg.model_args.str.replace('\\_n\\_jobs\\-[0-9]{1,2}','')
model_ord.model_args = model_ord.model_args.str.replace('\\_n\\_jobs\\-[0-9]{1,2}','')
# Remove extraneuous columns
model_reg = drop_zero_var(model_reg)
model_ord = drop_zero_var(model_ord)

# Do inference via the boostrap
tmp_aerr = df.assign(aerr=lambda x: np.abs(x.y-x.pred))[cn_reg+['aerr']]
tmp_aerr['idx'] = tmp_aerr.groupby(cn_reg).cumcount()
tmp_aerr = tmp_aerr.pivot_table('aerr',cn_reg[:-1]+['idx'],'lead').droplevel(2)

n_bs = 50
tmp_aerr_bs = tmp_aerr.sample(frac=n_bs,replace=True,random_state=n_bs)
tmp_aerr_bs.insert(0,'idx',np.repeat(range(n_bs),tmp_aerr.shape[0]))
tmp_aerr_bs = tmp_aerr_bs.set_index('idx',append=True)
# Average over the rows within the idx/year/woy
tmp_aerr_bs = tmp_aerr_bs.groupby(cn_reg[:-1]+['idx']).mean()
tmp_aerr_bs = tmp_aerr_bs.reset_index().melt(cn_reg[:-1]+['idx'])
tmp_aerr_bs.groupby(['idx']).value.quantile([0.25,0.5,0.75])


tmp_aerr.groupby(cn_reg[:-1]).mean().median()
perf_reg.query('metric=="MAE" & iqr=="mu"')

###################################
# --- (3) COMPARE PERFORMANCE --- #

cn_hp = ['dtrain','h_retrain']
cn_gg_reg = cn_gg + ['iqr']

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
    theme(subplots_adjust={'wspace': 0.15}) + 
    geom_point(aes(x='lead',y='value'),color='red',data=bl_reg))
gg_save('gg_hpf_reg.png', dir_figures, gg_hpf_reg, 8, 8)

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
    # Get the bootstrap distribution
    tmp



# Wrap on metric and h_retrain
tmp = sort_reg.rename(columns={'h_retrain':'Retrain_Hours','metric':'Metric'}).copy()
tmp = tmp.assign(lead2=lambda x: x.lead.astype(int)).query('lead2>=4')
# tmp = tmp.assign(lead=lambda x: x.lead.astype(int))

posd = position_dodge(0.5)
gg_hp_dayhour = (ggplot(tmp,aes(x='dtrain',y='mu',color='lead')) + 
    theme_bw() + geom_point(position=posd) + 
    labs(x='Training days', y='MAE/Spearman')  + 
    ggtitle('Retrain/Training set size (lead>=4)') + 
    facet_grid('Metric~Retrain_Hours',scales='free_y',labeller=label_both))
gg_save('gg_hp_dayhour.png', dir_figures, gg_hp_dayhour, 12, 6)











################################
# --- (3) PRECISION/RECALL --- #

# Escalation changes by sign (-1/0/+1)
cn_drop = ['year','month']
tmp = dat_ord.assign(pred=lambda x: np.sign(x.pred),y=lambda x: np.sign(x.y))
res_sp_agg = prec_recall_lbls(tmp.drop(columns=cn_drop), cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_idx=cn_date)
# Sign by month
res_sp_month = prec_recall_lbls(tmp, cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_idx=cn_date)
del tmp
# All escalations
res_sp_full = prec_recall_lbls(dat_ord.drop(columns=cn_drop), cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_idx=cn_date)

# - (i) Breakdown of TPs/FPs - #
tmp0 = pd.concat([res_sp_agg.query('pred==1').assign(pred=0), res_sp_full.query('pred>0')])
tmp1 = tmp0.query('metric=="prec"').assign(tp=lambda x: np.round(x.value*x.den,0).astype(int)).assign(fp=lambda x: x.den-x.tp).drop(columns=['den','value','metric'])
tmp2 = tmp0.query('metric=="sens"').assign(fn=lambda x: x.den-np.round(x.value*x.den,0).astype(int)).rename(columns={'den':'pos'}).drop(columns=['value','metric'])
res_tp_full = tmp1.merge(tmp2)
assert res_tp_full.assign(check=lambda x: x.tp+x.fn == x.pos).check.all()
res_tp_full = res_tp_full.melt(['lead','pred'],None,'metric','n')
del tmp0, tmp1, tmp2


# - (iii) Precision/recall curve - #
p_seq = np.arange(0.02,1.0,0.02)
cn_sign = ['pred','y']
holder = []
for p in p_seq:
    print('percentile: %0.2f' % p)
    tmp_month = ordinal_lbls(dat_pr, cn_date=cn_date, cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_se=cn_se, level=p)
    tmp_agg = ordinal_lbls(dat_pr.drop(columns=cn_drop), cn_date=cn_date, cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_se=cn_se, level=p)
    tmp = pd.concat([tmp_agg.assign(month=0,year=0), tmp_month])
    del tmp_agg, tmp_month
    tmp[cn_sign] = tmp[cn_sign].apply(lambda x: np.sign(x), 0)
    tmp = prec_recall_lbls(tmp, cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_idx=cn_date)
    tmp = tmp.query('pred==1').drop(columns='pred').assign(p=p)
    holder.append(tmp)
# Merge and plot
df_pr = pd.concat(holder).pivot_table('value',['year','month','p','lead'],'metric').reset_index()
df_pr.groupby('lead').head(1)

tmp = df_pr.query('month==0 & year==0')
gg_pr_agg = (ggplot(tmp, aes(x='sens',y='prec',color='lead',group='lead')) +
    theme_bw() + geom_line() + 
    labs(x='Recall',y='Precision') +
    geom_abline(slope=-1,intercept=1,linetype='--') +
    scale_color_cmap(name='Lead') +
    scale_x_continuous(limits=[0,1])+
    scale_y_continuous(limits=[0,1]) +
    ggtitle('Δ>0 in escalation (all test months)') +
    theme(legend_direction='horizontal', legend_position=(0.7, 0.8),
            legend_background=element_blank(),legend_key_size=10))
gg_save('gg_pr_agg.png',dir_figures,gg_pr_agg,5,4)

# Precision recall for year/month
tmp1 = df_pr.query('month>0 & year>0').reset_index(None, True)
tmp1 = tmp1.assign(ymon1=lambda x: (x.year+x.month/100).astype(str))

gg_pr_month = (ggplot(tmp1, aes(x='sens',y='prec',color='ymon1',group='ymon1')) +
    theme_bw() + geom_line() + 
    labs(x='Recall',y='Precision') +
    geom_abline(slope=-1,intercept=1,linetype='--') +
    # scale_color_cmap(name='Lead') +
    facet_wrap('~lead',labeller=label_both) + 
    scale_x_continuous(limits=[0,1])+
    scale_y_continuous(limits=[0,1]) +
    ggtitle('Δ>0 in escalation') +
    theme(legend_direction='horizontal',legend_position='bottom'))
gg_save('gg_pr_month.png',dir_figures,gg_pr_month,16,12)

# Relationship between average precision
gg = ['ymon','lead','metric']
tmp2 = tmp1.assign(ymon=lambda x: pd.to_datetime(x.year.astype(str)+'-'+x.month.astype('str')+'-01'))
tmp2 = tmp2.melt(['ymon','lead'],['prec','sens'])
tmp2 = tmp2.groupby(gg).value.apply(lambda x: 
    pd.Series({'mu':x.mean(),'lb':x.quantile(0.25),'ub':x.quantile(0.75)})).reset_index()
tmp2 = tmp2.pivot_table('value',gg,'level_'+str(len(gg))).reset_index()

gg_pr_mu = (ggplot(tmp2, aes(x='ymon',y='mu',color='metric')) + 
    theme_bw() + geom_point() + 
    geom_linerange(aes(ymin='lb',ymax='ub')) + 
    facet_wrap('~lead',labeller=label_both,nrow=4) + 
    labs(x='Year.Month',y='Precision/Recall') + 
    geom_hline(yintercept=0.5,linetype='--') + 
    ggtitle('IQR of Precision/Recall curve around average') + 
    scale_x_datetime(date_breaks='1 month',date_labels='%b, %y') + 
    scale_color_discrete(name='Metric',labels=['Precision','Recall']) + 
    theme(axis_text_x=element_text(angle=90),axis_title_x=element_blank()))
gg_save('gg_pr_mu.png',dir_figures,gg_pr_mu,18,10)

# Relationship of average precision to number of labels
tmp3 = res_sp_month.query('pred==1 & lead > 1')
tmp3.groupby(['year','month']).size()
tmp3 = tmp3.assign(ymon=lambda x: pd.to_datetime(x.year.astype(str)+'-'+x.month.astype('str')+'-01'))
tmp3.drop(columns=['pred','value','year','month'],inplace=True)
tmp3.query('ymon=="2020-04-01"')
tmp4 = tmp2.drop(columns=['lb','ub']).merge(tmp3,'inner',['ymon','lead','metric'])
tmp5 = tmp4.query('ymon>="2021"').assign(lbl=lambda x: x.ymon.dt.strftime('%b, %y'))

gg_pr_n = (ggplot(tmp4,aes(x='den',y='mu',color='metric')) + 
    theme_bw() + geom_point() + 
    labs(x='Number of labels',y='Average precision/recall') + 
    facet_wrap('~lead',labeller=label_both,nrow=4) + 
    ggtitle('Over 13 test year/months') + 
    geom_smooth(method='lm',se=False) + 
    scale_color_discrete(name='Metric',labels=['Precision','Recall']) + 
    geom_text(aes(label='lbl'),data=tmp5,size=8) + 
    geom_hline(yintercept=0.5,linetype='--'))
gg_save('gg_pr_n.png',dir_figures,gg_pr_n,18,10)



#######################
# --- (4) FIGURES --- #






tit = 'Prediction breakdown for Δ>0 in escalation'
tmp = res_tp_full.assign(pred=lambda x: pd.Categorical(x.pred.map(di_lblz),list(di_lblz.values())), metric=lambda x: pd.Categorical(x.metric.map(di_metric), list(di_metric.values())))
gg_tp_full = (ggplot(tmp, aes(x='lead', y='n', color='metric')) +
             theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
             scale_x_continuous(breaks=list(range(1,25))) +
             labs(x='Forecasting lead', y='Count') +
             facet_wrap('~pred',scales='free_y',labeller=label_both) +
              scale_color_manual(values=colz_gg, name='Metric') +
              theme(subplots_adjust={'wspace': 0.10}))
gg_save('gg_tp_full.png',dir_figures,gg_tp_full,14,7)



# - (ii) Precision/Recall with CI - #
cn_n, cn_val = 'den', 'value'
tmp1 = add_bin_CI(res_sp_agg.query('pred==1'), cn_n=cn_n, cn_val=cn_val, method='beta', alpha=0.05)
tmp1.pred = 0
tmp2 = add_bin_CI(res_sp_full.query('pred > 0'), cn_n=cn_n, cn_val=cn_val, method='beta', alpha=0.05)
tmp = pd.concat([tmp1, tmp2]).reset_index(None,True)

tit = 'Precision/Recall for escalation levels\n95% CI (beta method)'
gg_sp_full = (ggplot(tmp, aes(x='lead', y='value', color='pred.astype(str)')) +
    theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
    scale_x_continuous(breaks=list(range(1,25))) +
    labs(x='Forecasting lead', y='Precision/Recall') +
    geom_linerange(aes(ymin='lb', ymax='ub'),position=posd) +
    facet_wrap('~metric', labeller=labeller(metric=di_pr)) +
    scale_color_manual(name='Δ esclation',values=colz_gg, labels=lblz) +
    theme(legend_position=(0.5, -0.05),legend_direction='horizontal'))
gg_save('gg_sp_full.png',dir_figures,gg_sp_full,13,5)

xmi = (tmp1.lb.min()*100 // 5)*5/100
xmx = np.ceil(tmp1.ub.max()*100 / 5)*5/100
tit = 'Precision/Recall for Δ>0 in escalation\n95% CI (beta method)'
gg_sp_agg = (ggplot(tmp1, aes(x='lead', y='value', color='metric')) +
    theme_bw() + ggtitle(tit) +
    geom_point(size=2, position=posd) + 
    scale_color_discrete(labels=['Precision','Recall'],name='Metric') +
    scale_x_continuous(breaks=list(range(1,25))) +
    scale_y_continuous(limits=[0,1]) +  #xmi,xmx
    labs(x='Forecasting lead', y='Precision/Recall') + 
    geom_linerange(aes(ymin='lb', ymax='ub'),position=posd))
gg_save('gg_sp_agg.png',dir_figures,gg_sp_agg,9,5)

tmp = add_bin_CI(res_sp_month.query('pred==1'), cn_n=cn_n, cn_val=cn_val, method='beta', alpha=0.05)
tmp.reset_index(None,True).loc[0]
height = int(np.ceil(len(tmp.month.unique()) / 2)) * 3
gg_sp_month = (ggplot(tmp, aes(x='lead', y='value', color='metric')) +
             theme_bw() + geom_point(size=2, position=posd) + ggtitle(tit) +
             scale_color_discrete(labels=['Precision','Recall'],name='Metric') +
             scale_x_continuous(breaks=list(range(1,25))) +
             labs(x='Forecasting lead', y='Precision/Recall') +
             geom_linerange(aes(ymin='lb', ymax='ub'),position=posd) +
             facet_wrap('~year+month',ncol=3,labeller=label_both))
gg_save('gg_sp_month.png',dir_figures,gg_sp_month,16,height)



# # (ii) Find the most recent folder with GPY results
# fn_test = pd.Series(os.listdir(dir_test))
# fn_test = fn_test[fn_test.str.contains('^[0-9]{4}\\_[0-9]{2}\\_[0-9]{2}$')].reset_index(None,True)
# dates_test = pd.to_datetime(fn_test,format='%Y_%m_%d')
# df_fn_test = pd.DataFrame({'fn':fn_test,'dates':dates_test,'has_gpy':False})
# # Find out which have GP data
# for fn in fn_test:
#     fn_fold = pd.Series(os.listdir(os.path.join(dir_test,fn)))
#     mdls_fold = list(pd.Series(fn_fold.str.split('\\_',2,True).iloc[:,1].unique()).dropna())
#     if 'gpy' in mdls_fold:
#         df_fn_test.loc[df_fn_test.fn==fn,'has_gpy'] = True
# # Get largest date
# assert df_fn_test.has_gpy.any()
# fold_recent = df_fn_test.loc[df_fn_test.dates.idxmax()].fn
# print('Most recent folder: %s' % fold_recent)
# path_recent = os.path.join(dir_test,fold_recent)
# fn_recent = pd.Series(os.listdir(path_recent))

# # (iii) Load the predicted/actual
# fn_res = fn_recent[fn_recent.str.contains('^res\\_.*\\.csv$')]
# holder = []
# for fn in fn_res:
#     holder.append(pd.read_csv(os.path.join(path_recent,fn)))
# dat_recent = pd.concat(holder).reset_index(None,True)

# # Stability or recall by threshold
# tmp = df_pr.query('month>0').assign(p=lambda x: pd.Categorical(x.p))
# gg_recall_thresh = (ggplot(tmp,aes(x='p',y='sens')) + theme_bw() + 
#     geom_point(position=position_dodge(0.5),size=0.5) + 
#     facet_wrap('~lead',labeller=label_both) + 
#     ggtitle('Stability of precision across months'))
# gg_save('gg_recall_thresh.png',dir_figures,gg_recall_thresh,16,12)

####################################################################


# # (iv) Load the kernel weights
# path_kernel = os.path.join(dir_flow,'dat_kernel.csv')
# preload_kernel = True
# if preload_kernel:
#     dat_kernel = pd.read_csv(path_kernel)
# else:
#     assert 'pt' in list(fn_recent) # Check that coefficient weights are to be found
#     fold_pt = os.path.join(path_recent, 'pt')
#     fn_pt = pd.Series(os.listdir(fold_pt))
#     n_pt = len(fn_pt)
#     print('A total of %i model weights were found' % n_pt)
#     holder = []
#     for i, fn in enumerate(fn_pt):
#         if (i+1) % 50 == 0:
#             print('Kernel %i of %i' % (i+1, n_pt))
#         tmp_dict = torch.load(os.path.join(fold_pt,fn),map_location=device)
#         keys = list(tmp_dict.keys())
#         vals = torch.cat([v.flatten() for v in tmp_dict.values()]).detach().numpy()
#         day = fn.split('day_')[1].replace('.pth','')
#         lead = int(fn.split('lead_')[1].split('_dstart')[0])
#         tmp_df = pd.DataFrame({'dates':day, 'lead':lead, 'kernel':keys,'value':vals})
#         holder.append(tmp_df)
#     # Merge and save
#     dat_kernel = pd.concat(holder).reset_index(None,True)
#     dat_kernel['dates'] = pd.to_datetime(dat_kernel.dates)
#     # Apply the constraint transformer
#     dat_kernel.rename(columns = {'value':'raw'},errors='ignore', inplace=True)
#     dat_kernel['value'] = dat_kernel.raw.copy()
#     mdl_attr = pd.Series(dat_kernel.kernel.unique())  # Model attributes
#     u_attr = mdl_attr.copy()
#     mdl_attr = mdl_attr[mdl_attr != 'mean.constant']  # Only term without a constraint
#     tt_attr = mdl_attr.str.split('\\.',1,True).iloc[:,0].str.split('\\_',2,True).iloc[:,2].fillna('noise')
#     tt_attr = np.setdiff1d(tt_attr.unique(),['noise'])
#     print('GP uses the following feature types (tt): %s' % (', '.join(tt_attr)))
#     # Load in the model
#     n_samp = 50
#     x_samp = torch.tensor(np.random.randn(n_samp,2),dtype=torch.float32)
#     y_samp = torch.tensor(np.random.randn(n_samp),dtype=torch.float32)
#     ll_samp = gpytorch.likelihoods.GaussianLikelihood()
#     tt_samp = pd.Series(['trend','flow','date','mds','arr','CTAS'])
#     idx_samp = range(len(tt_samp))
#     cidx_samp = pd.DataFrame({'tt':tt_attr,'cn':'x'+tt_attr,'idx':idx_samp,'pidx':idx_samp})
#     mdl_gp = gp_real(train_x=x_samp, train_y=y_samp, likelihood=ll_samp, cidx=cidx_samp)
#     mdl_gp.load_state_dict(torch.load(os.path.join(fold_pt,fn_pt[0]),map_location=device))
#     mdl_gp.training = False
#     for param in mdl_gp.parameters():  # Turn off auto-grad
#         param.requires_grad = False
#     # Get the constraint for each hyperparameter
#     di_constraint = {}
#     for attr in mdl_attr:
#         terms = attr.split('.')
#         tmp_attr = copy.deepcopy(mdl_gp)
#         for j, term in enumerate(terms):
#             if j+1 < len(terms):
#                 tmp_attr = getattr(tmp_attr,term)
#             else:
#                 tmp_constraint = getattr(tmp_attr,term+'_constraint')
#                 tmp_attr = getattr(tmp_attr,term)            
#         di_constraint[attr] = tmp_constraint
#         # Assign the transformed values
#         idx_attr = np.where(dat_kernel.kernel == attr)[0]
#         raw_tens = torch.tensor(dat_kernel.loc[idx_attr,'raw'].values,dtype=torch.float32)
#         dat_kernel.loc[idx_attr,'value'] = di_constraint[attr].transform(raw_tens).numpy()
#     # Format the types
#     di_cn = u_attr.str.split('\\_',2,True).iloc[:,2].str.split('\\.',1,True).iloc[:,0].fillna('constant')
#     di_cn = dict(zip(u_attr,di_cn))
#     di_kern = u_attr.str.split('\\_',2,True).iloc[:,1].fillna('constant').replace('covar.raw','noise')
#     di_kern = dict(zip(u_attr,di_kern))
#     di_coef = u_attr.str.split('\\.',2,True).apply(lambda x: x[x.notnull().sum()-1],1).str.replace('raw_','')
#     di_coef = dict(zip(u_attr,di_coef))
#     di_lvl = u_attr.str.split('\\_',1,True).iloc[:,0]
#     di_lvl = dict(zip(u_attr,di_lvl))
#     # Apply to data.frame
#     dat_kernel['cn'] = dat_kernel.kernel.map(di_cn)
#     dat_kernel['kern'] = dat_kernel.kernel.map(di_kern)
#     dat_kernel['coef'] = dat_kernel.kernel.map(di_coef)
#     dat_kernel['lvl'] = dat_kernel.kernel.map(di_lvl)
#     # Check that these four levels covers all permuatiosn
#     assert dat_kernel.groupby(['cn','kern','coef','lvl']).size().unique().shape[0] == 1
#     print('Saving dat_kernel for later')
#     dat_kernel.to_csv(path_kernel,index=False)
# # Ensure its datetime
# dat_kernel.dates = pd.to_datetime(dat_kernel.dates)

#####################################
# --- (2) R-SQUARED PERFORMANCE --- #

# # (i) Daily R2 trend: 7 day rolling average
# df_r2 = pd.concat([dat_recent,date2ymd(dat_recent.dates)],1).groupby(cn_ymd+['lead']).apply(lambda x: r2_score(x.y, x.pred))
# df_r2 = df_r2.reset_index().rename(columns={0: 'r2'})
# # Get a 7 day average
# df_r2 = df_r2.assign(trend=lambda x: x.groupby('lead').r2.rolling(window=7,center=True).mean().values)
# df_r2 = df_r2.assign(date = ymd2date(df_r2), leads=lambda x: (x.lead-1)//6+1)
# tmp0 = pd.Series(range(1,25))
# tmp1 = (((tmp0-1) // 6)+1)
# tmp2 = ((tmp1-1)*6+1).astype(str) + '-' + (tmp1*6).astype(str)
# di_leads = dict(zip(tmp1, tmp2))
# df_r2.leads = pd.Categorical(df_r2.leads.map(di_leads),list(di_leads.values()))

# ### DAILY R2 TREND ###
# gg_r2_best = (ggplot(df_r2, aes(x='date', y='trend', color='lead', groups='lead.astype(str)')) +
#               theme_bw() + labs(y='R-squared') + geom_line() +
#               theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90),
#                     subplots_adjust={'wspace': 0.25}) +
#               ggtitle('Daily R2 performance by lead (7 day rolling average)') +
#               scale_x_datetime(date_breaks='1 month', date_labels='%b, %Y') +
#               scale_color_cmap(name='Lead',cmap_name='viridis') +
#               facet_wrap('~leads',labeller=label_both))
# gg_save('gg_r2_best.png',dir_figures,gg_r2_best,13,8)

######################################
# --- (3) KERNEL HYPERPARAMETERS --- #

# n_hour = 24*3

# n_kern = dat_kernel.groupby(['cn','kern','coef','lvl']).size().reset_index()

# for i, r in n_kern.iterrows():
#     print('row %i of %i' % (i+1,len(n_kern)))
#     cn, kern, coef, lvl = r['cn'], r['kern'], r['coef'], r['lvl']
#     tmp_data = dat_kernel.query('cn==@cn & kern==@kern & coef==@coef & lvl==@lvl')[cn_keep].reset_index(None,True)
#     tmp_data = tmp_data.assign(qlead=lambda x: pd.cut(x.lead,range(0,25,6)))
#     tmp_fn = 'cn-'+cn+'_'+'kern-'+kern+'_'+'coef-'+coef+'_'+'lvl-'+lvl+'.png'
#     gtit = 'cn='+cn+', '+'kern='+kern+', '+'coef='+coef+', '+'lvl='+lvl
#     gg_tmp = (ggplot(tmp_data,aes(x='dates',y='value',color='lead')) + 
#               theme_bw() + geom_line() + ggtitle(gtit) + 
#               scale_color_cmap(name='Lead') + 
#               facet_wrap('~qlead',nrow=2) + 
#               theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
#               scale_x_datetime(date_breaks='2 month', date_labels='%b, %Y'))
#     gg_save(tmp_fn,dir_kernel,gg_tmp,10,8)

# daily_y = act_y.assign(mu=lambda x: x.y_rt.rolling(n_hour).mean(), se=lambda x: x.y_rt.rolling(n_hour).std(1)).dropna()
# daily_y = daily_y.assign(y=lambda x: (x.y_rt-x.mu)/x.se, dates=lambda x: x.dates.dt.strftime('%Y-%m-%d'))
# daily_y = daily_y.groupby('dates').y.mean().reset_index().assign(dates=lambda x: pd.to_datetime(x.dates))

# cn_keep = ['dates','lead','value']
# # (i) Does distribution in means align with actual y data?
# dat_const = dat_kernel.query('cn=="constant"')[cn_keep].merge(daily_y)
# # dat_const = dat_const.melt(['dates','lead'],['value','y'],'tt')

# gg_const = (ggplot(dat_const,aes(x='y',y='value')) + 
#     theme_bw() + geom_point(size=0.5,alpha=0.5,color='blue') + 
#     facet_wrap('~lead',labeller=label_both,ncol=6))
# gg_save('gg_const.png',dir_kernel,gg_const,16,10)