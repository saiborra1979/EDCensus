# Performs analysis on the most recent year-month-day folder

import os
import copy
import math
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from plotnine import *
from sklearn.metrics import r2_score
from funs_support import ymdh2date, ymd2date, date2ymw, date2ymd, find_dir_olu, gg_color_hue, gg_save, makeifnot
from funs_stats import add_bin_CI, get_CI, ordinal_lbls, prec_recall_lbls, ols
from mdls.gpy import gp_real
import torch
import gpytorch

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_kernel = os.path.join(dir_figures, 'kernel')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
lst_dir = [dir_figures, dir_output, dir_flow, dir_test]
assert all([os.path.exists(z) for z in lst_dir])
makeifnot(dir_kernel)

cn_ymd = ['year', 'month', 'day']
cn_ymdh = cn_ymd + ['hour']
cn_ymdl = cn_ymd + ['lead']

use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

di_desc = {'25%': 'lb', '50%': 'med', '75%': 'ub'}
di_pr = {'prec': 'Precision', 'sens': 'Recall'}
di_lblz = dict(zip(range(4),lblz))
di_metric = {'tp':'TP', 'fp':'FP', 'fn':'FN', 'pos':'Positive'}

colz_gg = ['black'] + gg_color_hue(3)
lblz = ['≥0', '1', '2', '3']

#########################
# --- (1) LOAD DATA --- #

# (i) Load the real-time y
cn_ymdh = ['year', 'month', 'day', 'hour']
idx = pd.IndexSlice
act_y = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), header=[0,1], index_col=[0,1,2,3])
act_y = act_y.loc[:,idx['y','lead_0']].reset_index().droplevel(1,1)
act_y.rename(columns={'y':'y_rt'},inplace=True)
act_y = act_y.assign(date_rt=ymdh2date(act_y)).drop(columns=cn_ymdh)[['date_rt','y_rt']]

# (ii) Find the most recent folder with GPY results
fn_test = pd.Series(os.listdir(dir_test))
fn_test = fn_test[fn_test.str.contains('^[0-9]{4}\\_[0-9]{2}\\_[0-9]{2}$')].reset_index(None,True)
dates_test = pd.to_datetime(fn_test,format='%Y_%m_%d')
df_fn_test = pd.DataFrame({'fn':fn_test,'dates':dates_test,'has_gpy':False})
# Find out which have GP data
for fn in fn_test:
    fn_fold = pd.Series(os.listdir(os.path.join(dir_test,fn)))
    mdls_fold = list(pd.Series(fn_fold.str.split('\\_',2,True).iloc[:,1].unique()).dropna())
    if 'gpy' in mdls_fold:
        df_fn_test.loc[df_fn_test.fn==fn,'has_gpy'] = True
# Get largest date
assert df_fn_test.has_gpy.any()
fold_recent = df_fn_test.loc[df_fn_test.dates.idxmax()].fn
print('Most recent folder: %s' % fold_recent)
path_recent = os.path.join(dir_test,fold_recent)
fn_recent = pd.Series(os.listdir(path_recent))

# (iii) Load the predicted/actual
cn_dates = ['date_rt','date_pred']
fn_res = fn_recent[fn_recent.str.contains('^res\\_.*\\.csv$')]
holder = []
for fn in fn_res:
    holder.append(pd.read_csv(os.path.join(path_recent,fn)))
dat_recent = pd.concat(holder).reset_index(None,True)
dat_recent[cn_dates] = dat_recent[cn_dates].apply(pd.to_datetime)
# dat_recent.insert(0,'dates',pd.to_datetime(dat_recent.date + ' ' + dat_recent.hour.astype(str)+':00:00'))
assert np.all(dat_recent.model == 'gpy')
assert len(dat_recent.groups.unique()) == 1
assert len(dat_recent.dtrain.unique()) == 1
dat_recent.drop(columns = ['model','groups','dtrain'], inplace=True)
dat_recent = dat_recent.sort_values(['lead','date_rt']).reset_index(None,True)
# Add on the date columns
dat_recent = dat_recent.assign(doy=lambda x: x.date_rt.dt.strftime('%Y-%m-%d'))
dat_recent = pd.concat([dat_recent, date2ymw(dat_recent.date_rt)],1)
dat_recent = dat_recent.assign(year = lambda x: np.where((x.woy==53)&(x.month==1), x.year-1, x.year))
# Merge real-time y's
dat_recent = dat_recent.merge(act_y,'left')

# Keep only full week of years
freq_woy = dat_recent.groupby(['woy','year']).size().reset_index().rename(columns={0:'n'})
freq_woy = freq_woy.query('n == n.max()').drop(columns='n')
# Mappings of year/woy to date
lookup_woy = dat_recent.groupby(['year','woy']).date_rt.min().reset_index()
lookup_woy = lookup_woy.merge(freq_woy,'inner')

# Subset to only full weeks
dat_recent = dat_recent.merge(freq_woy,'inner')

# Get the y-ranges
ymi, ymx = dat_recent.pred.min()-1, dat_recent.pred.max()+1
ymi, ymx = min(ymi, act_y.y_rt.min()), max(ymx, act_y.y_rt.max())
esc_bins = [ymi - 1, 31, 38, 48, ymx + 1]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']
esc_lvls = [0,1,2,3]

# Calculate the different escalation levels
cn_esc = ['lead','date_rt','y_rt','y'] #,'date_pred'
act_esc = dat_recent[cn_esc].copy()
tmp = act_esc[['y_rt','y']].apply(lambda x: pd.Categorical(pd.cut(x, esc_bins, False, esc_lbls)).codes)
act_esc = pd.concat([act_esc,tmp.rename(columns={'y':'esc','y_rt':'esc_rt'})],1)
act_esc = act_esc.assign(esc = lambda x: x.esc - x.esc_rt)
act_esc = pd.concat([act_esc,date2ymw(act_esc.date_rt)],1)
act_esc = act_esc.assign(year = lambda x: np.where((x.woy==53)&(x.month==1), x.year-1, x.year))

# Calculate the change in escalation levels
cn_pr = ['date_rt','year','month','lead','y','y_rt','pred','se']
dat_pr = dat_recent[cn_pr].copy()
cn_date, cn_y, cn_y_rt, cn_pred, cn_se = 'date_rt', 'y', 'y_rt', 'pred', 'se'
dat_ord = ordinal_lbls(dat_pr, cn_date=cn_date, cn_y=cn_y, cn_y_rt=cn_y_rt, cn_pred=cn_pred, cn_se=cn_se, level=0.5)
# Double check it lines up with hand calculation
q1 = act_esc.query('lead==10').groupby(['year','month','esc']).size().reset_index()
q2 = dat_ord.query('lead==10').groupby(['year','month','y']).size().reset_index()
q3 = q1.rename(columns={0:'n1','esc':'y'}).merge(q2.rename(columns={0:'n2'}))
assert q3.query('n1 != n2').shape[0] == 0

######################################
# --- (2) FACTORS OF PERFORMANCE --- #

# (i) Spearman performance
perf_rho = dat_recent.groupby(['lead','year','woy']).apply(lambda x: spearmanr(x.y,x.pred)[0])
perf_rho = perf_rho.reset_index().rename(columns={0:'rho'}).merge(lookup_woy)

# # Time trend
# perf_rho.assign(tidx=lambda x: x.groupby('lead').cumcount()+1).groupby('lead').apply(lambda x: ols(y=x.rho.values,x=x.tidx.values)).reset_index()

# (ii) "baseline" easy of prediction: r-squared from using previous day's hour
dmin, dmax = dat_recent.date_rt.min(), dat_recent.date_rt.max()
yshift = act_y.query('date_rt>=@dmin & date_rt<=@dmax').assign(y_lead=lambda x: x.y_rt.shift(24)).dropna()
yshift.y_lead = yshift.y_lead.astype(int)
yshift = pd.concat([yshift,date2ymw(yshift.date_rt)],1).reset_index(None,True)
yshift = yshift.assign(hour=lambda x: x.date_rt.dt.hour)
bl_rho = yshift.groupby(['year','woy']).apply(lambda x: r2_score(x.y_rt,x.y_lead)).reset_index()
bl_rho = bl_rho.rename(columns={0:'r2'}).merge(lookup_woy)

# (ii) (a) Average level changes, (b) number of escalations, (c) number of esc jumps > 1


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

# (4.i) Spearman's correlation over time

gg_rho_time = (ggplot(perf_rho,aes(x='date_rt',y='rho')) + 
    theme_bw() + geom_line() + 
    facet_wrap('~lead',labeller=label_both,nrow=4) + 
    labs(y="Spearman's rho (weekly)") + 
    geom_line(aes(y='r2'),data=bl_rho,color='red') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
    scale_x_datetime(date_breaks='2 months',date_labels='%b, %y'))
gg_save('gg_rho_weekly.png',dir_figures,gg_rho_time,12,8)


gg_bl_rho = (ggplot(bl_rho,aes(x='date_rt',y='rho')) + 
    theme_bw() + geom_line() + 
    facet_wrap('~lead',labeller=label_both,nrow=4) + 
    labs(y="Spearman's rho (weekly)") + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
    scale_x_datetime(date_breaks='2 months',date_labels='%b, %y'))
gg_save('gg_rho_weekly.png',dir_figures,gg_rho_time,12,8)





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