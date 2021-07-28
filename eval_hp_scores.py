# Calls class from ~/mdls folder
import argparse
from plotnine.geoms.geom_hline import geom_hline
from plotnine.geoms.geom_linerange import geom_linerange
from plotnine.labels import ggtitle

from plotnine.themes.themeable import axis_text_x
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
import plotnine as pn
from funs_support import find_dir_olu, read_pickle, drop_zero_var, date2ymw, gg_save
from funs_esc import esc_bins, esc_lbls, get_esc_levels
from funs_stats import prec_recall_lbls, get_reg_score


dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
lst_dir = [dir_figures, dir_output, dir_flow, dir_test]
assert all([os.path.exists(z) for z in lst_dir])

cn_woy = ['year','woy']
cn_drop = cn_woy+['month']
cn_date =['date_rt','date_pred']

##########################
# --- (1) LOAD MODEL --- #

di_mdl = dict.fromkeys(model_list)

for model in model_list:
    dir_model = os.path.join(dir_test, model)
    dir_mdl = os.path.join(dir_model, 'model')
    fn_mdl = pd.Series(os.listdir(dir_mdl))
    di_mdl[model] = fn_mdl


###########################
# --- (2) LOAD SCORES --- #

holder = []
for model in model_list:
    print('model = %s' % model)
    dir_model = os.path.join(dir_test, model)
    fn_mdl = di_mdl[model].str.replace('mdl_','')
    holder_scores = []
    for fn in fn_mdl:
        path_pickle = os.path.join(dir_model, fn)
        di_fn = read_pickle(path_pickle)
        df_hp = di_fn['hp']
        df_scores = di_fn['scores']
        # print(fn)
        # print(df_scores.date_rt.min())
        # Multicolumn
        cn_scores = pd.MultiIndex.from_product([['base'],df_scores.columns])
        cn_hp = pd.MultiIndex.from_frame(df_hp.drop(columns='val'))
        df_scores.columns = cn_scores
        df_hp = df_hp.drop(columns=['tt','hp']).T.values
        df_hp = pd.DataFrame(np.tile(df_hp,[len(df_scores),1]),columns=cn_hp)
        df_scores = pd.concat([df_scores,df_hp],1)
        holder_scores.append(df_scores)
        del di_fn, df_scores
    # Merge
    res_scores = pd.concat(holder_scores).reset_index(None, True)
    res_scores.insert(0,'model',model)
    res_scores.columns = res_scores.columns.insert(0,('base','model')).drop(('model',''))    
    holder.append(res_scores)
    del res_scores
# Merge all
res_all = pd.concat(holder).reset_index(None, True)
# Get rid of zero var features
res_all = drop_zero_var(res_all)
# Drop multiindex
res_all.columns = res_all.columns.droplevel(0)
res_all.drop(columns=cn_drop, inplace=True)
res_all = res_all.sort_values(['lead','date_rt']).reset_index(None, True)
# Assign Year/WoY information
res_all = pd.concat([date2ymw(res_all['date_rt'], False), res_all],1)


#############################
# --- (3) LOAD BASELINE --- #

res_bl = pd.read_csv(os.path.join(dir_test, 'bl_scores.csv'))
assert len(np.setdiff1d(res_bl.columns, res_all.columns)) == 0
res_bl = res_bl.sort_values(['lead','date_rt']).reset_index(None, True)
# Datetime
res_bl[cn_date] = res_bl[cn_date].apply(pd.to_datetime)
# Assign Year/WoY information
res_bl = pd.concat([date2ymw(res_bl['date_rt'],False), res_bl],1)

# Get weeks
woy_n = res_bl.groupby(cn_woy).date_rt.apply(lambda x: pd.DataFrame({'x':x.min(),'n':len(x)},index=[0]))
woy_n = woy_n.reset_index().drop(columns='level_2')
woy_n = woy_n.query('n==n.max()').reset_index(None, True).drop(columns='n')


##################################
# --- (4) WEEKLY PERFORMANCE --- #

cn_y = ['y_delta', 'pred_delta', 'date_rt']
cn_gg = ['lead','year','woy','x']
cn_ord = cn_y + cn_gg

# Remove non-full weeks
res_bl = res_bl.merge(woy_n,'inner')
res_all = res_all.merge(woy_n,'inner')

# Get precision/recall
pr_bl = prec_recall_lbls(res_bl[cn_ord], 'y_delta', 'pred_delta', 'date_rt', 100)
pr_all = prec_recall_lbls(res_all[cn_ord], 'y_delta', 'pred_delta', 'date_rt', 100)
pr_bl = pr_bl.query('pred_delta==1').drop(columns='pred_delta')
pr_all = pr_all.query('pred_delta==1').drop(columns='pred_delta')
pr_both = pd.concat([pr_all.assign(tt='Model'), pr_bl.assign(tt='Baseline')],0)
pr_both['tt'] = pd.Categorical(pr_both.tt,['Model','Baseline'])

# Get regression metrics
reg_all = res_all.groupby(['lead']+cn_woy).apply(get_reg_score,add_n=True).reset_index()
reg_all.drop(columns=['n'], inplace=True)
reg_bl = res_bl.groupby(['lead']+cn_woy).apply(get_reg_score,add_n=True).reset_index()
reg_bl.drop(columns=['n'], inplace=True)
reg_both = pd.concat([reg_all.assign(tt='Model'), reg_bl.assign(tt='Baseline')],0)
reg_both['tt'] = pd.Categorical(reg_both.tt,['Model','Baseline'])
reg_both = reg_both.melt(['lead','year','woy','tt'],None,'metric')
reg_both = reg_both.merge(woy_n,'left')


###################################
# --- (5) ROLLING PERFORMANCE --- #

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint as prop_CI
from funs_support import gg_color_hue

prec_target = 0.5
p_seq = np.round(np.arange(0.05,1,0.05),2)
q_seq = norm.ppf(p_seq)
di_pq = dict(zip(p_seq, q_seq))

# How many days to use to calibrate "threshold"
dval = 14
val_offset = pd.DateOffset(days=dval)
cn_scores = ['lead','pred','se','date_rt','y_rt','y']
cn_keep = ['p','lead','date_rt','y_delta','pred_delta']
mdl_scores = res_all[cn_scores].copy()
dmin = mdl_scores.date_rt.min()
dmax = mdl_scores.date_rt.max()
drange = pd.date_range(dmin, dmax, freq='1D') # str(dval)+
drange = drange[drange >= dmin+val_offset]

holder = []
for ii in range(1,len(drange)-1):
    dmed = drange[ii]
    dlow = dmed - val_offset
    dhigh = drange[ii+1]
    print('%i of %i' % (ii, len(drange)-2))
    # break
    # (i) Use training sequence
    scores_dval = mdl_scores.query('date_rt >= @dlow & date_rt < @dmed')
    scores_dval = get_esc_levels(scores_dval,['y','y_rt'],esc_bins, esc_lbls)
    scores_dval = pd.concat([scores_dval.assign(pred=lambda x: x.pred+q*x.se, p=p) for q,p in zip(q_seq,p_seq)])
    scores_dval = get_esc_levels(scores_dval,['pred'],esc_bins, esc_lbls)
    scores_dval = scores_dval.assign(y_delta=lambda x: np.sign(x.esc_y - x.esc_y_rt),
                pred_delta = lambda x: np.sign(x.esc_pred - x.esc_y_rt) )
    scores_dval = scores_dval[cn_keep]
    prec_dval = prec_recall_lbls(scores_dval, 'y_delta', 'pred_delta', 'date_rt')
    prec_dval = prec_dval.query('pred_delta==1 & metric=="prec"')
    prec_dval.drop(columns=['pred_delta','metric'], inplace=True)
    prec_dval = prec_dval.assign(e2=lambda x: (x.value-prec_target)**2)
    pstar = prec_dval.groupby('lead').apply(lambda x: x.loc[x.e2.idxmin()])
    pstar = pstar.reset_index(None, True)[['lead','p']].assign(lead=lambda x: x.lead.astype(int))
    pstar = pstar.assign(p=lambda x: x.p.min())

    # (ii) Apply to "test" week
    scores_test = mdl_scores.query('date_rt >= @dmed & date_rt < @dhigh')
    scores_test = scores_test.merge(pstar).assign(q=lambda x: x.p.map(di_pq))
    scores_test = scores_test.assign(pred=lambda x: x.pred+x.q*x.se)
    # scores_test.drop(columns=['se','p','q'], inplace=True)
    scores_test = get_esc_levels(scores_test,['y','y_rt','pred'],esc_bins, esc_lbls)
    scores_test = scores_test.assign(y_delta=lambda x: np.sign(x.esc_y - x.esc_y_rt),
                    pred_delta = lambda x: np.sign(x.esc_pred - x.esc_y_rt) )
    scores_test = scores_test[cn_keep].assign(date=dmed)
    holder.append(scores_test)
# Merge and get weekly precision
esc_test = pd.concat(holder).reset_index(None,True)
esc_test = pd.concat([date2ymw(esc_test['date'],False), esc_test], 1)
esc_test.drop(columns=['date', 'p'], inplace=True)
pr_test = prec_recall_lbls(esc_test, 'y_delta', 'pred_delta', 'date_rt')
pr_test = pr_test.query('pred_delta==1').drop(columns='pred_delta')
pr_test = pr_test.merge(woy_n,'left').sort_values(['metric','lead','x'])
tmp = pd.concat(prop_CI(pr_test['den']*pr_test['value'], pr_test['den'], 0.05, 'beta'),1)
tmp.columns = ['lb', 'ub']
pr_test = pd.concat([pr_test,tmp],1)
pr_test = pr_test.assign(cover=lambda x: pd.Categorical(x.ub >= prec_target,[True,False]))

# # Merge precision/recall from test

# pr_test = pr_test.assign(aerr=lambda x: np.abs(prec_target - x.value))
# pr_test.groupby(['metric','lead']).cover.mean()
# pr_test.groupby(['metric','lead']).aerr.mean()

# Show precision for 24-hour ahead
tmp_data = pr_test.query('metric=="prec" & lead==12').drop(columns=['metric','lead'])
colz = [gg_color_hue(2)[1], gg_color_hue(2)[0]]
gg_prec_12 = (pn.ggplot(tmp_data,pn.aes(x='x',y='value',color='cover')) + 
    pn.theme_bw() + pn.geom_point(size=2) + 
    pn.scale_color_manual(name='≥ 50%',values=colz) + 
    pn.ggtitle('Rolling weekly precision (12 hour forecast)\nVertical lines shows 95% CI') + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),size=0.5) + 
    pn.geom_hline(yintercept=prec_target,color='black',linetype='--') + 
    pn.theme(axis_title_x=pn.element_blank(), axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_x_datetime(date_breaks='2 months',date_labels='%b, %y') + 
    pn.labs(y='Precision'))
gg_save('gg_prec_12.png', dir_figures, gg_prec_12, 6, 4)



#######################
# --- (6) FIGURES --- #

di_metric = {'prec':'Precision', 'sens':'Recall', 'MAE':'MAE','spearman':'Correlation'}

gg_pr_rolling = (pn.ggplot(pr_test,pn.aes(x='x',y='value',color='lead')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.geom_hline(yintercept=prec_target) + 
    pn.theme(axis_title_x=pn.element_blank(), axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_x_datetime(date_breaks='2 months',date_labels='%b, %y') + 
    pn.facet_wrap('~metric',labeller=pn.labeller(metric=di_metric),nrow=1) + 
    pn.labs(y='Percent'))
gg_save('gg_pr_rolling.png', dir_figures, gg_pr_rolling, 12, 5)




gg_pr_weekly = (pn.ggplot(pr_both, pn.aes(x='x', y='value', color='tt')) + 
    pn.theme_bw() + pn.labs(y='Percent') + 
    pn.theme(axis_title_x=pn.element_blank()) + 
    pn.geom_line() + 
    pn.scale_color_manual(name='Type',values=['red','black']) + 
    pn.facet_grid('lead~metric',labeller=pn.labeller(metric=di_metric)))
gg_save('gg_pr_weekly.png', dir_figures, gg_pr_weekly, 12, 24)

gg_reg_weekly = (pn.ggplot(reg_both, pn.aes(x='x', y='value', color='tt')) + 
    pn.theme_bw() + pn.labs(y='Value') + 
    pn.theme(axis_title_x=pn.element_blank(), axis_text_x=pn.element_text(angle=90)) + 
    pn.geom_line() + 
    pn.scale_x_datetime(date_breaks='3 months',date_labels='%m, %y') + 
    pn.scale_color_manual(name='Type',values=['red','black']) + 
    pn.facet_grid('metric~lead',labeller=pn.labeller(metric=di_metric),scales='free_y'))
gg_save('gg_reg_weekly.png', dir_figures, gg_reg_weekly, 36, 6)

















