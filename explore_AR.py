"""
FUNCTION TO ANALYZE THE TIME SERIES PROPERTIES OF THE Y OUTCOME
calibrated NN: https://arxiv.org/pdf/1803.09546.pdf
weighted trees: http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
"""

import os
import numpy as np
import pandas as pd
from time import time
from funs_support import gg_save, find_dir_olu

import matplotlib
import plotnine
from plotnine import *

from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from scipy.optimize import minimize_scalar

def find_alpha(alpha, dist, target):
    err = np.exp(-alpha*dist).sum() - target
    return err**2

# https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method
def weighted_quantile(x, weights, quantile=0.5):
    s_data, s_weights = map(np.array, zip(*sorted(zip(x, weights))))
    midpoint = quantile * sum(s_weights)
    if any(weights > midpoint):
        w_q = (x[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_q = np.mean(s_data[idx:idx+2])
        else:
            w_q = s_data[idx+1]
    return w_q


dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'mlmd')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
lst_dir = [dir_figures, dir_output]
assert all([os.path.exists(z) for z in lst_dir])

# --- LOAD DATA --- #
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), header=[0,1], index_col=[0,1,2,3])
# Get datetime from index
dates = pd.to_datetime(df_lead_lags.index.to_frame().astype(str).apply(lambda x: x.year+'-'+x.month+'-'+x.day+' '+x.hour+':00:00',1))
df_lead_lags.index = pd.MultiIndex.from_frame(pd.concat([df_lead_lags.index.to_frame(),
                                        pd.DataFrame(dates).rename(columns={0:'date'})],1))
#Ancillary
idx = pd.IndexSlice
cn_date = ['year', 'month', 'day', 'hour']

################################
# --- Q1: R2 + Coef by Lag --- #

nlags = 10
lags = np.arange(nlags)+1
y = df_lead_lags.loc[:,idx['y','lag_0']]

holder = []
for ll in lags:
    print('Lag %i of %i' % (ll, nlags))
    mdl = AutoReg(endog=y.values, lags=[ll], missing='drop', hold_back=nlags, trend='c').fit()
    rsq = 1-mdl.sigma2/mdl.model.endog.var()
    tmp = pd.DataFrame({'rsq':rsq, 'lag':ll, 'tt':['int','coef'], 'bhat':mdl.params, 'se':mdl.bse})
    holder.append(tmp); del tmp
dat_r2 = pd.concat(holder).reset_index(None,True).assign(zscore=lambda x: x.bhat/x.se)

############################################
# --- Q2: Quality of coverage by AR(p) --- #

# Is coverage as bad historically?
alpha = 0.05
crit = norm.ppf(1-alpha/2)
X = sm.add_constant(y.shift(6).values)
linreg = sm.OLS(endog=y.values, exog=X, hasconst=True, missing='drop').fit()
se_e = np.sqrt(linreg.mse_resid)
dat_cof = pd.DataFrame({'y':linreg.model.endog, 'yhat':linreg.fittedvalues, 'res':linreg.resid})
dat_cof = dat_cof.assign(lb=lambda x: x.yhat-crit*se_e,ub=lambda x: x.yhat+crit*se_e)
dat_cof = dat_cof.assign(viol_lb = lambda x: x.lb<=x.y, viol_ub=lambda x: x.ub>=x.y)
print(dat_cof[['viol_lb','viol_ub']].mean())
# Get the error distribution by bin
enc = KBinsDiscretizer(n_bins=10,strategy='quantile',encode='ordinal').fit(dat_cof[['y']])
dat_cof['yq'] = enc.transform(dat_cof[['y']]).astype(int).flatten()
print(dat_cof.groupby('yq')[['viol_lb','viol_ub']].mean().reset_index())

##########################################
# --- Q3: LINEAR QUANTILE REGRESSION --- #

mdl = smf.quantreg(formula='lead_6~lag_0', data=df_lead_lags.loc[:,idx['y',['lead_6','lag_0']]].droplevel(0,axis=1))
res = mdl.fit(q=0.05)
enc = KBinsDiscretizer(n_bins=5,strategy='quantile',encode='ordinal').fit(np.atleast_2d(res.model.endog).T)
dat_qreg = pd.DataFrame({'viol':res.model.endog < res.fittedvalues,'y':res.model.endog,'yhat':res.fittedvalues,
                         'yq':enc.transform(np.atleast_2d(res.model.endog).T).flatten().astype(int)})

###########################################
# --- Q4: BOOSTED QUANTILE REGRESSION --- #

mdl = GBR(loss='quantile', learning_rate=0.1, n_estimators=250, max_depth=3,
          min_samples_leaf=5, min_samples_split=5, alpha=0.05)
X = df_lead_lags.loc[:,idx[:,['lag_'+str(z) for z in range(nlags+1)]]].values
y = df_lead_lags.loc[:,idx['y','lead_6']].values
enc = KBinsDiscretizer(n_bins=5,strategy='quantile').fit(np.atleast_2d(y).T)
enc2 = KBinsDiscretizer(n_bins=5,strategy='quantile',encode='ordinal').fit(np.atleast_2d(y).T)
X = np.hstack([X,enc.transform(np.atleast_2d(y).T).toarray().astype(int)])
mdl.fit(X=X, y=y)
print(np.mean(y < mdl.predict(X)))
dat_qreg_GBR = pd.DataFrame({'viol':y < mdl.predict(X),'yhat':mdl.predict(X),'y':y,
                             'yq':enc2.transform(np.atleast_2d(y).T).flatten().astype(int)})
print(dat_qreg_GBR.groupby('yq').viol.mean())

##################################
# --- Q5: QUANTILE SMOOTHING --- #

# Fit a simple linear regression model
X = df_lead_lags.loc[:,idx[:,['lag_'+str(z) for z in range(nlags+1)]]].values
y = df_lead_lags.loc[:,idx['y','lead_6']].values
n = X.shape[0]

mdl = LinearRegression().fit(X,y)
dat = pd.DataFrame({'y':y,'yhat':mdl.predict(X)}).sort_values('yhat').reset_index(None,True)
# Normalize both
dat = pd.concat([dat,pd.DataFrame(StandardScaler().fit_transform(dat),columns=dat.columns).add_prefix('z_')],1)
nsub = 1000
sub = dat.sample(n=nsub,random_state=1234).sort_values('yhat').reset_index(None,True)

holder, kernel = np.zeros(nsub), np.zeros([nsub,2])
stime = time()
for ii in range(nsub):
    if (ii+1) % 250 == 0:
        rate = (ii+1) / (time() - stime)
        eta = (nsub - (ii+1)) / rate / 60
        print('Iteration %i of %i (ETA %0.1f minutes)' % (ii+1, nsub, eta))
    dist = (sub.z_yhat[ii]-sub.z_yhat)**2
    # Find a "normalized" weights ~ 200
    alpha_ii = minimize_scalar(fun=find_alpha, args=(dist, 20*np.log(dist.sum())), method='brent').x
    weights = np.exp(-alpha_ii*dist)**2  # For distance use "predicted" values
    holder[ii] = weighted_quantile(x=sub.y.values, weights=weights, quantile=0.05)
# Append on and visualize
sub['lb'] = holder

#########################################################
# --- Q6: ERROR-RANGE FOR ONE-DAY AHEAD PREDICTIONS --- #

cn_lead = ['lead_'+str(z+1) for z in range(nlags)]
cn_lag = ['lag_'+str(z) for z in range(nlags+1)]
Y = df_lead_lags.loc[:,idx['y',cn_lead]].values
X = df_lead_lags.loc[:,idx[:,cn_lag]].values
rho = np.corrcoef(X,rowvar=False)
rho2 = np.tril(rho,k=-1)

# Train on 2018-19 and test on 2020
idx_dates = pd.date_range(start='2020-01-01',end=dates.max(), freq='d')

holder = []
for ii, dd in enumerate(idx_dates):
    if (ii + 1) % 10 == 0:
        print('Test day %i of %i: %s' % (ii+1, len(idx_dates), dd.strftime('%b %d, %Y')))
    # Get train/test index
    idx_train = df_lead_lags.index.get_level_values('date') < dd
    dd2 = dd + pd.DateOffset(days=1)
    idx_test = (df_lead_lags.index.get_level_values('date') >= dd) & \
                    (df_lead_lags.index.get_level_values('date') < dd2)
    # Fit model on the different horizons
    Xtrain, Xtest = X[idx_train], X[idx_test]
    Ytrain, Ytest = Y[idx_train], Y[idx_test]
    mdl = Ridge(alpha=0.5,normalize=True).fit(X=Xtrain,y=Ytrain)
    tmp_pred = pd.DataFrame(mdl.predict(Xtest),columns=cn_lead).add_prefix('yhat_')
    tmp_true = pd.DataFrame(Ytest,columns=cn_lead).add_prefix('ytrue_')
    tmp = pd.concat([tmp_pred, tmp_true], 1).assign(date=df_lead_lags.index.get_level_values('date')[idx_test])
    holder.append(tmp)
dat_res = pd.concat(holder).reset_index(None, True)
del holder
dat_res = dat_res.melt('date',None,'tmp').assign(tt=lambda x: x.tmp.str.split('_',1,True).iloc[:,0],
                                                 lead=lambda x: x.tmp.str.split('_',2,True).iloc[:,2])
dat_res = dat_res.drop(columns=['tmp']).assign(lead=lambda x: x.lead.astype(int))
dat_wide = dat_res.pivot_table('value',['date','lead'],'tt').reset_index()
dat_wide = dat_wide.assign(res=lambda x: x.ytrue - x.yhat,
                           month=lambda x: x.date.dt.strftime('%m').astype(int),
                           hour=lambda x: x.date.dt.strftime('%H').astype(int))


dat_sum = dat_wide.groupby('lead').apply(lambda x:
           pd.Series({'mse':mse(x.ytrue, x.yhat),'r2':r2(x.ytrue, x.yhat)}))

# Error distribution by lead
gg_err_lead = (ggplot(dat_wide[dat_wide.lead<=9], aes(x='res',fill='month.astype(str)')) +
               geom_density(color='black',alpha=0.5) + theme_bw() +
               facet_wrap('~lead',ncol=3, labeller=label_both,scales='free_x') +
               ggtitle('Error distribution by horizon for 2020') +
               theme(panel_spacing_x=0.5) +
               scale_fill_brewer(palette='Blues',name='Month') +
               geom_vline(xintercept=-10,linetype='--') + geom_vline(xintercept=+10,linetype='--'))
gg_err_lead.save(os.path.join(dir_figures,'gg_err_lead.png'),width=11,height=7)

# Scatterplot
from mizani import palettes
colz = palettes.brewer_pal(palette=1)(dat_wide.month.unique().shape[0]+2)[2:]

gg_scat_lead = (ggplot(dat_wide[dat_wide.lead<=9], aes(x='yhat', y='ytrue',color='month.astype(str)')) +
               geom_point(alpha=0.5) + theme_bw() +
               facet_wrap('~lead',ncol=3, labeller=label_both,scales='free_x') +
               ggtitle('One-day-ahead prediction for 2020') +
               theme(panel_spacing_x=0.3) +
               scale_fill_brewer(palette='Blues',name='Month') +
               labs(x='Predicted',y='Actual') +
               geom_abline(intercept=0,slope=1,color='black') +
               scale_color_manual(name='Month',values=colz))
gg_scat_lead.save(os.path.join(dir_figures,'gg_scat_lead.png'),width=9,height=7)

#########################
# --- SANITY CHECKS --- #

# Check that we can replicate what statsmodels is doing under the hood
mdl_lag5 = AutoReg(endog=y.values,lags=[5], missing='drop', trend='c', seasonal=False).fit() #, hold_back=nlags
mdl_ls5 = sm.OLS(endog=y.values, exog=sm.add_constant(y.shift(5).values),hasconst=True, missing='drop').fit()
# The "innovations" are just the standard deviation of the residuals
print(np.sqrt(np.mean( (mdl_lag5.resid)**2 )))
# R-squared
print(1-mdl_lag5.sigma2/mdl_lag5.model.endog.var())
check = pd.DataFrame({'bhat_ar':mdl_lag5.params, 'bhat_ls':mdl_ls5.params,
                      'se_ar':mdl_lag5.bse, 'se_ls':mdl_ls5.bse})
print(check)


###########################
# --- STEP 7: FIGURES --- #

# --- (ii) R2 and Coef by Lag --- #
crit = norm.ppf(1-0.05/2)
tmp = dat_r2[dat_r2.tt=='coef'].melt(['lag','se'],['bhat','rsq'],'tt').assign(se=lambda x: np.where(x.tt=='rsq',np.NaN, x.se))
tmp = tmp.assign(lb = lambda x: x.value-crit*x.se, ub=lambda x: x.value+crit*x.se)
gg_r2_coef = (ggplot(tmp, aes(x='lag',y='value')) + geom_point() +
              theme_bw() + facet_wrap('~tt',labeller=labeller(tt={'rsq':'R2','bhat':'Coef'})) +
              labs(x='Lag order',y='Value') +
              scale_x_continuous(breaks=list(lags)) +
              ggtitle('AR(p) model coefficient and results'))
gg_r2_coef.save(os.path.join(dir_figures, 'gg_r2_coef.png'), width=7, height=4)

# --- (iii) Error distribution --- #
gg_err_dist = (ggplot(dat_cof, aes(x='yq.astype(str)', y= 'res')) +
               geom_boxplot() + theme_bw() +
               labs(x='Max hours quintile',y='Residual') +
               ggtitle('Distribution of errors by response level'))
gg_err_dist.save(os.path.join(dir_figures,'gg_err_dist.png'))

gg_err_ord = (ggplot(dat_cof, aes(x='yhat',y='y')) +
              geom_point(size=0.1,alpha=0.5) + geom_density_2d(color='blue',levels=10) +
              theme_bw())
gg_err_ord.save(os.path.join(dir_figures,'gg_err_ord.png'))

# --- (iv) Linear Quantile Regression --- #
tmp = dat_qreg.groupby('yq').viol.mean().reset_index()
gg_qreg = (ggplot(tmp, aes(x='yq.astype(str)',y='viol')) +
           theme_bw() + geom_point() +
           labs(x='Max hours quantile',y='Violation percentage') +
           ggtitle('Quantile regression errors at 5% levels') +
           geom_hline(yintercept=0.05,linetype='--',color='blue'))
gg_qreg.save(os.path.join(dir_figures,'gg_qreg.png'))

# --- (v) GBR Quantile Regression --- #

gg_qreg_linear = (ggplot(dat_qreg, aes(x='yhat',y='y')) + theme_bw()+
               geom_point(size=0.1,alpha=0.5) +
               geom_abline(intercept=0,slope=1,color='blue'))
gg_qreg_linear.save(os.path.join(dir_figures,'gg_qreg_linear.png'))

gg_qreg_GBR = (ggplot(dat_qreg_GBR, aes(x='yhat',y='y')) + theme_bw()+
               geom_point(size=0.1,alpha=0.5) +
               geom_abline(intercept=0,slope=1,color='blue'))
gg_qreg_GBR.save(os.path.join(dir_figures,'gg_qreg_GBR.png'))

# --- (vi) Non-parametric quantile --- #

gg_np_quant = (ggplot(sub, aes(x='z_yhat', y='y')) + theme_bw() +
                  geom_point(size=0.2, alpha=0.5) +
                  labs(x='Predicted (normalized)', y='Actual') +
                  geom_line(aes(x='z_yhat',y='lb'),color='blue',data=sub) +
                  ggtitle('Non-parametric quantile'))
gg_np_quant.save(os.path.join(dir_figures,'gg_np_quant.png'))