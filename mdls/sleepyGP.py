"""
GAUSSIAN PROCESS WHICH TUNES OVER BOTH THE LENGTH SCALE AND THE TRAINING WINDOW
"""

import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances
from funs_support import cvec

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

# Beta testing
from plotnine import *
import seaborn as sns
ktrain = 500
y1, y2, y3 = y_train[-ktrain:].copy(), y_valid.copy(), y_test.copy()
t1, t2, t3 = tmat[idx_train][-500:].copy(), tmat[idx_valid].copy(), tmat[idx_test].copy()
t1, t2, t3 = t1.reset_index(None,True).droplevel(1,1), t2.reset_index(None,True).droplevel(1,1), t3.reset_index(None,True).droplevel(1,1)
yy = pd.Series(y1).append(pd.Series(y2)).append(pd.Series(y3))

# Beta testing
df = pd.concat([t1,t2,t3],0).assign(y=yy)
ii = (df.index - df.reset_index().index)
df.insert(0,'tt',np.where(ii == 0, 'train', np.where(ii == -500, 'valid', 'test')))
df['hour'] = df.groupby('trend').cumcount()
df = df.reset_index(None,True).reset_index()

# Try noise downward sinusodial
np.random.seed(1234)
df = df[['index','tt']].assign(y=np.sin(df.index/25) +
                                 0.25*np.random.randn(df.shape[0]) +
                                 np.linspace(0,-1,df.shape[0]),
                               tt=lambda x: np.where(x.tt != 'train','test','train'))
# Set up the GP kenrel
gp_kernel = ExpSineSquared(length_scale=1, periodicity=25)
#                length_scale_bounds=(1e-2, 10), periodicity_bounds = (1e-2,24))
gp_kernel += WhiteKernel(noise_level=15)
gp_kernel += 1**2*RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=10, random_state=1234)
gpr.fit(X=cvec(df[df.tt=='train'].index),y=df[df.tt=='train'].y)
# Look at the new kernel estimates
print(gpr.kernel_)
mu, se = gpr.predict(cvec(df.index),True)
# fig = sns.distplot(gpr.sample_y(cvec(df.index[0]),n_samples=1000))
# fig.figure.savefig(os.path.join(dir_figures, 'tmp1.png'))


crit = norm.ppf(0.975)
dat = df.assign(yhat = mu, se=se, lb=mu-crit*se, ub=mu+crit*se)
dat_long = dat.melt(['index','tt','lb','ub'],['y','yhat'],'vv')
print(dat.groupby('tt').apply(lambda x: r2_score(x.y, x.yhat)))
print(dat.assign(acc=lambda x: (x.lb < x.y) & (x.ub > x.y)).groupby('tt').acc.mean())

# Plot
gg_ts = (ggplot(dat_long, aes(x='index', y='value', color='vv')) + theme_bw() +
         geom_line(alpha=0.5) + geom_point(size=1) +
         geom_vline(xintercept=ktrain) +
         geom_ribbon(aes(ymin='lb',ymax='ub'),color='black',fill='blue',alpha=0.2))
gg_ts.save(os.path.join(dir_figures, 'tmp.png'),width=8, height=5)

