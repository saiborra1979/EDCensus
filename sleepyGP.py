#################################
# --- MULTITASK EXPERIMENTS --- #

import os
from funs_support import gg_save, t2n, find_dir_olu

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')

import torch, gpytorch
from gpytorch.models import ExactGP, 
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import RBFKernel, IndexKernel #, LinearKernel, CosineKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import max_cg_iterations, fast_pred_var

from plotnine import *
import math
import random
import numpy as np
import pandas as pd
from copy import copy

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ntasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.task_covar_module = IndexKernel(num_tasks=ntasks, rank=1)

    # self, x, i = model, full_train_x, full_train_i
    def forward(self,x,i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)

ntrain = 50
p = 4
k = 3
torch.manual_seed(ntrain)
random.seed(ntrain)

# Data
full_train_x = [torch.rand(ntrain,p) for j in range(k)]
full_train_i = torch.cat([torch.full_like(x, dtype=torch.long, fill_value=j)  for j, x in enumerate(full_train_x)])
full_train_x = torch.cat(full_train_x)
full_train_y = torch.rand(ntrain*k)
assert full_train_x.shape == full_train_i.shape
assert len(full_train_y) == len(full_train_y)

# Model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood, ntasks=k)
model.train()
likelihood.train()
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training
training_iterations = 50
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

# Inference
model.eval()
likelihood.eval()
test_x1, test_x2 = copy(train_x1), copy(train_x2)
test_i_task1 = torch.full_like(test_x1, dtype=torch.long, fill_value=0)
test_i_task2 = torch.full_like(test_x2, dtype=torch.long, fill_value=1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y1 = likelihood(model(test_x1, test_i_task1))
    observed_pred_y2 = likelihood(model(test_x2, test_i_task2))

df = pd.DataFrame({'x':t2n(torch.cat([test_x1.mean(1),test_x2.mean(1)])).flat,'y':t2n(full_train_y),
                'pred':t2n(torch.cat([observed_pred_y1.mean,observed_pred_y2.mean])),
                'se':t2n(torch.cat([observed_pred_y1.stddev,observed_pred_y2.stddev])),
                'tt':t2n(torch.cat([test_i_task1.sum(1),test_i_task2.sum(1)]).flatten())})
df.tt = df.tt.astype(str)

gg_gp_test = (ggplot(df,aes('x',color='tt',fill='tt')) + theme_bw() + 
    geom_point(aes(y='y'),color='black') + 
    geom_line(aes(y='pred')) + 
    geom_ribbon(aes(ymin='pred-2*se',ymax='pred+2*se'),alpha=0.5))
gg_save('gg_gp_test.png',dir_figures,gg_gp_test,5,4)



# import numpy as np
# import pandas as pd
# from time import time
# from scipy.stats import norm
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import euclidean_distances
# from funs_support import cvec

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

# # Beta testing
# from plotnine import *
# import seaborn as sns
# ktrain = 500
# y1, y2, y3 = y_train[-ktrain:].copy(), y_valid.copy(), y_test.copy()
# t1, t2, t3 = tmat[idx_train][-500:].copy(), tmat[idx_valid].copy(), tmat[idx_test].copy()
# t1, t2, t3 = t1.reset_index(None,True).droplevel(1,1), t2.reset_index(None,True).droplevel(1,1), t3.reset_index(None,True).droplevel(1,1)
# yy = pd.Series(y1).append(pd.Series(y2)).append(pd.Series(y3))

# # Beta testing
# df = pd.concat([t1,t2,t3],0).assign(y=yy)
# ii = (df.index - df.reset_index().index)
# df.insert(0,'tt',np.where(ii == 0, 'train', 'test'))
# df['hour'] = df.groupby('trend').cumcount()
# df = df.reset_index(None,True).reset_index()
# enc = StandardScaler().fit(cvec(df.y[df.tt=='train']))
# df['y'] = enc.transform(cvec(df.y)).flatten()

# # # Try noise downward sinusodial
# # np.random.seed(1234)
# # df = df[['index','tt']].assign(y=np.sin(df.index/25) +
# #                                  0.25*np.random.randn(df.shape[0]) +
# #                                  np.linspace(0,-1,df.shape[0]),
# #                                tt=lambda x: np.where(x.tt != 'train','test','train'))

# # Set up the GP kenrel
# gp_kernel = ExpSineSquared(length_scale=1, periodicity=25)
# gp_kernel += WhiteKernel(noise_level=15)
# gp_kernel += 1**2*RBF(length_scale=1)
# gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=1234)
# gpr.fit(X=cvec(df[df.tt=='train'].index),y=df[df.tt=='train'].y)
# gpr.kernel_.k1.k2.noise_level = 1
# # Look at the new kernel estimates
# print(gpr.kernel_)

# mu, se = gpr.predict(cvec(df.index),True)

# crit = norm.ppf(0.975)
# dat = df.assign(yhat = mu, se=se, lb=mu-crit*se, ub=mu+crit*se)
# dat_long = dat.melt(['index','tt','lb','ub'],['y','yhat'],'vv')
# print(dat.groupby('tt').apply(lambda x: r2_score(x.y, x.yhat)))
# print(dat.assign(acc=lambda x: (x.lb < x.y) & (x.ub > x.y)).groupby('tt').acc.mean())

# # Plot
# gg_ts = (ggplot(dat_long, aes(x='index', y='value', color='vv')) + theme_bw() +
#          geom_line(alpha=0.5) + geom_point(size=1) +
#          geom_vline(xintercept=ktrain) +
#          geom_ribbon(aes(ymin='lb',ymax='ub'),color='black',fill='blue',alpha=0.2))
# gg_ts.save(os.path.join(dir_figures, 'tmp.png'),width=8, height=5)