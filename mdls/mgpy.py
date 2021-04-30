# SCRIPT TO TEST HOW WELL SIMPLE GP DOES WITH HOUR AND TREND ON ONE-DAY-AHEAD FORECASTS

import torch
import gpytorch
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from funs_support import t2n, normalizer
from scipy.stats import norm

from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, LinearKernel, RBFKernel, CosineKernel
from gpytorch.kernels import AdditiveKernel, ProductKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood



# Single slope: linear0
# Temporal interaction with RBF: linear1
# Similarity on hour: rfb0
# Temporal interaction with time: rbf1
class mgp_real(ExactGP):
    # train_x=Xtil; train_y=Ytil; likelihood=self.likelihood; cidx=self.cidx.copy()
    def __init__(self, train_x, train_y, likelihood, cidx):
        super(mgp_real, self).__init__(train_x, train_y, likelihood)
        self.cidx = cidx
        self.k = train_y.shape[1]
        self.mean = MultitaskMean(ConstantMean(), num_tasks=self.k)
        # trend have a linear+cosine, everything else gets RBF
        self.tt = pd.Series(np.setdiff1d(self.cidx.tt.unique(),['trend']))
        self.ngroups = len(self.tt)
        self.pidx_trend = int(self.cidx[self.cidx.tt=='trend'].pidx.values[0])
        # Number of linear kernels: 2 + ngroups
        # Number of cosine kernels: 2
        # Number of RBF kernels: 2*ngroups
        self.kern_linear_trend = MultitaskKernel(LinearKernel(), num_tasks=self.k, rank=1)
        self.kern_cosine_trend = MultitaskKernel(CosineKernel(), num_tasks=self.k, rank=1)
        
        for col in self.tt:
            setattr(self, 'kern_rbf_' + col, MultitaskKernel(RBFKernel(), num_tasks=self.k, rank=1))
            setattr(self, 'kern_linear_' + col, MultitaskKernel(LinearKernel(), num_tasks=self.k, rank=1))


    def forward(self, x):
        mean_x = self.mean(x)
        # Independent signals: trend, cyclical, RBF-freatures
        covar_x = self.kern_linear_trend(x[:, self.pidx_trend]) + \
                   self.kern_cosine_trend(x[:, self.pidx_trend])
        # Interaction of cyclical/features with trend
        for col in self.tt:
            idx = self.cidx[self.cidx.tt==col].pidx.values
            covar_x += getattr(self, 'kern_rbf_' + col)(x[:, idx])
        # Sum of independent, time-dependent, and interaction
        out = MultitaskMultivariateNormal(mean_x, covar_x)
        return out


class mdl():
    def __init__(self, model, cn, device, groups=None):
        self.encX, self.encY = normalizer(), normalizer()
        self.isfit, self.istrained = False, False
        self.model, self.cn, self.device = model, pd.Series(cn), device
        # Construct the unique filename
        self.fn = 'mdl_' + self.model + '.pkl'
        # Group the different column types:
        idx_trend = np.where(self.cn == 'date_trend')[0]  # (1) Continuous trend
        idx_date = np.setdiff1d(np.where(self.cn.str.contains('date_'))[0],idx_trend)  # (2) Other datetime
        idx_flow = np.where(self.cn.str.contains('census_|tt_'))[0]
        idx_mds = np.where(self.cn.str.contains('avgmd|u_mds'))[0]
        idx_health = np.where(self.cn.str.contains('diastolic_|num_meds_|systolic_|pulse_|resp_|temp_'))[0]
        idx_demo = np.where(self.cn.str.contains('age_|ret72_|weight_|sex_|DistSK_'))[0]
        idx_lang = np.where(self.cn.str.contains('language_'))[0]
        idx_CTAS = np.where(self.cn.str.contains('CTAS_'))[0]
        idx_arr = np.where(self.cn.str.contains('arr_method'))[0]
        idx_labs = np.where(self.cn.str.contains('labs_'))[0]
        idx_DI = np.where(self.cn.str.contains('DI_'))[0]
        self.di_cn = {'trend':idx_trend, 'date':idx_date, 'flow':idx_flow,
                 'mds':idx_mds, 'health':idx_health, 'demo':idx_demo,
                 'language':idx_lang, 'CTAS':idx_CTAS, 'arr':idx_arr,
                 'labs':idx_labs, 'DI':idx_DI}
        assert len(np.setdiff1d(range(len(self.cn)), np.concatenate(list(self.di_cn.values())))) == 0
        bl_groups = ['trend', 'date', 'flow']
        if groups is None:
            self.groups = bl_groups
        else:
            assert isinstance(groups, list)
            assert all([ll in self.di_cn for ll in groups])
            self.groups = groups + bl_groups
        # Subset to valid groups
        self.di_cn = {z: k for z,k in self.di_cn.items() if z in self.groups}
        self.cidx = np.concatenate(list(self.di_cn.values()))
        self.cidx = pd.concat([pd.DataFrame({'tt':k,'cn':self.cn[v],'idx':v}) for k,v in self.di_cn.items()])
        self.cidx = self.cidx.reset_index(None,True).rename_axis('pidx').reset_index()

    # self=mgp; X, Y = Xmat_tval.copy(), Ymat_tval.copy()
    def fit(self, X, Y):
        ntot = len(X)
        assert ntot == len(Y)
        self.k = Y.shape[1]
        Xtil, Ytil = X[:,self.cidx.idx].copy(), Y.copy()
        self.encX.fit(Xtil)
        self.encY.fit(Ytil)  # Learn scaling
        Xtil = torch.tensor(self.encX.transform(Xtil)).to(self.device)
        Ytil = torch.tensor(self.encY.transform(Ytil)).to(self.device)
        # Train model
        torch.manual_seed(1234)  # Seed because gpytorch is non-determinsit in matrix inversion
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.k)
        self.gp = mgp_real(train_x=Xtil, train_y=Ytil, likelihood=self.likelihood, cidx=self.cidx)
        self.gp.to(self.device)
        self.isfit = True

    def set_Xy(self, X, y):
        Xtil = torch.tensor(self.encX.transform(X)).to(self.device)
        ytil = torch.tensor(self.encY.transform(y)).to(self.device)
        self.gp.set_train_data(inputs=Xtil, targets=ytil, strict=False)

    # lr=0.01; max_iter=250
    def tune(self, max_iter=250, lr=0.01, get_train=False):
        optimizer = torch.optim.Adam(params=self.gp.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
        self.gp.train()
        self.likelihood.train()  # Set model to training mode
        self.gp.float()
        ldiff, lprev, tol = 1, 100, 0.05
        i, mi = 0, 100000
        while (i < max_iter):
            i += 1
            optimizer.zero_grad()  # Zero gradients from previous iteration
            with gpytorch.settings.max_cg_iterations(10000):
                output = self.gp(self.gp.train_inputs[0])  # Output from model
            loss = -mll(output, self.gp.train_targets)  # Calc loss and backprop gradients
            loss.backward()
            ll = loss.item()
            mi = min(mi, ll)
            lprev = loss.item()  # Reset
            optimizer.step()
            torch.cuda.empty_cache()
            if i % 5 == 0:
                print('Iter %d/%d - Loss: %.3f, min: %.4f' % (i, max_iter, ll, mi))
            if ll / mi - 1 > tol:
                print('Stopping model, loss is %i%% above min' % (tol*100))
                break
        # Get internal fit
        self.gp.eval()
        self.likelihood.eval()
        self.istrained = True

        if get_train:
            with torch.no_grad():
                pred = self.gp(self.gp.train_inputs[0])
            crit, cn = norm.ppf(0.975), ['mu','y','lb','ub']
            tmp1 = pd.DataFrame(self.encY.inverse_transform(t2n(pred.mean))).rename_axis('idx').reset_index().melt('idx',None,'lead','mu')
            tmp2 = pd.DataFrame(self.encY.inverse_transform(t2n(pred.stddev))).rename_axis('idx').reset_index().melt('idx',None,'lead','se')
            tmp3 = pd.DataFrame(self.encY.inverse_transform(t2n(self.gp.train_targets))).rename_axis('idx').reset_index().melt('idx',None,'lead','y')
            self.res_train = tmp1.merge(tmp2,'left',['idx','lead']).merge(tmp3,'left',['idx','lead'])
            self.res_train = self.res_train.assign(lb=lambda x: x.mu - crit * x.se, ub=lambda x: x.mu + crit * x.se)
            print(self.res_train.groupby('lead').apply(lambda x: pd.Series({'r2': r2_score(x.y, x.mu)})))

    # X, Y = Xmat_test.copy(), Ymat_test.copy()
    def predict(self, X, Y=None):
        assert self.isfit & self.istrained
        Xtil = torch.tensor(self.encX.transform(X[:,self.cidx.idx.values]),dtype=torch.float32).to(self.device)
        if Y is not None:
            Ytil = torch.tensor(self.encY.transform(Y),dtype=torch.float32).to(self.device)
        ntest = Xtil.shape[0]
        cn = ['mu', 'se']
        self.gp.eval()
        self.likelihood.eval()
        Mu, Sigma = np.zeros([ntest, self.k]), np.zeros([ntest, self.k])
        for i in range(ntest):
            xslice = Xtil[[i]]
            with torch.no_grad(), gpytorch.settings.max_cg_iterations(10000):
                pred = self.gp(xslice)
            Mu[i] = t2n(pred.mean).flat
            Sigma[i] = t2n(pred.stddev).flat
            # Append on test set
            if Y is not None:
                self.gp.set_train_data(inputs=torch.cat([self.gp.train_inputs[0],xslice]),
                                       targets=torch.cat([self.gp.train_targets, Ytil[[i]]]),strict=False)
        # Transform back to original scale (not we only need se to be multipled by sig from (X-mu)/sig
        Mu = self.encY.inverse_transform(Mu)
        Sigma = Sigma * self.encY.enc.scale_
        if Y is not None:
            r2_test = pd.DataFrame({'lead':range(self.k),'r2':r2_score(Y, Mu,multioutput='raw_values')})
            print('Test set R2: %s' % r2_test)
        # Tidy up and return
        tmp1 = pd.DataFrame(Mu).rename_axis('idx').reset_index().melt('idx',None,'lead','mu')
        tmp2 = pd.DataFrame(Sigma).rename_axis('idx').reset_index().melt('idx',None,'lead','se')
        tmp3 = pd.DataFrame(Y).rename_axis('idx').reset_index().melt('idx',None,'lead','y')
        df_test = tmp1.merge(tmp2,'left',['idx','lead']).merge(tmp3,'left',['idx','lead'])
        return df_test

    def save(self, folder):
        """
        SAVES THE COEFFICIENTS OF THE MODEL
        """
        assert self.isfit & self.istuned
        print('Pickling!')
        with open(os.path.join(folder, self.fn), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, folder):
        """
        SET EXISTING CLASS ATTRIBUTES TO MATCH
        """
