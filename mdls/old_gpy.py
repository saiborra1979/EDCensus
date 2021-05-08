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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import LinearKernel, RBFKernel, CosineKernel
# ,SpectralMixtureKernel, PeriodicKernel, ProductKernel, AdditiveKernel,
from gpytorch.kernels import ScaleKernel


# Single slope: linear0
# Temporal interaction with RBF: linear1
# Similarity on hour: rfb0
# Temporal interaction with time: rbf1
class gp_real(ExactGP):
    def __init__(self, train_x, train_y, likelihood, cidx):
        super(gp_real, self).__init__(train_x, train_y, likelihood)
        self.cidx = cidx
        self.mean = ConstantMean()
        # trend have a linear+cosine, everything else gets RBF
        self.tt = pd.Series(np.setdiff1d(self.cidx.tt.unique(),['trend']))
        self.ngroups = len(self.tt)
        self.pidx_trend = int(self.cidx[self.cidx.tt=='trend'].pidx.values[0])
        # Number of linear kernels: 2 + ngroups
        # Number of cosine kernels: 2
        # Number of RBF kernels: 2*ngroups
        self.kern1_linear_trend = LinearKernel()
        self.kern1_cosine_trend = CosineKernel()
        self.kern2_linear_trend = ScaleKernel(LinearKernel())
        self.kern2_cosine_trend = ScaleKernel(CosineKernel())
        for col in self.tt:
            setattr(self, 'kern1_rbf_' + col, RBFKernel())
            setattr(self, 'kern2_rbf_' + col, ScaleKernel(RBFKernel()))
            setattr(self, 'kern2_linear_' + col, ScaleKernel(LinearKernel()))


    def forward(self, x):
        mean_x = self.mean(x)
        # Independent signals: trend, cyclical, RBF-freatures
        covar_x1 = self.kern1_linear_trend(x[:, self.pidx_trend]) + \
                   self.kern1_cosine_trend(x[:, self.pidx_trend])
        # Interaction of cyclical/features with trend
        covar_x2 = self.kern2_linear_trend(x[:, self.pidx_trend])*self.kern2_cosine_trend(x[:, self.pidx_trend])
        for col in self.tt:
            idx = self.cidx[self.cidx.tt==col].pidx.values
            covar_x1 += getattr(self, 'kern1_rbf_' + col)(x[:, idx])
            covar_x2 += getattr(self, 'kern2_rbf_' + col)(x[:, idx]) * getattr(self, 'kern2_linear_' + col)(x[:, self.pidx_trend])
        # Sum of independent, time-dependent, and interaction
        covar_x = covar_x1 + covar_x2 + (covar_x1 * covar_x2)
        out = MultivariateNormal(mean_x, covar_x)
        return out


# self = mdl(model=model, lead=lead, cn=cn, device=device, groups = groups)
# self.fit(X=Xmat_tval, y=y_tval, ntrain=ntrain, nval=nval)
# self.tune(max_iter=250, lr=0.01)
# print(pd.Series([name for name, val in self.gp.named_parameters()]))
# qq=self.gp.forward(self.gp.train_inputs[0])
class mdl():
    def __init__(self, model, lead, cn, device, groups=None):  #, date
        self.encX, self.encY = normalizer(), normalizer()
        self.isfit, self.istrained = False, False
        self.model, self.lead, self.cn, self.device = model, lead, pd.Series(cn), device
        # Construct the unique filename
        self.fn = 'mdl_' + self.model + '_lead_' + str(self.lead) + '.pkl'  # + '_' + self.date.strftime('%Y_%m_%d')
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

    # X, y = Xmat_tval.copy(), y_tval.copy()
    def fit(self, X, y):
        ntot = len(X)
        assert ntot == len(y)
        Xtil, ytil = X[:,self.cidx.idx].copy(), y.copy()
        self.encX.fit(Xtil)
        self.encY.fit(ytil)  # Learn scaling
        Xtil = torch.tensor(self.encX.transform(Xtil)).to(self.device)
        ytil = torch.tensor(self.encY.transform(ytil)).to(self.device)
        # Train model
        torch.manual_seed(1234)  # Seed because gpytorch is non-determinsit in matrix inversion
        self.likelihood = GaussianLikelihood()
        self.gp = gp_real(train_x=Xtil, train_y=ytil, likelihood=self.likelihood, cidx=self.cidx)
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
        ldiff, lprev, tol = 1, 100, 1e-3
        i = 0
        while (ldiff > tol) & (i < max_iter):
            i += 1
            optimizer.zero_grad()  # Zero gradients from previous iteration
            with gpytorch.settings.max_cg_iterations(10000):
                output = self.gp(self.gp.train_inputs[0])  # Output from model
            loss = -mll(output, self.gp.train_targets)  # Calc loss and backprop gradients
            loss.backward()
            if (i + 1) % 5 == 0:
                ll = loss.item()
                ldiff = lprev - ll
                lprev = loss.item()
                print('Iter %d/%d - Loss: %.3f, ldiff: %.4f' % (i + 1, max_iter, ll, ldiff))
            optimizer.step()
            torch.cuda.empty_cache()
        #print('\n'.join([f"{name}: {param.item():0.3f}" for name, param in self.gp.named_parameters()]))
        # Get internal fit
        self.gp.eval()
        self.likelihood.eval()
        self.istrained = True

        if get_train:
            with torch.no_grad():
                pred = self.gp(self.gp.train_inputs[0])
            crit, cn = norm.ppf(0.975), ['mu','y','lb','ub']
            self.res_train = pd.DataFrame({'mu': t2n(pred.mean), 'se': t2n(pred.stddev),'y':t2n(self.gp.train_targets)}).rename_axis(
                'idx').reset_index().assign(tt=lambda x: np.where(x.idx < self.ntrain, 'train', 'valid'))
            self.res_train = self.res_train.assign(lb=lambda x: x.mu - crit * x.se, ub=lambda x: x.mu + crit * x.se)
            self.res_train[cn] = self.encY.inverse_transform(self.res_train[cn])
            print(self.res_train.groupby('tt').apply(lambda x: pd.Series({'r2': r2_score(x.y, x.mu)})))


    # X, y = Xmat_test.copy(), y_test.copy()
    def predict(self, X, y=None):
        assert self.isfit & self.istrained
        Xtil = torch.tensor(self.encX.transform(X[:,self.cidx.idx.values]),dtype=torch.float32).to(self.device)
        if y is not None:
            ytil = torch.tensor(self.encY.transform(y),dtype=torch.float32).to(self.device)
        ntest = Xtil.shape[0]
        cn = ['mu', 'se']
        self.gp.eval()
        self.likelihood.eval()
        res = np.zeros([ntest, 2])
        for i in range(ntest):
            xslice = Xtil[[i]]
            with torch.no_grad(), gpytorch.settings.max_cg_iterations(10000):
                pred = self.gp(xslice)
            res[i] = [pred.mean.item(), pred.stddev.item()]
            # Append on test set
            if y is not None:
                self.gp.set_train_data(inputs=torch.cat([self.gp.train_inputs[0],xslice]),
                                       targets=torch.cat([self.gp.train_targets, ytil[[i]]]),strict=False)
        res = pd.DataFrame(res, columns=cn)
        # Transform back to original scale (not we only need se to be multipled by sig from (X-mu)/sig
        res['mu'] = self.encY.inverse_transform(res.mu)
        res['se'] = res.se * self.encY.enc.scale_
        if y is not None:
            res.insert(0,'y',y)
            print('Test set R2: %0.3f' % r2_score(res.y, res.mu))
        return res

        # tmp = pd.concat([self.res_train.drop(columns=['se','idx']), res.assign(tt='test')])
        # tmp = tmp.reset_index(None, True).rename_axis('idx').reset_index()
        # from plotnine import *
        # gg_torch = (ggplot(tmp, aes(x='idx', y='mu', color='tt')) + theme_bw() + geom_line() +
        #             geom_vline(xintercept=ntrain) + geom_ribbon(aes(ymin='lb', ymax='ub'), alpha=0.5) +
        #             geom_point(aes(x='idx', y='y'), color='black', size=0.5, alpha=0.5))
        # gg_torch.save(os.path.join(dir_figures, 'test.png'),width=12,height=7)


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
