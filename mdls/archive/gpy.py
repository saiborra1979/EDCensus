# SCRIPT TO TEST HOW WELL SIMPLE GP DOES WITH HOUR AND TREND ON ONE-DAY-AHEAD FORECASTS

import numpy as np
import pandas as pd
import random
import torch
import gpytorch
from funs_support import t2n

from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import LinearKernel, RBFKernel, CosineKernel, ScaleKernel
from gpytorch.settings import max_cg_iterations, fast_pred_var

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, cn_X):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Determine column indexes
        self.cn_X = pd.Series(cn_X)
        self.cn_X = self.cn_X.rename_axis('cidx').reset_index().rename(columns={0:'cn'})
        self.cn_X = self.cn_X.assign(gg=lambda x: x.cn.str.split('\\_',2,True).iloc[:,0:2].apply(lambda x: x.str.cat(sep='_'),1))
        self.di_gg = self.cn_X.groupby('gg').apply(lambda x: x.cidx.to_list()).to_dict()
        # Set up kernels
        self.covar_modules = {}
        for gg in self.di_gg:
            self.covar_modules['RBF_'+gg] = ScaleKernel(RBFKernel())
        self.mean = ConstantMean()

    def forward(self, x):
        mean_x = self.mean(x)
        for ii, gg in enumerate(self.di_gg):
            if ii == 0:
                covar_RBF = self.covar_modules['RBF_'+gg](x[:,self.di_gg[gg]])
            else:
                covar_RBF += self.covar_modules['RBF_'+gg](x[:,self.di_gg[gg]])
        out = MultivariateNormal(mean_x, covar_RBF)
        return out

# self = mdl(encoder=enc_yX,device=device,leads=24)
class mdl():
    def __init__(self, encoder, device, dtype=torch.float32, leads=24, max_cg=10000):
        self.enc = encoder
        self.device = device
        self.dtype = dtype
        self.leads = np.arange(1,leads+1)
        self.k = leads
        self.max_cg = max_cg
        self.isfit = False

    # X, y = Xtrain.copy(), ytrain.copy()
    # n_steps=250; lr=0.01; tol=1e-4
    def fit(self, X, y, n_steps=250, lr=0.05, tol=1e-4):
        assert len(X) == len(y)
        # --- (i) Data --- #
        Ytil = torch.tensor(self.enc.transform_y(y=y,leads=self.k,rdrop=1),
                            dtype=self.dtype).to(self.device)
        assert Ytil.shape[1] == self.leads.max()
        Xtil = torch.tensor(self.enc.transform_X(X,rdrop=1),
                    dtype=self.dtype).to(self.device)
        assert len(Xtil) == len(Ytil)

        # --- (ii) Model --- #
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        self.di_ll = [GaussianLikelihood() for k in self.leads]
        self.di_mdl = [[] for k in self.leads]
        for k in range(self.k):
            yk = Ytil[:,k]
            idx_full = torch.where(~torch.isnan(yk))[0]
            self.di_mdl[k] = ExactGPModel(Xtil[idx_full], yk[idx_full], 
                                self.di_ll[k], self.enc.cn_X).to(self.device)
        self.di_mdl = IndependentModelList(*self.di_mdl)
        self.di_ll = LikelihoodList(*self.di_ll)
        self.di_mdl.training, self.di_ll.training = True, True
        mll = SumMarginalLogLikelihood(self.di_ll, self.di_mdl)
        optimizer = torch.optim.Adam([{'params': self.di_mdl.parameters()}], lr=lr)


        # --- (iii) Training --- #
        ll = 0
        for i in range(n_steps):
            optimizer.zero_grad()
            output = self.di_mdl(*self.di_mdl.train_inputs)
            loss = -mll(output, self.di_mdl.train_targets)
            loss.backward()
            print('Iter %d/%d - Loss: %.7f' % (i + 1, n_steps, loss.item()))
            optimizer.step()
            if (np.abs(ll - loss.item()) < tol) or (i >= n_steps):
                print('Early stopping')
                break
            ll = loss.item()
        print('Training finished')
        self.isfit = True

    # X = Xtrain[0:24].copy()
    def predict(self, X):
        assert self.isfit
        self.di_ll.eval()
        self.di_mdl.eval()
        Xtil = torch.tensor(self.enc.transform_X(X,rdrop=0),dtype=self.dtype).to(self.device)
        
        with torch.no_grad(), fast_pred_var(), max_cg_iterations(self.max_cg):
            pred = self.di_ll(*self.di_mdl(*[Xtil for k in self.leads]))
        [t2n(pred[k-1].stddev) for k in self.leads]
        out = pd.concat([pd.DataFrame({'lead':k,'pred':t2n(pred[k-1].mean),
            'se':t2n(pred[k-1].stddev)},index=range(len(pred[k-1].mean))) for k in self.leads])
        out[['pred','se']] = out[['pred','se']].apply(lambda x: self.enc.inverse_transform_y(x.values).flatten(), 0)
        out.reset_index(None,True,True)
        return out
        

