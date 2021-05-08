# SCRIPT TO TEST HOW WELL SIMPLE GP DOES WITH HOUR AND TREND ON ONE-DAY-AHEAD FORECASTS

import torch, gpytorch, random
from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import max_cg_iterations, fast_pred_var

import numpy as np
import pandas as pd
from funs_support import t2n

use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

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
        out = MultivariateNormal(mean_x, covar)
        return out


# self = mdl(encoder=enc_yX)
class mdl():
    def __init__(self, encoder, dtype=torch.float32, leads=24, max_cg=10000):
        self.enc = encoder
        self.dtype = torch.float32
        self.leads = np.arange(1,leads+1)
        self.max_cg = max_cg
        self.isfit, self.istrained = False, False

    # self, X, y = gp_model, Xtrain.copy(), ytrain.copy()
    # n_steps=250; lr=0.01; tol=0.0001
    def fit(self, X, y, n_steps=250, lr=0.05, tol=0.01):
        assert len(X) == len(y)
        # --- (i) Data --- #
        Ytil = torch.tensor(self.enc.transform_y(y,rdrop=1),dtype=self.dtype)
        assert Ytil.shape[1] == self.leads.max()
        Xtil = torch.tensor(self.enc.transform_X(X,rdrop=1).toarray(),dtype=self.dtype)
        assert len(Xtil) == len(Ytil)
        Ytil = torch.cat([Ytil[:,k-1] for k in self.leads])
        Itil = torch.cat([torch.zeros(Xtil.shape,dtype=torch.long)+k-1 for k in self.leads])
        Xtil = torch.cat([Xtil for k in self.leads])
        idx_full = list(np.where(~torch.isnan(Ytil))[0])
        Xtil, Itil, Ytil = Xtil[idx_full], Itil[idx_full], Ytil[idx_full]
        assert Xtil.shape == Itil.shape
        
        # --- (ii) Model --- #
        torch.manual_seed(self.leads.max())
        random.seed(self.leads.max())
        self.likelihood = GaussianLikelihood()
        self.model = MultitaskGPModel((Xtil, Itil), Ytil, self.likelihood, ntasks=self.leads.max())
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # --- (iii) Training --- #
        ll = 0
        for i in range(n_steps):
            optimizer.zero_grad()
            output = self.model(Xtil, Itil)
            loss = -mll(output, Ytil)
            loss.backward()
            print('Iter %d/%d - Loss: %.6f' % (i + 1, n_steps, loss.item()))
            optimizer.step()
            if (np.abs(ll - loss.item()) < tol) or (i >= n_steps):
                print('Early stopping')
                break
            ll = loss.item()
        print('Training finished')
        self.isfit = True

    # X = Xtrain.copy()
    def predict(self, X):
        assert self.isfit
        self.likelihood.eval()
        self.model.eval()
        Xtil1 = torch.tensor(self.enc.transform_X(X,rdrop=0).toarray(),dtype=self.dtype)
        Itil1 = torch.zeros(Xtil1.shape,dtype=torch.long)
        Itil = torch.cat([Itil1+k-1 for k in self.leads])
        Xtil = torch.cat([Xtil1 for k in self.leads])
        with torch.no_grad(), fast_pred_var(), max_cg_iterations(self.max_cg):
            pred = self.likelihood(self.model(Xtil, Itil))
        Mu, Sigma, idx = t2n(pred.mean), t2n(pred.stddev), t2n(Itil[:,0])
        # Reverse transform
        Mu = self.enc.inverse_transform_y(Mu)
        Sigma = self.enc.inverse_transform_y(Sigma)
        return Mu, Sigma, idx
        