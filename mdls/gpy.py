"""
SCRIPT TO TEST HOW WELL SIMPLE GP DOES WITH HOUR AND TREND ON ONE-DAY-AHEAD FORECASTS
"""

import torch
import gpytorch
import os
import pickle

import numpy as np
import pandas as pd

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import LinearKernel, RBFKernel, CosineKernel
# ,SpectralMixtureKernel, PeriodicKernel, ProductKernel, AdditiveKernel,
from gpytorch.kernels import ScaleKernel

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from funs_support import t2n, normalizer

from scipy.stats import norm


class gp_real(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(gp_real, self).__init__(train_x, train_y, likelihood)
        self.mean = ConstantMean()
        self.linear1 = LinearKernel()  # Single slope
        self.linear2 = ScaleKernel(LinearKernel())  # Temporal interaction with RBF
        self.rbf1 = RBFKernel()  # Similarity on hour (X)
        self.rbf2 = ScaleKernel(RBFKernel())  # Temporal interaction with time
        self.cosine1 = CosineKernel()
        self.cosine2 = ScaleKernel(CosineKernel())

    def forward(self, x):
        mean_x = self.mean(x)
        # Independent signals: trend, cyclical, RBF-freatures
        covar_x1 = self.linear1(x[:, 0]) + self.cosine1(x[:, 0]) + self.rbf1(x[:, 1:])
        # Interaction of cylical/features with trend
        covar_x2 = self.linear2(x[:, 0])*self.rbf2(x[:, 1:]) + self.linear2(x[:, 0])*self.cosine2(x[:, 0])
        # Sum of independent, time-dependent, and interaction
        covar_x = covar_x1 + covar_x2 + (covar_x1 * covar_x2)
        out = MultivariateNormal(mean_x, covar_x)
        return out


# self = mdl(model=model, lead=lead, date=d_test, cn=cn)
# self.fit(X=Xmat_tval, y=y_tval, ntrain=1080, nval=168)
# print(self.gp.train_inputs[0].var(0))
class mdl():
    def __init__(self, model, lead, cn):  #, date
        self.encX, self.encY = normalizer(), normalizer()
        self.isfit, self.istrained = False, False
        self.model, self.lead, self.cn = model, lead, cn  #self.date,
        # Construct the unique filename
        self.fn = 'mdl_' + self.model + '_lead_' + str(self.lead) + '.pkl'  # + '_' + self.date.strftime('%Y_%m_%d')

    # ntrain, nval, X, y, max_iter = 1080, 168, Xmat_tval.copy(), y_tval.copy(), 1000
    def fit(self, X, y, ntrain=1080, nval=168):
        self.ntrain, self.nval = ntrain, nval
        ntot = ntrain + nval
        Xtil, ytil = X[-ntot:], y[-ntot:]
        self.encX.fit(Xtil)
        self.encY.fit(ytil)  # Learn scaling
        Xtil, ytil = torch.tensor(self.encX.transform(Xtil)), torch.tensor(self.encY.transform(ytil))
        # Train model
        torch.manual_seed(1234)  # Seed because gptorch is non-determinsit in matrix inversion
        self.likelihood = GaussianLikelihood()
        self.gp = gp_real(train_x=Xtil, train_y=ytil, likelihood=self.likelihood)
        self.isfit = True

    def set_Xy(self, X, y):
        Xtil, ytil = torch.tensor(self.encX.transform(X)), torch.tensor(self.encY.transform(y))
        self.gp.set_train_data(inputs=Xtil, targets=ytil, strict=False)

    def tune(self, max_iter=100):
        optimizer = torch.optim.Adam(params=self.gp.parameters(), lr=0.01)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
        self.gp.train()
        self.likelihood.train()  # Set model to training mode
        self.gp.double()
        ldiff, lprev, tol = 1, 100, 1e-3
        i = 0
        while (ldiff > tol) & (i < max_iter):
            i += 1
            optimizer.zero_grad()  # Zero gradients from previous iteration
            output = self.gp(self.gp.train_inputs[0])  # Output from model
            loss = -mll(output, self.gp.train_targets)  # Calc loss and backprop gradients
            loss.backward()
            if (i + 1) % 5 == 0:
                ll = loss.item()
                ldiff = lprev - ll
                lprev = loss.item()
                print('Iter %d/%d - Loss: %.3f, ldiff: %.4f' % (i + 1, max_iter, ll, ldiff))
            optimizer.step()
        print('\n'.join([f"{name}: {param.item():0.3f}" for name, param in self.gp.named_parameters()]))
        # Get internal fit
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad():
            pred = self.likelihood(self.gp(self.gp.train_inputs[0]))
        crit, cn = norm.ppf(0.975), ['mu','y','lb','ub']
        self.res_train = pd.DataFrame({'mu': t2n(pred.mean), 'se': t2n(pred.stddev),'y':t2n(self.gp.train_targets)}).rename_axis(
            'idx').reset_index().assign(tt=lambda x: np.where(x.idx < self.ntrain, 'train', 'valid'))
        self.res_train = self.res_train.assign(lb=lambda x: x.mu - crit * x.se, ub=lambda x: x.mu + crit * x.se)
        self.res_train[cn] = self.encY.inverse_transform(self.res_train[cn])
        print(self.res_train.groupby('tt').apply(lambda x: pd.Series({'r2': r2_score(x.y, x.mu)})))
        self.istrained = True

    # X, y = Xmat_test.copy(), y_test
    def predict(self, X, y=None):
        assert self.isfit & self.istrained
        Xtil = torch.tensor(self.encX.transform(X))
        if y is not None:
            ytil = torch.tensor(self.encY.transform(y))
        ntest = Xtil.shape[0]
        crit, cn = norm.ppf(0.975), ['mu','lb','ub']
        self.gp.eval()
        self.likelihood.eval()
        res = np.zeros([ntest, 2])
        for i in range(ntest):
            xslice = Xtil[[i]]
            with torch.no_grad():
                pred = self.likelihood(self.gp(xslice))
            res[i] = [pred.mean.item(), pred.stddev.item()]
            # Append on test set
            if y is not None:
                self.gp.set_train_data(inputs=torch.cat([self.gp.train_inputs[0],xslice]),
                                       targets=torch.cat([self.gp.train_targets, ytil[[i]]]),strict=False)
        res = pd.DataFrame(res, columns=['mu', 'se']).assign(lb=lambda x: x.mu - crit * x.se, ub=lambda x: x.mu + crit * x.se)
        res = pd.DataFrame(self.encY.inverse_transform(res[cn]),columns=cn)
        if y is not None:
            res.insert(0,'y',y)
            print(r2_score(res.y, res.mu))
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
