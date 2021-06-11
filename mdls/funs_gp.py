# Load standard libaries
import torch
import numpy as np
import pandas as pd
import random
from funs_support import t2n
import copy
from sklearn import metrics

# Load gpytorch libaries
from gpytorch.settings import max_cg_iterations, fast_pred_var
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood, LikelihoodList, likelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel


##########################################
# ----- (1) BASELINE MODEL CLASSES ----- #

# Standard RBF kernel
class gp_real(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(gp_real, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
	
# Multitask RBF kernel
class mgp_real(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(mgp_real, self).__init__(train_x, train_y, likelihood)
        assert train_x.shape == train_y.shape
        self.k = train_y.shape[1]
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.k)
        self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=self.k, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

# Block-diagonal multitask GP
class mgp_batch(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.k = train_y.shape[1]
        self.mean_module = ConstantMean(batch_shape=torch.Size([self.k]))
        # ScaleKernel(,batch_shape=torch.Size([self.k]))
        self.covar_module = RBFKernel(batch_shape=torch.Size([self.k]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(
            MultivariateNormal(mean_x, covar_x))

##########################################
# ----- (2) OPTIMIZATION FUNCTIONS ----- #

# wrapper = self.gp.copy(); x_val=Eta_val.copy(); y_val=Ytil_val.copy() 
# max_iter=250; max_cg=10000; lr=0.01
def tune_gp(wrapper, x_val=None, y_val=None, 
            max_iter=250, max_cg=10000, lr=0.1):
    assert (x_val is None and y_val is None) or (x_val is not None and y_val is not None)
    # Set up optimizers
    wrapper.model.train(); wrapper.likelihood.train()
    optimizer = torch.optim.Adam([{'params': wrapper.model.parameters()}], lr=lr)
    mll = wrapper.ll(wrapper.likelihood, wrapper.model)
    # begin loop
    r2_prev, i, pct_change, lprev, check = 0, 0, 1, 0, True
    loss_seq, r2_seq = np.zeros(max_iter), np.NaN*np.zeros(max_iter)
    while (i < max_iter) & check:  # & (pct_change > tol)
        i += 1
        optimizer.zero_grad()  # Zero gradients from previous iteration
        with max_cg_iterations(max_cg):
            # output = model(model.train_inputs[0])
            output = wrapper.model(*wrapper.model.train_inputs)
        loss = -mll(output, wrapper.model.train_targets)
        loss.backward()
        ll = loss.item()
        ldiff = lprev - ll
        lprev = ll
        optimizer.step()
        torch.cuda.empty_cache()
        loss_seq[i-1] = loss.item()
        if (i % 5) == 0:
            pct_change = ldiff / ll
            print('Iter %d/%d - Loss: %.4f, pct: %.4f' %  (i, max_iter, ll, pct_change))
            # Check for convergence
            dseq = -np.diff(loss_seq[:i])
            # break
            if not np.all( dseq > 0 ):
                print('Change in nll is not positive')
                check = False
            if x_val is not None:
                mu_val, se_val = wrapper.predict(x_val)
                r2 = metrics.r2_score(y_val, mu_val)
                r2_seq[i-1] = r2
                print('val R2: %0.1f%%' % (r2*100))
                if r2 < r2_prev:
                    print('Validation R2 is not increasing')
                    check = False
                r2_prev = r2
    # Save the loss data
    df_loss = pd.DataFrame({'iter':np.arange(max_iter)+1,'loss':loss_seq})
    df_loss = df_loss.query('loss != 0')
    return df_loss

###############################
# ----- (3) GP WRAPPERS ----- #

# Wrapper for the MGP classes
class gp_wrapper():  # Uses multitask for likelihood
    def __init__(self, gp_class, train_x, train_y, tt='multi'):
        self.tt = tt
        assert tt in ['multi','list','univariate']
        if len(train_y.shape) == 2:
            self.k = train_y.shape[1]
        # Set likelihood
        if tt == 'multi':
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.k)
        elif tt == 'list':
            self.likelihood = LikelihoodList(*[GaussianLikelihood() for z in range(self.k)])
        else:
            self.likelihood = GaussianLikelihood()
        # Set marginal likelihood
        if self.tt == 'list':
            self.ll = SumMarginalLogLikelihood
        else:
            self.ll = ExactMarginalLogLikelihood
        # Set the device
        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        print('Using device: %s' % self.device)
        # Convert inputs to device
        train_x, train_y = self.to_tensor(train_x), self.to_tensor(train_y)
        # Initialize models and likelihoods
        self.call_seed()
        if tt == 'list':
            self.model = IndependentModelList(*[gp_class(train_x, train_y[:,jj], ll) for jj, ll in enumerate(self.likelihood.likelihoods)])
        else:
            self.model = gp_class(train_x, train_y, self.likelihood)
        self.model.to(self.device)
        # self.model.float()  # Set if errors show up

    def fit(self, x_val=None, y_val=None, max_iter=250, max_cg=10000, lr=0.01):
        self.loss = tune_gp(self, x_val=x_val, y_val=y_val, max_iter=max_iter, max_cg=max_cg, lr=lr)

    def forward(self, x):
        return self.mdl.forward(x)

    def to_tensor(self, xx):
        tens = torch.tensor(xx, dtype=torch.float32).to(self.device)
        if not tens.is_contiguous():
            tens = tens.contiguous()
        return tens

    def call_seed(self, seed=1234):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

    # X = Yhat.copy()
    def predict(self, X):
        self.likelihood.eval(); self.model.eval()
        x_pred = self.to_tensor(X)
        with torch.no_grad(), fast_pred_var():
            if self.tt == 'list':
                preds = self.likelihood(*self.model(*[x_pred for z in range(self.k)]))
                mean = np.vstack([t2n(pred.mean) for pred in preds]).T
                se = np.vstack([t2n(pred.stddev) for pred in preds]).T
            else:
                pred = self.likelihood(self.model(x_pred))
                mean = t2n(pred.mean)
                se = t2n(pred.stddev)
        self.likelihood.train(); self.model.train()
        return mean, se

    def copy(self):
        return copy.deepcopy(self)
