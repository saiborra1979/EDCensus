# Load standard libaries
import torch
import numpy as np
import pandas as pd
import random
from funs_support import t2n

# Load gpytorch libaries
from gpytorch.settings import max_cg_iterations, fast_pred_var
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood, LikelihoodList
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

# max_iter=250; max_cg=10000; lr=0.01
def tune_gp(model, likelihood, ll, max_iter=250, max_cg=10000, lr=0.1):
    # Set up optimizers
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    mll = ll(likelihood, model)
    # begin loop
    tol, i, pct_change, lprev = 1e-3, 0, 1, 0
    loss_seq = np.zeros(max_iter)
    while (i < max_iter):  # & (pct_change > tol)
        i += 1
        optimizer.zero_grad()  # Zero gradients from previous iteration
        with max_cg_iterations(max_cg):
            # output = model(model.train_inputs[0])
            output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward()
        ll = loss.item()
        ldiff = lprev - ll
        lprev = ll
        if (i + 1) % 5 == 0:
            pct_change = ldiff / ll
            print('Iter %d/%d - Loss: %.1f, ldiff: %.1f, pct: %.4f' % 
                        (i + 1, max_iter, ll, ldiff, pct_change))
        optimizer.step()
        torch.cuda.empty_cache()
        loss_seq[i-1] = loss.item()
    # Set to eval mode
    model.eval()
    likelihood.eval()
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
        if tt == 'multi':
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.k)
        elif tt == 'list':
            self.likelihood = LikelihoodList(*[GaussianLikelihood() for z in range(self.k)])
        else:
            self.likelihood = GaussianLikelihood()
        
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

    def fit(self, max_iter=250, max_cg=10000, lr=0.01):
        if self.tt == 'list':
            ll = SumMarginalLogLikelihood
        else:
            ll = ExactMarginalLogLikelihood
        self.loss = tune_gp(self.model, self.likelihood, ll, max_iter=max_iter, max_cg=max_cg, lr=lr)

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
        return mean, se



