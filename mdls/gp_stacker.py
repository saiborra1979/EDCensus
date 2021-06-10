# Load standard libaries
from mdls.funs_gp import mgp_real
from gpytorch.models import exact_gp
from funs_support import t2n
import torch
import numpy as np
import pandas as pd
from math import modf
from sklearn.preprocessing import StandardScaler

from mdls.funs_gp import gp_wrapper, gp_real  # , mgp_batch

# di_model={'base':'rxgboost', 'max_iter':'100', 'lr':'0.01',
#           'n_trees':'100', 'depth':'3', 'n_jobs':'7'}
# self = model(encoder=enc_yX,di_model=di_model)
# self.fit(Xtrain, ytrain)
class model():
    def __init__(self, encoder, di_model=None):
        self.encoder = encoder
        self.di_model = {'base':'xgboost', 'max_iter':250, 'lr':0.01, 'max_cg':10000}  # defaults
        # Model arguments either strings, floats, ints, or not part of default (left as string)
        if di_model is not None:
            for k in di_model.keys():
                if k in self.di_model:
                    try:
                        val_k = float(di_model[k])
                        frac, _ = modf(val_k)
                        if frac == 0:
                            val_k = int(val_k)
                        self.di_model[k] = val_k
                    except:
                        assert isinstance(di_model[k],str)
                        self.di_model[k] = di_model[k]                        
                else:
                    self.di_model[k] = di_model[k]  # Argument for base
        mclass = __import__('mdls.' + self.di_model['base'])
        self.base = getattr(getattr(mclass, self.di_model['base']), 'model')
        self.base = self.base(encoder=self.encoder, di_model=self.di_model)

    # X, y = Xtrain.copy(), ytrain.copy()
    def fit(self, X, y):
        ntot = len(X)
        assert ntot == len(y)

        # --- (i) Fit the baseline model --- #
        print('(i) Fitting baseline model')
        self.base.fit(y=y, X=X)
        Yhat, Ytil = self.base.predict(X=X, y=y)
        assert Yhat.shape == Ytil.shape
        self.k = Ytil.shape[1]
        # Fill missing values with predicted scores
        Ytil = np.where(np.isnan(Ytil),Yhat, Ytil)
        # Normalize the Y's
        self.enc_Y = StandardScaler().fit(Ytil)
        Yhat, Ytil = self.enc_Y.transform(Yhat), self.enc_Y.transform(Ytil)

        # --- (ii) GP Stacker --- #
        print('(ii) Fitting GP')
        print(np.std(Ytil[:,12] - Yhat[:,12]))
        self.gp = gp_wrapper(gp_class=gp_real, train_x=Yhat, train_y=Ytil, tt='list')
        self.gp.fit(lr=0.1,max_iter=25)
        for name, param in self.gp.model.named_parameters():
            with torch.no_grad():
                print('param: %s, value: %s' % (name, param.flatten()))
        self.gp.predict(Yhat)[1].mean()

    # X = Xtrain[-(lag+2):].copy()
    def predict(self, X):
        # Get predictions from baseline model
        Yhat = self.base.predict(X=X)
        assert Yhat.shape[1] == self.k
        # Normalize and tensorize
        Yhat = self.enc_Y.transform(Yhat)
        # Get predictions from multitask GP

# self.gp = gp_wrapper(gp_class=mgp_real, train_x=Yhat, train_y=Ytil, tt='multi')
# self.gp = gp_wrapper(gp_class=mgp_batch, train_x=Yhat, train_y=Ytil, tt='multi')
# self.gp = gp_wrapper(gp_class=gp_real, train_x=Yhat, train_y=Ytil[:,12], tt='univariate')



    #     # tmp = pd.concat([self.res_train.drop(columns=['se','idx']), res.assign(tt='test')])
    #     # tmp = tmp.reset_index(None, True).rename_axis('idx').reset_index()
    #     # from plotnine import *
    #     # gg_torch = (ggplot(tmp, aes(x='idx', y='mu', color='tt')) + theme_bw() + geom_line() +
    #     #             geom_vline(xintercept=ntrain) + geom_ribbon(aes(ymin='lb', ymax='ub'), alpha=0.5) +
    #     #             geom_point(aes(x='idx', y='y'), color='black', size=0.5, alpha=0.5))
    #     # gg_torch.save(os.path.join(dir_figures, 'test.png'),width=12,height=7)
