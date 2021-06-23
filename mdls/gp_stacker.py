# Load standard libaries
import copy
import dill
import numpy as np
import pandas as pd
from math import modf
from sklearn.preprocessing import StandardScaler

from mdls.funs_gp import gp_wrapper, gp_real

# di_model={'base':'rxgboost', 'nval':'168',
#           'max_iter':'10', 'lr':'0.01', 'max_cg':'10000',
#           'n_trees':'100', 'depth':'3', 'n_jobs':'7'}
# self = model(encoder=enc_yX,di_model=di_model)
# self.fit(Xtrain, ytrain)
class model():
    def __init__(self, encoder, di_model=None):
        self.encoder = encoder
        self.di_model = {'base':'xgboost', 'nval':168,  # Number of validation points
                        'max_iter':250, 'lr':0.01, 'max_cg':10000}  # defaults
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
        nval = self.di_model['nval']
        lr = self.di_model['lr']
        max_iter = self.di_model['max_iter']
        max_cg = self.di_model['max_cg']
        # --- (i) Fit the baseline model --- #
        print('(i) Fitting baseline model')
        if nval > 0:
            X_train, y_train = X[:-nval], y[:-nval]
            #X_val, y_val = X[-nval:], y[-nval:]
        else:
            X_train, y_train = X, y
            #X_val, y_val = None, None
        self.base.fit(y=y_train, X=X_train)
        Eta, Ytil = self.base.predict(X=X, y=y)
        assert Eta.shape == Ytil.shape
        self.k = Ytil.shape[1]
        # Fill missing values with predicted scores
        Ytil = np.where(np.isnan(Ytil),Eta, Ytil)
        # Normalize the Y's
        self.enc_Y = StandardScaler().fit(Ytil)
        Eta = self.enc_Y.transform(Eta)
        Ytil = self.enc_Y.transform(Ytil)
        # Split the nval
        if nval > 0:
            Eta_val, Ytil_val = Eta[-nval:], Ytil[-nval:]
            Eta_train, Ytil_train = Eta[:-nval], Ytil[:-nval]
        else:
            Eta_val, Ytil_val = None, None

        # --- (ii) GP Stacker --- #
        print('(ii) Fitting GP')
        self.gp = gp_wrapper(gp_class=gp_real, train_x=Eta_train, train_y=Ytil_train, tt='list')
        self.gp.fit(x_val=Eta_val, y_val=Ytil_val, max_iter=max_iter, max_cg=max_cg, lr=lr)
        # Refit baseline model
        print('Refitting baseline')
        self.base.fit(y=y, X=X)
        self.update_Xy(Xnew=X, ynew=y)

    # Xnew=X.copy(); ynew=y.copy()
    def update_Xy(self, Xnew, ynew):
        print('Updating GP X/Y')
        # (i) New pandas df to get yhats from xgboost
        Eta, Ytil = self.base.predict(X=Xnew, y=ynew)
        Ytil = np.where(np.isnan(Ytil), Eta, Ytil)
        self.enc_Y.fit(Ytil)
        Eta = self.enc_Y.transform(Eta)
        Ytil = self.enc_Y.transform(Ytil)
        # (ii) Update X/y within GP
        tmp_di = self.gp.model.state_dict()
        self.gp = gp_wrapper(gp_class=gp_real, train_x=Eta, train_y=Ytil, tt='list')
        self.gp.model.load_state_dict(tmp_di)
        

    # X = X_now.copy(); #Xtrain[-(lag+2):].copy()
    def predict(self, X):
        # Get predictions from baseline model
        Eta = self.base.predict(X=X)
        assert Eta.shape[1] == self.k
        # Normalize and tensorize
        Eta = self.enc_Y.transform(Eta)
        # Get predictions from multitask GP
        mu, se = self.gp.predict(Eta)
        df_mu = pd.DataFrame(self.enc_Y.inverse_transform(mu)).assign(tt='pred')
        df_se = pd.DataFrame(self.enc_Y.scale_ * se).assign(tt='se')
        df = pd.concat([df_mu,df_se],0).rename_axis('idx').reset_index()
        df = df.melt(['idx','tt'],None,'lead').pivot_table('value',['lead','idx'],'tt')
        df = df.reset_index().assign(lead=lambda x: x.lead+1)
        if df.idx.max() == 0:
            df.drop(columns = 'idx', inplace=True)
        return df

    def pickle_me(self, path):
        with open(path, 'wb') as file:
            dill.dump(self, file)

    def copy(self):
        return copy.deepcopy(self)

# with open(path, 'rb') as file:
#     tmp = dill.load(file)
# tmp.predict(X_now.copy())

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

# for name, param in self.gp.model.named_parameters():
#     with torch.no_grad():
#         print('param: %s, value: %s' % (name, param.flatten()))
