import numpy as np
from xgboost import XGBRegressor as GBR
from math import modf

# di_model={'core_buffer':'2', 'eta':'0.1', 'depth':'3'}
# self=model(encoder=enc_yX, lead=lead, lag=lag, di_model=di_model)
class model():
    def __init__(self, encoder, lead=24, lag=24, di_model=None):
        self.enc = encoder
        self.lead = np.arange(1,lead+1)
        self.k = lead
        self.lag = lag
        self.isfit = False
        self.rdrop = min(self.lead)
        # Set up with the defaults
        self.di_model = {'n_trees':100, 'depth':3, 'n_jobs':1, 'eta':0.3}
        if di_model is not None:
            for k in di_model.keys():
                if k not in self.di_model:
                    print(k) 
                assert k in self.di_model
                val_k = float(di_model[k])
                frac, _ = modf(val_k)
                if frac == 0:
                    val_k = int(val_k)
                self.di_model[k] = val_k

    # self = regressor; X=Xtrain.copy(); Y=Ytrain.copy()
    def fit(self, X, Y):
        assert len(X) == len(Y)
        self.k = range(Y.shape[1])
        Xtil = self.enc.transform_X(X, rdrop=self.rdrop)
        # Will lose the first self.lag rows and the last self.rdrop rows
        assert X.shape[0] - Xtil.shape[0] == self.rdrop + self.lag
        Ytil = [self.enc.transform_y(Y[:,k], rdrop=self.rdrop) for k in self.k]
        Ytil = dict(zip(self.k,Ytil))
        assert all([len(Xtil) == len(y) for y in Ytil.values()])
        
        # Fit model for each column and each lead
        self.di_mdl = dict.fromkeys(self.k)
        self.di_mdl = {k: {} for k,v in self.di_mdl.items()}
        for k in self.k:
            for l in self.lead:
                print('Column %i, lead %i' % (k+1, l))
                ytil = Ytil[k][:,l-1]
                idx_l = ~np.isnan(ytil)
                self.di_mdl[k][l] = GBR(random_state=l, 
                    n_estimators=self.di_model['n_trees'], 
                    max_depth=self.di_model['depth'],
                    n_jobs=self.di_model['n_jobs'],
                    learning_rate=self.di_model['eta'])
                self.di_mdl[k][l].fit(Xtil[idx_l],ytil[idx_l])
        
    # X = Xtrain[-(lag+2):].copy()
    def predict(self, X):
        Xtil = self.enc.transform_X(X,rdrop=0)
        holder_k = []
        for k in self.k:
            holder_l = []
            for l in self.lead:
                holder_l.append(self.di_mdl[k][l].predict(Xtil))
            pred_l = np.vstack(holder_l).T
            holder_k.append(pred_l)
        pred = dict(zip(self.k, holder_k))
        if hasattr(self.enc,'enc_Y'):
            pred = {k:self.enc.inverse_transform_y(v) for k,v in pred.items()}
        return pred

    def update_Xy(self, Xnew, Ynew):
        None

    def pickle_me(self):
        None
