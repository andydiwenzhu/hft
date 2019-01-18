from dutil import get_tickab
import pandas as pd
import numpy as np
import os
import sys
from sklearn.svm import SVC
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV

import logging
logger = logging.getLogger(__name__)

def _spread(df, n):
    dic = {}
    for ab in ['ask', 'bid']:
        dic['spread_0_%s_price' % ab] = df['n%sPrice_1' % ab.capitalize()] - df['nPrice']
        dic['spread_0_%s_volume' % ab] = df['n%sVolume_1' % ab.capitalize()]
    for i, ab, pv in product(range(n), ['ask', 'bid'], ['price', 'volume']):
        dic['spread_%d_%s_%s'%(i+1, ab, pv)] = (
            df['n%s%s_%d'%(ab.capitalize(), pv.capitalize(), i+1)] -
            df['n%s%s_%d'%(ab.capitalize(), pv.capitalize(), i+2)]
        )
    return pd.concat(dic, axis=1)


def _diff(df, n):
    dic = {}
    for i, ab, pv in product(range(n), ['ask', 'bid'], ['price', 'volume']):
        dic['delta_%d_%s_%s'%(i+1, ab, pv)] = df['n%s%s_%d'%(ab.capitalize(), pv.capitalize(), i+1)].diff()
    return pd.concat(dic, axis=1)


def _label(df, L=5):
    dic = {}
    dic['label_high'] = df['nPrice'].rolling(L).max().shift(-L)-df['nPrice']
    dic['label_low'] = df['nPrice']-df['nPrice'].rolling(L).min().shift(-L)
    dic['label_ha'] = df['nAskPrice_1'].rolling(L).min().shift(-L)-df['nAskPrice_1']
    dic['label_lb'] = df['nBidPrice_1']-df['nBidPrice_1'].rolling(L).max().shift(-L)
    return pd.concat(dic, axis=1)


def get_features(ticks, L, include_label=True):
    columns = [name+str(num) for name in ['nAskPrice_','nBidPrice_','nAskVolume_','nBidVolume_'] for num in range(1,11)]
    columns += ['nTime','nPrice']
    df = ticks[columns].copy()
    
    spread = _spread(df, 1)
    diff = _diff(df, 2)
    
    features = [spread, diff]

    if include_label:
        label = _label(df, L)
        features.append(label)
    
    return pd.concat(features, axis=1).dropna()


def get_filenames(path, dates, stock):
    return ['{path}/{date}/tickab_{stock}.{date}'.format(path=path, date=date, stock=stock) for date in dates]


def tick_filter(f):
    ticks = get_tickab(f)
    return ticks.ix[ticks.nTime>=93000000]

def get_data(path, L, dates, stock='000001'):
    dfs = []
    for fn in get_filenames(path, dates, stock):
        with open(fn, 'rb') as f:
            ticks = tick_filter(f)
        df = get_features(ticks, L)
        dfs.append(df.reset_index(drop=True))

    x = pd.concat(dfs)
    return x


def data_filter(x, f=10):
    return x.iloc[::f]


def top_bottom(x, k, threshold=None, balance=False):
    if not threshold:
	if k in ['label_high', 'label_low']:
            threshold = x[k].quantile(0.5)-0.00001
            logger.debug('quantile %s', x[k].quantile([0.5,0.7,0.9,0.95,0.99]))
	elif k in ['label_ha', 'label_lb']:
	    threshold = 0

    x['label'] = (x[k] >= threshold).astype(int)
    if balance:
	g = x.groupby('label')
	x = g.apply(lambda s: s.sample(g.size().min(), random_state=7)).reset_index(drop=True)

    label = x['label']
    features = [c for c in x.columns if not c.startswith('label')]
    data = x[features]
    return data, label, threshold


def data_ss(data, keys):
    '''
    x is standard scaled seperatedly for price and volume
    '''
    mlist = []
    ss = {}
    for key in keys:
        s = StandardScaler()
        cols = [c for c in data.columns if c.endswith(key)]
        m = data[cols].values        
        s.fit(m)
        ss[key] = s
        mlist.append(s.transform(m))
        
    return np.concatenate(mlist, axis=1), ss

def standard(data, keys, ss):
    mlist = []
    for key in keys:
        s = ss[key]
        cols = [c for c in data.columns if c.endswith(key)]
        m = data[cols].values        
        mlist.append(s.transform(m))
        
    return np.concatenate(mlist, axis=1)

def model_gs(train_data, train_label):
    model = SVC(gamma='auto', probability=False)
    param_grid = {
        'kernel': ['rbf', 'sigmoid', 'poly'],
        'C': [10, 100, 1000]
    }
    grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1)
    grid_search.fit(train_data, train_label)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        if para in param_grid:
            logger.debug('%s %s', para, val)
    return best_parameters



class Tick_SVM(object):
    def __init__(self, path, label_key, ss_keys=['price','volume']):
        self.path = path
	self.label_key = 'label_'+label_key

        self.ss_keys = ss_keys
        self.model = None

    def train(self, dates, stock, L):
        self.L = L
        self.stock = stock

        x = get_data(self.path, self.L, dates, self.stock)

        data, label, self.R = top_bottom(x, self.label_key, balance=True)
    
        train_data, self.ss = data_ss(data, self.ss_keys)
        train_label = label.values

        logger.info('1 ratio: %f, shape: %s', sum(train_label)*1.0/len(train_label), train_data.shape)

        self.model = SVC(C=100, gamma='auto', probability=True)
        self.model.fit(train_data, train_label)

    def test(self, dates):
        x = get_data(self.path, self.L, dates, self.stock)
        data, label, _ = top_bottom(x, self.label_key, self.R)
        test_data = standard(data, self.ss_keys, self.ss)
        test_label = label.values

        y = test_label
        y_ = self.model.predict(test_data)
#	y_ = [1 if r[1] > 0.6 else 0 for r in self.model.predict_proba(test_data)]

	
	logger.info('1 ratio: %f', sum(y_)*1.0/len(y_))
        logger.info('test accuracy: %f', accuracy_score(y, y_))
	logger.info('test recall: %f', recall_score(y, y_))

       
    def predict(self, df, prob=False):
        x = get_features(df, self.L, include_label=False)
        if len(x):
            x = standard(x, self.ss_keys, self.ss)
	    if prob:
	        y_ = self.model.predict_proba(x)
	    else:
		y_ = self.model.predict(x)
            return y_
        return None



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    h = Tick_SVM('/home/pub/tick/tony/etf/', 'ha')
    h.train(['20181105', '20181106', '20181107', '20181108'], '510500', 100)
    h.test(['20181109'])
