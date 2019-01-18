import pandas as pd
import numpy as np
from itertools import product
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def mincut(x):
    return min(x,1)

def time_filter(date):
    df = pd.read_csv('huobip/btc.usdt.2018-11-%s' % date)
    df['ts'] = df['timestamp']//1000
    x = df.loc[df.groupby('ts').ts.idxmax()]
    x.drop(['timestamp'], axis=1, inplace=True)
    print len(df), len(x)
    return x

def add_tmp_cols(x, n=10):
    x['_mid'] = (x['ask_0_p'] + x['bid_0_p'])/2
    for i in range(n):
        x['_ask_dist_%d'%i] = (x['ask_%d_p'%i] - x['_mid']) / (x['ask_0_p'] - x['_mid'])
        x['_bid_dist_%d'%i] = (x['_mid'] - x['bid_%d_p'%i]) / (x['_mid'] - x['bid_0_p'])


def power_imbalance(x, n=10, powers=[2]):
    ret = []
    for power in powers:
        key = 'pi_%d' % power
        ret.append(key)
        x[key] = 0
        for i in range(n):
           x[key] += x['bid_%d_v'%i]/x['_bid_dist_%d'%i]**power
           x[key] -= x['ask_%d_v'%i]/x['_ask_dist_%d'%i]**power
    return ret

def power_price(x, n=10, powers=[2]):    
    ret = []
    for power in powers:
        key = 'pp_%d' % power
        ret.append(key)
        x[key] = 0
        x['_dist_w'] = 0
        for i in range(n):
            x['_tmp'] = x['bid_%d_p'%i]*x['_bid_dist_%d'%i]**power
            x[key] += x['_tmp'] 
            x['_tmp'] = x['ask_%d_p'%i]*x['_ask_dist_%d'%i]**power
            x[key] += x['_tmp']
            x['_dist_w'] += x['_bid_dist_%d'%i]**power + x['_ask_dist_%d'%i]**power
        x[key] = np.log(x[key]/x['_dist_w']/x['_mid']) 
    return ret

def spread(x, n=10):
    ret = []
    for ab, pv in product(['ask', 'bid'],['p','v']):
        key = 'spread_0_%s_%s' % (ab, pv)
        ret.append(key)
        x[key] = x['%s_0_%s'%(ab,pv)] - x['last']

    for i, ab, pv in product(range(n-1), ['ask', 'bid'], ['p', 'v']):
        key = 'spread_%d_%s_%s' % (i+1, ab, pv)
        ret.append(key)
        x[key] = x['%s_%d_%s'%(ab, i+1, pv)] - x['%s_%d_%s'%(ab, i, pv)]

    return ret

def diff(x, n=10):
    ret = []
    for i, ab, pv in product(range(n), ['ask', 'bid'], ['p', 'v']):
        key = 'delta_%d_%s_%s'%(i, ab, pv)
        ret.append(key)
        x[key] = x['%s_%d_%s'%(ab, i, pv)].diff()
    return ret

def volume_order_imbalance(x):
    x['va'] = 0
    x['va'] += (x['ask_0_p'].diff() <= 0).astype(int)*x['ask_0_v'].diff()
    x['va'] += (x['ask_0_p'].diff() < 0).astype(int)*x['ask_0_v'].shift()       
    x['vb'] = 0
    x['vb'] += (x['bid_0_p'].diff() >= 0).astype(int)*x['bid_0_v'].diff()
    x['vb'] += (x['bid_0_p'].diff() > 0).astype(int)*x['bid_0_v'].shift()
    return ['va','vb']   

def ma_y(x, forwards=[5]):
    ret = []
    for f in forwards:
        key = 'y_%d'%f
        ret.append(key)
        x[key] = np.log(x['_mid'].rolling(f).mean().shift(-f) / x['_mid'])
    return ret

def run(date, forward, back, vol_limit, unify):
    x = time_filter(date)
    add_tmp_cols(x)    
    cols = []
    cols += power_imbalance(x, powers=[0,2,4,8])
    cols += power_price(x, powers=[0,2,4,8])
    cols += spread(x)
    cols += diff(x) 
    cols += volume_order_imbalance(x)
 
    labels = ma_y(x, forwards=[5]) 
    d = x[cols+labels].dropna()
#    print d.corrwith(d[labels[0]])

    X = d[cols]
    y = d['y_5']
    cut = len(y)/3*2
    train_data = X[:cut]
    test_data = X[cut:]
    train_label = y[:cut]
    test_label = y[cut:]
    
#    model = LinearRegression()
    model = XGBRegressor()
    model.fit(train_data, train_label)
    p = model.predict(train_data)
    print len(train_label), r2_score(train_label, p)
    p = model.predict(test_data)
    return len(test_label), r2_score(test_label, p)


def get_features(x):
    add_tmp_cols(x)    
    cols = []
    cols += power_imbalance(x, powers=[0,2,4,8])
    cols += power_price(x, powers=[0,2,4,8])
    cols += spread(x)
    cols += diff(x) 
    cols += volume_order_imbalance(x)
    return cols 

def get_data(date):
    x = time_filter(date)
    cols = get_features(x)
    labels = ma_y(x, forwards=[5])
    d = x[cols+labels].dropna()
    return d[cols], d['y_5']

import time

class Tick_X(object):
    def __init__(self):
        self.model = None

    def train(self, date):
        X, y = get_data(date)
        self.cols = X.columns
        self.model = XGBRegressor()
        self.model.fit(X, y)
        p = self.model.predict(X)
        print 'train size: %s, r2: %s' % (len(y), r2_score(y, p)) 

    def test(self, date):
	X, y = get_data(date)
        p = self.model.predict(X)
        print 'test size: %s, r2: %s' % (len(y), r2_score(y, p))
       
    def predict(self, x):
        cols = get_features(x)
        p = self.model.predict(x[cols])
        return p


if __name__ == '__main__':
    m = Tick_X()
    m.train(24)
    x = time_filter(25)
    res = []
    for i in range(500):
        if i%100==0:
            print i
        res.append(m.predict(x.iloc[i:i+2].copy()))    
    print pd.Series(res).describe()


def grid_search(path):
    res = []
    for date, forward, back, vol, unify in product([24, 25, 26],[5],[0],[0],['None']):
        l, r = run(date, forward, back, vol, unify)
        print date, forward, back, vol, unify, l, r
        res.append({
		    'date': date,
		    'forward': forward,
		    'back': back,
		    'vol': vol,
		    'unify': unify,
                    'l': l,
                    'r': r,		
		   })
    pd.DataFrame(res).to_pickle(path)
