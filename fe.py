import os
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import linregress
import datetime
from dtypes import *


class Reader(object):
    def __init__(self, path, region, market, combine_refresh, feature_refresh):
	self.path = path
	self.region = region
	self.market = market
	self.combine_refresh = combine_refresh
	self.feature_refresh = feature_refresh

    def get_data(self, instrument, date):
        fpath = '%s/%s/%s/feature/%s.%s' % (self.path, self.region, self.market, instrument, date)
        if not self.feature_refresh and os.path.isfile(fpath):
            x = pd.read_pickle(fpath)
            cols = [c for c in x.columns if c.startswith('f__')]
        else:	
            x = self.get_combine_data(instrument, date)
            cols = get_features(x)
            x.to_pickle(fpath)
        return x, cols
   
    def get_combine_data(self, instrument, date):
        fpath = '%s/%s/%s/combine/%s.%s' % (self.path, self.region, self.market, instrument, date)
        if not self.combine_refresh and os.path.isfile(fpath):
	    return pd.read_pickle(fpath)
        else:
	    return self.combine(instrument, date)

    def combine(self, instrument, date):
	raise NotImplementedError()

 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def mincut(x):
    return min(x,1)


def _oi(df):
    df['oi'] = ( (df['bid_0_p'] >= df['bid_0_p'].shift()).astype(int)*df['bid_0_v']
		-(df['bid_0_p'] <= df['bid_0_p'].shift()).astype(int)*df['bid_0_v'].shift()
		-(df['ask_0_p'] <= df['ask_0_p'].shift()).astype(int)*df['ask_0_v']
		+(df['ask_0_p'] >= df['ask_0_p'].shift()).astype(int)*df['ask_0_v'].shift())



def add_tmp_cols(x, n=10):
    x['_mid'] = (x['ask_0_p'] + x['bid_0_p'])/2
    for i in range(n):
        x['_ask_dist_%d'%i] = (x['ask_%d_p'%i] - x['_mid']) / (x['ask_0_p'] - x['_mid'])
        x['_bid_dist_%d'%i] = (x['_mid'] - x['bid_%d_p'%i]) / (x['_mid'] - x['bid_0_p'])

def power_imbalance(x, n=10, powers=[2]):
    ret = []        
    for power, i in product(powers, range(n)):
        key = 'f__pi_%d_%d' % (power, i)
        ret.append(key)
        if i == 0:
            x[key] = 0
	else:
	    x[key] = x['f__pi_%d_%d'%(power, i-1)]
        x[key] += x['bid_%d_v'%i]/x['_bid_dist_%d'%i]**power
        x[key] -= x['ask_%d_v'%i]/x['_ask_dist_%d'%i]**power
    return ret

def volume_imbalance(x, n=10, powers=[0, 2]):
    ret = []
    for power, i in product(powers, range(n)):
        key = 'f__vi_%d_%d' % (power, i)
        ret.append(key)
        if i == 0:
            x['_via'] = 0
            x['_vib'] = 0
        rr = 1 if power == 0 else np.exp(-i*1.0/power)
        x['_via'] += x['ask_%d_v'%i]*rr
        x['_vib'] += x['bid_%d_v'%i]*rr
        x[key] = (x['_vib'] - x['_via'])/(x['_vib'] + x['_via'])
        if power == 0:
	    x['f__mv_ask_%d'%i] = x['_via']/(i+1.0)
	    x['f__mv_bid_%d'%i] = x['_vib']/(i+1.0)
	    ret += ['f__mv_ask_%d'%i, 'f__mv_bid_%d'%i]
    return ret


def power_price(x, n=10, powers=[2]):    
    ret = []
    for power in powers:
       x['_dist_w'] = 0
       x['_dist_s'] = 0
       for i in range(n):
            key = 'f__pp_%d_%d' % (power, i)
            ret.append(key)
            x['_dist_s'] += x['bid_%d_p'%i]*x['_bid_dist_%d'%i]**power + x['ask_%d_p'%i]*x['_ask_dist_%d'%i]**power
            x['_dist_w'] += x['_bid_dist_%d'%i]**power + x['_ask_dist_%d'%i]**power
            x[key] = np.log(x['_dist_s']/x['_dist_w']/x['_mid']) 
    return ret


def spread(x, n=10):
    ret = []
    for ab, pv in product(['ask', 'bid'],['p']):
        key = 'f__spread_0_%s_%s' % (ab, pv)
        ret.append(key)
        x[key] = x['%s_0_%s'%(ab,pv)] - x['tp_last']

    for i, ab, pv in product(range(n-1), ['ask', 'bid'], ['p']):
        key = 'f__spread_%d_%s_%s' % (i+1, ab, pv)
        ret.append(key)
        x[key] = x['%s_%d_%s'%(ab, i+1, pv)] - x['%s_%d_%s'%(ab, i, pv)]

    return ret

def diff(x, n=10, ranges=[1]):
    ret = []
    for i, ab, pv, r in product(range(n), ['ask', 'bid'], ['p'], ranges):
        key = 'f__diff_%d_%s_%s_%d'%(i, ab, pv, r)
        ret.append(key)
        x[key] = x['%s_%d_%s'%(ab, i, pv)].diff(r)/r
    return ret

def ma_y(x, forwards=[5]):
    ret = []
    for f in forwards:
        key = 'l__ma_%d'%f
        ret.append(key)
        x[key] = np.log(x['_mid'].rolling(f).mean().shift(-f) / x['_mid'])
    return ret

def shift(x, cols, shifts):    
    ret = []
    for s, c in product(shifts, cols):
	key = '%s_s%d' % (c, s)
	ret.append(key)
	x[key] = x[c].shift(s)
    return ret

def _feat(x, c):
    key = 'f__%s' % c
    x[key] = x[c]
    return key

def feat(x, n):
    ret = []
    ret.append(_feat(x, '_mid'))
    for ab, i in product(['ask','bid'], range(n)):
	ret.append(_feat(x, '%s_%d_v'%(ab,i)))
    return ret

def _trend(x):
    x = [p for p in x if p > 0]
    if len(x) > 1:
        return linregress(range(len(x)), x)[0]
    return 0

def trend(x, ranges):
    ret = []
    for r in ranges:
        key = 'f__trade_trend_%d'%r
	ret.append(key)
	x[key] = x['tp_mean'].rolling(r).apply(_trend, raw=True)
    return ret

def trade_range(x, k, ranges):
    ret = []
    for r in ranges:
        key = 'f__trade_%s_%d'%(k,r)
        ret.append(key)
        if k == 'simple_mean':
	    x[key] = np.log(x['tp_sum'].rolling(r).sum()/(x['f__trade_count_%d'%r]+0.001)/x['_mid'])
	else:
	    x[key] = x[k].rolling(r).sum()
    return ret

def oi(x):
    if 'oi' not in x.columns:
	_oi(x)
    x['f__oi'] = x['oi']
    x['f__oip'] = x['oi']/(x['ask_0_v']+x['bid_0_v'])
    x['f__ois'] = x['oi']/(x['bid_0_p']-x['ask_0_p'])
    return ['f__oi', 'f__oip', 'f__ois']

def tt(x):
    x['f__tt'] = np.log(x['tp_last']/x['_mid'])
    return ['f__tt']



def get_features(x):
    watch = 10
    powers = [0,2,4,8]
    ranges = [1,5,10,30,60,120,180]
 
    add_tmp_cols(x, watch)     

    cols = []
    print 'tick features...'
    cols += tt(x)
    cols += oi(x)
    cols += volume_imbalance(x, watch)
    cols += power_imbalance(x, watch, powers)
    cols += power_price(x, watch, powers)
    cols += spread(x, watch)
    cols += diff(x, watch, ranges)
    print len(cols)
    print 'trade features...'
    cols += trade_range(x, 'count', ranges)
    cols += trade_range(x, 'aggressor', ranges)
    cols += trade_range(x, 'simple_mean', ranges)
    cols += trend(x, ranges)
    print len(cols)
    print 'feat features...'
    cols += feat(x, watch)
    print len(cols)
    labels = ma_y(x, forwards=[1,5,10,20,40,100])
    print labels
#    print 'shifting features...'
#    cols += shift(x, cols[:], [1,2,3,4,5])
#    print len(cols)
    return cols






def get_data_of_dates():
    pass




