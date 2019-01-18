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
    x = df.loc[df.groupby('ts').ts.idxmin()]
    x.set_index('ts',inplace=True)
    x['v'] = df.groupby('ts').sum()['volume']
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


def get_features(x):
    add_tmp_cols(x)    
    cols = []
    cols += power_imbalance(x)
    cols += power_price(x)
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
        if x.iloc[0]['v'] < 0.1:
            return None
        cols = get_features(x)
        p = self.model.predict(x.dropna()[cols])
        return p[0]


if __name__ == '__main__':
    m = Tick_X()
    m.train(24)
    m.test(25)
    x = time_filter(25)
    res = []
    for i in range(len(x)-1):
        if i%1000==0:
            print i
        res.append(m.predict(x.iloc[i:i+2].copy()))    
    print pd.DataFrame({'predict':res}).describe()


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



import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


if __name__ == '__main__':
    x = time_filter(read_file(['2018-11-24'], 'btc/huobip', 'btc.usdt'))
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, len(x.columns), 10, 1

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

