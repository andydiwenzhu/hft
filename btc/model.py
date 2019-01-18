import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge, RidgeCV, LinearRegression

def read_ticks(exchange, contract, date):
    path = '%s/%s.%s' % (exchange, contract, date)
    return pd.read_csv(path)	

def _vt(df, side):
    kp = side+'_0_p'
    kv = side+'_0_v'
    
    res = [0]
    mat = df[[kp, kv]].values
    
    for i in range(1, len(mat)):
	if side == 'bid' and mat[i][0] < mat[i-1][0] or side == 'ask' and mat[i][0] > mat[i-1][0]:
	    res.append(0)
	elif mat[i][0] == mat[i-1][0]:
	    res.append(mat[i][1] - mat[i-1][1])
	else:
	    res.append(mat[i][1])
    return res
    

def extend(df, b):
    for k in ['oi','roi']:
      for i in range(1, b+1):
	df['%s_%d'%(k,i)] = df[k].shift(i)
    return df




def get_features(df, back=0, forward=100, div_by_spread=False, include_label=True):
    df['mid'] = (df['ask_0_p'] + df['bid_0_p'])/2
#    print df['mid'].describe()
    df['label'] = df['mid'].rolling(forward).mean().shift(-forward) - df['mid']
#    print df['label'].describe()
#    exit()
#    print 'raw:', len(df)
#    cols = ['ask_0_p', 'ask_0_v', 'bid_0_p', 'bid_0_v']
#    df = df.loc[(df[cols] != df[cols].shift()).any(axis=1)].copy()
#    print 'silence removed:', len(df)
    df['d_ask_v'] = _vt(df, 'ask')
    df['d_bid_v'] = _vt(df, 'bid')
    df['oi'] = df['d_bid_v'] - df['d_ask_v']
    df['soi'] = df['d_bid_v'] + df['d_ask_v']
    df = df.ix[df.soi != 0]
    df = df.ix[(df.ask_0_v > 10)|(df.bid_0_v > 10)]
#    print 'div_by_zero removed:', len(df)
    df['roi'] = df['oi']/df['soi']
    df['sp'] = df['ask_0_p'] - df['bid_0_p']


    if div_by_spread:
        df['oi'] = df['oi']/df['sp']
        df['roi'] = df['roi']/df['sp']
	

    if include_label:
	return extend(df[['oi','roi','label']].copy(), back)

    return extend(df[['oi','roi']].copy(), back)
  


if __name__ == '__main__':
    df = read_ticks('huobip','btc.usdt','2018-11-23')
    for f in [1]:
        x = get_features(df,forward=f)
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.dropna()
 #   print 'nan removed:', len(x)
 #   print x.iloc[10000:10010]
        label = x['label']
    #data = x.drop(['label'], axis=1)
        data = x[['oi']]
        print data.shape, len(label)
    
        m = LinearRegression().fit(data.values, label)
        print m.score(data.values, label)
#    print m.coef_
#    print m.intercept_

'''

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
'''
