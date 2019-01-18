from xgboost import XGBRegressor
from sklearn.metrics import r2_score

class TickX(object):
    def __init__(self):
        self.model = None

    def select(self, x, cols, ft, vlimit, method='ma'):
        if vlimit is None:
   	    vlimit = x['tv_sum'].quantile(0.5)
	    print 'v_limit', vlimit
        x = x.loc[x.tv_sum > vlimit]
        label = 'l__%s_%d'%(method, ft)
        d = x.dropna(subset=[label]) 
        return d[cols], d[label]
        #print X.iloc[:5]        
        #ret = [c for c in ret if 'trade' not in c]
        #X.corrwith(y).to_pickle('tmp')
	#print len(ret),'/',len(X.columns)


    def train(self, reader, instrument, dates, ft, vlimit=None):
        data, cols = reader.get_data(instrument, dates[0])        
        X, y = self.select(data, cols, ft, vlimit)
        self.cols = X.columns
        self.model = XGBRegressor()
	#self.model = XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=6, gamma=0, reg_alpha=1e-4, subsample=0.9, colsample_bytree=0.9, seed=7)
        self.model.fit(X, y)
        p = self.model.predict(X)
        print 'train size: %s, r2: %s' % (len(y), r2_score(y, p)) 

    def test(self, reader, instrument, dates, ft, vlimit):
	data, _ = reader.get_data(instrument, dates[0])
        X, y = self.select(data, self.cols, ft, vlimit)
        p = self.model.predict(X)
        print 'test size: %s, r2: %s' % (len(y), r2_score(y, p))
       
    def predict(self, df):
	pass
        #if df.iloc[-1]['tv_sum'] < 3:
        #    return None
        #x = df.copy()
        #get_features(x)    
	#get_trade_features(x)
        #p = self.model.predict(x.iloc[-1:].fillna(0)[self.cols])
	#assert(len(p)==1)
        #return p[0]



