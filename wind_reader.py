from fe import Reader
from dtypes import *
import snappy
import pandas as pd


def get_transaction(array):
    x = pd.DataFrame(array)
    names = {'exts':'nTime','trade_price':'nTradePrice','bs':'chBSFlag','trade_volume':'nTradeVolume'}
    for name in names:
	x[name] = x[names[name]]      
    return x[names.keys()]

def get_tickab(array):
    df1 = pd.DataFrame(array['nAskPrice'], columns=['ask_%d_p'%i for i in range(10)])
    df2 = pd.DataFrame(array['nBidPrice'], columns=['bid_%d_p'%i for i in range(10)])
    df3 = pd.DataFrame(array['nAskVolume'], columns=['ask_%d_v'%i for i in range(10)])
    df4 = pd.DataFrame(array['nBidVolume'], columns=['bid_%d_v'%i for i in range(10)])
    df5 = pd.DataFrame(array['nTime'], columns=['timestamp'])
    df = pd.concat([df1,df2,df3,df4,df5], axis=1)
    return df.loc[(df.timestamp>=93000000)&(df.timestamp<=113000000)|(df.timestamp>=130000000)&(df.timestamp<=150000000)]

def to_dt(nt):
    nt /= 1000
    return datetime.datetime(2018,1,10,nt/10000,nt/100%100,nt%100)


class WindReader(Reader):
    __type = {'transaction':dt_Transaction, 'tickab':dt_TickAB}
    def trade_agg(self, x):
        r = pd.Series({
            'tp_max': x['trade_price'].max(),
            'tp_min': x['trade_price'].min(),
            'tp_last': x['trade_price'].iloc[-1] if len(x) else None,
	    'tv_sum': x['trade_volume'].sum(),
	    'count': len(x),
	    'tp_sum': x['trade_price'].sum(),
	    'tp_mean': x['trade_price'].mean(),
	    'aggressor': x.loc[x.bs=='B']['trade_volume'].sum() - x.loc[x.bs=='S']['trade_volume'].sum(),
	    })
        return r

    def read_file(self, dtype, instrument, date):
        date = '%s%s%s'%(date[:4], date[5:7], date[8:])
        fpath = '%s/%s/%s/%s/%s_%s.%s' % (self.path, self.region, self.market, dtype, dtype, instrument, date)
	array = np.fromstring(snappy.uncompress(open(fpath, 'rb').read()), dtype=self.__type[dtype])     
        if dtype == 'transaction':
	    df = get_transaction(array)
	elif dtype == 'tickab':
	    df = get_tickab(array)   
        return df

    def combine(self, instrument, date):
        ticks = self.read_file('tickab', instrument, date)   
        ticks.set_index('timestamp', inplace=True)
        trades = self.read_file('transaction', instrument, date)
        tradesagg = trades.groupby(pd.cut(trades['exts'], ticks.index)).apply(self.trade_agg)       
	tradesagg.set_index(tradesagg.index.right, inplace=True)

        df = ticks.join(tradesagg)
        print len(df)
        df.to_pickle('%s/%s/%s/combine/%s.%s' % (self.path, self.region, self.market, instrument, date))
        return df


