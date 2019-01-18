from ..reader import Reader


class OneTokenReader(Reader):
    def trade_agg(self, x):
        r = pd.Series({
            'tp_max': x['trade_price'].max(),
            'tp_min': x['trade_price'].min(),
            'tp_last': x['trade_price'].iloc[-1],
	    'tv_sum': x['trade_volume'].sum(),
	    'count': len(x),
	    'tp_sum': x['trade_price'].sum(),
	    'tp_mean': x['trade_price'].mean(),
	    'aggressor': x.loc[x.bs=='b']['trade_volume'].sum() - x.loc[x.bs=='s']['trade_volume'].sum(),
	    })
        return r

    def time_filter(self, df):
        _oi(df)
        df['ts'] = (df['timestamp']+500)//1000
        x = df.loc[df.groupby('ts').ts.idxmax()]
        x.set_index('ts',inplace=True)
        x['oi'] = df.groupby('ts').sum()['oi']
        x.drop(['timestamp','last','volume'], axis=1, inplace=True)
        return x

    def read_file(self, dtype, instrument, date):
        fpath = '%s/%s/%s/%s/%s.%s' % (self.path, self.region, self.market, dtype, instrument, date)
        if dtype == 'tick':
	    return pd.read_csv(fpath)
        elif dtype == 'trade':
	    return pd.read_csv(fpath, header=None, names=['extime','contract','trade_price','bs','trade_volume','exts','ttime','tts'])

    def combine(self, instrument, date):
        ticks = self.read_file('tick', instrument, date)            
        ticks = self.time_filter(ticks) 
        ticks = ticks.reindex(pd.RangeIndex(ticks.index.min(), ticks.index.max()+1))
        ticks.ffill(inplace=True)
        trades = self.read_file('trade', instrument, date)
        trades['ts'] = (trades['exts']).astype(int)+1
        trades = trades.groupby('ts').apply(self.trade_agg)
        df = ticks.join(trades)
        assert(len(df)>=86400)   
        df.to_pickle('%s/%s/%s/combine/%s.%s' % (self.path, self.region, self.market, instrument, date))
        return df




