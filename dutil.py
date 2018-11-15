import numpy as np

import dtypes

def TDB_Future(binary):
    return np.fromstring(binary, dtype=dtypes.dt_Future)

def TDB_FutureAB(binary):
    return np.fromstring(binary, dtype=dtypes.dt_FutureAB)

def TDB_OrderQueue(binary):
    return np.fromstring(binary, dtype=dtypes.dt_OrderQueue)

def TDB_TickAB(binary):
    return np.fromstring(binary, dtype=dtypes.dt_TickAB)

def TDB_Transaction(binary):
    return np.fromstring(binary, dtype=dtypes.dt_Transaction)

from datetime import datetime

def _to_datetime(nDate, nTime):
    return datetime(nDate//10000, (nDate%10000)//100, nDate%100, nTime//10000000,
            (nTime%10000000)//100000, (nTime%100000)//1000, (nTime%1000)*1000)

to_datetime = np.frompyfunc(_to_datetime, 2, 1)

def _to_interval_oc(nTime, nMin):
    h, m = nTime//10000000, ((nTime%10000000)//100000)//nMin*nMin
    if nTime > h*10000000+m*100000:
        if m+nMin >= 60:
            return (h+1)*10000+(m+nMin-60)*100
        return h*10000+(m+nMin)*100
    return h*10000+m*100

to_interval_oc = np.frompyfunc(_to_interval_oc, 2, 1)

def _to_interval_co(nTime, nMin):
    h, m = nTime//10000000, (((nTime%10000000)//100000)//nMin+1)*nMin
    if nTime < h*10000000+m*100000:
        if m >= 60:
            return (h+1)*10000+(m-60)*100
        return h*10000+m*100
    if m+nMin >= 60:
        return (h+1)*10000+(m+nMin-60)*100
    return h*10000+(m+nMin)*100

to_interval_co = np.frompyfunc(_to_interval_co, 2, 1)

def generate_intervals(interval, method='oc'):
    if method == 'co':
        am_9 = [i*100 for i in range(930, 960, interval)]
        am_10 = [i*100 for i in range(1000, 1060, interval)]
        am_11 = [i*100 for i in range(1100, 1130, interval)]
        pm_1 = [i*100 for i in range(1300, 1360, interval)]
        pm_2 = [i*100 for i in range(1400, 1460, interval)]
        return am_9+am_10+am_11+pm_1+pm_2
    else:
        am_9 = [i*100 for i in range(930+interval, 960, interval)]
        am_10 = [i*100 for i in range(1000, 1060, interval)]
        am_11 = [i*100 for i in range(1100, 1130+interval, interval)]
        pm_1 = [i*100 for i in range(1300+interval, 1360, interval)]
        pm_2 = [i*100 for i in range(1400, 1460, interval)]
        return am_9+am_10+am_11+pm_1+pm_2+[150000]

intervals = {
        (5, 'co'): set(generate_intervals(5, method='co')),
        (5, 'oc'): set(generate_intervals(5, method='oc')),
        }

import pandas as pd
import snappy
import logging
logger = logging.getLogger('wdhh')

def generate_path(path, name, stock, date):
    stock, date = stock[:6], str(date)
    YYYY, MM, DD = date[:4], date[4:6], date[6:8]
    return path.format(name=name, stock=stock, date=date,
            YYYY=YYYY, MM=MM, DD=DD)

def read_path(path, name=None, stock=None, date=None):
    if stock is None:
        return 'xxxxxx', 888888, snappy.uncompress(path.read())
    return stock, date, snappy.uncompress(open(generate_path(path, name, stock, date), 'rb').read())

def get_orderqueue(path, stock=None, date=None, **kwargs):
    stock, date, binary = read_path(path, 'orderqueue', stock, date)
    array = TDB_OrderQueue(binary)
    names = list(array.dtype.names)
    df1 = pd.DataFrame(array['nABVolume'],
            columns=['nABVolume_'+str(i+1) for i in range(50)])
    names.pop(names.index('nABVolume'))
    df2 = pd.DataFrame(array[names])

#    for t in df2['nABItems'].itertuples():
#       df1.iloc[t[0], t[1]:] = np.nan

    df = pd.concat([df1, df2], axis=1)
    df['nPrice'] /= 1e4

    X = []
    last = 0
    for t in df.itertuples():
        x = list(t)
        now = x[-5]+x[-4]
        if now <= last:
            continue
        else:
            last = now
#print x
        x[x[-1]+1:51] = [0]*(50-x[-1])
        X.append(x[1:])

    ndf = pd.DataFrame(X,columns=df.columns)

 #  df.iloc[:, :50] = df.iloc[:, :50][df.iloc[:, :50]>0]
    return format(ndf, **kwargs)

def get_tickab(path, stock=None, date=None, **kwargs):
    stock, date, binary = read_path(path, 'tickab', stock, date)
    array = TDB_TickAB(binary)
    names = list(array.dtype.names)
    df1 = pd.DataFrame(array['nAskPrice'],
            columns=['nAskPrice_'+str(i+1) for i in range(10)])
    df1 /= 1e4
    names.pop(names.index('nAskPrice'))
    df2 = pd.DataFrame(array['nAskVolume'],
            columns=['nAskVolume_'+str(i+1) for i in range(10)])
    names.pop(names.index('nAskVolume'))
    df3 = pd.DataFrame(array['nBidPrice'],
            columns=['nBidPrice_'+str(i+1) for i in range(10)])
    df3 /= 1e4
    names.pop(names.index('nBidPrice'))
    df4 = pd.DataFrame(array['nBidVolume'],
            columns=['nBidVolume_'+str(i+1) for i in range(10)])
    names.pop(names.index('nBidVolume'))
    df5 = pd.DataFrame(array[names])
    df = pd.concat([df1, df2, df3, df4, df5], axis=1)
    df['nPrice'] /= 1e4
    df['nHigh'] /= 1e4
    df['nLow'] /= 1e4
    df['nOpen'] /= 1e4
    df['nPreClose'] /= 1e4
    return format(df, **kwargs)

def get_transaction(path, stock=None, date=None, **kwargs):
    stock, date, binary = read_path(path, 'transaction', stock, date)
    df = pd.DataFrame(TDB_Transaction(binary))
    df['nTradePrice'] /= 1e4
    return format(df, **kwargs)

def get_transaction_nodivide(path, stock=None, date=None, **kwargs):
    stock, date, binary = read_path(path, 'transaction', stock, date)
    df = pd.DataFrame(TDB_Transaction(binary))
    return format(df, **kwargs)

def format(df, datetime_index=False, interval=None, method='oc', inplace=False):
    if not inplace:
        df = df.copy()
    if datetime_index:
        df.index = to_datetime(df.nDate, df.nTime)
    if interval:
        if method == 'co':
            df['nInterval'] = to_interval_co(df.nTime, interval)
        else:
            df['nInterval'] = to_interval_oc(df.nTime, interval)
    return df
