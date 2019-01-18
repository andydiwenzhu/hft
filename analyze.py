import pickle
import numpy as np
import os
import pandas as pd
<<<<<<< HEAD
=======
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
from configs import cfg
from dutil import get_tickab
from redis_feed import AShare
>>>>>>> f31f118d00beb5b3549367741e535afaa030338b
import logging
logger = logging.getLogger(__name__)

def basic(dic, debug=False):
    df = pd.DataFrame(dic['deals'])
 
    for bs in ['buy','sell']:
        x = df.ix[df.bsflag==bs]
        x_amt = (x['avg_price']*x['shares']).sum()
        x_shares = x['shares'].sum()
	if x_shares:
	    if debug:
		print len(x), bs, x_shares, x_amt/x_shares
	    else:
                logger.info('%s %s %s %s', len(x), bs, x_shares, x_amt / x_shares)



def check_twap(results, ins, date, interval, tasks, market):
    deals = sorted(results[ins]['deals'], key=lambda k: k['dt'])
    market.generate_intervals(date, interval, tasks)

    error = 100
    ms = 0
    for d in deals:
	bs = d['bsflag']
	market.shift_intervals(d['dt'])
	market.add_current(ins, bs, d['shares'])
	if d['shares'] > ms:
	   ms = d['shares']
	x = market.check_current(ins, bs, d['dt'])
	if x < error:
	    error = x

    print 'error:', error, 'max share:', ms



def draw(ins, date, result):
    path = cfg.path[AShare().get_type(ins)]
    fn = '{path}/{date}/tickab_{stock}.{date}'.format(path=path, date=date, stock=ins)
    with open(fn, 'rb') as f:
        df = get_tickab(f)
    deals = pd.DataFrame(result['deals'])
    deals['nt'] = deals['dt'].apply(lambda x: (((x.hour*100+x.minute)*100)+x.second)*1000).astype('int32')

    deals = deals.set_index('nt')
    deals = deals.reindex(df.nTime.unique(), method='backfill', limit=1).dropna()
    df = df.join(deals, on='nTime')
   
    t = df['nTime']
    x = list(df.index)
    p = df['nPrice']
    b = df['avg_price'].mask(df['bsflag']!='buy', None).dropna()
    s = df['avg_price'].mask(df['bsflag']!='sell', None).dropna()

    plt.figure(figsize=(20,10), dpi=80)
    plt.xticks(x[::600]+x[-1:],['%d:%02d'%(i/10000000, i/100000%100) for i in t[::600]]+['15:00'])
    plt.yticks(np.linspace(min(p),max(p),10,endpoint=True))
    plt.ylim(min(p), max(p))

    plt.grid(True, 'major', 'x')
    plt.scatter(list(b.index), b, 50, color='green') 
    plt.scatter(list(s.index), s, 50, color='red')
    plt.plot(x, p)
    
    mpld3.show(ip='192.168.128.21',port=8887) 
    



if __name__ == '__main__':
    filename = sorted(os.listdir('results/'))[-1]
    print filename
    with open('results/'+filename, 'rb') as f:
        results = pickle.load(f)

    draw('510500', '20181109', results['510500'])
    exit()
 
    for ins in results:
	print ins
	check_twap(results, ins, 20181109, 240, {'002475':{'buy':100000,'sell':100000}})
