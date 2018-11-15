import pickle
import os
import pandas as pd
from redis_feed import AShare
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



def check_twap(results, ins, date, interval, tasks, market=AShare()):
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


if __name__ == '__main__':
    filename = sorted(os.listdir('results/'))[-1]
    print filename
    with open('results/'+filename, 'rb') as f:
        results = pickle.load(f)

    for ins in results:
	print ins
	check_twap(results, ins, 20181109, 240, {'002475':{'buy':100000,'sell':100000}})
